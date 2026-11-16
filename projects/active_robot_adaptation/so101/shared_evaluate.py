"""Paired goal-directed control; no labels or simulator state given to policies."""
import argparse,json,time
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from shared_tasks import SharedTask
from run_store import create_run
SCALE=np.array([.2,.2,.1,.1,.2,.2,1,1,1,1,1,1,1],np.float32)
def vector(obs,action):return np.r_[np.asarray(obs,dtype=np.float32)/SCALE,np.asarray(action,dtype=np.float32)/.01]
def infer(w,x):return np.tanh(x@w[0]+w[1])@w[2]+w[3]
def setup(e,task,start,seed):
 f=e.reset(start['xy'],seed,start['pan']);xy=f[:2].copy();start_truth=e.truth()
 if not f[-1]:return None,None,None,start_truth
 if task=='push':
  reached=[e.move_to([xy[0]-.0116,xy[1]+.060,.06]),e.move_to([xy[0]-.0116,xy[1]+.060,.018])];grip=1.2
 else:
  reached=[e.move_to([xy[0]-.0116,xy[1]-.0124,.06]),e.move_to([xy[0]-.0116,xy[1]-.0124,.014])];grip=-.1
  for _ in range(15):e.env.step(np.zeros(5),grip)
 e.last=e.env.observe();f=e.features(e.last)
 return f,grip,(float(xy[1])-.04,reached),start_truth

def main():
 p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]);args=p.parse_args()
 cfg=json.load(open('so101/shared_protocol.json'));cfg['model_run']=args.models;cfg['evaluation_model_seeds']=args.seeds;cfg['methods']=['conventional','current','history','frozen_history'];cfg['planned_episodes']=len(args.seeds)*4*len(cfg['tasks'])*len(cfg['test_starts'])*len(cfg['test_conditions']);out=create_run('shared_evaluation',cfg);(out/'episodes').mkdir();rows=[]
 for seed in args.seeds:
  weights={}
  for model in ('current','history'):
   archive=np.load(Path(args.models)/f'{model}_seed{seed}.npz');weights[model]=[archive[f'p{i}'] for i in range(4)]
  for task in cfg['tasks']:
   for ci,condition in enumerate(cfg['test_conditions']):
    for si,start in enumerate(cfg['test_starts']):
     for method in cfg['methods']:
      e=SharedTask(task,**condition);begin=time.perf_counter();f,grip,setting,truth_start=setup(e,task,start,5000+si);frames=[];history=[];frozen=None;records=[];compute=0.;missing=0
      if f is None:
       row={'seed':seed,'task':task,'start':start,'condition':condition,'method':method,'success':False,'failure':'initial_visibility','trace':e.env.trace}
       with (out/'episodes'/f'{seed}_{task}_{ci}_{si}_{method}.json').open('x') as h:json.dump(row,h)
       rows.append(row);e.close();continue
      goal,reached=setting;frame=e.last.get('image');
      if frame is not None:frames.append(frame.copy())
      for i in range(cfg['eval_steps']):
       tic=time.perf_counter()
       if not f[-1]:action=np.zeros(2);missing+=1
       elif abs(float(f[1])-goal)<cfg['goal_tolerance_m']:action=np.zeros(2)
       elif method=='conventional':action=np.array([0.,np.clip(goal-f[1],-.012,.012)])
       else:
        model='current' if method=='current' else 'history';candidates=np.array([[0.,v] for v in (-.012,-.006,0.,.006,.012)])
        context=([np.zeros(15,np.float32)]*3+history)[-3:]
        if method=='frozen_history':
         if frozen is None and len(history)>=3:frozen=[x.copy() for x in context]
         if frozen is not None:context=frozen
        vals=[]
        for a in candidates:
         z=vector(f,a);inp=z if model=='current' else np.r_[z,*context];delta=infer(weights[model],inp)*.01
         vals.append(abs(float(f[1]+np.clip(delta[1],-.03,.03))-goal)+.001*abs(a[1]))
        action=candidates[int(np.argmin(vals))]
       compute+=time.perf_counter()-tic
       old=f.copy();f=e.step(action,grip);history.append(vector(old,action));records.append({'obs':old.tolist(),'action':action.tolist(),'next_obs':f.tolist(),'time_s':float(e.last['time_s'])})
       if e.last.get('image') is not None:frames.append(e.last['image'].copy())
      # Every controller is charged the same final 0.5-second settling window.
      for _ in range(5):f=e.step([0.,0.],grip)
      truth=e.truth();target=truth_start[1]-.04;err=abs(float(truth[1])-target);invalid=e.numerical_reset or any(e.physics_warnings)
      score=err<cfg['goal_tolerance_m'] and not invalid
      row={'seed':seed,'task':task,'start':start,'condition':condition,'method':method,'success':bool(score),'failure':'numerical' if invalid else None if score else 'goal_error',
       'target_y_observed':goal,'final_error_evaluator_m':err,'displacement_evaluator_m':(truth-truth_start).tolist(),'setup_reached':reached,'records':records,'trace':e.env.trace,
       'actions_including_setup':e.env.action_count,'movement_rad_including_setup':e.env.movement,'sim_s':float(e.env.data.time),'wall_s':time.perf_counter()-begin,'model_selection_compute_s':compute,'controller_compute_s':e.env.compute_s,'missing_observations':missing,
       'numerical_invalid':invalid,'warnings':e.physics_warnings,'self_collision_physics_steps':e.env.physics_collision_steps,'joint_limit_physics_steps':e.env.physics_limit_steps,'retries':0}
      name=f'{seed}_{task}_{ci}_{si}_{method}'
      with (out/'episodes'/f'{name}.json').open('x') as h:json.dump(row,h)
      if seed==args.seeds[0] and si==0 and ci==0 and frames:imageio.mimsave(out/f'{task}_{method}.gif',frames,duration=200,loop=0)
      rows.append(row);e.close();print(name,score,round(err,4),'invalid',invalid,flush=True)
 with (out/'results.json').open('x') as h:json.dump(rows,h)
 print('saved',out)
if __name__=='__main__':main()
