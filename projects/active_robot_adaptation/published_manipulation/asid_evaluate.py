"""Paired exploration-only diagnostic in the authors' unmodified environment."""
import argparse,copy,json,os,resource,sys,time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import yaml
from run_store import create

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--limit',type=int);a=p.parse_args();a.checkpoint=str(Path(a.checkpoint).resolve()) if a.checkpoint else None;src=Path(os.environ.get('ASID_SRC','/tmp/active-asid-src')).resolve();sys.path.insert(0,str(src));os.chdir(src)
 from env.wrappers.asid_vec import make_env
 from stable_baselines3 import SAC
 root=Path(__file__).resolve().parent;cfg=json.loads((root/'asid_protocol.json').read_text());cfg['checkpoint']=a.checkpoint;cfg['limit']=a.limit
 cases=cfg['cases'][:a.limit] if a.limit else cfg['cases'];methods=['random','author_short_trained'] if a.checkpoint else ['random'];out=create('asid_explore',cfg);rows=[]
 policy=SAC.load(a.checkpoint,device='cpu') if a.checkpoint else None
 for case in cases:
  for method in methods:
   robot=yaml.safe_load((src/'configs/robot/sim_2_dof.yaml').read_text());obj=yaml.safe_load((src/'configs/env/rod.yaml').read_text());param=yaml.safe_load((src/'configs/asid/inertia.yaml').read_text())
   t0=time.perf_counter();env=make_env(robot,obj,param,seed=case['seed']);env.parameter_dict['inertia']['value']=case['inertia_m'];env.params_set=True
   np.random.seed(case['seed']);obs=env.reset();rng=np.random.default_rng(case['seed']+100000);init_pose=env.get_obj_pose().copy();frames=[];steps=[];error=None
   if case==cases[0]:frames.append(env.render().copy())
   for i in range(20):
    action=rng.uniform(-.1,.1,size=2).astype(np.float32) if method=='random' else policy.predict(obs,deterministic=True)[0]
    before=obs.copy();start=time.perf_counter();obs,reward,done,info=env.step(action);elapsed=time.perf_counter()-start
    steps.append({'observation':before.tolist(),'action':np.asarray(action).tolist(),'next_observation':obs.tolist(),'author_information_reward':float(reward),'step_wall_s':elapsed,'simulation_s':float(env.unwrapped._robot.data.time),'rod_pose_evaluator':env.get_obj_pose().tolist()})
    if case==cases[0] and i%4==0:frames.append(env.render().copy())
    if not np.all(np.isfinite(obs)) or not np.isfinite(reward):error='nonfinite';break
    if done:break
   rod=env.get_obj_pose().copy();warning=[int(w.number) for w in env.unwrapped._robot.data.warning];row={'case':case,'method':method,'actual_parameter_evaluator':env.get_parameters().tolist(),'steps':steps,'complete':len(steps)==20,'failure':error,'author_reward_sum':sum(s['author_information_reward'] for s in steps),'rod_displacement_m_evaluator':float(np.linalg.norm(rod[:3]-init_pose[:3])),'elapsed_wall_s':time.perf_counter()-t0,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'physics_warnings':warning};rows.append(row)
   with (out/(case['id']+'_'+method+'.json')).open('x') as f:json.dump(row,f,indent=2)
   if frames:imageio.mimsave(out/(case['id']+'_'+method+'.gif'),frames,duration=300,loop=0)
   env.unwrapped._robot.close() if hasattr(env.unwrapped._robot,'close') else None
   print(case['id'],method,'return',row['author_reward_sum'],'displacement',row['rod_displacement_m_evaluator'],flush=True)
 with (out/'summary.json').open('x') as f:json.dump(rows,f,indent=2)
 print('saved',out,flush=True)
if __name__=='__main__':main()
