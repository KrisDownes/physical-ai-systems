"""One paired rod iteration; simulated push followed by idealized release."""
import argparse,gzip,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'published_manipulation')]
from run_store import create
from compare import yaw,wrap,fit,predict,features,choose_prior
from balance import evaluate

TRAIN_FILES=['results/published_runs/20261117T183441.744535Z_91b38b9b1287_rod_push_development_aa07c4fe/summary.json','results/published_runs/20261117T183600.677617Z_91b38b9b1287_rod_push_evaluation_3024b01b/summary.json']
def training():return [r for f in TRAIN_FILES for r in json.loads((ROOT/f).read_text())['episodes']]
def hist(row, donor=None):
 o=np.array(row['observations']);d=np.array((donor or row)['observations']);a=np.array((donor or row)['actions']);start=yaw(d[0,14:18]); rel=[wrap(yaw(d[i,14:18])-start) for i in (5,10,15,20)]
 # Full recipient final/current state and its last observed velocities remain intact.
 return np.r_[features(row,'current_state'),yaw(o[0,14:18]),o[0,12]-o[0,1],rel,np.sum(a,axis=0),d[-1,11:13]-d[0,11:13]]
def train_hist(rows):
 x=np.stack([hist(r) for r in rows]);y=np.array([.6*r['hidden_mass_offset_m_evaluator_training'] for r in rows]);mean=x.mean(0);scale=np.maximum(x.std(0),.02);z=(x-mean)/scale;X=np.c_[np.ones(len(z)),z];coef=np.linalg.solve(X.T@X+np.diag([0]+[1.]*(X.shape[1]-1)),X.T@y);return mean,scale,coef
def source(out):
 with gzip.open(out/'iteration_source.json.gz','wt') as f:json.dump({p.name:p.read_text() for p in Path(__file__).resolve().parent.glob('*.py')},f,sort_keys=True)
def outcome(theta,choice):return evaluate(theta,choice)[0]
def dev():
 rows=training();counts={'fitted_prior':0,'start_only':0,'center_prior':0};per=[];t=time.perf_counter()
 for theta in sorted(set(r['hidden_mass_offset_m_evaluator_training'] for r in rows)):
  fold=[r for r in rows if r['hidden_mass_offset_m_evaluator_training']!=theta];prior=choose_prior(fold);sm=fit(fold,'start_only')
  for r in rows:
   if r['hidden_mass_offset_m_evaluator_training']!=theta:continue
   choices={'fitted_prior':prior,'start_only':predict(sm,features(r,'start_only')),'center_prior':0.}
   for name,c in choices.items():
    ok=outcome(theta,c)['success'];counts[name]+=int(ok);per.append({'case_seed':r['seed'],'theta':theta,'method':name,'choice_m':c,'success':ok})
 # Tie-break: simpler centered prior, then fitted prior, then start-only.
 best=max(('center_prior','fitted_prior','start_only'),key=lambda k:counts[k]);out=create('rod_iteration_development',{'training_files':TRAIN_FILES,'split':'leave one mass offset out','prior_selection_counts':counts});source(out);(out/'summary.json').write_text(json.dumps({'counts':counts,'selected':best,'cases':per,'wall_s':time.perf_counter()-t},indent=2));print('dev',counts,'selected',best,'saved',out,flush=True)

def collect(n,protocol):
 import yaml,mujoco,imageio.v2 as imageio
 src=Path(os.environ.get('ASID_SRC','/tmp/active-asid-src')).resolve();sys.path.insert(0,str(src));from env.wrappers.asid_vec import make_env
 cfg=json.loads(Path(protocol).read_text());cases=cfg['cases'][:n];out=create('rod_iteration_push',{'protocol':cfg,'actual_case_count':n,'source_commit_at_execution':os.popen('git rev-parse HEAD').read().strip()});source(out);os.chdir(src);rows=[];start=time.perf_counter();original=mujoco.mj_step
 for i,c in enumerate(cases):
  robot=yaml.safe_load(Path('configs/robot/sim_2_dof.yaml').read_text());obj=yaml.safe_load(Path('configs/env/rod.yaml').read_text());par=yaml.safe_load(Path('configs/asid/inertia.yaml').read_text());par['reward']=False
  env=make_env(robot,obj,par,seed=c['seed']);m=env.unwrapped._robot.model;d=env.unwrapped._robot.data;marker=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'rod_visual_com');m.geom_rgba[marker,3]=0;env.parameter_dict['inertia']['value']=c['theta'];env.params_set=True;np.random.seed(c['seed']);obs=env.reset()
  # Check all robot self-contacts, excluding same/adjacent bodies and the two
  # neighboring fingers; table/rod contacts are not self-collisions.
  robot_bodies={j for j in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,j) or '') in {'link0','link1','link2','link3','link4','link5','link6','link7','hand','left_finger','right_finger'}}
  joints=[j for j in range(m.njnt) if int(m.jnt_bodyid[j]) in robot_bodies and m.jnt_limited[j]];audit={'steps':0,'limit_steps':0,'self_collision_steps':0,'pairs':{},'joint_excursions':{},'max_limit_excursion':0.,'rod_robot_contact_steps':0}
  def audited(model,data,nstep=1):
   for _ in range(nstep):
    original(model,data);audit['steps']+=1;bad=False
    for j in joints:
     value=float(data.qpos[model.jnt_qposadr[j]]);lo,hi=model.jnt_range[j];ex=max(0.,float(lo-value),float(value-hi));audit['max_limit_excursion']=max(audit['max_limit_excursion'],ex);bad|=ex>1e-5
     if ex>1e-5:
      name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j);entry=audit['joint_excursions'].setdefault(name,{'steps':0,'max':0.});entry['steps']+=1;entry['max']=max(entry['max'],ex)
    audit['limit_steps']+=int(bad);collisions=set();rodcontact=False
    for cc in data.contact[:data.ncon]:
     if cc.dist>0:continue
     b1=int(model.geom_bodyid[cc.geom1]);b2=int(model.geom_bodyid[cc.geom2]);names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,k) for k in (cc.geom1,cc.geom2)]
     if 'rod_geom_0' in names and (b1 in robot_bodies or b2 in robot_bodies):rodcontact=True
     if b1 not in robot_bodies or b2 not in robot_bodies or b1==b2:continue
     if int(model.body_parentid[b1])==b2 or int(model.body_parentid[b2])==b1:continue
     n1=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b1);n2=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b2)
     if {n1,n2}=={'left_finger','right_finger'}:continue
     collisions.add('/'.join(sorted((n1,n2))))
    audit['rod_robot_contact_steps']+=int(rodcontact);audit['self_collision_steps']+=bool(collisions)
    for pair in collisions:audit['pairs'][pair]=audit['pairs'].get(pair,0)+1
  observations=[obs.tolist()];actions=[];poses=[env.get_obj_pose().tolist()];target=obs[11:13].copy();frames=[];t=time.perf_counter()
  if i==0:frames.append(env.render().copy())
  try:
   mujoco.mj_step=audited
   for k in range(20):
    goal=target+np.array([-.09,0]) if k<6 else target+np.array([.10,0]);action=np.clip(goal-obs[:2],-.07,.07);obs,_,_,_=env.step(action);observations.append(obs.tolist());actions.append(action.tolist());poses.append(env.get_obj_pose().tolist())
    if i==0 and k%4==0:frames.append(env.render().copy())
  finally:mujoco.mj_step=original
  row={'case':i,'seed':c['seed'],'hidden_mass_offset_m_evaluator_training':c['theta'],'observations':observations,'actions':actions,'rod_pose_evaluator':poses,'audit':audit,'action_sim_seconds':2.,'wall_s':time.perf_counter()-t,'warnings':[int(w.number) for w in d.warning]};rows.append(row)
  (out/f'episode_{i:03}.json').write_text(json.dumps(row));
  if frames:imageio.mimsave(out/'push.gif',frames,duration=300,loop=0)
  print('case',i,'audit',audit['steps'],audit['limit_steps'],audit['self_collision_steps'],flush=True)
 (out/'summary.json').write_text(json.dumps({'episodes':rows,'wall_s':time.perf_counter()-start},indent=2));print('saved',out,flush=True)

def score(data_path,protocol):
 cfg=json.loads(Path(protocol).read_text());data=json.loads(Path(data_path).read_text())['episodes'];train=training();assert len(data)==cfg['count'];assert [r['seed'] for r in data]==[c['seed'] for c in cfg['cases']];assert all(r['hidden_mass_offset_m_evaluator_training']==c['theta'] for r,c in zip(data,cfg['cases']))
 assert not ({r['seed'] for r in data}&{r['seed'] for r in train});assert not ({r['hidden_mass_offset_m_evaluator_training'] for r in data}&{r['hidden_mass_offset_m_evaluator_training'] for r in train})
 for name,sha in cfg['source_sha256'].items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==sha,(name,'frozen code changed')
 models={'A':fit(train,'start_only'),'B':fit(train,'current_state'),'C':train_hist(train),'D':fit(train,'history')};out=create('rod_iteration_scored',{'protocol':cfg,'data':str(Path(data_path).resolve()),'train':TRAIN_FILES});source(out);rows=[];t=time.perf_counter()
 for i,r in enumerate(data):
  theta=r['hidden_mass_offset_m_evaluator_training'];choices={'A':predict(models['A'],features(r,'start_only')),'B':predict(models['B'],features(r,'current_state')),'C':predict(models['C'],hist(r)),'D':predict(models['D'],features(r,'history'))}
  for method,choice in choices.items():
   st=time.perf_counter();res=outcome(theta,choice);res.update({'case':i,'seed':r['seed'],'method':method,'support_choice_m':choice,'hidden_offset_evaluator':theta,'compute_s':time.perf_counter()-st,'push_actions':0 if method=='A' else 20,'stage_sim_s':2. if method=='A' else 4.});rows.append(res)
 # Compatibility uses only initial pose, tool position and action context.
 # Each donor contributes its intact 20-step response/action history; recipient
 # keeps its own initial and final observations. The resulting cross-time
 # inconsistency is measured, not hidden.
 rng=np.random.default_rng(cfg['shuffle_seed']);shuffles=[];bootstrap=np.random.default_rng(cfg['bootstrap_seed']);
 for i,r in enumerate(data):
  o=np.array(r['observations']);context=np.r_[yaw(o[0,14:18]),o[0,11:13]-o[0,:2]];dist=[]
  for j,s in enumerate(data):
   if j==i:continue
   q=np.array(s['observations']);v=np.r_[yaw(q[0,14:18]),q[0,11:13]-q[0,:2]];dist.append((float(np.linalg.norm((context-v)/np.array([.3,.08,.08]))),j))
  candidates=sorted(dist)[:cfg['compatible_donors']];own=wrap(yaw(o[-1,14:18])-yaw(o[0,14:18]))
  for draw in range(cfg['shuffle_draws']):
   dd,j=candidates[int(rng.integers(len(candidates)))];donor=data[j];z=np.array(donor['observations']);response=wrap(yaw(z[-1,14:18])-yaw(z[0,14:18]));choice=predict(models['C'],hist(r,donor));res=outcome(r['hidden_mass_offset_m_evaluator_training'],choice)
   shuffles.append({'case':i,'donor':j,'draw':draw,'compatibility_distance':dd,'end_yaw_inconsistency_rad':float(abs(wrap(response-own))),'choice_m':choice,'success':res['success'],'absolute_error_m':abs(res['placement_error_evaluator_m'])})
 summaries={};by={m:[r for r in rows if r['method']==m] for m in 'ABCD'}
 for m,rr in by.items():summaries[m]={'success':sum(x['success'] for x in rr),'mean_absolute_error_m':float(np.mean([abs(x['placement_error_evaluator_m']) for x in rr]))}
 paired={}
 for m in 'ACD':
  x=by[m];y=by['B'];delta=np.array([abs(a['placement_error_evaluator_m'])-abs(b['placement_error_evaluator_m']) for a,b in zip(x,y)]);samples=np.mean(delta[bootstrap.integers(0,len(delta),(cfg['bootstrap_draws'],len(delta)))],axis=1)
  paired[m+'_minus_B']={'mean_abs_error_difference_m':float(delta.mean()),'bootstrap_95_percentile_ci_m':np.quantile(samples,[.025,.975]).tolist(),'method_only_success':sum(a['success'] and not b['success'] for a,b in zip(x,y)),'B_only_success':sum(b['success'] and not a['success'] for a,b in zip(x,y)),'both_success':sum(a['success'] and b['success'] for a,b in zip(x,y))}
 shuffled_success=np.array([s['success'] for s in shuffles]).reshape(len(data),cfg['shuffle_draws']);shuffled_error=np.array([s['absolute_error_m'] for s in shuffles]).reshape(len(data),cfg['shuffle_draws']);cs=np.array([r['success'] for r in by['C']]);ce=np.array([abs(r['placement_error_evaluator_m']) for r in by['C']]);diag={'draws_per_recipient':cfg['shuffle_draws'],'mean_success_count':float(shuffled_success.sum(axis=0).mean()),'draw_success_counts':shuffled_success.sum(axis=0).tolist(),'correct_success_count':int(cs.sum()),'mean_shuffled_absolute_error_m':float(shuffled_error.mean()),'correct_mean_absolute_error_m':float(ce.mean()),'median_end_yaw_inconsistency_rad':float(np.median([s['end_yaw_inconsistency_rad'] for s in shuffles])),'note':'nearest-initial-geometry donor resampling, not a bijective permutation; incompatible endpoint observations make this an OOD diagnostic'}
 result={'summaries':summaries,'paired':paired,'shuffle_diagnostic':diag,'rows':rows,'shuffles':shuffles,'models':{k:{'mean':v[0].tolist(),'scale':v[1].tolist(),'coef':v[2].tolist()} for k,v in models.items()},'wall_s':time.perf_counter()-t};(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'summaries':summaries,'paired':paired,'shuffle':diag},indent=2));print('saved',out,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['dev','collect','score']);p.add_argument('--n',type=int,default=48);p.add_argument('--protocol',default=str(Path(__file__).with_name('ITERATION_PROTOCOL.json')));p.add_argument('--data');a=p.parse_args()
 if a.mode=='dev':dev()
 elif a.mode=='collect':collect(a.n,a.protocol)
 else:score(a.data,a.protocol)
if __name__=='__main__':main()
