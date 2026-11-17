"""Author environment diagnostic; all simulator truth here is evaluator-only."""
import copy, gzip, json, os, sys, time
from pathlib import Path
import numpy as np, yaml, mujoco
src=Path(os.environ.get('ASID_SRC','/tmp/active-asid-src')).resolve()
sys.path[:0]=[str(src),str(Path(__file__).resolve().parents[1]/'published_manipulation')]
from env.wrappers.asid_vec import make_env
from run_store import create

def run():
 os.chdir(src)
 robot=yaml.safe_load(Path('configs/robot/sim_2_dof.yaml').read_text()); obj=yaml.safe_load(Path('configs/env/rod.yaml').read_text()); par=yaml.safe_load(Path('configs/asid/inertia.yaml').read_text())
 out=create('rod_contact_diagnostic',{'seed':3101,'parameter_m':0.07,'kind':'development, evaluator truth diagnostic; not a controller','policies':['stationary','guided push']})
 with gzip.open(out/'rod_source.json.gz','wt') as f:json.dump({p.name:p.read_text() for p in Path(__file__).resolve().parent.glob('*.py')},f)
 env=make_env(robot,obj,par,seed=3101);env.parameter_dict['inertia']['value']=.07;env.params_set=True;np.random.seed(3101);obs=env.reset();raw=env.unwrapped; m=raw._robot.model;d=raw._robot.data
 print('ee',obs[:3], 'rod',env.get_obj_pose(), 'time',d.time, flush=True)
 g=env.exp_reward.estimate_loss_grad(env.get_full_state(),np.zeros(2),params=env.get_parameters()).reshape(-1)
 print('gradient',np.round(g,6).tolist(),flush=True)
 rows=[]
 # Move toward rod then through its centre. This is a diagnostic oracle trajectory,
 # not the action policy to be evaluated.
 initial=env.get_obj_pose().copy(); target=initial[:2].copy();
 for i in range(20):
  o=obs.copy(); goal=target+np.array([-.09,0]) if i<6 else target+np.array([.10,0]); action=np.clip(goal-obs[:2],-.07,.07)
  before=env.get_obj_pose().copy();obs,reward,done,_=env.step(action);d=raw._robot.data
  contacts=[]
  for k in range(d.ncon):
   c=d.contact[k]; n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1);n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)
   if 'rod' in str(n1) or 'rod' in str(n2):contacts.append([n1,n2,float(c.dist)])
  r={'step':i,'action':action.tolist(),'ee_observed':obs[:3].tolist(),'rod_truth':env.get_obj_pose().tolist(),'rod_delta':float(np.linalg.norm(env.get_obj_pose()[:3]-before[:3])),'contacts':contacts,'reward':float(reward),'ctrl':d.ctrl.tolist()};rows.append(r)
  print(i,'ee',np.round(obs[:3],3),'rod',np.round(env.get_obj_pose()[:3],4),'contacts',contacts,'reward',reward,flush=True)
 (out/'trace.json').write_text(json.dumps({'initial_rod_pose':initial.tolist(),'gradient':g.tolist(),'rows':rows},indent=2));print('saved',out,flush=True)
if __name__=='__main__':run()
