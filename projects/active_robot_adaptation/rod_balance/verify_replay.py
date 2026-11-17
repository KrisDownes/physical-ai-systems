"""Replay stored actions with the upstream visual mass marker hidden; evaluator only."""
import argparse,gzip,json,os,sys,time
from pathlib import Path
import imageio.v2 as imageio,numpy as np,yaml,mujoco
src=Path(os.environ.get('ASID_SRC','/tmp/active-asid-src')).resolve();sys.path[:0]=[str(src),str(Path(__file__).resolve().parents[1]/'published_manipulation')]
from env.wrappers.asid_vec import make_env
from run_store import create

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();path=Path(a.data).resolve();data=json.loads(path.read_text())['episodes'];out=create('rod_marker_hidden_replay',{'data':str(path),'scope':'evaluator verification: hide non-colliding visual marker, replay fixed recorded commands; no new selection or tuning'});os.chdir(src)
 with gzip.open(out/'rod_source.json.gz','wt') as f:json.dump({p.name:p.read_text() for p in Path(__file__).resolve().parent.glob('*.py')},f)
 rows=[];start=time.perf_counter()
 for n,r in enumerate(data):
  robot=yaml.safe_load(Path('configs/robot/sim_2_dof.yaml').read_text());obj=yaml.safe_load(Path('configs/env/rod.yaml').read_text());par=yaml.safe_load(Path('configs/asid/inertia.yaml').read_text());par['reward']=False
  env=make_env(robot,obj,par,seed=r['seed']);m=env.unwrapped._robot.model;idx=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'rod_visual_com');assert idx>=0 and m.geom_contype[idx]==0 and m.geom_conaffinity[idx]==0;m.geom_rgba[idx,3]=0
  env.parameter_dict['inertia']['value']=r['hidden_mass_offset_m_evaluator_training'];env.params_set=True;np.random.seed(r['seed']);obs=env.reset();observations=[obs.copy()];poses=[env.get_obj_pose().copy()];frames=[]
  if n==0:frames.append(env.render().copy())
  for i,action in enumerate(r['actions']):
   obs,_,_,_=env.step(np.array(action));observations.append(obs.copy());poses.append(env.get_obj_pose().copy())
   if n==0 and i%4==0:frames.append(env.render().copy())
  diff=float(np.max(np.abs(np.array(observations)-np.array(r['observations']))));pdiff=float(np.max(np.abs(np.array(poses)-np.array(r['rod_pose_evaluator']))));rows.append({'case':n,'observation_max_abs_difference':diff,'pose_max_abs_difference':pdiff,'physics_warnings':[int(w.number) for w in env.unwrapped._robot.data.warning]});print(n,'obs',diff,'pose',pdiff,flush=True)
  if frames:imageio.mimsave(out/'marker_hidden.gif',frames,duration=300,loop=0)
 (out/'summary.json').write_text(json.dumps({'input':str(path),'rows':rows,'max_observation_difference':max(r['observation_max_abs_difference'] for r in rows),'max_pose_difference':max(r['pose_max_abs_difference'] for r in rows),'wall_s':time.perf_counter()-start},indent=2));print('saved',out,flush=True)
if __name__=='__main__':main()
