"""Fixed, observation-guided rod push: author robot environment; no FIM reward."""
import argparse,gzip,json,os,sys,time
from pathlib import Path
import numpy as np,yaml,mujoco,imageio.v2 as imageio
src=Path(os.environ.get('ASID_SRC','/tmp/active-asid-src')).resolve()
sys.path[:0]=[str(src),str(Path(__file__).resolve().parents[1]/'published_manipulation')]
from env.wrappers.asid_vec import make_env
from run_store import create

def main():
 p=argparse.ArgumentParser();p.add_argument('--set',choices=['smoke','development','evaluation'],required=True);a=p.parse_args();os.chdir(src)
 if a.set=='smoke':cases=[(-.08,4101),(.08,4101)]
 elif a.set=='development':cases=[(x,4101+3*i+j) for i,x in enumerate([-.1,-.07,-.04,0,.04,.07,.1]) for j in range(3)]
 else:cases=[(x,5101+3*i+j) for i,x in enumerate([-.085,-.055,-.025,.025,.055,.085]) for j in range(3)]
 config={'scope':'development interaction data; hidden theta for evaluator/training only','set':a.set,'cases':cases,'20_actions':True,'push':'move to observed rod center minus 0.09m world x for 6 steps; move toward center plus 0.10m for 14 steps; each delta clipped ±0.07m','seeds':sorted(set(s for _,s in cases)),'sensor':'author noisy 18d robot/rod pose, no images; no hidden theta or forces in deployment features','reward':False}
 out=create('rod_push_'+a.set,config)
 with gzip.open(out/'rod_source.json.gz','wt') as f:json.dump({p.name:p.read_text() for p in Path(__file__).resolve().parent.glob('*.py')},f)
 rows=[];t0=time.perf_counter()
 for n,(theta,seed) in enumerate(cases):
  robot=yaml.safe_load(Path('configs/robot/sim_2_dof.yaml').read_text());obj=yaml.safe_load(Path('configs/env/rod.yaml').read_text());par=yaml.safe_load(Path('configs/asid/inertia.yaml').read_text());par['reward']=False
  env=make_env(robot,obj,par,seed=seed);env.parameter_dict['inertia']['value']=theta;env.params_set=True;np.random.seed(seed);obs=env.reset();d=env.unwrapped._robot.data;m=env.unwrapped._robot.model
  initial=env.get_obj_pose().copy();target=obs[11:13].copy();observations=[obs.tolist()];actions=[];poses=[initial.tolist()];contacts=0;frames=[];t=time.perf_counter();
  if n in (0,len(cases)-1): frames.append(env.render().copy())
  for i in range(20):
   goal=target+np.array([-.09,0]) if i<6 else target+np.array([.10,0]);action=np.clip(goal-obs[:2],-.07,.07);obs,_,done,_=env.step(action);d=env.unwrapped._robot.data
   names=[(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)) for c in d.contact[:d.ncon]]
   contacts+=any(('rod_geom_0' in pair and 'vention_table' not in pair) for pair in names)
   observations.append(obs.tolist());actions.append(action.tolist());poses.append(env.get_obj_pose().tolist())
   if n in (0,len(cases)-1) and i%4==0:frames.append(env.render().copy())
  row={'case':n,'hidden_mass_offset_m_evaluator_training':theta,'seed':seed,'observations':observations,'actions':actions,'rod_pose_evaluator':poses,'contact_observed_physics_steps_sampled_at_actions':contacts,'action_sim_seconds':2.,'wall_s':time.perf_counter()-t,'warnings':[int(w.number) for w in d.warning]};rows.append(row)
  (out/f'episode_{n:03}.json').write_text(json.dumps(row,indent=2))
  if frames:imageio.mimsave(out/f'episode_{n:03}.gif',frames,duration=300,loop=0)
  print(n,theta,seed,'contact samples',contacts,'displacement',np.linalg.norm(np.array(poses[-1][:3])-initial[:3]),'wall',round(row['wall_s'],2),flush=True)
 (out/'summary.json').write_text(json.dumps({'episodes':rows,'wall_s':time.perf_counter()-t0},indent=2));print('saved',out,flush=True)
if __name__=='__main__':main()
