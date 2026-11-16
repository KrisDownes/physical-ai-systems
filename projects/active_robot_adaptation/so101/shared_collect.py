"""Collect visual transitions, retaining missingness and evaluator-only episode logs."""
import json,time
import numpy as np
from shared_tasks import SharedTask
from run_store import create_run

cfg=json.load(open('so101/shared_protocol.json'));out=create_run('shared_data',cfg);episodes=[]
for split in ('train','validation','test'):
 for task in cfg['tasks']:
  for ci,cond in enumerate(cfg[split+'_conditions']):
   for si,start in enumerate(cfg[split+'_starts']):
    reps=cfg['train_repeats'] if split=='train' else cfg['validation_repeats'] if split=='validation' else 1
    for rep in range(reps):
     seed=cfg['seed_base']+len(episodes);rng=np.random.default_rng(seed);e=SharedTask(task,**cond);f=e.reset(start['xy'],seed,start['pan']);xy=f[:2].copy()
     if not f[-1]:
      episodes.append({'split':split,'task':task,'condition':cond,'start':start,'seed':seed,'failure':'initial_visibility','transitions':[]});e.close();continue
     if task=='push':
      setup=[e.move_to([xy[0]-.0116,xy[1]+.060,.06]),e.move_to([xy[0]-.0116,xy[1]+.060,.018])];grip=1.2
     else:
      setup=[e.move_to([xy[0]-.0116,xy[1]-.0124,.06]),e.move_to([xy[0]-.0116,xy[1]-.0124,.014])];grip=-.1
      for _ in range(15):e.env.step(np.zeros(5),grip)
     e.last=e.env.observe();f=e.features(e.last);transitions=[];t0=e.truth()
     for t in range(cfg['steps_per_collection_episode']):
      # Same perturbation distribution in both tasks, with task-directed commands mixed in.
      action=np.array([rng.uniform(-.003,.003),rng.choice([-.012,-.009,-.006,.003,.009])],dtype=float)
      nxt=e.step(action,grip);transitions.append({'observation':f.tolist(),'action':action.tolist(),'next_observation':nxt.tolist(),'time_s':float(e.last['time_s'])});f=nxt
     episode={'split':split,'task':task,'condition_evaluator':cond,'start_evaluator':start,'seed':seed,'setup_reached':setup,'transitions':transitions,'object_displacement_evaluator_m':(e.truth()-t0).tolist(),'actions_including_setup':e.env.action_count,'movement_rad_including_setup':e.env.movement,'self_collision_physics_steps':e.env.physics_collision_steps,'joint_limit_physics_steps':e.env.physics_limit_steps,'numerical_invalid':e.numerical_reset or any(e.physics_warnings),'physics_warnings':e.physics_warnings}
     episodes.append(episode);e.close();print(split,task,ci,si,rep,'visibility',sum(t['observation'][-1] for t in transitions),flush=True)
     with (out/f'episode_{len(episodes):03d}.json').open('x') as h:json.dump(episode,h)
with (out/'episodes.json').open('x') as h:json.dump(episodes,h)
print('saved',out)
