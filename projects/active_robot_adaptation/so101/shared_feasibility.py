"""Two explicitly bounded setup checks, with evaluator truth only in output."""
import json
import numpy as np
from shared_tasks import SharedTask
from run_store import create_run
out=create_run('shared_feasibility',{'tasks':['push','pull'],'seed':501,'planned_interaction_actions':12,'direction_world_y':-.01})
rows=[]
for kind in ('push','pull'):
 e=SharedTask(kind);f=e.reset(seed=501);xy=f[:2].copy();start=e.truth();print(kind,'start',xy,start,flush=True)
 if kind=='push':
  # Fixed open-jaw pusher placed on the +y side of the observed object.
  a=e.move_to([xy[0]-.0116,xy[1]+.060,.06]);b=e.move_to([xy[0]-.0116,xy[1]+.060,.018]);grip=1.2
 else:
  a=e.move_to([xy[0]-.0116,xy[1]-.0124,.06]);b=e.move_to([xy[0]-.0116,xy[1]-.0124,.014]);grip=-.1
  for _ in range(15):e.env.step(np.zeros(5),grip)
  e.last=e.env.observe()
 before=e.truth();vis=[]
 for i in range(12):
  obs=e.step([0.,-.01],grip);vis.append(int(obs[-1]));print(kind,i,'true y',e.truth()[1],'visible',vis[-1],flush=True)
 after=e.truth();r={'kind':kind,'start_evaluator_m':start.tolist(),'pre_motion_evaluator_m':before.tolist(),'end_evaluator_m':after.tolist(),'displacement_m':(after-before).tolist(),'setup_reached':[a,b],'visibility':vis,'self_collision_physics_steps':e.env.physics_collision_steps,'joint_limit_physics_steps':e.env.physics_limit_steps,'action_count':e.env.action_count,'trace':e.env.trace}
 rows.append(r);e.close()
with (out/'results.json').open('x') as f:json.dump(rows,f,indent=2)
print('saved',out)
