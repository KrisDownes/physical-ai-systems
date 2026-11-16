"""Evaluator-only physical contact validation, excluded from training."""
import json
import numpy as np
import mujoco
from shared_tasks import SharedTask
from run_store import create_run
out=create_run('shared_contact_check',{'tasks':['push','pull'],'seed':71,'purpose':'development; inspect actual jaw/slider contact'})
rows=[]
for task in ('push','pull'):
 e=SharedTask(task);f=e.reset(seed=71);xy=f[:2]
 y=xy[1]+.060 if task=='push' else xy[1]-.0124
 e.move_to([xy[0]-.0116,y,.06]);e.move_to([xy[0]-.0116,y,.018 if task=='push' else .014]);grip=1.2 if task=='push' else -.1
 if task=='pull':
  for _ in range(15):e.env.step(np.zeros(5),grip)
 e.last=e.env.observe();start=e.truth().copy();contacts=[]
 prior_audit=e.env._audit_physics_contacts
 def audit():
  prior_audit()
  for c in e.env.data.contact[:e.env.data.ncon]:
   ids=[int(e.env.model.geom_bodyid[c.geom1]),int(e.env.model.geom_bodyid[c.geom2])]
   if e.env.object_id not in ids:continue
   other=c.geom2 if ids[0]==e.env.object_id else c.geom1
   body=mujoco.mj_id2name(e.env.model,mujoco.mjtObj.mjOBJ_BODY,int(e.env.model.geom_bodyid[other]))
   if body and body not in contacts:contacts.append(body)
 e.env._audit_physics_contacts=audit
 for _ in range(3):
  e.step([0,-.01],grip)
 rows.append({'task':task,'contact_bodies':contacts,'displacement_evaluator_m':(e.truth()-start).tolist(),'numerical_invalid':e.numerical_reset or any(e.physics_warnings),'warnings':e.physics_warnings,'self_collision_physics_steps':e.env.physics_collision_steps,'joint_limit_physics_steps':e.env.physics_limit_steps});e.close()
with (out/'results.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps(rows,indent=2));print('saved',out)
