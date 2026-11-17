"""Evaluator-only feasibility and setup checks for the frozen resistance pair."""
import json
from pathlib import Path
import numpy as np
import mujoco
from shared_tasks import SharedTask
from shared_evaluate import setup
from run_store import create_run
cfg=json.loads(Path('so101/decision_audit_protocol.json').read_text());out=create_run('decision_feasibility',cfg);rows=[]
for condition in cfg['condition_pair']:
 start=cfg['paired_start'];e=SharedTask('pull',**condition);f,grip,setting,initial=setup(e,'pull',start,cfg['paired_seed']);goal,reached=setting
 contacts=[]
 for c in e.env.data.contact[:e.env.data.ncon]:
  b1=int(e.env.model.geom_bodyid[c.geom1]);b2=int(e.env.model.geom_bodyid[c.geom2])
  if e.env.object_id not in (b1,b2):continue
  other=b2 if b1==e.env.object_id else b1;name=mujoco.mj_id2name(e.env.model,mujoco.mjtObj.mjOBJ_BODY,other)
  if name and name not in contacts:contacts.append(name)
 truth=[e.truth().tolist()];features=[f.tolist()]
 for a in cfg['paired_actions']:
  f=e.step([0.,a],grip);features.append(f.tolist());truth.append(e.truth().tolist())
 rows.append({'condition_evaluator':condition,'setup_reached':reached,'contact_bodies_after_setup_evaluator':contacts,'observations':features,'truth_positions_evaluator':truth,'goal_y_observed':goal,'slider_joint_rad_or_m':float(e.env.data.qpos[6]),'slider_joint_range_m':e.env.model.jnt_range[6].tolist(),'numerical_invalid':e.numerical_reset or any(e.physics_warnings),'warnings':e.physics_warnings,'joint_limit_physics_steps':e.env.physics_limit_steps,'self_collision_physics_steps':e.env.physics_collision_steps,'total_actions':e.env.action_count,'simulation_s':float(e.env.data.time)})
 e.close()
with (out/'results.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps(rows,indent=2));print('saved',out)
