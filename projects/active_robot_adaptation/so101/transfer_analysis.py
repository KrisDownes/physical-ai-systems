"""Evaluator-only retrospective scoring; never called by the controller."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from environment import Environment,Q_HOME

p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();run=Path(args.run)
rows=[json.loads(x.read_text()) for x in sorted((run/'episodes').glob('*.json'))]
assert len(rows)==36 and (run/'summary.json').exists(), 'complete frozen run required'
output=[]
for row in rows:
    case=row['case'];c=row['condition'];r=row['result']; loc=r.get('localization')
    if not loc: output.append({'case':case['id'],'condition':c['name'],'success':False,'failure':r.get('failure')});continue
    env=Environment(tool_marker=True,render=False,tool_offset=c['tool_offset_wrist_m'],camera_shift=c['camera_shift_m'])
    q=Q_HOME.copy();q[0]=case['initial_pan'];env.reset(case['object_xy'],q,seed=case['seed'])
    body=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'gripper');base=env.data.xmat[body].reshape(3,3).copy()
    obj_est=np.asarray(loc['object_estimate_xyz'][:2]); camera_error=np.asarray(loc['offset_xy'])-np.asarray(c['camera_shift_m'])
    stages={}
    for t in row['trace']:
        stage=t.get('stage')
        if stage not in ('approach','reach','lift','transport','lower'):continue
        env.data.qpos[:6]=t['joint_rad'];mujoco.mj_forward(env.model,env.data)
        rot=env.data.xmat[body].reshape(3,3); angle=float(np.degrees(np.arccos(np.clip((np.trace(base.T@rot)-1)/2,-1,1))))
        nominal,_=env.controller_fk(t['joint_rad']);actual=env.data.site_xpos[env.site].copy();truth_obj=np.asarray(t['object_xyz_evaluator'])
        # Predicted relative grasp-site versus observed object's actual relative position.
        relative=(nominal[:2]-obj_est)-(actual[:2]-truth_obj[:2])
        s=stages.setdefault(stage,[]);s.append({'angle':angle,'tool_mm':float(np.linalg.norm(nominal-actual)*1000),'relative_mm':float(np.linalg.norm(relative)*1000)})
    agg={k:{'max_rotation_deg':max(x['angle'] for x in v),'max_tool_error_mm':max(x['tool_mm'] for x in v),'last_relative_error_mm':v[-1]['relative_mm']} for k,v in stages.items()}
    snaps=r.get('evaluation',{}).get('snapshots',{}); before=snaps.get('before_close');after=snaps.get('after_settle')
    closure=None
    if before:
        env.data.qpos[:6]=before['joints_rad'];mujoco.mj_forward(env.model,env.data);nom,_=env.controller_fk(before['joints_rad']);actual=env.data.site_xpos[env.site].copy();true_obj=np.asarray(before['object_xyz']);closure=float(np.linalg.norm((nom[:2]-obj_est)-(actual[:2]-true_obj[:2]))*1000)
    output.append({'case':case['id'],'condition':c['name'],'references':c['references'],'success':r['success'],'failure':r.get('failure'),
       'camera_parameter_error_mm':(camera_error*1000).tolist(),'grasp_relative_prediction_error_mm':closure,
       'closure_site_minus_object_mm':None if not before else ((np.asarray(before['site_xyz'])-np.asarray(before['object_xyz']))*1000).tolist(),
       'final_placement_error_mm':None if not after else float(np.linalg.norm(np.asarray(after['object_xyz'][:2])-[.22,-.05])*1000),
       'stages':agg,'actions':row['actions'],'movement_rad':row['movement_rad'],'sim_s':row['simulation_s'],'compute_s':row['compute_s'],'wall_s':row['wall_s'],
       'calibration_events':r.get('calibration_events',[]),'self_collision_physics_steps':row['self_collision_physics_steps'],'joint_limit_physics_steps':row['joint_limit_physics_steps']})
    env.close()
with (run/'transfer_metrics.json').open('x') as f: json.dump(output,f,indent=2)
for ref in ('visible','unavailable'):
    group=[x for x in output if x.get('references')==ref]
    print(ref,'success',sum(x['success'] for x in group),'/',len(group),'closure error mm range',min(x['grasp_relative_prediction_error_mm'] for x in group),max(x['grasp_relative_prediction_error_mm'] for x in group),'max placement error mm',max(x['final_placement_error_mm'] for x in group if x['final_placement_error_mm'] is not None),'joint-limit steps',sum(x['joint_limit_physics_steps'] for x in group),'collisions',sum(x['self_collision_physics_steps'] for x in group))
