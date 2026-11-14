"""Evaluator-only local geometric sensitivity; does not establish visual detectability."""
import json,sys
from pathlib import Path
import numpy as np
import mujoco
from environment import Environment,Q_HOME,CAMERA_HEIGHT,CAMERA_X

run=Path(sys.argv[1]); row=json.loads((run/'episodes'/'unavailable_c0_t0_holdout-a_calibration.json').read_text())
env=Environment(tool_marker=True,render=False);q=Q_HOME.copy();q[0]=row['case']['initial_pan'];env.reset(row['case']['object_xy'],q,seed=row['case']['seed'])
body=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'gripper');parent=env.model.body_parentid[body]
marker=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_SITE,'tool_marker_site')
def matrix(q):
    env.data.qpos[:6]=q;mujoco.mj_forward(env.model,env.data)
    x,y,z=env.data.site_xpos[marker];d=CAMERA_HEIGHT-z
    # Pixel Jacobian divided by focal length: [u/f,v/f] in metres/metres.
    p=np.array([[1/d,0,(x-CAMERA_X)/d**2],[0,-1/d,-y/d**2]])
    camera=np.array([[-1/d,0],[0,1/d]])
    return np.hstack([camera,p@env.data.xmat[parent].reshape(3,3)])
initial=matrix(env.data.qpos[:6]);samples=[initial]
for t in row['trace'][::5]:
    if t.get('stage') in ('approach','reach','lift','transport','lower'):samples.append(matrix(t['joint_rad']))
a=np.vstack(samples);sv=np.linalg.svd(a,compute_uv=False)
result={'interpretation':'Idealized local projection sensitivity, assumes accurately detected marker at every sampled state; actual occlusion and pixel noise reduce available information. Parameters are camera xy and parent-wrist offset xyz; unknown fixed overhead orientation and focal length are held fixed.', 'case':row['case'],'samples':len(samples),'single_observation_rank':int(np.linalg.matrix_rank(initial)), 'trajectory_rank':int(np.linalg.matrix_rank(a)),'trajectory_singular_values_divided_by_focal':sv.tolist(),'weakest_parameter_direction_camera_xy_then_parent_offset_xyz':np.linalg.svd(a)[2][-1].tolist()}
with (run/'observability.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2));env.close()
