"""Evaluator-only feasibility check exposing true mounting geometry to FK."""
import json
import time
import numpy as np
import mujoco
from environment import Environment,Q_HOME
from nominal import run
from run_store import create_run

config={'tool_offsets_wrist_m':[[.006,0,0],[-.006,0,0]],'object_xy':[.216,.025],
        'initial_pan':-.6,'camera_shift_m':[0,0],'seed':921,
        'privilege':'diagnostic only: controller FK uses true disturbed model'}
out=create_run('geometry_oracle',config);rows=[]
for offset in config['tool_offsets_wrist_m']:
    env=Environment(tool_marker=True,tool_offset=offset,action_budget=200,movement_budget=8)
    oracle_data=mujoco.MjData(env.model)
    def oracle_fk(q):
        oracle_data.qpos[:6]=q;mujoco.mj_forward(env.model,oracle_data)
        jp,jr=np.zeros((3,env.model.nv)),np.zeros((3,env.model.nv))
        mujoco.mj_jacSite(env.model,oracle_data,jp,jr,env.site)
        return oracle_data.site_xpos[env.site].copy(),jp[:,:5].copy()
    env.controller_fk=oracle_fk
    q=Q_HOME.copy();q[0]=config['initial_pan'];start=time.perf_counter()
    r=run(env,config['object_xy'],q,method='fixed2',seed=config['seed'])
    r.pop('frames',None)
    rows.append({'tool_offset':offset,'result':r,'actions':env.action_count,
      'simulation_s':float(env.data.time),'wall_s':time.perf_counter()-start,
      'self_collision_physics_steps':env.physics_collision_steps})
    print(offset,r['success'],r['failure'],flush=True);env.close()
(out/'oracle_geometry.json').write_text(json.dumps(rows,indent=2)+'\n')
print('saved',out)
