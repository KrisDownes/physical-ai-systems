"""Replay previous discrepancy with corrected pitch-response Jacobian."""
import json
from environment import Environment,Q_HOME
from nominal import run
from diagnose import overlay
from run_store import create_run

cfg={'case':'previous heldout-a, now development','object_xy':[.219,.017],
     'initial_pan':-.56,'camera_shift_m':[.008,.010],'seed':811,
     'methods':['nominal','online'],'tool_offset_wrist_m':[0,0,0]}
out=create_run('discrepancy_replay',cfg);rows=[]
for method in cfg['methods']:
    env=Environment(tool_marker=True,camera_shift=cfg['camera_shift_m'],action_budget=200,movement_budget=8)
    q=Q_HOME.copy();q[0]=cfg['initial_pan']
    r=run(env,cfg['object_xy'],q,method=method,seed=cfg['seed'],diagnostics=True)
    frames=r.pop('diagnostic_frames',[]);r.pop('frames',None)
    if frames:overlay(frames,cfg['camera_shift_m'],out/f'{method}_overlay.gif')
    rows.append({'method':method,'result':r,'trace':env.trace,'actions':env.action_count,
                 'collision_physics_steps':env.physics_collision_steps})
    print(method,r['success'],r['failure'],flush=True);env.close()
with (out/'replay.json').open('x') as f:json.dump(rows,f,indent=2)
print('saved',out)
