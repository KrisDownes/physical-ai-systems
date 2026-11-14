"""Check sampled evaluation joint states for the known internal self-collision."""
import json
import argparse
from pathlib import Path
import mujoco
import numpy as np
from environment import load_model

parser=argparse.ArgumentParser()
parser.add_argument('--transfer',action='store_true')
parser.add_argument('--run',required=True,help='Unique run directory containing traces')
args=parser.parse_args()
out=Path(args.run)
prefix='transfer' if args.transfer else 'comparison'
traces=json.loads((out/f'{prefix}_traces.json').read_text())
model=load_model(); data=mujoco.MjData(model)
rows={}
for key,trace in traces.items():
    count=0; max_elbow=0.
    for sample in trace:
        mujoco.mj_resetData(model,data)
        data.qpos[:6]=sample['joint_rad']
        mujoco.mj_forward(model,data)
        max_elbow=max(max_elbow,abs(float(data.qpos[2])))
        hit=False
        for contact in data.contact[:data.ncon]:
            b1,b2=model.geom_bodyid[contact.geom1],model.geom_bodyid[contact.geom2]
            names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b1),
                   mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b2)}
            if 'shoulder' in names and names.intersection({'lower_arm','wrist','gripper'}) and contact.dist<0:
                hit=True;break
        count+=hit
    rows[key]={'sampled_states':len(trace),'sampled_self_collision_states':count,'max_abs_elbow_rad':max_elbow}
with (out/f'{prefix}_self_collision_audit.json').open('x') as f: f.write(json.dumps(rows,indent=2)+'\n')
print('episodes with sampled self-collision',sum(r['sampled_self_collision_states']>0 for r in rows.values()),'/',len(rows))
