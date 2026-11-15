"""Evaluator-only paired analysis of frozen bottle runs."""
import gzip,json,sys
from pathlib import Path
import numpy as np
from run_store import create_run

def read(path):
    path=Path(path); files=sorted((path/'episodes').glob('*.json'))
    if files:return [json.loads(p.read_text()) for p in files]
    with gzip.open(path/'episodes_full.json.gz','rt') as f:return json.load(f)
a,b=map(read,sys.argv[1:3]);assert len(a)==len(b)==18
result=[]
for x,y in zip(a,b):
    assert x['condition']==y['condition'] and x['case']==y['case']
    pair={'condition':x['condition'],'case':x['case']['id']}
    for name,r in [('baseline',x),('conservative',y)]:
        e=r['result'].get('evaluation',{});s=e.get('snapshots',{});pre=s.get('before_close',{});close=s.get('after_close',{});lift=s.get('after_lift',{})
        c=lift.get('contacts',[]);held=any((z or '').startswith('fixed_jaw') for z in c) and any((z or '').startswith('moving_jaw') for z in c)
        shift=float(np.linalg.norm(np.asarray(close['object_xyz'])[:2]-np.asarray(pre['object_xyz'])[:2])) if pre and close else None
        t=r['telemetry_evaluator']
        pair[name]={'success':r['result']['success'],'failure':r['result'].get('failure'),'acquired':e.get('grasp_contact',False),'lifted':e.get('lifted',False),'held_at_lift_end':held,
         'dropped_during_transport':bool(e.get('lifted',False) and t['transport_table_contact_steps']>0),
         'transport_table_contact_physics_steps':t['transport_table_contact_steps'],'closure_push_m':shift,
         'relative_motion_m':t['max_grip_relative_motion_m'],'max_tilt_deg':t['max_tilt_deg'],'placement_error_m':r['placement_error_m'],
         'actions':r['actions'],'movement_rad':r['movement_rad'],'sim_s':r['sim_s'],'wall_s':r['wall_s'],'compute_s':r['compute_s'],
         'normal_force_peak_N':t['peak_normal_N'],'normal_impulse_Ns':t['normal_impulse_Ns'],
         'max_joint_tracking_error_rad':t['max_joint_tracking_error_rad'],'actuator_peak_Nm':t['max_actuator_effort_Nm'],
         'force_saturation_steps':t['force_saturation_steps'],'command_saturation_actions':r['command_saturation_actions'],
         'self_collision_steps':r['self_collision_physics_steps'],'joint_limit_steps':r['joint_limit_physics_steps'],'observed_friction':t['friction_observed']}
    result.append(pair)
out=create_run('bottle_analysis',{'inputs':sys.argv[1:3],'purpose':'paired evaluator-only analysis'})
with (out/'paired.json').open('x') as f:json.dump(result,f,indent=2)
for fill in ('empty','half','full'):
 for grip in ('dry','wet'):
    g=[x for x in result if x['condition']=={'fill':fill,'grip':grip}]
    print(fill,grip, 'baseline',sum(x['baseline']['success'] for x in g),'conservative',sum(x['conservative']['success'] for x in g),'n',len(g))
print('saved',out)
