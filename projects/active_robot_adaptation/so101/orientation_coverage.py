"""Evaluator-only analysis of already recorded task trajectories."""
import gzip,json
import numpy as np
import mujoco
from environment import Environment,Q_HOME
from run_store import create_run
from pathlib import Path

source=Path('results/so101_runs/20261114T165904.465176Z_ec797c5d8ac7_integrated_d805a79d/episodes_full.json.gz')
rows=json.load(gzip.open(source,'rt'))
out=create_run('orientation_coverage',{'source':str(source),'methods':['integrated'],
                                       'purpose':'retrospective evaluator-only diagnostics'})
results=[]
for r in rows:
    if r['method']!='integrated':continue
    c=r['condition'];case=r['case']
    env=Environment(tool_marker=True,render=False,tool_offset=c['tool_offset_wrist_m'],camera_shift=c['camera_shift_m'])
    q=Q_HOME.copy();q[0]=case['initial_pan'];obs=env.reset(case['object_xy'],q,seed=case['seed'])
    body=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'gripper')
    base=env.data.xmat[body].reshape(3,3).copy();calq=obs['joint_rad'].copy()
    offset=np.array(r['result']['localization']['offset_xy']);cam=np.array(c['camera_shift_m'])
    stage={}
    for t in r['trace']:
        name=t.get('stage','closure_or_release')
        env.data.qpos[:6]=t['joint_rad'];mujoco.mj_forward(env.model,env.data)
        rotation=env.data.xmat[body].reshape(3,3)
        angle=np.arccos(np.clip((np.trace(base.T@rotation)-1)/2,-1,1))
        nominal,_=env.controller_fk(t['joint_rad'])
        err=env.data.site_xpos[env.site].copy()-nominal
        x=stage.setdefault(name,{'rotations_deg':[],'tool_errors_m':[],'wrist_roll_rad':[]})
        x['rotations_deg'].append(float(np.rad2deg(angle)));x['tool_errors_m'].append(err.tolist());x['wrist_roll_rad'].append(t['joint_rad'][4])
    summary={k:{'samples':len(v['rotations_deg']),'max_rotation_deg':max(v['rotations_deg']),
                'max_tool_prediction_error_mm':float(np.max(np.linalg.norm(v['tool_errors_m'],axis=1))*1000),
                'wrist_roll_range_rad':[min(v['wrist_roll_rad']),max(v['wrist_roll_rad'])]} for k,v in stage.items()}
    results.append({'case':case['id'],'condition':c['name'],'camera_estimation_error_mm':((offset-cam)*1000).tolist(),
                    'calibration_joint_rad':calq.tolist(),'success':r['result']['success'],'stage':summary})
    env.close()
(out/'coverage.json').write_text(json.dumps(results,indent=2)+'\n')
print('saved',out)
print('max rotations by stage',{s:round(max((v['stage'][s]['max_rotation_deg'] for v in results if s in v['stage']),default=0),2) for s in ('approach','reach','transport','lower')})
print('max tool prediction error mm',max(max((q['max_tool_prediction_error_mm'] for q in v['stage'].values()),default=0) for v in results))
