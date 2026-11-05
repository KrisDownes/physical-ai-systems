"""Independent entire-mission judge. Truth is used here only, never for handoff."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import numpy as np
import inspection_judge
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def wrap(a):return np.arctan2(np.sin(a),np.cos(a))

def score(out):
    run=load(out/'run.json');cfg=load(out/'private/evaluation.json')
    allrows=rows(out/'judge.jsonl') if (out/'judge.jsonl').exists() else []
    truth=[x for x in allrows if x['type']=='truth']
    if len(truth)<2:
        (out/'outcome.json').write_text(json.dumps(dict(complete_contract=False,infrastructure_failure=True,reason='missing_physics'),indent=2)+'\n');return
    t=np.array([r['time'] for r in truth]);xy=np.array([[r['x'],r['y']] for r in truth]);q=np.array([r['q'] for r in truth])
    yaw=np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)))
    path=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    contacts=sum(r['type']=='contacts' and r['count']>0 for r in allrows)
    heart=[r['time'] for r in allrows if r['type']=='wheel_contacts' and r['count']>0];live=bool(heart and heart[-1]>=t[-1]-.25)
    clearance=min(float(np.min(np.hypot(np.maximum(abs(xy[:,0]-ox)-sx/2,0),np.maximum(abs(xy[:,1]-oy)-sy/2,0))-.23)) for ox,oy,sx,sy in cfg['physical_obstacles'])
    acq=load(out/'acquisition/acquisition_done.json') if (out/'acquisition/acquisition_done.json').exists() else {}
    dwell=0.;start=None
    for j in range(1,len(t)):
        dt=t[j]-t[j-1];valid=dt>0 and dt<.15 and np.linalg.norm(xy[j]-[-2,-2])<=.15 and abs(wrap(yaw[j]))<=math.radians(10) and np.linalg.norm(xy[j]-xy[j-1])/dt<.025 and abs(yaw[j]-yaw[j-1])/dt<.05
        if valid:
            if start is None:start=t[j-1]
            dwell=t[j]-start
        else:start=None;dwell=0.
    result=dict(case=cfg['case'],return_required=run['return_only'],correct_physical_inspection=False,complete_contract=False,success_claim=False,false_completion=False,wrong_asset_acceptances=0,
        contacts=contacts,contact_system_live=live,clearance_m=clearance,total_sim_s=float(t[-1]),total_path_m=float(path[-1]),
        acquisition=acq,acquisition_time_s=acq.get('time'),acquisition_physical_distance_m=float(np.interp(acq.get('time',t[-1]),t,path)),
        physical_dock_return=bool(dwell>=2),end_dock_dwell_s=dwell,handoff_success=False,interventions=run['interventions'],resets_after_start=run['resets_after_start'],
        infrastructure_failure=run['stop_reason'].startswith(('infrastructure','child_exit','wall_guard')),termination=run['stop_reason'])
    safe=contacts==0 and live and clearance>0
    if run['return_only']:
        result['complete_contract']=bool(safe and dwell>=2 and acq.get('reason')=='returned_to_dock' and acq['time']<=240 and acq['wheel_distance_m']<=20)
    elif (out/'inspection/task_done.json').exists():
        with contextlib.redirect_stdout(io.StringIO()):inspection_judge.score(out/'inspection')
        judged=load(out/'inspection/outcome.json');done=load(out/'inspection/task_done.json');handoff=load(out/'handoff.json')
        history=rows(out/'inspection/method.jsonl');ready=next((x for x in history if x['event']=='localization_ready'),None)
        h=handoff['estimated_state'];stamp=h['pose_time'];actual=np.array([np.interp(stamp,t,xy[:,0]),np.interp(stamp,t,xy[:,1]),np.interp(stamp,t,yaw)])
        seed=np.array(handoff['amcl_initial_pose']);seed_error=seed-actual;seed_error[2]=wrap(seed_error[2])
        before=np.array([np.interp(h['time'],t,xy[:,0]),np.interp(h['time'],t,xy[:,1])])
        end=ready['time'] if ready else h['time'];after=np.array([np.interp(end,t,xy[:,0]),np.interp(end,t,xy[:,1])])
        motion=float(np.linalg.norm(before-after));gap=t[(t>=h['time'])&(t<=end)]
        continuous=bool(len(gap)>1 and np.max(np.diff(gap))<=.15 and run['simulator_launches']==1 and run['resets_after_start']==0)
        acquired_valid=acq.get('reason')=='mapping_endpoint_ready' and acq.get('time',1e9)<=240 and acq.get('wheel_distance_m',1e9)<=20
        handoff_ok=bool(ready and continuous and motion<.05 and not handoff['ground_truth_used'])
        overall=bool(judged['correct_physical_task_completion'] and acquired_valid and handoff_ok and safe and done['time']<=650 and t[-1]<=650 and path[-1]<=35 and run['interventions']==0 and not result['infrastructure_failure'])
        result.update(correct_physical_inspection=judged['valid_inspection_obtained'],complete_contract=overall,success_claim=judged['success_claim'],false_completion=bool(judged['success_claim'] and not overall),
            wrong_asset_acceptances=judged['wrong_asset_acceptances'],inspection_abstentions=judged['inspection_abstentions'],incorrect_numeric_acceptances=judged['incorrect_numeric_acceptances'],
            observations=judged['captures'],recoveries=judged['recovery_actions'],navigation_failure=judged['navigation_failure'],inspection_reason=done['reason'],honest_noncompletion=judged['honest_noncompletion'],
            handoff_success=handoff_ok,handoff_physical_motion_m=motion,handoff_continuous_truth_coverage=continuous,
            handoff_seed_position_error_m=float(np.linalg.norm(seed_error[:2])),handoff_seed_heading_error_deg=float(np.degrees(abs(seed_error[2]))),
            first_localization_ready=ready,inspection_time_s=done['time']-h['time'],inspection_path_m=done['wheel_path_m'],task_decision_time_s=done['time'],
            decision_physical_path_m=float(np.interp(done['time'],t,path)),all_observations=judged['observations'])
    (out/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['all_observations','first_localization_ready','acquisition']},indent=2))

if __name__=='__main__':score(Path(sys.argv[1]))
