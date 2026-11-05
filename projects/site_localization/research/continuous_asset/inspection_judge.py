"""Independent physical task judge. Never loaded by the robot policy."""
import json
import math
from pathlib import Path
import sys
import numpy as np


def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]


def score(out):
    contract=load(out/'private/evaluation.json');done=load(out/'task_done.json') if (out/'task_done.json').exists() else {}
    judge=rows(out.parent/'judge.jsonl');truth=[r for r in judge if r['type']=='truth']
    if len(truth)<2:raise ValueError('No physical trajectory')
    t=np.array([r['time'] for r in truth]);xy=np.array([[r['x'],r['y']] for r in truth]);q=np.array([r['q'] for r in truth])
    yaw=np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)))
    path=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    contacts=[r for r in judge if r['type']=='contacts' and r['count']]
    wheel=[r for r in judge if r['type']=='wheel_contacts' and r['count']]
    live=bool(wheel and wheel[-1]['time']>=t[-1]-.25)
    clearance=min(float(np.min(np.hypot(np.maximum(abs(xy[:,0]-ox)-sx/2,0),np.maximum(abs(xy[:,1]-oy)-sy/2,0))-.23))
                  for ox,oy,sx,sy in contract['physical_obstacles'])
    captures=rows(out/'private/sensor_truth.jsonl');metrics=[]
    for capture in captures:
        vp=out/f'vision_{capture["id"]:03d}.json';vision=load(vp) if vp.exists() else {}
        intended=next((a for a in capture['assets'] if a['actual_id']==contract['intended_id']),None)
        reading=vision.get('reading',{});accept=reading.get('decision')=='accept'
        value=reading.get('reading_value');low,high=contract['reference_interval']
        error=[max(low-value,0,value-high),max(abs(value-low),abs(value-high))] if accept else None
        valid_reading=bool(accept and error[1]<=contract['tolerance_MPa'])
        end=int(np.searchsorted(t,capture['time'],side='right'))-1;start=end
        for j in range(end,0,-1):
            dt=t[j]-t[j-1]
            if not(0<dt<.15 and np.linalg.norm(xy[j]-xy[j-1])/dt<.025 and abs(yaw[j]-yaw[j-1])/dt<.05):break
            start=j-1
        stationary=float(t[end]-t[start])
        physical=False;bearing=None;distance=None
        if intended:
            d=np.array(intended['asset_pose'][:2])-np.array(capture['robot_pose'][:2])
            distance=float(np.linalg.norm(d));bearing=abs(math.atan2(math.sin(math.atan2(d[1],d[0])-capture['robot_pose'][2]),math.cos(math.atan2(d[1],d[0])-capture['robot_pose'][2])))
            physical=bool(intended.get('usable_image') and .9<=distance<=1.3 and bearing<=math.radians(10)
                          and stationary>=2 and capture['time']<=contract['max_sim_s'])
        metrics.append(dict(id=capture['id'],time=capture['time'],observed_ids=vision.get('observed_ids',[]),
            identity_decision=vision.get('identity_decision','missing'),reading_decision=reading.get('decision','not_requested'),
            reading_value=value,error_interval_MPa=error,reading_valid=valid_reading,
            actual_target_visible=bool(intended and intended.get('usable_image')),distance_m=distance,
            heading_error_deg=None if bearing is None else math.degrees(bearing),stationary_before_capture_s=stationary,
            valid_physical_inspection=physical and valid_reading,
            wrong_asset_reading_accept=bool(accept and (intended is None or not intended.get('usable_image'))),
            incorrect_numeric_accept=bool(accept and error[0]>contract['tolerance_MPa'])))
    handoff=load(out.parent/'handoff.json');start=handoff['estimated_state']['time']
    safe=bool(not contacts and live and clearance>0 and done.get('time',float('inf'))<=650 and done.get('time',float('inf'))-start<=420 and done.get('wheel_path_m',float('inf'))<=18 and float(np.interp(done.get('time',t[-1]),t,path))<=35)
    valid=any(m['valid_physical_inspection'] for m in metrics)
    claim=bool(done.get('success_claim',False));reason=done.get('reason','missing_policy_result')
    events=rows(out/'method.jsonl')
    detections=[e for e in events if e.get('event')=='observation' and e.get('observed_asset_map')]
    first_detection=detections[0]['sensor_time'] if detections else None
    result=dict(case=contract['case'],target_present=contract['target_present'],success_claim=claim,
        correct_physical_task_completion=bool(claim and valid and safe),valid_inspection_obtained=valid,
        false_completion_claim=bool(claim and not(valid and safe)),safe_navigation=safe,
        wrong_asset_acceptances=sum(m['wrong_asset_reading_accept'] for m in metrics),
        incorrect_numeric_acceptances=sum(m['incorrect_numeric_accept'] for m in metrics),
        negative_detection_observations=sum(m['identity_decision']=='not_observed' for m in metrics),
        inspection_abstentions=sum(m['reading_decision']=='abstain' for m in metrics),
        observations=metrics,captures=len(captures),recovery_actions=done.get('recovery_actions'),
        search_positions=done.get('visited_search_positions'),first_target_detection_sim_s=first_detection,
        distance_to_first_target_detection_m=None if first_detection is None else float(np.interp(first_detection,t,path)),
        physical_path_m=float(np.interp(done.get('time',t[-1]),t,path)),wheel_path_m=done.get('wheel_path_m'),
        simulation_time_s=done.get('time',float(t[-1])),contact_events=len(contacts),contact_system_live=live,
        minimum_clearance_m=clearance,termination_reason=reason,
        navigation_failure=reason.startswith('navigation_'),infrastructure_failure=reason.startswith('infrastructure_') or not done,
        honest_noncompletion=bool(not claim and reason in ['target_not_observed_at_old_view','search_budget_exhausted',
            'search_candidates_exhausted_no_absence_proof','target_detected_no_reachable_approach','identity_uncertain',
            'identity_not_reacquired','identity_pose_inconsistent','inspection_unreadable_or_viewpoint_inadequate','stale_target_observation']),
        observable_coverage_certified=False,absence_claimed=False,
        scope='continuous acquired-map mission; full trajectory safety, ideal calibration, synthetic inspection')
    with (out/'outcome.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='observations'},indent=2))


if __name__=='__main__':score(Path(sys.argv[1]))
