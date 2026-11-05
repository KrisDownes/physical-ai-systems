"""Independent simulation outcomes; never imported by policy or verifier."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
import score as physical_score
from frames import compose,inverse


def load(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in path.read_text().splitlines()]
def wrapped(a):return np.arctan2(np.sin(a),np.cos(a))


def main(out):
    with contextlib.redirect_stdout(io.StringIO()):physical_score.main(out)
    navigation=load(out/'outcome.json');cfg=load(out/'cell.json')
    done=load(out/'integration_done.json') if (out/'integration_done.json').exists() else {}
    contract=load(out/'evaluation_contract.json');sensor=lines(out/'sensor_truth.jsonl')
    truth=[r for r in lines(out/'judge.jsonl') if r['type']=='truth']
    times=np.array([r['time'] for r in truth]);xy=np.array([[r['x'],r['y']] for r in truth])
    q=np.array([r['q'] for r in truth]);yaw=np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)))
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    rotation=np.r_[0,np.cumsum(abs(np.diff(yaw)))]
    observations=[]
    for s in sensor:
        pred=load(out/f'verifier_{s["id"]}.json') if (out/f'verifier_{s["id"]}.json').exists() else {}
        accepted=pred.get('decision')=='accept';value=pred.get('reading_value')
        lower,upper=contract['reference_interval'];error=None
        if accepted:error=[max(lower-value,0,value-upper),max(abs(value-lower),abs(value-upper))]
        valid=bool(accepted and error[1]<=contract['tolerance_MPa'])
        wrong=bool(accepted and error[0]>contract['tolerance_MPa'])
        identity=s['actual_asset_id']==contract['intended_asset_id']
        observations.append(dict(id=s['id'],time=s['time'],decision=pred.get('decision','missing'),
            reason=pred.get('reason'),reading_value=value,error_interval_MPa=error,
            numerical_valid_for_all_reference_values=valid,definitely_incorrect_numerical_acceptance=wrong,
            geometric_usability=s['geometrically_usable'],visibility=s['visibility'],
            actual_asset_correct=identity,identity_verified_by_method=False,
            usable_reading=bool(valid and identity and s['geometrically_usable']),
            incorrect_visibility_acceptance=bool(accepted and not s['geometrically_usable'])))
    recovery=None
    if done.get('recovery_goal_map') is not None:
        target=compose(inverse(cfg['map_from_mission']),done['recovery_goal_map'])
        after=observations[0]['time'] if observations else 0
        best=0.;start=None
        for j in range(1,len(times)):
            dt=times[j]-times[j-1]
            valid=(times[j]>=after and 0<dt<.15 and np.linalg.norm(xy[j]-target[:2])<=cfg['judge_position_m']
                   and abs(wrapped(yaw[j]-target[2]))<=cfg['judge_yaw_rad']
                   and np.linalg.norm(xy[j]-xy[j-1])/dt<.025 and abs(yaw[j]-yaw[j-1])/dt<.05)
            if valid:
                if start is None:start=times[j]
                best=max(best,times[j]-start)
            else:start=None
        endpoint=observations[-1]['time'] if len(observations)==2 else done.get('time',times[-1])
        p=np.array([np.interp(endpoint,times,xy[:,0]),np.interp(endpoint,times,xy[:,1])])
        recovery=dict(commanded_pose_mission=target.tolist(),physical_position_error_m=float(np.linalg.norm(p-target[:2])),
            heading_error_deg=float(np.rad2deg(abs(wrapped(np.interp(endpoint,times,yaw)-target[2])))),
            longest_valid_stationary_dwell_s=float(best),physical_arrival_pass=best>=cfg['dwell_s'])
    capture_success=any(o['usable_reading'] and o['time']<=cfg['max_sim_s'] for o in observations)
    actual=bool(navigation['mission_success'] and capture_success)
    first=observations[0]['time'] if observations else None
    end=done.get('time',times[-1])
    added_motion=float(np.interp(end,times,distance)-np.interp(first,times,distance)) if first is not None else None
    added_rotation=float(np.interp(end,times,rotation)-np.interp(first,times,rotation)) if first is not None else None
    controls=lines(out/'controller.jsonl')
    result=dict(scope='one synthetic-camera development mission; not physical camera validation',
        physical_initial_navigation_success=navigation['mission_success'],initial_stop=navigation['stops'],
        initial_claim_errors=navigation['claimed_stop_errors'],
        actual_simulated_inspection_completion=actual,arrival_or_policy_claim=done.get('claimed_task_complete',False),
        incorrect_task_completion_claim=bool(done.get('claimed_task_complete',False) and not actual),
        actual_asset_identity='intended fixture only; no identity challenge',identity_verified_by_method=False,
        observations=observations,abstentions=sum(o['decision']=='abstain' for o in observations),
        incorrect_numerical_acceptances=sum(o['definitely_incorrect_numerical_acceptance'] for o in observations),
        incorrect_visibility_acceptances=sum(o['incorrect_visibility_acceptance'] for o in observations),
        recovery_attempted=done.get('recovery_attempted',False),recovery_arrival=recovery,
        recovery_success=bool(len(observations)==2 and not observations[0]['usable_reading'] and observations[1]['usable_reading']
                              and recovery and recovery['physical_arrival_pass']),
        total_physical_path_m=float(np.interp(end,times,distance)),
        added_physical_motion_after_first_capture_m=added_motion,
        added_rotation_after_first_capture_rad=added_rotation,
        time_to_first_capture_s=first,final_policy_time_s=end,
        added_sim_time_after_first_capture_s=end-first if first is not None else None,
        recovery_wheel_path_m=done.get('recovery_wheel_path_m'),
        contact_messages=navigation['body_contact_messages'],contact_system_live=navigation['contact_system_live'],
        minimum_clearance_m=navigation['minimum_conservative_clearance_m'],
        obstacle_stop_ticks=sum(r['status']=='obstacle_stop' for r in controls),
        waiting_for_sensor_ticks=sum(r['status']=='waiting_for_sensors' for r in controls),
        navigation_failure=bool(not navigation['mission_success'] or (recovery is not None and not recovery['physical_arrival_pass'])),
        termination_reason=done.get('reason','missing_policy_result'),reset_count=1,
        caveats=['static photographic texture; pressure truth is only an existing visual interval',
                 'exact commissioned asset location and ideal pinhole/planar model',
                 'single asset, no deployable identity verification','shortened one-stop task, no dock/full-route claim'])
    (out/'integration_outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
