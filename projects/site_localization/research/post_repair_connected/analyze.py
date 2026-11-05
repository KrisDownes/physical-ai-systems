"""Evaluator-only analysis of the fixed opened-layout development executions."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from continuous_asset_analysis import csv_pose,error_summary,truth_pose

OUT=ROOT/'results/post_repair_connected_20261027'
NAMES=['visible_infra_retry01','moved','occluded','wrong_asset','absent']
load=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(s) for s in p.read_text().splitlines()]

def at_path(time,truth,cum):
    return float(np.interp(time,truth[:,0],cum))

def main():
    results=[]
    for name in NAMES:
        p=OUT/('mission_'+name);result=load(p/'outcome.json');run=load(p/'run.json')
        truth=truth_pose(p);frame=load(p/'handoff.json')['site_from_slam']
        slam=csv_pose(p/'acquisition/mapper/mapper_tf_pose.csv')
        slam[:,1:3]+=np.array(frame[:2]);local=error_summary(slam,truth)
        amcl=error_summary(csv_pose(p/'inspection/estimates.csv'),truth)
        events=rows(p/'inspection/method.jsonl')
        sensors=rows(p/'inspection/private/sensor_truth.jsonl')
        capture_by_id={s['id']:s for s in sensors}
        target_errors=[]
        for e in events:
            if e['event']!='observation' or not e.get('observed_asset_map'):
                continue
            actual=next((a for a in capture_by_id[e['id']]['assets'] if a['actual_id']==7),None)
            if actual:
                target_errors.append(dict(id=e['id'],time=e['sensor_time'],
                    error_m=float(np.linalg.norm(np.array(e['observed_asset_map']['position'])-actual['asset_pose'][:2]))))
        detected=[dict(id=e['id'],time=e['sensor_time'],ids=e['vision'].get('observed_ids',[]))
                  for e in events if e['event']=='observation' and e['vision'].get('observed_ids')]
        final_id=(load(p/'inspection/task_done.json').get('final_observation') or {}).get('id')
        final_obs=next((o for o in result['all_observations'] if o['id']==final_id),None)
        results.append(dict(case=name,complete_contract=result['complete_contract'],reason=result['inspection_reason'],
            false_completion=result['false_completion'],wrong_asset_acceptances=result['wrong_asset_acceptances'],
            contacts=result['contacts'],clearance_m=result['clearance_m'],
            total_sim_s=result['total_sim_s'],total_path_m=result['total_path_m'],
            acquisition_sim_s=result['acquisition_time_s'],acquisition_physical_m=result['acquisition_physical_distance_m'],
            inspection_wheel_m=result['inspection_path_m'],captures=result['observations'],
            handoff_seed_error_m=result['handoff_seed_position_error_m'],
            acquisition_localization=local,inspection_localization=amcl,
            decoded_marker_events=detected,target_projection_errors=target_errors,
            final_observation_id=final_id,final_observation_physically_valid=bool(final_obs and final_obs['valid_physical_inspection']),
            resets=run['resets_after_start'],interventions=run['interventions'],
            all_processes_reaped=run['all_owned_processes_reaped']))
    wrong=OUT/'mission_wrong_asset';t=truth_pose(wrong)
    cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(t[:,1:3],axis=0),axis=1))]
    ev=rows(wrong/'inspection/method.jsonl')
    approach=next(e for e in ev if e['event']=='plan' and e['purpose']=='approach')
    handoff=load(wrong/'handoff.json')['estimated_state']['time']
    used=at_path(approach['time'],t,cum)-at_path(handoff,t,cum)
    target=next(a for a in load(wrong/'private/scene.json')['assets'] if a['id']==7)
    first=next(e for e in ev if e['event']=='observation' and 7 in e['vision'].get('observed_ids',[]))
    last_sensor=next(s for s in rows(wrong/'inspection/private/sensor_truth.jsonl') if s['id']==31)
    asset=next(a for a in last_sensor['assets'] if a['actual_id']==7)
    mechanism=dict(first_target_detection=dict(capture=first['id'],time_s=first['sensor_time']),
        confirmed_at_s=approach['time'],planned_approach_length_m=approach['path']['length_m'],
        evaluator_physical_inspection_path_used_at_confirmation_m=used,
        evaluator_remaining_of_18m_at_confirmation_m=18-used,
        planned_approach_exceeds_remaining_by_m=approach['path']['length_m']-(18-used),
        target_map_estimate_error_at_confirmation_m=next(q['error_m'] for q in results[3]['target_projection_errors'] if q['id']==31),
        final_detection_view=dict(physical_distance_m=asset['distance_m'],face_diameter_px=asset['face_diameter_px'],
            visible_fractions=asset['visibility'],usable_image=asset['usable_image']),
        finish_wheel_path_m=load(wrong/'inspection/task_done.json')['wheel_path_m'],
        target_pose_evaluator_only=target['pose'])
    summary=dict(protocol='docs/post_repair_connected/protocol.md',cases=results,
                 wrong_asset_failure_mechanism=mechanism,
                 attempts_started=sum(e['event']=='attempt_started' for e in rows(OUT/'attempts.jsonl')),
                 infrastructure_only_attempt='mission_visible; sandbox ROS getifaddrs denied; no physics; retained',
                 interpretation='Opened development cases in one shared layout; no reliability/generalization estimate')
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    for r in results:print(r['case'],r['complete_contract'],r['reason'],'acq_max',round(r['acquisition_localization']['position_max_m'],3),
                           'amcl_p95',round(r['inspection_localization']['position_p95_m'],3),'captures',r['captures'])
    print('wrong_asset',json.dumps(mechanism,indent=2))

if __name__=='__main__':main()
