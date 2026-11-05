"""Open a completed transfer cohort once; all later diagnosis is development."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def trajectory(path):
    rows=[json.loads(s) for s in (path/'judge.jsonl').read_text().splitlines()]
    truth=[r for r in rows if r['type']=='truth']
    t=np.array([r['time'] for r in truth]);q=np.array([r['q'] for r in truth])
    yaw=np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
    xy=np.array([[r['x'],r['y']] for r in truth])
    return t,xy,yaw


def stationary_dwell(t,xy,yaw,target,lower,upper,pos_tol,yaw_tol):
    best=0.;begin=None;completed=None
    for j in range(1,len(t)):
        dt=t[j]-t[j-1]
        speed=np.linalg.norm(xy[j]-xy[j-1])/max(dt,1e-12)
        rotation=abs(np.arctan2(np.sin(yaw[j]-yaw[j-1]),np.cos(yaw[j]-yaw[j-1])))/max(dt,1e-12)
        angle=abs(np.arctan2(np.sin(yaw[j]-target[2]),np.cos(yaw[j]-target[2])))
        valid=(lower<=t[j]<=upper and 0<dt<.15 and speed<.025 and rotation<.05
               and np.linalg.norm(xy[j]-target[:2])<=pos_tol and angle<=yaw_tol)
        if valid:
            if begin is None:begin=t[j]
            best=max(best,t[j]-begin)
            if completed is None and t[j]-begin>=2:completed=float(t[j])
        else:begin=None
    return float(best),completed


def mission(path):
    cfg=json.loads((path/'cell.json').read_text());run=json.loads((path/'run.json').read_text())
    outcome=json.loads((path/'outcome.json').read_text()) if (path/'outcome.json').exists() else None
    parts=path.name.split('__');role=parts[-1];condition=parts[1] if len(parts)>2 else 'unchanged'
    row=dict(run=path.name,artifact_directory=str(path),layout=parts[0],condition=condition,role=role,
        wall_s=run['wall_s'],cpu_s=run['children_cpu_user_s']+run['children_cpu_system_s'],
        largest_child_rss_kib=run['children_max_rss_kib'],stop_reason=run['stop_reason'],
        resets=run['reset_count'],serialization=run['serialization'],
        operator_events=cfg['operator_events'],all_owned_processes_reaped=run['all_owned_processes_reaped'])
    if outcome is None:return dict(**row,missing_outcome=True)
    row.update(mission_success=outcome['mission_success'],physical_stops=outcome['physical_stops_completed'],
        required_stops=outcome['required_stops'],dock_after_completed_inspections=outcome.get('dock_return_success'),
        dock_completion_s=outcome.get('dock_completion_time_s'),body_contacts=outcome['body_contact_messages'],
        min_clearance_m=outcome['minimum_conservative_clearance_m'],observed_sim_s=outcome['observed_sim_span_s'],
        claimed_stop_errors=outcome['claimed_stop_errors'])
    claims={e['goal']:e for e in outcome['claimed_stop_errors'] if e['reference_supported']}
    row['max_claim_position_m']=max((e['physical_position_error_m'] for e in claims.values()),default=None)
    row['max_claim_heading_deg']=max((e['physical_heading_error_deg'] for e in claims.values()),default=None)
    row['unchanged_claim_errors']={str(i):claims.get(i) for i in [0,1,4,5]}
    row['unchanged_position_regressions_m']={}
    for before,after in [(0,5),(1,4)]:
        if before in claims and after in claims:
            row['unchanged_position_regressions_m'][str(before)+'->'+str(after)]=claims[after]['physical_position_error_m']-claims[before]['physical_position_error_m']
    t,xy,yaw=trajectory(path)
    row['last_reference_time_s']=float(t[-1])
    row['wall_guard_before_sim_deadline']=(run['stop_reason']=='wall_guard' and t[-1]<cfg['max_sim_s'])
    row['physical_distance_m']=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
    row['visit_window_dwell_s']={}
    for i,claim in claims.items():
        lower=max((v['claim_time_s'] for k,v in claims.items() if k<i),default=float(t[0]))
        dwell,_=stationary_dwell(t,xy,yaw,np.array(cfg['goals'][i]),lower,claim['claim_time_s']+.15,
                                cfg['judge_position_m'],cfg['judge_yaw_rad'])
        row['visit_window_dwell_s'][str(i)]=dwell
    if 5 in claims:
        dock=cfg['dock_completion']
        dwell,completed=stationary_dwell(t,xy,yaw,np.array(dock['pose']),claims[5]['claim_time_s'],float(t[-1]),
                dock['position_tolerance_m'],dock['yaw_tolerance_rad'])
        row['physical_dock_after_last_claim']=bool(completed is not None and completed<=cfg['max_sim_s'])
        row['physical_dock_after_last_claim_s']=completed;row['physical_dock_dwell_s']=dwell
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    status=json.loads((root/'collection_status.json').read_text())
    if status['state']!='collection_complete_outcomes_not_inspected':raise ValueError('Collection incomplete; do not inspect outcomes')
    gate=root/'collection_gate.json'
    if gate.exists() and not json.loads(gate.read_text())['allowed_to_open']:raise ValueError('Setup recovery incomplete')
    marker=root/'outcomes_opened.json'
    if not marker.exists():marker.write_text(json.dumps(dict(
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline=status['baseline_revision'],
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='All cohort A outcomes now development evidence for subsequent methods. Final validation cohort remains ungenerated.'),indent=2)+'\n')
    rows=[mission(path.parent) for path in sorted(root.rglob('run.json'))]
    byname={r['run']:r for r in rows}
    registrations={e['run']:e for e in status['events'] if e['kind']=='registration'}
    for row in rows:
        if not row['role'].endswith('deployment'):continue
        method=row['role'].replace('_deployment','');prefix=row['layout']+'__'+row['condition']
        acquisition=byname.get(prefix+'__'+method+'_acquisition')
        row['candidate_acquisition_run']=acquisition['run'] if acquisition else None
        row['commissioning_shared_run']=row['layout']+'__commission' if method!='full' else None
        row['historical_information']='Privileged original map; empty remapping graph' if method=='full' else 'Acquired commissioning graph and its original alignment'
        row['recorded_repair_and_mission_wall_s']=row['wall_s']+(acquisition['wall_s'] if acquisition else 0)+registrations.get(acquisition['run'] if acquisition else '',{}).get('wall_s',0)
        row['acquisition_wall_s']=acquisition['wall_s'] if acquisition else 0
        row['registration_wall_s']=registrations.get(acquisition['run'] if acquisition else '',{}).get('wall_s',0)
        row['repair_wall_s']=row['acquisition_wall_s']+row['registration_wall_s']
        row['workflow_dock_reset_count']=1+(1 if acquisition else 0)
        if acquisition:
            row['acquisition_physical_distance_m']=acquisition.get('physical_distance_m')
            row['acquisition_completed_inspection_and_dock']=acquisition.get('mission_success')
    classifications={};observables=[];unjudged=[]
    for layout in ['maintenance_bays','l_workshop']:
        for condition in ['unchanged','additions','rearrangement']:
            prefix=layout+'__'+condition;gate=root/(prefix+'__full_acquisition')/'preacceptance.json'
            deployment=byname.get(prefix+'__full_deployment')
            if not gate.exists() or deployment is None:continue
            pre=json.loads(gate.read_text())
            diagnostics=pre['diagnostics'];fit=json.loads((gate.parent/'alignment.json').read_text())
            covariance={}
            log=gate.parent/'mapper/mapper_pose.csv'
            if log.exists():
                data=np.atleast_1d(np.genfromtxt(log,delimiter=',',names=True))
                for name in ['cov_xx','cov_yy','cov_yawyaw']:
                    values=data[name];values=values[np.isfinite(values)]
                    covariance[name]=dict(min=float(values.min()),median=float(np.median(values)),max=float(values.max())) if len(values) else None
            observables.append(dict(layout=layout,condition=condition,
                ranking_rmse_m=float(np.sqrt(fit['selected']['trimmed_objective_m2'])),
                retained_rmse_m=diagnostics['retained_rmse_m'],source_overlap=diagnostics['source_inlier_fraction'],
                target_overlap=diagnostics['target_inlier_fraction'],retained_bin_fraction=diagnostics['retained_target_bin_fraction'],
                retained_count=diagnostics['retained_count'],near_optimal_candidates=diagnostics['near_optimal_candidate_count'],
                max_competing_displacement_m=max((x['max_map_corner_displacement_m'] for x in diagnostics['near_optimal_candidates']),default=0),
                conditioning_available=diagnostics['existing_conditioning'].get('available'),
                mapper_pose_covariance_diagonal=covariance,
                covariance_scope='Official mapper local pose covariance; not inter-map transform covariance or calibrated integrity. Auxiliary AMCL is not the acquisition controller.',
                decisions={name:value['decision'] for name,value in pre['rules'].items()}))
            if 'mission_success' not in deployment:
                unjudged.append(deployment['run']);continue
            for name,decision in pre['rules'].items():
                item=classifications.setdefault(name,dict(candidates=0,accepted=0,useful_accepts=0,
                    incorrect_accepts=0,unnecessary_rejects=0,rejected_failures=0,missions=[]))
                accepted=decision['decision']=='provisionally_accept';success=bool(deployment.get('mission_success'))
                item['candidates']+=1;item['accepted']+=accepted
                key=('useful_accepts' if success else 'incorrect_accepts') if accepted else ('unnecessary_rejects' if success else 'rejected_failures')
                item[key]+=1;item['missions'].append(dict(run=deployment['run'],category=key,reasons=decision['reasons']))
    for item in classifications.values():
        item['acceptance_coverage']=item['accepted']/item['candidates']
        item['useful_operation_per_planned_cell']=item['useful_accepts']/6
    report=dict(split_at_collection='heldout_transfer_A',after_opening='development_for_any_subsequent_method',
        baseline=status['baseline_revision'],analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),missions=rows,acceptance_comparators=classifications,
        alignment_observables=observables,unjudged_acceptance_candidates=unjudged,
        missing_dependencies=[e for e in status['events'] if e['kind']=='dependency_missing' and not e.get('transport_recovery_record')],
        inference_limits='Purposive two-layout screen; one mapping realization per cell/method. Shared graphs and correlated stops. Counts are not real-world failure probabilities.',
        cost_limits='Concurrent wall time includes contention. Largest-child RSS not total peak memory. Diagnostic-AMCL overhead included. Preacceptance diagnostic time not separately tracked. Human reset/relocation seconds unmeasured; commissioning shared per layout.')
    (root/'transfer_result.json').write_text(json.dumps(report,indent=2,default=lambda value:value.item() if isinstance(value,np.generic) else str(value))+'\n')
    fields=['run','layout','condition','role','mission_success','physical_stops','required_stops',
        'dock_after_completed_inspections','physical_dock_after_last_claim','max_claim_position_m',
        'max_claim_heading_deg','body_contacts','min_clearance_m','observed_sim_s','physical_distance_m',
        'wall_s','acquisition_wall_s','registration_wall_s','repair_wall_s','recorded_repair_and_mission_wall_s','workflow_dock_reset_count','cpu_s','largest_child_rss_kib','stop_reason',
        'last_reference_time_s','wall_guard_before_sim_deadline']
    with (root/'missions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(missions=len(rows),acceptance={k:{x:v for x,v in item.items() if x!='missions'} for k,item in classifications.items()}),indent=2))


if __name__=='__main__':main()
