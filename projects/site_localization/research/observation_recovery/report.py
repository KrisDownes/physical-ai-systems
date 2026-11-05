"""Summarize retained attempts only; no robot, camera or verifier execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(study,out):
    out.mkdir(parents=True,exist_ok=False)
    freeze=load(study/'freeze.json');rows=[];capture_checks=[]
    for job in freeze['jobs']:
        folder=Path(job['output']);source=folder/'integration_outcome.json'
        if not source.exists():source=study/'scoring_recovery'/job['name']/'integration_outcome.json'
        result=load(source);done=load(folder/'integration_done.json');run=load(folder/'run.json')
        obs=result['observations'];first,last=obs[0]['time'],obs[-1]['time']
        truth=[r for r in lines(folder/'judge.jsonl') if r['type']=='truth']
        t=np.array([r['time'] for r in truth]);xy=np.array([[r['x'],r['y']] for r in truth])
        q=np.array([r['q'] for r in truth])
        yaw=np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)))
        path=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        rotation=np.r_[0,np.cumsum(abs(np.diff(yaw)))]
        for ob in obs:
            end=int(np.searchsorted(t,ob['time'],side='right'))-1;start=end
            for j in range(end,0,-1):
                dt=t[j]-t[j-1]
                if not(0<dt<.15 and np.linalg.norm(xy[j]-xy[j-1])/dt<.025 and abs(yaw[j]-yaw[j-1])/dt<.05):break
                start=j-1
            capture_checks.append(dict(case=job['name'],capture=ob['id'],capture_time=ob['time'],
                                       stationary_s_ending_at_capture=float(t[end]-t[start])))
        row=dict(case=job['name'],condition=job['condition'],policy=job['mode'],
            physical_arrival_pass=result['physical_initial_navigation_success'],
            initial_position_error_m=result['initial_stop'][0]['position_error_m'],
            initial_heading_error_deg=result['initial_stop'][0]['yaw_error_deg'],
            inspection_complete=result['actual_simulated_inspection_completion'],
            incorrect_task_claim=result['incorrect_task_completion_claim'],
            observations=len(obs),verifier_sequence=' -> '.join(o['decision'] for o in obs),
            abstentions=result['abstentions'],incorrect_reading_acceptances=result['incorrect_numerical_acceptances'],
            incorrect_visibility_acceptances=result['incorrect_visibility_acceptances'],
            recovery_attempted=result['recovery_attempted'],recovery_success=result['recovery_success'],
            physical_path_to_final_capture_m=float(np.interp(last,t,path)),
            added_path_between_captures_m=float(np.interp(last,t,path)-np.interp(first,t,path)),
            added_rotation_between_captures_rad=float(np.interp(last,t,rotation)-np.interp(first,t,rotation)),
            first_capture_sim_s=first,final_capture_sim_s=last,extra_capture_sim_s=last-first,
            verifier_total_wall_s=sum(o['verifier_wall_s'] for o in done['observations']),
            launcher_wall_s=run['wall_s'],contacts=result['contact_messages'],
            minimum_clearance_m=result['minimum_clearance_m'],navigation_failure=result['navigation_failure'],
            termination_reason=result['termination_reason'],original_score_returncode=run['score_returncode'],
            outcome_source=str(source.relative_to(study)))
        rows.append(row)
    pairs=[]
    for condition in ['clear','obstructed']:
        a,b=[r for r in rows if r['condition']==condition]
        pairs.append(dict(condition=condition,arrival_success=a['inspection_complete'],
            recovery_policy_success=b['inspection_complete'],
            difference_final_capture_sim_s=b['final_capture_sim_s']-a['final_capture_sim_s'],
            difference_total_physical_path_m=b['physical_path_to_final_capture_m']-a['physical_path_to_final_capture_m']))
    project=study.parents[1]
    protected=load(study/'preservation_before.json')
    mismatches=[p for p,h in protected.items() if sha(project/p)!=h]
    freeze_mismatches=[p for p,h in freeze['files'].items() if sha(Path(p))!=h]
    assert not mismatches and not freeze_mismatches
    equivalence=[]
    for condition in ['clear','obstructed']:
        a=study/'prepared'/f'{condition}__arrival_only';b=study/'prepared'/f'{condition}__recover_on_abstention'
        same={n:sha(a/n)==sha(b/n) for n in ['controller_config.json','map.txt','cell.sdf','sensor_scene.json','cell.json','evaluation_contract.json']}
        ca,cb=load(a/'policy_config.json'),load(b/'policy_config.json');ca.pop('mode');cb.pop('mode')
        same['policy_except_mode']=ca==cb
        assert all(same.values());equivalence.append(dict(condition=condition,same=same))
    summary=dict(scope='four development missions in one environment, two conditions, one pair each',
        missions=rows,paired_comparisons=pairs,capture_stationary_checks=capture_checks,
        physical_runs=4,resets=4,physical_retries=0,fresh_map_acquisitions=0,
        independent_environments=1,conditions=2,verifier_calls=sum(r['observations'] for r in rows),
        accepted_readings=sum(r['observations']-r['abstentions'] for r in rows),
        numerical_error_bounds_MPa=[0,.2958477104965569],reference_interval_unchanged=[6,6.5],
        identity_verified_by_method=False,real_camera_observations=0,
        all_attempts_preserved=True,owned_processes_reaped=True,
        original_scoring_failures=1,scoring_recovery='NumPy Boolean JSON serialization only; frozen scorer and original run retained',
        timing_caution='Use actual sensor capture times for recovery latency. Cached ROS clock in synchronous verifier callback is not end-to-end decision latency. Verifier wall times include process startup; reported separately.',
        contact_caution='Ground-truth conservative base-footprint clearance; overhead panel lies above base and lidar.',
        protected_files=len(protected),protection_mismatches=mismatches,freeze_mismatches=freeze_mismatches,
        input_equivalence=equivalence,freeze_sha256=sha(study/'freeze.json'))
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'missions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(pairs=pairs,missions=4,protected_files=len(protected)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.study.resolve(),a.out.resolve())
