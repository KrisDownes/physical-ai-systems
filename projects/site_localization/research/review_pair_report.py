"""Analyze the completed fixed matched pair. Never used by either method."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from frames import compose, inverse
from transfer_report import mission, trajectory, stationary_dwell


def load(path):
    return json.loads(Path(path).read_text())


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def report(root):
    assert load(root/'collection_complete.json')['attempts'] == 2
    plan = load(root/'plan.json')
    reports, configs = [], []
    for job in plan['jobs']:
        p = Path(job['output'])
        cfg, run = load(p/'cell.json'), load(p/'run.json')
        controller_cfg = load(p/'controller_config.json')
        configs.append(controller_cfg)
        selection = load(p/'calibration_selection.json')
        t, xy, yaw = trajectory(p)
        heading = np.unwrap(yaw)
        events = [r['event'] for r in map(json.loads, (p/'judge.jsonl').read_text().splitlines())
                  if r['type'] == 'event' and r['event']['type'] == 'claimed_stop']
        metrics = mission(p)
        visits = []
        for j,e in enumerate(events):
            stamp, i = e['time'], e['goal']
            hi = int(np.searchsorted(t, stamp, side='right')); lo = hi-1
            supported = 0 <= lo < hi < len(t) and t[hi]-t[lo] <= .15
            row = dict(goal=i, claim_time_s=stamp, reference_supported=bool(supported))
            if supported:
                physical = np.array([np.interp(stamp,t,xy[:,0]),np.interp(stamp,t,xy[:,1]),
                                     np.interp(stamp,t,heading)])
                estimate = compose(selection['target_from_source'], e['estimate'])
                goal = np.array(cfg['goals'][i])
                residual = estimate[:2]-goal[:2]
                bias = physical[:2]-estimate[:2]
                error = physical[:2]-goal[:2]
                assert np.max(abs(residual+bias-error)) < 1e-12
                lower = events[j-1]['time'] if j else float(t[0])
                dwell, completed = stationary_dwell(t,xy,yaw,goal,lower,stamp+.15,
                                                    cfg['judge_position_m'],cfg['judge_yaw_rad'])
                row.update(physical_position_error_m=float(np.linalg.norm(error)),
                    physical_heading_error_deg=float(np.rad2deg(abs(wrap(physical[2]-goal[2])))),
                    stationary_dwell_in_visit_window_s=dwell,
                    individual_visit_meets_pose_heading_dwell=completed is not None,
                    controller_residual_site_m=residual.tolist(),
                    controller_residual_norm_m=float(np.linalg.norm(residual)),
                    physical_minus_estimated_site_m=bias.tolist(),
                    localization_calibration_error_norm_m=float(np.linalg.norm(bias)),
                    physical_position_error_site_m=error.tolist(),
                    controller_heading_residual_deg=float(np.rad2deg(wrap(estimate[2]-goal[2]))),
                    physical_minus_estimated_heading_deg=float(np.rad2deg(wrap(physical[2]-estimate[2]))),
                    estimated_site_pose=estimate.tolist(),physical_site_pose=physical.tolist(),
                    map_estimate=e['estimate'])
            visits.append(row)
        outcome = load(p/'outcome.json')
        for row, original in zip(visits, outcome['claimed_stop_errors']):
            assert row['goal'] == original['goal']
            if row['reference_supported']:
                assert abs(row['physical_position_error_m']-original['physical_position_error_m']) < 1e-10
        final = np.array(outcome['final_truth_pose'])
        dock = np.array(cfg['dock_completion']['pose'])
        reports.append(dict(objective=job['objective'],candidate_index=job['candidate_index'],
            artifact_directory=str(p),selection=selection['selected'],
            mission_success=outcome['mission_success'],ordered_stops=outcome['physical_stops_completed'],
            required_stops=outcome['required_stops'],dock_after_completed_inspections=outcome['dock_return_success'],
            physical_mission_completion_time_s=outcome['dock_completion_time_s'],
            physical_dock_after_last_claim=metrics['physical_dock_after_last_claim'] if 5 in [e['goal'] for e in events] else None,
            physical_dock_dwell_s=metrics.get('physical_dock_dwell_s'),
            final_dock_position_error_m=float(np.linalg.norm(final[:2]-dock[:2])),
            final_dock_heading_error_deg=float(np.rad2deg(abs(wrap(final[2]-dock[2])))),
            body_contacts=outcome['body_contact_messages'],contact_system_live=outcome['contact_system_live'],
            minimum_conservative_clearance_m=outcome['minimum_conservative_clearance_m'],
            final_reference_time_s=float(t[-1]),observed_sim_span_s=outcome['observed_sim_span_s'],
            controller_completion_time_s=load(p/'controller_done.json')['time'] if (p/'controller_done.json').exists() else None,
            stop_reason=run['stop_reason'],score_returncode=run['score_returncode'],reset_count=run['reset_count'],
            wall_s=run['wall_s'],child_cpu_s=run['children_cpu_user_s']+run['children_cpu_system_s'],
            largest_child_rss_kib=run['children_max_rss_kib'],all_processes_reaped=run['all_owned_processes_reaped'],
            per_visit=visits,outcome_sha256=hashlib.sha256((p/'outcome.json').read_bytes()).hexdigest()))
    assert {k:v for k,v in configs[0].items() if k!='map_from_mission'} == {
        k:v for k,v in configs[1].items() if k!='map_from_mission'}
    result = dict(scope='One fixed-map matched development pair, one unchanged environment. Visits are correlated.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        intended_runtime_config_differences=['map_from_mission'],actual_other_controller_inputs_match=True,
        attempts=load(root/'attempts.json'),missions=reports,reserved_access=False)
    (root/'paired_result.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'visits.csv').open('w') as f:
        keys=['objective','goal','claim_time_s','physical_position_error_m','physical_heading_error_deg',
              'stationary_dwell_in_visit_window_s','individual_visit_meets_pose_heading_dwell',
              'controller_residual_norm_m','localization_calibration_error_norm_m']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader()
        for r in reports:
            for v in r['per_visit']:writer.writerow(dict(objective=r['objective'],**v))
    print(json.dumps([{k:v for k,v in r.items() if k not in ['per_visit','selection']} for r in reports],indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    report(parser.parse_args().root.resolve())
