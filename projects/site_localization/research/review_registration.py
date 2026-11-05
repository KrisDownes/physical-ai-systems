"""R1 fixed-candidate information ablation; offline, development evidence only.

Sensor-only ranking is written before importing any physical outcome trace.
The subsequent rigid fit is a privileged diagnostic, never a deployable method.
See docs/technical_review/experiment.md. No ROS or reserved-cohort access.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from frames import compose, inverse
from register_maps import occupied_points, transform, fit_rigid

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode()


def load(path):
    return json.loads(read(path))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def distance_summary(d):
    return dict(rms_m=float(np.sqrt(np.mean(d*d))), max_m=float(d.max()))


def orientation_groups(path):
    """Descriptive target-face proxy from free neighbors, not correspondences."""
    with path.open() as f:
        w, h, *_ = np.fromstring(f.readline(), sep=' ')
        a = np.loadtxt(f, dtype=int).reshape(int(h), int(w))
    free = a == -1
    x, y = np.zeros_like(free), np.zeros_like(free)
    x[:, 1:] |= free[:, :-1]; x[:, :-1] |= free[:, 1:]
    y[1:] |= free[:-1]; y[:-1] |= free[1:]
    group = np.full(a.shape, 'other', dtype='<U12')
    group[x & ~y] = 'vertical'
    group[y & ~x] = 'horizontal'
    group[x & y] = 'corner_like'
    return group[a == 1]


def rank(acq):
    a = load(acq/'surface_alignment.json')
    anchor = load(acq/'dock_anchor.json')['target_from_source']
    for p in ['source_surface.txt', 'target_surface.txt']:
        read(acq/p)
    source, sm = occupied_points(acq/'source_surface.txt')
    target, tm = occupied_points(acq/'target_surface.txt')
    assert sm['sha256'] == a['source']['sha256'] and tm['sha256'] == a['target']['sha256']
    assert sm['used_count'] == sm['occupied_count']  # orientation indexing unsampled
    assert tm['used_count'] == tm['occupied_count']
    tree = cKDTree(target)
    poses = [c['pose'] for c in a['candidates']]
    distances, indices = zip(*(tree.query(transform(source, np.array(p))) for p in poses))
    distances, indices = np.array(distances), np.array(indices)
    n = int(np.ceil(.7*len(source)))
    support = np.argsort(distances[0], kind='stable')[:n]
    discarded = np.setdiff1d(np.arange(len(source)), support)
    groups = orientation_groups(acq/'target_surface.txt')[indices[0]]
    rows = []
    for i, (p, d) in enumerate(zip(poses, distances)):
        scores = dict(trimmed_m2=float(np.mean(np.sort(d)[:n]**2)),
                      capped_m2=float(np.mean(np.minimum(d, .3)**2)),
                      huber_m2=float(np.mean(np.where(d <= .08, .5*d*d, .08*(d-.04)))))
        assert abs(scores['trimmed_m2']-a['candidates'][i]['trimmed_objective_m2']) < 1e-12
        assert abs(scores['capped_m2']-a['candidates'][i]['objective_m2']) < 1e-12
        rows.append(dict(index=i, pose=p, converged=a['candidates'][i]['converged'], **scores,
                         baseline_retained=distance_summary(d[support]),
                         baseline_discarded=distance_summary(d[discarded]),
                         by_baseline_target_face={k: distance_summary(d[groups == k])
                            for k in sorted(set(groups))}))
    winners = {k: min(range(len(rows)), key=lambda i: (rows[i][k], i))
               for k in ['trimmed_m2', 'capped_m2', 'huber_m2']}
    np.savez_compressed(acq_out := OUTPUT/acq.name.replace('__acquisition', '__support.npz'),
                        source=source, target=target, candidate_poses=poses,
                        nearest_distances=distances, nearest_target_indices=indices,
                        baseline_retained=support, baseline_discarded=discarded,
                        baseline_target_face=groups)
    return dict(acquisition=str(acq.relative_to(ROOT)), anchor=anchor,
                source_cells=len(source), target_cells=len(target), winners=winners,
                retained_count=n, discarded_count=len(discarded), candidates=rows,
                support_artifact=acq_out.name)


def claims(path):
    cfg = load(path/'controller_config.json')
    judge_cfg = load(path/'cell.json')
    assert judge_cfg['judge_position_m'] == .15 and judge_cfg['dwell_s'] == 2
    assert abs(judge_cfg['judge_yaw_rad']-np.deg2rad(10)) < 1e-12
    events, truth = [], []
    for line in read(path/'judge.jsonl').splitlines():
        r = json.loads(line)
        if r['type'] == 'truth': truth.append(r)
        if r['type'] == 'event' and r['event']['type'] == 'claimed_stop': events.append(r['event'])
    assert [e['goal'] for e in events] == list(range(6))
    times = np.array([r['time'] for r in truth])
    assert (np.diff(times) >= 0).all()
    q = np.array([r['q'] for r in truth])
    yaw = np.unwrap(np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2)))
    ps = np.c_[[r['x'] for r in truth], [r['y'] for r in truth], yaw]
    physical = []
    for e in events:
        t = e['time']; hi = np.searchsorted(times, t, side='right'); lo = hi-1
        assert 0 <= lo < hi < len(times) and times[hi]-times[lo] <= .15
        physical.append([np.interp(t, times, ps[:, k]) for k in range(3)])
    physical = np.array(physical)
    estimates = np.array([e['estimate'] for e in events])
    targets = np.array(cfg['goals'])
    s_from_m = inverse(cfg['map_from_mission'])
    site_estimates = compose(s_from_m, estimates)
    map_targets = compose(cfg['map_from_mission'], targets)
    assert np.max(abs(compose(s_from_m, map_targets)-targets)) < 1e-12
    outcome = load(path/'outcome.json')
    error = np.linalg.norm(physical[:, :2]-targets[:, :2], axis=1)
    assert np.max(abs(error-np.array([r['physical_position_error_m'] for r in outcome['claimed_stop_errors']]))) < 1e-10
    decomposition = []
    for i in range(6):
        decomposition.append(dict(goal=i, time_s=events[i]['time'], target_site=targets[i].tolist(),
            estimate_map=estimates[i].tolist(), physical_site=physical[i].tolist(),
            controller_residual_site=(site_estimates[i, :2]-targets[i, :2]).tolist(),
            physical_minus_estimated_site=(physical[i, :2]-site_estimates[i, :2]).tolist(),
            physical_error_site=(physical[i, :2]-targets[i, :2]).tolist()))
    return dict(outcome=outcome, decomposition=decomposition, estimates=estimates,
                physical=physical, cfg=cfg)


def evaluate(ranking, hard, free):
    h, f = claims(hard), claims(free)
    calibrations = {'dock': ranking['anchor']}
    for name, i in ranking['winners'].items(): calibrations[name] = ranking['candidates'][i]['pose']
    # Privileged diagnostic: actual poses, first four outbound visits only.
    oracle = fit_rigid(h['estimates'][:4, :2], h['physical'][:4, :2], np.ones(4))
    calibrations['oracle_outbound_only'] = oracle.tolist()
    residuals = {}
    for name, pose in calibrations.items():
        estimated = compose(pose, h['estimates'])
        d = np.linalg.norm(estimated[:, :2]-h['physical'][:, :2], axis=1)
        residuals[name] = dict(pose=pose, per_claim_position_m=d.tolist(),
                              outbound=distance_summary(d[:4]), later_return=distance_summary(d[4:]),
                              all_claims=distance_summary(d))
    # Predicted target displacement under the exact dock-frame diagnostic.
    targets = np.array(f['cfg']['goals'])
    predicted = compose(compose(ranking['anchor'], np.array(f['cfg']['map_from_mission'])), targets)
    shift = predicted[:, :2]-targets[:, :2]
    observed = f['physical'][:, :2]-targets[:, :2]
    return dict(hard_dock_decomposition=h['decomposition'], free_fit_decomposition=f['decomposition'],
                calibration_only_on_fixed_hard_dock_trace=residuals,
                predicted_dock_frame_goal_shift_m=shift.tolist(), observed_free_goal_error_m=observed.tolist(),
                prediction_difference_m=np.linalg.norm(shift-observed, axis=1).tolist(),
                hard_outcome={k:h['outcome'][k] for k in ['mission_success','physical_stops_completed','dock_return_success']},
                free_outcome={k:f['outcome'][k] for k in ['mission_success','physical_stops_completed','dock_return_success']})


def run(out):
    global OUTPUT
    OUTPUT = out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'sensor_rankings.json').exists(): raise FileExistsError('Use a new output directory')
    start, cpu = time.monotonic(), time.process_time()
    plan = load(ROOT/'results/correction_surface_policy_20261022/plan.json')
    rankings = [rank(Path(j['acquisition'])) for j in plan['jobs']]
    write(out/'sensor_rankings.json', rankings)  # before loading outcome/truth files
    write(out/'sensor_input_hashes.json', INPUTS.copy())
    evaluations = []
    for job, r in zip(plan['jobs'], rankings):
        hard = Path(job['acquisition'].replace('__acquisition', '__deployment'))
        evaluations.append(dict(layout=job['layout'], repeat=job['repeat'],
                                **evaluate(r, hard, Path(job['artifact_directory']))))
    write(out/'diagnostic_evaluation.json', evaluations)
    ledger = load(ROOT/'results/correction_cycle_20261022/ledger.json')
    for r in ledger['rows']:
        p = Path(r['artifact_directory'])/'outcome.json'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['outcome_sha256']
    write(out/'provenance.json', dict(inputs=INPUTS, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_wall_s=time.monotonic()-start, cpu_s=time.process_time()-cpu,
        original_outcome_hashes_verified=len(ledger['rows']), new_missions=0,
        warning='All analysis is development. Calibration residuals do not establish counterfactual mission outcomes.'))
    for j, r, e in zip(plan['jobs'], rankings, evaluations):
        print(j['layout'], j['repeat'], 'winner indices', r['winners'])
        print({k:round(v['all_claims']['max_m'],4) for k,v in e['calibration_only_on_fixed_hard_dock_trace'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
