"""Conventional completion + surface-fit screen; no mission certificate.

Consumes only map inputs, registration, and controller acquisition artifacts.
Physical outcomes are deliberately absent from the acceptance API.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from register_maps import conditioning, occupied_points, transform
from frames import compose, inverse

POLICY = dict(version='correction-development-v2-health-primary', retained_rmse_max_m=.05,
              inlier_distance_m=.12, minimum_source_overlap=.5,
              minimum_target_overlap=.5, minimum_surface_points=3,
              minimum_centered_point_rank=2, rank_tolerance_m=1e-6,
              require_controller_survey_completion=True,
              free_fit_comparator_requires_convergence=True,
              primary_requires_fixed_dock_frame=True,
              dock_stationary_position_tolerance_m=.01,
              dock_stationary_yaw_tolerance_rad=.01,
              ambiguity_and_conditioning_are_hard_gates=False)


def _read(value):
    return json.loads(Path(value).read_text()) if not isinstance(value, dict) else value


def _surface_points(path, max_points):
    with Path(path).open() as stream:
        h = np.fromstring(stream.readline(), sep=' ')
        grid = np.loadtxt(stream, dtype=int).reshape(int(h[1]), int(h[0]))
    free = grid == -1
    adjacent = np.zeros(grid.shape, dtype=bool)
    adjacent[1:] |= free[:-1]
    adjacent[:-1] |= free[1:]
    adjacent[:, 1:] |= free[:, :-1]
    adjacent[:, :-1] |= free[:, 1:]
    y, x = np.nonzero((grid == 1) & adjacent)
    points = np.c_[h[3] + x * h[2], h[4] + y * h[2]]
    if len(points) > max_points:
        points = points[np.linspace(0, len(points)-1, max_points, dtype=int)]
    return points


def _decision(reasons):
    return dict(decision='reject' if reasons else 'provisionally_accept',
                reasons=reasons, calibrated_integrity=False,
                physical_mission_certified=False)


def _stats(source, target, pose, parameters):
    if min(len(source), len(target)) < 3:
        return dict(available=False, reason='fewer_than_three_surface_points',
                    source_count=len(source), target_count=len(target))
    moved = transform(source, pose)
    tree = cKDTree(target)
    distances, ids = tree.query(moved)
    reverse, _ = cKDTree(moved).query(target)
    eligible = np.flatnonzero(distances <= parameters['max_correspondence'])
    ordered = eligible[np.argsort(distances[eligible], kind='stable')]
    retained = ordered[:max(3, int(np.ceil(parameters['trim_fraction']*len(ordered))))]
    rank = lambda x: int(np.linalg.matrix_rank(x-x.mean(axis=0), tol=POLICY['rank_tolerance_m']))
    return dict(available=True, source_count=len(source), target_count=len(target),
                retained_count=len(retained), eligible_count=len(eligible),
                retained_rmse_m=float(np.sqrt(np.mean(distances[retained]**2))) if len(retained) else None,
                source_overlap=float(np.mean(distances <= POLICY['inlier_distance_m'])),
                target_overlap=float(np.mean(reverse <= POLICY['inlier_distance_m'])),
                source_centered_rank=rank(source), target_centered_rank=rank(target),
                retained_source_centered_rank=rank(source[retained]) if len(retained) >= 3 else 0,
                local_geometry=conditioning(moved, target, tree, distances, ids,
                    SimpleNamespace(inlier_distance=POLICY['inlier_distance_m'])))


def _fit_reasons(stats, converged):
    reasons = []
    if not converged:
        reasons.append('registration_not_converged')
    if not stats['available']:
        return reasons + [stats['reason']]
    if stats['retained_rmse_m'] is None or stats['retained_rmse_m'] > POLICY['retained_rmse_max_m']:
        reasons.append('retained_rms_above_0.05m')
    if stats['source_overlap'] < POLICY['minimum_source_overlap']:
        reasons.append('source_overlap_below_0.5')
    if stats['target_overlap'] < POLICY['minimum_target_overlap']:
        reasons.append('target_overlap_below_0.5')
    return reasons


def _anchor_health(survey):
    """Check operational frame-gauge evidence, never simulator state."""
    try:
        with (survey/'mapper/mapper_pose.csv').open() as stream:
            first = next(csv.DictReader(stream))
        stamp = float(first['timestamp'])
        pose = np.array([float(first[k]) for k in ['x', 'y', 'yaw']])
        with (survey/'mapper/wheel_odometry.csv').open() as stream:
            prefix = [r for r in csv.DictReader(stream) if float(r['timestamp']) <= stamp]
        if not prefix:
            return dict(verified=False, reason='no_wheel_evidence_before_first_mapper_pose')
        odom = np.array([[float(r[k]) for k in ['x', 'y', 'yaw']] for r in prefix])
        position_change = float(np.max(np.linalg.norm(odom[:, :2]-odom[0, :2], axis=1)))
        yaw_change = float(np.max(np.abs(np.arctan2(np.sin(odom[:, 2]-odom[0, 2]),
                                                     np.cos(odom[:, 2]-odom[0, 2])))))
        with (survey/'controller.jsonl').open() as stream:
            # A command issued after receiving the first mapper observation can
            # share its simulation timestamp. Only strictly earlier commands
            # establish commanded motion before that frame observation.
            commands = [r for r in map(json.loads, stream) if r['time'] < stamp]
        no_commanded_motion = bool(commands) and all(abs(r['v']) <= 1e-9 and abs(r['w']) <= 1e-9 for r in commands)
        verified = bool(np.isfinite(pose).all() and no_commanded_motion
            and np.linalg.norm(pose[:2]) <= POLICY['dock_stationary_position_tolerance_m']
            and abs(pose[2]) <= POLICY['dock_stationary_yaw_tolerance_rad']
            and position_change <= POLICY['dock_stationary_position_tolerance_m']
            and yaw_change <= POLICY['dock_stationary_yaw_tolerance_rad'])
        return dict(verified=verified, first_mapper_pose=pose.tolist(), first_mapper_time_s=stamp,
                    max_wheel_position_change_m=position_change, max_wheel_yaw_change_rad=yaw_change,
                    no_commanded_motion=no_commanded_motion,
                    limitation='Checks first-node gauge and apparent stationarity, not physical docking calibration or repeatability.')
    except (OSError, StopIteration, KeyError, ValueError) as error:
        return dict(verified=False, reason='missing_or_invalid_anchor_evidence', detail=str(error))


def analyze(alignment, source_map, target_map, survey_dir, primary_transform=None,
            anchor_survey_dir=None, frame_policy='dock_anchor'):
    """Screen one candidate, optionally filtering original rasters in memory.

    On an old fit this changes ONLY gate inputs, leaving the fit unchanged. On
    a corrected surface fit the exact input-map hashes must still match. Never
    attach a different fitted transform to an old physical mission outcome.
    """
    if frame_policy not in ['dock_anchor', 'surface_fit']:
        raise ValueError('Unknown frame policy')
    a = _read(alignment)
    parameters = a['parameters']
    pose = np.asarray(a['target_from_source'], dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError('Nonfinite/malformed registration pose')
    if parameters.get('stable_region') is not None:
        raise ValueError('Whole-map common-dock configuration required')
    source, smeta = occupied_points(source_map, parameters['max_points'])
    target, tmeta = occupied_points(target_map, parameters['max_points'])
    for label, metadata in [('source', smeta), ('target', tmeta)]:
        if a[label]['sha256'] != metadata['sha256']:
            raise ValueError(f'{label} map hash differs from registration')
    ss = _surface_points(source_map, parameters['max_points'])
    ts = _surface_points(target_map, parameters['max_points'])
    raw = _stats(source, target, pose, parameters)
    surface = _stats(ss, ts, pose, parameters)
    survey = Path(survey_dir)
    done = survey/'controller_done.json'
    completion = dict(controller_done_present=done.exists(),
                      evidence='Controller route completion only; does not certify physical inspection or dock arrival.')
    completion_reasons = []
    if not done.exists():
        completion_reasons.append('controller_survey_incomplete')
    else:
        stamp = _read(done).get('time')
        completion['controller_done_time_s'] = stamp
        if not isinstance(stamp, (float, int)) or not np.isfinite(stamp):
            completion_reasons.append('controller_done_timestamp_invalid')
    converged = bool(a['selected']['converged'])
    primary_pose = pose if primary_transform is None else np.asarray(primary_transform, dtype=float)
    if primary_pose.shape != (3,) or not np.isfinite(primary_pose).all():
        raise ValueError('Nonfinite/malformed primary transform')
    primary_stats = _stats(ss, ts, primary_pose, parameters)
    anchor_survey = Path(anchor_survey_dir) if anchor_survey_dir is not None else survey
    anchor = _anchor_health(anchor_survey)
    anchor['survey_directory'] = str(anchor_survey.resolve())
    anchor_done = anchor_survey/'controller_done.json'
    anchor['commissioning_completion_present'] = anchor_done.exists()
    if not anchor_done.exists():
        anchor.update(verified=False, reason='anchor_commissioning_survey_incomplete')
    anchor['lineage_requirement'] = ('For reused graphs, caller must verify this immutable commissioning record '
        'belongs to the loaded serialized graph; a later relocalization pose is not a first graph node.')
    if primary_transform is not None and anchor['verified']:
        config = _read(anchor_survey/'controller_config.json')
        expected = compose(np.asarray(config['start']), inverse(np.asarray(anchor['first_mapper_pose'])))
        error = compose(inverse(expected), primary_pose)
        anchor['expected_target_from_source'] = expected.tolist()
        anchor['supplied_transform_error_m_rad'] = error.tolist()
        if frame_policy == 'dock_anchor' and (np.linalg.norm(error[:2]) > POLICY['dock_stationary_position_tolerance_m']
                or abs(error[2]) > POLICY['dock_stationary_yaw_tolerance_rad']):
            anchor.update(verified=False, reason='supplied_transform_violates_commissioned_dock_correspondence')
    # A measured fixed transform has no optimizer convergence condition.
    # The independent free-fit convergence remains an explicit diagnostic.
    health_reasons = completion_reasons.copy()
    if primary_transform is None:
        health_reasons.append('canonical_dock_transform_not_supplied' if frame_policy == 'dock_anchor'
                              else 'actual_surface_transform_not_supplied')
    if frame_policy == 'surface_fit':
        if not converged:
            health_reasons.append('surface_registration_not_converged')
        if not np.array_equal(primary_pose, pose):
            health_reasons.append('actual_transform_differs_from_selected_surface_fit')
    if not anchor['verified']:
        health_reasons.append('dock_frame_evidence_not_verified')
    if not primary_stats['available']:
        health_reasons.append(primary_stats['reason'])
    if primary_stats['available']:
        if min(primary_stats['source_count'], primary_stats['target_count']) < POLICY['minimum_surface_points']:
            health_reasons.append('insufficient_surface_support')
        if min(primary_stats['source_centered_rank'], primary_stats['target_centered_rank']) < POLICY['minimum_centered_point_rank']:
            health_reasons.append('collinear_surface_support')
    policy = POLICY.copy()
    if frame_policy == 'surface_fit':
        policy.update(version='surface-fit-development-v1', primary_requires_fixed_dock_frame=False,
                      primary_transform_must_equal_selected_fit=True,
                      primary_requires_registration_convergence=True,
                      dock_contract_role='Initial gauge/prior evidence only; not a metric alignment constraint')
    return dict(policy=policy, frame_policy=frame_policy,
                source_sha256=smeta['sha256'], target_sha256=tmeta['sha256'],
                alignment_implementation_sha256=a.get('implementation_sha256'),
                acceptance_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fitted_pose_unchanged=pose.tolist(), primary_transform=primary_pose.tolist(),
                primary_frame=('converged_surface_fit' if frame_policy == 'surface_fit' else
                               ('verified_dock_gauge' if primary_transform is not None else 'free_surface_fit')),
                survey=completion, anchor_evidence=anchor,
                diagnostics=dict(raw=raw, surface=surface, primary_surface=primary_stats,
                    independent_free_fit_converged=converged,
                    original_multistart_candidates=a.get('candidates', []),
                    note='Candidate losses retain their original fitting representation; local geometry is not calibrated covariance.'),
                rules=dict(finite_output=_decision([]),
                    residual_overlap=_decision(_fit_reasons(raw, converged)),
                    surface_residual_overlap=_decision(_fit_reasons(surface, converged)),
                    primary_surface_residual_overlap=_decision(_fit_reasons(primary_stats, True if primary_transform is not None else converged)),
                    fixed_pose_residual_and_health=_decision(
                        health_reasons + _fit_reasons(primary_stats, True if primary_transform is not None else converged)),
                    health_only=_decision(health_reasons),
                    operational=_decision(health_reasons)),
                limitations=['Prespecified conventional heuristics, selected using development evidence only.',
                    'Survey completion is not physical success; sensor/frame/calibration assumptions remain.',
                    'Majority surface overlap can reject valid heavily changed or incompletely observed sites.',
                    'Three noncollinear points establish only algebraic support, not spatial independence or robustness.',
                    'No confidence probability, mission-error protection level, free-space contradiction or scan-history integrity check.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--survey', type=Path, required=True)
    p.add_argument('--anchor-survey', type=Path,
        help='Immutable commissioning frame evidence for continued/reused graphs; current survey completion is still required')
    p.add_argument('--pose', type=float, nargs=3, help='Operational historical-from-map dock transform; never fit using physical judge')
    p.add_argument('--frame-policy', choices=['dock_anchor', 'surface_fit'], default='dock_anchor')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.alignment, args.source, args.target, args.survey, args.pose,
                     args.anchor_survey, args.frame_policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['rules'], indent=2))
