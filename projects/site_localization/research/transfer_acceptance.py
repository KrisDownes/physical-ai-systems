#!/usr/bin/env python3
"""Pre-outcome conventional registration screens, not calibrated integrity.

Only consumes the candidate alignment and its two input maps. Does not load
simulator state, mission outcomes, inspection targets or the next cohort.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from .register_maps import occupied_points, transform, wrap
except ImportError:
    from register_maps import occupied_points, transform, wrap


POLICY = {
    'version': '2026-10-22-prespecified-v1',
    'retained_rmse_max_m': .05,
    'source_inlier_fraction_min': .5,
    'target_inlier_fraction_min': .5,
    'spatial_bin_m': 1.,
    'retained_target_bins_fraction_min': .5,
    'near_optimal_extra_loss_m2': .025 ** 2,
    'competing_alignment_max_displacement_m': .10,
    'minimum_geometry_rank': 3,
    'maximum_geometry_condition_number': 1000.,
}


def _bins(points, size):
    return set(map(tuple, np.floor(points / size).astype(int)))


def _decision(reasons):
    return {'decision': 'reject' if reasons else 'provisionally_accept',
            'reasons': reasons, 'integrity_guarantee': False}


def analyze_alignment(alignment, source_map, target_map):
    """Return reproducible diagnostics and fixed conventional screen decisions.

    Paths can be strings or Paths; alignment can be a JSON path or a dict.
    Reconstruct final matches, not the optimizer's penultimate iteration.
    """
    a = (json.loads(Path(alignment).read_text())
         if not isinstance(alignment, dict) else alignment)
    p = a['parameters']
    source, smeta = occupied_points(source_map, p['max_points'])
    target, tmeta = occupied_points(target_map, p['max_points'])
    for name, actual in [('source', smeta), ('target', tmeta)]:
        if a[name]['sha256'] != actual['sha256']:
            raise ValueError(f'{name} map differs from alignment input')
    if p.get('stable_region') is not None:
        raise ValueError('Transfer screens require the frozen whole-map baseline')
    pose = np.asarray(a['target_from_source'], dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError('Nonfinite or malformed fitted transform')
    moved = transform(source, pose)
    distances, ids = cKDTree(target).query(moved)
    reverse, _ = cKDTree(moved).query(target)
    eligible = np.flatnonzero(distances <= p['max_correspondence'])
    order = eligible[np.argsort(distances[eligible], kind='stable')]
    retained = order[:max(3, int(np.ceil(p['trim_fraction'] * len(order))))]
    retained_rmse = (float(np.sqrt(np.mean(distances[retained] ** 2)))
                     if len(retained) else None)
    target_bins = _bins(target, POLICY['spatial_bin_m'])
    retained_bins = _bins(target[ids[retained]], POLICY['spatial_bin_m'])
    coverage = len(retained_bins) / len(target_bins)
    source_overlap = float(np.mean(distances <= p['inlier_distance']))
    target_overlap = float(np.mean(reverse <= p['inlier_distance']))
    # Use actual map-wide displacement so yaw disagreement has units of metres.
    # Corner probes are derived solely from occupied-map extent, not task truth.
    lo, hi = source.min(axis=0), source.max(axis=0)
    probes = np.array([[x, y] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]]])
    selected_loss = float(a['selected']['trimmed_objective_m2'])
    near = []
    for i, candidate in enumerate(a['candidates']):
        loss = float(candidate['trimmed_objective_m2'])
        if loss <= selected_loss + POLICY['near_optimal_extra_loss_m2']:
            displacement = float(np.max(np.linalg.norm(
                transform(probes, np.asarray(candidate['pose'])) - transform(probes, pose), axis=1)))
            near.append({'candidate_index': i, 'loss_m2': loss,
                         'max_map_corner_displacement_m': displacement,
                         'translation_difference_m': float(np.linalg.norm(np.asarray(candidate['pose'])[:2] - pose[:2])),
                         'yaw_difference_rad': float(abs(wrap(candidate['pose'][2] - pose[2])))})
    ambiguous = any(c['max_map_corner_displacement_m'] >
                    POLICY['competing_alignment_max_displacement_m'] for c in near)
    # Existing pipeline had no residual/coverage/conditioning acceptance rule.
    # Successfully constructed finite registration JSON was used provisionally.
    rules = {'legacy_output': _decision([])}
    fit_reasons = []
    if not a['selected']['converged']:
        fit_reasons.append('optimizer_not_converged')
    if retained_rmse is None or retained_rmse > POLICY['retained_rmse_max_m']:
        fit_reasons.append('retained_residual_above_one_grid_cell')
    if source_overlap < POLICY['source_inlier_fraction_min']:
        fit_reasons.append('source_overlap_below_majority')
    if target_overlap < POLICY['target_inlier_fraction_min']:
        fit_reasons.append('target_overlap_below_majority')
    rules['residual_overlap'] = _decision(fit_reasons)
    support_reasons = fit_reasons.copy()
    if coverage < POLICY['retained_target_bins_fraction_min']:
        support_reasons.append('retained_support_covers_less_than_half_occupied_bins')
    if ambiguous:
        support_reasons.append('competitive_multistart_alignment_exceeds_10cm_map_displacement')
    rules['residual_overlap_support_ambiguity'] = _decision(support_reasons)
    geometry_reasons = support_reasons.copy()
    cond = a.get('conditioning', {})
    if not cond.get('available'):
        geometry_reasons.append('local_geometry_diagnostic_unavailable')
    elif (cond['rank'] < POLICY['minimum_geometry_rank'] or
          cond['condition_number'] is None or
          cond['condition_number'] > POLICY['maximum_geometry_condition_number']):
        geometry_reasons.append('local_geometry_rank_or_condition_gate')
    rules['residual_overlap_support_ambiguity_geometry'] = _decision(geometry_reasons)
    return {
        'policy': POLICY.copy(),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': smeta['sha256'], 'target_sha256': tmeta['sha256'],
        'diagnostics': {
            'source_count': len(source), 'target_count': len(target),
            'eligible_correspondence_count': len(eligible), 'retained_count': len(retained),
            'retained_source_indices': retained.tolist(),
            'retained_target_indices': ids[retained].tolist(),
            'retained_rmse_m': retained_rmse,
            'retained_unique_target_count': len(np.unique(ids[retained])),
            'source_inlier_fraction': source_overlap, 'target_inlier_fraction': target_overlap,
            'occupied_target_bin_count': len(target_bins),
            'retained_target_bin_count': len(retained_bins),
            'retained_target_bin_fraction': coverage,
            'retained_target_bins': sorted([list(map(int, b)) for b in retained_bins]),
            'near_optimal_candidates': near,
            'near_optimal_candidate_count': len(near),
            'ambiguous_within_sampled_starts': ambiguous,
            'existing_conditioning': cond,
        },
        'rules': rules,
        'limits': [
            'Thresholds are prespecified engineering screens, not empirically calibrated probabilities.',
            'Only 27 local starts; no claim of exhaustive competing alignments or global uniqueness.',
            'Occupied-cell counts and 1 m bins are spatial support, not independent sensor evidence.',
            'Final optimizer support differs from trimmed ALL-source candidate-ranking support.',
            'Conditioning is local and dimension-normalized; it is not covariance or physical confidence.',
            'A passed screen does not establish physical mission correctness or detect nonrigid map errors.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze_alignment(args.alignment, args.source, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['rules'], indent=2))
