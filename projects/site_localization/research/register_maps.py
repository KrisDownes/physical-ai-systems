#!/usr/bin/env python3
"""Classical SE(2) occupancy-map registration; no ground-truth input.

Point-to-point ICP: Besl & McKay (1992), doi:10.1109/34.121791.
Trimming: Chetverikov et al. (2002), doi:10.1109/ICPR.2002.1047997.
Huber residual weighting is standard robust estimation, not a novel method.
Native map coordinates follow offline_amcl.cpp: origin + index * resolution
(no half-cell offset). Positive cells are occupied; zero is unknown.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree


def occupied_points(path, max_points=12000):
    with Path(path).open() as f:
        header = np.fromstring(f.readline(), sep=' ')
        if len(header) != 5:
            raise ValueError('Expected width height resolution origin_x origin_y')
        width, height = map(int, header[:2])
        res, ox, oy = header[2:]
        if width < 1 or height < 1 or res <= 0 or not np.isfinite(header).all():
            raise ValueError('Invalid map header')
        cells = np.loadtxt(f, dtype=int)
    if cells.size != width * height or not np.isin(cells, [-1, 0, 1]).all():
        raise ValueError('Invalid native map cells')
    y, x = np.nonzero(cells.reshape(height, width) == 1)
    points = np.c_[ox + x * res, oy + y * res]
    if len(points) < 3:
        raise ValueError('Need at least three occupied points')
    # Deterministic evenly spaced indexing, same point budget for both methods.
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    return points, dict(resolution_m=float(res), occupied_count=len(x),
                        used_count=len(points), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def transform(points, pose):
    return points @ rotation(pose[2]).T + pose[:2]


def wrap(yaw):
    return math.atan2(math.sin(yaw), math.cos(yaw))


def fit_rigid(source, target, weights):
    weights = weights / weights.sum()
    a, b = weights @ source, weights @ target
    u, _, vt = np.linalg.svd((source - a).T @ ((target - b) * weights[:, None]))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = b - r @ a
    return np.array([*t, math.atan2(r[1, 0], r[0, 0])])


def matches(distances, args, robust_phase=True):
    keep = distances <= args.max_correspondence
    if args.method == 'robust' and robust_phase and keep.sum() >= 3:
        ids = np.flatnonzero(keep)
        order = ids[np.argsort(distances[ids], kind='stable')]
        keep[:] = False
        keep[order[:max(3, int(np.ceil(args.trim_fraction * len(order))))]] = True
    return keep


def icp(source, target, tree, initial, args):
    pose = np.array(initial, dtype=float)
    converged = False
    iterations = 0
    for iterations in range(1, args.iterations + 1):
        # Standard coarse-to-fine initialization: do not discard the sparse
        # corners that constrain along-wall translation before initial fitting.
        robust_phase = iterations > min(15, args.iterations // 3)
        moved = transform(source, pose)
        distances, ids = tree.query(moved)
        keep = matches(distances, args, robust_phase)
        if keep.sum() < 3:
            break
        weights = np.ones(keep.sum())
        if args.method == 'robust' and robust_phase:
            weights = np.minimum(1., args.huber_delta / np.maximum(distances[keep], 1e-12))
        step = fit_rigid(moved[keep], target[ids[keep]], weights)
        pose[:2] = rotation(step[2]) @ pose[:2] + step[:2]
        pose[2] = wrap(pose[2] + step[2])
        if (np.linalg.norm(step[:2]) < args.translation_tolerance and abs(step[2]) < args.yaw_tolerance
                and (args.method == 'plain' or robust_phase)):
            converged = True
            break
    distances, _ = tree.query(transform(source, pose))
    # Bounded loss over ALL source cells penalizes unmatched geometry, rather
    # than rewarding an alignment supported by only a tiny surviving subset.
    loss = float(np.mean(np.minimum(distances, args.score_cap) ** 2))
    return dict(pose=pose.tolist(), objective_m2=loss, converged=converged,
                iterations=iterations, inlier_fraction=float(np.mean(distances <= args.inlier_distance)))


def summary(distances, inlier_distance):
    mask = distances <= inlier_distance
    return dict(rmse_m=float(np.sqrt(np.mean(distances ** 2))),
                median_m=float(np.median(distances)), p95_m=float(np.quantile(distances, .95)),
                inlier_fraction=float(mask.mean()), inlier_count=int(mask.sum()),
                inlier_rmse_m=float(np.sqrt(np.mean(distances[mask] ** 2))) if mask.any() else None)


def conditioning(moved, target, tree, distances, ids, args):
    keep = distances <= args.inlier_distance
    if keep.sum() < 3:
        return {'available': False, 'reason': 'fewer than three inliers'}
    q = moved[keep]
    center = np.mean(q, axis=0)
    radius = max(float(np.sqrt(np.mean(np.sum((q - center) ** 2, axis=1)))), 1e-6)
    # Point-to-plane diagnostic with local target normals reveals corridor
    # tangential weakness that fixed point correspondences can conceal.
    k = min(12, len(target))
    _, neighbours = tree.query(target[ids[keep]], k=k)
    local = target[neighbours]
    local = local - local.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', local, local)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    planar = eigenvalues[:, 0] <= .2 * np.maximum(eigenvalues[:, 1], 1e-12)
    normals = eigenvectors[:, :, 0]
    dq = (q - center) / radius
    jac = np.c_[normals, -normals[:, 0] * dq[:, 1] + normals[:, 1] * dq[:, 0]]
    jac = jac[planar]
    if len(jac) < 3:
        return {'available': False, 'reason': 'fewer than three locally linear target neighbourhoods'}
    info = jac.T @ jac / len(jac)
    eig = np.linalg.eigvalsh(info)
    return dict(available=True, normalized_point_to_plane_eigenvalues=eig.tolist(),
                condition_number=float(eig[-1] / eig[0]) if eig[0] > 1e-12 else None,
                rank=int(np.sum(eig > max(eig[-1] * 1e-6, 1e-12))),
                linear_neighbourhood_count=len(jac), rotation_length_scale_m=radius,
                warning='Local linearized geometry only; not covariance, calibration or global uniqueness.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='New/acquired native map')
    p.add_argument('--target', type=Path, required=True, help='Historical mission native map')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior', nargs=3, type=float, required=True, metavar=('X', 'Y', 'YAW'),
                   help='Dock-provided target_from_source prior; yaw radians')
    p.add_argument('--method', choices=['plain', 'robust'], default='robust')
    p.add_argument('--selection-objective',choices=['capped','trimmed'],default='capped',
                   help='Capped all-source loss or conventional fixed-fraction trimmed nearest-neighbour loss')
    p.add_argument('--stable-region',type=float,nargs=4,metavar=('XMIN','XMAX','YMIN','YMAX'),
                   help='Extra operator-supplied unchanged overlap in target frame; not inferred from simulator truth')
    p.add_argument('--translation-radius', type=float, default=.25,
                   help='Multistart x/y offsets {-radius, 0, +radius} metres')
    p.add_argument('--yaw-radius', type=float, default=.1, help='Multistart yaw offsets, radians')
    p.add_argument('--max-points', type=int, default=12000)
    p.add_argument('--max-correspondence', type=float, default=.6)
    p.add_argument('--trim-fraction', type=float, default=.7)
    p.add_argument('--huber-delta', type=float, default=.08)
    p.add_argument('--inlier-distance', type=float, default=.12)
    p.add_argument('--score-cap', type=float, default=.3)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--translation-tolerance', type=float, default=1e-5)
    p.add_argument('--yaw-tolerance', type=float, default=1e-5)
    args = p.parse_args()
    vals = [*args.prior, args.translation_radius, args.yaw_radius, args.max_correspondence,
            args.trim_fraction, args.huber_delta, args.inlier_distance, args.score_cap,
            args.translation_tolerance, args.yaw_tolerance]
    if (not np.isfinite(vals).all() or args.translation_radius < 0 or args.yaw_radius < 0
            or not 0 < args.trim_fraction <= 1 or args.max_points < 3 or args.iterations < 1
            or min(vals[5:]) <= 0):
        p.error('Require finite prior, nonnegative search radii, positive parameters and trim fraction in (0,1]')
    started = time.monotonic()
    source, smeta = occupied_points(args.source, args.max_points)
    target, tmeta = occupied_points(args.target, args.max_points)
    if args.stable_region:
        xmin,xmax,ymin,ymax=args.stable_region
        if not np.isfinite(args.stable_region).all() or xmin>=xmax or ymin>=ymax:p.error('Invalid stable region')
        def inside(points):return (points[:,0]>=xmin)&(points[:,0]<=xmax)&(points[:,1]>=ymin)&(points[:,1]<=ymax)
        source=source[inside(transform(source,np.array(args.prior)))];target=target[inside(target)]
        if min(len(source),len(target))<3:p.error('Stable region has insufficient occupied support')
        smeta['filtered_used_count']=len(source);tmeta['filtered_used_count']=len(target)
    tree = cKDTree(target)
    offsets = itertools.product(sorted(set([-args.translation_radius, 0., args.translation_radius])),
                                sorted(set([-args.translation_radius, 0., args.translation_radius])),
                                sorted(set([-args.yaw_radius, 0., args.yaw_radius])))
    candidates = [icp(source, target, tree, np.array(args.prior) + offset, args) for offset in offsets]
    for candidate in candidates:
        d,_=tree.query(transform(source,candidate['pose']))
        candidate['trimmed_objective_m2']=float(np.mean(np.sort(d)[:max(3,int(np.ceil(args.trim_fraction*len(d))))]**2))
    # Standard trimmed matching selects by its retained-support loss. Preserve
    # old capped-loss runs, rather than silently changing their interpretation.
    key='trimmed_objective_m2' if args.selection_objective=='trimmed' else 'objective_m2'
    candidates.sort(key=lambda c: (c[key], np.linalg.norm(np.array(c['pose']) - args.prior)))
    best = candidates[0]
    moved = transform(source, best['pose'])
    distances, ids = tree.query(moved)
    reverse, _ = cKDTree(moved).query(target)
    pose = best['pose']
    homogeneous = np.eye(3)
    homogeneous[:2, :2] = rotation(pose[2])
    homogeneous[:2, 2] = pose[:2]
    result = dict(method=args.method, target_from_source=pose,
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  assistance=('Externally declared unchanged target-frame region; additional assistance with unmeasured human time' if args.stable_region else 'Common dock prior only'),
                  target_from_source_matrix=homogeneous.tolist(), source=smeta, target=tmeta,
                  parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_to_target=summary(distances, args.inlier_distance),
                  target_to_source=summary(reverse, args.inlier_distance),
                  conditioning=conditioning(moved, target, tree, distances, ids, args),
                  candidates=candidates, selected=best, elapsed_wall_s=time.monotonic() - started,
                  frame_convention='p_target = R(yaw) p_source + [x,y]; radians; no half-cell offset',
                  limitations=['Local ICP; dock prior and overlap must be operationally available.',
                               'Symmetry and coherent site changes can produce wrong low-residual alignments.',
                               'Low residual and local conditioning do not establish physical mission correctness.',
                               'Reverse coverage reflects missing survey area as well as changed geometry.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['method', 'target_from_source', 'source_to_target', 'elapsed_wall_s']}))


if __name__ == '__main__':
    main()
