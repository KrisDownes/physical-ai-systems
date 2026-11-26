"""Phase 5 Zero-Training Tactile-Amplifier Quantitative Diagnostic.

Compares matched BlindSpiral rollouts under Stiff (`ts = 1.0`) vs. `GravComp-TL` (`alpha = 0.35`)
across Splits A (bolted), B (unbolted), and C (tight_0.5mm) — zero training required — to verify
whether `GravComp-TL` lifts rim-contact `|tracking_error|` cleanly above the 12-bit STS3215
magnetic encoder quantization floor (`1 count = 2*pi/4096 = 0.001534 rad`):
  - Mean & median |tracking_error| (both L2 norm and max-joint abs) in rad and 12-bit encoder counts
    grouped by contact class (`free`, `plate_contact`, `rim_contact`, `entering`).
  - Fraction of `rim_contact` steps exceeding `2 encoder counts` (`> 0.00307 rad`).
  - Exact Mann-Whitney AUROC of |tracking_error| alone for separating `rim_contact` vs `plate_contact`
    (and `rim_contact` vs `free`) under Stiff vs `GravComp-TL`.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_insertion.benchmarks import (
    build_regime_a_bolted,
    build_regime_b_unbolted,
    build_split_c_tight_0_5mm,
)
from projects.blind_insertion.controllers import BlindSpiralController
from projects.blind_insertion.env import ENCODER_QUANT_RAD, InsertionTaskSpec
from projects.blind_insertion.phase1_train import _init_worker_single_thread
from projects.blind_insertion.phase3_gravcomp import GravCompBlindInsertionEnv
from projects.blind_insertion.phase3_train import GravCompBlindSpiralController

CONTACT_GROUPS: tuple[str, ...] = ('free', 'plate_contact', 'rim_contact', 'entering')


def exact_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC (handles ties cleanly, zero external dependencies)."""
    pos = np.asarray(pos_scores, dtype=float).reshape(-1)
    neg = np.asarray(neg_scores, dtype=float).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    all_s = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size, dtype=int), np.zeros(neg.size, dtype=int)])
    order = np.argsort(all_s, kind='mergesort')
    s_sorted = all_s[order]
    y_sorted = labels[order]

    # Assign average ranks for tied values (1-based)
    n = len(s_sorted)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_r = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_r
        i = j

    r_pos = float(np.sum(ranks[y_sorted == 1]))
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    u = r_pos - 0.5 * n_pos * (n_pos + 1.0)
    return float(u / (n_pos * n_neg))


def _collect_tactile_worker(args: tuple[str, InsertionTaskSpec]) -> dict:
    mode, task = args
    env = GravCompBlindInsertionEnv(task=task, render=False)
    try:
        obs, info = env.reset()
        ctrl = (
            BlindSpiralController(torque_scale=1.0)
            if mode == 'stiff_1.0'
            else GravCompBlindSpiralController(alpha=0.35)
        )
        ctrl.reset(obs)
        rec = {c: {'l2': [], 'max_j': [], 'arm_l2': []} for c in CONTACT_GROUPS}
        term = trunc = False
        while not (term or trunc):
            act = ctrl.act(obs, info)
            obs, _, term, trunc, info = env.step(act)
            c = info['privileged']['contact_class']
            if c in rec:
                err = np.abs(np.asarray(obs['tracking_error'][:5], dtype=float))
                rec[c]['l2'].append(float(np.linalg.norm(err)))
                rec[c]['max_j'].append(float(np.max(err)))
                rec[c]['arm_l2'].append(float(np.linalg.norm(err[1:4])))
        return {'rec': rec, 'success': bool(info['success'])}
    finally:
        env.close()


def _summarize_group(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {'n': 0}
    lsb = ENCODER_QUANT_RAD
    return {
        'n': int(arr.size),
        'mean_rad': round(float(np.mean(arr)), 6),
        'median_rad': round(float(np.median(arr)), 6),
        'mean_counts': round(float(np.mean(arr) / lsb), 3),
        'median_counts': round(float(np.median(arr) / lsb), 3),
        'p90_counts': round(float(np.percentile(arr, 90) / lsb), 3),
        'frac_gt_2counts': round(float(np.mean(arr > 2.0 * lsb)), 4),
        'frac_gt_5counts': round(float(np.mean(arr > 5.0 * lsb)), 4),
    }


def run_tactile_snr_diagnostic(workers: int = 10) -> dict:
    t0 = time.perf_counter()
    suites = {
        'split_a_bolted': build_regime_a_bolted(),
        'split_b_unbolted': build_regime_b_unbolted(),
        'split_c_tight_0.5mm': build_split_c_tight_0_5mm(),
    }
    modes = ('stiff_1.0', 'gravcomp_0.35')
    out: dict = {
        'encoder_12bit_count_rad': round(ENCODER_QUANT_RAD, 6),
        'two_counts_rad': round(2.0 * ENCODER_QUANT_RAD, 6),
        'by_suite': {},
        'pooled_abc': {},
    }

    ctx = mp.get_context('spawn')
    with ctx.Pool(workers, initializer=_init_worker_single_thread) as pool:
        pooled = {m: {c: {'l2': [], 'max_j': [], 'arm_l2': []} for c in CONTACT_GROUPS} for m in modes}
        for sk, trials in suites.items():
            out['by_suite'][sk] = {}
            for m in modes:
                rows = pool.map(_collect_tactile_worker, [(m, t) for t in trials])
                agg = {c: {'l2': [], 'max_j': [], 'arm_l2': []} for c in CONTACT_GROUPS}
                for r in rows:
                    for c in CONTACT_GROUPS:
                        for k in ('l2', 'max_j', 'arm_l2'):
                            agg[c][k].extend(r['rec'][c][k])
                            pooled[m][c][k].extend(r['rec'][c][k])
                suite_stats = {
                    'success_rate': round(sum(r['success'] for r in rows) / len(rows), 4),
                    'by_class_l2': {c: _summarize_group(np.asarray(agg[c]['l2'])) for c in CONTACT_GROUPS},
                    'by_class_max_j': {c: _summarize_group(np.asarray(agg[c]['max_j'])) for c in CONTACT_GROUPS},
                    'auroc_rim_vs_plate_l2': round(exact_auroc(np.asarray(agg['rim_contact']['l2']), np.asarray(agg['plate_contact']['l2'])), 4),
                    'auroc_rim_vs_plate_max_j': round(exact_auroc(np.asarray(agg['rim_contact']['max_j']), np.asarray(agg['plate_contact']['max_j'])), 4),
                    'auroc_rim_vs_free_l2': round(exact_auroc(np.asarray(agg['rim_contact']['l2']), np.asarray(agg['free']['l2'])), 4),
                }
                out['by_suite'][sk][m] = suite_stats

        for m in modes:
            out['pooled_abc'][m] = {
                'by_class_l2': {c: _summarize_group(np.asarray(pooled[m][c]['l2'])) for c in CONTACT_GROUPS},
                'by_class_max_j': {c: _summarize_group(np.asarray(pooled[m][c]['max_j'])) for c in CONTACT_GROUPS},
                'auroc_rim_vs_plate_l2': round(exact_auroc(np.asarray(pooled[m]['rim_contact']['l2']), np.asarray(pooled[m]['plate_contact']['l2'])), 4),
                'auroc_rim_vs_plate_max_j': round(exact_auroc(np.asarray(pooled[m]['rim_contact']['max_j']), np.asarray(pooled[m]['plate_contact']['max_j'])), 4),
                'auroc_rim_vs_free_l2': round(exact_auroc(np.asarray(pooled[m]['rim_contact']['l2']), np.asarray(pooled[m]['free']['l2'])), 4),
                'auroc_entering_vs_plate_l2': round(exact_auroc(np.asarray(pooled[m]['entering']['l2']), np.asarray(pooled[m]['plate_contact']['l2'])), 4),
            }

    # Explicit amplification ratios (GravComp / Stiff)
    p_st = out['pooled_abc']['stiff_1.0']
    p_gc = out['pooled_abc']['gravcomp_0.35']
    out['amplification_ratios_pooled'] = {
        c: {
            'median_l2_ratio': round(p_gc['by_class_l2'][c]['median_rad'] / max(p_st['by_class_l2'][c]['median_rad'], 1e-9), 3),
            'mean_l2_ratio': round(p_gc['by_class_l2'][c]['mean_rad'] / max(p_st['by_class_l2'][c]['mean_rad'], 1e-9), 3),
            'median_max_j_ratio': round(p_gc['by_class_max_j'][c]['median_rad'] / max(p_st['by_class_max_j'][c]['median_rad'], 1e-9), 3),
        }
        for c in CONTACT_GROUPS
    }
    out['wall_s'] = round(time.perf_counter() - t0, 2)
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--out', type=str, default=str(THIS_DIR / 'phase5_tactile_snr.json'))
    args = parser.parse_args()

    res = run_tactile_snr_diagnostic(workers=args.workers)
    Path(args.out).write_text(json.dumps(res, indent=2) + '\n')

    print(f'=== Phase 5 Tactile-Amplifier Diagnostic ({res["wall_s"]:.1f}s, 1 count = {ENCODER_QUANT_RAD:.6f} rad) ===')
    for m in ('stiff_1.0', 'gravcomp_0.35'):
        p = res['pooled_abc'][m]
        print(f'\n[{m}] Pooled Splits A+B+C (90 matched trials):')
        print(f'  AUROC (rim vs plate): L2={p["auroc_rim_vs_plate_l2"]:.4f} | max_j={p["auroc_rim_vs_plate_max_j"]:.4f} | AUROC (rim vs free, L2)={p["auroc_rim_vs_free_l2"]:.4f}')
        for c in CONTACT_GROUPS:
            g = p['by_class_l2'][c]
            gm = p['by_class_max_j'][c]
            print(
                f'  {c:14s} (n={g["n"]:5d}) | L2: mean={g["mean_rad"]:.5f}rad ({g["mean_counts"]:5.1f} cts), med={g["median_rad"]:.5f}rad ({g["median_counts"]:5.1f} cts), >2cts={100*g["frac_gt_2counts"]:5.1f}%'
                f' | max_j: med={gm["median_rad"]:.5f}rad ({gm["median_counts"]:5.1f} cts), >2cts={100*gm["frac_gt_2counts"]:5.1f}%'
            )
    print('\nAmplification Ratios (GravComp_0.35 / Stiff_1.0):')
    for c, rr in res['amplification_ratios_pooled'].items():
        print(f'  {c:14s}: median_L2={rr["median_l2_ratio"]:.2f}x | mean_L2={rr["mean_l2_ratio"]:.2f}x | median_max_j={rr["median_max_j_ratio"]:.2f}x')
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
