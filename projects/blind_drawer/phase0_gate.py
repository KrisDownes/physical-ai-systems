"""Phase 0 Physics Gate & Zero-Training Cross-Task Tactile-Amplifier Diagnostic for `projects/blind_drawer/`.

Evaluates on 8 CPU workers (`render=False`, bounded per-worker XML cache -> < 2 GB total RSS):
  1. Reference controllers across Split A (`bolted_sticky`, n=30), Split B (`unbolted_cabinet`, n=30),
     and Split OOD (`ood_light_height_angle`, n=20):
     - `Nominal_Stiff_1.0`         (straight-back pull, ts=1.0)
     - `Nominal_RawLowTorque_0.35` (straight-back pull, raw ts=0.35)
     - `Nominal_GravComp_0.35`     (straight-back pull, fixed GravComp-TL alpha=0.35)
     - `Privileged_Oracle`         (pulls along true rail_yaw with lateral yield + detent ramp)
  2. Zero-Training Tactile-Amplifier Diagnostic: median & mean `|tracking_error|` (in rad &
     12-bit STS3215 encoder counts) across `free_slide`, `off_axis_bind`, and `detent_catch`
     under Stiff (`ts=1.0`) vs. `GravComp-TL` (`alpha=0.35`), plus AUROC & amplification ratios.
  3. Headless throughput (steps/s), total RSS (GB), safety (`0/0`), and JSON report.
"""
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_drawer.benchmarks import (
    build_split_a_bolted_sticky,
    build_split_b_unbolted_cabinet,
    build_split_ood_light_height_angle,
)
from projects.blind_drawer.controllers import (
    NominalStraightBackController,
    PrivilegedOracleController,
)
from projects.blind_drawer.env import ENCODER_QUANT_RAD, BlindDrawerEnv, DrawerTaskSpec

# Bounded per-worker model cache keyed ONLY by structural topology (bolted, height_shift_mm) -> <= 4 models/worker!
_WORKER_ENV_CACHE: dict[tuple[bool, int], BlindDrawerEnv] = {}


def _init_worker() -> None:
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def get_rss_mb(pid: int | None = None) -> float:
    p = 'self' if pid is None else str(int(pid))
    try:
        with open(f'/proc/{p}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def _get_reusable_env(task: DrawerTaskSpec) -> BlindDrawerEnv:
    """Reuse compiled lean MjModel and update physics parameters in-place (zero XML recompile!)."""
    key = (bool(task.bolted), int(round(task.height_shift_m * 1000.0)))
    env = _WORKER_ENV_CACHE.get(key)
    if env is None:
        env = BlindDrawerEnv(task=task, render=False)
        _WORKER_ENV_CACHE[key] = env
    else:
        env.task = task
        env.servo_cfg = type(env.servo_cfg)(backlash_rad=task.backlash_rad)
        # Update drawer friction/damping, cabinet mass/inertia, table friction, and bolted pose in-place!
        m = env.model
        m.dof_frictionloss[env.drawer_dof_adr] = float(task.f_rail_N)
        m.dof_damping[env.drawer_dof_adr] = float(task.rail_damping)
        table_gid = int(m.geom('table').id)
        m.geom_friction[table_gid, 0] = float(task.mu_table)
        m_c = float(task.cabinet_mass_kg)
        scale = m_c / max(float(m.body_mass[env.cab_body_id]), 1e-9)
        m.body_mass[env.cab_body_id] = m_c
        m.body_inertia[env.cab_body_id] *= scale
        if task.bolted:
            import math
            h = 0.5 * math.radians(task.rail_yaw_deg)
            m.body_pos[env.cab_body_id] = [float(task.cabinet_xy[0]), float(task.cabinet_xy[1]), float(task.height_shift_m)]
            m.body_quat[env.cab_body_id] = [math.cos(h), 0.0, 0.0, math.sin(h)]
        import mujoco
        mujoco.mj_setConst(m, env.data)
    return env


def exact_auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if pos.size == 0 or neg.size == 0:
        return None
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)])
    order = np.argsort(scores, kind='mergesort')
    s_sorted = scores[order]
    l_sorted = labels[order]
    ranks = np.empty(scores.size, dtype=float)
    i = 0
    n = scores.size
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + 1 + j)
        i = j
    r_pos = float(np.sum(ranks[l_sorted == 1]))
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return float((r_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def _make_controller(cname: str):
    if cname == 'Nominal_Stiff_1.0':
        return NominalStraightBackController(mode='stiff', torque_scale=1.0)
    if cname == 'Nominal_RawLowTorque_0.35':
        return NominalStraightBackController(mode='low_torque', torque_scale=0.35)
    if cname == 'Nominal_GravComp_0.35':
        return NominalStraightBackController(mode='gravcomp', gravcomp_alpha=0.35)
    if cname == 'Privileged_Oracle':
        return PrivilegedOracleController()
    raise ValueError(f'Unknown controller {cname!r}')


def _eval_episode_worker(job: tuple[str, DrawerTaskSpec, bool]) -> dict:
    cname, task, record_snr = job
    env = _get_reusable_env(task)
    obs, info = env.reset()
    ctrl = _make_controller(cname)
    ctrl.reset(obs)
    terminated = False
    truncated = False
    snr_l2: dict[str, list[float]] = {'free_slide': [], 'off_axis_bind': [], 'detent_catch': []}
    snr_maxj: dict[str, list[float]] = {'free_slide': [], 'off_axis_bind': [], 'detent_catch': []}
    while not (terminated or truncated):
        act = ctrl.act(obs, info)
        obs, _, terminated, truncated, info = env.step(act)
        if record_snr:
            cc = info['privileged']['contact_class']
            if cc in snr_l2:
                te = np.asarray(obs['tracking_error'][:5], dtype=float)
                snr_l2[cc].append(float(np.linalg.norm(te)))
                snr_maxj[cc].append(float(np.max(np.abs(te))))
    priv = info['privileged']
    rec = {
        'trial_id': task.trial_id,
        'seed': task.seed,
        'bolted': task.bolted,
        'yaw_deg': round(task.rail_yaw_deg, 2),
        'height_shift_mm': round(task.height_shift_m * 1000.0, 1),
        'success': bool(info['success']),
        'failure_mode': info['failure_mode'],
        'steps': int(env.step_count),
        'final_travel_mm': round(float(priv['drawer_travel_mm']), 2),
        'time_s': round(float(info['time_s']), 3),
        'peak_contact_force_N': round(float(info['episode_peak_contact_force_N']), 2),
        'max_cab_disp_mm': round(float(info['episode_max_cab_disp_mm']), 2),
        'max_cab_tilt_deg': round(float(info['episode_max_cab_tilt_deg']), 2),
        'self_collision_steps': int(info['self_collision_steps']),
        'joint_limit_steps': int(info['joint_limit_steps']),
    }
    if record_snr:
        rec['snr_l2'] = snr_l2
        rec['snr_maxj'] = snr_maxj
    return rec


def summarize_suite(rows: list[dict]) -> dict:
    succ = [r for r in rows if r['success']]
    fms: dict[str, int] = {}
    for r in rows:
        fms[r['failure_mode']] = fms.get(r['failure_mode'], 0) + 1
    return {
        'trials': len(rows),
        'successes': len(succ),
        'success_rate': len(succ) / max(len(rows), 1),
        'median_travel_mm': round(float(np.median([r['final_travel_mm'] for r in rows])), 2),
        'median_time_s': round(float(np.median([r['time_s'] for r in succ])), 3) if succ else None,
        'median_peak_force_N': round(float(np.median([r['peak_contact_force_N'] for r in rows])), 2),
        'median_cab_disp_mm': round(float(np.median([r['max_cab_disp_mm'] for r in rows])), 2),
        'median_cab_tilt_deg': round(float(np.median([r['max_cab_tilt_deg'] for r in rows])), 2),
        'shove_failures': fms.get('cabinet_shoved', 0),
        'tip_failures': fms.get('cabinet_tipped', 0),
        'slip_failures': fms.get('handle_slip', 0),
        'self_collision_steps': int(sum(r['self_collision_steps'] for r in rows)),
        'joint_limit_steps': int(sum(r['joint_limit_steps'] for r in rows)),
        'total_steps': int(sum(r['steps'] for r in rows)),
        'failure_modes': fms,
    }


def _stats_for_arr(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {'n': 0}
    cts = arr / ENCODER_QUANT_RAD
    return {
        'n': int(arr.size),
        'mean_rad': round(float(np.mean(arr)), 6),
        'median_rad': round(float(np.median(arr)), 6),
        'mean_counts': round(float(np.mean(cts)), 2),
        'median_counts': round(float(np.median(cts)), 2),
        'frac_gt_2counts': round(float(np.mean(cts > 2.0)), 4),
        'frac_gt_5counts': round(float(np.mean(cts > 5.0)), 4),
    }


def main() -> None:
    t0 = time.perf_counter()
    suites = {
        'split_a_bolted_sticky': build_split_a_bolted_sticky(),
        'split_b_unbolted_cabinet': build_split_b_unbolted_cabinet(),
        'split_ood_light_height_angle': build_split_ood_light_height_angle(),
    }
    controllers = (
        'Nominal_Stiff_1.0',
        'Nominal_RawLowTorque_0.35',
        'Nominal_GravComp_0.35',
        'Privileged_Oracle',
    )
    snr_controllers = {'Nominal_Stiff_1.0', 'Nominal_GravComp_0.35'}
    snr_raw: dict[str, dict[str, dict[str, list[float]]]] = {
        c: {'l2': {'free_slide': [], 'off_axis_bind': [], 'detent_catch': []},
            'maxj': {'free_slide': [], 'off_axis_bind': [], 'detent_catch': []}}
        for c in snr_controllers
    }

    print('=== Phase 0 Blind Drawer Physics Gate & Zero-Training Tactile-Amplifier Check (8 CPU workers) ===', flush=True)
    results: dict[str, dict] = {}
    total_steps = 0
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=8, initializer=_init_worker) as pool:
        t_roll = time.perf_counter()
        for cname in controllers:
            results[cname] = {}
            rec_snr = cname in snr_controllers
            parts = []
            for sk, trials in suites.items():
                rows = pool.map(_eval_episode_worker, [(cname, t, rec_snr) for t in trials])
                if rec_snr:
                    for r in rows:
                        for k, vals in r.pop('snr_l2').items():
                            snr_raw[cname]['l2'][k].extend(vals)
                        for k, vals in r.pop('snr_maxj').items():
                            snr_raw[cname]['maxj'][k].extend(vals)
                summ = summarize_suite(rows)
                total_steps += summ['total_steps']
                results[cname][sk] = {'summary': summ, 'episodes': rows}
                parts.append(
                    f'{sk[:9]}={summ["successes"]:2d}/{summ["trials"]}({100*summ["success_rate"]:5.1f}%,'
                    f'F={summ["median_peak_force_N"]:4.1f}N,shv={summ["shove_failures"]:2d},slp={summ["slip_failures"]:2d})'
                )
            print(f'  {cname:26s} | ' + ' | '.join(parts), flush=True)

        # Measure exact total RSS across parent + all 8 live workers before closing pool!
        p_mb = get_rss_mb(None)
        w_mb = sum(get_rss_mb(w.pid) for w in pool._pool if w.pid is not None)
        tot_rss_gb = round((p_mb + w_mb) / 1024.0, 3)

        # Steady-state warm throughput benchmark (160 episodes across 8 workers with 100% cache hit!)
        t_bench = time.perf_counter()
        warm_jobs = [('Nominal_GravComp_0.35', t, False) for t in (suites['split_a_bolted_sticky'] + suites['split_b_unbolted_cabinet']) * 2]
        warm_rows = pool.map(_eval_episode_worker, warm_jobs)
        warm_steps = sum(r['steps'] for r in warm_rows)
        warm_wall = max(time.perf_counter() - t_bench, 1e-6)
        warm_sps = round(warm_steps / warm_wall, 1)

        p_mb2 = get_rss_mb(None)
        w_mb2 = sum(get_rss_mb(w.pid) for w in pool._pool if w.pid is not None)
        post_bench_rss_gb = round((p_mb2 + w_mb2) / 1024.0, 3)

    # Tactile-Amplifier SNR Summary (pooled across all 80 matched Split A + B + OOD trials)
    snr_summary: dict = {'encoder_12bit_count_rad': round(ENCODER_QUANT_RAD, 6), 'pooled_all_80': {}}
    for cname in ('Nominal_Stiff_1.0', 'Nominal_GravComp_0.35'):
        d_l2 = {k: np.asarray(v, dtype=float) for k, v in snr_raw[cname]['l2'].items()}
        d_mj = {k: np.asarray(v, dtype=float) for k, v in snr_raw[cname]['maxj'].items()}
        snr_summary['pooled_all_80'][cname] = {
            'by_class_l2': {k: _stats_for_arr(a) for k, a in d_l2.items()},
            'by_class_maxj': {k: _stats_for_arr(a) for k, a in d_mj.items()},
            'auroc_bind_vs_free_l2': exact_auroc(d_l2['off_axis_bind'], d_l2['free_slide']),
            'auroc_detent_vs_free_l2': exact_auroc(d_l2['detent_catch'], d_l2['free_slide']),
            'auroc_bind_vs_detent_l2': exact_auroc(d_l2['off_axis_bind'], d_l2['detent_catch']),
        }

    ratios = {}
    for k in ('free_slide', 'off_axis_bind', 'detent_catch'):
        st = snr_summary['pooled_all_80']['Nominal_Stiff_1.0']['by_class_l2'][k]
        gc_s = snr_summary['pooled_all_80']['Nominal_GravComp_0.35']['by_class_l2'][k]
        if st.get('n', 0) > 0 and gc_s.get('n', 0) > 0 and st['median_rad'] > 0:
            ratios[k] = round(gc_s['median_rad'] / st['median_rad'], 2)
    snr_summary['amplification_ratios_l2'] = ratios

    total_col = sum(results[c][sk]['summary']['self_collision_steps'] for c in controllers for sk in suites)
    total_jlim = sum(results[c][sk]['summary']['joint_limit_steps'] for c in controllers for sk in suites)
    wall_s = round(time.perf_counter() - t0, 2)

    print(f'\n=== Zero-Training Cross-Task Tactile-Amplifier SNR (80 trials A+B+OOD, 1 count = {ENCODER_QUANT_RAD:.6f} rad) ===')
    for k in ('free_slide', 'off_axis_bind', 'detent_catch'):
        st = snr_summary['pooled_all_80']['Nominal_Stiff_1.0']['by_class_l2'][k]
        gc_s = snr_summary['pooled_all_80']['Nominal_GravComp_0.35']['by_class_l2'][k]
        if st.get('n', 0) and gc_s.get('n', 0):
            print(
                f'  {k:15s} (n={st["n"]:4d}/{gc_s["n"]:4d}): Stiff med={st["median_counts"]:5.1f} cts (>2ct={100*st["frac_gt_2counts"]:5.1f}%, >5ct={100*st["frac_gt_5counts"]:5.1f}%) | '
                f'GC-0.35 med={gc_s["median_counts"]:5.1f} cts (>2ct={100*gc_s["frac_gt_2counts"]:5.1f}%, >5ct={100*gc_s["frac_gt_5counts"]:5.1f}%) | '
                f'ratio={ratios.get(k)}x'
            )

    print(
        f'\n=== PHASE 0 GATE FINISHED ({wall_s}s) | warm_throughput={warm_sps:.0f} st/s (8 workers) '
        f'| total_RSS={tot_rss_gb:.2f}GB -> {post_bench_rss_gb:.2f}GB | self_col={total_col}, jlim={total_jlim} ==='
    )

    out = {
        'wall_s': wall_s,
        'warm_throughput_steps_per_sec_8w': warm_sps,
        'total_rss_gb_eval': tot_rss_gb,
        'total_rss_gb_post_bench': post_bench_rss_gb,
        'total_eval_rollouts': len(controllers) * sum(len(v) for v in suites.values()),
        'total_eval_self_collision_steps': total_col,
        'total_eval_joint_limit_steps': total_jlim,
        'controllers': results,
        'tactile_snr': snr_summary,
    }
    out_path = THIS_DIR / 'phase0_results.json'
    out_path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'Saved results to {out_path}')


if __name__ == '__main__':
    main()
