"""Phase 0 Gate Verification & Benchmarks for Blind Proprioceptive Peg-in-Hole Insertion.

Runs:
1. Exact Clearance Drop Test across all 6 shapes at 1.0 mm and 0.5 mm clearance (12/12 gate).
2. Reference Controllers on Regime A (30 bolted) & Regime B (30 unbolted):
   - Nominal Straight-Down (torque_scale=1.0)
   - Classical Blind Spiral Search (Fixed Stiffness, torque_scale=1.0)
   - Classical Blind Spiral Search (Low Fixed Torque, torque_scale=0.35)
   - Privileged Oracle (knows exact socket center)
   Plus Privileged Oracle on the 6-trial tight-clearance (0.5 mm) suite.
3. Headless Multiprocessing Throughput Benchmark (8 and 10 CPU workers: steps/sec & episodes/sec).
4. Saves full machine-readable results to `projects/blind_insertion/phase0_results.json`.
"""
from __future__ import annotations

from collections import Counter
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
    build_tight_clearance_suite,
)
from projects.blind_insertion.controllers import (
    BlindSpiralController,
    NominalStraightDownController,
    PrivilegedOracleController,
    run_episode,
)
from projects.blind_insertion.env import BlindInsertionEnv, InsertionTaskSpec
from projects.blind_insertion.geometry import run_all_drop_tests


def _make_controller(name: str):
    if name == 'Nominal_StraightDown':
        return NominalStraightDownController(torque_scale=1.0)
    if name == 'BlindSpiral_Stiff_1.0':
        return BlindSpiralController(torque_scale=1.0)
    if name == 'BlindSpiral_LowTorque_0.35':
        return BlindSpiralController(torque_scale=0.35)
    if name == 'Privileged_Oracle':
        return PrivilegedOracleController()
    raise ValueError(f'Unknown controller: {name}')


def _eval_worker(args: tuple[str, InsertionTaskSpec]) -> dict:
    ctrl_name, task = args
    env = BlindInsertionEnv(task=task, render=False)
    try:
        ctrl = _make_controller(ctrl_name)
        return run_episode(env, task, ctrl)
    finally:
        env.close()


_BENCH_ENV: BlindInsertionEnv | None = None


def _init_throughput_worker(bolted: bool) -> None:
    global _BENCH_ENV
    spec = InsertionTaskSpec(shape='round', clearance_m=0.0012, bolted=bolted, max_steps=150)
    _BENCH_ENV = BlindInsertionEnv(task=spec, render=False)


def _throughput_episode_worker(seed: int) -> int:
    assert _BENCH_ENV is not None
    env = _BENCH_ENV
    obs, _ = env.reset()
    steps = 0
    ctrl = BlindSpiralController(torque_scale=0.65)
    ctrl.reset(obs)
    for _ in range(120):
        action = ctrl.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    return steps


def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    succ = [r for r in rows if r['success']]
    seen = [r for r in rows if r['split'] == 'seen']
    unseen = [r for r in rows if r['split'] == 'unseen']
    modes = dict(Counter(r['failure_mode'] for r in rows))
    by_shape = {}
    for shape in sorted(set(r['shape'] for r in rows)):
        sub = [r for r in rows if r['shape'] == shape]
        by_shape[shape] = {
            'successes': sum(r['success'] for r in sub),
            'trials': len(sub),
            'success_rate': sum(r['success'] for r in sub) / len(sub),
            'modes': dict(Counter(r['failure_mode'] for r in sub)),
        }
    return {
        'successes': len(succ),
        'trials': n,
        'success_rate': len(succ) / max(n, 1),
        'seen_successes': sum(r['success'] for r in seen),
        'seen_trials': len(seen),
        'unseen_successes': sum(r['success'] for r in unseen),
        'unseen_trials': len(unseen),
        'median_time_to_insert_s': float(np.median([r['time_s'] for r in succ])) if succ else None,
        'mean_time_s_all': float(np.mean([r['time_s'] for r in rows])),
        'median_peak_contact_force_N': float(np.median([r['peak_contact_force_N'] for r in rows])),
        'max_peak_contact_force_N': float(np.max([r['peak_contact_force_N'] for r in rows])),
        'shove_failures': modes.get('socket_shoved', 0),
        'tip_failures': modes.get('socket_tipped', 0),
        'median_socket_disp_mm': float(np.median([r['max_socket_disp_mm'] for r in rows])),
        'max_socket_disp_mm': float(np.max([r['max_socket_disp_mm'] for r in rows])),
        'self_collision_steps': sum(r['self_collision_steps'] for r in rows),
        'joint_limit_steps': sum(r['joint_limit_steps'] for r in rows),
        'failure_modes': modes,
        'by_shape': by_shape,
    }


def benchmark_headless_throughput(worker_counts: tuple[int, ...] = (8, 10), episodes_per_run: int = 160) -> dict:
    ctx = mp.get_context('spawn')
    results = {}
    for nw in worker_counts:
        with ctx.Pool(nw, initializer=_init_throughput_worker, initargs=(True,)) as pool:
            # Warmup one episode per worker
            pool.map(_throughput_episode_worker, list(range(nw)))
            t0 = time.perf_counter()
            step_counts = pool.map(_throughput_episode_worker, list(range(episodes_per_run)))
            dt = time.perf_counter() - t0
        total_steps = int(sum(step_counts))
        results[f'{nw}_workers_bolted'] = {
            'workers': nw,
            'episodes': episodes_per_run,
            'total_policy_steps': total_steps,
            'total_physics_steps_1khz': total_steps * 20,
            'wall_s': round(dt, 3),
            'steps_per_sec': round(total_steps / max(dt, 1e-9), 1),
            'episodes_per_sec': round(episodes_per_run / max(dt, 1e-9), 2),
        }
    return results


def main() -> None:
    t0_all = time.perf_counter()

    print('=== 1. Exact Clearance Gravity Drop-Test Gate (6 shapes x {1.0 mm, 0.5 mm}) ===', flush=True)
    drop_tests = run_all_drop_tests(clearances_m=(0.0010, 0.0005))
    n_drop_pass = sum(r['passed'] for r in drop_tests)
    for r in drop_tests:
        print(
            f"  c={r['clearance_mm']:.1f}mm {r['shape']:10s} ({r['split']:6s}) | "
            f"depth={r['depth_mm']:5.2f}mm drift={r['max_xy_drift_mm']:.2f}mm tilt={r['max_tilt_deg']:.2f}deg | "
            f"{'PASS' if r['passed'] else 'FAIL'}",
            flush=True,
        )
    print(f'Drop-test gate: {n_drop_pass}/{len(drop_tests)} PASS\n', flush=True)

    regime_a = build_regime_a_bolted()
    regime_b = build_regime_b_unbolted()
    tight_suite = build_tight_clearance_suite()

    controllers = [
        'Nominal_StraightDown',
        'BlindSpiral_Stiff_1.0',
        'BlindSpiral_LowTorque_0.35',
        'Privileged_Oracle',
    ]

    ctx = mp.get_context('spawn')
    bench_results = {'regime_a_bolted': {}, 'regime_b_unbolted': {}, 'tight_0.5mm': {}}

    print('=== 2. Reference Controllers on Regime A (30 bolted) & Regime B (30 unbolted) ===', flush=True)
    with ctx.Pool(10) as pool:
        for reg_key, trials in [('regime_a_bolted', regime_a), ('regime_b_unbolted', regime_b)]:
            print(f'\n--- {reg_key} (n={len(trials)}) ---', flush=True)
            for cname in controllers:
                t0 = time.perf_counter()
                rows = pool.map(_eval_worker, [(cname, t) for t in trials])
                dt = time.perf_counter() - t0
                summ = summarize_rows(rows)
                bench_results[reg_key][cname] = {
                    'summary': summ,
                    'wall_s': round(dt, 2),
                    'episodes': rows,
                }
                t_ins = f"{summ['median_time_to_insert_s']:.2f}s" if summ['median_time_to_insert_s'] is not None else '—'
                print(
                    f"  {cname:28s} | succ={summ['successes']:2d}/30 ({100*summ['success_rate']:5.1f}%, "
                    f"seen={summ['seen_successes']:2d}/15 unseen={summ['unseen_successes']:2d}/15) | "
                    f"med_t={t_ins:5s} med_F={summ['median_peak_contact_force_N']:5.1f}N | "
                    f"shove={summ['shove_failures']:2d} tip={summ['tip_failures']:2d} med_disp={summ['median_socket_disp_mm']:5.1f}mm | "
                    f"col={summ['self_collision_steps']} jlim={summ['joint_limit_steps']} ({dt:.1f}s)",
                    flush=True,
                )

        # Tight-clearance 0.5 mm sanity check with Privileged Oracle & BlindSpiral_Stiff
        print('\n--- tight_0.5mm (n=6, all 6 shapes @ 0.5 mm clearance) ---', flush=True)
        for cname in ('Nominal_StraightDown', 'BlindSpiral_Stiff_1.0', 'Privileged_Oracle'):
            rows = pool.map(_eval_worker, [(cname, t) for t in tight_suite])
            summ = summarize_rows(rows)
            bench_results['tight_0.5mm'][cname] = {'summary': summ, 'episodes': rows}
            print(
                f"  {cname:28s} | succ={summ['successes']}/6 ({100*summ['success_rate']:5.1f}%) | "
                f"modes={summ['failure_modes']} col={summ['self_collision_steps']} jlim={summ['joint_limit_steps']}",
                flush=True,
            )

    print('\n=== 3. Headless Throughput Benchmark (8 & 10 CPU workers, 1 kHz / 50 Hz) ===', flush=True)
    throughput = benchmark_headless_throughput(worker_counts=(8, 10), episodes_per_run=160)
    for k, v in throughput.items():
        print(
            f"  {k:20s}: {v['episodes']} eps ({v['total_policy_steps']} policy steps) in {v['wall_s']:.2f}s "
            f"-> {v['steps_per_sec']:.1f} steps/s | {v['episodes_per_sec']:.2f} ep/s",
            flush=True,
        )

    total_col = sum(
        v['summary']['self_collision_steps']
        for reg in bench_results.values()
        for v in reg.values()
    )
    total_jlim = sum(
        v['summary']['joint_limit_steps']
        for reg in bench_results.values()
        for v in reg.values()
    )

    report = {
        'total_wall_s': round(time.perf_counter() - t0_all, 2),
        'drop_tests': {
            'passed': n_drop_pass,
            'total': len(drop_tests),
            'all_passed': n_drop_pass == len(drop_tests),
            'cases': drop_tests,
        },
        'benchmarks': bench_results,
        'throughput': throughput,
        'total_self_collision_steps': total_col,
        'total_joint_limit_steps': total_jlim,
    }
    out_path = THIS_DIR / 'phase0_results.json'
    out_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'\n=== PHASE 0 VERIFICATION FINISHED ({report["total_wall_s"]:.1f}s) | self_col={total_col}, jlim={total_jlim} ===', flush=True)
    print(f'Saved results to {out_path}', flush=True)


if __name__ == '__main__':
    main()
