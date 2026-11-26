"""Phase 2 Targeted Experiment for `projects/blind_drawer/`: Per-Joint 5-DoF GravComp-TL (`D3_Res_GravComp_5DoF`).

Tests whether per-joint 5-DoF `alpha in [0.10, 1.0]^5` ([pan, lift, elbow, wflex, wroll]) fixes the
Phase 1 Split OOD downward-pinning bottleneck while preserving Split A off-axis compliance.
Reuses already-trained Phase 0 baselines and Phase 1 `D1_Res_Stiff` / `D2_Res_GravComp_TL` results
directly (ZERO retraining of D1 or D2).
"""
from __future__ import annotations

import gc
import json
import math
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_drawer.benchmarks import (
    TRAIN_SEED_OFFSET,
    build_split_a_bolted_sticky,
    build_split_b_unbolted_cabinet,
    build_split_ood_light_height_angle,
)
from projects.blind_drawer.controllers import NominalStraightBackController
from projects.blind_drawer.env import DrawerTaskSpec
from projects.blind_drawer.phase0_gate import (
    _get_reusable_env,
    _init_worker,
    summarize_suite,
)
from projects.blind_drawer.phase1_policy import (
    CRITIC_PRIV_DIM,
    DRAWER_CONTACT_TO_IDX,
    GRU_HIDDEN_DIM,
    JOINT_ALPHA_NAMES,
    DrawerObsBuilder,
    DrawerResidualGRUPPO,
)
from projects.blind_drawer.phase1_train import (
    _pool_rss_mb,
    train_one_seed,
)

PHASES = ('free_slide', 'off_axis_bind', 'detent_catch')
PHASE_SHORT = {'free_slide': 'free', 'off_axis_bind': 'bind', 'detent_catch': 'detent'}


def _eval_rl_5dof_worker(job: dict) -> dict:
    task: DrawerTaskSpec = job['task']
    mode: str = job['mode']

    env = _get_reusable_env(task)
    obs, info = env.reset()
    base_ctrl = NominalStraightBackController(mode='stiff', pull_speed_m_s=0.028, down_preload_m=0.0012)
    base_ctrl.reset(obs)

    builder = DrawerObsBuilder(mode=mode)
    builder.reset()
    pol = DrawerResidualGRUPPO(obs_dim=builder.obs_dim, mode=mode, critic_priv_dim=CRITIC_PRIV_DIM, hidden_dim=GRU_HIDDEN_DIM)
    pol.set_actor_weights(job['actor_weights'])
    h = pol.init_hidden()

    terminated = False
    truncated = False
    aux_correct = 0
    aux_total = 0
    yaw_err_deg_list: list[float] = []
    alpha_by: dict[str, list[tuple[float, ...]]] = {ph: [] for ph in PHASES}

    while not (terminated or truncated):
        base_a = base_ctrl.act(obs)[:3]
        x_vec = builder.build(obs, base_a)
        cc = info['privileged']['contact_class']
        true_cls = DRAWER_CONTACT_TO_IDX[cc]
        true_yaw_rad = float(info['privileged']['rail_yaw_rad'])

        env_act, _, alpha_5, _, h, pred_cls, pred_yaw_rad = pol.act_step(x_vec, h, base_a, deterministic=True)
        builder.set_prev_alpha(alpha_5)

        aux_correct += int(pred_cls == true_cls)
        aux_total += 1
        yaw_err_deg_list.append(abs(math.degrees(pred_yaw_rad - true_yaw_rad)))
        if alpha_5 is not None and cc in alpha_by:
            alpha_by[cc].append(alpha_5)

        obs, _, terminated, truncated, info = env.step(env_act)

    priv = info['privileged']
    row = {
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
        'aux_contact_acc': aux_correct / max(aux_total, 1),
        'aux_rail_yaw_mae_deg': float(np.mean(yaw_err_deg_list)) if yaw_err_deg_list else 0.0,
    }
    for ph in PHASES:
        sh = PHASE_SHORT[ph]
        arr = np.asarray(alpha_by[ph], dtype=float) if alpha_by[ph] else None
        for j, jname in enumerate(JOINT_ALPHA_NAMES):
            row[f'alpha_{jname}_{sh}'] = float(np.mean(arr[:, j])) if arr is not None else None
    return row


def summarize_rl_5dof_suite(rows: list[dict]) -> dict:
    base = summarize_suite(rows)
    base['mean_aux_contact_acc'] = round(float(np.mean([r['aux_contact_acc'] for r in rows])), 4)
    base['mean_aux_rail_yaw_mae_deg'] = round(float(np.mean([r['aux_rail_yaw_mae_deg'] for r in rows])), 2)
    for ph in PHASES:
        sh = PHASE_SHORT[ph]
        for jname in JOINT_ALPHA_NAMES:
            k = f'alpha_{jname}_{sh}'
            vals = [r[k] for r in rows if r.get(k) is not None]
            base[f'mean_{k}'] = round(float(np.mean(vals)), 3) if vals else None
    return base


def aggregate_seeds_5dof(runs: list[dict], suite_key: str) -> dict:
    summs = [r['eval'][suite_key]['summary'] for r in runs]
    rates = [s['success_rate'] for s in summs]
    times = [s['median_time_s'] for s in summs if s['median_time_s'] is not None]
    out = {
        'trials': summs[0]['trials'],
        'per_seed_successes': [s['successes'] for s in summs],
        'mean_success_rate': float(np.mean(rates)),
        'std_success_rate': float(np.std(rates)),
        'mean_median_travel_mm': float(np.mean([s['median_travel_mm'] for s in summs])),
        'mean_median_time_s': float(np.mean(times)) if times else None,
        'mean_median_force_N': float(np.mean([s['median_peak_force_N'] for s in summs])),
        'mean_median_disp_mm': float(np.mean([s['median_cab_disp_mm'] for s in summs])),
        'mean_median_tilt_deg': float(np.mean([s['median_cab_tilt_deg'] for s in summs])),
        'mean_shove_failures': float(np.mean([s['shove_failures'] for s in summs])),
        'mean_tip_failures': float(np.mean([s['tip_failures'] for s in summs])),
        'mean_slip_failures': float(np.mean([s['slip_failures'] for s in summs])),
        'mean_aux_contact_acc': float(np.mean([s['mean_aux_contact_acc'] for s in summs])),
        'mean_aux_rail_yaw_mae_deg': float(np.mean([s['mean_aux_rail_yaw_mae_deg'] for s in summs])),
        'total_self_collision_steps': int(sum(s['self_collision_steps'] for s in summs)),
        'total_joint_limit_steps': int(sum(s['joint_limit_steps'] for s in summs)),
    }
    for ph in PHASES:
        sh = PHASE_SHORT[ph]
        for jname in JOINT_ALPHA_NAMES:
            k = f'mean_alpha_{jname}_{sh}'
            vals = [s[k] for s in summs if s.get(k) is not None]
            out[k] = round(float(np.mean(vals)), 3) if vals else None
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--target-steps', type=int, default=1_500_000)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=250)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--out-prefix', type=str, default='phase2')
    parser.add_argument('--seed-offset', type=int, default=100000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--single-run', action='store_true')
    parser.add_argument('--log-every-steps', type=int, default=None)
    args = parser.parse_args()

    t_total = time.perf_counter()
    ckpt_dir = THIS_DIR / f'{args.out_prefix}_checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    eval_suites = {
        'split_a_bolted_sticky': build_split_a_bolted_sticky(),
        'split_b_unbolted_cabinet': build_split_b_unbolted_cabinet(),
        'split_ood_light_height_angle': build_split_ood_light_height_angle(),
    }
    label = 'D2_PerJoint'
    mode = 'res_gravcomp_tl_5dof'

    if not args.single_run and len(args.seeds) > 1:
        print(
            f'Launching Phase 2 Blind Drawer Isolated Subprocess Orchestrator ({args.workers} workers/run, '
            f'{label} x {len(args.seeds)} seeds x ~{args.target_steps:,} steps, resume={args.resume})...',
            flush=True,
        )
        runs: list[dict] = []
        total_eps = 0
        for seed in args.seeds:
            ckpt_p = ckpt_dir / f'{label}_seed{seed}.npz'
            res_p = ckpt_dir / f'{label}_seed{seed}_result.json'
            if args.resume and ckpt_p.exists() and res_p.exists():
                rec = json.loads(res_p.read_text())
                runs.append(rec)
                total_eps += int(rec['meta']['episodes_budget'])
                print(f'  SKIP [RESUME] {label} s={seed}', flush=True)
                continue
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                '--single-run', '--workers', str(args.workers),
                '--target-steps', str(args.target_steps),
                '--batch-size', str(args.batch_size),
                '--max-steps', str(args.max_steps),
                '--seeds', str(seed),
                '--out-prefix', args.out_prefix,
                '--seed-offset', str(args.seed_offset),
            ]
            if args.log_every_steps is not None:
                cmd += ['--log-every-steps', str(args.log_every_steps)]
            print(f'\n--- Spawning isolated subprocess for {label} s={seed} ---', flush=True)
            subprocess.run(cmd, check=True)
            rec = json.loads(res_p.read_text())
            runs.append(rec)
            total_eps += int(rec['meta']['episodes_budget'])
    else:
        runs = []
        total_eps = 0
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
            for seed in args.seeds:
                seed_slot = {101: 0, 202: 1, 303: 2}.get(int(seed), int(seed) % 10)
                seed_cursor = TRAIN_SEED_OFFSET + int(args.seed_offset) + (20 + seed_slot) * 100_000
                pol, meta, n_eps = train_one_seed(
                    pool, label, mode, seed=seed, seed_cursor=seed_cursor,
                    target_steps=args.target_steps, batch_size=args.batch_size,
                    max_steps=args.max_steps, log_every_steps=args.log_every_steps,
                    actor_lr_init=2.5e-4, target_kl=0.035,
                )
                total_eps += n_eps
                ckpt_p = ckpt_dir / f'{label}_seed{seed}.npz'
                res_p = ckpt_dir / f'{label}_seed{seed}_result.json'
                pol.save(ckpt_p)
                meta['checkpoint'] = str(ckpt_p.relative_to(THIS_DIR))

                aw = pol.get_actor_weights()
                eval_out = {}
                parts = []
                for sk, trials in eval_suites.items():
                    rows = pool.map(_eval_rl_5dof_worker, [{'task': t, 'mode': mode, 'actor_weights': aw} for t in trials])
                    summ = summarize_rl_5dof_suite(rows)
                    eval_out[sk] = {'summary': summ, 'episodes': rows}
                    parts.append(
                        f'{sk[:9]}={summ["successes"]:2d}/{summ["trials"]}({100*summ["success_rate"]:5.1f}%,'
                        f'F={summ["median_peak_force_N"]:4.1f}N,shv={summ["shove_failures"]:2d},aux={100*summ["mean_aux_contact_acc"]:4.1f}%)'
                    )
                _, _, tot_mb = _pool_rss_mb(pool)
                print(f'  EVAL [{label:20s} s={seed}] ({meta["wall_s"]:.0f}s, RAM={tot_mb/1024:.2f}GB) | ' + ' | '.join(parts), flush=True)
                rec = {'meta': meta, 'eval': eval_out}
                res_p.write_text(json.dumps(rec, indent=2) + '\n')
                runs.append(rec)
                gc.collect()
        if args.single_run:
            return

    # Load frozen Phase 1 results (D1_Res_Stiff, D2_Res_GravComp_TL, Phase 0 baselines) for direct comparison
    p1_path = THIS_DIR / 'phase1_results.json'
    p1_data = json.loads(p1_path.read_text()) if p1_path.exists() else {}

    d3_agg = {
        'mode': mode,
        'obs_dim': runs[0]['meta']['obs_dim'],
        'act_dim': runs[0]['meta']['act_dim'],
        'hidden_dim': runs[0]['meta']['hidden_dim'],
        'actor_param_count': runs[0]['meta']['actor_param_count'],
        'critic_param_count': runs[0]['meta']['critic_param_count'],
        'mean_total_steps': float(np.mean([r['meta']['total_steps'] for r in runs])),
        'mean_train_wall_s': float(np.mean([r['meta']['wall_s'] for r in runs])),
        'mean_steps_per_sec': float(np.mean([r['meta']['steps_per_sec'] for r in runs])),
        'max_peak_total_rss_gb': float(max(r['meta']['peak_total_rss_gb'] for r in runs)),
        'eval': {sk: aggregate_seeds_5dof(runs, sk) for sk in eval_suites},
    }

    rl_aggregate = dict(p1_data.get('rl_aggregate', {}))
    rl_aggregate[label] = d3_agg

    total_col = sum(d3_agg['eval'][sk]['total_self_collision_steps'] for sk in eval_suites)
    total_jlim = sum(d3_agg['eval'][sk]['total_joint_limit_steps'] for sk in eval_suites)

    report = {
        'total_wall_s': round(time.perf_counter() - t_total, 2),
        'n_train_episodes_total': total_eps,
        'total_eval_self_collision_steps': total_col,
        'total_eval_joint_limit_steps': total_jlim,
        'reused_from_phase1': list(p1_data.get('rl_aggregate', {}).keys()),
        'baselines': p1_data.get('baselines', {}),
        'rl_aggregate': rl_aggregate,
        'rl_runs': {label: runs},
    }
    out_path = THIS_DIR / f'{args.out_prefix}_results.json'
    out_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'\n=== {args.out_prefix.upper()} BLIND DRAWER FINISHED ({report["total_wall_s"]:.1f}s) | self_col={total_col}, jlim={total_jlim} ===', flush=True)
    print(f'Saved results to {out_path}', flush=True)


if __name__ == '__main__':
    main()
