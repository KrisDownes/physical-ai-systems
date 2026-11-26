"""Phase 1 Training & 3-Split Evaluation Harness for `projects/blind_drawer/` (`D1` vs `D2`).

Memory-safe, isolated-subprocess architecture (strictly flat < 1.0 GB total RSS on 8 CPU workers):
  - Reuses at most 2 lean MjModels per worker (`bolted=True` and `bolted=False`), updating friction,
    damping, cabinet mass, table friction, and yaw in-place!
  - Isolated short-lived subprocess per `(method, seed)` with incremental `--resume` saving.
  - Strictly in-distribution training (`height_shift_m = 0.0`, `|yaw| <= 18 deg`, `mass 400-750 g`,
    `mu_table 0.40-0.65`, `seed >= 100,000`) so `Split OOD` (`280-350 g, mu=0.35, +-15 mm, +-22 deg`)
    is 100% strictly out-of-distribution!

Trains 2 matched methods x 3 seeds `[101, 202, 303]` x `800k` policy steps/run (= `4.8M` steps):
  1. `D1_Res_PosOnly` (`res_pos_only`):       Stiff `ts=1.0` + 3-DoF relative-kinematics residual GRU(64)
  2. `D2_Res_GravComp_TL` (`res_gravcomp_tl`): Dynamic 2-group `GravComp-TL` `[alpha_lat, alpha_arm]` + GRU(64)
"""
from __future__ import annotations

from collections import deque
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
    _eval_episode_worker as _eval_p0_baseline_worker,
    _get_reusable_env,
    _init_worker,
    get_rss_mb,
    summarize_suite,
)
from projects.blind_drawer.phase1_policy import (
    BPTT_CHUNK_LEN,
    CRITIC_PRIV_DIM,
    DRAWER_CONTACT_TO_IDX,
    GRU_HIDDEN_DIM,
    DrawerObsBuilder,
    DrawerResidualGRUPPO,
    build_privileged_critic_features,
)


def compute_drawer_step_reward(
    prev_priv: dict,
    cur_priv: dict,
    d_xyz: np.ndarray,
    prev_dxyz: np.ndarray,
    info: dict,
) -> float:
    """Identical clean reward for D1 and D2 (progress + success - disp/tilt/force/smoothness penalties)."""
    d_trv_m = (float(cur_priv['drawer_travel_mm']) - float(prev_priv['drawer_travel_mm'])) * 0.001
    r = 50.0 * float(np.clip(d_trv_m, -0.006, 0.008))

    # Mild peak-force regularization above 35 N
    f_pk = float(cur_priv['peak_contact_force_N'])
    if f_pk > 35.0:
        r -= 0.012 * min((f_pk - 35.0) / 20.0, 2.0)

    # Soft penalties on unbolted cabinet displacement (> 8 mm) and tilt (> 4 deg)
    if not cur_priv['bolted']:
        disp = float(cur_priv['cabinet_disp_mm'])
        if disp > 8.0:
            r -= 0.05 * min((disp - 8.0) / 17.0, 1.5)
        tilt = float(cur_priv['cabinet_tilt_deg'])
        if tilt > 4.0:
            r -= 0.05 * min((tilt - 4.0) / 6.0, 1.5)

    # Smooth action penalty & mild time cost
    r -= 0.002 * float(np.linalg.norm(d_xyz - prev_dxyz) * 100.0)
    r -= 0.004

    if info['success']:
        r += 10.0
    elif info['failure_mode'] in ('cabinet_shoved', 'cabinet_tipped', 'handle_slip'):
        r -= 4.0
    return float(r)


def sample_drawer_train_batch(
    rng: np.random.Generator,
    batch_size: int,
    seed_cursor: int,
    yaw_max_deg: float = 18.0,
    max_steps: int = 250,
) -> tuple[list[DrawerTaskSpec], int]:
    """Strictly in-distribution training batch (height_shift=0.0, |yaw|<=18 deg, 50% bolted / 50% unbolted)."""
    if seed_cursor < TRAIN_SEED_OFFSET:
        raise ValueError(f'seed_cursor {seed_cursor} must be >= {TRAIN_SEED_OFFSET}')
    trials: list[DrawerTaskSpec] = []
    for i in range(batch_size):
        bolted = (i % 2 == 0)
        ymax = 8.0 if rng.random() < 0.20 else float(yaw_max_deg)
        yaw = float(rng.uniform(-ymax, ymax))
        if bolted:
            f_rail = float(rng.uniform(2.5, 5.5))
            f_det = float(rng.uniform(5.0, 9.0))
            det_w = 6.0
            rail_damp = float(rng.uniform(10.0, 18.0))
            mu_tbl = 0.55
            m_cab = 0.600
        else:
            f_rail = float(rng.uniform(0.8, 1.8))
            f_det = float(rng.uniform(3.0, 7.0))
            det_w = 2.5
            rail_damp = float(rng.uniform(8.0, 12.0))
            mu_tbl = float(rng.uniform(0.40, 0.65))
            m_cab = float(rng.uniform(0.400, 0.750))
        trials.append(DrawerTaskSpec(
            trial_id=f'drw-tr-{seed_cursor}',
            seed=seed_cursor,
            bolted=bolted,
            cabinet_xy=(float(0.336 + rng.uniform(-0.008, 0.008)), float(rng.uniform(-0.012, 0.012))),
            height_shift_m=0.0,  # NEVER shift height during training -> Split OOD +-15 mm is 100% OOD!
            rail_yaw_deg=yaw,
            f_rail_N=f_rail,
            rail_damping=rail_damp,
            rail_bind_mu=1.80,
            detent_travel_mm=float(rng.uniform(10.0, 40.0)),
            detent_force_N=f_det,
            detent_width_mm=det_w,
            mu_jaw_handle=0.85,
            mu_table=mu_tbl,
            cabinet_mass_kg=m_cab,
            backlash_rad=0.0045,
            max_steps=max_steps,
        ))
        seed_cursor += 1
    return trials, seed_cursor


def _train_rollout_worker(job: dict) -> dict:
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

    rng = np.random.default_rng(job['rollout_seed'])
    ar_rho = float(job.get('ar_rho', 0.35))
    innov = math.sqrt(max(1.0 - ar_rho * ar_rho, 0.0))
    eps = rng.normal(0.0, 1.0, size=pol.act_dim)

    obs_buf, priv_buf, z_buf, rew_buf, cls_buf, yaw_buf = [], [], [], [], [], []
    prev_priv = info['privileged']
    prev_dxyz = np.zeros(3, dtype=float)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        base_a = base_ctrl.act(obs)[:3]
        x_vec = builder.build(obs, base_a)
        p_vec = build_privileged_critic_features(obs, info['privileged'])
        cls_idx = DRAWER_CONTACT_TO_IDX[info['privileged']['contact_class']]
        yaw_rad = float(info['privileged']['rail_yaw_rad'])

        eps = ar_rho * eps + innov * rng.normal(0.0, 1.0, size=pol.act_dim)
        env_act, d_xyz, alpha_pair, z, h, _, _ = pol.act_step(x_vec, h, base_a, deterministic=False, eps=eps)
        builder.set_prev_alpha(alpha_pair)

        obs, _, terminated, truncated, info = env.step(env_act)
        r = compute_drawer_step_reward(prev_priv, info['privileged'], d_xyz, prev_dxyz, info)

        obs_buf.append(x_vec)
        priv_buf.append(p_vec)
        z_buf.append(z)
        rew_buf.append(r)
        cls_buf.append(cls_idx)
        yaw_buf.append([yaw_rad])
        prev_priv = info['privileged']
        prev_dxyz = d_xyz

    last_base = base_ctrl.act(obs)[:3]
    return {
        'obs': np.asarray(obs_buf, dtype=float),
        'priv': np.asarray(priv_buf, dtype=float),
        'last_obs': builder.build(obs, last_base),
        'last_priv': build_privileged_critic_features(obs, info['privileged']),
        'z': np.asarray(z_buf, dtype=float),
        'rewards': np.asarray(rew_buf, dtype=float),
        'contact_labels': np.asarray(cls_buf, dtype=np.int64),
        'rail_yaws': np.asarray(yaw_buf, dtype=float),
        'terminated': bool(terminated),
        'success': bool(info['success']),
        'bolted': bool(task.bolted),
        'failure_mode': info['failure_mode'],
        'steps': int(env.step_count),
    }


def _eval_rl_worker(job: dict) -> dict:
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
    a_lat_by: dict[str, list[float]] = {'free_slide': [], 'off_axis_bind': [], 'detent_catch': []}
    a_arm_by: dict[str, list[float]] = {'free_slide': [], 'off_axis_bind': [], 'detent_catch': []}

    while not (terminated or truncated):
        base_a = base_ctrl.act(obs)[:3]
        x_vec = builder.build(obs, base_a)
        cc = info['privileged']['contact_class']
        true_cls = DRAWER_CONTACT_TO_IDX[cc]
        true_yaw_rad = float(info['privileged']['rail_yaw_rad'])

        env_act, _, alpha_pair, _, h, pred_cls, pred_yaw_rad = pol.act_step(x_vec, h, base_a, deterministic=True)
        builder.set_prev_alpha(alpha_pair)

        aux_correct += int(pred_cls == true_cls)
        aux_total += 1
        yaw_err_deg_list.append(abs(math.degrees(pred_yaw_rad - true_yaw_rad)))
        if alpha_pair is not None and cc in a_lat_by:
            a_lat_by[cc].append(alpha_pair[0])
            a_arm_by[cc].append(alpha_pair[1])

        obs, _, terminated, truncated, info = env.step(env_act)

    priv = info['privileged']
    return {
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
        'alpha_lat_free': float(np.mean(a_lat_by['free_slide'])) if a_lat_by['free_slide'] else None,
        'alpha_arm_free': float(np.mean(a_arm_by['free_slide'])) if a_arm_by['free_slide'] else None,
        'alpha_lat_bind': float(np.mean(a_lat_by['off_axis_bind'])) if a_lat_by['off_axis_bind'] else None,
        'alpha_arm_bind': float(np.mean(a_arm_by['off_axis_bind'])) if a_arm_by['off_axis_bind'] else None,
        'alpha_lat_detent': float(np.mean(a_lat_by['detent_catch'])) if a_lat_by['detent_catch'] else None,
        'alpha_arm_detent': float(np.mean(a_arm_by['detent_catch'])) if a_arm_by['detent_catch'] else None,
    }


def summarize_rl_suite(rows: list[dict]) -> dict:
    base = summarize_suite(rows)
    base['mean_aux_contact_acc'] = round(float(np.mean([r['aux_contact_acc'] for r in rows])), 4)
    base['mean_aux_rail_yaw_mae_deg'] = round(float(np.mean([r['aux_rail_yaw_mae_deg'] for r in rows])), 2)
    for k in ('alpha_lat_free', 'alpha_arm_free', 'alpha_lat_bind', 'alpha_arm_bind', 'alpha_lat_detent', 'alpha_arm_detent'):
        vals = [r[k] for r in rows if r.get(k) is not None]
        base[f'mean_{k}'] = round(float(np.mean(vals)), 3) if vals else None
    return base


def aggregate_seeds_drawer(runs: list[dict], suite_key: str) -> dict:
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
    for k in ('mean_alpha_lat_free', 'mean_alpha_arm_free', 'mean_alpha_lat_bind', 'mean_alpha_arm_bind', 'mean_alpha_lat_detent', 'mean_alpha_arm_detent'):
        vals = [s[k] for s in summs if s.get(k) is not None]
        out[k] = round(float(np.mean(vals)), 3) if vals else None
    return out


def _pool_rss_mb(pool) -> tuple[float, float, float]:
    p_mb = get_rss_mb(None)
    w_mb = sum(get_rss_mb(w.pid) for w in pool._pool if w.pid is not None)
    return p_mb, w_mb, p_mb + w_mb


def train_one_seed(
    pool,
    label: str,
    mode: str,
    seed: int,
    seed_cursor: int,
    target_steps: int = 800_000,
    batch_size: int = 20,
    max_steps: int = 250,
    log_every_steps: int | None = None,
    actor_lr_init: float = 3.5e-4,
    target_kl: float = 0.025,
) -> tuple[DrawerResidualGRUPPO, dict, int]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    builder = DrawerObsBuilder(mode=mode)
    pol = DrawerResidualGRUPPO(
        obs_dim=builder.obs_dim, mode=mode, critic_priv_dim=CRITIC_PRIV_DIM,
        hidden_dim=GRU_HIDDEN_DIM, seed=seed, init_log_std=-0.95,
    )
    history = []
    episodes_cum = 0
    cum_steps = 0
    upd = 0
    next_log_steps = 0 if log_every_steps is None else int(log_every_steps)
    rolling_succ: deque[int] = deque(maxlen=100)
    peak_rss_mb = 0.0

    while cum_steps < target_steps:
        progress = min(1.0, cum_steps / max(target_steps, 1))
        yaw_max = 6.0 + 12.0 * min(1.0, progress / 0.25)  # smooth 6 -> 18 deg over first 25%
        aw = pol.get_actor_weights()
        ar_rho = 0.40 * max(0.0, 1.0 - progress / 0.50)
        log_std_cap = -0.90 + (-1.55 - (-0.90)) * min(1.0, progress / 0.90)

        trajs: list[dict] = []
        batch_steps = 0
        while batch_steps < 1800:
            wave_trials, seed_cursor = sample_drawer_train_batch(
                rng, batch_size=batch_size, seed_cursor=seed_cursor, yaw_max_deg=yaw_max, max_steps=max_steps,
            )
            episodes_cum += len(wave_trials)
            rseeds = rng.integers(1, 2**31 - 1, size=batch_size)
            jobs = [
                {'task': t, 'mode': mode, 'actor_weights': aw, 'rollout_seed': int(rs), 'ar_rho': ar_rho}
                for t, rs in zip(wave_trials, rseeds)
            ]
            wave_trajs = pool.map(_train_rollout_worker, jobs)
            trajs.extend(wave_trajs)
            batch_steps += sum(tr['steps'] for tr in wave_trajs)

        if upd == 0:
            pol.obs_rms.update(np.concatenate([tr['obs'] for tr in trajs], axis=0))
            pol.critic_rms.update(np.concatenate([np.concatenate([tr['obs'], tr['priv']], axis=-1) for tr in trajs], axis=0))

        lr_scale = 0.20 + 0.80 * (1.0 - progress)
        stats = pol.ppo_update(
            trajs, rng=rng,
            actor_lr=actor_lr_init * lr_scale, critic_lr=1.2e-3 * lr_scale,
            log_std_max=log_std_cap,
            target_kl=target_kl,
        )
        upd += 1
        cum_steps += stats['n_steps']
        for tr in trajs:
            rolling_succ.append(int(tr['success']))
        roll_100 = float(np.mean(rolling_succ)) if rolling_succ else 0.0
        succ_all = sum(tr['success'] for tr in trajs) / len(trajs)
        b_tr = [tr for tr in trajs if tr['bolted']]
        u_tr = [tr for tr in trajs if not tr['bolted']]
        succ_b = sum(tr['success'] for tr in b_tr) / max(len(b_tr), 1)
        succ_u = sum(tr['success'] for tr in u_tr) / max(len(u_tr), 1)
        shoves = sum(tr['failure_mode'] == 'cabinet_shoved' for tr in u_tr)
        del trajs

        history.append({
            'update': upd, 'steps_cum': cum_steps, 'episodes_cum': episodes_cum,
            'yaw_max_deg': round(yaw_max, 2), 'roll100_succ': round(roll_100, 3),
            'train_succ_bolted': succ_b, 'train_succ_unbolted': succ_u, 'train_shoves': shoves,
            **stats,
        })

        if cum_steps >= next_log_steps or cum_steps >= target_steps:
            elapsed = time.perf_counter() - t0
            p_mb, w_mb, tot_mb = _pool_rss_mb(pool)
            peak_rss_mb = max(peak_rss_mb, tot_mb)
            print(
                f'  [{label:19s} s={seed}] upd {upd:3d} ({cum_steps:7d}/{target_steps} st, {cum_steps/max(elapsed,1e-6):5.0f} st/s, '
                f'RAM={tot_mb/1024:.2f}GB[p={p_mb:.0f}M,w={w_mb/1024:.2f}G]) | yaw<={yaw_max:4.1f}d '
                f'succ={100*succ_all:5.1f}% (r100={100*roll_100:4.1f}%, B={100*succ_b:4.0f}% U={100*succ_u:4.0f}% shv={shoves:2d}) | '
                f'ret={stats["mean_return"]:+.2f} ev={stats["explained_var"]:+.2f} aux={100*stats["aux_cls_acc"]:4.1f}% kl={stats["approx_kl"]:.4f}',
                flush=True,
            )
            if log_every_steps is not None:
                while next_log_steps <= cum_steps:
                    next_log_steps += int(log_every_steps)
            else:
                next_log_steps = cum_steps + max(target_steps // 6, 1)

    wall_s = time.perf_counter() - t0
    meta = {
        'label': label, 'mode': mode, 'seed': seed,
        'obs_dim': builder.obs_dim, 'act_dim': pol.act_dim, 'hidden_dim': pol.hidden_dim,
        'actor_param_count': pol.actor_param_count, 'critic_param_count': pol.critic_param_count,
        'n_updates': upd, 'episodes_budget': episodes_cum, 'total_steps': cum_steps,
        'target_steps': target_steps, 'peak_total_rss_gb': round(peak_rss_mb / 1024.0, 3),
        'wall_s': round(wall_s, 2), 'steps_per_sec': round(cum_steps / max(wall_s, 1e-6), 1),
        'history': history,
    }
    return pol, meta, episodes_cum


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--target-steps', type=int, default=1_500_000)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=250)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--methods', type=str, nargs='+', default=['D1_Res_Stiff', 'D2_Res_GravComp_TL'])
    parser.add_argument('--out-prefix', type=str, default='phase1')
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
    all_specs = [
        ('D1_Res_Stiff', 'res_pos_only'),
        ('D2_Res_GravComp_TL', 'res_gravcomp_tl'),
    ]
    method_idx_map = {lbl: i for i, (lbl, _) in enumerate(all_specs)}
    rl_specs = [(lbl, m) for lbl, m in all_specs if lbl in set(args.methods)]
    baseline_names = ('Nominal_Stiff_1.0', 'Nominal_RawLowTorque_0.35', 'Nominal_GravComp_0.35', 'Privileged_Oracle')
    baselines_path = ckpt_dir / 'baselines_eval.json'

    if not args.single_run and (len(rl_specs) * len(args.seeds) > 1):
        print(
            f'Launching Phase 1 Blind Drawer Isolated Subprocess Orchestrator ({args.workers} workers/run, '
            f'{len(rl_specs)} methods x {len(args.seeds)} seeds x ~{args.target_steps:,} steps, resume={args.resume})...',
            flush=True,
        )
        if args.resume and baselines_path.exists():
            baselines = json.loads(baselines_path.read_text())
        else:
            p0_path = THIS_DIR / 'phase0_results.json'
            if p0_path.exists():
                baselines = json.loads(p0_path.read_text())['controllers']
                baselines_path.write_text(json.dumps(baselines, indent=2) + '\n')
                print(f'Loaded Phase 0 reference baselines from {p0_path.name}', flush=True)
            else:
                ctx = mp.get_context('spawn')
                baselines = {}
                with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
                    for cname in baseline_names:
                        baselines[cname] = {}
                        for sk, trials in eval_suites.items():
                            rows = pool.map(_eval_p0_baseline_worker, [(cname, t, False) for t in trials])
                            baselines[cname][sk] = {'summary': summarize_suite(rows), 'episodes': rows}
                baselines_path.write_text(json.dumps(baselines, indent=2) + '\n')

        rl_runs: dict[str, list[dict]] = {lbl: [] for lbl, _ in rl_specs}
        total_eps = 0
        for label, _ in rl_specs:
            for seed in args.seeds:
                ckpt_p = ckpt_dir / f'{label}_seed{seed}.npz'
                res_p = ckpt_dir / f'{label}_seed{seed}_result.json'
                if args.resume and ckpt_p.exists() and res_p.exists():
                    rec = json.loads(res_p.read_text())
                    rl_runs[label].append(rec)
                    total_eps += int(rec['meta']['episodes_budget'])
                    print(f'  SKIP [RESUME] {label} s={seed}', flush=True)
                    continue
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    '--single-run', '--workers', str(args.workers),
                    '--target-steps', str(args.target_steps),
                    '--batch-size', str(args.batch_size),
                    '--max-steps', str(args.max_steps),
                    '--methods', label, '--seeds', str(seed),
                    '--out-prefix', args.out_prefix,
                    '--seed-offset', str(args.seed_offset),
                ]
                if args.log_every_steps is not None:
                    cmd += ['--log-every-steps', str(args.log_every_steps)]
                print(f'\n--- Spawning isolated subprocess for {label} s={seed} ---', flush=True)
                subprocess.run(cmd, check=True)
                rec = json.loads(res_p.read_text())
                rl_runs[label].append(rec)
                total_eps += int(rec['meta']['episodes_budget'])
    else:
        baselines = {}
        rl_runs = {lbl: [] for lbl, _ in rl_specs}
        total_eps = 0
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
            for label, mode in rl_specs:
                m_idx = method_idx_map[label]
                for seed in args.seeds:
                    seed_slot = {101: 0, 202: 1, 303: 2}.get(int(seed), int(seed) % 10)
                    seed_cursor = TRAIN_SEED_OFFSET + int(args.seed_offset) + (m_idx * 10 + seed_slot) * 100_000
                    pol, meta, n_eps = train_one_seed(
                        pool, label, mode, seed=seed, seed_cursor=seed_cursor,
                        target_steps=args.target_steps, batch_size=args.batch_size,
                        max_steps=args.max_steps, log_every_steps=args.log_every_steps,
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
                        rows = pool.map(_eval_rl_worker, [{'task': t, 'mode': mode, 'actor_weights': aw} for t in trials])
                        summ = summarize_rl_suite(rows)
                        eval_out[sk] = {'summary': summ, 'episodes': rows}
                        parts.append(
                            f'{sk[:9]}={summ["successes"]:2d}/{summ["trials"]}({100*summ["success_rate"]:5.1f}%,'
                            f'F={summ["median_peak_force_N"]:4.1f}N,shv={summ["shove_failures"]:2d},aux={100*summ["mean_aux_contact_acc"]:4.1f}%)'
                        )
                    _, _, tot_mb = _pool_rss_mb(pool)
                    print(f'  EVAL [{label:19s} s={seed}] ({meta["wall_s"]:.0f}s, RAM={tot_mb/1024:.2f}GB) | ' + ' | '.join(parts), flush=True)
                    rec = {'meta': meta, 'eval': eval_out}
                    res_p.write_text(json.dumps(rec, indent=2) + '\n')
                    rl_runs[label].append(rec)
                    gc.collect()
        if args.single_run:
            return

    rl_aggregate = {}
    for label, mode in rl_specs:
        runs = rl_runs[label]
        rl_aggregate[label] = {
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
            'eval': {sk: aggregate_seeds_drawer(runs, sk) for sk in eval_suites},
        }

    total_col = sum(v['eval'][sk]['total_self_collision_steps'] for v in rl_aggregate.values() for sk in eval_suites)
    total_jlim = sum(v['eval'][sk]['total_joint_limit_steps'] for v in rl_aggregate.values() for sk in eval_suites)

    report = {
        'total_wall_s': round(time.perf_counter() - t_total, 2),
        'n_train_episodes_total': total_eps,
        'total_eval_self_collision_steps': total_col,
        'total_eval_joint_limit_steps': total_jlim,
        'baselines': baselines,
        'rl_aggregate': rl_aggregate,
        'rl_runs': rl_runs,
    }
    out_path = THIS_DIR / f'{args.out_prefix}_results.json'
    out_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'\n=== {args.out_prefix.upper()} BLIND DRAWER FINISHED ({report["total_wall_s"]:.1f}s) | self_col={total_col}, jlim={total_jlim} ===', flush=True)
    print(f'Saved results to {out_path}', flush=True)


if __name__ == '__main__':
    main()
