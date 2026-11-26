"""Phase 3 Zero-Training GravComp-TL Physics Gate & 2.0M-Step Converged RL Showdown.

1. Zero-Training Physics Verification Gate (~10 s on 10 CPU workers):
   Evaluates BlindSpiral with fixed GravComp-TL at alpha in {0.10, 0.20, 0.35} alongside
   BlindSpiral_Stiff (ts=1.0), BlindSpiral_LowTorque (ts=0.35), and Privileged_Oracle across
   Split A (bolted), Split B (unbolted), Split C (tight_0.5mm), and OOD Slippery (mu=0.25, 120 g).
2. Converged RL Showdown (2.0M policy steps/run x 2 seeds [101, 202] = 12.0M steps total):
   - M1_PosOnly_2M:     Phase 2 M1 (pos/vel + FK + ee_rel_start, fixed ts=1.0) scaled to 2.0M steps
   - M4_GravComp_TL:    M1 + tau_ext + d_tau_ext + tracking_error + active 2-group GravComp-TL [alpha_lat, alpha_arm]
   - M4_Fixed_GravComp: M1 + tau_ext + d_tau_ext + tracking_error + fixed GravComp-TL alpha=0.20

Keeps geometry.py, env.py, benchmarks.py, and controllers.py completely untouched, and preserves
Phase 2's exact training distribution ([0.35, 0.55] mu_table, [180, 320] g, seen shapes only) so
OOD Slippery (mu=0.25, 120 g) remains strictly OOD and Phase 2 -> Phase 3 is 100% apples-to-apples.
"""
from __future__ import annotations

from collections import deque
import json
import math
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
    TRAIN_SEED_OFFSET,
    build_regime_a_bolted,
    build_regime_b_unbolted,
    build_split_c_tight_0_5mm,
)
from projects.blind_insertion.controllers import (
    BlindSpiralController,
    NominalStraightDownController,
    PrivilegedOracleController,
    run_episode,
)
from projects.blind_insertion.env import InsertionTaskSpec
from projects.blind_insertion.geometry import SEEN_SHAPES, UNSEEN_SHAPES
from projects.blind_insertion.phase1_policy import CONTACT_CLASS_TO_IDX
from projects.blind_insertion.phase1_train import (
    HOLE_OUTER_RADIUS_BY_SHAPE,
    _init_worker_single_thread,
    reset_cached_env_inplace,
    verify_train_disjointness,
)
from projects.blind_insertion.phase2_policy import (
    BPTT_CHUNK_LEN,
    GRU_HIDDEN_DIM,
    PRIVILEGED_CRITIC_DIM_P2,
    build_privileged_critic_features_p2,
)
from projects.blind_insertion.phase2_train import (
    P2_PLATE_HARD_HALF_M,
    aggregate_seeds_p2,
    build_slippery_unbolted_ood_suite,
    compute_smooth_r_max_m,
    compute_step_reward_p2,
    sample_training_batch_p2,
    summarize_suite_p2,
)
from projects.blind_insertion.phase3_gravcomp import GravCompBlindInsertionEnv
from projects.blind_insertion.phase3_policy import (
    M4_FIXED_ALPHA,
    Phase3GravCompPPO,
    Phase3ProprioBuilder,
)

_P3_WORKER_ENV_CACHE: dict[tuple[str, float, bool], GravCompBlindInsertionEnv] = {}


def _get_cached_env_p3(task: InsertionTaskSpec) -> GravCompBlindInsertionEnv:
    key = (task.shape, round(task.clearance_m, 6), bool(task.bolted))
    env = _P3_WORKER_ENV_CACHE.get(key)
    if env is None:
        env = GravCompBlindInsertionEnv(task=task, render=False)
        _P3_WORKER_ENV_CACHE[key] = env
    return env


class GravCompBlindSpiralController(BlindSpiralController):
    """Exact Phase 0 BlindSpiralController outputting GravComp-TL fixed contact margin `alpha`."""

    def __init__(self, alpha: float = 0.20, **kwargs):
        super().__init__(torque_scale=1.0, **kwargs)
        self.alpha = float(alpha)

    def act(self, obs: dict, info: dict | None = None) -> dict:
        base_act = super().act(obs, info)
        return {
            'delta_xyz': base_act[:3].copy(),
            'gravcomp_alpha': np.array([self.alpha, self.alpha], dtype=float),
        }


def _eval_p3_baseline_worker(args: tuple[str, InsertionTaskSpec]) -> dict:
    cname, task = args
    env = GravCompBlindInsertionEnv(task=task, render=False)
    try:
        if cname == 'Nominal_StraightDown':
            ctrl = NominalStraightDownController(torque_scale=1.0)
        elif cname == 'BlindSpiral_Stiff_1.0':
            ctrl = BlindSpiralController(torque_scale=1.0)
        elif cname == 'BlindSpiral_LowTorque_0.35':
            ctrl = BlindSpiralController(torque_scale=0.35)
        elif cname == 'BlindSpiral_GravComp_0.10':
            ctrl = GravCompBlindSpiralController(alpha=0.10)
        elif cname == 'BlindSpiral_GravComp_0.20':
            ctrl = GravCompBlindSpiralController(alpha=0.20)
        elif cname == 'BlindSpiral_GravComp_0.35':
            ctrl = GravCompBlindSpiralController(alpha=0.35)
        elif cname == 'Privileged_Oracle':
            ctrl = PrivilegedOracleController()
        else:
            raise ValueError(cname)
        return run_episode(env, task, ctrl)
    finally:
        env.close()


def _train_rollout_worker_p3(job: dict) -> dict:
    task: InsertionTaskSpec = job['task']
    mode: str = job['mode']
    fixed_alpha: float = float(job.get('fixed_alpha', M4_FIXED_ALPHA))

    env = _get_cached_env_p3(task)
    obs, info = reset_cached_env_inplace(env, task)
    start_xyz = np.asarray(obs['ee_pose_fk'][:3], dtype=float).copy()

    builder = Phase3ProprioBuilder(mode=mode, fixed_alpha=fixed_alpha)
    pol = Phase3GravCompPPO(
        obs_dim=builder.obs_dim,
        mode=mode,
        fixed_alpha=fixed_alpha,
        critic_dim=PRIVILEGED_CRITIC_DIM_P2,
        hidden_dim=GRU_HIDDEN_DIM,
    )
    pol.set_weights(job['actor_weights'])
    h = pol.init_hidden()

    rng = np.random.default_rng(job['rollout_seed'])
    ar_rho = float(job.get('ar_rho', 0.0))
    innov = math.sqrt(max(1.0 - ar_rho * ar_rho, 0.0))
    eps = rng.normal(0.0, 1.0, size=pol.act_dim)

    obs_buf, crit_buf, z_buf, rew_buf, cls_buf, dir_buf, dmask_buf = [], [], [], [], [], [], []
    prev_priv = info['privileged']
    prev_dxyz = np.zeros(3, dtype=float)
    milestones = {'rim': False, 'entering': False}
    visited_plate_cells: set[tuple[int, int]] = set()
    terminated = False
    truncated = False
    off_plate_term = False

    while not (terminated or truncated):
        x_vec = builder.step(obs)
        c_vec = build_privileged_critic_features_p2(obs, info['privileged'], start_xyz, builder.visited_fraction)
        cls_idx = CONTACT_CLASS_TO_IDX[info['privileged']['contact_class']]
        xy_e = np.asarray(info['privileged']['socket_xy_error'], dtype=float)
        xy_n = float(np.linalg.norm(xy_e))
        if xy_n > 5e-4:
            dir_t = -xy_e / xy_n
            d_m = 1.0
        else:
            dir_t = np.zeros(2, dtype=float)
            d_m = 0.0

        eps = ar_rho * eps + innov * rng.normal(0.0, 1.0, size=pol.act_dim)
        env_act, d_xyz, alpha_pair, z, h, _, _ = pol.act_step_p3(x_vec, h, deterministic=False, eps=eps)
        if alpha_pair is not None:
            builder.set_prev_alpha(alpha_pair[0], alpha_pair[1])

        obs, _, terminated, truncated, info = env.step(env_act)
        ee_rel_xy = np.asarray(obs['ee_pose_fk'][:2], dtype=float) - start_xyz[:2]
        r, off_plate_term = compute_step_reward_p2(
            prev_priv, info['privileged'], d_xyz, prev_dxyz, info,
            env.step_count, task.max_steps, milestones,
            visited_plate_cells=visited_plate_cells, ee_rel_xy=ee_rel_xy,
        )
        if off_plate_term and not info['success']:
            terminated = True

        obs_buf.append(x_vec)
        crit_buf.append(c_vec)
        z_buf.append(z)
        rew_buf.append(r)
        cls_buf.append(cls_idx)
        dir_buf.append(dir_t)
        dmask_buf.append(d_m)

        prev_priv = info['privileged']
        prev_dxyz = d_xyz

    last_c = build_privileged_critic_features_p2(obs, info['privileged'], start_xyz, builder.visited_fraction)
    return {
        'obs': np.asarray(obs_buf, dtype=float),
        'critic_obs': np.asarray(crit_buf, dtype=float),
        'last_critic_obs': last_c,
        'z': np.asarray(z_buf, dtype=float),
        'rewards': np.asarray(rew_buf, dtype=float),
        'contact_labels': np.asarray(cls_buf, dtype=np.int64),
        'dir_targets': np.asarray(dir_buf, dtype=float),
        'dir_mask': np.asarray(dmask_buf, dtype=float),
        'terminated': bool(terminated),
        'success': bool(info['success']),
        'bolted': bool(task.bolted),
        'failure_mode': 'off_plate' if (off_plate_term and not info['success']) else info['failure_mode'],
        'steps': int(env.step_count),
    }


def _eval_rl_worker_p3(job: dict) -> dict:
    """Deterministic held-out evaluation on a fresh unmodified GravCompBlindInsertionEnv."""
    task: InsertionTaskSpec = job['task']
    mode: str = job['mode']
    fixed_alpha: float = float(job.get('fixed_alpha', M4_FIXED_ALPHA))

    env = GravCompBlindInsertionEnv(task=task, render=False)
    try:
        obs, info = env.reset()
        builder = Phase3ProprioBuilder(mode=mode, fixed_alpha=fixed_alpha)
        pol = Phase3GravCompPPO(
            obs_dim=builder.obs_dim,
            mode=mode,
            fixed_alpha=fixed_alpha,
            critic_dim=PRIVILEGED_CRITIC_DIM_P2,
            hidden_dim=GRU_HIDDEN_DIM,
        )
        pol.set_weights(job['actor_weights'])
        h = pol.init_hidden()

        terminated = False
        truncated = False
        max_depth_mm = float(info['privileged']['peg_depth'] * 1000.0)
        hole_r = HOLE_OUTER_RADIUS_BY_SHAPE[task.shape]
        went_off_plate = False
        aux_correct = 0
        aux_total = 0
        dir_cos_list: list[float] = []
        ts_by_phase: dict[str, list[float]] = {'search_rim': [], 'insert_seat': [], 'free': []}
        a_lat_by_phase: dict[str, list[float]] = {'search_rim': [], 'insert_seat': [], 'free': []}
        a_arm_by_phase: dict[str, list[float]] = {'search_rim': [], 'insert_seat': [], 'free': []}

        while not (terminated or truncated):
            x_vec = builder.step(obs)
            true_cls_name = info['privileged']['contact_class']
            true_cls = CONTACT_CLASS_TO_IDX[true_cls_name]
            env_act, _, alpha_pair, _, h, pred_cls, pred_dir = pol.act_step_p3(x_vec, h, deterministic=True)
            if alpha_pair is not None:
                builder.set_prev_alpha(alpha_pair[0], alpha_pair[1])

            aux_correct += int(pred_cls == true_cls)
            aux_total += 1
            xy_e = np.asarray(info['privileged']['socket_xy_error'], dtype=float)
            xy_n = float(np.linalg.norm(xy_e))
            pn = float(np.linalg.norm(pred_dir))
            if xy_n > 5e-4 and pn > 1e-6:
                dir_cos_list.append(float(np.dot(pred_dir / pn, -xy_e / xy_n)))

            obs, _, terminated, truncated, info = env.step(env_act)
            ts_mean = float(np.mean(obs['torque_scale'][:5]))
            if true_cls_name in ('plate_contact', 'rim_contact'):
                p_key = 'search_rim'
            elif true_cls_name in ('entering', 'seated'):
                p_key = 'insert_seat'
            else:
                p_key = 'free'
            ts_by_phase[p_key].append(ts_mean)
            if alpha_pair is not None:
                a_lat_by_phase[p_key].append(alpha_pair[0])
                a_arm_by_phase[p_key].append(alpha_pair[1])

            cur_p = info['privileged']
            d_mm = float(cur_p['peg_depth'] * 1000.0)
            if d_mm > max_depth_mm:
                max_depth_mm = d_mm
            cur_xy_v = np.asarray(cur_p['socket_xy_error'], dtype=float)
            cur_xy = float(np.linalg.norm(cur_xy_v))
            cur_linf = float(np.max(np.abs(cur_xy_v)))
            if cur_linf > P2_PLATE_HARD_HALF_M or (float(cur_p['peg_depth']) > 0.0020 and cur_xy > hole_r + 0.0030):
                went_off_plate = True

        priv = info['privileged']
        return {
            'trial_id': task.trial_id,
            'seed': task.seed,
            'shape': task.shape,
            'split': 'seen' if task.shape in SEEN_SHAPES else 'unseen',
            'bolted': task.bolted,
            'clearance_mm': round(task.clearance_m * 1000.0, 3),
            'misalign_mm': round(float(np.linalg.norm(task.misalign_xy) * 1000.0), 2),
            'success': bool(info['success']),
            'failure_mode': info['failure_mode'],
            'went_off_plate': bool(went_off_plate),
            'steps': int(env.step_count),
            'time_s': round(float(info['time_s']), 3),
            'final_depth_mm': round(float(priv['peg_depth'] * 1000.0), 2),
            'max_depth_mm': round(max_depth_mm, 2),
            'peak_contact_force_N': round(float(info['episode_peak_contact_force_N']), 2),
            'max_socket_disp_mm': round(float(info['episode_max_socket_disp_mm']), 2),
            'max_socket_tilt_deg': round(float(info['episode_max_socket_tilt_deg']), 2),
            'self_collision_steps': int(info['self_collision_steps']),
            'joint_limit_steps': int(info['joint_limit_steps']),
            'aux_contact_acc': aux_correct / max(aux_total, 1),
            'aux_dir_cosine': float(np.mean(dir_cos_list)) if dir_cos_list else 0.0,
            'ts_search_rim': float(np.mean(ts_by_phase['search_rim'])) if ts_by_phase['search_rim'] else None,
            'ts_insert_seat': float(np.mean(ts_by_phase['insert_seat'])) if ts_by_phase['insert_seat'] else None,
            'ts_free': float(np.mean(ts_by_phase['free'])) if ts_by_phase['free'] else None,
            'alpha_lat_free': float(np.mean(a_lat_by_phase['free'])) if a_lat_by_phase['free'] else None,
            'alpha_arm_free': float(np.mean(a_arm_by_phase['free'])) if a_arm_by_phase['free'] else None,
            'alpha_lat_search_rim': float(np.mean(a_lat_by_phase['search_rim'])) if a_lat_by_phase['search_rim'] else None,
            'alpha_arm_search_rim': float(np.mean(a_arm_by_phase['search_rim'])) if a_arm_by_phase['search_rim'] else None,
            'alpha_lat_insert_seat': float(np.mean(a_lat_by_phase['insert_seat'])) if a_lat_by_phase['insert_seat'] else None,
            'alpha_arm_insert_seat': float(np.mean(a_arm_by_phase['insert_seat'])) if a_arm_by_phase['insert_seat'] else None,
        }
    finally:
        env.close()


def summarize_suite_p3(rows: list[dict]) -> dict:
    summ = summarize_suite_p2(rows)
    summ['jammed_on_rim_count'] = summ['failure_modes'].get('jammed_on_rim', 0)
    summ['jammed_on_plate_count'] = summ['failure_modes'].get('jammed_on_plate', 0)
    summ['hole_reach_rate'] = (summ['successes'] + summ['jammed_on_rim_count']) / max(summ['trials'], 1)
    for k in (
        'alpha_lat_free', 'alpha_arm_free',
        'alpha_lat_search_rim', 'alpha_arm_search_rim',
        'alpha_lat_insert_seat', 'alpha_arm_insert_seat',
    ):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            summ[f'mean_{k}'] = float(np.mean(vals))
    return summ


def aggregate_seeds_p3(runs: list[dict], suite_key: str) -> dict:
    agg = aggregate_seeds_p2(runs, suite_key)
    sums = [r['eval'][suite_key]['summary'] for r in runs]
    agg['mean_jammed_on_rim'] = float(np.mean([s['jammed_on_rim_count'] for s in sums]))
    agg['total_jammed_on_rim'] = int(sum(s['jammed_on_rim_count'] for s in sums))
    agg['mean_jammed_on_plate'] = float(np.mean([s['jammed_on_plate_count'] for s in sums]))
    agg['mean_hole_reach_rate'] = float(np.mean([s['hole_reach_rate'] for s in sums]))
    for k in (
        'mean_alpha_lat_free', 'mean_alpha_arm_free',
        'mean_alpha_lat_search_rim', 'mean_alpha_arm_search_rim',
        'mean_alpha_lat_insert_seat', 'mean_alpha_arm_insert_seat',
    ):
        vals = [s[k] for s in sums if s.get(k) is not None]
        if vals:
            agg[k] = float(np.mean(vals))
    return agg


def train_one_seed_p3(
    pool,
    label: str,
    mode: str,
    seed: int,
    seed_cursor: int,
    target_steps: int = 2_000_000,
    batch_size: int = 20,
    max_steps: int = 220,
    fixed_alpha: float = M4_FIXED_ALPHA,
) -> tuple[Phase3GravCompPPO, dict, int, list[InsertionTaskSpec]]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    builder = Phase3ProprioBuilder(mode=mode, fixed_alpha=fixed_alpha)
    pol = Phase3GravCompPPO(
        obs_dim=builder.obs_dim,
        mode=mode,
        fixed_alpha=fixed_alpha,
        critic_dim=PRIVILEGED_CRITIC_DIM_P2,
        hidden_dim=GRU_HIDDEN_DIM,
        seed=seed,
        init_log_std=-0.65,
    )
    history = []
    all_trials: list[InsertionTaskSpec] = []
    cum_steps = 0
    upd = 0
    next_log_steps = 0
    rolling_succ: deque[int] = deque(maxlen=100)

    while cum_steps < target_steps:
        progress = min(1.0, cum_steps / max(target_steps, 1))
        aw = pol.get_actor_weights()
        ar_rho_cur = 0.70 * max(0.0, 1.0 - progress / 0.70)
        log_std_cap = -0.55 + (-1.15 - (-0.55)) * min(1.0, progress / 0.90)

        trajs: list[dict] = []
        batch_steps = 0
        eff_r_max = compute_smooth_r_max_m(progress)
        while batch_steps < 1800:
            wave_trials, seed_cursor, eff_r_max = sample_training_batch_p2(
                rng, batch_size=batch_size, seed_cursor=seed_cursor, progress=progress, max_steps=max_steps,
            )
            all_trials.extend(wave_trials)
            rseeds = rng.integers(1, 2**31 - 1, size=batch_size)
            jobs = [
                {
                    'task': t, 'mode': mode, 'fixed_alpha': fixed_alpha,
                    'actor_weights': aw, 'rollout_seed': int(rs), 'ar_rho': ar_rho_cur,
                }
                for t, rs in zip(wave_trials, rseeds)
            ]
            wave_trajs = pool.map(_train_rollout_worker_p3, jobs)
            trajs.extend(wave_trajs)
            batch_steps += sum(tr['steps'] for tr in wave_trajs)

        if upd == 0:
            pol.obs_rms.update(np.concatenate([tr['obs'] for tr in trajs], axis=0))
            pol.critic_rms.update(np.concatenate([tr['critic_obs'] for tr in trajs], axis=0))

        lr_scale = 0.20 + 0.80 * (1.0 - progress)
        stats = pol.ppo_update(
            trajs,
            rng=rng,
            actor_lr=6.5e-4 * lr_scale,
            critic_lr=1.6e-3 * lr_scale,
            gamma=0.995,
            lam=0.95,
            clip_eps=0.2,
            epochs=4,
            chunk_len=BPTT_CHUNK_LEN,
            chunk_minibatch=8,
            critic_minibatch=256,
            entropy_coef=0.0005,
            aux_cls_coef=0.08,
            aux_dir_coef=0.08,
            max_grad_norm=0.5,
            target_kl=0.03,
            weight_decay=1e-4,
            log_std_max=log_std_cap,
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
        shoves = sum(tr['failure_mode'] == 'socket_shoved' for tr in u_tr)
        off_plates = sum(tr['failure_mode'] == 'off_plate' for tr in trajs)

        entry = {
            'update': upd,
            'episodes_batch': len(trajs),
            'episodes_cum': len(all_trials),
            'steps_cum': cum_steps,
            'r_max_mm': round(eff_r_max * 1000.0, 2),
            'roll100_succ': round(roll_100, 3),
            'train_succ': succ_all,
            'train_succ_bolted': succ_b,
            'train_succ_unbolted': succ_u,
            'train_unbolted_shoves': shoves,
            'train_off_plate': off_plates,
            **stats,
        }
        history.append(entry)
        if cum_steps >= next_log_steps or cum_steps >= target_steps:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{label:18s} s={seed}] upd {upd:3d} ({cum_steps:7d}/{target_steps} st, r<={eff_r_max*1000:4.1f}mm, {cum_steps/max(elapsed,1e-6):5.0f} st/s) | "
                f"succ={100*succ_all:5.1f}% (r100={100*roll_100:4.1f}%, B={100*succ_b:4.0f}% U={100*succ_u:4.0f}% shv={shoves:2d} off={off_plates:2d}/{len(trajs)}) | "
                f"ret={stats['mean_return']:+.2f} ev={stats['explained_var']:+.2f} aux={100*stats['aux_cls_acc']:4.1f}% "
                f"kl={stats['approx_kl']:.4f} std={np.round(stats['std'], 2)}",
                flush=True,
            )
            next_log_steps = cum_steps + max(target_steps // 8, 1)

    wall_s = time.perf_counter() - t0
    total_off_plate_train = sum(e['train_off_plate'] for e in history)
    meta = {
        'label': label,
        'mode': mode,
        'fixed_alpha': fixed_alpha if mode == 'gravcomp_fixed' else None,
        'seed': seed,
        'obs_dim': builder.obs_dim,
        'act_dim': pol.act_dim,
        'hidden_dim': pol.hidden_dim,
        'actor_param_count': pol.actor_param_count,
        'critic_param_count': pol.critic_param_count,
        'n_updates': upd,
        'batch_size': batch_size,
        'episodes_budget': len(all_trials),
        'total_steps': cum_steps,
        'target_steps': target_steps,
        'final_r_max_mm': round(compute_smooth_r_max_m(1.0) * 1000.0, 2),
        'train_off_plate_rate_all': total_off_plate_train / max(len(all_trials), 1),
        'train_off_plate_rate_last10': float(np.mean([e['train_off_plate'] / max(e['episodes_batch'], 1) for e in history[-10:]])),
        'wall_s': round(wall_s, 2),
        'steps_per_sec': round(cum_steps / max(wall_s, 1e-6), 1),
        'episodes_per_sec': round(len(all_trials) / max(wall_s, 1e-6), 2),
        'history': history,
    }
    return pol, meta, seed_cursor, all_trials


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--target-steps', type=int, default=2_000_000)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=220)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202])
    parser.add_argument('--methods', type=str, nargs='+', default=['M1_PosOnly_2M', 'M4_GravComp_TL', 'M4_Fixed_GravComp'])
    parser.add_argument('--out-prefix', type=str, default='phase3')
    parser.add_argument('--seed-offset', type=int, default=240000)
    parser.add_argument('--gate-only', action='store_true')
    args = parser.parse_args()

    t_total = time.perf_counter()
    ckpt_dir = THIS_DIR / f'{args.out_prefix}_checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    eval_suites = {
        'split_a_bolted': build_regime_a_bolted(),
        'split_b_unbolted': build_regime_b_unbolted(),
        'split_c_tight_0.5mm': build_split_c_tight_0_5mm(),
        'ood_slippery_mu0.25_120g': build_slippery_unbolted_ood_suite(),
    }

    all_rl_specs = [
        ('M1_PosOnly_2M', 'pos_only'),
        ('M4_GravComp_TL', 'gravcomp_tl'),
        ('M4_Fixed_GravComp', 'gravcomp_fixed'),
    ]
    rl_specs = [(lbl, m) for lbl, m in all_rl_specs if lbl in set(args.methods)]

    ctx = mp.get_context('spawn')
    seed_cursor = TRAIN_SEED_OFFSET + int(args.seed_offset)
    all_train_trials: list[InsertionTaskSpec] = []
    rl_runs: dict[str, list[dict]] = {label: [] for label, _ in rl_specs}

    baseline_names = (
        'Nominal_StraightDown',
        'BlindSpiral_Stiff_1.0',
        'BlindSpiral_LowTorque_0.35',
        'BlindSpiral_GravComp_0.10',
        'BlindSpiral_GravComp_0.20',
        'BlindSpiral_GravComp_0.35',
        'Privileged_Oracle',
    )

    with ctx.Pool(args.workers, initializer=_init_worker_single_thread) as pool:
        t_gate = time.perf_counter()
        print('\n=== 1. Zero-Training Physics Verification Gate: GravComp-TL BlindSpiral (alpha in {0.10, 0.20, 0.35}) ===', flush=True)
        baselines = {}
        for cname in baseline_names:
            baselines[cname] = {}
            parts = []
            for sk, trials in eval_suites.items():
                rows = pool.map(_eval_p3_baseline_worker, [(cname, t) for t in trials])
                summ = summarize_suite_p3(rows)
                baselines[cname][sk] = {'summary': summ, 'episodes': rows}
                parts.append(
                    f'{sk[:8]}={summ["successes"]:2d}/30({100*summ["success_rate"]:4.1f}%,'
                    f'F={summ["median_peak_contact_force_N"]:4.1f}N,shv={summ["shove_failures"]:2d},d={summ["median_socket_disp_mm"]:4.1f}mm)'
                )
            print(f'  {cname:27s} | ' + ' | '.join(parts), flush=True)
        print(f'Zero-Training Physics Gate completed in {time.perf_counter() - t_gate:.1f}s!', flush=True)

        if args.gate_only:
            out_gate = THIS_DIR / f'{args.out_prefix}_gate.json'
            out_gate.write_text(json.dumps({'baselines': baselines}, indent=2) + '\n')
            print(f'Saved gate table to {out_gate}', flush=True)
            return

        print(
            f'\nLaunching Phase 3 Converged RL Showdown ({args.workers} headless CPU workers, '
            f'{len(rl_specs)} methods x {len(args.seeds)} seeds x ~{args.target_steps:,} steps, max_steps={args.max_steps})...',
            flush=True,
        )
        for label, mode in rl_specs:
            print(f'\n=== 2. Training {label} (GRU-{GRU_HIDDEN_DIM}, mode={mode}, {len(args.seeds)} seeds x {args.target_steps:,} steps) ===', flush=True)
            for seed in args.seeds:
                pol, meta, seed_cursor, used_trials = train_one_seed_p3(
                    pool, label, mode, seed=seed, seed_cursor=seed_cursor,
                    target_steps=args.target_steps, batch_size=args.batch_size, max_steps=args.max_steps,
                )
                all_train_trials.extend(used_trials)
                ckpt_path = ckpt_dir / f'{label}_seed{seed}.npz'
                pol.save(ckpt_path)
                meta['checkpoint'] = str(ckpt_path.relative_to(THIS_DIR))

                aw = pol.get_actor_weights()
                eval_out = {}
                parts = []
                for sk, trials in eval_suites.items():
                    jobs = [{'task': t, 'mode': mode, 'fixed_alpha': M4_FIXED_ALPHA, 'actor_weights': aw} for t in trials]
                    rows = pool.map(_eval_rl_worker_p3, jobs)
                    summ = summarize_suite_p3(rows)
                    eval_out[sk] = {'summary': summ, 'episodes': rows}
                    parts.append(
                        f'{sk[:8]}={summ["successes"]:2d}/30({100*summ["success_rate"]:4.1f}%,'
                        f'rim={summ["jammed_on_rim_count"]:2d},F={summ["median_peak_contact_force_N"]:4.1f}N,shv={summ["shove_failures"]:2d})'
                    )
                print(f'  EVAL [{label:18s} s={seed}] ({meta["wall_s"]:.0f}s) | ' + ' | '.join(parts), flush=True)
                rl_runs[label].append({'meta': meta, 'eval': eval_out})

    verify_train_disjointness(all_train_trials)
    print(f'\nVerified strict seed & shape disjointness across {len(all_train_trials)} training episodes (SEEN shapes only)!', flush=True)

    rl_aggregate = {}
    for label, _ in rl_specs:
        runs = rl_runs[label]
        rl_aggregate[label] = {
            'mode': runs[0]['meta']['mode'],
            'obs_dim': runs[0]['meta']['obs_dim'],
            'act_dim': runs[0]['meta']['act_dim'],
            'hidden_dim': runs[0]['meta']['hidden_dim'],
            'actor_param_count': runs[0]['meta']['actor_param_count'],
            'critic_param_count': runs[0]['meta']['critic_param_count'],
            'mean_episodes_per_seed': float(np.mean([r['meta']['episodes_budget'] for r in runs])),
            'mean_total_steps': float(np.mean([r['meta']['total_steps'] for r in runs])),
            'mean_train_wall_s': float(np.mean([r['meta']['wall_s'] for r in runs])),
            'mean_steps_per_sec': float(np.mean([r['meta']['steps_per_sec'] for r in runs])),
            'eval': {sk: aggregate_seeds_p3(runs, sk) for sk in eval_suites},
        }

    total_col = sum(
        v['eval'][sk]['total_self_collision_steps'] for v in rl_aggregate.values() for sk in eval_suites
    ) + sum(
        baselines[b][sk]['summary']['self_collision_steps'] for b in baselines for sk in eval_suites
    )
    total_jlim = sum(
        v['eval'][sk]['total_joint_limit_steps'] for v in rl_aggregate.values() for sk in eval_suites
    ) + sum(
        baselines[b][sk]['summary']['joint_limit_steps'] for b in baselines for sk in eval_suites
    )

    report = {
        'total_wall_s': round(time.perf_counter() - t_total, 2),
        'n_train_episodes_total': len(all_train_trials),
        'train_shapes_only': list(SEEN_SHAPES),
        'heldout_unseen_shapes': list(UNSEEN_SHAPES),
        'total_eval_self_collision_steps': total_col,
        'total_eval_joint_limit_steps': total_jlim,
        'baselines': baselines,
        'rl_aggregate': rl_aggregate,
        'rl_runs': rl_runs,
    }
    out_path = THIS_DIR / f'{args.out_prefix}_results.json'
    out_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'\n=== {args.out_prefix.upper()} FINISHED ({report["total_wall_s"]:.1f}s) | self_col={total_col}, jlim={total_jlim} ===', flush=True)
    print(f'Saved results to {out_path}', flush=True)


if __name__ == '__main__':
    main()
