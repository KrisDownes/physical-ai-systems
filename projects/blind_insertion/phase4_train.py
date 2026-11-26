"""Phase 4 Clean-Kinematics Residual RL + Spiral Brake Gate (`g_spiral`) + Dynamic `GravComp-TL`.

Clean 2-way ablation (1.5M policy steps/run x 3 seeds [101, 202, 303] = 9.0M steps total):
  1. `R1_Res_PosOnly` (`res_pos_only`):       Residual + `g_spiral` brake gate on frozen `BlindSpiral_Stiff` (`ts=1.0`)
  2. `R2_Res_GravComp_TL` (`res_gravcomp_tl`): Residual + `g_spiral` brake gate on frozen `BlindSpiral_GravComp (alpha=0.35)`
                                               + dynamic `[alpha_lat, alpha_arm] in [0.08, 1.0]`

Both methods share identical GRU(64) backbone, clean high-SNR kinematic + `tracking_error` + `a_base_t` sensing
(ZERO raw `actuator_force_noisy`, ZERO `d_tau_ext`), asymmetric critic, dual aux decoder, hole-gated depth shaping,
off-plate guard, and exact Phase 2/3 training distribution (`SEEN_SHAPES`, `mu_table in [0.35, 0.55]`, `180-320 g`)
so OOD Slippery (`mu=0.25, 120 g`) remains strictly out-of-distribution.
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
from projects.blind_insertion.controllers import BlindSpiralController
from projects.blind_insertion.env import InsertionTaskSpec
from projects.blind_insertion.geometry import SEEN_SHAPES, UNSEEN_SHAPES
from projects.blind_insertion.phase1_policy import CONTACT_CLASS_TO_IDX
from projects.blind_insertion.phase1_train import (
    HOLE_OUTER_RADIUS_BY_SHAPE,
    TRAIN_CLEARANCES_M,
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
    build_slippery_unbolted_ood_suite,
    compute_step_reward_p2,
)
from projects.blind_insertion.phase3_gravcomp import GravCompBlindInsertionEnv
from projects.blind_insertion.phase3_train import (
    _eval_p3_baseline_worker,
    _get_cached_env_p3,
    aggregate_seeds_p3,
    summarize_suite_p3,
)
from projects.blind_insertion.phase4_policy import (
    BASE_SPIRAL_ALPHA,
    Phase4ResidualBuilder,
    Phase4ResidualPPO,
)


class PausableBaseSpiralController(BlindSpiralController):
    """Frozen BlindSpiral base controller whose spiral angle `theta` pauses proportionally to `g_spiral`.

    When `g_spiral == 1.0` (step-0 zero-init), this is 100% bit-identical to `BlindSpiralController`!
    When the residual policy drops `g_spiral -> 0.0` upon catching a 0.5 mm rim, `theta` pauses
    right at the rim radius instead of running away while the policy micro-centers.
    """

    def __init__(self, torque_scale: float = 1.0):
        super().__init__(torque_scale=torque_scale)
        self._pause_scale: float = 1.0

    def reset(self, obs: dict) -> None:
        super().reset(obs)
        self._pause_scale = 1.0

    def set_spiral_gate(self, g_spiral: float) -> None:
        self._pause_scale = float(np.clip(g_spiral, 0.0, 1.0))

    def act_base(self, obs: dict) -> np.ndarray:
        """Return base `d_xyz_3` with exact BlindSpiralController state machine & gated theta advance."""
        prev_theta = self.theta
        base_4 = super().act(obs, None)
        if self.phase == 'spiral' and self._pause_scale < 1.0:
            # Undo full dtheta and apply paused advance `pause_scale * dtheta`
            dtheta = self.theta - prev_theta
            self.theta = prev_theta + self._pause_scale * dtheta
        return base_4[:3].copy()


def sample_training_batch_p4(
    rng: np.random.Generator,
    batch_size: int,
    seed_cursor: int,
    max_steps: int = 250,
) -> tuple[list[InsertionTaskSpec], int]:
    """Exact Phase 2/3 training distribution (SEEN shapes, uniform [0.5, 1.5] mm, [0.35, 0.55] mu, [180, 320] g).

    Because episode 0 starts exact at 70%/73%/33% via the frozen base spiral, we train directly on
    full [6.0, 16.5] mm (with 20% <= 10 mm replay) from step 0.
    """
    if seed_cursor < TRAIN_SEED_OFFSET:
        raise ValueError(f'seed_cursor must be >= {TRAIN_SEED_OFFSET}')

    trials: list[InsertionTaskSpec] = []
    for i in range(batch_size):
        shape = SEEN_SHAPES[i % len(SEEN_SHAPES)]
        bolted = (i % 2 == 0)
        clearance = float(TRAIN_CLEARANCES_M[int(rng.integers(0, len(TRAIN_CLEARANCES_M)))])
        if rng.random() < 0.20:
            r = float(rng.uniform(0.0010, 0.0100))
        else:
            r = float(rng.uniform(0.0060, 0.0165))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        trials.append(InsertionTaskSpec(
            trial_id=f'p4-train-{seed_cursor}',
            seed=seed_cursor,
            shape=shape,
            clearance_m=clearance,
            bolted=bolted,
            socket_xy=(float(0.220 + rng.uniform(-0.012, 0.012)), float(rng.uniform(-0.015, 0.015))),
            misalign_xy=(r * math.cos(th), r * math.sin(th)),
            start_clearance_z_m=float(rng.uniform(0.001, 0.0025)),
            grasp_offset_xy=(float(rng.uniform(-0.0015, 0.0015)), float(rng.uniform(-0.0015, 0.0015))),
            mu_peg_hole=float(rng.uniform(0.15, 0.45)),
            mu_table=float(rng.uniform(0.35, 0.55)),
            socket_mass_kg=float(rng.uniform(0.180, 0.320)),
            backlash_rad=float(rng.uniform(0.003, 0.006)),
            max_steps=max_steps,
        ))
        seed_cursor += 1
    return trials, seed_cursor


def _train_rollout_worker_p4(job: dict) -> dict:
    task: InsertionTaskSpec = job['task']
    mode: str = job['mode']

    env = _get_cached_env_p3(task)
    obs, info = reset_cached_env_inplace(env, task)
    start_xyz = np.asarray(obs['ee_pose_fk'][:3], dtype=float).copy()

    base_ctrl = PausableBaseSpiralController(torque_scale=1.0)
    base_ctrl.reset(obs)

    builder = Phase4ResidualBuilder(mode=mode)
    pol = Phase4ResidualPPO(
        obs_dim=builder.obs_dim,
        mode=mode,
        base_alpha=BASE_SPIRAL_ALPHA,
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
        base_dxyz = base_ctrl.act_base(obs)
        base_r_m = min((base_ctrl.pitch_m / (2.0 * math.pi)) * base_ctrl.theta, base_ctrl.max_radius_m)

        x_vec = builder.step(obs, base_dxyz, base_ctrl.phase, base_r_m, base_ctrl.insert_stall_steps)
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
        env_act, d_xyz, g_spiral, alpha_pair, z, h, _, _ = pol.act_step_p4(x_vec, h, base_dxyz, deterministic=False, eps=eps)
        base_ctrl.set_spiral_gate(g_spiral)
        builder.set_prev_controls(g_spiral, alpha_pair)

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
        'tight': bool(task.clearance_m <= 0.00055),
        'failure_mode': 'off_plate' if (off_plate_term and not info['success']) else info['failure_mode'],
        'steps': int(env.step_count),
    }


def _eval_rl_worker_p4(job: dict) -> dict:
    """Deterministic held-out evaluation on a fresh unmodified GravCompBlindInsertionEnv."""
    task: InsertionTaskSpec = job['task']
    mode: str = job['mode']

    env = GravCompBlindInsertionEnv(task=task, render=False)
    try:
        obs, info = env.reset()
        base_ctrl = PausableBaseSpiralController(torque_scale=1.0)
        base_ctrl.reset(obs)

        builder = Phase4ResidualBuilder(mode=mode)
        pol = Phase4ResidualPPO(
            obs_dim=builder.obs_dim,
            mode=mode,
            base_alpha=BASE_SPIRAL_ALPHA,
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
        ts_by_phase: dict[str, list[float]] = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}
        g_by_phase: dict[str, list[float]] = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}
        a_lat_by_phase: dict[str, list[float]] = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}
        a_arm_by_phase: dict[str, list[float]] = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}

        while not (terminated or truncated):
            base_dxyz = base_ctrl.act_base(obs)
            base_r_m = min((base_ctrl.pitch_m / (2.0 * math.pi)) * base_ctrl.theta, base_ctrl.max_radius_m)

            x_vec = builder.step(obs, base_dxyz, base_ctrl.phase, base_r_m, base_ctrl.insert_stall_steps)
            true_cls_name = info['privileged']['contact_class']
            true_cls = CONTACT_CLASS_TO_IDX[true_cls_name]
            env_act, _, g_spiral, alpha_pair, _, h, pred_cls, pred_dir = pol.act_step_p4(x_vec, h, base_dxyz, deterministic=True)
            base_ctrl.set_spiral_gate(g_spiral)
            builder.set_prev_controls(g_spiral, alpha_pair)

            aux_correct += int(pred_cls == true_cls)
            aux_total += 1
            xy_e = np.asarray(info['privileged']['socket_xy_error'], dtype=float)
            xy_n = float(np.linalg.norm(xy_e))
            pn = float(np.linalg.norm(pred_dir))
            if xy_n > 5e-4 and pn > 1e-6:
                dir_cos_list.append(float(np.dot(pred_dir / pn, -xy_e / xy_n)))

            obs, _, terminated, truncated, info = env.step(env_act)
            ts_mean = float(np.mean(obs['torque_scale'][:5]))
            if true_cls_name == 'plate_contact':
                p_key = 'plate'
            elif true_cls_name == 'rim_contact':
                p_key = 'rim'
            elif true_cls_name in ('entering', 'seated'):
                p_key = 'insert_seat'
            else:
                p_key = 'free'
            ts_by_phase[p_key].append(ts_mean)
            g_by_phase[p_key].append(g_spiral)
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
        sr_ts = ts_by_phase['plate'] + ts_by_phase['rim']
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
            'ts_free': float(np.mean(ts_by_phase['free'])) if ts_by_phase['free'] else None,
            'ts_search_rim': float(np.mean(sr_ts)) if sr_ts else None,
            'ts_insert_seat': float(np.mean(ts_by_phase['insert_seat'])) if ts_by_phase['insert_seat'] else None,
            'g_spiral_free': float(np.mean(g_by_phase['free'])) if g_by_phase['free'] else None,
            'g_spiral_plate': float(np.mean(g_by_phase['plate'])) if g_by_phase['plate'] else None,
            'g_spiral_rim': float(np.mean(g_by_phase['rim'])) if g_by_phase['rim'] else None,
            'g_spiral_insert_seat': float(np.mean(g_by_phase['insert_seat'])) if g_by_phase['insert_seat'] else None,
            'alpha_lat_free': float(np.mean(a_lat_by_phase['free'])) if a_lat_by_phase['free'] else None,
            'alpha_arm_free': float(np.mean(a_arm_by_phase['free'])) if a_arm_by_phase['free'] else None,
            'alpha_lat_plate': float(np.mean(a_lat_by_phase['plate'])) if a_lat_by_phase['plate'] else None,
            'alpha_arm_plate': float(np.mean(a_arm_by_phase['plate'])) if a_arm_by_phase['plate'] else None,
            'alpha_lat_rim': float(np.mean(a_lat_by_phase['rim'])) if a_lat_by_phase['rim'] else None,
            'alpha_arm_rim': float(np.mean(a_arm_by_phase['rim'])) if a_arm_by_phase['rim'] else None,
            'alpha_lat_insert_seat': float(np.mean(a_lat_by_phase['insert_seat'])) if a_lat_by_phase['insert_seat'] else None,
            'alpha_arm_insert_seat': float(np.mean(a_arm_by_phase['insert_seat'])) if a_arm_by_phase['insert_seat'] else None,
        }
    finally:
        env.close()


P4_PHASE_KEYS = (
    'g_spiral_free', 'g_spiral_plate', 'g_spiral_rim', 'g_spiral_insert_seat',
    'alpha_lat_free', 'alpha_arm_free',
    'alpha_lat_plate', 'alpha_arm_plate',
    'alpha_lat_rim', 'alpha_arm_rim',
    'alpha_lat_insert_seat', 'alpha_arm_insert_seat',
)


def summarize_suite_p4(rows: list[dict]) -> dict:
    summ = summarize_suite_p3(rows)
    for k in P4_PHASE_KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            summ[f'mean_{k}'] = float(np.mean(vals))
    return summ


def aggregate_seeds_p4(runs: list[dict], suite_key: str) -> dict:
    agg = aggregate_seeds_p3(runs, suite_key)
    sums = [r['eval'][suite_key]['summary'] for r in runs]
    for k in P4_PHASE_KEYS:
        mk = f'mean_{k}'
        vals = [s[mk] for s in sums if s.get(mk) is not None]
        if vals:
            agg[mk] = float(np.mean(vals))
    return agg


def train_one_seed_p4(
    pool,
    label: str,
    mode: str,
    seed: int,
    seed_cursor: int,
    target_steps: int = 1_500_000,
    batch_size: int = 20,
    max_steps: int = 250,
) -> tuple[Phase4ResidualPPO, dict, int, list[InsertionTaskSpec]]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    builder = Phase4ResidualBuilder(mode=mode)
    pol = Phase4ResidualPPO(
        obs_dim=builder.obs_dim,
        mode=mode,
        base_alpha=BASE_SPIRAL_ALPHA,
        critic_dim=PRIVILEGED_CRITIC_DIM_P2,
        hidden_dim=GRU_HIDDEN_DIM,
        seed=seed,
        init_log_std=-1.10,  # std=0.33: gentle residual exploration around exact frozen base spiral!
    )
    history = []
    all_trials: list[InsertionTaskSpec] = []
    cum_steps = 0
    upd = 0
    next_log_steps = 0
    rolling_succ: deque[int] = deque(maxlen=100)
    rolling_tight: deque[int] = deque(maxlen=100)

    while cum_steps < target_steps:
        progress = min(1.0, cum_steps / max(target_steps, 1))
        aw = pol.get_actor_weights()
        ar_rho_cur = 0.40 * max(0.0, 1.0 - progress / 0.50)
        log_std_cap = -1.00 + (-1.55 - (-1.00)) * min(1.0, progress / 0.90)

        trajs: list[dict] = []
        batch_steps = 0
        while batch_steps < 1800:
            wave_trials, seed_cursor = sample_training_batch_p4(
                rng, batch_size=batch_size, seed_cursor=seed_cursor, max_steps=max_steps,
            )
            all_trials.extend(wave_trials)
            rseeds = rng.integers(1, 2**31 - 1, size=batch_size)
            jobs = [
                {'task': t, 'mode': mode, 'actor_weights': aw, 'rollout_seed': int(rs), 'ar_rho': ar_rho_cur}
                for t, rs in zip(wave_trials, rseeds)
            ]
            wave_trajs = pool.map(_train_rollout_worker_p4, jobs)
            trajs.extend(wave_trajs)
            batch_steps += sum(tr['steps'] for tr in wave_trajs)

        if upd == 0:
            pol.obs_rms.update(np.concatenate([tr['obs'] for tr in trajs], axis=0))
            pol.critic_rms.update(np.concatenate([tr['critic_obs'] for tr in trajs], axis=0))

        lr_scale = 0.20 + 0.80 * (1.0 - progress)
        stats = pol.ppo_update(
            trajs,
            rng=rng,
            actor_lr=5.0e-4 * lr_scale,
            critic_lr=1.4e-3 * lr_scale,
            gamma=0.995,
            lam=0.95,
            clip_eps=0.2,
            epochs=4,
            chunk_len=BPTT_CHUNK_LEN,
            chunk_minibatch=8,
            critic_minibatch=256,
            entropy_coef=0.0003,
            aux_cls_coef=0.08,
            aux_dir_coef=0.08,
            max_grad_norm=0.5,
            target_kl=0.025,
            weight_decay=1e-4,
            log_std_max=log_std_cap,
        )
        upd += 1
        cum_steps += stats['n_steps']
        for tr in trajs:
            rolling_succ.append(int(tr['success']))
            if tr['tight']:
                rolling_tight.append(int(tr['success']))
        roll_100 = float(np.mean(rolling_succ)) if rolling_succ else 0.0
        roll_tight = float(np.mean(rolling_tight)) if rolling_tight else 0.0
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
            'roll100_succ': round(roll_100, 3),
            'roll100_tight': round(roll_tight, 3),
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
                f"  [{label:18s} s={seed}] upd {upd:3d} ({cum_steps:7d}/{target_steps} st, {cum_steps/max(elapsed,1e-6):5.0f} st/s) | "
                f"succ={100*succ_all:5.1f}% (r100={100*roll_100:4.1f}%, tight={100*roll_tight:4.1f}%, B={100*succ_b:4.0f}% U={100*succ_u:4.0f}% shv={shoves:2d} off={off_plates:2d}/{len(trajs)}) | "
                f"ret={stats['mean_return']:+.2f} ev={stats['explained_var']:+.2f} aux={100*stats['aux_cls_acc']:4.1f}% "
                f"kl={stats['approx_kl']:.4f} std={np.round(stats['std'], 2)}",
                flush=True,
            )
            next_log_steps = cum_steps + max(target_steps // 6, 1)

    wall_s = time.perf_counter() - t0
    total_off_plate_train = sum(e['train_off_plate'] for e in history)
    meta = {
        'label': label,
        'mode': mode,
        'base_alpha': BASE_SPIRAL_ALPHA if mode == 'res_gravcomp_tl' else None,
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
    parser.add_argument('--target-steps', type=int, default=1_500_000)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=250)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--methods', type=str, nargs='+', default=['R1_Res_PosOnly', 'R2_Res_GravComp_TL'])
    parser.add_argument('--out-prefix', type=str, default='phase4')
    parser.add_argument('--seed-offset', type=int, default=350000)
    parser.add_argument('--skip-baselines', action='store_true')
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
        ('R1_Res_PosOnly', 'res_pos_only'),
        ('R2_Res_GravComp_TL', 'res_gravcomp_tl'),
    ]
    rl_specs = [(lbl, m) for lbl, m in all_rl_specs if lbl in set(args.methods)]

    ctx = mp.get_context('spawn')
    seed_cursor = TRAIN_SEED_OFFSET + int(args.seed_offset)
    all_train_trials: list[InsertionTaskSpec] = []
    rl_runs: dict[str, list[dict]] = {label: [] for label, _ in rl_specs}

    baseline_names = (
        'BlindSpiral_Stiff_1.0',
        'BlindSpiral_GravComp_0.35',
        'Privileged_Oracle',
    )

    print(
        f'Launching Phase 4 Clean-Kinematics Residual GRU({GRU_HIDDEN_DIM}) + Spiral Gate + GravComp-TL '
        f'({args.workers} headless CPU workers, {len(rl_specs)} methods x {len(args.seeds)} seeds x ~{args.target_steps:,} steps)...',
        flush=True,
    )
    with ctx.Pool(args.workers, initializer=_init_worker_single_thread) as pool:
        baselines = {}
        if not args.skip_baselines:
            print('\n=== 1. Evaluating Reference Base & Oracle Controllers Across All 4 Suites ===', flush=True)
            for cname in baseline_names:
                baselines[cname] = {}
                parts = []
                for sk, trials in eval_suites.items():
                    rows = pool.map(_eval_p3_baseline_worker, [(cname, t) for t in trials])
                    summ = summarize_suite_p4(rows)
                    baselines[cname][sk] = {'summary': summ, 'episodes': rows}
                    parts.append(
                        f'{sk[:8]}={summ["successes"]:2d}/30({100*summ["success_rate"]:4.1f}%,'
                        f'F={summ["median_peak_contact_force_N"]:4.1f}N,shv={summ["shove_failures"]:2d})'
                    )
                print(f'  {cname:27s} | ' + ' | '.join(parts), flush=True)

        for label, mode in rl_specs:
            print(f'\n=== 2. Training {label} (mode={mode}, {len(args.seeds)} seeds x {args.target_steps:,} steps) ===', flush=True)
            for seed in args.seeds:
                pol, meta, seed_cursor, used_trials = train_one_seed_p4(
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
                    jobs = [{'task': t, 'mode': mode, 'actor_weights': aw} for t in trials]
                    rows = pool.map(_eval_rl_worker_p4, jobs)
                    summ = summarize_suite_p4(rows)
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
            'eval': {sk: aggregate_seeds_p4(runs, sk) for sk in eval_suites},
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
