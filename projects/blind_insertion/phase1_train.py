"""Phase 1 Headless Asymmetric PPO Training & 3-Split Evaluation Harness.

Head-to-head comparison of 3 demo-free RL methods (identical 1D-CNN backbone, asymmetric
privileged critic, dual auxiliary decoder, reward hygiene, compute budget, and seeds):
  - M1_PosOnly:       SOTA Blind Precise Assembly (arXiv:2610.23849) replica (zero force, zero tracking_error, ts=1.0)
  - M2_Current:       Adds actuator_force_noisy + tracking_error (fixed ts=1.0)
  - M3_Current_TL:    Adds actuator_force_noisy + tracking_error + active torque_scale in [0.20, 1.0]

Evaluated alongside Phase 0 reference controllers (Nominal, Spiral Stiff, Spiral Low-Torque, Oracle)
across Split A (`bolted_socket`, n=30), Split B (`unbolted_socket`, n=30), and Split C (`tight_0.5mm`, n=30).
Training uses strictly SEEN shapes (`round`, `square`, `rectangle`) and disjoint seeds >= 10000.
"""
from __future__ import annotations

from collections import Counter, deque
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
import mujoco
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
    build_tight_clearance_suite,
)
from projects.blind_insertion.controllers import (
    BlindSpiralController,
    NominalStraightDownController,
    PrivilegedOracleController,
    run_episode,
)
from projects.blind_insertion.env import (
    PEG_MOUNT_BASE_GRIP_FRAME,
    BlindInsertionEnv,
    InsertionTaskSpec,
    STS3215Config,
)
from projects.blind_insertion.geometry import ALL_SHAPES, SEEN_SHAPES, UNSEEN_SHAPES, hole_outer_radius_m
from projects.blind_insertion.phase1_policy import (
    CONTACT_CLASS_TO_IDX,
    PRIVILEGED_CRITIC_DIM,
    TemporalAsymmetricPPO,
    TemporalProprioBuilder,
    build_privileged_critic_features,
)

# Uniform grid over [0.5, 1.5] mm so worker MuJoCo models stay cached in RAM (zero XML recompile!).
TRAIN_CLEARANCES_M: tuple[float, ...] = (0.0005, 0.00075, 0.0010, 0.00125, 0.0015)
HOLE_OUTER_RADIUS_BY_SHAPE: dict[str, float] = {s: hole_outer_radius_m(s) for s in ALL_SHAPES}
OFF_PLATE_SOFT_RADIUS_M = 0.022   # soft boundary penalty approaching plate edge
OFF_PLATE_HARD_RADIUS_M = 0.028   # off-plate guard & training termination (> 28 mm)
CURRICULUM_STAGES_M: tuple[float, ...] = (0.006, 0.010, 0.014, 0.016)
CURRICULUM_PROMOTE_RATE = 0.75    # promote only when rolling 100-ep success >= 75%

_WORKER_ENV_CACHE: dict[tuple[str, float, bool], BlindInsertionEnv] = {}


def reset_cached_env_inplace(env: BlindInsertionEnv, task: InsertionTaskSpec) -> tuple[dict, dict]:
    """Update continuous physics parameters in-place on a cached geometry-matched env, then reset."""
    assert env.task.shape == task.shape and abs(env.task.clearance_m - task.clearance_m) < 1e-9 and env.task.bolted == task.bolted
    m = env.model
    z_plate = env._geom_spec.bottom_thickness_m + env._geom_spec.plate_thickness_m / 2.0
    m.body_pos[env.socket_body_id] = np.array([task.socket_xy[0], task.socket_xy[1], z_plate], dtype=float)
    m.body_pos[env.peg_body_id] = PEG_MOUNT_BASE_GRIP_FRAME + np.array([task.grasp_offset_xy[0], task.grasp_offset_xy[1], 0.0], dtype=float)
    for gid in env._peg_geoms:
        m.geom_friction[gid, 0] = task.mu_peg_hole
    for gid in env._socket_geoms:
        m.geom_friction[gid, 0] = task.mu_peg_hole
    table_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'table')
    m.geom_friction[table_gid, 0] = task.mu_table
    if not task.bolted:
        cur_mass = float(m.body_mass[env.socket_body_id])
        scale = float(task.socket_mass_kg) / max(cur_mass, 1e-9)
        m.body_mass[env.socket_body_id] *= scale
        m.body_inertia[env.socket_body_id] *= scale
    mujoco.mj_setConst(m, env.data)
    env.task = task
    env.servo_cfg = STS3215Config(backlash_rad=task.backlash_rad)
    return env.reset()


def _get_cached_env(task: InsertionTaskSpec) -> BlindInsertionEnv:
    key = (task.shape, round(task.clearance_m, 6), bool(task.bolted))
    env = _WORKER_ENV_CACHE.get(key)
    if env is None:
        env = BlindInsertionEnv(task=task, render=False)
        _WORKER_ENV_CACHE[key] = env
    return env


def verify_train_disjointness(train_trials: list[InsertionTaskSpec]) -> None:
    """Assert strict zero overlap with Splits A/B/C and strictly SEEN shapes only in training."""
    heldout_seeds = (
        {t.seed for t in build_regime_a_bolted()}
        | {t.seed for t in build_regime_b_unbolted()}
        | {t.seed for t in build_tight_clearance_suite()}
        | {t.seed for t in build_split_c_tight_0_5mm()}
    )
    tr_seeds = {t.seed for t in train_trials}
    assert all(s >= TRAIN_SEED_OFFSET for s in tr_seeds), 'Found training seed < TRAIN_SEED_OFFSET'
    assert tr_seeds.isdisjoint(heldout_seeds), f'Training seeds overlap held-out: {tr_seeds & heldout_seeds}'
    tr_shapes = {t.shape for t in train_trials}
    assert tr_shapes.issubset(set(SEEN_SHAPES)), f'Unseen shapes leaked into training: {tr_shapes - set(SEEN_SHAPES)}'


def _init_worker_single_thread() -> None:
    """Keep MuJoCo rollout workers single-threaded while parent uses multi-threaded BLAS for PPO."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def sample_training_batch(
    rng: np.random.Generator,
    batch_size: int,
    seed_cursor: int,
    progress_frac: float,
    max_steps: int = 180,
    r_max_override: float | None = None,
) -> tuple[list[InsertionTaskSpec], int]:
    """Sample strictly SEEN shapes, 50/50 bolted/unbolted, uniform clearances in [0.5, 1.5] mm.

    Performance-gated stepwise curriculum stages: 6 mm -> 10 mm -> 14 mm -> 16 mm
    (identical across M1, M2, M3), with 20% near-hole precision replay once r_max >= 10 mm.
    """
    if seed_cursor < TRAIN_SEED_OFFSET:
        raise ValueError(f'seed_cursor must be >= {TRAIN_SEED_OFFSET}')
    r_max = float(np.clip(r_max_override if r_max_override is not None else CURRICULUM_STAGES_M[0], 0.006, 0.016))
    r_min = max(0.0, r_max - 0.010)  # 6mm->[0,6], 10mm->[0,10], 14mm->[4,14], 16mm->[6,16]

    trials: list[InsertionTaskSpec] = []
    for i in range(batch_size):
        shape = SEEN_SHAPES[i % len(SEEN_SHAPES)]
        bolted = (i % 2 == 0)
        clearance = float(TRAIN_CLEARANCES_M[int(rng.integers(0, len(TRAIN_CLEARANCES_M)))])
        if r_max >= 0.010 and rng.random() < 0.20:
            r = float(rng.uniform(0.0010, 0.0060))
        else:
            r = float(rng.uniform(r_min, r_max))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        trials.append(InsertionTaskSpec(
            trial_id=f'train-{seed_cursor}',
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


def compute_step_reward(
    prev_priv: dict,
    cur_priv: dict,
    env_act: np.ndarray,
    prev_env_act: np.ndarray,
    info: dict,
    step_idx: int,
    max_steps: int,
    milestones: dict,
    visited_plate_cells: set[tuple[int, int]] | None = None,
    ee_rel_xy: np.ndarray | None = None,
) -> tuple[float, bool]:
    """Identical exploit-free reward function across M1, M2, and M3 (strictly from info['privileged']).

    Phase 1.5 fixes:
    1. Hole-gated depth shaping: positive +Delta(peg_depth) awarded ONLY when over/inside the hole
       (`radial_err < hole_outer_r + 1.5 mm` and `peg_depth > 0`).
    2. Off-plate guard & termination: if `radial_err > 28 mm` OR (`peg_depth > 2.0 mm` while
       `radial_err > hole_outer_r + 3.0 mm`), immediately terminate with `-2.5` penalty.
    3. Clean on-plate radial progress potential (`radial_err <= 28 mm`), seating bonus `+10.0`,
       soft peak-force quadratic penalty (`> 35 N`), and unbolted shove penalty (`> 10 mm` soft,
       `> 25 mm` failure `-2.5`).

    Returns:
      (step_reward, off_plate_terminated)
    """
    prev_xy = float(np.linalg.norm(prev_priv['socket_xy_error']))
    cur_xy = float(np.linalg.norm(cur_priv['socket_xy_error']))
    hole_r = HOLE_OUTER_RADIUS_BY_SHAPE[cur_priv['shape']]

    prev_depth_signed = float(prev_priv['peg_depth'])
    cur_depth_signed = float(cur_priv['peg_depth'])

    # Off-plate guard & immediate termination condition
    dropped_off_plate = (cur_depth_signed > 0.0020) and (cur_xy > hole_r + 0.0030)
    off_plate_terminated = bool((cur_xy > OFF_PLATE_HARD_RADIUS_M) or dropped_off_plate)

    # 1. Hole-gated positive depth shaping (ONLY when radial_err < hole_outer_r + 1.5 mm and peg_depth > 0)
    prev_in_hole = max(0.0, prev_depth_signed) if (prev_xy < hole_r + 0.0015) else 0.0
    cur_in_hole = max(0.0, cur_depth_signed) if (cur_xy < hole_r + 0.0015) else 0.0
    # Pre-contact descent capped strictly at plate top (0.0) so dropping off outer edge pays 0
    prev_plate_z = min(max(prev_depth_signed, -0.008), 0.0)
    cur_plate_z = min(max(cur_depth_signed, -0.008), 0.0)

    r = 8.0 * (cur_plate_z - prev_plate_z)
    r += 45.0 * (cur_in_hole - prev_in_hole)

    # 2. Clean bounded on-plate radial progress potential (ONLY while on plate <= 28 mm)
    if (prev_xy <= OFF_PLATE_HARD_RADIUS_M) and (cur_xy <= OFF_PLATE_HARD_RADIUS_M):
        r += 14.0 * (prev_xy - cur_xy)

    # Anti-floating penalty when peg hovers > 4 mm above top plate
    if cur_depth_signed < -0.004:
        r -= 0.025 * min((-cur_depth_signed - 0.004) / 0.010, 2.0)

    # Soft boundary penalty approaching plate edge (22..28 mm)
    if cur_xy > OFF_PLATE_SOFT_RADIUS_M:
        r -= 0.04 * min((cur_xy - OFF_PLATE_SOFT_RADIUS_M) / 0.006, 2.0)

    # Small one-time on-plate novel-cell coverage bonus (strictly on plate, capped at 16 cells = +0.48 max)
    if (
        visited_plate_cells is not None
        and ee_rel_xy is not None
        and (not off_plate_terminated)
        and (-0.003 <= cur_depth_signed <= 0.002)
        and (cur_xy <= OFF_PLATE_SOFT_RADIUS_M)
        and len(visited_plate_cells) < 16
    ):
        cell = (int(math.floor(float(ee_rel_xy[0]) / 0.006)), int(math.floor(float(ee_rel_xy[1]) / 0.006)))
        if cell not in visited_plate_cells and float(np.linalg.norm(ee_rel_xy)) <= 0.020:
            visited_plate_cells.add(cell)
            r += 0.03

    c_cls = cur_priv['contact_class']
    if (not milestones['rim']) and c_cls in ('rim_contact', 'entering', 'seated'):
        milestones['rim'] = True
        r += 0.30
    if (not milestones['entering']) and c_cls in ('entering', 'seated'):
        milestones['entering'] = True
        r += 0.90

    # 3. Unbolted socket displacement hygiene (> 10 mm soft penalty, > 25 mm failure -2.5)
    if not cur_priv['bolted']:
        disp = float(cur_priv['socket_displacement_mm'])
        prev_disp = float(prev_priv['socket_displacement_mm'])
        if disp > 10.0:
            r -= 0.03 * ((disp - 10.0) / 15.0)
            r -= 0.06 * max(0.0, disp - prev_disp)
        if float(cur_priv['socket_tilt_deg']) > 3.0:
            r -= 0.025 * (float(cur_priv['socket_tilt_deg']) - 3.0)

    # 4. Excessive peak contact force soft quadratic penalty above 35 N
    excess_f = max(0.0, float(cur_priv['peak_contact_force_N']) - 35.0)
    if excess_f > 0.0:
        r -= 0.02 * ((excess_f / 35.0) ** 2)

    # 5. Smoothness penalty on 3D motion jerk + step time cost (identical across M1, M2, M3)
    d_act = (env_act[:3] - prev_env_act[:3]) / np.array([0.004, 0.004, 0.003])
    r -= 0.003 * float(np.sum(d_act ** 2))
    r -= 0.005

    if info['success']:
        r += 10.0 + 1.0 * (1.0 - step_idx / max(max_steps, 1))
    elif off_plate_terminated:
        r -= 2.5
    elif info['failure_mode'] in ('socket_shoved', 'socket_tipped'):
        r -= 2.5
    elif info['failure_mode'] == 'safety_violation':
        r -= 3.0
    return float(r), off_plate_terminated


def _train_rollout_worker(job: dict) -> dict:
    task: InsertionTaskSpec = job['task']
    mode: str = job['mode']
    use_tl = (mode == 'current_torque')

    env = _get_cached_env(task)
    obs, info = reset_cached_env_inplace(env, task)
    start_xyz = np.asarray(obs['ee_pose_fk'][:3], dtype=float).copy()

    builder = TemporalProprioBuilder(mode=mode, max_steps=task.max_steps)
    pol = TemporalAsymmetricPPO(
        step_dim=builder.step_dim,
        spatial_dim=builder.spatial_dim,
        critic_dim=PRIVILEGED_CRITIC_DIM,
        use_torque_limit=use_tl,
    )
    pol.set_weights(job['actor_weights'])

    rng = np.random.default_rng(job['rollout_seed'])
    ar_rho = float(job.get('ar_rho', 0.85))
    innov = math.sqrt(max(1.0 - ar_rho * ar_rho, 0.0))
    eps = rng.normal(0.0, 1.0, size=pol.act_dim)

    hist_buf, spat_buf, crit_buf, z_buf, rew_buf, cls_buf, dir_buf, dmask_buf = [], [], [], [], [], [], [], []
    prev_priv = info['privileged']
    prev_env_act = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    milestones = {'rim': False, 'entering': False}
    visited_plate_cells: set[tuple[int, int]] = set()
    terminated = False
    truncated = False
    off_plate_term = False

    while not (terminated or truncated):
        h_20, s_ctx = builder.step(obs)
        c_vec = build_privileged_critic_features(obs, info['privileged'], start_xyz, max_steps=task.max_steps)
        cls_idx = CONTACT_CLASS_TO_IDX[info['privileged']['contact_class']]
        xy_e = np.asarray(info['privileged']['socket_xy_error'], dtype=float)
        xy_n = float(np.linalg.norm(xy_e))
        if xy_n > 5e-4:
            dir_t = -xy_e / xy_n  # unit vector pointing TOWARD socket center
            d_m = 1.0
        else:
            dir_t = np.zeros(2, dtype=float)
            d_m = 0.0

        eps = ar_rho * eps + innov * rng.normal(0.0, 1.0, size=pol.act_dim)
        env_act, z, _, _ = pol.act_step(h_20, s_ctx, deterministic=False, eps=eps)

        obs, _, terminated, truncated, info = env.step(env_act)
        ee_rel_xy = np.asarray(obs['ee_pose_fk'][:2], dtype=float) - start_xyz[:2]
        r, off_plate_term = compute_step_reward(
            prev_priv, info['privileged'], env_act, prev_env_act, info,
            env.step_count, task.max_steps, milestones,
            visited_plate_cells=visited_plate_cells, ee_rel_xy=ee_rel_xy,
        )
        if off_plate_term and not info['success']:
            terminated = True
        hist_buf.append(h_20)
        spat_buf.append(s_ctx)
        crit_buf.append(c_vec)
        z_buf.append(z)
        rew_buf.append(r)
        cls_buf.append(cls_idx)
        dir_buf.append(dir_t)
        dmask_buf.append(d_m)

        prev_priv = info['privileged']
        prev_env_act = env_act

    last_c = build_privileged_critic_features(obs, info['privileged'], start_xyz, max_steps=task.max_steps)
    return {
        'hist': np.asarray(hist_buf, dtype=float),
        'spatial': np.asarray(spat_buf, dtype=float),
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


def _eval_rl_worker(job: dict) -> dict:
    """Deterministic held-out evaluation on a fresh unmodified BlindInsertionEnv."""
    task: InsertionTaskSpec = job['task']
    mode: str = job['mode']
    use_tl = (mode == 'current_torque')

    env = BlindInsertionEnv(task=task, render=False)
    try:
        obs, info = env.reset()
        builder = TemporalProprioBuilder(mode=mode, max_steps=task.max_steps)
        pol = TemporalAsymmetricPPO(
            step_dim=builder.step_dim,
            spatial_dim=builder.spatial_dim,
            critic_dim=PRIVILEGED_CRITIC_DIM,
            use_torque_limit=use_tl,
        )
        pol.set_weights(job['actor_weights'])

        terminated = False
        truncated = False
        max_depth_mm = float(info['privileged']['peg_depth'] * 1000.0)
        hole_r = HOLE_OUTER_RADIUS_BY_SHAPE[task.shape]
        went_off_plate = False
        aux_correct = 0
        aux_total = 0
        dir_cos_list: list[float] = []
        ts_by_phase: dict[str, list[float]] = {'search_rim': [], 'insert_seat': [], 'free': []}

        while not (terminated or truncated):
            h_20, s_ctx = builder.step(obs)
            true_cls_name = info['privileged']['contact_class']
            true_cls = CONTACT_CLASS_TO_IDX[true_cls_name]
            env_act, _, pred_cls, pred_dir = pol.act_step(h_20, s_ctx, deterministic=True)

            aux_correct += int(pred_cls == true_cls)
            aux_total += 1
            xy_e = np.asarray(info['privileged']['socket_xy_error'], dtype=float)
            xy_n = float(np.linalg.norm(xy_e))
            pn = float(np.linalg.norm(pred_dir))
            if xy_n > 5e-4 and pn > 1e-6:
                dir_cos_list.append(float(np.dot(pred_dir / pn, -xy_e / xy_n)))

            ts_mean = float(np.mean(env_act[3:8]))
            if true_cls_name in ('plate_contact', 'rim_contact'):
                ts_by_phase['search_rim'].append(ts_mean)
            elif true_cls_name in ('entering', 'seated'):
                ts_by_phase['insert_seat'].append(ts_mean)
            else:
                ts_by_phase['free'].append(ts_mean)

            obs, _, terminated, truncated, info = env.step(env_act)
            cur_p = info['privileged']
            d_mm = float(cur_p['peg_depth'] * 1000.0)
            if d_mm > max_depth_mm:
                max_depth_mm = d_mm
            cur_xy = float(np.linalg.norm(cur_p['socket_xy_error']))
            if cur_xy > OFF_PLATE_HARD_RADIUS_M or (float(cur_p['peg_depth']) > 0.0020 and cur_xy > hole_r + 0.0030):
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
        }
    finally:
        env.close()


def _eval_baseline_worker(args: tuple[str, InsertionTaskSpec]) -> dict:
    cname, task = args
    env = BlindInsertionEnv(task=task, render=False)
    try:
        if cname == 'Nominal_StraightDown':
            ctrl = NominalStraightDownController(torque_scale=1.0)
        elif cname == 'BlindSpiral_Stiff_1.0':
            ctrl = BlindSpiralController(torque_scale=1.0)
        elif cname == 'BlindSpiral_LowTorque_0.35':
            ctrl = BlindSpiralController(torque_scale=0.35)
        elif cname == 'Privileged_Oracle':
            ctrl = PrivilegedOracleController()
        else:
            raise ValueError(cname)
        return run_episode(env, task, ctrl)
    finally:
        env.close()


def summarize_suite(rows: list[dict]) -> dict:
    n = len(rows)
    succ = [r for r in rows if r['success']]
    seen = [r for r in rows if r['split'] == 'seen']
    unseen = [r for r in rows if r['split'] == 'unseen']
    unbolted = [r for r in rows if not r['bolted']]
    bolted = [r for r in rows if r['bolted']]
    modes = dict(Counter(r['failure_mode'] for r in rows))
    by_shape = {}
    for s in sorted(set(r['shape'] for r in rows)):
        sub = [r for r in rows if r['shape'] == s]
        by_shape[s] = {
            'successes': sum(r['success'] for r in sub),
            'trials': len(sub),
            'success_rate': sum(r['success'] for r in sub) / len(sub),
        }
    out = {
        'successes': len(succ),
        'trials': n,
        'success_rate': len(succ) / max(n, 1),
        'seen_successes': sum(r['success'] for r in seen),
        'seen_trials': len(seen),
        'seen_rate': sum(r['success'] for r in seen) / max(len(seen), 1),
        'unseen_successes': sum(r['success'] for r in unseen),
        'unseen_trials': len(unseen),
        'unseen_rate': sum(r['success'] for r in unseen) / max(len(unseen), 1),
        'bolted_successes': sum(r['success'] for r in bolted),
        'bolted_trials': len(bolted),
        'unbolted_successes': sum(r['success'] for r in unbolted),
        'unbolted_trials': len(unbolted),
        'median_time_to_insert_s': float(np.median([r['time_s'] for r in succ])) if succ else None,
        'median_peak_contact_force_N': float(np.median([r['peak_contact_force_N'] for r in rows])),
        'shove_failures': modes.get('socket_shoved', 0),
        'tip_failures': modes.get('socket_tipped', 0),
        'median_socket_disp_mm': float(np.median([r['max_socket_disp_mm'] for r in unbolted])) if unbolted else 0.0,
        'self_collision_steps': sum(r['self_collision_steps'] for r in rows),
        'joint_limit_steps': sum(r['joint_limit_steps'] for r in rows),
        'failure_modes': modes,
        'by_shape': by_shape,
    }
    if 'aux_contact_acc' in rows[0]:
        out['mean_aux_contact_acc'] = float(np.mean([r['aux_contact_acc'] for r in rows]))
        out['mean_aux_dir_cosine'] = float(np.mean([r['aux_dir_cosine'] for r in rows]))
        out['off_plate_rate'] = float(np.mean([r.get('went_off_plate', False) for r in rows]))
        ts_sr = [r['ts_search_rim'] for r in rows if r['ts_search_rim'] is not None]
        ts_is = [r['ts_insert_seat'] for r in rows if r['ts_insert_seat'] is not None]
        ts_fr = [r['ts_free'] for r in rows if r['ts_free'] is not None]
        out['mean_ts_search_rim'] = float(np.mean(ts_sr)) if ts_sr else None
        out['mean_ts_insert_seat'] = float(np.mean(ts_is)) if ts_is else None
        out['mean_ts_free'] = float(np.mean(ts_fr)) if ts_fr else None
    return out


def aggregate_seeds(runs: list[dict], suite_key: str) -> dict:
    sums = [r['eval'][suite_key]['summary'] for r in runs]
    rates = [s['success_rate'] for s in sums]
    med_ts = [s['median_time_to_insert_s'] for s in sums if s['median_time_to_insert_s'] is not None]
    out = {
        'mean_success_rate': float(np.mean(rates)),
        'std_success_rate': float(np.std(rates)),
        'per_seed_successes': [s['successes'] for s in sums],
        'trials': sums[0]['trials'],
        'mean_seen_rate': float(np.mean([s['seen_rate'] for s in sums])),
        'std_seen_rate': float(np.std([s['seen_rate'] for s in sums])),
        'mean_unseen_rate': float(np.mean([s['unseen_rate'] for s in sums])),
        'std_unseen_rate': float(np.std([s['unseen_rate'] for s in sums])),
        'per_seed_seen': [s['seen_successes'] for s in sums],
        'per_seed_unseen': [s['unseen_successes'] for s in sums],
        'mean_bolted_succ': float(np.mean([s['bolted_successes'] for s in sums])),
        'mean_unbolted_succ': float(np.mean([s['unbolted_successes'] for s in sums])),
        'mean_shove_failures': float(np.mean([s['shove_failures'] for s in sums])),
        'mean_tip_failures': float(np.mean([s['tip_failures'] for s in sums])),
        'mean_median_disp_mm': float(np.mean([s['median_socket_disp_mm'] for s in sums])),
        'mean_median_force_N': float(np.mean([s['median_peak_contact_force_N'] for s in sums])),
        'mean_median_insert_time_s': float(np.mean(med_ts)) if med_ts else None,
        'total_self_collision_steps': sum(s['self_collision_steps'] for s in sums),
        'total_joint_limit_steps': sum(s['joint_limit_steps'] for s in sums),
    }
    if 'mean_aux_contact_acc' in sums[0]:
        out['mean_aux_contact_acc'] = float(np.mean([s['mean_aux_contact_acc'] for s in sums]))
        out['mean_aux_dir_cosine'] = float(np.mean([s['mean_aux_dir_cosine'] for s in sums]))
        out['mean_off_plate_rate'] = float(np.mean([s.get('off_plate_rate', 0.0) for s in sums]))
        sr = [s['mean_ts_search_rim'] for s in sums if s['mean_ts_search_rim'] is not None]
        ins = [s['mean_ts_insert_seat'] for s in sums if s['mean_ts_insert_seat'] is not None]
        fr = [s['mean_ts_free'] for s in sums if s['mean_ts_free'] is not None]
        out['mean_ts_search_rim'] = float(np.mean(sr)) if sr else None
        out['mean_ts_insert_seat'] = float(np.mean(ins)) if ins else None
        out['mean_ts_free'] = float(np.mean(fr)) if fr else None
    return out


def train_one_seed(
    pool,
    label: str,
    mode: str,
    seed: int,
    seed_cursor: int,
    target_steps: int = 500_000,
    batch_size: int = 20,
    max_steps: int = 180,
) -> tuple[TemporalAsymmetricPPO, dict, int, list[InsertionTaskSpec]]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    builder = TemporalProprioBuilder(mode=mode, max_steps=max_steps)
    pol = TemporalAsymmetricPPO(
        step_dim=builder.step_dim,
        spatial_dim=builder.spatial_dim,
        critic_dim=PRIVILEGED_CRITIC_DIM,
        use_torque_limit=(mode == 'current_torque'),
        seed=seed,
        init_log_std=-0.60,
    )
    history = []
    all_trials: list[InsertionTaskSpec] = []
    cum_steps = 0
    upd = 0
    next_log_steps = 0
    stage_idx = 0  # stepwise performance-gated curriculum: 6 mm -> 10 mm -> 14 mm -> 16 mm
    rolling_succ: deque[int] = deque(maxlen=100)
    stage_transitions: list[dict] = []

    while cum_steps < target_steps:
        progress = min(1.0, cum_steps / max(target_steps, 1))
        eff_r_max = CURRICULUM_STAGES_M[stage_idx]
        batch_trials, seed_cursor = sample_training_batch(
            rng, batch_size=batch_size, seed_cursor=seed_cursor, progress_frac=progress,
            max_steps=max_steps, r_max_override=eff_r_max,
        )
        all_trials.extend(batch_trials)

        aw = pol.get_actor_weights()
        rseeds = rng.integers(1, 2**31 - 1, size=batch_size)
        jobs = [
            {'task': t, 'mode': mode, 'actor_weights': aw, 'rollout_seed': int(rs), 'ar_rho': 0.88}
            for t, rs in zip(batch_trials, rseeds)
        ]
        trajs = pool.map(_train_rollout_worker, jobs)

        if upd == 0:
            pol.step_rms.update(np.concatenate([tr['hist'][:, -1, :] for tr in trajs], axis=0))
            pol.spatial_rms.update(np.concatenate([tr['spatial'] for tr in trajs], axis=0))
            pol.critic_rms.update(np.concatenate([tr['critic_obs'] for tr in trajs], axis=0))

        lr_scale = 0.20 + 0.80 * (1.0 - progress)
        stats = pol.ppo_update(
            trajs,
            rng=rng,
            actor_lr=8e-4 * lr_scale,
            critic_lr=1.8e-3 * lr_scale,
            gamma=0.995,
            lam=0.95,
            clip_eps=0.2,
            epochs=4,
            minibatch_size=256,
            entropy_coef=0.001,
            aux_cls_coef=0.15,
            aux_dir_coef=0.15,
            max_grad_norm=0.5,
            target_kl=0.05,
            weight_decay=1e-4,
        )
        upd += 1
        cum_steps += stats['n_steps']
        for tr in trajs:
            rolling_succ.append(int(tr['success']))
        roll_100 = float(np.mean(rolling_succ)) if rolling_succ else 0.0
        succ_all = sum(tr['success'] for tr in trajs) / len(trajs)

        # Promote curriculum stage ONLY when rolling 100-episode success rate >= 75%!
        if (
            stage_idx + 1 < len(CURRICULUM_STAGES_M)
            and len(rolling_succ) == 100
            and roll_100 >= CURRICULUM_PROMOTE_RATE
        ):
            stage_idx += 1
            rolling_succ.clear()
            stage_transitions.append({
                'update': upd,
                'steps_cum': cum_steps,
                'new_r_max_mm': round(CURRICULUM_STAGES_M[stage_idx] * 1000.0, 1),
                'trigger_roll100': round(roll_100, 3),
            })

        b_tr = [tr for tr in trajs if tr['bolted']]
        u_tr = [tr for tr in trajs if not tr['bolted']]
        succ_b = sum(tr['success'] for tr in b_tr) / max(len(b_tr), 1)
        succ_u = sum(tr['success'] for tr in u_tr) / max(len(u_tr), 1)
        shoves = sum(tr['failure_mode'] == 'socket_shoved' for tr in u_tr)
        off_plates = sum(tr['failure_mode'] == 'off_plate' for tr in trajs)

        entry = {
            'update': upd,
            'episodes_cum': upd * batch_size,
            'steps_cum': cum_steps,
            'r_max_mm': round(eff_r_max * 1000.0, 1),
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
                f"  [{label:16s} s={seed}] upd {upd:3d} ({cum_steps:6d}/{target_steps} st, r<={CURRICULUM_STAGES_M[stage_idx]*1000:4.1f}mm, {cum_steps/max(elapsed,1e-6):5.0f} st/s) | "
                f"succ={100*succ_all:5.1f}% (r100={100*roll_100:4.1f}%, B={100*succ_b:4.0f}% U={100*succ_u:4.0f}% shv={shoves:2d} off={off_plates:2d}) | "
                f"ret={stats['mean_return']:+.2f} ev={stats['explained_var']:+.2f} aux={100*stats['aux_cls_acc']:4.1f}% "
                f"kl={stats['approx_kl']:.4f} std={np.round(stats['std'][:4], 2)}",
                flush=True,
            )
            next_log_steps = cum_steps + max(target_steps // 4, 1)

    wall_s = time.perf_counter() - t0
    total_off_plate_train = sum(e['train_off_plate'] for e in history)
    meta = {
        'label': label,
        'mode': mode,
        'seed': seed,
        'step_dim': builder.step_dim,
        'actor_param_count': pol.actor_param_count,
        'critic_param_count': pol.critic_param_count,
        'n_updates': upd,
        'batch_size': batch_size,
        'episodes_budget': upd * batch_size,
        'total_steps': cum_steps,
        'target_steps': target_steps,
        'final_r_max_mm': round(CURRICULUM_STAGES_M[stage_idx] * 1000.0, 1),
        'stage_transitions': stage_transitions,
        'train_off_plate_rate_all': total_off_plate_train / max(upd * batch_size, 1),
        'train_off_plate_rate_last10': float(np.mean([e['train_off_plate'] / batch_size for e in history[-10:]])),
        'wall_s': round(wall_s, 2),
        'steps_per_sec': round(cum_steps / max(wall_s, 1e-6), 1),
        'episodes_per_sec': round((upd * batch_size) / max(wall_s, 1e-6), 2),
        'history': history,
    }
    return pol, meta, seed_cursor, all_trials


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--target-steps', type=int, default=400_000)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=180)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--out-prefix', type=str, default='phase1_5')
    parser.add_argument('--seed-offset', type=int, default=50000)
    args = parser.parse_args()

    t_total = time.perf_counter()
    ckpt_dir = THIS_DIR / f'{args.out_prefix}_checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    eval_suites = {
        'split_a_bolted': build_regime_a_bolted(),
        'split_b_unbolted': build_regime_b_unbolted(),
        'split_c_tight_0.5mm': build_split_c_tight_0_5mm(),
    }

    rl_specs = [
        ('M1_PosOnly', 'pos_only'),
        ('M2_Current', 'current'),
        ('M3_Current_TL', 'current_torque'),
    ]

    ctx = mp.get_context('spawn')
    seed_cursor = TRAIN_SEED_OFFSET + int(args.seed_offset)
    all_train_trials: list[InsertionTaskSpec] = []
    rl_runs: dict[str, list[dict]] = {label: [] for label, _ in rl_specs}

    print(
        f'Launching Phase 1 blind insertion RL ({args.workers} headless CPU workers, '
        f'{len(args.seeds)} seeds x ~{args.target_steps:,} policy steps/run)...',
        flush=True,
    )
    with ctx.Pool(args.workers, initializer=_init_worker_single_thread) as pool:
        # 1. Exact reference baselines across Splits A, B, and C
        print('\n=== 1. Evaluating Reference Controllers on Splits A (bolted), B (unbolted), and C (tight_0.5mm) ===', flush=True)
        baselines = {}
        for cname in ('Nominal_StraightDown', 'BlindSpiral_Stiff_1.0', 'BlindSpiral_LowTorque_0.35', 'Privileged_Oracle'):
            baselines[cname] = {}
            parts = []
            for sk, trials in eval_suites.items():
                rows = pool.map(_eval_baseline_worker, [(cname, t) for t in trials])
                summ = summarize_suite(rows)
                baselines[cname][sk] = {'summary': summ, 'episodes': rows}
                parts.append(
                    f'{sk[:7]}={summ["successes"]:2d}/30({100*summ["success_rate"]:4.1f}%,'
                    f's={summ["seen_successes"]}/15,u={summ["unseen_successes"]}/15,shv={summ["shove_failures"]})'
                )
            print(f'  {cname:28s} | ' + ' | '.join(parts), flush=True)

        # 2. Train & evaluate M1, M2, M3 across identical seeds and step budget
        for label, mode in rl_specs:
            print(f'\n=== 2. Training {label} (mode={mode}, {len(args.seeds)} seeds x {args.target_steps:,} steps) ===', flush=True)
            for seed in args.seeds:
                pol, meta, seed_cursor, used_trials = train_one_seed(
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
                    rows = pool.map(_eval_rl_worker, jobs)
                    summ = summarize_suite(rows)
                    eval_out[sk] = {'summary': summ, 'episodes': rows}
                    parts.append(
                        f'{sk[:7]}={summ["successes"]:2d}/30({100*summ["success_rate"]:4.1f}%,'
                        f's={summ["seen_successes"]}/15,u={summ["unseen_successes"]}/15,shv={summ["shove_failures"]})'
                    )
                print(f'  EVAL [{label:16s} s={seed}] ({meta["wall_s"]:.0f}s) | ' + ' | '.join(parts), flush=True)
                rl_runs[label].append({'meta': meta, 'eval': eval_out})

    verify_train_disjointness(all_train_trials)
    print(f'\nVerified strict seed & shape disjointness across {len(all_train_trials)} training episodes (SEEN shapes only)!', flush=True)

    rl_aggregate = {}
    for label, _ in rl_specs:
        runs = rl_runs[label]
        rl_aggregate[label] = {
            'mode': runs[0]['meta']['mode'],
            'step_dim': runs[0]['meta']['step_dim'],
            'actor_param_count': runs[0]['meta']['actor_param_count'],
            'critic_param_count': runs[0]['meta']['critic_param_count'],
            'mean_episodes_per_seed': float(np.mean([r['meta']['episodes_budget'] for r in runs])),
            'mean_total_steps': float(np.mean([r['meta']['total_steps'] for r in runs])),
            'mean_train_wall_s': float(np.mean([r['meta']['wall_s'] for r in runs])),
            'mean_steps_per_sec': float(np.mean([r['meta']['steps_per_sec'] for r in runs])),
            'eval': {sk: aggregate_seeds(runs, sk) for sk in eval_suites},
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
