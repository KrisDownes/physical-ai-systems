"""Phase 2 Headless Recurrent GRU Asymmetric PPO Training & 3-Split + Slippery-OOD Evaluation Harness.

Eliminates the two Phase 1.5 bottlenecks while keeping geometry.py, env.py, and Splits A/B/C
in benchmarks.py completely untouched:
1. Short-Memory Fix: Episode-long GRU(80) carried across the full 5.0 s (250-step) episode +
   controller-legal spatial anchor (`ee_rel_start = ee_pos_fk - ee_start_pos_fk`) + 8x8 (5 mm)
   visited-XY grid + local 8-neighbor unvisited compass + kinematic dip memory (identical across
   M1, M2, M3).
2. Smooth Curriculum & Full Horizon Fix:
   - Removes the brittle 75% rolling gate: linearly expands training misalignment radius `r_max`
     from 6.0 mm -> 16.5 mm over the first 35% of total training steps (keeping a 20% replay mix
     of <= 10 mm starts), so every seed spends the remaining 65% of training on full 6-16.5 mm.
   - Full 250-step (5.0 s) training horizon matching eval's 250-step horizon.
   - Compact 2-group active torque-limit action [ts_lat, ts_arm] in [0.20, 1.0] (init 0.75) for
     M3, cutting action dim from 8 -> 5 while separating lateral compliance from vertical gravity.
3. Slippery Unbolted Generalization Ablation:
   - Clones Split B's 30 unbolted tasks with `mu_table = 0.25` and `socket_mass_kg = 0.120`
     (leaving benchmarks.py untouched) and evaluates BlindSpiral_Stiff, BlindSpiral_LowTorque,
     M1, M2, and M3 head-to-head on OOD slippery sockets.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
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
from projects.blind_insertion.env import BlindInsertionEnv, InsertionTaskSpec
from projects.blind_insertion.geometry import SEEN_SHAPES, UNSEEN_SHAPES
from projects.blind_insertion.phase1_policy import CONTACT_CLASS_TO_IDX
from projects.blind_insertion.phase1_train import (
    HOLE_OUTER_RADIUS_BY_SHAPE,
    OFF_PLATE_HARD_RADIUS_M,
    OFF_PLATE_SOFT_RADIUS_M,
    TRAIN_CLEARANCES_M,
    _eval_baseline_worker,
    _get_cached_env,
    _init_worker_single_thread,
    aggregate_seeds,
    reset_cached_env_inplace,
    summarize_suite,
    verify_train_disjointness,
)
from projects.blind_insertion.phase2_policy import (
    BPTT_CHUNK_LEN,
    GRU_HIDDEN_DIM,
    PRIVILEGED_CRITIC_DIM_P2,
    Phase2ProprioBuilder,
    RecurrentAsymmetricPPO,
    build_privileged_critic_features_p2,
)

# Smooth linear curriculum: 6.0 mm -> 16.5 mm over first 35% of steps, full 6-16.5 mm for final 65%
CURRICULUM_R_START_M = 0.0060
CURRICULUM_R_END_M = 0.0165
CURRICULUM_RAMP_FRAC = 0.35
CURRICULUM_REPLAY_PROB = 0.20
CURRICULUM_REPLAY_MAX_M = 0.0100

# Actual socket top-plate half-width is 42 mm (plate_size = 84 mm).
# A blind searcher starting at r = 16.5 mm sweeping an 18.5 mm spiral legitimately reaches
# cur_xy up to 35 mm while still safely on the 42 mm plate! Keep the off-plate guard right
# near the true plate edge (38 mm Chebyshev / off-hole drop > 2 mm) with a strong -8.0 penalty
# so off-plate termination is strictly dominated by staying on the plate for all 250 steps.
P2_PLATE_SOFT_HALF_M = 0.033
P2_PLATE_HARD_HALF_M = 0.038
P2_SEARCH_SOFT_REL_R_M = 0.020
P2_OFF_PLATE_PENALTY = 8.0


def compute_smooth_r_max_m(progress: float) -> float:
    """Linearly expand r_max from 6.0 mm -> 16.5 mm over the first 35% of training steps."""
    alpha = float(np.clip(progress / CURRICULUM_RAMP_FRAC, 0.0, 1.0))
    return CURRICULUM_R_START_M + (CURRICULUM_R_END_M - CURRICULUM_R_START_M) * alpha


def build_slippery_unbolted_ood_suite() -> list[InsertionTaskSpec]:
    """Deterministic 30-trial OOD slippery unbolted suite (mu_table=0.25, mass=120 g).

    Clones Split B tasks in-memory without touching benchmarks.py so every geometry,
    shape split (15 seen / 15 unseen), clearance, and misalignment is 100% matched.
    """
    base = build_regime_b_unbolted()
    return [
        replace(
            t,
            trial_id=f'slippery-{t.trial_id}',
            mu_table=0.25,
            socket_mass_kg=0.120,
        )
        for t in base
    ]


def sample_training_batch_p2(
    rng: np.random.Generator,
    batch_size: int,
    seed_cursor: int,
    progress: float,
    max_steps: int = 250,
) -> tuple[list[InsertionTaskSpec], int, float]:
    """Sample strictly SEEN shapes, 50/50 bolted/unbolted, uniform clearances in [0.5, 1.5] mm.

    Linearly ramps r_max from 6.0 mm -> 16.5 mm over progress in [0.0, 0.35] with a 20%
    replay mix of <= 10 mm starts once r_max > 10 mm, spending 65% of training on full 6-16.5 mm.
    """
    if seed_cursor < TRAIN_SEED_OFFSET:
        raise ValueError(f'seed_cursor must be >= {TRAIN_SEED_OFFSET}')
    r_max = compute_smooth_r_max_m(progress)
    r_min = max(0.0, r_max - 0.0105)  # 6.0mm->[0,6.0], ..., 16.5mm->[6.0,16.5]

    trials: list[InsertionTaskSpec] = []
    for i in range(batch_size):
        shape = SEEN_SHAPES[i % len(SEEN_SHAPES)]
        bolted = (i % 2 == 0)
        clearance = float(TRAIN_CLEARANCES_M[int(rng.integers(0, len(TRAIN_CLEARANCES_M)))])
        if r_max > CURRICULUM_REPLAY_MAX_M and rng.random() < CURRICULUM_REPLAY_PROB:
            r = float(rng.uniform(0.0010, CURRICULUM_REPLAY_MAX_M))
        else:
            r = float(rng.uniform(r_min, r_max))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        trials.append(InsertionTaskSpec(
            trial_id=f'p2-train-{seed_cursor}',
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
    return trials, seed_cursor, r_max


def compute_step_reward_p2(
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
    """Identical exploit-free Phase 1.5/2 reward across M1, M2, and M3 (strictly training-only).

    Preserves 100% of Phase 1.5's exploit-free guards:
    - Hole-gated positive depth shaping (ONLY when radial_err < hole_outer_r + 1.5 mm and peg_depth > 0)
    - Off-plate guard & immediate termination (> 28 mm OR off-hole depth > 2 mm -> -2.5 penalty)
    - Bounded on-plate radial progress potential (<= 28 mm)
    - 5 mm on-plate novel-cell coverage bonus aligned to 8x8 search disk (capped at 36 cells = +0.72 max,
      strictly inside 22 mm on-plate zone so walking off plate remains strictly negative < -2.0!)
    - Seating bonus (+10.0 + speed bonus), soft force penalty (> 35 N), unbolted shove hygiene.
    """
    prev_xy_vec = np.asarray(prev_priv['socket_xy_error'], dtype=float)
    cur_xy_vec = np.asarray(cur_priv['socket_xy_error'], dtype=float)
    prev_xy = float(np.linalg.norm(prev_xy_vec))
    cur_xy = float(np.linalg.norm(cur_xy_vec))
    cur_linf = float(np.max(np.abs(cur_xy_vec)))
    hole_r = HOLE_OUTER_RADIUS_BY_SHAPE[cur_priv['shape']]

    prev_depth_signed = float(prev_priv['peg_depth'])
    cur_depth_signed = float(cur_priv['peg_depth'])

    dropped_off_plate = (cur_depth_signed > 0.0020) and (cur_xy > hole_r + 0.0030)
    off_plate_terminated = bool((cur_linf > P2_PLATE_HARD_HALF_M) or dropped_off_plate)

    # 1. Hole-gated positive depth shaping (ONLY when radial_err < hole_outer_r + 1.5 mm and peg_depth > 0)
    prev_in_hole = max(0.0, prev_depth_signed) if (prev_xy < hole_r + 0.0015) else 0.0
    cur_in_hole = max(0.0, cur_depth_signed) if (cur_xy < hole_r + 0.0015) else 0.0
    prev_plate_z = min(max(prev_depth_signed, -0.008), 0.0)
    cur_plate_z = min(max(cur_depth_signed, -0.008), 0.0)

    r = 8.0 * (cur_plate_z - prev_plate_z)
    r += 45.0 * (cur_in_hole - prev_in_hole)

    # 2. Clean bounded on-plate radial progress potential (ONLY while on plate <= 38 mm)
    if (float(np.max(np.abs(prev_xy_vec))) <= P2_PLATE_HARD_HALF_M) and (cur_linf <= P2_PLATE_HARD_HALF_M):
        r += 14.0 * (prev_xy - cur_xy)

    if cur_depth_signed < -0.004:
        r -= 0.02 * min((-cur_depth_signed - 0.004) / 0.010, 2.0)

    # Soft nudge back from actual plate edges (33..38 mm) and beyond 20 mm start-relative search disk
    if cur_linf > P2_PLATE_SOFT_HALF_M:
        r -= 0.03 * min((cur_linf - P2_PLATE_SOFT_HALF_M) / 0.005, 2.0)
    if ee_rel_xy is not None:
        rel_r = float(np.linalg.norm(ee_rel_xy))
        if rel_r > P2_SEARCH_SOFT_REL_R_M:
            r -= 0.025 * min((rel_r - P2_SEARCH_SOFT_REL_R_M) / 0.006, 2.0)

    # 5 mm on-plate novel-cell coverage bonus covering full 18.5 mm search disk (capped at 36 cells = +0.72)
    if (
        visited_plate_cells is not None
        and ee_rel_xy is not None
        and (not off_plate_terminated)
        and (-0.003 <= cur_depth_signed <= 0.002)
        and (cur_linf <= P2_PLATE_SOFT_HALF_M)
        and len(visited_plate_cells) < 36
    ):
        cell = (int(math.floor(float(ee_rel_xy[0]) / 0.005)), int(math.floor(float(ee_rel_xy[1]) / 0.005)))
        if cell not in visited_plate_cells and float(np.linalg.norm(ee_rel_xy)) <= 0.0185:
            visited_plate_cells.add(cell)
            r += 0.02

    c_cls = cur_priv['contact_class']
    if (not milestones['rim']) and c_cls in ('rim_contact', 'entering', 'seated'):
        milestones['rim'] = True
        r += 0.30
    if (not milestones['entering']) and c_cls in ('entering', 'seated'):
        milestones['entering'] = True
        r += 0.90

    if not cur_priv['bolted']:
        disp = float(cur_priv['socket_displacement_mm'])
        prev_disp = float(prev_priv['socket_displacement_mm'])
        if disp > 10.0:
            r -= 0.02 * ((disp - 10.0) / 15.0)
            r -= 0.05 * max(0.0, disp - prev_disp)
        if float(cur_priv['socket_tilt_deg']) > 3.0:
            r -= 0.02 * (float(cur_priv['socket_tilt_deg']) - 3.0)

    # Bounded soft force penalty (> 35 N) + gentle step costs so 250-step timeout stays ~-1.5..-2.5
    # (strictly better than -8.0 off-plate penalty, eliminating the suicide attractor!)
    excess_f = max(0.0, float(cur_priv['peak_contact_force_N']) - 35.0)
    if excess_f > 0.0:
        r -= 0.008 * min((excess_f / 35.0) ** 2, 3.0)

    d_act = (env_act[:3] - prev_env_act[:3]) / np.array([0.004, 0.004, 0.003])
    r -= 0.002 * min(float(np.sum(d_act ** 2)), 4.0)
    r -= 0.002

    if info['success']:
        r += 10.0 + 1.0 * (1.0 - step_idx / max(max_steps, 1))
    elif off_plate_terminated:
        r -= P2_OFF_PLATE_PENALTY
    elif info['failure_mode'] in ('socket_shoved', 'socket_tipped'):
        r -= 3.5
    elif info['failure_mode'] == 'safety_violation':
        r -= 5.0
    return float(r), off_plate_terminated


def summarize_suite_p2(rows: list[dict]) -> dict:
    """Extends summarize_suite with <=10mm vs >10mm breakdown and M3 2-group torque stats."""
    summ = summarize_suite(rows)
    le10 = [r for r in rows if r['misalign_mm'] <= 10.0]
    gt10 = [r for r in rows if r['misalign_mm'] > 10.0]
    summ['le10_successes'] = sum(r['success'] for r in le10)
    summ['le10_trials'] = len(le10)
    summ['le10_rate'] = summ['le10_successes'] / max(len(le10), 1)
    summ['gt10_successes'] = sum(r['success'] for r in gt10)
    summ['gt10_trials'] = len(gt10)
    summ['gt10_rate'] = summ['gt10_successes'] / max(len(gt10), 1)
    for gkey in ('ts_lat_search_rim', 'ts_arm_search_rim', 'ts_lat_insert_seat', 'ts_arm_insert_seat'):
        vals = [r[gkey] for r in rows if r.get(gkey) is not None]
        if vals:
            summ[f'mean_{gkey}'] = float(np.mean(vals))
    return summ


def aggregate_seeds_p2(runs: list[dict], suite_key: str) -> dict:
    agg = aggregate_seeds(runs, suite_key)
    sums = [r['eval'][suite_key]['summary'] for r in runs]
    agg['mean_le10_rate'] = float(np.mean([s['le10_rate'] for s in sums]))
    agg['mean_gt10_rate'] = float(np.mean([s['gt10_rate'] for s in sums]))
    agg['per_seed_le10'] = [s['le10_successes'] for s in sums]
    agg['per_seed_gt10'] = [s['gt10_successes'] for s in sums]
    for gkey in ('mean_ts_lat_search_rim', 'mean_ts_arm_search_rim', 'mean_ts_lat_insert_seat', 'mean_ts_arm_insert_seat'):
        vals = [s[gkey] for s in sums if s.get(gkey) is not None]
        if vals:
            agg[gkey] = float(np.mean(vals))
    return agg


def _train_rollout_worker_p2(job: dict) -> dict:
    task: InsertionTaskSpec = job['task']
    mode: str = job['mode']
    use_tl = (mode == 'current_torque')

    env = _get_cached_env(task)
    obs, info = reset_cached_env_inplace(env, task)
    start_xyz = np.asarray(obs['ee_pose_fk'][:3], dtype=float).copy()

    builder = Phase2ProprioBuilder(mode=mode)
    pol = RecurrentAsymmetricPPO(
        obs_dim=builder.obs_dim,
        critic_dim=PRIVILEGED_CRITIC_DIM_P2,
        use_torque_limit=use_tl,
        hidden_dim=GRU_HIDDEN_DIM,
    )
    pol.set_weights(job['actor_weights'])
    h = pol.init_hidden()

    rng = np.random.default_rng(job['rollout_seed'])
    ar_rho = float(job.get('ar_rho', 0.88))
    innov = math.sqrt(max(1.0 - ar_rho * ar_rho, 0.0))
    eps = rng.normal(0.0, 1.0, size=pol.act_dim)

    obs_buf, crit_buf, z_buf, rew_buf, cls_buf, dir_buf, dmask_buf = [], [], [], [], [], [], []
    prev_priv = info['privileged']
    prev_env_act = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
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
        env_act, z, h, _, _ = pol.act_step(x_vec, h, deterministic=False, eps=eps)

        obs, _, terminated, truncated, info = env.step(env_act)
        ee_rel_xy = np.asarray(obs['ee_pose_fk'][:2], dtype=float) - start_xyz[:2]
        r, off_plate_term = compute_step_reward_p2(
            prev_priv, info['privileged'], env_act, prev_env_act, info,
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
        prev_env_act = env_act

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


def _eval_rl_worker_p2(job: dict) -> dict:
    """Deterministic held-out evaluation on a fresh unmodified BlindInsertionEnv."""
    task: InsertionTaskSpec = job['task']
    mode: str = job['mode']
    use_tl = (mode == 'current_torque')

    env = BlindInsertionEnv(task=task, render=False)
    try:
        obs, info = env.reset()
        builder = Phase2ProprioBuilder(mode=mode)
        pol = RecurrentAsymmetricPPO(
            obs_dim=builder.obs_dim,
            critic_dim=PRIVILEGED_CRITIC_DIM_P2,
            use_torque_limit=use_tl,
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
        ts_lat_by_phase: dict[str, list[float]] = {'search_rim': [], 'insert_seat': [], 'free': []}
        ts_arm_by_phase: dict[str, list[float]] = {'search_rim': [], 'insert_seat': [], 'free': []}

        while not (terminated or truncated):
            x_vec = builder.step(obs)
            true_cls_name = info['privileged']['contact_class']
            true_cls = CONTACT_CLASS_TO_IDX[true_cls_name]
            env_act, _, h, pred_cls, pred_dir = pol.act_step(x_vec, h, deterministic=True)

            aux_correct += int(pred_cls == true_cls)
            aux_total += 1
            xy_e = np.asarray(info['privileged']['socket_xy_error'], dtype=float)
            xy_n = float(np.linalg.norm(xy_e))
            pn = float(np.linalg.norm(pred_dir))
            if xy_n > 5e-4 and pn > 1e-6:
                dir_cos_list.append(float(np.dot(pred_dir / pn, -xy_e / xy_n)))

            ts_mean = float(np.mean(env_act[3:8]))
            ts_lat = float(env_act[3])
            ts_arm = float(env_act[4])
            if true_cls_name in ('plate_contact', 'rim_contact'):
                p_key = 'search_rim'
            elif true_cls_name in ('entering', 'seated'):
                p_key = 'insert_seat'
            else:
                p_key = 'free'
            ts_by_phase[p_key].append(ts_mean)
            ts_lat_by_phase[p_key].append(ts_lat)
            ts_arm_by_phase[p_key].append(ts_arm)

            obs, _, terminated, truncated, info = env.step(env_act)
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
            'ts_lat_search_rim': float(np.mean(ts_lat_by_phase['search_rim'])) if ts_lat_by_phase['search_rim'] else None,
            'ts_arm_search_rim': float(np.mean(ts_arm_by_phase['search_rim'])) if ts_arm_by_phase['search_rim'] else None,
            'ts_lat_insert_seat': float(np.mean(ts_lat_by_phase['insert_seat'])) if ts_lat_by_phase['insert_seat'] else None,
            'ts_arm_insert_seat': float(np.mean(ts_arm_by_phase['insert_seat'])) if ts_arm_by_phase['insert_seat'] else None,
        }
    finally:
        env.close()


def train_one_seed_p2(
    pool,
    label: str,
    mode: str,
    seed: int,
    seed_cursor: int,
    target_steps: int = 1_000_000,
    batch_size: int = 20,
    max_steps: int = 250,
) -> tuple[RecurrentAsymmetricPPO, dict, int, list[InsertionTaskSpec]]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    builder = Phase2ProprioBuilder(mode=mode)
    pol = RecurrentAsymmetricPPO(
        obs_dim=builder.obs_dim,
        critic_dim=PRIVILEGED_CRITIC_DIM_P2,
        use_torque_limit=(mode == 'current_torque'),
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
        # Smooth correlated exploration (0.70 -> 0.0 by 70%) for broad 16.5 mm discovery,
        # then pure on-policy i.i.d. PPO for the final 300k steps + gentle std cap (-0.55 -> -1.10).
        ar_rho_cur = 0.70 * max(0.0, 1.0 - progress / 0.70)
        log_std_cap = -0.55 + (-1.10 - (-0.55)) * min(1.0, progress / 0.90)

        # Collect at least min_batch_steps=1800 steps per PPO update (~400 updates per 1M steps)
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
                {'task': t, 'mode': mode, 'actor_weights': aw, 'rollout_seed': int(rs), 'ar_rho': ar_rho_cur}
                for t, rs in zip(wave_trials, rseeds)
            ]
            wave_trajs = pool.map(_train_rollout_worker_p2, jobs)
            trajs.extend(wave_trajs)
            batch_steps += sum(tr['steps'] for tr in wave_trajs)

        if upd == 0:
            pol.obs_rms.update(np.concatenate([tr['obs'] for tr in trajs], axis=0))
            pol.critic_rms.update(np.concatenate([tr['critic_obs'] for tr in trajs], axis=0))

        lr_scale = 0.25 + 0.75 * (1.0 - progress)
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
                f"  [{label:16s} s={seed}] upd {upd:3d} ({cum_steps:7d}/{target_steps} st, r<={eff_r_max*1000:4.1f}mm, {cum_steps/max(elapsed,1e-6):5.0f} st/s) | "
                f"succ={100*succ_all:5.1f}% (r100={100*roll_100:4.1f}%, B={100*succ_b:4.0f}% U={100*succ_u:4.0f}% shv={shoves:2d} off={off_plates:2d}/{len(trajs)}) | "
                f"ret={stats['mean_return']:+.2f} ev={stats['explained_var']:+.2f} aux={100*stats['aux_cls_acc']:4.1f}% "
                f"kl={stats['approx_kl']:.4f} std={np.round(stats['std'], 2)}",
                flush=True,
            )
            next_log_steps = cum_steps + max(target_steps // 5, 1)

    wall_s = time.perf_counter() - t0
    total_off_plate_train = sum(e['train_off_plate'] for e in history)
    meta = {
        'label': label,
        'mode': mode,
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
    parser.add_argument('--target-steps', type=int, default=1_000_000)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=250)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202])
    parser.add_argument('--methods', type=str, nargs='+', default=['M1_PosOnly', 'M2_Current', 'M3_Current_TL'])
    parser.add_argument('--out-prefix', type=str, default='phase2')
    parser.add_argument('--seed-offset', type=int, default=140000)
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
        ('M1_PosOnly', 'pos_only'),
        ('M2_Current', 'current'),
        ('M3_Current_TL', 'current_torque'),
    ]
    rl_specs = [(lbl, m) for lbl, m in all_rl_specs if lbl in set(args.methods)]

    ctx = mp.get_context('spawn')
    seed_cursor = TRAIN_SEED_OFFSET + int(args.seed_offset)
    all_train_trials: list[InsertionTaskSpec] = []
    rl_runs: dict[str, list[dict]] = {label: [] for label, _ in rl_specs}

    print(
        f'Launching Phase 2 Recurrent GRU({GRU_HIDDEN_DIM}) blind insertion RL ({args.workers} headless CPU workers, '
        f'{len(rl_specs)} methods x {len(args.seeds)} seeds x ~{args.target_steps:,} policy steps, max_steps={args.max_steps}, '
        f'smooth linear r_max 6.0->16.5mm over 35%)...',
        flush=True,
    )
    with ctx.Pool(args.workers, initializer=_init_worker_single_thread) as pool:
        baselines = {}
        if not args.skip_baselines:
            print('\n=== 1. Evaluating Reference Controllers on Splits A, B, C & OOD Slippery (mu=0.25, 120g) ===', flush=True)
            for cname in ('Nominal_StraightDown', 'BlindSpiral_Stiff_1.0', 'BlindSpiral_LowTorque_0.35', 'Privileged_Oracle'):
                baselines[cname] = {}
                parts = []
                for sk, trials in eval_suites.items():
                    rows = pool.map(_eval_baseline_worker, [(cname, t) for t in trials])
                    summ = summarize_suite_p2(rows)
                    baselines[cname][sk] = {'summary': summ, 'episodes': rows}
                    parts.append(
                        f'{sk[:8]}={summ["successes"]:2d}/30({100*summ["success_rate"]:4.1f}%,'
                        f'shv={summ["shove_failures"]},d={summ["median_socket_disp_mm"]:4.1f}mm)'
                    )
                print(f'  {cname:28s} | ' + ' | '.join(parts), flush=True)

        for label, mode in rl_specs:
            print(f'\n=== 2. Training {label} (GRU-{GRU_HIDDEN_DIM} + ee_rel_start + 4x4 mem, mode={mode}, {len(args.seeds)} seeds x {args.target_steps:,} steps) ===', flush=True)
            for seed in args.seeds:
                pol, meta, seed_cursor, used_trials = train_one_seed_p2(
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
                    rows = pool.map(_eval_rl_worker_p2, jobs)
                    summ = summarize_suite_p2(rows)
                    eval_out[sk] = {'summary': summ, 'episodes': rows}
                    parts.append(
                        f'{sk[:8]}={summ["successes"]:2d}/30({100*summ["success_rate"]:4.1f}%,'
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
            'obs_dim': runs[0]['meta']['obs_dim'],
            'act_dim': runs[0]['meta']['act_dim'],
            'hidden_dim': runs[0]['meta']['hidden_dim'],
            'actor_param_count': runs[0]['meta']['actor_param_count'],
            'critic_param_count': runs[0]['meta']['critic_param_count'],
            'mean_episodes_per_seed': float(np.mean([r['meta']['episodes_budget'] for r in runs])),
            'mean_total_steps': float(np.mean([r['meta']['total_steps'] for r in runs])),
            'mean_train_wall_s': float(np.mean([r['meta']['wall_s'] for r in runs])),
            'mean_steps_per_sec': float(np.mean([r['meta']['steps_per_sec'] for r in runs])),
            'eval': {sk: aggregate_seeds_p2(runs, sk) for sk in eval_suites},
        }

    main_suites = ('split_a_bolted', 'split_b_unbolted', 'split_c_tight_0.5mm')
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
        'main_suites': list(main_suites),
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
