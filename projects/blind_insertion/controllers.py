"""Phase 0 reference controllers (strictly legal obs for baselines; privileged for Oracle).

Controllers:
1. NominalStraightDownController: pushes straight down at torque_scale=1.0 (confirms <10% success).
2. BlindSpiralController(torque_scale=1.0): strong classical blind Archimedean spiral search
   using ONLY controller-legal `obs` (touchdown via tracking error / actuator force, spiral
   search with gentle downward bias, drop-in via ee_pose_fk[2], smooth downward seating).
3. BlindSpiralController(torque_scale=0.35): same blind spiral with fixed low torque limit.
4. PrivilegedOracleController: reads `info['privileged']['socket_xy_error']` to align and seat.
"""
from __future__ import annotations

import math
import numpy as np

from .env import ALLOWED_OBS_KEYS, FORBIDDEN_PRIVILEGED_KEYS, BlindInsertionEnv, InsertionTaskSpec


def _assert_legal_obs(obs: dict) -> None:
    keys = set(obs.keys())
    leaked = keys & FORBIDDEN_PRIVILEGED_KEYS
    if leaked:
        raise AssertionError(f'Privileged keys leaked into controller obs: {sorted(leaked)}')
    unknown = keys - ALLOWED_OBS_KEYS
    if unknown:
        raise AssertionError(f'Unexpected keys in controller obs: {sorted(unknown)}')


class NominalStraightDownController:
    """Open-loop straight-down insertion at fixed full stiffness (torque_scale=1.0)."""

    def __init__(self, dz_per_step: float = -0.0012, torque_scale: float = 1.0):
        self.dz_per_step = float(dz_per_step)
        self.torque_scale = float(torque_scale)

    def reset(self, obs: dict) -> None:
        _assert_legal_obs(obs)

    def act(self, obs: dict, info: dict | None = None) -> np.ndarray:
        _assert_legal_obs(obs)
        return np.array([0.0, 0.0, self.dz_per_step, self.torque_scale], dtype=float)


class BlindSpiralController:
    """Classical blind proprioceptive Archimedean spiral search (controller-legal obs only).

    State machine:
      1. `descend`: move slowly down until vertical contact is detected via arm tracking-error
         or actuator-force change -> record `z_contact` and `xy0`.
      2. `spiral`: track a fine-pitch (`1.6 mm` < `2 * 1.0 mm` minimum capture diameter)
         Archimedean spiral `r(t) = b * theta` covering `0 -> 18.5 mm` inside the 5.0 s budget,
         applying gentle downward virtual preload so the peg drops into the hole upon alignment.
         Whenever `ee_pose_fk[2]` drops > 1.8 mm below `z_contact`, switch immediately to `insert`.
      3. `insert`: apply small centering dither if stalled on the rim and descend smoothly to seat.
    """

    def __init__(
        self,
        torque_scale: float = 1.0,
        pitch_m: float = 0.0017,
        linear_speed_m_s: float = 0.14,
        max_radius_m: float = 0.0185,
        press_depth_m: float = 0.0012,
    ):
        self.torque_scale = float(torque_scale)
        self.pitch_m = float(pitch_m)
        self.linear_speed_m_s = float(linear_speed_m_s)
        self.max_radius_m = float(max_radius_m)
        self.press_depth_m = float(press_depth_m)
        self.phase = 'descend'
        self.xy0 = np.zeros(2, dtype=float)
        self.z_contact = 0.025
        self.f_free = np.zeros(6, dtype=float)
        self.theta = 0.0
        self.step_idx = 0
        self.insert_stall_steps = 0
        self.prev_z = 0.0

    def reset(self, obs: dict) -> None:
        _assert_legal_obs(obs)
        self.phase = 'descend'
        self.xy0 = obs['ee_pose_fk'][:2].copy()
        self.z_contact = float(obs['ee_pose_fk'][2])
        self.f_free = obs['actuator_force_noisy'].copy()
        self.theta = 0.0
        self.step_idx = 0
        self.insert_stall_steps = 0
        self.prev_z = float(obs['ee_pose_fk'][2])

    def act(self, obs: dict, info: dict | None = None) -> np.ndarray:
        _assert_legal_obs(obs)
        self.step_idx += 1
        ee_xyz = obs['ee_pose_fk'][:3]
        vz = float(obs['ee_twist_fk'][2])
        err_arm = float(np.linalg.norm(obs['tracking_error'][1:4]))
        df_arm = float(np.linalg.norm(obs['actuator_force_noisy'][1:4] - self.f_free[1:4]))

        if self.phase == 'descend':
            if self.step_idx >= 3 and (err_arm > 0.009 or df_arm > 0.14 or (self.step_idx >= 8 and abs(vz) < 0.008)):
                self.phase = 'spiral'
                self.xy0 = ee_xyz[:2].copy()
                self.z_contact = float(ee_xyz[2])
                self.theta = 0.0
            else:
                return np.array([0.0, 0.0, -0.0012, self.torque_scale], dtype=float)

        if self.phase == 'spiral':
            if ee_xyz[2] < self.z_contact - 0.0018:
                self.phase = 'insert'
                self.prev_z = float(ee_xyz[2])
                self.insert_stall_steps = 0
            else:
                b = self.pitch_m / (2.0 * math.pi)
                r_cur = min(b * self.theta, self.max_radius_m)
                dtheta = (self.linear_speed_m_s * 0.02) / max(r_cur, 0.0016)
                self.theta += dtheta
                r_next = min(b * self.theta, self.max_radius_m)
                target_xy = self.xy0 + np.array([r_next * math.cos(self.theta), r_next * math.sin(self.theta)])
                # Virtual spring target offset from current quantized EE position (compensated for backlash)
                d_xy = np.clip(1.35 * (target_xy - ee_xyz[:2]), -0.0038, 0.0038)
                z_target = self.z_contact - self.press_depth_m
                dz = float(np.clip(z_target - ee_xyz[2], -0.0015, 0.0010))
                return np.array([d_xy[0], d_xy[1], dz, self.torque_scale], dtype=float)

        # `insert` phase: descend smoothly; if stalled on rim/wall, apply small micro-dither.
        if abs(ee_xyz[2] - self.prev_z) < 0.00025:
            self.insert_stall_steps += 1
        else:
            self.insert_stall_steps = 0
        self.prev_z = float(ee_xyz[2])

        if self.insert_stall_steps >= 4:
            ang = 1.1 * self.step_idx
            d_xy = np.array([0.0012 * math.cos(ang), 0.0012 * math.sin(ang)])
        else:
            d_xy = np.zeros(2, dtype=float)
        return np.array([d_xy[0], d_xy[1], -0.0018, self.torque_scale], dtype=float)


class PrivilegedOracleController:
    """Privileged upper-bound controller knowing exact `info['privileged']['socket_xy_error']`.

    Centers above the hole before descending smoothly with compliant torque limits so both
    bolted (Regime A) and unbolted (Regime B) sockets seat cleanly without shoving.
    """

    def __init__(self, align_tol_m: float = 0.00035):
        self.align_tol_m = float(align_tol_m)
        self.phase = 'align'
        self.z_hover = 0.0265

    def reset(self, obs: dict) -> None:
        self.phase = 'align'
        self.z_hover = float(obs['ee_pose_fk'][2]) + 0.0015

    def act(self, obs: dict, info: dict) -> np.ndarray:
        priv = info['privileged']
        xy_err = np.asarray(priv['socket_xy_error'], dtype=float)
        rad_err = float(np.linalg.norm(xy_err))
        ee_z = float(obs['ee_pose_fk'][2])
        ts = 0.75 if priv['bolted'] else 0.55

        if self.phase == 'align':
            if rad_err <= self.align_tol_m:
                self.phase = 'insert'
            else:
                # Overcome gear backlash cleanly when error is small without overshooting.
                d_xy = -1.15 * xy_err
                norm = float(np.linalg.norm(d_xy))
                if 1e-5 < norm < 0.0009:
                    d_xy = d_xy * (0.0009 / norm)
                d_xy = np.clip(d_xy, -0.0032, 0.0032)
                dz = float(np.clip(self.z_hover - ee_z, -0.0015, 0.0020))
                return np.array([d_xy[0], d_xy[1], dz, ts], dtype=float)

        d_xy = np.clip(-1.1 * xy_err, -0.0020, 0.0020)
        dz = -0.0020 if rad_err < 0.0008 else -0.0006
        return np.array([d_xy[0], d_xy[1], dz, ts], dtype=float)


def run_episode(
    env: BlindInsertionEnv,
    task: InsertionTaskSpec,
    controller,
) -> dict:
    """Run one complete episode and return structured Phase 0 metrics."""
    obs, info = env.reset(task=task)
    controller.reset(obs)
    terminated = False
    truncated = False
    max_depth_mm = float(info['privileged']['peg_depth'] * 1000.0)

    while not (terminated or truncated):
        action = controller.act(obs, info)
        obs, _, terminated, truncated, info = env.step(action)
        d_mm = float(info['privileged']['peg_depth'] * 1000.0)
        if d_mm > max_depth_mm:
            max_depth_mm = d_mm

    priv = info['privileged']
    return {
        'trial_id': task.trial_id,
        'seed': task.seed,
        'shape': task.shape,
        'split': 'seen' if task.shape in ('round', 'square', 'rectangle') else 'unseen',
        'bolted': task.bolted,
        'clearance_mm': round(task.clearance_m * 1000.0, 3),
        'misalign_mm': round(float(np.linalg.norm(task.misalign_xy) * 1000.0), 2),
        'mu_peg_hole': round(task.mu_peg_hole, 3),
        'mu_table': round(task.mu_table, 3),
        'socket_mass_g': round(task.socket_mass_kg * 1000.0, 1),
        'success': bool(info['success']),
        'failure_mode': info['failure_mode'],
        'steps': int(env.step_count),
        'time_s': round(float(info['time_s']), 3),
        'final_depth_mm': round(float(priv['peg_depth'] * 1000.0), 2),
        'max_depth_mm': round(max_depth_mm, 2),
        'peak_contact_force_N': round(float(info['episode_peak_contact_force_N']), 2),
        'max_socket_disp_mm': round(float(info['episode_max_socket_disp_mm']), 2),
        'max_socket_tilt_deg': round(float(info['episode_max_socket_tilt_deg']), 2),
        'self_collision_steps': int(info['self_collision_steps']),
        'joint_limit_steps': int(info['joint_limit_steps']),
    }
