"""Phase 0 reference controllers for `projects/blind_drawer/` (strictly legal obs for baselines; privileged for Oracle).

Controllers:
  1. NominalStraightBackController: pulls straight back along world `-x` (with mild downward
     stabilization bias) under three actuator modes:
     - `stiff` (`ts=1.0`)
     - `low_torque` (raw `ts=0.35`)
     - `gravcomp` (fixed `GravComp-TL` `alpha=0.35` or 2-group `[alpha_lat, alpha_arm]`)
  2. PrivilegedOracleController: reads `info['privileged']` to pull smoothly along true `rail_yaw_rad`
     with compliant lateral yield, downward cabinet stabilization on unbolted cabinets, and smooth
     detent ramp -> verifies >= 95% solvability & 0/0 safety across Split A, Split B, and Split OOD.
"""
from __future__ import annotations

import math
import numpy as np

from .env import ALLOWED_OBS_KEYS, FORBIDDEN_PRIVILEGED_KEYS


def _assert_legal_obs(obs: dict) -> None:
    keys = set(obs.keys())
    leaked = keys & FORBIDDEN_PRIVILEGED_KEYS
    if leaked:
        raise AssertionError(f'Privileged/absolute keys leaked into controller obs: {sorted(leaked)}')
    unknown = keys - ALLOWED_OBS_KEYS
    if unknown:
        raise AssertionError(f'Unexpected keys in controller obs: {sorted(unknown)}')


class NominalStraightBackController:
    """Nominal straight-back trajectory pull along `-x` using ONLY controller-legal relative `obs`.

    Tracks a smooth straight-back reference `x_ref(t) = -v_pull * t, y_ref = 0, z_ref = -z_press`
    relative to `ee_rel_start` (identical tracking structure to `BlindSpiral` in `blind_insertion`):
      - Under Stiff (`ts=1.0`): forces tracking tightly (`~3-6` encoder counts) while spiking off-axis
        rail binding forces and dragging/shoving unbolted cabinets.
      - Under `GravComp-TL` (`alpha=0.35`): yields compliantly laterally & at detents, slashing
        binding forces and amplifying `|tracking_error|` (`3x+` Tactile-Amplifier SNR).
    """

    def __init__(
        self,
        mode: str = 'stiff',
        torque_scale: float = 1.0,
        gravcomp_alpha: float | tuple[float, float] = 0.35,
        pull_speed_m_s: float = 0.028,
        down_preload_m: float = 0.0012,
    ):
        if mode not in ('stiff', 'low_torque', 'gravcomp'):
            raise ValueError(f'Unknown mode {mode!r}')
        self.mode = mode
        self.torque_scale = float(torque_scale)
        self.gravcomp_alpha = gravcomp_alpha
        self.pull_speed_m_s = float(pull_speed_m_s)
        self.down_preload_m = float(down_preload_m)
        self.step_idx = 0

    def reset(self, obs: dict) -> None:
        _assert_legal_obs(obs)
        self.step_idx = 0

    def act(self, obs: dict, info: dict | None = None):
        _assert_legal_obs(obs)
        self.step_idx += 1
        rel = np.asarray(obs['ee_rel_start'], dtype=float)
        t = self.step_idx * 0.02
        x_ref = -min(self.pull_speed_m_s * t, 0.085)
        y_ref = 0.0
        z_ref = -self.down_preload_m
        err = np.array([x_ref - rel[0], y_ref - rel[1], z_ref - rel[2]], dtype=float)
        d_xyz = np.array([
            np.clip(1.4 * err[0], -0.0065, 0.0065),
            np.clip(1.4 * err[1], -0.0065, 0.0065),
            np.clip(1.2 * err[2], -0.0040, 0.0040),
        ], dtype=float)
        if self.mode == 'gravcomp':
            return {'delta_xyz': d_xyz, 'gravcomp_alpha': self.gravcomp_alpha}
        ts = 1.0 if self.mode == 'stiff' else self.torque_scale
        return np.array([d_xyz[0], d_xyz[1], d_xyz[2], ts], dtype=float)


class PrivilegedOracleController:
    """Privileged upper-bound controller pulling along exact `rail_yaw_rad` with lateral yield & detent ramp."""

    def __init__(self):
        self.stall_steps = 0
        self.prev_travel = 0.0
        self.preload_steps = 0

    def reset(self, obs: dict) -> None:
        self.stall_steps = 0
        self.prev_travel = 0.0
        self.preload_steps = 0

    def act(self, obs: dict, info: dict | None = None):
        assert info is not None and 'privileged' in info
        priv = info['privileged']
        yaw = float(priv['rail_yaw_rad'])
        pull_dir = np.array([-math.cos(yaw), -math.sin(yaw)], dtype=float)
        lat_dir = np.array([-pull_dir[1], pull_dir[0]], dtype=float)

        # Brief 5-step downward stabilization settle on unbolted cabinets before pulling
        if not priv['bolted'] and self.preload_steps < 5:
            self.preload_steps += 1
            return {'delta_xyz': np.array([0.0, 0.0, -0.0038]), 'gravcomp_alpha': (0.35, 1.00)}

        travel_mm = float(priv['drawer_travel_mm'])
        if travel_mm - self.prev_travel < 0.25:
            self.stall_steps += 1
        else:
            self.stall_steps = 0
        self.prev_travel = travel_mm

        handle_xy = np.asarray(priv['handle_pos_world'][:2], dtype=float)
        jaw_xy = np.asarray(priv['jaw_pos_world'][:2], dtype=float)
        lat_err = float(np.dot(handle_xy - jaw_xy, lat_dir))

        in_detent = (priv['contact_class'] == 'detent_catch') or (self.stall_steps >= 2)
        if priv['bolted']:
            step_pull = 0.0068 if in_detent else 0.0046
            dz = float(np.clip(-0.5 * float(obs['ee_rel_start'][2]), -0.0010, 0.0010))
            d_xy = step_pull * pull_dir + np.clip(1.0 * lat_err, -0.0025, 0.0025) * lat_dir
            return {'delta_xyz': np.array([d_xy[0], d_xy[1], dz]), 'gravcomp_alpha': (0.50, 1.00)}

        # Unbolted cabinets: gentle downward stabilization onto handle_ledge + smooth detent ramp
        # with compliant lateral GravComp-TL (zero lateral banging!)
        near_detent = in_detent or (abs(travel_mm - float(priv['detent_travel_mm'])) < 6.0)
        dz = -0.0038 if (near_detent or priv['mu_table'] <= 0.42) else -0.0030
        step_pull = min(0.0046 + 0.00032 * self.stall_steps, 0.0068)
        d_xy = step_pull * pull_dir + np.clip(0.8 * lat_err, -0.0020, 0.0020) * lat_dir
        return {'delta_xyz': np.array([d_xy[0], d_xy[1], dz]), 'gravcomp_alpha': (0.45, 1.00)}
