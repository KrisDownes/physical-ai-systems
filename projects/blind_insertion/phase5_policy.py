"""Phase 5 Residual GRU(64) with Sigmoid Spiral Braking (`logit_g = +1.0` init) + Fine Micro-Centering.

Surgical upgrade over Phase 4 fixing the spiral-drag bottleneck (`g_spiral` stuck at 0.85):
- True spiral braking: `g_spiral = sigmoid(1.0 + 2.5 * z[3])`:
  * At step-0 zero-init (`z[3] = 0`): `logit_g = +1.0` -> `g_spiral = 0.7311` (non-saturating gradient!)
  * Smoothly drops to `< 0.10` (true brake/pause) at `z[3] <= -1.28`, or ramps to `0.97` at `z[3] = +1.0`.
- Fine micro-centering when braked:
  * Lateral residual cap scales cleanly with `g_spiral`: `res_cap_xy = 0.0015 + 0.0010 * g_spiral`
    (`±1.5 mm` crisp micro-centering when `g_spiral -> 0` so `z` makes `0.2-1.5 mm` corrections
    without fighting an 85%-speed base spiral; `±2.5 mm` when cruising at `g_spiral -> 1`).
- Exact same clean high-SNR kinematic + `tracking_error` + `a_base_t` observation interface from Phase 4
  (ZERO raw `actuator_force_noisy`, ZERO noisy `d_tau_ext`) and exact `[alpha_lat, alpha_arm] in [0.08, 1.0]`
  initialized at `0.35` for `R2_v2`.
"""
from __future__ import annotations

import numpy as np

from .env import MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_Z_M
from .phase1_policy import LOG_STD_MAX, LOG_STD_MIN
from .phase2_policy import GRU_HIDDEN_DIM, PRIVILEGED_CRITIC_DIM_P2
from .phase4_policy import (
    BASE_SPIRAL_ALPHA,
    P4_ALPHA_MAX,
    P4_ALPHA_MIN,
    RES_ALPHA_GAIN,
    RES_MAX_Z_M,
    Phase4ResidualBuilder,
    Phase4ResidualPPO,
)

P5_LOGIT_G_INIT = 1.0     # sigmoid(+1.0) = 0.7311 -> non-saturating, smoothly brakes to < 0.10!
P5_LOGIT_G_GAIN = 2.5     # z[3] = -1.3 -> logit = -2.25 -> g_spiral = 0.095 (< 0.10)
P5_RES_FINE_XY_M = 0.0015 # 0.2-1.5 mm fine micro-centering cap when g_spiral -> 0
P5_RES_FAST_ADD_M = 0.0010


def _scalar_sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20.0, 20.0)))))


class Phase5ResidualPPO(Phase4ResidualPPO):
    """Phase 5 Residual GRU(64) with `g_spiral = sigmoid(1.0 + 2.5 * z_g)` braking + fine micro-centering."""

    def __init__(
        self,
        obs_dim: int,
        mode: str = 'res_gravcomp_tl',
        base_alpha: float = BASE_SPIRAL_ALPHA,
        critic_dim: int = PRIVILEGED_CRITIC_DIM_P2,
        hidden_dim: int = GRU_HIDDEN_DIM,
        critic_h1: int = 128,
        critic_h2: int = 96,
        seed: int = 0,
        init_log_std: float = -1.00,
    ):
        super().__init__(
            obs_dim=obs_dim,
            mode=mode,
            base_alpha=base_alpha,
            critic_dim=critic_dim,
            hidden_dim=hidden_dim,
            critic_h1=critic_h1,
            critic_h2=critic_h2,
            seed=seed,
            init_log_std=init_log_std,
        )

    @classmethod
    def load(cls, path) -> 'Phase5ResidualPPO':
        data = np.load(path, allow_pickle=False)
        pol = cls(
            obs_dim=int(data['obs_dim']),
            mode=str(data['mode']),
            base_alpha=float(data['base_alpha']),
            critic_dim=int(data['critic_dim']),
            hidden_dim=int(data['hidden_dim']),
            critic_h1=int(data['critic_h1']),
            critic_h2=int(data['critic_h2']),
        )
        meta = {'obs_dim', 'mode', 'base_alpha', 'critic_dim', 'use_torque_limit', 'hidden_dim', 'critic_h1', 'critic_h2'}
        pol.set_weights({k: data[k] for k in data.files if k not in meta})
        return pol

    def compose_action(
        self,
        base_dxyz: np.ndarray,
        z: np.ndarray,
    ) -> tuple[dict | np.ndarray, np.ndarray, float, tuple[float, float] | None]:
        """Compose sigmoid-braked spiral + fine micro-centering residual + dynamic GravComp-TL."""
        logit_g = P5_LOGIT_G_INIT + P5_LOGIT_G_GAIN * float(z[3])
        g_spiral = _scalar_sigmoid(logit_g)

        res_xy_cap = P5_RES_FINE_XY_M + P5_RES_FAST_ADD_M * g_spiral
        res_xy = np.clip(z[:2], -1.0, 1.0) * res_xy_cap
        res_z = float(np.clip(z[2], -1.0, 1.0)) * RES_MAX_Z_M

        d_xyz = np.array([
            float(np.clip(g_spiral * base_dxyz[0] + res_xy[0], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M)),
            float(np.clip(g_spiral * base_dxyz[1] + res_xy[1], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M)),
            float(np.clip(g_spiral * base_dxyz[2] + res_z, -MAX_TASK_DELTA_Z_M, 0.0020)),
        ], dtype=float)

        if self.mode == 'res_pos_only':
            env_act = np.concatenate([d_xyz, np.ones(5, dtype=float)])
            return env_act, d_xyz, g_spiral, None

        a_lat = float(np.clip(self.base_alpha + RES_ALPHA_GAIN * z[4], P4_ALPHA_MIN, P4_ALPHA_MAX))
        a_arm = float(np.clip(self.base_alpha + RES_ALPHA_GAIN * z[5], P4_ALPHA_MIN, P4_ALPHA_MAX))
        env_act = {
            'delta_xyz': d_xyz,
            'gravcomp_alpha': np.array([a_lat, a_arm], dtype=float),
        }
        return env_act, d_xyz, g_spiral, (a_lat, a_arm)
