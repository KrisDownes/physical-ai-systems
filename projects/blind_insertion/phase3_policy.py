"""Phase 3 Recurrent GRU(64) Actor-Critic with Gravity-Subtracted Current Sensing (tau_ext, d_tau_ext) & GravComp-TL.

Exact 3-way Phase 3 showdown on identical GRU(64) backbone, ee_rel_start + 4x4 legal spatial memory,
asymmetric privileged critic, dual auxiliary contact decoder, smooth linear curriculum, and reward:
- `pos_only` (`M1_PosOnly_2M`):       Phase 2 M1 scaled to 2.0M steps (obs_dim=69, act_dim=3, fixed ts=1.0).
                                      Strictly ZERO actuator_force, ZERO tau_ext, ZERO tracking_error.
- `gravcomp_fixed` (`M4_Fixed_GravComp`): M1 features + tau_ext + d_tau_ext + multi-scale tau_ext EMAs +
                                      tracking_error (obs_dim=97, act_dim=3), fixed GravComp-TL alpha=0.20.
- `gravcomp_tl` (`M4_GravComp_TL`):   Same gravity-subtracted features + prev_alpha (obs_dim=99, act_dim=5),
                                      outputs active 2-group GravComp-TL [alpha_lat, alpha_arm] in [0.04, 1.0]
                                      initialized at alpha=0.35 on episode 0.
"""
from __future__ import annotations

import math
import numpy as np

from .env import (
    ALLOWED_OBS_KEYS,
    FORBIDDEN_PRIVILEGED_KEYS,
    MAX_TASK_DELTA_XY_M,
    MAX_TASK_DELTA_Z_M,
)
from .phase1_policy import CLOCK_PERIOD_S, FOURIER_FREQS, LOG_STD_MAX, LOG_STD_MIN
from .phase2_policy import (
    GRID_HALF_EXTENT_M,
    GRID_N,
    GRU_HIDDEN_DIM,
    NEIGHBOR_OFFSETS,
    PRIVILEGED_CRITIC_DIM_P2,
    SPATIAL_MEM_DIM,
    RecurrentAsymmetricPPO,
)
from .phase3_gravcomp import (
    GRAVCOMP_ALPHA_MAX,
    GRAVCOMP_ALPHA_MIN,
    compute_arm_gravity_torque,
)

M4_ALPHA_INIT = 0.35
M4_FIXED_ALPHA = 0.20


class Phase3ProprioBuilder:
    """Controller-legal per-step feature builder with gravity-subtracted external torque (tau_ext, d_tau_ext)."""

    def __init__(self, mode: str, fixed_alpha: float = M4_FIXED_ALPHA):
        if mode not in ('pos_only', 'gravcomp_fixed', 'gravcomp_tl'):
            raise ValueError(f'Unknown Phase 3 mode {mode!r}')
        self.mode = mode
        self.fixed_alpha = float(fixed_alpha)
        self.reset()

    @property
    def obs_dim(self) -> int:
        if self.mode == 'pos_only':
            return SPATIAL_MEM_DIM + 25  # 69 (100% identical to Phase 2 M1)
        if self.mode == 'gravcomp_fixed':
            return SPATIAL_MEM_DIM + 25 + 28  # 97
        return SPATIAL_MEM_DIM + 25 + 30      # 99 (+ prev_alpha_lat, prev_alpha_arm)

    def reset(self) -> None:
        self._start_xyz: np.ndarray | None = None
        self._prev_ee_pos: np.ndarray | None = None
        self._z_min_rel: float = 0.0
        self._xy_at_z_min: np.ndarray = np.zeros(2, dtype=float)
        self._z_plate_est: float | None = None
        self._stall_steps: int = 0
        self._step_count: int = 0
        self._grid = np.zeros((GRID_N, GRID_N), dtype=float)
        self._ema_vel_fast = np.zeros(3, dtype=float)
        self._ema_vel_slow = np.zeros(3, dtype=float)
        self._prev_tau_ext: np.ndarray | None = None
        self._ema_tau_fast = np.zeros(5, dtype=float)
        self._ema_tau_slow = np.zeros(5, dtype=float)
        self._prev_alpha = np.array([M4_ALPHA_INIT, M4_ALPHA_INIT], dtype=float)
        self.last_tau_ext = np.zeros(5, dtype=float)
        self.last_d_tau_ext = np.zeros(5, dtype=float)
        self.last_obs_keys_seen: set[str] = set()

    @property
    def visited_fraction(self) -> float:
        return float(np.mean(self._grid > 0.0))

    def set_prev_alpha(self, alpha_lat: float, alpha_arm: float) -> None:
        self._prev_alpha = np.array([float(alpha_lat), float(alpha_arm)], dtype=float)

    def step(self, obs: dict) -> np.ndarray:
        """Return 1D controller-legal feature vector x_t."""
        keys = set(obs.keys())
        self.last_obs_keys_seen |= keys
        leaked = keys & FORBIDDEN_PRIVILEGED_KEYS
        if leaked:
            raise AssertionError(f'Privileged keys leaked into actor obs: {sorted(leaked)}')
        unknown = keys - ALLOWED_OBS_KEYS
        if unknown:
            raise AssertionError(f'Unexpected keys in actor obs: {sorted(unknown)}')

        self._step_count += 1
        q = np.asarray(obs['q_quantized'][:5], dtype=float)
        qdot = np.asarray(obs['q_dot_quantized'][:5], dtype=float)
        ee_pos = np.asarray(obs['ee_pose_fk'][:3], dtype=float)
        ee_vel = np.asarray(obs['ee_twist_fk'][:3], dtype=float)
        prev_xyz = np.asarray(obs['prev_action'][:3], dtype=float)
        time_s = float(obs['time_s'])

        if self._start_xyz is None:
            self._start_xyz = ee_pos.copy()
            self._prev_ee_pos = ee_pos.copy()
            self._z_min_rel = 0.0
            self._xy_at_z_min = ee_pos[:2].copy()

        ee_rel = ee_pos - self._start_xyz
        delta_ee = ee_pos - self._prev_ee_pos
        self._prev_ee_pos = ee_pos.copy()

        if self._z_plate_est is None and self._step_count >= 5 and abs(ee_vel[2]) < 0.008 and ee_rel[2] < -0.0004:
            self._z_plate_est = float(ee_pos[2])

        if ee_rel[2] < self._z_min_rel - 1e-5:
            self._z_min_rel = float(ee_rel[2])
            self._xy_at_z_min = ee_pos[:2].copy()

        z_plate_drop = (self._z_plate_est - float(ee_pos[2])) if self._z_plate_est is not None else 0.0
        xy_to_dip = self._xy_at_z_min - ee_pos[:2]

        speed_3d = float(np.linalg.norm(ee_vel))
        if speed_3d < 0.008:
            self._stall_steps = min(self._stall_steps + 1, 25)
        else:
            self._stall_steps = max(self._stall_steps - 2, 0)
        stall_frac = self._stall_steps / 25.0

        r_rel = float(np.linalg.norm(ee_rel[:2]))
        if r_rel > 1e-6:
            c_th, s_th = float(ee_rel[0] / r_rel), float(ee_rel[1] / r_rel)
        else:
            c_th, s_th = 0.0, 0.0

        gx = int(np.clip(math.floor(((ee_rel[0] + GRID_HALF_EXTENT_M) / (2.0 * GRID_HALF_EXTENT_M)) * GRID_N), 0, GRID_N - 1))
        gy = int(np.clip(math.floor(((ee_rel[1] + GRID_HALF_EXTENT_M) / (2.0 * GRID_HALF_EXTENT_M)) * GRID_N), 0, GRID_N - 1))
        self._grid[gx, gy] = min(1.0, self._grid[gx, gy] + 0.20)

        cell_w = (2.0 * GRID_HALF_EXTENT_M) / GRID_N
        compass = np.zeros(len(NEIGHBOR_OFFSETS), dtype=float)
        for idx_n, (dx_i, dy_i) in enumerate(NEIGHBOR_OFFSETS):
            nx, ny = gx + dx_i, gy + dy_i
            if 0 <= nx < GRID_N and 0 <= ny < GRID_N:
                cx = -GRID_HALF_EXTENT_M + (nx + 0.5) * cell_w
                cy = -GRID_HALF_EXTENT_M + (ny + 0.5) * cell_w
                if math.hypot(cx, cy) <= 0.0185:
                    compass[idx_n] = 1.0 - self._grid[nx, ny]

        t_frac = min(time_s / CLOCK_PERIOD_S, 1.0)
        fourier = np.array(
            [v for f in FOURIER_FREQS for v in (math.sin(2.0 * math.pi * f * t_frac), math.cos(2.0 * math.pi * f * t_frac))],
            dtype=float,
        )

        self._ema_vel_fast = 0.35 * ee_vel + 0.65 * self._ema_vel_fast
        self._ema_vel_slow = 0.06 * ee_vel + 0.94 * self._ema_vel_slow

        spatial_mem = np.concatenate([
            np.array([
                ee_rel[0] / 0.02,
                ee_rel[1] / 0.02,
                ee_rel[2] / 0.015,
                r_rel / 0.02,
                c_th,
                s_th,
                self._z_min_rel / 0.015,
                z_plate_drop / 0.010,
                xy_to_dip[0] / 0.015,
                xy_to_dip[1] / 0.015,
                stall_frac,
                t_frac,
            ], dtype=float),
            fourier,
            self._grid.reshape(-1),
            compass,
        ])

        vec_m1 = np.concatenate([
            spatial_mem,
            q / 1.5,
            qdot / 1.5,
            ee_vel / 0.10,
            prev_xyz / 0.004,
            self._ema_vel_fast / 0.10,
            self._ema_vel_slow / 0.10,
            delta_ee / 0.003,
        ])
        if self.mode == 'pos_only':
            return vec_m1

        # Controller-legal gravity-subtracted external torque & 1-step derivative (zero privileged info!)
        tau_grav = compute_arm_gravity_torque(obs['q_quantized'])
        f_noisy = np.asarray(obs['actuator_force_noisy'][:5], dtype=float)
        tau_ext = f_noisy - tau_grav
        if self._prev_tau_ext is None:
            d_tau_ext = np.zeros(5, dtype=float)
        else:
            d_tau_ext = tau_ext - self._prev_tau_ext
        self._prev_tau_ext = tau_ext.copy()
        self.last_tau_ext = tau_ext.copy()
        self.last_d_tau_ext = d_tau_ext.copy()

        self._ema_tau_fast = 0.35 * tau_ext + 0.65 * self._ema_tau_fast
        self._ema_tau_slow = 0.06 * tau_ext + 0.94 * self._ema_tau_slow

        err = np.asarray(obs['tracking_error'][:5], dtype=float)
        vec_m4_base = np.concatenate([
            vec_m1,                        # 69
            tau_ext / 0.80,                # 5
            d_tau_ext / 0.40,              # 5
            self._ema_tau_fast / 0.80,     # 5
            self._ema_tau_slow / 0.80,     # 5
            err / 0.03,                    # 5
            np.array([
                float(np.linalg.norm(tau_ext)) / 1.0,
                float(np.linalg.norm(d_tau_ext)) / 0.5,
                float(np.linalg.norm(err)) / 0.05,
            ], dtype=float),               # 3
        ])  # 97 dims
        if self.mode == 'gravcomp_fixed':
            return vec_m4_base
        return np.concatenate([vec_m4_base, self._prev_alpha])  # 99 dims


class Phase3GravCompPPO(RecurrentAsymmetricPPO):
    """Phase 3 Recurrent GRU(64) PPO supporting `pos_only`, `gravcomp_fixed`, and `gravcomp_tl`."""

    def __init__(
        self,
        obs_dim: int,
        mode: str = 'gravcomp_tl',
        fixed_alpha: float = M4_FIXED_ALPHA,
        critic_dim: int = PRIVILEGED_CRITIC_DIM_P2,
        hidden_dim: int = GRU_HIDDEN_DIM,
        critic_h1: int = 128,
        critic_h2: int = 96,
        seed: int = 0,
        init_log_std: float = -0.65,
    ):
        if mode not in ('pos_only', 'gravcomp_fixed', 'gravcomp_tl'):
            raise ValueError(f'Unknown Phase 3 mode {mode!r}')
        self.mode = mode
        self.fixed_alpha = float(fixed_alpha)
        super().__init__(
            obs_dim=obs_dim,
            critic_dim=critic_dim,
            use_torque_limit=(mode == 'gravcomp_tl'),
            hidden_dim=hidden_dim,
            critic_h1=critic_h1,
            critic_h2=critic_h2,
            seed=seed,
            init_log_std=init_log_std,
        )

    def save(self, path) -> None:
        np.savez(
            path,
            obs_dim=self.obs_dim,
            mode=self.mode,
            fixed_alpha=self.fixed_alpha,
            critic_dim=self.critic_dim,
            use_torque_limit=int(self.use_torque_limit),
            hidden_dim=self.hidden_dim,
            critic_h1=self.critic_h1,
            critic_h2=self.critic_h2,
            **self.get_all_weights(),
        )

    @classmethod
    def load(cls, path) -> 'Phase3GravCompPPO':
        data = np.load(path, allow_pickle=False)
        pol = cls(
            obs_dim=int(data['obs_dim']),
            mode=str(data['mode']),
            fixed_alpha=float(data['fixed_alpha']),
            critic_dim=int(data['critic_dim']),
            hidden_dim=int(data['hidden_dim']),
            critic_h1=int(data['critic_h1']),
            critic_h2=int(data['critic_h2']),
        )
        meta = {'obs_dim', 'mode', 'fixed_alpha', 'critic_dim', 'use_torque_limit', 'hidden_dim', 'critic_h1', 'critic_h2'}
        pol.set_weights({k: data[k] for k in data.files if k not in meta})
        return pol

    def map_z_to_p3_action(self, z: np.ndarray) -> tuple[dict | np.ndarray, np.ndarray, tuple[float, float] | None]:
        """Map pre-action z -> (env_action, d_xyz_3, (alpha_lat, alpha_arm) or None)."""
        d_xy = np.clip(z[:2], -1.0, 1.0) * MAX_TASK_DELTA_XY_M
        d_z = float(np.clip(z[2], -1.0, 0.5)) * MAX_TASK_DELTA_Z_M
        d_xyz = np.array([d_xy[0], d_xy[1], d_z], dtype=float)

        if self.mode == 'pos_only':
            return np.concatenate([d_xyz, np.ones(5, dtype=float)]), d_xyz, None
        if self.mode == 'gravcomp_fixed':
            a = float(np.clip(self.fixed_alpha, GRAVCOMP_ALPHA_MIN, GRAVCOMP_ALPHA_MAX))
            return {'delta_xyz': d_xyz, 'gravcomp_alpha': np.array([a, a], dtype=float)}, d_xyz, (a, a)
        # M4_GravComp_TL: zero-init (z=0) -> alpha=0.35; z in [-0.7, +1.45] spans full [0.04, 1.00]
        a_lat = float(np.clip(M4_ALPHA_INIT + 0.45 * z[3], GRAVCOMP_ALPHA_MIN, GRAVCOMP_ALPHA_MAX))
        a_arm = float(np.clip(M4_ALPHA_INIT + 0.45 * z[4], GRAVCOMP_ALPHA_MIN, GRAVCOMP_ALPHA_MAX))
        return {'delta_xyz': d_xyz, 'gravcomp_alpha': np.array([a_lat, a_arm], dtype=float)}, d_xyz, (a_lat, a_arm)

    def act_step_p3(
        self,
        x_raw: np.ndarray,
        h_prev: np.ndarray,
        deterministic: bool = True,
        eps: np.ndarray | None = None,
    ):
        x_norm = self.obs_rms.normalize(x_raw)
        h_next = self.step_gru_single(x_norm, h_prev)
        mu = h_next @ self.W_mu + self.b_mu
        aux_cls = h_next @ self.W_aux_cls + self.b_aux_cls
        aux_dir = h_next @ self.W_aux_dir + self.b_aux_dir
        if deterministic:
            z = mu.copy()
        else:
            std = np.exp(np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
            noise = np.zeros(self.act_dim, dtype=float) if eps is None else eps
            z = mu + std * noise
        env_act, d_xyz, alpha_pair = self.map_z_to_p3_action(z)
        return env_act, d_xyz, alpha_pair, z, h_next, int(np.argmax(aux_cls)), aux_dir.copy()
