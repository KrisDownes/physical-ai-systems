"""Phase 4 Clean-Kinematics Residual GRU(64) + Base-Spiral Brake Gate (`g_spiral`) + Dynamic `GravComp-TL`.

Decoupled architecture synthesizing Phases 0-3:
- Frozen Base Controller:
  * `R1 · Residual-PosOnly` (`res_pos_only`):   Frozen `BlindSpiral_Stiff (ts=1.0)` + residual `[dx, dy, dz, g_spiral]`,
                                                 fixed `ts = 1.0` (`act_dim = 4`, `obs_dim = 84`).
  * `R2 · Residual-GravComp-TL` (`res_gravcomp_tl`): Frozen `BlindSpiral_GravComp (alpha=0.35)` + residual
                                                 `[dx, dy, dz, g_spiral, alpha_lat, alpha_arm]` with `alpha in [0.08, 1.0]`
                                                 initialized at `0.35` (`act_dim = 6`, `obs_dim = 86`).
- Clean High-SNR Sensing (ZERO raw `actuator_force_noisy`, ZERO noisy `d_tau_ext`):
  * Actor observes ONLY clean, controller-legal kinematic features (`q_quantized`, `q_dot_quantized`,
    `ee_pose_fk`, `ee_twist_fk`, `ee_rel_start`, `prev_action`, 4x4 spatial memory, Fourier clock) +
    `a_base_t` (base spiral action/phase/radius/stall) + `tracking_error = q_cmd - q_quantized` +
    `prev_g_spiral` (+ `prev_alpha` for R2).
- Exact Episode-0 Identity:
  * Policy head `W_mu, b_mu` is strictly zero-initialized -> `z = 0` outputs `res_dxyz = 0`,
    `g_spiral = 1.0`, and `alpha = (0.35, 0.35)` -> bit-identical to the frozen base controller!
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
from .phase3_gravcomp import GRAVCOMP_ALPHA_MAX

BASE_SPIRAL_ALPHA = 0.35
P4_ALPHA_MIN = 0.08
P4_ALPHA_MAX = GRAVCOMP_ALPHA_MAX  # 1.00
RES_MAX_XY_M = 0.0028   # full lateral micro-centering & pitch-filling authority when g_spiral brakes
RES_MAX_Z_M = 0.0020    # unload (+dz) out of rim jams or drive tight 0.5 mm seating
RES_GATE_GAIN = 0.60    # z_g=0 -> exact g_spiral=1.0; z_g=-1.67 -> full brake g_spiral=0.0
RES_ALPHA_GAIN = 0.35   # z_a=0 -> exact alpha=0.35; z_a=-0.77 -> 0.08; z_a=+1.86 -> 1.00


class Phase4ResidualBuilder:
    """Clean high-SNR kinematic + tracking_error + base-spiral feature builder (ZERO actuator_force_noisy, ZERO d_tau_ext)."""

    def __init__(self, mode: str):
        if mode not in ('res_pos_only', 'res_gravcomp_tl'):
            raise ValueError(f'Unknown Phase 4 mode {mode!r}')
        self.mode = mode
        self.reset()

    @property
    def obs_dim(self) -> int:
        # 69 (M1 clean kinematics) + 8 (base spiral ctx) + 6 (tracking_error + norm) + 1 (prev_g_spiral) = 84
        if self.mode == 'res_pos_only':
            return SPATIAL_MEM_DIM + 25 + 8 + 6 + 1       # 84
        return SPATIAL_MEM_DIM + 25 + 8 + 6 + 1 + 2       # 86 (+ prev_alpha_lat, prev_alpha_arm)

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
        self._prev_g_spiral: float = 1.0
        self._prev_alpha = np.array([BASE_SPIRAL_ALPHA, BASE_SPIRAL_ALPHA], dtype=float)
        self.last_obs_keys_seen: set[str] = set()

    @property
    def visited_fraction(self) -> float:
        return float(np.mean(self._grid > 0.0))

    def set_prev_controls(self, g_spiral: float, alpha_pair: tuple[float, float] | None = None) -> None:
        self._prev_g_spiral = float(g_spiral)
        if alpha_pair is not None:
            self._prev_alpha = np.array([float(alpha_pair[0]), float(alpha_pair[1])], dtype=float)

    def step(
        self,
        obs: dict,
        base_dxyz: np.ndarray,
        base_phase: str,
        base_r_m: float,
        base_stall_steps: int,
    ) -> np.ndarray:
        """Build clean kinematic + tracking_error + base-spiral obs vector (strictly zero actuator_force_noisy)."""
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
        err = np.asarray(obs['tracking_error'][:5], dtype=float)
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

        p_oh = np.array([
            1.0 if base_phase == 'descend' else 0.0,
            1.0 if base_phase == 'spiral' else 0.0,
            1.0 if base_phase == 'insert' else 0.0,
        ], dtype=float)
        base_ctx = np.concatenate([
            np.asarray(base_dxyz[:3], dtype=float) / np.array([MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_Z_M]),
            p_oh,
            np.array([min(float(base_r_m) / 0.0185, 1.2), min(float(base_stall_steps) / 10.0, 1.0)], dtype=float),
        ])  # 8 dims

        vec_r1 = np.concatenate([
            spatial_mem,                # 44
            q / 1.5,                    # 5
            qdot / 1.5,                 # 5
            ee_vel / 0.10,              # 3
            prev_xyz / 0.004,           # 3
            self._ema_vel_fast / 0.10,  # 3
            self._ema_vel_slow / 0.10,  # 3
            delta_ee / 0.003,           # 3
            base_ctx,                   # 8
            err / 0.03,                 # 5 (clean zero-lag encoder tracking error)
            np.array([float(np.linalg.norm(err)) / 0.05, self._prev_g_spiral], dtype=float),  # 2
        ])  # 84 dims (ZERO actuator_force_noisy, ZERO d_tau_ext!)

        if self.mode == 'res_pos_only':
            return vec_r1
        return np.concatenate([vec_r1, self._prev_alpha])  # 86 dims


class Phase4ResidualPPO(RecurrentAsymmetricPPO):
    """Zero-initialized Residual GRU(64) + Spiral Gate `g_spiral` (+ Dynamic `GravComp-TL` for R2)."""

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
        init_log_std: float = -1.10,
    ):
        if mode not in ('res_pos_only', 'res_gravcomp_tl'):
            raise ValueError(f'Unknown Phase 4 mode {mode!r}')
        self.mode = mode
        self.base_alpha = float(base_alpha)
        super().__init__(
            obs_dim=obs_dim,
            critic_dim=critic_dim,
            use_torque_limit=(mode == 'res_gravcomp_tl'),
            hidden_dim=hidden_dim,
            critic_h1=critic_h1,
            critic_h2=critic_h2,
            seed=seed,
            init_log_std=init_log_std,
        )
        # Override act_dim cleanly: R1 = 4 ([dx, dy, dz, g_spiral]), R2 = 6 ([dx, dy, dz, g_spiral, a_lat, a_arm])
        self.act_dim = 6 if self.mode == 'res_gravcomp_tl' else 4
        H = self.hidden_dim
        self.W_mu = np.zeros((H, self.act_dim), dtype=float)
        self.b_mu = np.zeros(self.act_dim, dtype=float)
        init_ls = float(np.clip(init_log_std, LOG_STD_MIN, LOG_STD_MAX))
        self.log_std = np.full(self.act_dim, init_ls, dtype=float)
        self._m_actor = {k: np.zeros_like(v) for k, v in self.actor_params().items()}
        self._v_actor = {k: np.zeros_like(v) for k, v in self.actor_params().items()}

    def save(self, path) -> None:
        np.savez(
            path,
            obs_dim=self.obs_dim,
            mode=self.mode,
            base_alpha=self.base_alpha,
            critic_dim=self.critic_dim,
            use_torque_limit=int(self.use_torque_limit),
            hidden_dim=self.hidden_dim,
            critic_h1=self.critic_h1,
            critic_h2=self.critic_h2,
            **self.get_all_weights(),
        )

    @classmethod
    def load(cls, path) -> 'Phase4ResidualPPO':
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
        """Compose gated base-spiral + bounded residual motion + optional GravComp-TL `[alpha_lat, alpha_arm]`.

        At z = 0 (step-0 zero-init):
          - `g_spiral = 1.000` (exact base spiral!)
          - `res_dxyz = 0.000`
          - `R1` outputs exact `[base_dxyz, ts=1.0]` (bit-identical to `BlindSpiral_Stiff`)
          - `R2` outputs exact `{'delta_xyz': base_dxyz, 'gravcomp_alpha': [0.35, 0.35]}`
            (bit-identical to `BlindSpiral_GravComp_0.35`)
        """
        g_spiral = float(np.clip(1.0 + RES_GATE_GAIN * z[3], 0.0, 1.0))
        res_xy = np.clip(z[:2], -1.0, 1.0) * RES_MAX_XY_M
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

    def act_step_p4(
        self,
        x_raw: np.ndarray,
        h_prev: np.ndarray,
        base_dxyz: np.ndarray,
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
        env_act, d_xyz, g_spiral, alpha_pair = self.compose_action(base_dxyz, z)
        return env_act, d_xyz, g_spiral, alpha_pair, z, h_next, int(np.argmax(aux_cls)), aux_dir.copy()
