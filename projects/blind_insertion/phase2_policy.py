"""Phase 2 Recurrent GRU Asymmetric Actor-Critic PPO + High-Res Legal Spatial Memory.

Eliminates Phase 1.5's short-memory bottleneck (0.4 s CNN + coarse 4x4 10 mm grid) while
keeping the exact 3-way ablation ladder, strict observation boundary, asymmetric privileged
critic, dual auxiliary contact decoder, and exploit-free reward hygiene:

- M1 (`pos_only`):       SOTA Blind Precise Assembly (arXiv:2610.23849) upgraded with
                         episode-long GRU(64) + 8x8 (5 mm) legal visited-XY memory + dip memory.
                         Strictly ZERO actuator_force_noisy, ZERO tracking_error, ZERO torque_scale.
                         Fixed torque_scale = 1.0.
- M2 (`current`):        Identical GRU(64) + 8x8 spatial memory, adding actuator_force_noisy +
                         tracking_error + multi-scale legal EMAs. Fixed torque_scale = 1.0.
- M3 (`current_torque`): Identical GRU(64) + 8x8 spatial memory + current/tracking_error +
                         active 2-group STS3215 torque_scale [ts_lat, ts_arm] in [0.20, 1.0]
                         initialized at 0.75 (mapped to [ts_lat, ts_arm, ts_arm, ts_arm, ts_lat]).

Architecture:
- Per-step input encoder: `x_emb = tanh(x_norm @ W_in + b_in)` (D_in -> 64)
- Episode-long recurrent GRU cell carried across the full 5.0 s (250-step) episode:
    `h_{t+1} = GRU(x_emb_t, h_t)` (64 -> 64), trained with masked truncated-BPTT (L=32)
    using exact carried rollout initial hidden states `h_init`.
- Actor heads from `h_{t+1}`:
    * Policy mean `mu` (zero-initialized final layer, log_std clamped in [-3.0, -0.5])
    * Aux Contact-Class Decoder (5-class cross-entropy: free/plate/rim/entering/seated)
    * Aux Socket-Direction Decoder (2D unit vector MSE toward socket center)
- Asymmetric Privileged Critic `V(obs, priv)`: 2-layer tanh MLP (128 -> 96 -> 1), training-only.
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
from .phase1_policy import (
    CLOCK_PERIOD_S,
    CONTACT_CLASSES,
    CONTACT_CLASS_TO_IDX,
    FOURIER_FREQS,
    LOG_STD_MAX,
    LOG_STD_MIN,
    M3_TORQUE_MAX,
    M3_TORQUE_MIN,
    RunningMeanStd,
    clip_grad_norm,
    compute_gae,
)

GRID_N = 4
GRID_HALF_EXTENT_M = 0.020  # 4x4 compact visited-XY grid over [-20, +20] mm (high-SNR alongside GRU(64) memory)
NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
SPATIAL_MEM_DIM = 12 + 2 * len(FOURIER_FREQS) + GRID_N * GRID_N + len(NEIGHBOR_OFFSETS)  # 44

GRU_HIDDEN_DIM = 64
BPTT_CHUNK_LEN = 32
PRIVILEGED_CRITIC_DIM_P2 = 50


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    ex = np.exp(np.clip(shifted, -30.0, 30.0))
    return ex / np.sum(ex, axis=-1, keepdims=True)


class Phase2ProprioBuilder:
    """Builds controller-legal per-step feature vector with 8x8 (5 mm) spatial memory.

    Strictly enforces the observation boundary across all 3 methods:
    - Shared kinematic spatial memory (91 dims):
        * ee_rel (3), r_rel (1), cos/sin_th (2), z_min_rel (1), z_plate_drop (1),
          xy_to_z_min_dip (2), stall_frac (1) = 11 dims
        * t_frac + 4-harmonic fixed-5s Fourier clock -> wait: t_frac is inside the 11 dims?
          Let's keep exact accounting: 12 scalar/vec + 8 Fourier + 64 grid + 8 local compass = 92 dims.
    - M1 (`pos_only`):       92 + 25 = 117 dims (q, qdot, ee_vel, prev_xyz, fast/slow vel EMAs, delta_ee_pos)
                             STRICTLY ZERO actuator_force_noisy, ZERO tracking_error, ZERO torque_scale!
    - M2 (`current`):        117 + 27 = 144 dims (+ tracking_error, f_noisy, df_free, fast/slow df EMAs, norms)
    - M3 (`current_torque`): 144 + 5 = 149 dims (+ obs['torque_scale'][:5])
    """

    def __init__(self, mode: str):
        if mode not in ('pos_only', 'current', 'current_torque'):
            raise ValueError(f'Unknown mode {mode!r}')
        self.mode = mode
        self.reset()

    @property
    def obs_dim(self) -> int:
        if self.mode == 'pos_only':
            return SPATIAL_MEM_DIM + 25  # 69
        if self.mode == 'current':
            return SPATIAL_MEM_DIM + 52  # 96
        return SPATIAL_MEM_DIM + 57      # 101

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
        self._f_free: np.ndarray | None = None
        self._ema_df_fast = np.zeros(5, dtype=float)
        self._ema_df_slow = np.zeros(5, dtype=float)
        self.last_obs_keys_seen: set[str] = set()

    @property
    def visited_fraction(self) -> float:
        return float(np.mean(self._grid > 0.0))

    def step(self, obs: dict) -> np.ndarray:
        """Return 1D feature vector x_t strictly from controller-legal obs."""
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

        # Kinematic touchdown estimate (purely from ee_pose_fk / ee_twist_fk — 100% M1-legal)
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

        # 8x8 (5.0 mm) visited-XY grid over [-20, +20] mm + 8-neighbor unvisited compass
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

        # Multi-scale kinematic velocity EMAs (0.06 s & 0.33 s)
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
            ], dtype=float),  # 12
            fourier,          # 8
            self._grid.reshape(-1),  # 64
            compass,          # 8
        ])  # 92 dims

        # M1 core vector (117 dims): STRICTLY zero actuator_force, zero tracking_error, zero torque_scale!
        vec_m1 = np.concatenate([
            spatial_mem,                  # 92
            q / 1.5,                      # 5
            qdot / 1.5,                   # 5
            ee_vel / 0.10,                # 3
            prev_xyz / 0.004,             # 3
            self._ema_vel_fast / 0.10,    # 3
            self._ema_vel_slow / 0.10,    # 3
            delta_ee / 0.003,             # 3
        ])
        if self.mode == 'pos_only':
            return vec_m1

        err = np.asarray(obs['tracking_error'][:5], dtype=float)
        f_noisy = np.asarray(obs['actuator_force_noisy'][:5], dtype=float)
        if self._f_free is None:
            self._f_free = f_noisy.copy()
        df = f_noisy - self._f_free
        self._ema_df_fast = 0.35 * df + 0.65 * self._ema_df_fast
        self._ema_df_slow = 0.06 * df + 0.94 * self._ema_df_slow

        vec_m2 = np.concatenate([
            vec_m1,                        # 117
            err / 0.03,                    # 5
            f_noisy / 1.5,                 # 5
            df / 1.0,                      # 5
            self._ema_df_fast / 1.0,       # 5
            self._ema_df_slow / 1.0,       # 5
            np.array([float(np.linalg.norm(df)) / 1.5, float(np.linalg.norm(err)) / 0.05], dtype=float),  # 2
        ])  # 144 dims
        if self.mode == 'current':
            return vec_m2

        ts = np.asarray(obs['torque_scale'][:5], dtype=float)
        return np.concatenate([vec_m2, ts])  # 149 dims


def build_privileged_critic_features_p2(
    obs: dict,
    priv: dict,
    start_xyz: np.ndarray,
    visited_frac: float,
) -> np.ndarray:
    """Training-only asymmetric critic vector V(obs, priv) — 50 dims, discarded at test time."""
    q = np.asarray(obs['q_quantized'][:5], dtype=float) / 1.5
    qdot = np.asarray(obs['q_dot_quantized'][:5], dtype=float) / 1.5
    err = np.asarray(obs['tracking_error'][:5], dtype=float) / 0.03
    f = np.asarray(obs['actuator_force_noisy'][:5], dtype=float) / 1.5
    ts = np.asarray(obs['torque_scale'][:5], dtype=float)
    ee_rel = (np.asarray(obs['ee_pose_fk'][:3], dtype=float) - start_xyz) / 0.02
    ee_vel = np.asarray(obs['ee_twist_fk'][:3], dtype=float) / 0.10

    xy_err = np.asarray(priv['socket_xy_error'], dtype=float)
    rad_err = float(np.linalg.norm(xy_err))
    c_oh = np.zeros(len(CONTACT_CLASSES), dtype=float)
    c_oh[CONTACT_CLASS_TO_IDX[priv['contact_class']]] = 1.0

    priv_vec = np.array([
        xy_err[0] / 0.015,
        xy_err[1] / 0.015,
        rad_err / 0.015,
        float(priv['peg_depth']) / 0.015,
        float(priv['peak_contact_force_N']) / 50.0,
        float(priv['socket_displacement_mm']) / 25.0,
        float(priv['socket_tilt_deg']) / 15.0,
        1.0 if priv['bolted'] else 0.0,
        float(priv['clearance_m']) / 0.0015,
        float(priv['mu_peg_hole']) / 0.5,
        float(priv['mu_table']) / 0.6,
        float(priv['socket_mass_kg']) / 0.35,
        float(obs['time_s']) / CLOCK_PERIOD_S,
        float(visited_frac),
    ], dtype=float)
    return np.concatenate([q, qdot, err, f, ts, ee_rel, ee_vel, c_oh, priv_vec])


def _ortho_init(rng: np.random.Generator, rows: int, cols: int, gain: float = 1.0) -> np.ndarray:
    """Orthogonal matrix init (standard Recurrent PPO hygiene so GRU spectral radius starts at 1.0)."""
    a = rng.normal(0.0, 1.0, size=(rows, cols))
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == (rows, cols) else vt
    return gain * q.reshape(rows, cols)


class RecurrentAsymmetricPPO:
    """Episode-long GRU(64) Actor + Dual Aux Decoder + Asymmetric Privileged Critic."""

    def __init__(
        self,
        obs_dim: int,
        critic_dim: int = PRIVILEGED_CRITIC_DIM_P2,
        use_torque_limit: bool = False,
        hidden_dim: int = GRU_HIDDEN_DIM,
        critic_h1: int = 128,
        critic_h2: int = 96,
        seed: int = 0,
        init_log_std: float = -0.65,
    ):
        self.obs_dim = int(obs_dim)
        self.critic_dim = int(critic_dim)
        self.use_torque_limit = bool(use_torque_limit)
        # M1/M2: 3 dims [dx, dy, dz]; M3: 5 dims [dx, dy, dz, ts_lat, ts_arm]
        self.act_dim = 5 if self.use_torque_limit else 3
        self.hidden_dim = int(hidden_dim)
        self.critic_h1 = int(critic_h1)
        self.critic_h2 = int(critic_h2)
        self.n_contact_classes = len(CONTACT_CLASSES)

        rng = np.random.default_rng(seed)
        H = self.hidden_dim

        # Input encoder: D_in -> H (tanh), orthogonal init
        self.W_in = _ortho_init(rng, self.obs_dim, H, gain=1.0)
        self.b_in = np.zeros(H, dtype=float)

        # GRU cell: per-gate orthogonal blocks so initial spectral norm of W_h_r/z/n is exactly 0.90!
        self.W_x = np.concatenate([_ortho_init(rng, H, H, gain=0.9) for _ in range(3)], axis=1)
        self.b_x = np.zeros(3 * H, dtype=float)
        self.W_h = np.concatenate([_ortho_init(rng, H, H, gain=0.9) for _ in range(3)], axis=1)
        self.b_h = np.zeros(3 * H, dtype=float)
        # Slight positive bias on update gate u (retain memory across multi-second sweeps!)
        self.b_h[H:2 * H] = 0.5

        # Policy mean head: STRICTLY ZERO-INITIALIZED!
        self.W_mu = np.zeros((H, self.act_dim), dtype=float)
        self.b_mu = np.zeros(self.act_dim, dtype=float)
        init_ls = float(np.clip(init_log_std, LOG_STD_MIN, LOG_STD_MAX))
        self.log_std = np.full(self.act_dim, init_ls, dtype=float)

        # Training-only auxiliary decoders from recurrent latent h_t
        self.W_aux_cls = rng.normal(0.0, 0.05, size=(H, self.n_contact_classes))
        self.b_aux_cls = np.zeros(self.n_contact_classes, dtype=float)
        self.W_aux_dir = rng.normal(0.0, 0.05, size=(H, 2))
        self.b_aux_dir = np.zeros(2, dtype=float)

        # Asymmetric Privileged Critic V(obs, priv)
        self.C_W1 = rng.normal(0.0, np.sqrt(1.0 / self.critic_dim), size=(self.critic_dim, self.critic_h1))
        self.C_b1 = np.zeros(self.critic_h1, dtype=float)
        self.C_W2 = rng.normal(0.0, np.sqrt(1.0 / self.critic_h1), size=(self.critic_h1, self.critic_h2))
        self.C_b2 = np.zeros(self.critic_h2, dtype=float)
        self.C_Wv = rng.normal(0.0, 0.01, size=(self.critic_h2, 1))
        self.C_bv = np.zeros(1, dtype=float)

        # Running normalizers (obs_rms saved in ckpt, frozen in eval; critic_rms training-only)
        self.obs_rms = RunningMeanStd(self.obs_dim)
        self.critic_rms = RunningMeanStd(self.critic_dim)

        self._m_actor = {k: np.zeros_like(v) for k, v in self.actor_params().items()}
        self._v_actor = {k: np.zeros_like(v) for k, v in self.actor_params().items()}
        self._t_actor = 0
        self._m_critic = {k: np.zeros_like(v) for k, v in self.critic_params().items()}
        self._v_critic = {k: np.zeros_like(v) for k, v in self.critic_params().items()}
        self._t_critic = 0

    def actor_params(self) -> dict[str, np.ndarray]:
        return {
            'W_in': self.W_in, 'b_in': self.b_in,
            'W_x': self.W_x, 'b_x': self.b_x,
            'W_h': self.W_h, 'b_h': self.b_h,
            'W_mu': self.W_mu, 'b_mu': self.b_mu,
            'W_aux_cls': self.W_aux_cls, 'b_aux_cls': self.b_aux_cls,
            'W_aux_dir': self.W_aux_dir, 'b_aux_dir': self.b_aux_dir,
            'log_std': self.log_std,
        }

    def critic_params(self) -> dict[str, np.ndarray]:
        return {
            'C_W1': self.C_W1, 'C_b1': self.C_b1,
            'C_W2': self.C_W2, 'C_b2': self.C_b2,
            'C_Wv': self.C_Wv, 'C_bv': self.C_bv,
        }

    @property
    def actor_param_count(self) -> int:
        return int(sum(v.size for v in self.actor_params().values()))

    @property
    def critic_param_count(self) -> int:
        return int(sum(v.size for v in self.critic_params().values()))

    def init_hidden(self) -> np.ndarray:
        return np.zeros(self.hidden_dim, dtype=float)

    def get_actor_weights(self) -> dict[str, np.ndarray]:
        out = {k: v.copy() for k, v in self.actor_params().items()}
        out.update(self.obs_rms.get_state('obs_rms'))
        return out

    def get_all_weights(self) -> dict[str, np.ndarray]:
        out = self.get_actor_weights()
        out.update({k: v.copy() for k, v in self.critic_params().items()})
        out.update(self.critic_rms.get_state('critic_rms'))
        return out

    def set_weights(self, w: dict[str, np.ndarray]) -> None:
        for k in ('W_in', 'b_in', 'W_x', 'b_x', 'W_h', 'b_h', 'W_mu', 'b_mu', 'W_aux_cls', 'b_aux_cls', 'W_aux_dir', 'b_aux_dir'):
            setattr(self, k, np.asarray(w[k], dtype=float).copy())
        self.log_std = np.clip(np.asarray(w['log_std'], dtype=float).copy(), LOG_STD_MIN, LOG_STD_MAX)
        if 'obs_rms_mean' in w:
            self.obs_rms.set_state(w, 'obs_rms')
        if 'C_W1' in w:
            for k in ('C_W1', 'C_b1', 'C_W2', 'C_b2', 'C_Wv', 'C_bv'):
                setattr(self, k, np.asarray(w[k], dtype=float).copy())
        if 'critic_rms_mean' in w:
            self.critic_rms.set_state(w, 'critic_rms')

    def save(self, path) -> None:
        np.savez(
            path,
            obs_dim=self.obs_dim,
            critic_dim=self.critic_dim,
            use_torque_limit=int(self.use_torque_limit),
            hidden_dim=self.hidden_dim,
            critic_h1=self.critic_h1,
            critic_h2=self.critic_h2,
            **self.get_all_weights(),
        )

    @classmethod
    def load(cls, path) -> 'RecurrentAsymmetricPPO':
        data = np.load(path)
        pol = cls(
            obs_dim=int(data['obs_dim']),
            critic_dim=int(data['critic_dim']),
            use_torque_limit=bool(int(data['use_torque_limit'])),
            hidden_dim=int(data['hidden_dim']),
            critic_h1=int(data['critic_h1']),
            critic_h2=int(data['critic_h2']),
        )
        meta = {'obs_dim', 'critic_dim', 'use_torque_limit', 'hidden_dim', 'critic_h1', 'critic_h2'}
        pol.set_weights({k: data[k] for k in data.files if k not in meta})
        return pol

    def map_z_to_env_action(self, z: np.ndarray) -> np.ndarray:
        """Map pre-action z -> [dx, dy, dz, ts_0..ts_4].

        M3 maps 2 coupled STS3215 torque-limit groups [ts_lat, ts_arm] initialized at 0.75:
          - ts_lat -> shoulder_pan (0) and wrist_roll (4) [zero gravity load, pure lateral/yaw compliance]
          - ts_arm -> shoulder_lift (1), elbow_flex (2), wrist_flex (3) [planar reach + gravity support]
        """
        d_xy = np.clip(z[:2], -1.0, 1.0) * MAX_TASK_DELTA_XY_M
        d_z = float(np.clip(z[2], -1.0, 0.5)) * MAX_TASK_DELTA_Z_M
        if not self.use_torque_limit:
            ts = np.ones(5, dtype=float)
        else:
            ts_lat = float(np.clip(0.75 + 0.40 * z[3], M3_TORQUE_MIN, M3_TORQUE_MAX))
            ts_arm = float(np.clip(0.75 + 0.40 * z[4], M3_TORQUE_MIN, M3_TORQUE_MAX))
            ts = np.array([ts_lat, ts_arm, ts_arm, ts_arm, ts_lat], dtype=float)
        return np.concatenate([d_xy, [d_z], ts])

    def step_gru_single(self, x_norm: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Single-step GRU update for fast rollouts."""
        H = self.hidden_dim
        x_emb = np.tanh(x_norm @ self.W_in + self.b_in)
        pre_x = x_emb @ self.W_x + self.b_x
        pre_h = h_prev @ self.W_h + self.b_h
        r = _sigmoid(pre_x[:H] + pre_h[:H])
        u = _sigmoid(pre_x[H:2 * H] + pre_h[H:2 * H])
        n = np.tanh(pre_x[2 * H:] + r * pre_h[2 * H:])
        return (1.0 - u) * n + u * h_prev

    def act_step(
        self,
        x_raw: np.ndarray,
        h_prev: np.ndarray,
        deterministic: bool = True,
        eps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
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
        return self.map_z_to_env_action(z), z, h_next, int(np.argmax(aux_cls)), aux_dir.copy()

    def forward_actor_seq(self, X_norm: np.ndarray, h0: np.ndarray):
        """Vectorized sequence forward over (B, L, D_in) starting from h0 (B, H)."""
        B, L, _ = X_norm.shape
        H = self.hidden_dim
        X_emb = np.tanh(X_norm @ self.W_in + self.b_in)    # (B, L, H)
        pre_x = X_emb @ self.W_x + self.b_x                # (B, L, 3H)

        H_prev = np.empty((B, L, H), dtype=float)
        H_out = np.empty((B, L, H), dtype=float)
        R_arr = np.empty((B, L, H), dtype=float)
        U_arr = np.empty((B, L, H), dtype=float)
        N_arr = np.empty((B, L, H), dtype=float)
        Hn_arr = np.empty((B, L, H), dtype=float)

        h = h0.copy()
        for t in range(L):
            H_prev[:, t, :] = h
            pre_h = h @ self.W_h + self.b_h
            px_t = pre_x[:, t, :]
            r = _sigmoid(px_t[:, :H] + pre_h[:, :H])
            u = _sigmoid(px_t[:, H:2 * H] + pre_h[:, H:2 * H])
            h_n = pre_h[:, 2 * H:]
            n = np.tanh(px_t[:, 2 * H:] + r * h_n)
            h = (1.0 - u) * n + u * h

            R_arr[:, t, :] = r
            U_arr[:, t, :] = u
            N_arr[:, t, :] = n
            Hn_arr[:, t, :] = h_n
            H_out[:, t, :] = h

        mu = H_out @ self.W_mu + self.b_mu
        aux_cls = H_out @ self.W_aux_cls + self.b_aux_cls
        aux_dir = H_out @ self.W_aux_dir + self.b_aux_dir
        cache = (X_norm, X_emb, H_prev, H_out, R_arr, U_arr, N_arr, Hn_arr)
        return mu, aux_cls, aux_dir, H_out, cache

    def forward_critic(self, C_norm: np.ndarray):
        h1 = np.tanh(C_norm @ self.C_W1 + self.C_b1)
        h2 = np.tanh(h1 @ self.C_W2 + self.C_b2)
        v = (h2 @ self.C_Wv + self.C_bv).reshape(-1)
        return v, h1, h2

    def seq_log_prob(self, X_norm: np.ndarray, h0: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact per-step log-prob and H_out for sequence batch (B, L, ...)."""
        mu, _, _, H_out, _ = self.forward_actor_seq(X_norm, h0)
        log_std = np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        inv_var = np.exp(-2.0 * log_std)
        diff = Z - mu
        lp = -0.5 * np.sum((diff ** 2) * inv_var + 2.0 * log_std + np.log(2.0 * np.pi), axis=-1)
        return lp, H_out

    def actor_and_aux_seq_loss_and_grad(
        self,
        X_norm: np.ndarray,       # (B, L, D_in)
        h0: np.ndarray,           # (B, H)
        mask: np.ndarray,         # (B, L) in {0.0, 1.0}
        Z: np.ndarray,            # (B, L, act_dim)
        old_logp: np.ndarray,     # (B, L)
        adv: np.ndarray,          # (B, L)
        cls_labels: np.ndarray,   # (B, L) int64
        dir_targets: np.ndarray,  # (B, L, 2)
        dir_mask: np.ndarray,     # (B, L)
        clip_eps: float = 0.2,
        entropy_coef: float = 0.001,
        aux_cls_coef: float = 0.08,
        aux_dir_coef: float = 0.08,
        mu_sat_coef: float = 0.002,
        weight_decay: float = 1e-4,
    ):
        B, L, _ = X_norm.shape
        H = self.hidden_dim
        mu, aux_cls, aux_dir, H_out, cache = self.forward_actor_seq(X_norm, h0)
        _, X_emb, H_prev, _, R_arr, U_arr, N_arr, Hn_arr = cache

        m_sum = max(float(np.sum(mask)), 1.0)
        log_std = np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        inv_var = np.exp(-2.0 * log_std)
        diff = Z - mu
        logp = -0.5 * np.sum((diff ** 2) * inv_var + 2.0 * log_std + np.log(2.0 * np.pi), axis=-1)

        log_ratio = np.clip(logp - old_logp, -10.0, 10.0)
        ratio = np.exp(log_ratio)
        clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surr1 = ratio * adv
        surr2 = clipped * adv
        use_unclipped = (surr1 <= surr2)

        clip_frac = float(np.sum((~use_unclipped) * mask) / m_sum)
        approx_kl = float(np.sum(((ratio - 1.0) - log_ratio) * mask) / m_sum)

        # Aux (a): 5-class contact_class cross-entropy
        probs = _softmax(aux_cls)  # (B, L, 5)
        p_true = np.take_along_axis(probs, cls_labels[:, :, None], axis=-1).squeeze(-1)
        aux_ce = -float(np.sum(mask * np.log(np.clip(p_true, 1e-12, 1.0))) / m_sum)
        aux_acc = float(np.sum(mask * (np.argmax(aux_cls, axis=-1) == cls_labels)) / m_sum)

        # Aux (b): 2D unit direction MSE
        d_eff_mask = mask * dir_mask
        dm_sum = max(float(np.sum(d_eff_mask)), 1.0)
        dir_err = aux_dir - dir_targets
        aux_dir_mse = 0.5 * float(np.sum(d_eff_mask[:, :, None] * (dir_err ** 2)) / dm_sum)

        # Smooth C1 deadzone penalty outside [-1.5, +1.5] so full +-4 mm search is uninhibited while preventing runaway
        mu_over = np.maximum(np.abs(mu) - 1.5, 0.0)
        mu_sat_loss = 0.5 * mu_sat_coef * float(np.sum(mask[:, :, None] * (mu_over ** 2)) / m_sum)

        entropy = float(np.sum(log_std + 0.5 * (1.0 + np.log(2.0 * np.pi))))
        wd_loss = 0.5 * weight_decay * (
            float(np.sum(self.W_in ** 2))
            + float(np.sum(self.W_x ** 2))
            + float(np.sum(self.W_h ** 2))
            + float(np.sum(self.W_mu ** 2))
            + float(np.sum(self.W_aux_cls ** 2))
            + float(np.sum(self.W_aux_dir ** 2))
        )
        loss = (
            -float(np.sum(mask * np.minimum(surr1, surr2)) / m_sum)
            - entropy_coef * entropy
            + aux_cls_coef * aux_ce
            + aux_dir_coef * aux_dir_mse
            + mu_sat_loss
            + wd_loss
        )

        # Head gradients over (B, L, ...)
        d_logp = np.where(use_unclipped, -(ratio * adv) * mask / m_sum, 0.0)  # (B, L)
        d_mu = d_logp[:, :, None] * (diff * inv_var) + (mu_sat_coef / m_sum) * mask[:, :, None] * np.sign(mu) * mu_over
        d_log_std = np.sum(d_logp[:, :, None] * ((diff ** 2) * inv_var - 1.0), axis=(0, 1)) - entropy_coef * np.ones(self.act_dim)

        d_cls = probs.copy()
        bi = np.arange(B)[:, None]
        li = np.arange(L)[None, :]
        d_cls[bi, li, cls_labels] -= 1.0
        d_cls *= (aux_cls_coef / m_sum) * mask[:, :, None]

        d_dir = (aux_dir_coef / dm_sum) * (d_eff_mask[:, :, None] * dir_err)

        H_out_2d = H_out.reshape(B * L, H)
        d_mu_2d = d_mu.reshape(B * L, self.act_dim)
        d_cls_2d = d_cls.reshape(B * L, self.n_contact_classes)
        d_dir_2d = d_dir.reshape(B * L, 2)

        d_W_mu = H_out_2d.T @ d_mu_2d + weight_decay * self.W_mu
        d_b_mu = np.sum(d_mu_2d, axis=0)
        d_W_aux_cls = H_out_2d.T @ d_cls_2d + weight_decay * self.W_aux_cls
        d_b_aux_cls = np.sum(d_cls_2d, axis=0)
        d_W_aux_dir = H_out_2d.T @ d_dir_2d + weight_decay * self.W_aux_dir
        d_b_aux_dir = np.sum(d_dir_2d, axis=0)

        # Backprop through time (BPTT) across L steps
        d_H_out = (d_mu @ self.W_mu.T) + (d_cls @ self.W_aux_cls.T) + (d_dir @ self.W_aux_dir.T)  # (B, L, H)
        d_pre_x = np.empty((B, L, 3 * H), dtype=float)
        d_pre_h = np.empty((B, L, 3 * H), dtype=float)
        d_h_next = np.zeros((B, H), dtype=float)

        W_h_T = self.W_h.T
        for t in range(L - 1, -1, -1):
            d_h_tot = d_H_out[:, t, :] + d_h_next
            r = R_arr[:, t, :]
            u = U_arr[:, t, :]
            n = N_arr[:, t, :]
            h_n = Hn_arr[:, t, :]
            h_prev = H_prev[:, t, :]

            d_n = d_h_tot * (1.0 - u)
            d_u = d_h_tot * (h_prev - n)
            d_h_prev = d_h_tot * u

            d_pre_n = d_n * (1.0 - n * n)
            d_h_n = d_pre_n * r
            d_r = d_pre_n * h_n
            d_pre_r = d_r * r * (1.0 - r)
            d_pre_z = d_u * u * (1.0 - u)

            d_pre_x[:, t, :H] = d_pre_r
            d_pre_x[:, t, H:2 * H] = d_pre_z
            d_pre_x[:, t, 2 * H:] = d_pre_n

            d_pre_h[:, t, :H] = d_pre_r
            d_pre_h[:, t, H:2 * H] = d_pre_z
            d_pre_h[:, t, 2 * H:] = d_h_n

            d_h_next = d_h_prev + (d_pre_h[:, t, :] @ W_h_T)

        H_prev_2d = H_prev.reshape(B * L, H)
        X_emb_2d = X_emb.reshape(B * L, H)
        X_norm_2d = X_norm.reshape(B * L, self.obs_dim)
        d_pre_x_2d = d_pre_x.reshape(B * L, 3 * H)
        d_pre_h_2d = d_pre_h.reshape(B * L, 3 * H)

        d_W_h = H_prev_2d.T @ d_pre_h_2d + weight_decay * self.W_h
        d_b_h = np.sum(d_pre_h_2d, axis=0)
        d_W_x = X_emb_2d.T @ d_pre_x_2d + weight_decay * self.W_x
        d_b_x = np.sum(d_pre_x_2d, axis=0)

        d_x_emb = d_pre_x_2d @ self.W_x.T
        d_pre_in = d_x_emb * (1.0 - X_emb_2d * X_emb_2d)
        d_W_in = X_norm_2d.T @ d_pre_in + weight_decay * self.W_in
        d_b_in = np.sum(d_pre_in, axis=0)

        grads = {
            'W_in': d_W_in, 'b_in': d_b_in,
            'W_x': d_W_x, 'b_x': d_b_x,
            'W_h': d_W_h, 'b_h': d_b_h,
            'W_mu': d_W_mu, 'b_mu': d_b_mu,
            'W_aux_cls': d_W_aux_cls, 'b_aux_cls': d_b_aux_cls,
            'W_aux_dir': d_W_aux_dir, 'b_aux_dir': d_b_aux_dir,
            'log_std': d_log_std,
        }
        return loss, grads, clip_frac, approx_kl, aux_ce, aux_acc, aux_dir_mse

    def critic_loss_and_grad(
        self,
        C_norm: np.ndarray,
        old_v: np.ndarray,
        ret: np.ndarray,
        clip_eps: float = 0.2,
        weight_decay: float = 1e-4,
    ):
        B = len(C_norm)
        v_pred, h1, h2 = self.forward_critic(C_norm)
        v_clipped = old_v + np.clip(v_pred - old_v, -clip_eps, clip_eps)
        err_u = v_pred - ret
        err_c = v_clipped - ret
        sq_u = err_u ** 2
        sq_c = err_c ** 2

        wd_loss = 0.5 * weight_decay * (
            float(np.sum(self.C_W1 ** 2)) + float(np.sum(self.C_W2 ** 2)) + float(np.sum(self.C_Wv ** 2))
        )
        loss = 0.5 * float(np.mean(np.maximum(sq_u, sq_c))) + wd_loss

        in_band = np.abs(v_pred - old_v) <= clip_eps
        d_v = np.where(sq_u >= sq_c, err_u, np.where(in_band, err_c, 0.0)) / B
        d_v2 = d_v[:, None]

        d_C_Wv = h2.T @ d_v2 + weight_decay * self.C_Wv
        d_C_bv = np.sum(d_v2, axis=0)
        d_h2 = d_v2 @ self.C_Wv.T
        d_pre2 = d_h2 * (1.0 - h2 ** 2)
        d_C_W2 = h1.T @ d_pre2 + weight_decay * self.C_W2
        d_C_b2 = np.sum(d_pre2, axis=0)
        d_h1 = d_pre2 @ self.C_W2.T
        d_pre1 = d_h1 * (1.0 - h1 ** 2)
        d_C_W1 = C_norm.T @ d_pre1 + weight_decay * self.C_W1
        d_C_b1 = np.sum(d_pre1, axis=0)

        grads = {
            'C_W1': d_C_W1, 'C_b1': d_C_b1,
            'C_W2': d_C_W2, 'C_b2': d_C_b2,
            'C_Wv': d_C_Wv, 'C_bv': d_C_bv,
        }
        return loss, grads

    @staticmethod
    def _adam_step(params, grads, m, v, t: int, lr: float) -> int:
        t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k, p in params.items():
            g = grads[k]
            m[k] = beta1 * m[k] + (1.0 - beta1) * g
            v[k] = beta2 * v[k] + (1.0 - beta2) * (g ** 2)
            m_hat = m[k] / (1.0 - beta1 ** t)
            v_hat = v[k] / (1.0 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return t

    def ppo_update(
        self,
        trajectories: list[dict],
        rng: np.random.Generator,
        actor_lr: float = 8e-4,
        critic_lr: float = 1.8e-3,
        gamma: float = 0.995,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        chunk_len: int = BPTT_CHUNK_LEN,
        chunk_minibatch: int = 16,
        critic_minibatch: int = 256,
        entropy_coef: float = 0.001,
        aux_cls_coef: float = 0.15,
        aux_dir_coef: float = 0.15,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.025,
        weight_decay: float = 3e-4,
        log_std_max: float = LOG_STD_MAX,
    ) -> dict:
        valid = [tr for tr in trajectories if len(tr['obs']) > 0]
        if not valid:
            return {'n_steps': 0, 'mean_return': 0.0, 'std': np.exp(self.log_std).tolist()}

        X_raw_l, C_raw_l = [], []
        C_n_l, old_v_l, ret_l = [], [], []
        ep_chunks_meta = []
        all_adv_flat = []
        ep_returns = []

        # 1. Exact full-episode forward pass -> GAE + exact carried h_init for every chunk
        for tr in valid:
            X_raw = np.asarray(tr['obs'], dtype=float)          # (T, D_in)
            C_raw = np.asarray(tr['critic_obs'], dtype=float)   # (T, 50)
            Z = np.asarray(tr['z'], dtype=float)                # (T, act_dim)
            R = np.asarray(tr['rewards'], dtype=float)          # (T,)
            CLS = np.asarray(tr['contact_labels'], dtype=np.int64)
            DIR = np.asarray(tr['dir_targets'], dtype=float)
            DMASK = np.asarray(tr['dir_mask'], dtype=float)

            X_n = self.obs_rms.normalize(X_raw)
            C_n = self.critic_rms.normalize(C_raw)
            v_old, _, _ = self.forward_critic(C_n)
            h0_ep = np.zeros((1, self.hidden_dim), dtype=float)
            lp_old_2d, H_out_3d = self.seq_log_prob(X_n[None, :, :], h0_ep, Z[None, :, :])
            lp_old = lp_old_2d[0]
            H_out = H_out_3d[0]  # (T, H)

            last_v = 0.0
            if not tr['terminated'] and 'last_critic_obs' in tr:
                last_c = self.critic_rms.normalize(tr['last_critic_obs'])[None, :]
                last_v = float(self.forward_critic(last_c)[0][0])
            adv_ep, ret_ep = compute_gae(R, v_old, terminated=bool(tr['terminated']), last_value=last_v, gamma=gamma, lam=lam)

            X_raw_l.append(X_raw)
            C_raw_l.append(C_raw)
            C_n_l.append(C_n)
            old_v_l.append(v_old)
            ret_l.append(ret_ep)
            all_adv_flat.append(adv_ep)
            ep_returns.append(float(np.sum(R)))
            ep_chunks_meta.append((X_n, H_out, Z, lp_old, adv_ep, CLS, DIR, DMASK))

        C_all = np.concatenate(C_n_l, axis=0)
        old_v_all = np.concatenate(old_v_l, axis=0)
        ret_all = np.concatenate(ret_l, axis=0)
        adv_flat = np.concatenate(all_adv_flat, axis=0)
        N_steps = len(adv_flat)

        var_y = float(np.var(ret_all))
        explained_var = float(1.0 - np.var(ret_all - old_v_all) / (var_y + 1e-8)) if var_y > 1e-8 else 0.0

        adv_mean = float(np.mean(adv_flat))
        adv_std = float(np.std(adv_flat))
        adv_inv = (1.0 / adv_std) if adv_std > 1e-8 else 1.0

        # 2. Slice episodes into masked L=chunk_len BPTT chunks with exact carried h_init
        c_X, c_h0, c_m, c_Z, c_lp, c_adv, c_cls, c_dir, c_dm = [], [], [], [], [], [], [], [], []
        for X_n, H_out, Z, lp_old, adv_ep, CLS, DIR, DMASK in ep_chunks_meta:
            T = len(X_n)
            adv_norm = (adv_ep - adv_mean) * adv_inv
            for s in range(0, T, chunk_len):
                e = min(s + chunk_len, T)
                k = e - s
                h_init = np.zeros(self.hidden_dim, dtype=float) if s == 0 else H_out[s - 1].copy()

                x_pad = np.zeros((chunk_len, self.obs_dim), dtype=float)
                m_pad = np.zeros(chunk_len, dtype=float)
                z_pad = np.zeros((chunk_len, self.act_dim), dtype=float)
                lp_pad = np.zeros(chunk_len, dtype=float)
                adv_pad = np.zeros(chunk_len, dtype=float)
                cls_pad = np.zeros(chunk_len, dtype=np.int64)
                dir_pad = np.zeros((chunk_len, 2), dtype=float)
                dm_pad = np.zeros(chunk_len, dtype=float)

                x_pad[:k] = X_n[s:e]
                m_pad[:k] = 1.0
                z_pad[:k] = Z[s:e]
                lp_pad[:k] = lp_old[s:e]
                adv_pad[:k] = adv_norm[s:e]
                cls_pad[:k] = CLS[s:e]
                dir_pad[:k] = DIR[s:e]
                dm_pad[:k] = DMASK[s:e]

                c_X.append(x_pad)
                c_h0.append(h_init)
                c_m.append(m_pad)
                c_Z.append(z_pad)
                c_lp.append(lp_pad)
                c_adv.append(adv_pad)
                c_cls.append(cls_pad)
                c_dir.append(dir_pad)
                c_dm.append(dm_pad)

        CH_X = np.asarray(c_X, dtype=float)
        CH_h0 = np.asarray(c_h0, dtype=float)
        CH_m = np.asarray(c_m, dtype=float)
        CH_Z = np.asarray(c_Z, dtype=float)
        CH_lp = np.asarray(c_lp, dtype=float)
        CH_adv = np.asarray(c_adv, dtype=float)
        CH_cls = np.asarray(c_cls, dtype=np.int64)
        CH_dir = np.asarray(c_dir, dtype=float)
        CH_dm = np.asarray(c_dm, dtype=float)
        n_chunks = len(CH_X)

        clip_fracs, kls, v_losses, aux_accs, aux_dirs = [], [], [], [], []
        early_stopped = False

        for _ in range(epochs):
            # Recurrent actor + aux BPTT minibatches with true post-step KL trust-region rollback
            c_perm = rng.permutation(n_chunks)
            ep_kls = []
            for start in range(0, n_chunks, chunk_minibatch):
                idx = c_perm[start:start + chunk_minibatch]
                if len(idx) == 0:
                    continue
                snapshot = {k: v.copy() for k, v in self.actor_params().items()}
                _, a_grads, cf, kl, _, aux_acc, aux_dir_mse = self.actor_and_aux_seq_loss_and_grad(
                    CH_X[idx], CH_h0[idx], CH_m[idx], CH_Z[idx], CH_lp[idx], CH_adv[idx],
                    CH_cls[idx], CH_dir[idx], CH_dm[idx],
                    clip_eps=clip_eps, entropy_coef=entropy_coef,
                    aux_cls_coef=aux_cls_coef, aux_dir_coef=aux_dir_coef, weight_decay=weight_decay,
                )
                clip_grad_norm(a_grads, max_norm=max_grad_norm)
                self._t_actor = self._adam_step(self.actor_params(), a_grads, self._m_actor, self._v_actor, self._t_actor, lr=actor_lr)
                self.log_std[:] = np.clip(self.log_std, LOG_STD_MIN, min(LOG_STD_MAX, float(log_std_max)))

                # Verify post-step KL on this minibatch; rollback immediately if trust region violated!
                lp_new, _ = self.seq_log_prob(CH_X[idx], CH_h0[idx], CH_Z[idx])
                lr_post = np.clip(lp_new - CH_lp[idx], -10.0, 10.0)
                m_b = CH_m[idx]
                kl_post = float(np.sum(((np.exp(lr_post) - 1.0) - lr_post) * m_b) / max(float(np.sum(m_b)), 1.0))
                if kl_post > 1.8 * target_kl:
                    for k, v in snapshot.items():
                        self.actor_params()[k][:] = v
                    early_stopped = True
                    break

                clip_fracs.append(cf)
                kls.append(kl_post)
                ep_kls.append(kl_post)
                aux_accs.append(aux_acc)
                aux_dirs.append(aux_dir_mse)

            # Asymmetric privileged critic minibatches
            s_perm = rng.permutation(N_steps)
            for start in range(0, N_steps, critic_minibatch):
                idx = s_perm[start:start + critic_minibatch]
                if len(idx) < 16:
                    continue
                v_loss, c_grads = self.critic_loss_and_grad(C_all[idx], old_v_all[idx], ret_all[idx], clip_eps=clip_eps, weight_decay=weight_decay)
                clip_grad_norm(c_grads, max_norm=max_grad_norm)
                self._t_critic = self._adam_step(self.critic_params(), c_grads, self._m_critic, self._v_critic, self._t_critic, lr=critic_lr)
                v_losses.append(v_loss)

            if early_stopped or (ep_kls and float(np.mean(ep_kls)) > target_kl):
                early_stopped = True
                break

        self.obs_rms.update(np.concatenate(X_raw_l, axis=0))
        self.critic_rms.update(np.concatenate(C_raw_l, axis=0))

        return {
            'n_steps': int(N_steps),
            'n_chunks': int(n_chunks),
            'mean_return': float(np.mean(ep_returns)),
            'value_loss': float(np.mean(v_losses)) if v_losses else 0.0,
            'explained_var': explained_var,
            'clip_frac': float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            'approx_kl': float(np.mean(kls)) if kls else 0.0,
            'aux_cls_acc': float(np.mean(aux_accs)) if aux_accs else 0.0,
            'aux_dir_mse': float(np.mean(aux_dirs)) if aux_dirs else 0.0,
            'early_stopped': early_stopped,
            'std': np.exp(self.log_std).tolist(),
        }
