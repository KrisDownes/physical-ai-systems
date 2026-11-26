"""Phase 1 1D-CNN Asymmetric Actor-Critic PPO + Auxiliary Contact-State & Direction Decoder.

Implements the exact 3-way ablation ladder on identical backbone, asymmetric critic,
auxiliary decoder, and reward hygiene:
- M1 (`pos_only`):       SOTA replica of Blind Precise Assembly (arXiv:2610.23849).
                         Actor sees ONLY q_quantized, q_dot_quantized, ee_pose_fk,
                         ee_twist_fk, prev_action (ZERO actuator_force, ZERO tracking_error).
                         Fixed torque_scale = 1.0.
- M2 (`current`):        Adds actuator_force_noisy + tracking_error. Fixed torque_scale = 1.0.
- M3 (`current_torque`): Adds actuator_force_noisy + tracking_error AND outputs active
                         per-joint torque_scale in [0.20, 1.0] alongside motion deltas.

Architecture:
- Temporal 1D-CNN (kernel=4, stride=2, 32 filters) over the recent 0.4 s (20-step)
  proprioceptive history -> dual temporal pooling (last + mean + max = 96 dims) +
  controller-legal start-relative spatial search context (24 dims) -> 64-dim latent `z_h`.
- Actor heads from `z_h`:
    * Policy mean `mu` (zero-initialized final layer, log_std clamped in [-3.0, -0.5])
    * Aux Contact-Class Decoder (5-class cross-entropy: free/plate/rim/entering/seated)
    * Aux Socket-Direction Decoder (2D unit vector MSE toward socket center)
- Asymmetric Privileged Critic `V(obs, priv)`: 2-layer tanh MLP (128 -> 96 -> 1), training-only.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np

from .env import (
    ALLOWED_OBS_KEYS,
    FORBIDDEN_PRIVILEGED_KEYS,
    MAX_TASK_DELTA_XY_M,
    MAX_TASK_DELTA_Z_M,
)

CONTACT_CLASSES: tuple[str, ...] = ('free', 'plate_contact', 'rim_contact', 'entering', 'seated')
CONTACT_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CONTACT_CLASSES)}

HISTORY_LEN = 20        # 0.40 s at 50 Hz
CONV_KERNEL = 4         # 80 ms receptive field per conv window
CONV_STRIDE = 2         # 9 temporal output windows across 20 steps: (20 - 4)//2 + 1 = 9
CONV_OUT_STEPS = (HISTORY_LEN - CONV_KERNEL) // CONV_STRIDE + 1
GRID_N = 4
GRID_HALF_EXTENT_M = 0.020
CLOCK_PERIOD_S = 5.0  # Fixed physical 5.0 s horizon so Fourier phase matches identically across train & eval
FOURIER_FREQS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)  # cycles per 5.0 s (controller-legal from time_s)
SPATIAL_CTX_DIM = 8 + 2 * len(FOURIER_FREQS) + GRID_N * GRID_N  # 32

M3_TORQUE_MIN = 0.20
M3_TORQUE_MAX = 1.00
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5

PRIVILEGED_CRITIC_DIM = 49


class RunningMeanStd:
    """Welford online mean/variance normalizer (updated in training, frozen in eval)."""

    def __init__(self, dim: int, eps: float = 1e-4):
        self.dim = int(dim)
        self.mean = np.zeros(self.dim, dtype=float)
        self.var = np.ones(self.dim, dtype=float)
        self.count = float(eps)

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] == 0:
            return
        b_mean = np.mean(arr, axis=0)
        b_var = np.var(arr, axis=0)
        b_count = float(arr.shape[0])
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * (b_count / tot)
        m2 = self.var * self.count + b_var * b_count + (delta ** 2) * (self.count * b_count / tot)
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray, clip_val: float = 5.0) -> np.ndarray:
        return np.clip((np.asarray(x, dtype=float) - self.mean) / np.sqrt(self.var + 1e-8), -clip_val, clip_val)

    def get_state(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f'{prefix}_mean': self.mean.copy(),
            f'{prefix}_var': self.var.copy(),
            f'{prefix}_count': np.array([self.count], dtype=float),
        }

    def set_state(self, state: dict[str, np.ndarray], prefix: str) -> None:
        self.mean = np.asarray(state[f'{prefix}_mean'], dtype=float).copy()
        self.var = np.asarray(state[f'{prefix}_var'], dtype=float).copy()
        self.count = float(np.asarray(state[f'{prefix}_count']).reshape(-1)[0])


class TemporalProprioBuilder:
    """Builds 20-step (0.4 s) temporal proprioceptive tensor + 24-dim spatial context.

    Strictly enforces controller-legal observation boundary:
    - M1 (`pos_only`):       19 channels/step = q[:5](5) + qdot[:5](5) + ee_rel(3) + ee_vel(3) + prev_xyz(3)
                             ZERO actuator_force_noisy, ZERO tracking_error, ZERO torque_scale!
    - M2 (`current`):        34 channels/step = M1(19) + tracking_error[:5](5) + f_noisy[:5](5) + df_free[:5](5)
    - M3 (`current_torque`): 39 channels/step = M2(34) + prev_torque_scale[:5](5)
    """

    def __init__(self, mode: str, max_steps: int = 250):
        if mode not in ('pos_only', 'current', 'current_torque'):
            raise ValueError(f'Unknown mode {mode!r}')
        self.mode = mode
        self.max_steps = int(max_steps)
        self.reset()

    @property
    def step_dim(self) -> int:
        if self.mode == 'pos_only':
            return 19
        if self.mode == 'current':
            return 34
        return 39

    @property
    def spatial_dim(self) -> int:
        return SPATIAL_CTX_DIM

    def reset(self) -> None:
        self._start_xyz: np.ndarray | None = None
        self._f_free: np.ndarray | None = None
        self._z_min_rel: float = 0.0
        self._grid = np.zeros((GRID_N, GRID_N), dtype=float)
        self._hist: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
        self.last_obs_keys_seen: set[str] = set()

    def step(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (hist_20xC, spatial_ctx_24) strictly from controller-legal obs."""
        keys = set(obs.keys())
        self.last_obs_keys_seen |= keys
        leaked = keys & FORBIDDEN_PRIVILEGED_KEYS
        if leaked:
            raise AssertionError(f'Privileged keys leaked into actor obs: {sorted(leaked)}')
        unknown = keys - ALLOWED_OBS_KEYS
        if unknown:
            raise AssertionError(f'Unexpected keys in actor obs: {sorted(unknown)}')

        q = np.asarray(obs['q_quantized'][:5], dtype=float)
        qdot = np.asarray(obs['q_dot_quantized'][:5], dtype=float)
        ee_pos = np.asarray(obs['ee_pose_fk'][:3], dtype=float)
        ee_vel = np.asarray(obs['ee_twist_fk'][:3], dtype=float)
        prev_xyz = np.asarray(obs['prev_action'][:3], dtype=float)
        time_s = float(obs['time_s'])

        if self._start_xyz is None:
            self._start_xyz = ee_pos.copy()
            self._z_min_rel = 0.0

        ee_rel = ee_pos - self._start_xyz
        if ee_rel[2] < self._z_min_rel:
            self._z_min_rel = float(ee_rel[2])

        r_rel = float(np.linalg.norm(ee_rel[:2]))
        if r_rel > 1e-6:
            c_th, s_th = float(ee_rel[0] / r_rel), float(ee_rel[1] / r_rel)
        else:
            c_th, s_th = 0.0, 0.0

        gx = int(np.clip(math.floor(((ee_rel[0] + GRID_HALF_EXTENT_M) / (2.0 * GRID_HALF_EXTENT_M)) * GRID_N), 0, GRID_N - 1))
        gy = int(np.clip(math.floor(((ee_rel[1] + GRID_HALF_EXTENT_M) / (2.0 * GRID_HALF_EXTENT_M)) * GRID_N), 0, GRID_N - 1))
        self._grid[gx, gy] = min(1.0, self._grid[gx, gy] + 0.12)

        t_frac = min(time_s / CLOCK_PERIOD_S, 1.0)
        fourier = np.array(
            [v for f in FOURIER_FREQS for v in (math.sin(2.0 * math.pi * f * t_frac), math.cos(2.0 * math.pi * f * t_frac))],
            dtype=float,
        )
        spatial_ctx = np.concatenate([
            np.array([
                ee_rel[0] / 0.02,
                ee_rel[1] / 0.02,
                ee_rel[2] / 0.015,
                r_rel / 0.02,
                c_th,
                s_th,
                self._z_min_rel / 0.015,
                t_frac,
            ], dtype=float),
            fourier,
            self._grid.reshape(-1),
        ])

        # M1 core frame (STRICTLY zero actuator_force, zero tracking_error, zero torque_scale!)
        frame_m1 = np.concatenate([
            q / 1.5,
            qdot / 1.5,
            ee_rel / 0.02,
            ee_vel / 0.10,
            prev_xyz / 0.004,
        ])

        if self.mode == 'pos_only':
            frame = frame_m1
        else:
            err = np.asarray(obs['tracking_error'][:5], dtype=float)
            f_noisy = np.asarray(obs['actuator_force_noisy'][:5], dtype=float)
            if self._f_free is None:
                self._f_free = f_noisy.copy()
            df = f_noisy - self._f_free
            frame_m2 = np.concatenate([frame_m1, err / 0.03, f_noisy / 1.5, df / 1.0])
            if self.mode == 'current':
                frame = frame_m2
            else:
                ts = np.asarray(obs['torque_scale'][:5], dtype=float)
                frame = np.concatenate([frame_m2, ts])

        if not self._hist:
            for _ in range(HISTORY_LEN):
                self._hist.append(frame.copy())
        else:
            self._hist.append(frame.copy())

        hist_arr = np.asarray(self._hist, dtype=float)  # (20, C)
        return hist_arr, spatial_ctx


def build_privileged_critic_features(obs: dict, priv: dict, start_xyz: np.ndarray, max_steps: int = 250) -> np.ndarray:
    """Training-only asymmetric critic vector V(obs, priv) — 49 dims, discarded at test time."""
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
    ], dtype=float)
    return np.concatenate([q, qdot, err, f, ts, ee_rel, ee_vel, c_oh, priv_vec])


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: bool,
    last_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=float)
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        next_v = values[t + 1] if (t + 1 < T) else (0.0 if terminated else float(last_value))
        delta = rewards[t] + gamma * next_v - values[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    return adv, adv + values


def clip_grad_norm(grads: dict[str, np.ndarray], max_norm: float = 0.5) -> float:
    total_sq = sum(float(np.sum(g * g)) for g in grads.values())
    norm = float(np.sqrt(total_sq))
    if norm > max_norm and norm > 1e-12:
        scale = max_norm / norm
        for k in grads:
            grads[k] = grads[k] * scale
    return norm


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    ex = np.exp(np.clip(shifted, -30.0, 30.0))
    return ex / np.sum(ex, axis=-1, keepdims=True)


def _im2col_1d(H: np.ndarray) -> np.ndarray:
    """Zero-copy strided view converting (B, T=20, C) into (B, L=9, K*C) patches for 1D-CNN."""
    B, T, C = H.shape
    H_c = np.ascontiguousarray(H)
    s0, s1, s2 = H_c.strides
    windows = np.lib.stride_tricks.as_strided(
        H_c,
        shape=(B, CONV_OUT_STEPS, CONV_KERNEL, C),
        strides=(s0, CONV_STRIDE * s1, s1, s2),
    )
    return windows.reshape(B, CONV_OUT_STEPS, CONV_KERNEL * C)


class TemporalAsymmetricPPO:
    """1D-CNN Temporal Actor + Dual Aux Decoder (Contact Class + XY Dir) + Asymmetric Critic."""

    def __init__(
        self,
        step_dim: int,
        spatial_dim: int = SPATIAL_CTX_DIM,
        critic_dim: int = PRIVILEGED_CRITIC_DIM,
        use_torque_limit: bool = False,
        conv_filters: int = 32,
        latent_dim: int = 64,
        critic_h1: int = 128,
        critic_h2: int = 96,
        seed: int = 0,
        init_log_std: float = -0.65,
    ):
        self.step_dim = int(step_dim)
        self.spatial_dim = int(spatial_dim)
        self.critic_dim = int(critic_dim)
        self.use_torque_limit = bool(use_torque_limit)
        self.act_dim = 8 if self.use_torque_limit else 3
        self.conv_filters = int(conv_filters)
        self.latent_dim = int(latent_dim)
        self.critic_h1 = int(critic_h1)
        self.critic_h2 = int(critic_h2)
        self.n_contact_classes = len(CONTACT_CLASSES)

        rng = np.random.default_rng(seed)

        # 1D-CNN over (20, step_dim) -> (9, conv_filters) -> [last, mean, max] (3*F) + spatial_dim -> latent_dim
        conv_in = CONV_KERNEL * self.step_dim
        self.W_conv = rng.normal(0.0, np.sqrt(1.0 / conv_in), size=(conv_in, self.conv_filters))
        self.b_conv = np.zeros(self.conv_filters, dtype=float)

        fc_in = 3 * self.conv_filters + self.spatial_dim
        self.W_fc = rng.normal(0.0, np.sqrt(1.0 / fc_in), size=(fc_in, self.latent_dim))
        self.b_fc = np.zeros(self.latent_dim, dtype=float)

        # Policy mean head: STRICTLY ZERO-INITIALIZED!
        self.W_mu = np.zeros((self.latent_dim, self.act_dim), dtype=float)
        self.b_mu = np.zeros(self.act_dim, dtype=float)
        init_ls = float(np.clip(init_log_std, LOG_STD_MIN, LOG_STD_MAX))
        self.log_std = np.full(self.act_dim, init_ls, dtype=float)

        # Training-only auxiliary decoders from temporal latent z_h:
        # (a) 5-class contact_class logits; (b) 2D unit socket_xy_error direction prediction.
        self.W_aux_cls = rng.normal(0.0, 0.05, size=(self.latent_dim, self.n_contact_classes))
        self.b_aux_cls = np.zeros(self.n_contact_classes, dtype=float)
        self.W_aux_dir = rng.normal(0.0, 0.05, size=(self.latent_dim, 2))
        self.b_aux_dir = np.zeros(2, dtype=float)

        # Asymmetric Privileged Critic V(obs, priv)
        self.C_W1 = rng.normal(0.0, np.sqrt(1.0 / self.critic_dim), size=(self.critic_dim, self.critic_h1))
        self.C_b1 = np.zeros(self.critic_h1, dtype=float)
        self.C_W2 = rng.normal(0.0, np.sqrt(1.0 / self.critic_h1), size=(self.critic_h1, self.critic_h2))
        self.C_b2 = np.zeros(self.critic_h2, dtype=float)
        self.C_Wv = rng.normal(0.0, 0.01, size=(self.critic_h2, 1))
        self.C_bv = np.zeros(1, dtype=float)

        # Running normalizers (step_rms & spatial_rms saved in ckpt, frozen at eval; critic_rms training-only)
        self.step_rms = RunningMeanStd(self.step_dim)
        self.spatial_rms = RunningMeanStd(self.spatial_dim)
        self.critic_rms = RunningMeanStd(self.critic_dim)

        self._m_actor = {k: np.zeros_like(v) for k, v in self.actor_params().items()}
        self._v_actor = {k: np.zeros_like(v) for k, v in self.actor_params().items()}
        self._t_actor = 0
        self._m_critic = {k: np.zeros_like(v) for k, v in self.critic_params().items()}
        self._v_critic = {k: np.zeros_like(v) for k, v in self.critic_params().items()}
        self._t_critic = 0

    def actor_params(self) -> dict[str, np.ndarray]:
        return {
            'W_conv': self.W_conv, 'b_conv': self.b_conv,
            'W_fc': self.W_fc, 'b_fc': self.b_fc,
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

    def get_actor_weights(self) -> dict[str, np.ndarray]:
        out = {k: v.copy() for k, v in self.actor_params().items()}
        out.update(self.step_rms.get_state('step_rms'))
        out.update(self.spatial_rms.get_state('spatial_rms'))
        return out

    def get_all_weights(self) -> dict[str, np.ndarray]:
        out = self.get_actor_weights()
        out.update({k: v.copy() for k, v in self.critic_params().items()})
        out.update(self.critic_rms.get_state('critic_rms'))
        return out

    def set_weights(self, w: dict[str, np.ndarray]) -> None:
        for k in ('W_conv', 'b_conv', 'W_fc', 'b_fc', 'W_mu', 'b_mu', 'W_aux_cls', 'b_aux_cls', 'W_aux_dir', 'b_aux_dir'):
            setattr(self, k, np.asarray(w[k], dtype=float).copy())
        self.log_std = np.clip(np.asarray(w['log_std'], dtype=float).copy(), LOG_STD_MIN, LOG_STD_MAX)
        if 'step_rms_mean' in w:
            self.step_rms.set_state(w, 'step_rms')
        if 'spatial_rms_mean' in w:
            self.spatial_rms.set_state(w, 'spatial_rms')
        if 'C_W1' in w:
            for k in ('C_W1', 'C_b1', 'C_W2', 'C_b2', 'C_Wv', 'C_bv'):
                setattr(self, k, np.asarray(w[k], dtype=float).copy())
        if 'critic_rms_mean' in w:
            self.critic_rms.set_state(w, 'critic_rms')

    def save(self, path) -> None:
        np.savez(
            path,
            step_dim=self.step_dim,
            spatial_dim=self.spatial_dim,
            critic_dim=self.critic_dim,
            use_torque_limit=int(self.use_torque_limit),
            conv_filters=self.conv_filters,
            latent_dim=self.latent_dim,
            critic_h1=self.critic_h1,
            critic_h2=self.critic_h2,
            **self.get_all_weights(),
        )

    @classmethod
    def load(cls, path) -> 'TemporalAsymmetricPPO':
        data = np.load(path)
        pol = cls(
            step_dim=int(data['step_dim']),
            spatial_dim=int(data['spatial_dim']),
            critic_dim=int(data['critic_dim']),
            use_torque_limit=bool(int(data['use_torque_limit'])),
            conv_filters=int(data['conv_filters']),
            latent_dim=int(data['latent_dim']),
            critic_h1=int(data['critic_h1']),
            critic_h2=int(data['critic_h2']),
        )
        meta = {'step_dim', 'spatial_dim', 'critic_dim', 'use_torque_limit', 'conv_filters', 'latent_dim', 'critic_h1', 'critic_h2'}
        pol.set_weights({k: data[k] for k in data.files if k not in meta})
        return pol

    def normalize_actor_inputs(self, H_raw: np.ndarray, S_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H_norm = self.step_rms.normalize(H_raw)
        S_norm = self.spatial_rms.normalize(S_raw)
        return H_norm, S_norm

    def forward_actor(self, H_norm: np.ndarray, S_norm: np.ndarray):
        """Forward 1D-CNN + pooled fusion + latent z_h + policy & aux heads."""
        patches = _im2col_1d(H_norm)                          # (B, L=9, K*C)
        conv_act = np.tanh(patches @ self.W_conv + self.b_conv)  # (B, L=9, F)
        f_last = conv_act[:, -1, :]                           # (B, F)
        f_mean = np.mean(conv_act, axis=1)                    # (B, F)
        max_idx = np.argmax(conv_act, axis=1)                 # (B, F)
        B, _, F = conv_act.shape
        f_max = conv_act[np.arange(B)[:, None], max_idx, np.arange(F)[None, :]]  # (B, F)

        fused = np.concatenate([f_last, f_mean, f_max, S_norm], axis=1)  # (B, 3F + S)
        z_h = np.tanh(fused @ self.W_fc + self.b_fc)          # (B, latent_dim)
        mu = z_h @ self.W_mu + self.b_mu                      # (B, act_dim)
        aux_cls = z_h @ self.W_aux_cls + self.b_aux_cls       # (B, 5)
        aux_dir = z_h @ self.W_aux_dir + self.b_aux_dir       # (B, 2)
        cache = (patches, conv_act, max_idx, fused, z_h)
        return mu, aux_cls, aux_dir, cache

    def forward_critic(self, C_norm: np.ndarray):
        h1 = np.tanh(C_norm @ self.C_W1 + self.C_b1)
        h2 = np.tanh(h1 @ self.C_W2 + self.C_b2)
        v = (h2 @ self.C_Wv + self.C_bv).reshape(-1)
        return v, h1, h2

    def map_z_to_env_action(self, z: np.ndarray) -> np.ndarray:
        """Map pre-action z -> [dx, dy, dz, ts_0..ts_4] with M3 ts in [0.20, 1.0] initialized at 0.75."""
        d_xy = np.clip(z[:2], -1.0, 1.0) * MAX_TASK_DELTA_XY_M
        # Asymmetric vertical cap: full downward insertion speed (-3 mm/step), bounded upward lift (+1.5 mm/step)
        d_z = float(np.clip(z[2], -1.0, 0.5)) * MAX_TASK_DELTA_Z_M
        if not self.use_torque_limit:
            ts = np.ones(5, dtype=float)
        else:
            # Zero-init (z=0) starts at ts=0.75 (overcomes bolted friction on ep 0),
            # while remaining free to drop smoothly down to 0.20-0.40 on unbolted sockets/rim contact.
            ts = np.clip(0.75 + 0.40 * z[3:8], M3_TORQUE_MIN, M3_TORQUE_MAX)
        return np.concatenate([d_xy, [d_z], ts])

    def act_step(
        self,
        hist_20xC: np.ndarray,
        spatial_ctx: np.ndarray,
        deterministic: bool = True,
        eps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        H_n, S_n = self.normalize_actor_inputs(hist_20xC[None, :, :], spatial_ctx[None, :])
        mu, aux_cls, aux_dir, _ = self.forward_actor(H_n, S_n)
        mu0 = mu[0]
        if deterministic:
            z = mu0.copy()
        else:
            std = np.exp(np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
            noise = np.zeros(self.act_dim, dtype=float) if eps is None else eps
            z = mu0 + std * noise
        return self.map_z_to_env_action(z), z, int(np.argmax(aux_cls[0])), aux_dir[0].copy()

    def step_log_prob(self, H_norm: np.ndarray, S_norm: np.ndarray, Z: np.ndarray) -> np.ndarray:
        mu, _, _, _ = self.forward_actor(H_norm, S_norm)
        log_std = np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        inv_var = np.exp(-2.0 * log_std)
        diff = Z - mu
        return -0.5 * np.sum((diff ** 2) * inv_var + 2.0 * log_std + np.log(2.0 * np.pi), axis=1)

    def actor_and_aux_loss_and_grad(
        self,
        H_norm: np.ndarray,
        S_norm: np.ndarray,
        Z: np.ndarray,
        old_logp: np.ndarray,
        adv: np.ndarray,
        cls_labels: np.ndarray,
        dir_targets: np.ndarray,
        dir_mask: np.ndarray,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.002,
        aux_cls_coef: float = 0.15,
        aux_dir_coef: float = 0.15,
        weight_decay: float = 1e-4,
    ):
        B = len(H_norm)
        mu, aux_cls, aux_dir, (patches, conv_act, max_idx, fused, z_h) = self.forward_actor(H_norm, S_norm)
        log_std = np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        inv_var = np.exp(-2.0 * log_std)
        diff = Z - mu
        logp = -0.5 * np.sum((diff ** 2) * inv_var + 2.0 * log_std + np.log(2.0 * np.pi), axis=1)

        log_ratio = np.clip(logp - old_logp, -10.0, 10.0)
        ratio = np.exp(log_ratio)
        clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surr1 = ratio * adv
        surr2 = clipped * adv
        use_unclipped = (surr1 <= surr2)
        clip_frac = float(np.mean(~use_unclipped))
        approx_kl = float(np.mean((ratio - 1.0) - log_ratio))

        # Aux (a): 5-class contact_class cross-entropy
        probs = _softmax(aux_cls)
        rows = np.arange(B)
        aux_ce = -float(np.mean(np.log(np.clip(probs[rows, cls_labels], 1e-12, 1.0))))
        aux_acc = float(np.mean(np.argmax(aux_cls, axis=1) == cls_labels))

        # Aux (b): normalized socket_xy_error direction MSE (masked when xy_err ~ 0)
        dir_err = aux_dir - dir_targets
        mask_sum = max(float(np.sum(dir_mask)), 1.0)
        aux_dir_mse = 0.5 * float(np.sum(dir_mask[:, None] * (dir_err ** 2)) / mask_sum)

        entropy = float(np.sum(log_std + 0.5 * (1.0 + np.log(2.0 * np.pi))))
        wd_loss = 0.5 * weight_decay * (
            float(np.sum(self.W_conv ** 2))
            + float(np.sum(self.W_fc ** 2))
            + float(np.sum(self.W_mu ** 2))
            + float(np.sum(self.W_aux_cls ** 2))
            + float(np.sum(self.W_aux_dir ** 2))
        )
        loss = (
            -float(np.mean(np.minimum(surr1, surr2)))
            - entropy_coef * entropy
            + aux_cls_coef * aux_ce
            + aux_dir_coef * aux_dir_mse
            + wd_loss
        )

        # Policy branch gradients
        d_logp = np.where(use_unclipped, -(ratio * adv) / B, 0.0)
        d_mu = d_logp[:, None] * (diff * inv_var)
        d_W_mu = z_h.T @ d_mu + weight_decay * self.W_mu
        d_b_mu = np.sum(d_mu, axis=0)
        d_log_std = np.sum(d_logp[:, None] * ((diff ** 2) * inv_var - 1.0), axis=0) - entropy_coef * np.ones(self.act_dim)

        # Aux contact-class branch gradients
        d_cls = probs.copy()
        d_cls[rows, cls_labels] -= 1.0
        d_cls *= (aux_cls_coef / B)
        d_W_aux_cls = z_h.T @ d_cls + weight_decay * self.W_aux_cls
        d_b_aux_cls = np.sum(d_cls, axis=0)

        # Aux direction branch gradients
        d_dir = (aux_dir_coef / mask_sum) * (dir_mask[:, None] * dir_err)
        d_W_aux_dir = z_h.T @ d_dir + weight_decay * self.W_aux_dir
        d_b_aux_dir = np.sum(d_dir, axis=0)

        # Latent FC backprop
        d_zh = d_mu @ self.W_mu.T + d_cls @ self.W_aux_cls.T + d_dir @ self.W_aux_dir.T
        d_pre_fc = d_zh * (1.0 - z_h ** 2)
        d_W_fc = fused.T @ d_pre_fc + weight_decay * self.W_fc
        d_b_fc = np.sum(d_pre_fc, axis=0)
        d_fused = d_pre_fc @ self.W_fc.T

        # Unpool into conv_act (B, L=9, F)
        F = self.conv_filters
        d_last = d_fused[:, :F]
        d_mean = d_fused[:, F:2 * F]
        d_max = d_fused[:, 2 * F:3 * F]

        d_conv_act = np.broadcast_to((d_mean / CONV_OUT_STEPS)[:, None, :], conv_act.shape).copy()
        d_conv_act[:, -1, :] += d_last
        np.add.at(d_conv_act, (np.arange(B)[:, None], max_idx, np.arange(F)[None, :]), d_max)

        d_pre_conv = d_conv_act * (1.0 - conv_act ** 2)  # (B, L, F)
        patches_2d = patches.reshape(B * CONV_OUT_STEPS, -1)
        d_pre_2d = d_pre_conv.reshape(B * CONV_OUT_STEPS, F)
        d_W_conv = patches_2d.T @ d_pre_2d + weight_decay * self.W_conv
        d_b_conv = np.sum(d_pre_2d, axis=0)

        grads = {
            'W_conv': d_W_conv, 'b_conv': d_b_conv,
            'W_fc': d_W_fc, 'b_fc': d_b_fc,
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
        actor_lr: float = 1.2e-3,
        critic_lr: float = 2.5e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        minibatch_size: int = 256,
        entropy_coef: float = 0.002,
        aux_cls_coef: float = 0.15,
        aux_dir_coef: float = 0.15,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.03,
        weight_decay: float = 1e-4,
    ) -> dict:
        valid = [tr for tr in trajectories if len(tr['hist']) > 0]
        if not valid:
            return {'n_steps': 0, 'mean_return': 0.0, 'std': np.exp(self.log_std).tolist()}

        H_raw_l, S_raw_l, C_raw_l = [], [], []
        H_n_l, S_n_l, C_n_l = [], [], []
        Z_l, old_lp_l, old_v_l, adv_l, ret_l, cls_l, dir_l, dmask_l = [], [], [], [], [], [], [], []
        ep_returns = []

        for tr in valid:
            H_raw = np.asarray(tr['hist'], dtype=float)       # (T, 20, C)
            S_raw = np.asarray(tr['spatial'], dtype=float)    # (T, 24)
            C_raw = np.asarray(tr['critic_obs'], dtype=float) # (T, 49)
            Z = np.asarray(tr['z'], dtype=float)
            R = np.asarray(tr['rewards'], dtype=float)
            CLS = np.asarray(tr['contact_labels'], dtype=np.int64)
            DIR = np.asarray(tr['dir_targets'], dtype=float)
            DMASK = np.asarray(tr['dir_mask'], dtype=float)

            H_n, S_n = self.normalize_actor_inputs(H_raw, S_raw)
            C_n = self.critic_rms.normalize(C_raw)
            v_old, _, _ = self.forward_critic(C_n)
            lp_old = self.step_log_prob(H_n, S_n, Z)
            last_v = 0.0
            if not tr['terminated'] and 'last_critic_obs' in tr:
                last_c = self.critic_rms.normalize(tr['last_critic_obs'])[None, :]
                last_v = float(self.forward_critic(last_c)[0][0])
            adv_ep, ret_ep = compute_gae(R, v_old, terminated=bool(tr['terminated']), last_value=last_v, gamma=gamma, lam=lam)

            H_raw_l.append(H_raw[:, -1, :])  # latest step frame for Welford step_rms update
            S_raw_l.append(S_raw)
            C_raw_l.append(C_raw)
            H_n_l.append(H_n)
            S_n_l.append(S_n)
            C_n_l.append(C_n)
            Z_l.append(Z)
            old_lp_l.append(lp_old)
            old_v_l.append(v_old)
            adv_l.append(adv_ep)
            ret_l.append(ret_ep)
            cls_l.append(CLS)
            dir_l.append(DIR)
            dmask_l.append(DMASK)
            ep_returns.append(float(np.sum(R)))

        H_all = np.concatenate(H_n_l, axis=0)
        S_all = np.concatenate(S_n_l, axis=0)
        C_all = np.concatenate(C_n_l, axis=0)
        Z_all = np.concatenate(Z_l, axis=0)
        old_lp_all = np.concatenate(old_lp_l, axis=0)
        old_v_all = np.concatenate(old_v_l, axis=0)
        adv_all = np.concatenate(adv_l, axis=0)
        ret_all = np.concatenate(ret_l, axis=0)
        cls_all = np.concatenate(cls_l, axis=0)
        dir_all = np.concatenate(dir_l, axis=0)
        dmask_all = np.concatenate(dmask_l, axis=0)

        var_y = float(np.var(ret_all))
        explained_var = float(1.0 - np.var(ret_all - old_v_all) / (var_y + 1e-8)) if var_y > 1e-8 else 0.0

        adv_std = float(np.std(adv_all))
        if adv_std > 1e-8:
            adv_all = (adv_all - float(np.mean(adv_all))) / adv_std

        N = len(H_all)
        clip_fracs, kls, v_losses, aux_accs, aux_dirs = [], [], [], [], []
        early_stopped = False

        for _ in range(epochs):
            perm = rng.permutation(N)
            ep_kls = []
            for start in range(0, N, minibatch_size):
                idx = perm[start:start + minibatch_size]
                if len(idx) < 16:
                    continue
                _, a_grads, cf, kl, _, aux_acc, aux_dir_mse = self.actor_and_aux_loss_and_grad(
                    H_all[idx], S_all[idx], Z_all[idx], old_lp_all[idx], adv_all[idx],
                    cls_all[idx], dir_all[idx], dmask_all[idx],
                    clip_eps=clip_eps, entropy_coef=entropy_coef,
                    aux_cls_coef=aux_cls_coef, aux_dir_coef=aux_dir_coef, weight_decay=weight_decay,
                )
                clip_grad_norm(a_grads, max_norm=max_grad_norm)
                self._t_actor = self._adam_step(self.actor_params(), a_grads, self._m_actor, self._v_actor, self._t_actor, lr=actor_lr)
                self.log_std[:] = np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)

                v_loss, c_grads = self.critic_loss_and_grad(C_all[idx], old_v_all[idx], ret_all[idx], clip_eps=clip_eps, weight_decay=weight_decay)
                clip_grad_norm(c_grads, max_norm=max_grad_norm)
                self._t_critic = self._adam_step(self.critic_params(), c_grads, self._m_critic, self._v_critic, self._t_critic, lr=critic_lr)

                clip_fracs.append(cf)
                kls.append(kl)
                ep_kls.append(kl)
                v_losses.append(v_loss)
                aux_accs.append(aux_acc)
                aux_dirs.append(aux_dir_mse)

            if ep_kls and float(np.mean(ep_kls)) > target_kl:
                early_stopped = True
                break

        # Update normalizers AFTER PPO update so current batch's old_logp/old_v remain exact.
        self.step_rms.update(np.concatenate(H_raw_l, axis=0))
        self.spatial_rms.update(np.concatenate(S_raw_l, axis=0))
        self.critic_rms.update(np.concatenate(C_raw_l, axis=0))

        return {
            'n_steps': int(N),
            'mean_return': float(np.mean(ep_returns)),
            'clip_fraction': float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            'approx_kl': float(np.mean(kls)) if kls else 0.0,
            'value_loss': float(np.mean(v_losses)) if v_losses else 0.0,
            'explained_var': explained_var,
            'aux_cls_acc': float(np.mean(aux_accs)) if aux_accs else 0.0,
            'aux_dir_mse': float(np.mean(aux_dirs)) if aux_dirs else 0.0,
            'early_stopped': early_stopped,
            'std': np.exp(self.log_std).tolist(),
        }
