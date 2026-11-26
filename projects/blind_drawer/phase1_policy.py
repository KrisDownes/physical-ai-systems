"""Phase 1 Relative-Kinematics Residual GRU(64) PPO Policy for `projects/blind_drawer/`.

Strictly controller-legal relative-kinematics actor observation (37-dim, ZERO absolute `ee_pose_fk`,
ZERO noisy current, ZERO privileged info):
  - q_quantized[:5]        (5)
  - q_dot_quantized[:5]    (5)
  - ee_rel_start           (3)  <- relative 3D FK displacement ONLY (height/placement invariant!)
  - ee_twist_fk            (6)
  - tracking_error[:5]     (5)  <- 4.84x Tactile-Amplifier channel under GravComp-TL
  - prev_action            (8)
  - a_base_t               (3)  <- frozen NominalStraightBackController base delta_xyz
  - prev_alpha             (2)  <- [alpha_lat, alpha_arm] (or [1.0, 1.0] for D1 Stiff)

Two matched residual methods:
  - `D1_Res_PosOnly` (`res_pos_only`):       3-DoF residual (dx, dy, dz) on top of frozen base, stiff ts=1.0
  - `D2_Res_GravComp_TL` (`res_gravcomp_tl`): 3-DoF residual (dx, dy, dz) + 2-DoF dynamic [alpha_lat, alpha_arm]
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from .env import ALLOWED_OBS_KEYS, FORBIDDEN_PRIVILEGED_KEYS

DRAWER_CONTACT_CLASSES: tuple[str, ...] = (
    'free_slide',
    'off_axis_bind',
    'detent_catch',
    'handle_slip',
    'opened',
)
DRAWER_CONTACT_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(DRAWER_CONTACT_CLASSES)}

D1_ALPHA_MIN = 0.10
D1_ALPHA_MAX = 1.00
D1_INIT_ALPHA_LAT = 0.35
D1_INIT_ALPHA_ARM = 0.35
D3_INIT_ALPHA_5DOF = (0.30, 0.45, 0.45, 0.35, 0.35)  # [pan, lift, elbow, wrist_flex, wrist_roll]
JOINT_ALPHA_NAMES: tuple[str, ...] = ('pan', 'lift', 'elbow', 'wflex', 'wroll')

RES_XY_CAP_M = 0.0050
RES_Z_CAP_M = 0.0050
GRU_HIDDEN_DIM = 64
BPTT_CHUNK_LEN = 32
CRITIC_PRIV_DIM = 18


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_c = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_c))


def _logit(p: float) -> float:
    p_c = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p_c / (1.0 - p_c))


def _orth_init(rng: np.random.Generator, rows: int, cols: int, gain: float = 1.0) -> np.ndarray:
    a = rng.normal(0.0, 1.0, size=(rows, cols))
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == (rows, cols) else vt
    return (gain * q[:rows, :cols]).astype(float)


class RunningMeanStd:
    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=float)
        self.var = np.ones(dim, dtype=float)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] == 0:
            return
        b_mean = arr.mean(axis=0)
        b_var = arr.var(axis=0)
        b_cnt = float(arr.shape[0])
        delta = b_mean - self.mean
        tot = self.count + b_cnt
        new_mean = self.mean + delta * (b_cnt / tot)
        m_a = self.var * self.count
        m_b = b_var * b_cnt
        m2 = m_a + m_b + (delta ** 2) * (self.count * b_cnt / tot)
        self.mean = new_mean
        self.var = np.maximum(m2 / tot, 1e-6)
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-6), -8.0, 8.0)


class DrawerObsBuilder:
    """Builds strictly controller-legal relative-kinematics actor features (ZERO absolute ee_pos)."""

    def __init__(self, mode: str):
        if mode not in ('res_pos_only', 'res_gravcomp_tl', 'res_gravcomp_tl_5dof'):
            raise ValueError(f'Unknown mode {mode!r}')
        self.mode = mode
        self.alpha_dim = 5 if mode == 'res_gravcomp_tl_5dof' else 2
        self.obs_dim = 35 + self.alpha_dim
        self.prev_alpha = self._init_alpha()

    def _init_alpha(self) -> np.ndarray:
        if self.mode == 'res_gravcomp_tl_5dof':
            return np.array(D3_INIT_ALPHA_5DOF, dtype=float)
        if self.mode == 'res_gravcomp_tl':
            return np.array([D1_INIT_ALPHA_LAT, D1_INIT_ALPHA_ARM], dtype=float)
        return np.array([1.0, 1.0], dtype=float)

    def reset(self) -> None:
        self.prev_alpha = self._init_alpha()

    def set_prev_alpha(self, alpha_val: tuple[float, ...] | np.ndarray | None) -> None:
        if alpha_val is not None:
            arr = np.asarray(alpha_val, dtype=float).reshape(-1)
            if arr.size != self.alpha_dim:
                raise ValueError(f'Expected prev_alpha len {self.alpha_dim}, got {arr.size}')
            self.prev_alpha = arr.copy()

    def build(self, obs: dict, a_base_xyz: np.ndarray) -> np.ndarray:
        keys = set(obs.keys())
        leaked = keys & FORBIDDEN_PRIVILEGED_KEYS
        if leaked:
            raise AssertionError(f'Forbidden keys leaked into actor obs: {sorted(leaked)}')
        if keys != ALLOWED_OBS_KEYS:
            raise AssertionError(f'Unexpected actor obs keys: {sorted(keys)}')
        return np.concatenate([
            np.asarray(obs['q_quantized'][:5], dtype=float),
            np.asarray(obs['q_dot_quantized'][:5], dtype=float) * 0.25,
            np.asarray(obs['ee_rel_start'][:3], dtype=float) * 20.0,
            np.asarray(obs['ee_twist_fk'][:6], dtype=float) * 5.0,
            np.asarray(obs['tracking_error'][:5], dtype=float) * 25.0,
            np.asarray(obs['prev_action'][:8], dtype=float) * np.array([150.0, 150.0, 150.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.asarray(a_base_xyz[:3], dtype=float) * 150.0,
            self.prev_alpha,
        ]).astype(float)


def build_privileged_critic_features(obs: dict, priv: dict) -> np.ndarray:
    """18-dim training-only privileged critic vector (NEVER exposed to actor)."""
    yaw = float(priv['rail_yaw_rad'])
    oh = np.zeros(len(DRAWER_CONTACT_CLASSES), dtype=float)
    oh[DRAWER_CONTACT_TO_IDX[priv['contact_class']]] = 1.0
    return np.concatenate([
        [
            float(priv['drawer_travel_mm']) / 70.0,
            math.sin(yaw),
            math.cos(yaw),
            float(priv['cabinet_disp_mm']) / 25.0,
            float(priv['cabinet_tilt_deg']) / 10.0,
            float(priv['lateral_bind_force_N']) / 20.0,
            float(priv['peak_contact_force_N']) / 50.0,
            1.0 if priv['bolted'] else 0.0,
            float(priv['mu_table']),
            float(priv['cabinet_mass_kg']),
            float(priv['detent_force_N']) / 10.0,
            float(priv['detent_travel_mm']) / 70.0,
            float(priv['f_rail_N']) / 6.0,
        ],
        oh,
    ]).astype(float)


class _Adam:
    def __init__(self, params: dict[str, np.ndarray]):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float, weight_decay: float = 0.0) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for k, g in grads.items():
            if weight_decay > 0.0 and params[k].ndim >= 2:
                g = g + weight_decay * params[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class DrawerResidualGRUPPO:
    """Clean standalone NumPy GRU(64) Residual PPO with Asymmetric Critic & Dual Aux Heads."""

    def __init__(
        self,
        obs_dim: int = 37,
        mode: str = 'res_gravcomp_tl',
        critic_priv_dim: int = CRITIC_PRIV_DIM,
        hidden_dim: int = GRU_HIDDEN_DIM,
        seed: int = 0,
        init_log_std: float = -0.95,
    ):
        self.obs_dim = int(obs_dim)
        self.mode = mode
        if mode == 'res_gravcomp_tl_5dof':
            self.act_dim = 8
        elif mode == 'res_gravcomp_tl':
            self.act_dim = 5
        elif mode == 'res_pos_only':
            self.act_dim = 3
        else:
            raise ValueError(f'Unknown mode {mode!r}')
        self.hidden_dim = int(hidden_dim)
        self.critic_priv_dim = int(critic_priv_dim)
        self.critic_in_dim = self.obs_dim + self.critic_priv_dim
        self.n_cls = len(DRAWER_CONTACT_CLASSES)

        rng = np.random.default_rng(seed)
        H = self.hidden_dim
        D = self.obs_dim

        # Per-gate orthogonal recurrent weights (prevents spectral-norm collapse!)
        W_h = np.concatenate([
            _orth_init(rng, H, H, gain=0.9),
            _orth_init(rng, H, H, gain=0.9),
            _orth_init(rng, H, H, gain=0.9),
        ], axis=1)
        W_x = _orth_init(rng, D, 3 * H, gain=1.0)
        b_g = np.zeros(3 * H, dtype=float)
        b_g[:H] = 0.5  # update gate bias slightly open for smooth episode memory

        # Zero-initialized residual head so step-0 starts cleanly near nominal base pull!
        W_act = np.zeros((H, self.act_dim), dtype=float)
        b_act = np.zeros(self.act_dim, dtype=float)
        if self.mode == 'res_gravcomp_tl':
            p_lat = (D1_INIT_ALPHA_LAT - D1_ALPHA_MIN) / (D1_ALPHA_MAX - D1_ALPHA_MIN)
            p_arm = (D1_INIT_ALPHA_ARM - D1_ALPHA_MIN) / (D1_ALPHA_MAX - D1_ALPHA_MIN)
            b_act[3] = _logit(p_lat)
            b_act[4] = _logit(p_arm)
        elif self.mode == 'res_gravcomp_tl_5dof':
            for j, a0 in enumerate(D3_INIT_ALPHA_5DOF):
                pj = (a0 - D1_ALPHA_MIN) / (D1_ALPHA_MAX - D1_ALPHA_MIN)
                b_act[3 + j] = _logit(pj)

        self.actor_params: dict[str, np.ndarray] = {
            'W_x': W_x,
            'W_h': W_h,
            'b_g': b_g,
            'W_act': W_act,
            'b_act': b_act,
            'log_std': np.full(self.act_dim, float(init_log_std), dtype=float),
            'W_cls': _orth_init(rng, H, self.n_cls, gain=0.5),
            'b_cls': np.zeros(self.n_cls, dtype=float),
            'W_yaw': _orth_init(rng, H, 1, gain=0.5),
            'b_yaw': np.zeros(1, dtype=float),
        }

        C = 96
        self.critic_params: dict[str, np.ndarray] = {
            'W1': _orth_init(rng, self.critic_in_dim, C, gain=math.sqrt(2.0)),
            'b1': np.zeros(C, dtype=float),
            'W2': _orth_init(rng, C, C, gain=math.sqrt(2.0)),
            'b2': np.zeros(C, dtype=float),
            'W3': _orth_init(rng, C, 1, gain=0.1),
            'b3': np.zeros(1, dtype=float),
        }

        self.obs_rms = RunningMeanStd(self.obs_dim)
        self.critic_rms = RunningMeanStd(self.critic_in_dim)
        self._opt_actor = _Adam(self.actor_params)
        self._opt_critic = _Adam(self.critic_params)

    @property
    def actor_param_count(self) -> int:
        return int(sum(v.size for v in self.actor_params.values()))

    @property
    def critic_param_count(self) -> int:
        return int(sum(v.size for v in self.critic_params.values()))

    def init_hidden(self) -> np.ndarray:
        return np.zeros(self.hidden_dim, dtype=float)

    def get_actor_weights(self) -> dict[str, np.ndarray]:
        out = {k: v.copy() for k, v in self.actor_params.items()}
        out['_obs_mean'] = self.obs_rms.mean.copy()
        out['_obs_var'] = self.obs_rms.var.copy()
        return out

    def set_actor_weights(self, weights: dict[str, np.ndarray]) -> None:
        for k in self.actor_params:
            self.actor_params[k] = np.asarray(weights[k], dtype=float).copy()
        if '_obs_mean' in weights:
            self.obs_rms.mean = np.asarray(weights['_obs_mean'], dtype=float).copy()
            self.obs_rms.var = np.asarray(weights['_obs_var'], dtype=float).copy()

    def _gru_step(self, x_n: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, tuple]:
        H = self.hidden_dim
        p = self.actor_params
        gx = x_n @ p['W_x']
        gh = h_prev @ p['W_h']
        z_g = _sigmoid(gx[..., :H] + gh[..., :H] + p['b_g'][:H])
        r_g = _sigmoid(gx[..., H:2 * H] + gh[..., H:2 * H] + p['b_g'][H:2 * H])
        gh_r = (r_g * h_prev) @ p['W_h'][:, 2 * H:]
        n_g = np.tanh(gx[..., 2 * H:] + gh_r + p['b_g'][2 * H:])
        h_new = (1.0 - z_g) * n_g + z_g * h_prev
        return h_new, (x_n, h_prev, z_g, r_g, n_g)

    def decode_action(self, z: np.ndarray, a_base_xyz: np.ndarray) -> tuple[dict | np.ndarray, np.ndarray, tuple[float, ...] | None]:
        res_xy = RES_XY_CAP_M * np.tanh(z[:2])
        res_z = RES_Z_CAP_M * float(np.tanh(z[2]))
        d_xyz = np.array([
            float(a_base_xyz[0] + res_xy[0]),
            float(a_base_xyz[1] + res_xy[1]),
            float(a_base_xyz[2] + res_z),
        ], dtype=float)
        if self.mode == 'res_gravcomp_tl':
            a_lat = D1_ALPHA_MIN + (D1_ALPHA_MAX - D1_ALPHA_MIN) * float(_sigmoid(z[3]))
            a_arm = D1_ALPHA_MIN + (D1_ALPHA_MAX - D1_ALPHA_MIN) * float(_sigmoid(z[4]))
            return {'delta_xyz': d_xyz, 'gravcomp_alpha': (a_lat, a_arm)}, d_xyz, (a_lat, a_arm)
        if self.mode == 'res_gravcomp_tl_5dof':
            a5 = D1_ALPHA_MIN + (D1_ALPHA_MAX - D1_ALPHA_MIN) * _sigmoid(z[3:8])
            a5_tup = tuple(float(v) for v in a5)
            return {'delta_xyz': d_xyz, 'gravcomp_alpha': a5}, d_xyz, a5_tup
        return np.array([d_xyz[0], d_xyz[1], d_xyz[2], 1.0], dtype=float), d_xyz, None

    def act_step(
        self,
        x_raw: np.ndarray,
        h_prev: np.ndarray,
        a_base_xyz: np.ndarray,
        deterministic: bool = False,
        eps: np.ndarray | None = None,
    ) -> tuple[dict | np.ndarray, np.ndarray, tuple[float, float] | None, np.ndarray, np.ndarray, int, float]:
        x_n = self.obs_rms.normalize(x_raw)
        h_new, _ = self._gru_step(x_n, h_prev)
        p = self.actor_params
        mean = h_new @ p['W_act'] + p['b_act']
        std = np.exp(p['log_std'])
        if deterministic:
            z = mean.copy()
        else:
            noise = np.zeros_like(mean) if eps is None else eps
            z = mean + std * noise
        env_act, d_xyz, alpha_pair = self.decode_action(z, a_base_xyz)
        logits = h_new @ p['W_cls'] + p['b_cls']
        pred_cls = int(np.argmax(logits))
        pred_yaw = float((h_new @ p['W_yaw'] + p['b_yaw'])[0])
        return env_act, d_xyz, alpha_pair, z, h_new, pred_cls, pred_yaw

    def critic_forward(self, c_n: np.ndarray) -> tuple[np.ndarray, tuple]:
        cp = self.critic_params
        h1 = np.tanh(c_n @ cp['W1'] + cp['b1'])
        h2 = np.tanh(h1 @ cp['W2'] + cp['b2'])
        v = (h2 @ cp['W3'] + cp['b3']).squeeze(-1)
        return v, (c_n, h1, h2)

    def _actor_seq_forward_and_bptt(
        self,
        x_seq: np.ndarray,      # (B, T, D) normalized
        h0: np.ndarray,         # (B, H)
        mask: np.ndarray,       # (B, T) float 0/1
        z_act: np.ndarray,      # (B, T, A)
        old_lp: np.ndarray,     # (B, T)
        adv: np.ndarray,        # (B, T)
        cls_t: np.ndarray,      # (B, T) int
        yaw_t: np.ndarray,      # (B, T, 1) float (rail_yaw_rad)
        clip_eps: float,
        entropy_coef: float,
        aux_cls_coef: float,
        aux_yaw_coef: float,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        B, T, _ = x_seq.shape
        H = self.hidden_dim
        p = self.actor_params
        std = np.exp(p['log_std'])
        inv_var = 1.0 / (std * std + 1e-12)

        # Forward unroll
        hs = np.empty((B, T, H), dtype=float)
        caches = []
        h = h0
        for t in range(T):
            h, cache = self._gru_step(x_seq[:, t, :], h)
            hs[:, t, :] = h
            caches.append(cache)

        mean = hs @ p['W_act'] + p['b_act']  # (B, T, A)
        diff = z_act - mean
        lp = -0.5 * np.sum((diff * diff) * inv_var + 2.0 * p['log_std'] + math.log(2.0 * math.pi), axis=-1)
        log_ratio = np.clip(lp - old_lp, -20.0, 20.0)
        ratio = np.exp(log_ratio)

        # Clipped PPO surrogate gradient w.r.t. lp
        m_sum = max(float(np.sum(mask)), 1.0)
        unclipped = ratio * adv
        clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        use_unclipped = (unclipped <= clipped).astype(float)
        d_lp = -(use_unclipped * ratio * adv * mask) / m_sum  # (B, T)

        d_mean = d_lp[..., None] * (diff * inv_var)  # (B, T, A)
        d_log_std = np.sum(d_lp[..., None] * ((diff * diff) * inv_var - 1.0), axis=(0, 1)) - entropy_coef * np.ones(self.act_dim)

        # Aux classification cross-entropy
        logits = hs @ p['W_cls'] + p['b_cls']
        logits_s = logits - np.max(logits, axis=-1, keepdims=True)
        ex = np.exp(logits_s)
        probs = ex / np.sum(ex, axis=-1, keepdims=True)
        cls_acc = float(np.sum((np.argmax(probs, axis=-1) == cls_t) * mask) / m_sum)
        d_logits = probs.copy()
        bi = np.arange(B)[:, None]
        ti = np.arange(T)[None, :]
        d_logits[bi, ti, cls_t] -= 1.0
        d_logits *= (aux_cls_coef * mask[..., None]) / m_sum

        # Aux rail_yaw_rad MSE loss (training supervision only)
        pred_yaw = hs @ p['W_yaw'] + p['b_yaw']  # (B, T, 1)
        d_pred_yaw = (aux_yaw_coef * 2.0 * (pred_yaw - yaw_t) * mask[..., None]) / m_sum

        # Gradients into output heads & hs
        hs_flat = hs.reshape(B * T, H)
        d_W_act = hs_flat.T @ d_mean.reshape(B * T, self.act_dim)
        d_b_act = np.sum(d_mean, axis=(0, 1))
        d_W_cls = hs_flat.T @ d_logits.reshape(B * T, self.n_cls)
        d_b_cls = np.sum(d_logits, axis=(0, 1))
        d_W_yaw = hs_flat.T @ d_pred_yaw.reshape(B * T, 1)
        d_b_yaw = np.sum(d_pred_yaw, axis=(0, 1))

        d_hs = d_mean @ p['W_act'].T + d_logits @ p['W_cls'].T + d_pred_yaw @ p['W_yaw'].T

        # Backward through time (GRU BPTT)
        d_W_x = np.zeros_like(p['W_x'])
        d_W_h = np.zeros_like(p['W_h'])
        d_b_g = np.zeros_like(p['b_g'])
        dh_next = np.zeros((B, H), dtype=float)
        W_hz = p['W_h'][:, :H]
        W_hr = p['W_h'][:, H:2 * H]
        W_hn = p['W_h'][:, 2 * H:]

        for t in range(T - 1, -1, -1):
            x_n, h_prev, z_g, r_g, n_g = caches[t]
            dh = d_hs[:, t, :] + dh_next
            dn = dh * (1.0 - z_g) * (1.0 - n_g * n_g)
            dz = dh * (h_prev - n_g) * z_g * (1.0 - z_g)
            d_rh = dn @ W_hn.T
            dr = d_rh * h_prev * r_g * (1.0 - r_g)

            dg = np.concatenate([dz, dr, dn], axis=-1)  # (B, 3H)
            d_W_x += x_n.T @ dg
            d_b_g += np.sum(dg, axis=0)
            d_W_h[:, :H] += h_prev.T @ dz
            d_W_h[:, H:2 * H] += h_prev.T @ dr
            d_W_h[:, 2 * H:] += (r_g * h_prev).T @ dn
            dh_next = dh * z_g + dz @ W_hz.T + dr @ W_hr.T + d_rh * r_g

        approx_kl = float(np.sum(((ratio - 1.0) - log_ratio) * mask) / m_sum)
        grads = {
            'W_x': d_W_x, 'W_h': d_W_h, 'b_g': d_b_g,
            'W_act': d_W_act, 'b_act': d_b_act, 'log_std': d_log_std,
            'W_cls': d_W_cls, 'b_cls': d_b_cls, 'W_yaw': d_W_yaw, 'b_yaw': d_b_yaw,
        }
        return grads, {'approx_kl': approx_kl, 'aux_cls_acc': cls_acc}

    def ppo_update(
        self,
        trajs: list[dict],
        rng: np.random.Generator,
        actor_lr: float = 5.0e-4,
        critic_lr: float = 1.4e-3,
        gamma: float = 0.995,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        chunk_len: int = BPTT_CHUNK_LEN,
        chunk_minibatch: int = 8,
        critic_minibatch: int = 256,
        entropy_coef: float = 0.0004,
        aux_cls_coef: float = 0.08,
        aux_yaw_coef: float = 0.25,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.025,
        weight_decay: float = 1e-4,
        log_std_max: float = -0.90,
        log_std_min: float = -2.20,
    ) -> dict[str, float | list[float]]:
        # 1. Compute GAE advantages & returns + pre-update old log-probs + chunk initial hidden states
        all_c_in, all_ret, all_val = [], [], []
        chunks: list[dict] = []
        ep_returns = []
        p = self.actor_params
        std_old = np.exp(p['log_std'])
        inv_var_old = 1.0 / (std_old * std_old + 1e-12)

        for tr in trajs:
            obs = tr['obs']
            priv = tr['priv']
            rew = tr['rewards']
            T_ep = obs.shape[0]
            ep_returns.append(float(np.sum(rew)))
            c_in = np.concatenate([obs, priv], axis=-1)
            c_n = self.critic_rms.normalize(c_in)
            vals, _ = self.critic_forward(c_n)
            last_c = np.concatenate([tr['last_obs'], tr['last_priv']], axis=-1)[None, :]
            last_v = 0.0 if tr['terminated'] else float(self.critic_forward(self.critic_rms.normalize(last_c))[0][0])

            adv = np.zeros(T_ep, dtype=float)
            gae = 0.0
            for t in range(T_ep - 1, -1, -1):
                v_next = last_v if t == T_ep - 1 else float(vals[t + 1])
                delta = float(rew[t]) + gamma * v_next - float(vals[t])
                gae = delta + gamma * lam * gae
                adv[t] = gae
            ret = adv + vals
            all_c_in.append(c_n)
            all_ret.append(ret)
            all_val.append(vals)

            # Roll GRU forward to record exact chunk-boundary h0 & old_lp
            x_n = self.obs_rms.normalize(obs)
            h = self.init_hidden()
            hs_pre = np.empty((T_ep, self.hidden_dim), dtype=float)
            old_lp = np.empty(T_ep, dtype=float)
            for t in range(T_ep):
                hs_pre[t] = h
                h, _ = self._gru_step(x_n[t], h)
                mean_t = h @ p['W_act'] + p['b_act']
                diff_t = tr['z'][t] - mean_t
                old_lp[t] = -0.5 * float(np.sum(diff_t * diff_t * inv_var_old + 2.0 * p['log_std'] + math.log(2.0 * math.pi)))

            for s in range(0, T_ep, chunk_len):
                e = min(s + chunk_len, T_ep)
                L = e - s
                pad = chunk_len - L
                mask = np.zeros(chunk_len, dtype=float)
                mask[:L] = 1.0
                def _pad(a, shape_rest):
                    out = np.zeros((chunk_len, *shape_rest), dtype=a.dtype)
                    out[:L] = a[s:e]
                    return out
                chunks.append({
                    'x_n': _pad(x_n, (self.obs_dim,)),
                    'h0': hs_pre[s].copy(),
                    'mask': mask,
                    'z': _pad(tr['z'], (self.act_dim,)),
                    'old_lp': _pad(old_lp, ()),
                    'adv': _pad(adv, ()),
                    'cls_t': _pad(tr['contact_labels'], ()),
                    'yaw_t': _pad(tr['rail_yaws'], (1,)),
                })

        # Normalize advantages globally
        all_adv_vals = np.concatenate([c['adv'][c['mask'] > 0.5] for c in chunks])
        a_mean = float(np.mean(all_adv_vals))
        a_std = float(np.std(all_adv_vals)) + 1e-8
        for c in chunks:
            c['adv'] = np.where(c['mask'] > 0.5, (c['adv'] - a_mean) / a_std, 0.0)

        # 2. Critic updates
        C_all = np.concatenate(all_c_in, axis=0)
        R_all = np.concatenate(all_ret, axis=0)
        V_all = np.concatenate(all_val, axis=0)
        n_steps = int(C_all.shape[0])
        ev = float(1.0 - np.var(R_all - V_all) / (np.var(R_all) + 1e-8))

        for _ in range(epochs):
            perm = rng.permutation(n_steps)
            for s in range(0, n_steps, critic_minibatch):
                idx = perm[s:s + critic_minibatch]
                v_pred, (c_b, h1, h2) = self.critic_forward(C_all[idx])
                dv = (2.0 * (v_pred - R_all[idx]) / float(idx.size))[:, None]
                cp = self.critic_params
                dW3 = h2.T @ dv
                db3 = np.sum(dv, axis=0)
                dh2 = (dv @ cp['W3'].T) * (1.0 - h2 * h2)
                dW2 = h1.T @ dh2
                db2 = np.sum(dh2, axis=0)
                dh1 = (dh2 @ cp['W2'].T) * (1.0 - h1 * h1)
                dW1 = c_b.T @ dh1
                db1 = np.sum(dh1, axis=0)
                cgrads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
                gn = math.sqrt(sum(float(np.sum(g * g)) for g in cgrads.values()))
                if gn > max_grad_norm:
                    sc = max_grad_norm / (gn + 1e-12)
                    cgrads = {k: g * sc for k, g in cgrads.items()}
                self._opt_critic.step(cp, cgrads, lr=critic_lr, weight_decay=weight_decay)

        # 3. Actor GRU-BPTT PPO updates with true post-step KL trust-region rollback
        n_chunks = len(chunks)
        last_kl = 0.0
        last_aux = 0.0
        early_stop = False
        for _ in range(epochs):
            if early_stop:
                break
            perm = rng.permutation(n_chunks)
            for s in range(0, n_chunks, chunk_minibatch):
                batch = [chunks[i] for i in perm[s:s + chunk_minibatch]]
                x_b = np.stack([c['x_n'] for c in batch], axis=0)
                h0_b = np.stack([c['h0'] for c in batch], axis=0)
                m_b = np.stack([c['mask'] for c in batch], axis=0)
                z_b = np.stack([c['z'] for c in batch], axis=0)
                olp_b = np.stack([c['old_lp'] for c in batch], axis=0)
                adv_b = np.stack([c['adv'] for c in batch], axis=0)
                cls_b = np.stack([c['cls_t'] for c in batch], axis=0)
                yaw_b = np.stack([c['yaw_t'] for c in batch], axis=0)

                snap = {k: v.copy() for k, v in self.actor_params.items()}
                grads, info_b = self._actor_seq_forward_and_bptt(
                    x_b, h0_b, m_b, z_b, olp_b, adv_b, cls_b, yaw_b,
                    clip_eps=clip_eps, entropy_coef=entropy_coef,
                    aux_cls_coef=aux_cls_coef, aux_yaw_coef=aux_yaw_coef,
                )
                gn = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
                if gn > max_grad_norm:
                    sc = max_grad_norm / (gn + 1e-12)
                    grads = {k: g * sc for k, g in grads.items()}
                self._opt_actor.step(self.actor_params, grads, lr=actor_lr, weight_decay=weight_decay)
                self.actor_params['log_std'] = np.clip(self.actor_params['log_std'], log_std_min, log_std_max)

                _, post_info = self._actor_seq_forward_and_bptt(
                    x_b, h0_b, m_b, z_b, olp_b, adv_b, cls_b, yaw_b,
                    clip_eps=clip_eps, entropy_coef=entropy_coef,
                    aux_cls_coef=aux_cls_coef, aux_yaw_coef=aux_yaw_coef,
                )
                last_aux = info_b['aux_cls_acc']
                if post_info['approx_kl'] > 1.8 * target_kl:
                    for k in self.actor_params:
                        self.actor_params[k] = snap[k]
                    early_stop = True
                    break
                last_kl = post_info['approx_kl']

        return {
            'n_steps': n_steps,
            'mean_return': float(np.mean(ep_returns)),
            'explained_var': ev,
            'approx_kl': last_kl,
            'aux_cls_acc': last_aux,
            'std': np.exp(self.actor_params['log_std']).tolist(),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            'obs_dim': np.array([self.obs_dim]),
            'hidden_dim': np.array([self.hidden_dim]),
            'critic_priv_dim': np.array([self.critic_priv_dim]),
            'mode': np.array([self.mode]),
            'obs_mean': self.obs_rms.mean,
            'obs_var': self.obs_rms.var,
            'obs_count': np.array([self.obs_rms.count]),
            'critic_mean': self.critic_rms.mean,
            'critic_var': self.critic_rms.var,
            'critic_count': np.array([self.critic_rms.count]),
        }
        for k, v in self.actor_params.items():
            payload[f'actor_{k}'] = v
        for k, v in self.critic_params.items():
            payload[f'critic_{k}'] = v
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> 'DrawerResidualGRUPPO':
        d = np.load(path, allow_pickle=False)
        pol = cls(
            obs_dim=int(d['obs_dim'][0]),
            mode=str(d['mode'][0]),
            critic_priv_dim=int(d['critic_priv_dim'][0]),
            hidden_dim=int(d['hidden_dim'][0]),
        )
        pol.obs_rms.mean = d['obs_mean'].copy()
        pol.obs_rms.var = d['obs_var'].copy()
        pol.obs_rms.count = float(d['obs_count'][0])
        pol.critic_rms.mean = d['critic_mean'].copy()
        pol.critic_rms.var = d['critic_var'].copy()
        pol.critic_rms.count = float(d['critic_count'][0])
        for k in pol.actor_params:
            pol.actor_params[k] = d[f'actor_{k}'].copy()
        for k in pol.critic_params:
            pol.critic_params[k] = d[f'critic_{k}'].copy()
        return pol
