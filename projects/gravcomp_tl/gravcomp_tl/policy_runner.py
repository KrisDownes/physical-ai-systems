# SPDX-License-Identifier: MIT
"""Zero-PyTorch pure-NumPy inference runner (< 0.06 ms/step) for frozen GravComp-TL GRU(64) checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

from .core import expand_alpha

ALPHA_MIN = 0.10
ALPHA_MAX = 1.00


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass
class PolicyStepOutput:
    delta_xyz: np.ndarray
    alpha_5dof: np.ndarray
    raw_action: np.ndarray
    aux_contact_idx: int
    aux_extra: np.ndarray


class GravCompPolicyRunner:
    """Standalone NumPy GRU(64) policy runner supporting `blind_drawer` & `blind_insertion` `.npz` files."""

    def __init__(self, ckpt_path: str | Path):
        self.ckpt_path = Path(ckpt_path)
        d = np.load(self.ckpt_path, allow_pickle=False)
        files = set(d.files)

        if 'actor_W_x' in files:
            self.schema = 'blind_drawer'
            self.obs_dim = int(np.asarray(d['obs_dim']).reshape(-1)[0])
            self.hidden_dim = int(np.asarray(d['hidden_dim']).reshape(-1)[0])
            self.mode = str(np.asarray(d['mode']).reshape(-1)[0])
            self.obs_mean = d['obs_mean'].astype(float).copy()
            self.obs_var = d['obs_var'].astype(float).copy()
            self.W_in = self.b_in = self.b_x = None
            self.W_x = d['actor_W_x'].astype(float).copy()
            self.W_h = d['actor_W_h'].astype(float).copy()
            self.b_g = d['actor_b_g'].astype(float).copy()
            self.W_act = d['actor_W_act'].astype(float).copy()
            self.b_act = d['actor_b_act'].astype(float).copy()
            self.W_cls = d['actor_W_cls'].astype(float).copy()
            self.b_cls = d['actor_b_cls'].astype(float).copy()
            self.W_extra = d['actor_W_yaw'].astype(float).copy()
            self.b_extra = d['actor_b_yaw'].astype(float).copy()
        elif 'W_in' in files:
            self.schema = 'blind_insertion'
            self.obs_dim = int(np.asarray(d['obs_dim']).reshape(-1)[0])
            self.hidden_dim = int(np.asarray(d['hidden_dim']).reshape(-1)[0])
            self.mode = str(np.asarray(d['mode']).reshape(-1)[0])
            self.obs_mean = d['obs_rms_mean'].astype(float).copy()
            self.obs_var = d['obs_rms_var'].astype(float).copy()
            self.W_in = d['W_in'].astype(float).copy()
            self.b_in = d['b_in'].astype(float).copy()
            self.W_x = d['W_x'].astype(float).copy()
            self.b_x = d['b_x'].astype(float).copy()
            self.W_h = d['W_h'].astype(float).copy()
            self.b_g = d['b_h'].astype(float).copy()
            self.W_act = d['W_mu'].astype(float).copy()
            self.b_act = d['b_mu'].astype(float).copy()
            self.W_cls = d['W_aux_cls'].astype(float).copy()
            self.b_cls = d['b_aux_cls'].astype(float).copy()
            self.W_extra = d['W_aux_dir'].astype(float).copy()
            self.b_extra = d['b_aux_dir'].astype(float).copy()
        else:
            raise ValueError(f'Unrecognized GravComp-TL checkpoint schema in {self.ckpt_path}')

        self.act_dim = int(self.W_act.shape[1])
        self.h = np.zeros(self.hidden_dim, dtype=float)

    def reset(self) -> None:
        self.h = np.zeros(self.hidden_dim, dtype=float)

    def _gru_step(self, x_n: np.ndarray) -> np.ndarray:
        H = self.hidden_dim
        if self.schema == 'blind_drawer':
            gx = x_n @ self.W_x
            gh = self.h @ self.W_h
            z_g = _sigmoid(gx[:H] + gh[:H] + self.b_g[:H])
            r_g = _sigmoid(gx[H:2 * H] + gh[H:2 * H] + self.b_g[H:2 * H])
            n_g = np.tanh(gx[2 * H:] + (r_g * self.h) @ self.W_h[:, 2 * H:] + self.b_g[2 * H:])
        else:
            e = np.tanh(x_n @ self.W_in + self.b_in)
            gx = e @ self.W_x + self.b_x
            gh = self.h @ self.W_h + self.b_g
            z_g = _sigmoid(gx[:H] + gh[:H])
            r_g = _sigmoid(gx[H:2 * H] + gh[H:2 * H])
            n_g = np.tanh(gx[2 * H:] + r_g * gh[2 * H:])
        self.h = (1.0 - z_g) * n_g + z_g * self.h
        return self.h

    def step(self, x_raw: np.ndarray, a_base_xyz: np.ndarray | None = None) -> PolicyStepOutput:
        x_in = np.asarray(x_raw, dtype=float).reshape(self.obs_dim)
        x_n = np.clip((x_in - self.obs_mean) / np.sqrt(self.obs_var + 1e-6), -8.0, 8.0)
        h = self._gru_step(x_n)
        mu = h @ self.W_act + self.b_act
        base = np.zeros(3, dtype=float) if a_base_xyz is None else np.asarray(a_base_xyz, dtype=float).reshape(3)

        if self.schema == 'blind_drawer':
            d_xyz = base + 0.0050 * np.tanh(mu[:3])
            if self.mode == 'res_gravcomp_tl_5dof':
                a5 = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * _sigmoid(mu[3:8])
            elif self.mode == 'res_gravcomp_tl':
                a5 = expand_alpha(ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * _sigmoid(mu[3:5]))
            else:
                a5 = np.ones(5, dtype=float)
        else:
            g_spiral = float(_sigmoid(mu[3]))
            d_xyz = np.array([
                g_spiral * base[0] + 0.0035 * float(np.tanh(mu[0])),
                g_spiral * base[1] + 0.0035 * float(np.tanh(mu[1])),
                base[2] + 0.0020 * float(np.tanh(mu[2])),
            ], dtype=float)
            if self.mode == 'res_gravcomp_tl':
                a5 = expand_alpha(ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * _sigmoid(mu[4:6]))
            else:
                a5 = np.ones(5, dtype=float)

        return PolicyStepOutput(
            delta_xyz=d_xyz,
            alpha_5dof=a5,
            raw_action=mu,
            aux_contact_idx=int(np.argmax(h @ self.W_cls + self.b_cls)),
            aux_extra=(h @ self.W_extra + self.b_extra).reshape(-1),
        )

    def benchmark_latency_ms(self, n_steps: int = 1000) -> float:
        x = np.zeros(self.obs_dim, dtype=float)
        base = np.zeros(3, dtype=float)
        self.reset()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            self.step(x, base)
        return 1000.0 * (time.perf_counter() - t0) / float(n_steps)
