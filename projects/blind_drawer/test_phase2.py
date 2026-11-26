"""Unit tests for Phase 2 Per-Joint 5-DoF GravComp-TL (`D3_Res_GravComp_5DoF`)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from projects.blind_drawer.benchmarks import build_split_a_bolted_sticky
from projects.blind_drawer.controllers import NominalStraightBackController
from projects.blind_drawer.env import BlindDrawerEnv
from projects.blind_drawer.gravcomp import STS3215_MAX_TORQUE_NM, compute_gravcomp_torque_limits_nm
from projects.blind_drawer.phase1_policy import (
    D3_INIT_ALPHA_5DOF,
    DrawerObsBuilder,
    DrawerResidualGRUPPO,
)


class Phase2BlindDrawer5DoFTest(unittest.TestCase):
    def test_5dof_init_torque_limits_and_checkpoint_replay(self):
        task = build_split_a_bolted_sticky()[0]
        env = BlindDrawerEnv(task)
        obs, _ = env.reset()
        ctrl = NominalStraightBackController(mode='stiff')
        ctrl.reset(obs)
        base_a = ctrl.act(obs)[:3]

        builder = DrawerObsBuilder(mode='res_gravcomp_tl_5dof')
        self.assertEqual(builder.obs_dim, 40)
        pol = DrawerResidualGRUPPO(obs_dim=40, mode='res_gravcomp_tl_5dof', seed=17)
        self.assertEqual(pol.act_dim, 8)
        h = pol.init_hidden()
        x = builder.build(obs, base_a)

        env_act, d_xyz, a5, z, h1, _, _ = pol.act_step(x, h, base_a, deterministic=True)
        self.assertIsInstance(env_act, dict)
        self.assertEqual(len(a5), 5)
        np.testing.assert_allclose(d_xyz, base_a, atol=1e-9)
        np.testing.assert_allclose(a5, D3_INIT_ALPHA_5DOF, atol=1e-5)

        # Verify distinct per-joint torque limits are actually applied in BlindDrawerEnv
        obs2, _, _, _, _ = env.step(env_act)
        lim_expected, _ = compute_gravcomp_torque_limits_nm(obs['q_quantized'], np.asarray(a5))
        np.testing.assert_allclose(obs2['prev_action'][3:8], lim_expected / STS3215_MAX_TORQUE_NM, atol=1e-6)
        self.assertLess(obs2['prev_action'][3], obs2['prev_action'][4])  # pan softer than shoulder_lift

        # Deterministic save/load replay
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'd3.npz'
            pol.save(p)
            pol2 = DrawerResidualGRUPPO.load(p)
        env_act2, d_xyz2, a5_2, z2, h1_2, _, _ = pol2.act_step(x, h, base_a, deterministic=True)
        np.testing.assert_allclose(d_xyz2, d_xyz, atol=1e-12)
        np.testing.assert_allclose(a5_2, a5, atol=1e-12)
        np.testing.assert_allclose(h1_2, h1, atol=1e-12)

    def test_5dof_bptt_gradients_match_finite_difference(self):
        rng = np.random.default_rng(7)
        pol = DrawerResidualGRUPPO(obs_dim=40, mode='res_gravcomp_tl_5dof', hidden_dim=8, seed=7)
        pol.actor_params['W_act'] = rng.normal(0.0, 0.25, size=pol.actor_params['W_act'].shape)
        B, T = 2, 5
        x_seq = rng.normal(0.0, 1.0, size=(B, T, 40))
        h0 = rng.normal(0.0, 0.3, size=(B, 8))
        mask = np.ones((B, T), dtype=float)
        z_act = rng.normal(0.0, 0.5, size=(B, T, 8))
        old_lp = rng.normal(-5.0, 0.2, size=(B, T))
        adv = rng.normal(0.0, 1.0, size=(B, T))
        cls_t = rng.integers(0, 5, size=(B, T))
        yaw_t = rng.normal(0.0, 0.2, size=(B, T, 1))

        grads, _ = pol._actor_seq_forward_and_bptt(
            x_seq, h0, mask, z_act, old_lp, adv, cls_t, yaw_t,
            clip_eps=0.2, entropy_coef=0.001, aux_cls_coef=0.1, aux_yaw_coef=0.2,
        )

        def loss_fn() -> float:
            H = pol.hidden_dim
            p = pol.actor_params
            std = np.exp(p['log_std'])
            inv_var = 1.0 / (std * std + 1e-12)
            hs = np.empty((B, T, H), dtype=float)
            h = h0
            for t in range(T):
                h, _ = pol._gru_step(x_seq[:, t, :], h)
                hs[:, t, :] = h
            mean = hs @ p['W_act'] + p['b_act']
            diff = z_act - mean
            lp = -0.5 * np.sum((diff * diff) * inv_var + 2.0 * p['log_std'] + np.log(2.0 * np.pi), axis=-1)
            ratio = np.exp(np.clip(lp - old_lp, -20.0, 20.0))
            m_sum = float(np.sum(mask))
            surr = np.minimum(ratio * adv, np.clip(ratio, 0.8, 1.2) * adv)
            l_ppo = -float(np.sum(surr * mask) / m_sum)
            l_ent = -0.001 * float(np.sum(p['log_std']))
            logits = hs @ p['W_cls'] + p['b_cls']
            logits -= np.max(logits, axis=-1, keepdims=True)
            logp = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
            l_cls = -0.1 * float(np.sum(logp[np.arange(B)[:, None], np.arange(T)[None, :], cls_t] * mask) / m_sum)
            pred_yaw = hs @ p['W_yaw'] + p['b_yaw']
            l_yaw = 0.2 * float(np.sum(((pred_yaw - yaw_t) ** 2) * mask[..., None]) / m_sum)
            return l_ppo + l_ent + l_cls + l_yaw

        eps = 1e-6
        for name in ('W_act', 'b_act', 'log_std', 'W_h'):
            param = pol.actor_params[name]
            idx = tuple(0 for _ in param.shape)
            orig = float(param[idx])
            param[idx] = orig + eps
            lp = loss_fn()
            param[idx] = orig - eps
            lm = loss_fn()
            param[idx] = orig
            fd = (lp - lm) / (2.0 * eps)
            self.assertAlmostEqual(float(grads[name][idx]), fd, places=5, msg=f'Grad mismatch on {name}')


if __name__ == '__main__':
    unittest.main()
