"""Unit tests for Phase 1 blind drawer RL: GRU BPTT finite-diff grads, relative obs guard, init, replay."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_drawer.benchmarks import (
    TRAIN_SEED_OFFSET,
    build_split_a_bolted_sticky,
    build_split_b_unbolted_cabinet,
    build_split_ood_light_height_angle,
)
from projects.blind_drawer.controllers import NominalStraightBackController
from projects.blind_drawer.env import BlindDrawerEnv, DrawerTaskSpec
from projects.blind_drawer.phase1_policy import (
    D1_INIT_ALPHA_ARM,
    D1_INIT_ALPHA_LAT,
    DrawerObsBuilder,
    DrawerResidualGRUPPO,
)
from projects.blind_drawer.phase1_train import sample_drawer_train_batch


class Phase1BlindDrawerTest(unittest.TestCase):
    def test_obs_builder_rejects_absolute_ee_pos_and_privileged_keys(self):
        env = BlindDrawerEnv(task=DrawerTaskSpec(seed=1, bolted=False, rail_yaw_deg=14.0), render=False)
        try:
            obs, _ = env.reset()
            b = DrawerObsBuilder(mode='res_gravcomp_tl')
            vec = b.build(obs, np.array([-0.002, 0.0, -0.001]))
            self.assertEqual(vec.shape, (37,))
            self.assertTrue(np.all(np.isfinite(vec)))

            bad_obs = dict(obs)
            bad_obs['ee_pose_fk'] = np.zeros(7)
            with self.assertRaises(AssertionError):
                b.build(bad_obs, np.zeros(3))
        finally:
            env.close()

    def test_d2_zero_init_alpha_and_checkpoint_deterministic_replay(self):
        pol = DrawerResidualGRUPPO(obs_dim=37, mode='res_gravcomp_tl', seed=12)
        h = pol.init_hidden()
        base = np.array([-0.0025, 0.0, -0.001], dtype=float)
        act, d_xyz, alpha_pair, _, h1, cls1, dir1 = pol.act_step(np.zeros(37), h, base, deterministic=True)
        np.testing.assert_allclose(d_xyz, base, atol=1e-9)
        self.assertIsNotNone(alpha_pair)
        self.assertAlmostEqual(alpha_pair[0], D1_INIT_ALPHA_LAT, places=4)
        self.assertAlmostEqual(alpha_pair[1], D1_INIT_ALPHA_ARM, places=4)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'ckpt.npz'
            pol.save(p)
            pol2 = DrawerResidualGRUPPO.load(p)
            act2, d_xyz2, alpha_pair2, _, h2, cls2, yaw2 = pol2.act_step(np.zeros(37), pol2.init_hidden(), base, deterministic=True)
            np.testing.assert_array_equal(d_xyz, d_xyz2)
            self.assertEqual(alpha_pair, alpha_pair2)
            np.testing.assert_array_equal(h1, h2)
            self.assertEqual(cls1, cls2)
            self.assertEqual(dir1, yaw2)

    def test_gru_hidden_state_resets_cleanly_at_episode_boundaries(self):
        pol = DrawerResidualGRUPPO(obs_dim=37, mode='res_gravcomp_tl', seed=19)
        rng = np.random.default_rng(19)
        h = pol.init_hidden()
        base = np.array([-0.002, 0.0, -0.001])
        for _ in range(10):
            _, _, _, _, h, _, _ = pol.act_step(rng.normal(0.0, 1.0, 37), h, base, deterministic=True)
        self.assertGreater(float(np.linalg.norm(h)), 0.1)
        x0 = rng.normal(0.0, 1.0, 37)
        _, d_reset, a_reset, _, h_reset, _, _ = pol.act_step(x0, pol.init_hidden(), base, deterministic=True)
        _, d_fresh, a_fresh, _, h_fresh, _, _ = pol.act_step(x0, np.zeros_like(h), base, deterministic=True)
        np.testing.assert_array_equal(d_reset, d_fresh)
        self.assertEqual(a_reset, a_fresh)
        np.testing.assert_array_equal(h_reset, h_fresh)

    def test_train_batch_strictly_in_distribution_and_disjoint_from_ood(self):
        eval_seeds = {
            t.seed
            for fn in (build_split_a_bolted_sticky, build_split_b_unbolted_cabinet, build_split_ood_light_height_angle)
            for t in fn()
        }
        rng = np.random.default_rng(99)
        trials, nxt = sample_drawer_train_batch(rng, batch_size=40, seed_cursor=TRAIN_SEED_OFFSET + 100000, yaw_max_deg=18.0)
        self.assertEqual(len(trials), 40)
        self.assertEqual(nxt, TRAIN_SEED_OFFSET + 100040)
        for t in trials:
            self.assertNotIn(t.seed, eval_seeds)
            self.assertEqual(t.height_shift_m, 0.0)
            self.assertLessEqual(abs(t.rail_yaw_deg), 18.0 + 1e-9)
            if not t.bolted:
                self.assertGreaterEqual(t.cabinet_mass_kg, 0.400)
                self.assertGreaterEqual(t.mu_table, 0.40)

    def test_gru_bptt_analytic_gradients_match_finite_difference(self):
        rng = np.random.default_rng(7)
        pol = DrawerResidualGRUPPO(obs_dim=8, mode='res_gravcomp_tl', critic_priv_dim=4, hidden_dim=6, seed=7)
        pol.actor_params['W_act'] = rng.normal(0.0, 0.2, size=pol.actor_params['W_act'].shape)

        B, T = 2, 5
        x_b = rng.normal(0.0, 1.0, size=(B, T, 8))
        h0_b = rng.normal(0.0, 0.3, size=(B, 6))
        m_b = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=float)
        z_b = rng.normal(0.0, 0.5, size=(B, T, 5))
        olp_b = rng.normal(-3.0, 0.5, size=(B, T))
        adv_b = rng.normal(0.0, 1.0, size=(B, T))
        cls_b = rng.integers(0, 5, size=(B, T))
        yaw_b = rng.normal(0.0, 0.25, size=(B, T, 1))

        def eval_loss() -> float:
            p = pol.actor_params
            std = np.exp(p['log_std'])
            inv_var = 1.0 / (std * std + 1e-12)
            hs = np.empty((B, T, 6))
            h = h0_b
            for t in range(T):
                h, _ = pol._gru_step(x_b[:, t, :], h)
                hs[:, t, :] = h
            mean = hs @ p['W_act'] + p['b_act']
            diff = z_b - mean
            lp = -0.5 * np.sum((diff * diff) * inv_var + 2.0 * p['log_std'] + np.log(2.0 * np.pi), axis=-1)
            ratio = np.exp(np.clip(lp - olp_b, -20.0, 20.0))
            ms = float(np.sum(m_b))
            ppo_l = -float(np.sum(np.minimum(ratio * adv_b, np.clip(ratio, 0.8, 1.2) * adv_b) * m_b) / ms)
            ent_l = -0.001 * float(np.sum(p['log_std']))
            logits = hs @ p['W_cls'] + p['b_cls']
            ls = logits - np.max(logits, axis=-1, keepdims=True)
            logp = ls - np.log(np.sum(np.exp(ls), axis=-1, keepdims=True))
            cls_l = -0.08 * float(np.sum(logp[np.arange(B)[:, None], np.arange(T)[None, :], cls_b] * m_b) / ms)
            pred_y = hs @ p['W_yaw'] + p['b_yaw']
            yaw_l = 0.25 * float(np.sum(((pred_y - yaw_b) ** 2) * m_b[..., None]) / ms)
            return ppo_l + ent_l + cls_l + yaw_l

        grads, _ = pol._actor_seq_forward_and_bptt(
            x_b, h0_b, m_b, z_b, olp_b, adv_b, cls_b, yaw_b,
            clip_eps=0.2, entropy_coef=0.001, aux_cls_coef=0.08, aux_yaw_coef=0.25,
        )
        eps = 1e-6
        for k in ('W_x', 'W_h', 'b_g', 'W_act', 'b_act', 'log_std', 'W_cls', 'W_yaw'):
            arr = pol.actor_params[k]
            idx = tuple(0 for _ in range(arr.ndim))
            orig = float(arr[idx])
            arr[idx] = orig + eps
            lp_v = eval_loss()
            arr[idx] = orig - eps
            lm_v = eval_loss()
            arr[idx] = orig
            fd = (lp_v - lm_v) / (2.0 * eps)
            self.assertAlmostEqual(float(grads[k][idx]), fd, places=4, msg=f'Grad mismatch on {k}')


if __name__ == '__main__':
    unittest.main()
