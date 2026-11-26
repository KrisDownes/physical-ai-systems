"""Unit tests for Phase 1 blind insertion RL: M1/M2/M3 obs boundaries, 1D-CNN gradients, disjointness, replay."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_insertion.benchmarks import (
    build_regime_a_bolted,
    build_regime_b_unbolted,
    build_split_c_tight_0_5mm,
)
from projects.blind_insertion.env import BlindInsertionEnv, InsertionTaskSpec
from projects.blind_insertion.geometry import SEEN_SHAPES, UNSEEN_SHAPES
from projects.blind_insertion.phase1_policy import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    M3_TORQUE_MAX,
    M3_TORQUE_MIN,
    PRIVILEGED_CRITIC_DIM,
    TemporalAsymmetricPPO,
    TemporalProprioBuilder,
    compute_gae,
)
from projects.blind_insertion.phase1_train import (
    compute_step_reward,
    reset_cached_env_inplace,
    sample_training_batch,
    verify_train_disjointness,
)


class Phase1BlindInsertionTest(unittest.TestCase):
    def test_m1_pos_only_zero_force_zero_tracking_error_and_zero_privileged_keys(self):
        base_obs = {
            'q_quantized': np.array([0.05, -0.1, 0.3, 1.3, 0.0, 0.35]),
            'q_dot_quantized': np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.0]),
            'ee_pose_fk': np.array([0.22, 0.0, 0.026, 1.0, 0.0, 0.0, 0.0]),
            'ee_twist_fk': np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
            'prev_action': np.array([0.001, -0.001, -0.001, 1.0, 1.0, 1.0, 1.0, 1.0]),
            'tracking_error': np.zeros(6),
            'actuator_force_noisy': np.zeros(6),
            'torque_scale': np.ones(5),
            'time_s': 0.10,
        }
        obs_pert = dict(base_obs)
        obs_pert['actuator_force_noisy'] = np.array([2.9, -2.9, 2.5, -1.8, 1.2, -2.0])
        obs_pert['tracking_error'] = np.array([0.15, -0.20, 0.18, -0.12, 0.09, 0.05])
        obs_pert['torque_scale'] = np.array([0.20, 0.25, 0.35, 0.45, 0.55])

        # M1 must be 100% invariant to actuator_force_noisy, tracking_error, AND torque_scale!
        b1 = TemporalProprioBuilder('pos_only')
        b2 = TemporalProprioBuilder('pos_only')
        h1, s1 = b1.step(base_obs)
        h2, s2 = b2.step(obs_pert)
        np.testing.assert_allclose(h1, h2, atol=1e-12)
        np.testing.assert_allclose(s1, s2, atol=1e-12)

        # M2 must see force & tracking_error, and ignore torque_scale
        bm2_a = TemporalProprioBuilder('current')
        bm2_b = TemporalProprioBuilder('current')
        hm2_a, _ = bm2_a.step(base_obs)
        obs_ts_only = dict(base_obs)
        obs_ts_only['torque_scale'] = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        hm2_b, _ = bm2_b.step(obs_ts_only)
        np.testing.assert_allclose(hm2_a, hm2_b, atol=1e-12)

        # Runtime privileged-key guard fires on all 3 modes (M1, M2, M3)
        bad_obs = dict(base_obs)
        bad_obs['socket_xy_error'] = np.zeros(2)
        for mode in ('pos_only', 'current', 'current_torque'):
            with self.assertRaises(AssertionError):
                TemporalProprioBuilder(mode).step(bad_obs)

    def test_split_c_composition_and_unseen_shapes_never_in_training(self):
        sc = build_split_c_tight_0_5mm()
        self.assertEqual(len(sc), 30)
        self.assertEqual(sum(t.bolted for t in sc), 15)
        self.assertEqual(sum(not t.bolted for t in sc), 15)
        self.assertEqual(sum(t.shape in SEEN_SHAPES for t in sc), 15)
        self.assertEqual(sum(t.shape in UNSEEN_SHAPES for t in sc), 15)
        self.assertTrue(all(abs(t.clearance_m - 0.0005) < 1e-9 for t in sc))

        # Disjoint seeds across Splits A, B, C
        sa = {t.seed for t in build_regime_a_bolted()}
        sb = {t.seed for t in build_regime_b_unbolted()}
        sc_seeds = {t.seed for t in sc}
        self.assertTrue(sa.isdisjoint(sb) and sa.isdisjoint(sc_seeds) and sb.isdisjoint(sc_seeds))

        # Training batches contain zero unseen shapes and zero overlap with Splits A/B/C
        rng = np.random.default_rng(7)
        batch, cur = sample_training_batch(rng, batch_size=96, seed_cursor=10000, progress_frac=0.5)
        self.assertEqual(cur, 10096)
        verify_train_disjointness(batch)

    def test_1d_cnn_actor_aux_and_critic_analytic_gradients_match_finite_difference(self):
        rng = np.random.default_rng(23)
        pol = TemporalAsymmetricPPO(
            step_dim=6, spatial_dim=8, critic_dim=10, use_torque_limit=True,
            conv_filters=6, latent_dim=8, critic_h1=10, critic_h2=8, seed=4,
        )
        pol.W_mu[:] = rng.normal(0.0, 0.1, size=pol.W_mu.shape)
        pol.b_mu[:] = rng.normal(0.0, 0.1, size=pol.b_mu.shape)

        H = rng.normal(0.0, 1.0, size=(8, 20, 6))
        S = rng.normal(0.0, 1.0, size=(8, 8))
        C = rng.normal(0.0, 1.0, size=(8, 10))
        Z = rng.normal(0.0, 0.5, size=(8, pol.act_dim))
        old_lp = pol.step_log_prob(H, S, Z) + rng.normal(0.0, 0.05, size=8)
        adv = rng.normal(0.0, 1.0, size=8)
        cls_lbl = rng.integers(0, 5, size=8)
        dir_tgt = rng.normal(0.0, 1.0, size=(8, 2))
        dir_tgt /= np.maximum(np.linalg.norm(dir_tgt, axis=1, keepdims=True), 1e-6)
        dir_mask = np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=float)

        _, a_grads, _, _, _, _, _ = pol.actor_and_aux_loss_and_grad(
            H, S, Z, old_lp, adv, cls_lbl, dir_tgt, dir_mask,
        )
        eps = 1e-6
        for name, param in pol.actor_params().items():
            num = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                orig = param[idx]
                param[idx] = orig + eps
                lp, _, _, _, _, _, _ = pol.actor_and_aux_loss_and_grad(H, S, Z, old_lp, adv, cls_lbl, dir_tgt, dir_mask)
                param[idx] = orig - eps
                lm, _, _, _, _, _, _ = pol.actor_and_aux_loss_and_grad(H, S, Z, old_lp, adv, cls_lbl, dir_tgt, dir_mask)
                param[idx] = orig
                num[idx] = (lp - lm) / (2.0 * eps)
                it.iternext()
            np.testing.assert_allclose(a_grads[name], num, atol=1e-5, rtol=1e-4)

        old_v, _, _ = pol.forward_critic(C)
        old_v = old_v + rng.normal(0.0, 0.05, size=8)
        ret = rng.normal(0.5, 0.5, size=8)
        _, c_grads = pol.critic_loss_and_grad(C, old_v, ret)
        for name, param in pol.critic_params().items():
            num = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                orig = param[idx]
                param[idx] = orig + eps
                vp, _ = pol.critic_loss_and_grad(C, old_v, ret)
                param[idx] = orig - eps
                vm, _ = pol.critic_loss_and_grad(C, old_v, ret)
                param[idx] = orig
                num[idx] = (vp - vm) / (2.0 * eps)
                it.iternext()
            np.testing.assert_allclose(c_grads[name], num, atol=1e-5, rtol=1e-4)

    def test_cached_env_exactness_and_checkpoint_deterministic_replay(self):
        t1 = InsertionTaskSpec(
            trial_id='c1', seed=10100, shape='square', clearance_m=0.0005, bolted=False,
            socket_xy=(0.215, 0.005), misalign_xy=(0.008, -0.006), grasp_offset_xy=(0.001, -0.001),
            mu_peg_hole=0.20, mu_table=0.40, socket_mass_kg=0.200, backlash_rad=0.004,
        )
        t2 = InsertionTaskSpec(
            trial_id='c2', seed=10101, shape='square', clearance_m=0.0005, bolted=False,
            socket_xy=(0.228, -0.009), misalign_xy=(-0.010, 0.007), grasp_offset_xy=(-0.0012, 0.0008),
            mu_peg_hole=0.42, mu_table=0.52, socket_mass_kg=0.310, backlash_rad=0.0055,
        )
        ec = BlindInsertionEnv(task=t1, render=False)
        ef = BlindInsertionEnv(task=t2, render=False)
        try:
            oc, ic = reset_cached_env_inplace(ec, t2)
            of, if_ = ef.reset()
            np.testing.assert_allclose(oc['q_quantized'], of['q_quantized'], atol=1e-12)
            np.testing.assert_allclose(ic['privileged']['socket_xy_error'], if_['privileged']['socket_xy_error'], atol=1e-12)
        finally:
            ec.close()
            ef.close()

        # Deterministic checkpoint replay + frozen RMS + M3 torque bounds [0.20, 1.0]
        rng = np.random.default_rng(13)
        b = TemporalProprioBuilder('current_torque')
        pol = TemporalAsymmetricPPO(step_dim=b.step_dim, use_torque_limit=True, seed=9)
        # Verify zero-init actor head outputs exact 0 motion delta
        h0 = np.zeros((20, b.step_dim))
        s0 = np.zeros(b.spatial_dim)
        act0, _, _, _ = pol.act_step(h0, s0, deterministic=True)
        np.testing.assert_allclose(act0[:3], 0.0, atol=1e-12)
        self.assertTrue(np.all(act0[3:8] >= M3_TORQUE_MIN) and np.all(act0[3:8] <= M3_TORQUE_MAX))

        pol.W_mu[:] = rng.normal(0.0, 0.25, size=pol.W_mu.shape)
        pol.step_rms.update(rng.normal(0.1, 1.2, size=(100, b.step_dim)))
        pol.spatial_rms.update(rng.normal(0.0, 1.0, size=(100, b.spatial_dim)))
        rms_before = pol.step_rms.get_state('step_rms')

        with tempfile.TemporaryDirectory() as tmp:
            p_ckpt = Path(tmp) / 'm3.npz'
            pol.save(p_ckpt)
            loaded = TemporalAsymmetricPPO.load(p_ckpt)
            outs = []
            for p in (pol, loaded):
                env = BlindInsertionEnv(task=t2, render=False)
                try:
                    obs, info = env.reset()
                    tb = TemporalProprioBuilder('current_torque')
                    trace = []
                    for _ in range(12):
                        h, s = tb.step(obs)
                        ea, _, c_p, _ = p.act_step(h, s, deterministic=True)
                        self.assertTrue(np.all(ea[3:8] >= M3_TORQUE_MIN - 1e-12) and np.all(ea[3:8] <= M3_TORQUE_MAX + 1e-12))
                        obs, _, _, _, info = env.step(ea)
                        trace.append((ea.tolist(), c_p, info['privileged']['peg_depth']))
                    outs.append(trace)
                finally:
                    env.close()
            self.assertEqual(outs[0], outs[1])
            np.testing.assert_array_equal(pol.step_rms.mean, rms_before['step_rms_mean'])
            self.assertTrue(np.all(loaded.log_std >= LOG_STD_MIN) and np.all(loaded.log_std <= LOG_STD_MAX))

    def test_off_plate_depth_reward_exploit_is_blocked(self):
        # 1. Analytical step check: dropping off outer edge (radial_err=44 mm, peg_depth=+24 mm)
        prev_priv = {
            'socket_xy_error': np.array([0.041, 0.0]),
            'peg_depth': 0.000,
            'contact_class': 'plate_contact',
            'peak_contact_force_N': 10.0,
            'socket_displacement_mm': 0.0,
            'socket_tilt_deg': 0.0,
            'shape': 'round',
            'bolted': True,
        }
        cur_priv_off = dict(prev_priv, socket_xy_error=np.array([0.044, 0.0]), peg_depth=0.024, contact_class='free')
        act = np.array([0.003, 0.0, -0.003, 1.0, 1.0, 1.0, 1.0, 1.0])
        info = {'success': False, 'failure_mode': None}
        r_off, off_term = compute_step_reward(prev_priv, cur_priv_off, act, act, info, 10, 180, {'rim': False, 'entering': False})
        self.assertTrue(off_term)
        self.assertLess(r_off, -2.0)

        # 2. Full MuJoCo episode check: walking a peg straight off the plate edge (radial_err > 30 mm, z -> 0)
        #    MUST yield strictly NEGATIVE total episode return (never positive shaping!).
        task = InsertionTaskSpec(
            trial_id='test-walkoff', seed=10555, shape='square', clearance_m=0.0010,
            bolted=True, socket_xy=(0.220, 0.0), misalign_xy=(0.014, 0.0),
        )
        env = BlindInsertionEnv(task=task, render=False)
        try:
            obs, info_e = env.reset()
            prev_p = info_e['privileged']
            prev_a = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            milestones = {'rim': False, 'entering': False}
            visited: set[tuple[int, int]] = set()
            start_xy = obs['ee_pose_fk'][:2].copy()
            total_ret = 0.0
            triggered_off = False
            for st in range(1, 60):
                a = np.array([0.004, 0.0, -0.002, 1.0, 1.0, 1.0, 1.0, 1.0])
                obs, _, term, trunc, info_e = env.step(a)
                r_st, off_t = compute_step_reward(
                    prev_p, info_e['privileged'], a, prev_a, info_e, st, 180, milestones,
                    visited_plate_cells=visited, ee_rel_xy=obs['ee_pose_fk'][:2] - start_xy,
                )
                total_ret += r_st
                prev_p = info_e['privileged']
                prev_a = a
                if off_t:
                    triggered_off = True
                    break
                if term or trunc:
                    break
            self.assertTrue(triggered_off, 'Expected off-plate guard to trigger when walking off plate')
            self.assertLess(total_ret, -2.0, f'Walking off plate must give strictly negative total return, got {total_ret}')
        finally:
            env.close()

        # 3. Genuine hole entry (radial_err=0.5 mm < hole_r, peg_depth=+5 mm) must give positive reward!
        prev_in = dict(prev_priv, socket_xy_error=np.array([0.0005, 0.0]), peg_depth=0.0)
        cur_in = dict(prev_in, peg_depth=0.005, contact_class='entering')
        r_in, off_term_in = compute_step_reward(prev_in, cur_in, act, act, info, 10, 180, {'rim': True, 'entering': True})
        self.assertFalse(off_term_in)
        self.assertGreater(r_in, 0.15)

        # 4. Verify M3 episode-0 zero-init outputs exact torque_scale = 0.75
        b = TemporalProprioBuilder('current_torque')
        pol_m3 = TemporalAsymmetricPPO(step_dim=b.step_dim, use_torque_limit=True, seed=1)
        ea0, _, _, _ = pol_m3.act_step(np.zeros((20, b.step_dim)), np.zeros(b.spatial_dim), deterministic=True)
        np.testing.assert_allclose(ea0[:3], 0.0, atol=1e-12)
        np.testing.assert_allclose(ea0[3:8], 0.75, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
