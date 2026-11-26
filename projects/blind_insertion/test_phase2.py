"""Unit tests for Phase 2 Recurrent GRU blind insertion RL: ee_rel_start legality, GRU reset, BPTT gradients, curriculum, replay."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_insertion.env import BlindInsertionEnv, InsertionTaskSpec
from projects.blind_insertion.phase1_policy import LOG_STD_MAX, LOG_STD_MIN, M3_TORQUE_MAX, M3_TORQUE_MIN
from projects.blind_insertion.phase1_train import verify_train_disjointness
from projects.blind_insertion.phase2_policy import Phase2ProprioBuilder, RecurrentAsymmetricPPO
from projects.blind_insertion.phase2_train import (
    build_slippery_unbolted_ood_suite,
    compute_smooth_r_max_m,
    compute_step_reward_p2,
    sample_training_batch_p2,
)


class Phase2BlindInsertionTest(unittest.TestCase):
    def test_ee_rel_start_legality_m1_invariance_and_zero_privileged_keys(self):
        base_obs = {
            'q_quantized': np.array([0.05, -0.1, 0.3, 1.3, 0.0, 0.35]),
            'q_dot_quantized': np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.0]),
            'ee_pose_fk': np.array([0.22, 0.0, 0.026, 1.0, 0.0, 0.0, 0.0]),
            'ee_twist_fk': np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
            'prev_action': np.array([0.001, -0.001, -0.001, 1.0, 1.0, 1.0, 1.0, 1.0]),
            'tracking_error': np.zeros(6),
            'actuator_force_noisy': np.zeros(6),
            'torque_scale': np.ones(5),
            'time_s': 0.00,
        }
        obs_pert = dict(base_obs)
        obs_pert['actuator_force_noisy'] = np.array([2.9, -2.9, 2.5, -1.8, 1.2, -2.0])
        obs_pert['tracking_error'] = np.array([0.15, -0.20, 0.18, -0.12, 0.09, 0.05])
        obs_pert['torque_scale'] = np.array([0.20, 0.25, 0.35, 0.45, 0.55])

        # Verify ee_rel_start is exact step-0-anchored FK displacement and M1 is 100% invariant to force/err/ts
        b1 = Phase2ProprioBuilder('pos_only')
        b2 = Phase2ProprioBuilder('pos_only')
        x0 = b1.step(base_obs)
        x0_p = b2.step(obs_pert)
        np.testing.assert_allclose(x0[:3], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(x0, x0_p, atol=1e-12)

        moved_obs = dict(base_obs, ee_pose_fk=np.array([0.230, -0.006, 0.020, 1.0, 0.0, 0.0, 0.0]), time_s=0.02)
        moved_pert = dict(obs_pert, ee_pose_fk=moved_obs['ee_pose_fk'], time_s=0.02)
        x1 = b1.step(moved_obs)
        x1_p = b2.step(moved_pert)
        np.testing.assert_allclose(x1[:3], [0.010 / 0.02, -0.006 / 0.02, -0.006 / 0.015], atol=1e-12)
        np.testing.assert_allclose(x1, x1_p, atol=1e-12)

        # M2 must see force/tracking_error, and ignore torque_scale
        bm2_a = Phase2ProprioBuilder('current')
        bm2_b = Phase2ProprioBuilder('current')
        x_a = bm2_a.step(base_obs)
        obs_ts = dict(base_obs, torque_scale=np.array([0.25, 0.25, 0.25, 0.25, 0.25]))
        x_b = bm2_b.step(obs_ts)
        np.testing.assert_allclose(x_a, x_b, atol=1e-12)

        # Runtime privileged-key guard fires on all 3 modes
        bad_obs = dict(base_obs, socket_xy_error=np.zeros(2))
        for mode in ('pos_only', 'current', 'current_torque'):
            with self.assertRaises(AssertionError):
                Phase2ProprioBuilder(mode).step(bad_obs)

    def test_gru_hidden_state_resets_cleanly_at_episode_boundaries(self):
        rng = np.random.default_rng(77)
        b = Phase2ProprioBuilder('current_torque')
        pol = RecurrentAsymmetricPPO(obs_dim=b.obs_dim, use_torque_limit=True, seed=11)
        pol.W_mu[:] = rng.normal(0.0, 0.3, size=pol.W_mu.shape)

        # Episode 1 runs with nonzero inputs so GRU hidden state h becomes strongly non-zero
        h = pol.init_hidden()
        for _ in range(12):
            _, _, h, _, _ = pol.act_step(rng.normal(0.0, 1.0, size=b.obs_dim), h, deterministic=True)
        self.assertGreater(float(np.linalg.norm(h)), 0.5)

        # Without reset vs with reset on the same new Episode 2 input
        x_ep2 = rng.normal(0.0, 1.0, size=b.obs_dim)
        act_carried, _, _, _, _ = pol.act_step(x_ep2, h, deterministic=True)
        h_reset = pol.init_hidden()
        np.testing.assert_allclose(h_reset, 0.0, atol=1e-12)
        act_reset_a, _, h_a, _, _ = pol.act_step(x_ep2, h_reset, deterministic=True)
        act_reset_b, _, h_b, _, _ = pol.act_step(x_ep2, np.zeros_like(h), deterministic=True)

        # Carried hidden state genuinely changes action (proving recurrent memory is active)
        self.assertGreater(float(np.linalg.norm(act_carried - act_reset_a)), 1e-4)
        # Clean boundary reset gives bit-identical output independent of Episode 1 history
        np.testing.assert_allclose(act_reset_a, act_reset_b, atol=1e-12)
        np.testing.assert_allclose(h_a, h_b, atol=1e-12)

    def test_gru_bptt_actor_aux_and_critic_analytic_gradients_match_finite_difference(self):
        rng = np.random.default_rng(31)
        pol = RecurrentAsymmetricPPO(
            obs_dim=7, critic_dim=9, use_torque_limit=True,
            hidden_dim=5, critic_h1=8, critic_h2=6, seed=12,
        )
        pol.W_mu[:] = rng.normal(0.0, 0.15, size=pol.W_mu.shape)
        pol.b_mu[:] = rng.normal(0.0, 0.10, size=pol.b_mu.shape)

        B, L = 3, 6
        X = rng.normal(0.0, 1.0, size=(B, L, 7))
        h0 = rng.normal(0.0, 0.5, size=(B, 5))
        mask = np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ], dtype=float)
        Z = rng.normal(0.0, 0.5, size=(B, L, pol.act_dim))
        old_lp, _ = pol.seq_log_prob(X, h0, Z)
        old_lp = old_lp + rng.normal(0.0, 0.05, size=(B, L))
        adv = rng.normal(0.0, 1.0, size=(B, L))
        cls_lbl = rng.integers(0, 5, size=(B, L))
        dir_tgt = rng.normal(0.0, 1.0, size=(B, L, 2))
        dir_tgt /= np.maximum(np.linalg.norm(dir_tgt, axis=-1, keepdims=True), 1e-6)
        dir_mask = np.array([
            [1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ], dtype=float)

        _, a_grads, _, _, _, _, _ = pol.actor_and_aux_seq_loss_and_grad(
            X, h0, mask, Z, old_lp, adv, cls_lbl, dir_tgt, dir_mask,
        )
        eps = 1e-6
        for name, param in pol.actor_params().items():
            num = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                orig = param[idx]
                param[idx] = orig + eps
                lp, _, _, _, _, _, _ = pol.actor_and_aux_seq_loss_and_grad(
                    X, h0, mask, Z, old_lp, adv, cls_lbl, dir_tgt, dir_mask,
                )
                param[idx] = orig - eps
                lm, _, _, _, _, _, _ = pol.actor_and_aux_seq_loss_and_grad(
                    X, h0, mask, Z, old_lp, adv, cls_lbl, dir_tgt, dir_mask,
                )
                param[idx] = orig
                num[idx] = (lp - lm) / (2.0 * eps)
                it.iternext()
            np.testing.assert_allclose(a_grads[name], num, atol=1e-5, rtol=1e-4, err_msg=f'Mismatch in {name}')

        C = rng.normal(0.0, 1.0, size=(8, 9))
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
            np.testing.assert_allclose(c_grads[name], num, atol=1e-5, rtol=1e-4, err_msg=f'Mismatch in {name}')

    def test_off_plate_reward_exploit_blocked_and_m3_two_group_init(self):
        task = InsertionTaskSpec(
            trial_id='p2-walkoff', seed=150555, shape='square', clearance_m=0.0010,
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
            triggered = False
            for st in range(1, 60):
                a = np.array([0.004, 0.0, -0.002, 1.0, 1.0, 1.0, 1.0, 1.0])
                obs, _, term, trunc, info_e = env.step(a)
                r_st, off_t = compute_step_reward_p2(
                    prev_p, info_e['privileged'], a, prev_a, info_e, st, 250, milestones,
                    visited_plate_cells=visited, ee_rel_xy=obs['ee_pose_fk'][:2] - start_xy,
                )
                total_ret += r_st
                prev_p = info_e['privileged']
                prev_a = a
                if off_t:
                    triggered = True
                    break
                if term or trunc:
                    break
            self.assertTrue(triggered)
            self.assertLess(total_ret, -7.0)
        finally:
            env.close()

        b = Phase2ProprioBuilder('current_torque')
        pol = RecurrentAsymmetricPPO(obs_dim=b.obs_dim, use_torque_limit=True, seed=3)
        ea0, _, _, _, _ = pol.act_step(np.zeros(b.obs_dim), pol.init_hidden(), deterministic=True)
        np.testing.assert_allclose(ea0[:3], 0.0, atol=1e-12)
        np.testing.assert_allclose(ea0[3:8], 0.75, atol=1e-12)
        ea_g = pol.map_z_to_env_action(np.array([0.0, 0.0, 0.0, -1.0, 0.5]))
        np.testing.assert_allclose(ea_g[3:8], [0.35, 0.95, 0.95, 0.95, 0.35], atol=1e-12)

    def test_smooth_linear_curriculum_disjointness_and_slippery_ood_suite(self):
        self.assertAlmostEqual(compute_smooth_r_max_m(0.0), 0.0060, places=6)
        self.assertAlmostEqual(compute_smooth_r_max_m(0.175), 0.01125, places=6)
        self.assertAlmostEqual(compute_smooth_r_max_m(0.35), 0.0165, places=6)
        self.assertAlmostEqual(compute_smooth_r_max_m(1.0), 0.0165, places=6)

        rng = np.random.default_rng(19)
        batch, cur, r_max = sample_training_batch_p2(rng, batch_size=200, seed_cursor=150000, progress=0.50)
        self.assertEqual(cur, 150200)
        self.assertAlmostEqual(r_max, 0.0165, places=6)
        verify_train_disjointness(batch)
        radii = [float(np.linalg.norm(t.misalign_xy)) for t in batch]
        # Verify both <=10 mm replay and >14 mm full-range starts are represented
        self.assertTrue(any(r < 0.006 for r in radii))
        self.assertTrue(any(r > 0.0145 for r in radii))

        slip = build_slippery_unbolted_ood_suite()
        self.assertEqual(len(slip), 30)
        self.assertTrue(all((not t.bolted) and abs(t.mu_table - 0.25) < 1e-9 and abs(t.socket_mass_kg - 0.120) < 1e-9 for t in slip))

    def test_recurrent_checkpoint_deterministic_replay_and_frozen_rms(self):
        rng = np.random.default_rng(42)
        b = Phase2ProprioBuilder('current_torque')
        pol = RecurrentAsymmetricPPO(obs_dim=b.obs_dim, use_torque_limit=True, seed=5)
        pol.W_mu[:] = rng.normal(0.0, 0.25, size=pol.W_mu.shape)
        pol.obs_rms.update(rng.normal(0.1, 1.1, size=(120, b.obs_dim)))
        rms_before = pol.obs_rms.get_state('obs_rms')

        task = InsertionTaskSpec(
            trial_id='p2-replay', seed=150101, shape='square', clearance_m=0.0005, bolted=False,
            socket_xy=(0.220, -0.005), misalign_xy=(-0.010, 0.007),
        )
        with tempfile.TemporaryDirectory() as tmp:
            p_ckpt = Path(tmp) / 'p2_m3.npz'
            pol.save(p_ckpt)
            loaded = RecurrentAsymmetricPPO.load(p_ckpt)
            outs = []
            for p in (pol, loaded):
                env = BlindInsertionEnv(task=task, render=False)
                try:
                    obs, info = env.reset()
                    tb = Phase2ProprioBuilder('current_torque')
                    h = p.init_hidden()
                    trace = []
                    for _ in range(15):
                        x = tb.step(obs)
                        ea, _, h, c_p, _ = p.act_step(x, h, deterministic=True)
                        self.assertTrue(np.all(ea[3:8] >= M3_TORQUE_MIN - 1e-12) and np.all(ea[3:8] <= M3_TORQUE_MAX + 1e-12))
                        obs, _, _, _, info = env.step(ea)
                        trace.append((ea.tolist(), c_p, info['privileged']['peg_depth']))
                    outs.append(trace)
                finally:
                    env.close()
            self.assertEqual(outs[0], outs[1])
            np.testing.assert_array_equal(pol.obs_rms.mean, rms_before['obs_rms_mean'])
            self.assertTrue(np.all(loaded.log_std >= LOG_STD_MIN) and np.all(loaded.log_std <= LOG_STD_MAX))


if __name__ == '__main__':
    unittest.main()
