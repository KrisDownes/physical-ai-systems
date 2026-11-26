"""Unit tests for Phase 4 Clean-Kinematics Residual GRU + Spiral Brake Gate (`g_spiral`) + Dynamic `GravComp-TL`."""
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
from projects.blind_insertion.controllers import BlindSpiralController
from projects.blind_insertion.env import InsertionTaskSpec
from projects.blind_insertion.phase1_train import verify_train_disjointness
from projects.blind_insertion.phase3_gravcomp import GravCompBlindInsertionEnv
from projects.blind_insertion.phase3_train import GravCompBlindSpiralController
from projects.blind_insertion.phase4_policy import (
    BASE_SPIRAL_ALPHA,
    P4_ALPHA_MAX,
    P4_ALPHA_MIN,
    Phase4ResidualBuilder,
    Phase4ResidualPPO,
)
from projects.blind_insertion.phase4_train import (
    PausableBaseSpiralController,
    sample_training_batch_p4,
)


class Phase4ResidualGravCompTest(unittest.TestCase):
    def test_clean_sensing_zero_noisy_current_invariance_and_privileged_guard(self):
        base_obs = {
            'q_quantized': np.array([0.05, -0.08, 0.30, 1.35, 0.0, 0.35]),
            'q_dot_quantized': np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.0]),
            'ee_pose_fk': np.array([0.22, 0.0, 0.026, 1.0, 0.0, 0.0, 0.0]),
            'ee_twist_fk': np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
            'prev_action': np.array([0.001, -0.001, -0.001, 0.35, 0.35, 0.35, 0.35, 0.35]),
            'tracking_error': np.array([0.01, -0.02, 0.015, 0.0, 0.0, 0.0]),
            'actuator_force_noisy': np.zeros(6),
            'torque_scale': np.full(5, 0.35),
            'time_s': 0.02,
        }
        obs_pert = dict(
            base_obs,
            actuator_force_noisy=np.array([2.9, -2.9, 2.5, -1.8, 1.2, -2.0]),
            torque_scale=np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
        )
        base_dxyz = np.array([0.001, -0.001, -0.0012])

        # BOTH R1 and R2 must be 100% invariant to noisy actuator_force_noisy (zero d_tau_ext / zero noisy current!)
        for mode in ('res_pos_only', 'res_gravcomp_tl'):
            b1, b2 = Phase4ResidualBuilder(mode), Phase4ResidualBuilder(mode)
            x1 = b1.step(base_obs, base_dxyz, 'spiral', 0.005, 0)
            x2 = b2.step(obs_pert, base_dxyz, 'spiral', 0.005, 0)
            np.testing.assert_allclose(x1, x2, atol=1e-12)

        # Runtime privileged-key guard fires on both modes
        bad_obs = dict(base_obs, socket_xy_error=np.zeros(2))
        for mode in ('res_pos_only', 'res_gravcomp_tl'):
            with self.assertRaises(AssertionError):
                Phase4ResidualBuilder(mode).step(bad_obs, base_dxyz, 'spiral', 0.005, 0)

    def test_step0_untrained_policy_reproduces_frozen_base_bit_for_bit(self):
        # Verify BOTH R1 (matches BlindSpiral_Stiff) and R2 (matches BlindSpiral_GravComp alpha=0.35)
        # across Bolted A, Unbolted B, and Tight 0.5mm C trials at step-0 initialization!
        sample_tasks = [
            build_regime_a_bolted()[0],
            build_regime_b_unbolted()[0],
            build_split_c_tight_0_5mm()[0],
        ]
        for mode in ('res_pos_only', 'res_gravcomp_tl'):
            for task in sample_tasks:
                env_base = GravCompBlindInsertionEnv(task=task, render=False)
                env_res = GravCompBlindInsertionEnv(task=task, render=False)
                try:
                    obs_b, _ = env_base.reset()
                    ctrl_b = (
                        BlindSpiralController(torque_scale=1.0)
                        if mode == 'res_pos_only'
                        else GravCompBlindSpiralController(alpha=BASE_SPIRAL_ALPHA)
                    )
                    ctrl_b.reset(obs_b)

                    obs_r, _ = env_res.reset()
                    ctrl_r = PausableBaseSpiralController(torque_scale=1.0)
                    ctrl_r.reset(obs_r)
                    builder = Phase4ResidualBuilder(mode)
                    pol = Phase4ResidualPPO(obs_dim=builder.obs_dim, mode=mode, seed=7)
                    h = pol.init_hidden()

                    for _ in range(35):
                        ab = ctrl_b.act(obs_b, None)
                        obs_b, _, tb, trb, info_b = env_base.step(ab)

                        bdxyz = ctrl_r.act_base(obs_r)
                        x = builder.step(obs_r, bdxyz, ctrl_r.phase, 0.0, ctrl_r.insert_stall_steps)
                        ar, _, g_sp, ap, _, h, _, _ = pol.act_step_p4(x, h, bdxyz, deterministic=True)
                        self.assertAlmostEqual(g_sp, 1.0, places=12)
                        if mode == 'res_gravcomp_tl':
                            self.assertEqual(ap, (BASE_SPIRAL_ALPHA, BASE_SPIRAL_ALPHA))
                        ctrl_r.set_spiral_gate(g_sp)
                        builder.set_prev_controls(g_sp, ap)
                        obs_r, _, tr, trr, info_r = env_res.step(ar)

                        np.testing.assert_allclose(obs_b['q_quantized'], obs_r['q_quantized'], atol=1e-12)
                        self.assertEqual(info_b['privileged']['peg_depth'], info_r['privileged']['peg_depth'])
                        if tb or trb or tr or trr:
                            break
                finally:
                    env_base.close()
                    env_res.close()

    def test_phase4_disjointness_gate_braking_and_checkpoint_replay(self):
        rng = np.random.default_rng(123)
        batch, cur = sample_training_batch_p4(rng, batch_size=200, seed_cursor=360000)
        self.assertEqual(cur, 360200)
        verify_train_disjointness(batch)
        self.assertTrue(all(0.35 <= t.mu_table <= 0.55 and 0.180 <= t.socket_mass_kg <= 0.320 for t in batch))

        # Verify g_spiral braking pauses spiral cleanly and alpha stays in [0.08, 1.0]
        b = Phase4ResidualBuilder('res_gravcomp_tl')
        pol = Phase4ResidualPPO(obs_dim=b.obs_dim, mode='res_gravcomp_tl', seed=19)
        _, d_braked, g_braked, ap_low = pol.compose_action(
            np.array([0.003, -0.002, -0.001]),
            np.array([0.0, 0.0, 0.0, -2.0, -1.0, -1.0]),
        )
        self.assertEqual(g_braked, 0.0)
        np.testing.assert_allclose(d_braked, [0.0, 0.0, 0.0], atol=1e-12)
        self.assertEqual(ap_low, (P4_ALPHA_MIN, P4_ALPHA_MIN))

        # Deterministic save/load replay with non-zero residual weights
        pol.W_mu[:] = rng.normal(0.0, 0.25, size=pol.W_mu.shape)
        pol.obs_rms.update(rng.normal(0.0, 1.0, size=(60, b.obs_dim)))
        task = InsertionTaskSpec(
            trial_id='p4-rep', seed=360111, shape='hex', clearance_m=0.0005, bolted=False,
            socket_xy=(0.220, 0.004), misalign_xy=(0.011, -0.008), mu_table=0.25, socket_mass_kg=0.120,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p_ckpt = Path(tmp) / 'r2.npz'
            pol.save(p_ckpt)
            loaded = Phase4ResidualPPO.load(p_ckpt)
            outs = []
            for p in (pol, loaded):
                env = GravCompBlindInsertionEnv(task=task, render=False)
                try:
                    obs, info = env.reset()
                    bc = PausableBaseSpiralController(torque_scale=1.0)
                    bc.reset(obs)
                    tb = Phase4ResidualBuilder('res_gravcomp_tl')
                    h = p.init_hidden()
                    trace = []
                    for _ in range(15):
                        bdxyz = bc.act_base(obs)
                        x = tb.step(obs, bdxyz, bc.phase, 0.0, bc.insert_stall_steps)
                        ea, d_xyz, g_sp, ap, _, h, c_p, _ = p.act_step_p4(x, h, bdxyz, deterministic=True)
                        bc.set_spiral_gate(g_sp)
                        tb.set_prev_controls(g_sp, ap)
                        self.assertTrue(0.0 <= g_sp <= 1.0)
                        self.assertTrue(P4_ALPHA_MIN <= ap[0] <= P4_ALPHA_MAX)
                        self.assertTrue(P4_ALPHA_MIN <= ap[1] <= P4_ALPHA_MAX)
                        obs, _, _, _, info = env.step(ea)
                        trace.append((d_xyz.tolist(), g_sp, ap, c_p, info['privileged']['peg_depth']))
                    outs.append(trace)
                finally:
                    env.close()
            self.assertEqual(outs[0], outs[1])


if __name__ == '__main__':
    unittest.main()
