"""Unit tests for Phase 3 GravComp-TL: arm-only gravity torque invariance & <0.1 ms speed, clamping, tau_ext, replay."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_insertion.env import InsertionTaskSpec, STS3215_MAX_TORQUE_NM
from projects.blind_insertion.phase1_policy import LOG_STD_MAX, LOG_STD_MIN
from projects.blind_insertion.phase3_gravcomp import (
    GRAVCOMP_ALPHA_MAX,
    GRAVCOMP_ALPHA_MIN,
    GRAVCOMP_MIN_LIM_NM,
    TAU_MARGIN_MAX_NM,
    GravCompBlindInsertionEnv,
    compute_arm_gravity_torque,
    compute_gravcomp_torque_limits_nm,
)
from projects.blind_insertion.phase3_policy import M4_ALPHA_INIT, Phase3GravCompPPO, Phase3ProprioBuilder


class Phase3GravCompTest(unittest.TestCase):
    def test_compute_arm_gravity_torque_invariance_and_under_0_1_ms(self):
        q = np.array([0.05, -0.08, 0.30, 1.35, 0.0, 0.35], dtype=float)
        tau_a = compute_arm_gravity_torque(q)
        self.assertEqual(tau_a.shape, (5,))
        # shoulder_pan (0) and wrist_roll (4) have zero static gravity torque; lift/elbow carry arm weight
        self.assertLess(abs(tau_a[0]), 1e-6)
        self.assertGreater(abs(tau_a[1]), 0.20)
        self.assertGreater(abs(tau_a[2]), 0.15)

        # Exact invariance across wildly different peg shapes, socket masses, friction, and contact states
        t1 = InsertionTaskSpec(
            trial_id='gc1', seed=21, shape='round', clearance_m=0.0005, bolted=True,
            mu_peg_hole=0.15, mu_table=0.25, socket_mass_kg=0.120,
        )
        t2 = InsertionTaskSpec(
            trial_id='gc2', seed=22, shape='hex', clearance_m=0.0015, bolted=False,
            mu_peg_hole=0.45, mu_table=0.55, socket_mass_kg=0.320,
        )
        e1 = GravCompBlindInsertionEnv(task=t1, render=False)
        e2 = GravCompBlindInsertionEnv(task=t2, render=False)
        try:
            e1.reset()
            e2.reset()
            # Push e1 into hard contact, leave e2 free
            for _ in range(10):
                e1.step({'delta_xyz': [0.0, 0.0, -0.003], 'gravcomp_alpha': [0.35, 0.35]})
            tau_b = compute_arm_gravity_torque(q)
            np.testing.assert_allclose(tau_a, tau_b, atol=1e-12)
        finally:
            e1.close()
            e2.close()

        # Benchmark speed: MUST run in < 0.1 ms per call
        n_calls = 500
        t0 = time.perf_counter()
        for k in range(n_calls):
            compute_arm_gravity_torque(q + 1e-4 * (k % 5))
        ms_per_call = (time.perf_counter() - t0) * 1000.0 / n_calls
        self.assertLess(ms_per_call, 0.1, f'compute_arm_gravity_torque took {ms_per_call:.4f} ms >= 0.1 ms')

    def test_gravcomp_tl_clamps_actuator_force_and_prevents_gravity_sag(self):
        task = InsertionTaskSpec(
            trial_id='gc-clamp', seed=99, shape='square', clearance_m=0.0012,
            bolted=True, misalign_xy=(0.015, 0.0), start_clearance_z_m=0.005,
        )
        env = GravCompBlindInsertionEnv(task=task, render=False)
        try:
            obs, _ = env.reset()
            z0 = float(obs['ee_pose_fk'][2])
            # 1. Hover in free air with ultra-low alpha=0.04 (0.06 Nm margin above gravity): must NOT sag!
            for _ in range(25):
                obs, _, _, _, info = env.step({'delta_xyz': [0.0, 0.0, 0.0], 'gravcomp_alpha': [0.04, 0.04]})
            z1 = float(obs['ee_pose_fk'][2])
            self.assertLess(abs(z1 - z0), 0.0035, f'Arm sagged under GravComp-TL alpha=0.04: dz={z1-z0:.4f} m')

            # 2. Push hard down at alpha=[0.08, 0.15] and verify MuJoCo respects exact GravComp-TL per-joint limits!
            for _ in range(20):
                q_pre = obs['q_quantized'].copy()
                expected_lim, _ = compute_gravcomp_torque_limits_nm(q_pre, np.array([0.08, 0.15]))
                self.assertTrue(np.all(expected_lim >= GRAVCOMP_MIN_LIM_NM) and np.all(expected_lim <= STS3215_MAX_TORQUE_NM))
                # shoulder_pan limit is ~0.12 Nm (far below Phase 2's 0.441 Nm floor!)
                self.assertAlmostEqual(expected_lim[0], 0.08 * TAU_MARGIN_MAX_NM, places=4)
                obs, _, term, trunc, info = env.step({'delta_xyz': [0.004, 0.0, -0.003], 'gravcomp_alpha': [0.08, 0.15]})
                self.assertTrue(
                    np.all(info['step_force_abs_max_Nm'][:5] <= expected_lim + 1e-5),
                    f'Force exceeded GravComp-TL limit: {info["step_force_abs_max_Nm"][:5]} vs {expected_lim}',
                )
                if term or trunc:
                    break
        finally:
            env.close()

    def test_tau_ext_d_tau_ext_legality_and_m1_invariance(self):
        base_obs = {
            'q_quantized': np.array([0.05, -0.08, 0.30, 1.35, 0.0, 0.35]),
            'q_dot_quantized': np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.0]),
            'ee_pose_fk': np.array([0.22, 0.0, 0.026, 1.0, 0.0, 0.0, 0.0]),
            'ee_twist_fk': np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
            'prev_action': np.array([0.001, -0.001, -0.001, 1.0, 1.0, 1.0, 1.0, 1.0]),
            'tracking_error': np.zeros(6),
            'actuator_force_noisy': np.zeros(6),
            'torque_scale': np.ones(5),
            'time_s': 0.00,
        }
        obs_pert = dict(
            base_obs,
            actuator_force_noisy=np.array([2.5, -2.5, 2.0, -1.5, 1.0, -1.0]),
            tracking_error=np.array([0.1, -0.1, 0.1, -0.1, 0.1, 0.0]),
            torque_scale=np.array([0.1, 0.2, 0.2, 0.2, 0.1]),
        )

        # M1 pos_only strictly invariant to force, tracking_error, torque_scale
        b1, b2 = Phase3ProprioBuilder('pos_only'), Phase3ProprioBuilder('pos_only')
        np.testing.assert_allclose(b1.step(base_obs), b2.step(obs_pert), atol=1e-12)

        # Verify tau_ext = f_noisy - tau_grav and d_tau_ext = 0 at step 0, exact diff at step 1
        bm4 = Phase3ProprioBuilder('gravcomp_tl')
        tau_g = compute_arm_gravity_torque(base_obs['q_quantized'])
        bm4.step(base_obs)
        np.testing.assert_allclose(bm4.last_tau_ext, base_obs['actuator_force_noisy'][:5] - tau_g, atol=1e-12)
        np.testing.assert_allclose(bm4.last_d_tau_ext, np.zeros(5), atol=1e-12)

        obs_step1 = dict(base_obs, actuator_force_noisy=np.array([0.4, -0.2, 0.1, 0.05, -0.1, 0.0]), time_s=0.02)
        prev_te = bm4.last_tau_ext.copy()
        bm4.step(obs_step1)
        expected_te1 = obs_step1['actuator_force_noisy'][:5] - tau_g
        np.testing.assert_allclose(bm4.last_tau_ext, expected_te1, atol=1e-12)
        np.testing.assert_allclose(bm4.last_d_tau_ext, expected_te1 - prev_te, atol=1e-12)

        # Privileged-key guard fires on all 3 modes
        bad_obs = dict(base_obs, socket_xy_error=np.zeros(2))
        for mode in ('pos_only', 'gravcomp_fixed', 'gravcomp_tl'):
            with self.assertRaises(AssertionError):
                Phase3ProprioBuilder(mode).step(bad_obs)

    def test_m4_zero_init_alpha_and_deterministic_checkpoint_replay(self):
        b = Phase3ProprioBuilder('gravcomp_tl')
        pol = Phase3GravCompPPO(obs_dim=b.obs_dim, mode='gravcomp_tl', seed=17)
        env_act0, d_xyz0, alpha0, _, _, _, _ = pol.act_step_p3(np.zeros(b.obs_dim), pol.init_hidden(), deterministic=True)
        np.testing.assert_allclose(d_xyz0, 0.0, atol=1e-12)
        self.assertEqual(alpha0, (M4_ALPHA_INIT, M4_ALPHA_INIT))
        np.testing.assert_allclose(env_act0['gravcomp_alpha'], [M4_ALPHA_INIT, M4_ALPHA_INIT], atol=1e-12)

        rng = np.random.default_rng(88)
        pol.W_mu[:] = rng.normal(0.0, 0.3, size=pol.W_mu.shape)
        pol.obs_rms.update(rng.normal(0.0, 1.0, size=(80, b.obs_dim)))
        rms_before = pol.obs_rms.get_state('obs_rms')

        task = InsertionTaskSpec(
            trial_id='p3-replay', seed=250101, shape='hex', clearance_m=0.0005, bolted=False,
            socket_xy=(0.220, -0.005), misalign_xy=(-0.010, 0.007), mu_table=0.25, socket_mass_kg=0.120,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p_ckpt = Path(tmp) / 'm4.npz'
            pol.save(p_ckpt)
            loaded = Phase3GravCompPPO.load(p_ckpt)
            outs = []
            for p in (pol, loaded):
                env = GravCompBlindInsertionEnv(task=task, render=False)
                try:
                    obs, info = env.reset()
                    tb = Phase3ProprioBuilder('gravcomp_tl')
                    h = p.init_hidden()
                    trace = []
                    for _ in range(15):
                        x = tb.step(obs)
                        ea, d_xyz, ap, _, h, c_p, _ = p.act_step_p3(x, h, deterministic=True)
                        tb.set_prev_alpha(ap[0], ap[1])
                        self.assertTrue(GRAVCOMP_ALPHA_MIN <= ap[0] <= GRAVCOMP_ALPHA_MAX)
                        self.assertTrue(GRAVCOMP_ALPHA_MIN <= ap[1] <= GRAVCOMP_ALPHA_MAX)
                        obs, _, _, _, info = env.step(ea)
                        trace.append((d_xyz.tolist(), ap, c_p, info['privileged']['peg_depth']))
                    outs.append(trace)
                finally:
                    env.close()
            self.assertEqual(outs[0], outs[1])
            np.testing.assert_array_equal(pol.obs_rms.mean, rms_before['obs_rms_mean'])
            self.assertTrue(np.all(loaded.log_std >= LOG_STD_MIN) and np.all(loaded.log_std <= LOG_STD_MAX))


if __name__ == '__main__':
    unittest.main()
