"""Unit tests for Phase 0 blind proprioceptive peg-in-hole insertion benchmark."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_insertion.controllers import (
    BlindSpiralController,
    NominalStraightDownController,
    PrivilegedOracleController,
    run_episode,
)
from projects.blind_insertion.env import (
    ALLOWED_OBS_KEYS,
    ENCODER_QUANT_RAD,
    FORBIDDEN_PRIVILEGED_KEYS,
    STS3215_MAX_TORQUE_NM,
    BlindInsertionEnv,
    InsertionTaskSpec,
    STS3215Config,
)
from projects.blind_insertion.geometry import ALL_SHAPES, run_all_drop_tests


class Phase0BlindInsertionTest(unittest.TestCase):
    def test_exact_clearance_drop_tests_pass_100_percent(self):
        results = run_all_drop_tests(clearances_m=(0.0010, 0.0005))
        self.assertEqual(len(results), 2 * len(ALL_SHAPES))
        for r in results:
            self.assertTrue(r['passed'], f'Drop test failed: {r}')
            self.assertGreaterEqual(r['depth_mm'], 19.5)

    def test_zero_privileged_keys_in_obs_and_privileged_schema(self):
        task = InsertionTaskSpec(
            trial_id='test-obs', seed=42, shape='hex', clearance_m=0.0012,
            bolted=False, misalign_xy=(0.010, -0.006),
        )
        env = BlindInsertionEnv(task=task, render=False)
        try:
            obs, info = env.reset()
            self.assertEqual(set(obs.keys()), ALLOWED_OBS_KEYS)
            self.assertTrue(set(obs.keys()).isdisjoint(FORBIDDEN_PRIVILEGED_KEYS))

            priv = info['privileged']
            for req in ('socket_xy_error', 'peg_depth', 'contact_class', 'peak_contact_force_N', 'socket_displacement_mm'):
                self.assertIn(req, priv)
            self.assertIn(priv['contact_class'], ('free', 'plate_contact', 'rim_contact', 'entering', 'seated'))

            obs2, _, _, _, info2 = env.step(np.array([0.0, 0.0, -0.0015, 0.5]))
            self.assertEqual(set(obs2.keys()), ALLOWED_OBS_KEYS)
            self.assertTrue(set(obs2.keys()).isdisjoint(FORBIDDEN_PRIVILEGED_KEYS))
            self.assertIn('privileged', info2)

            # Verify 12-bit encoder quantization exact grid on q_quantized
            ratio = obs2['q_quantized'] / ENCODER_QUANT_RAD
            np.testing.assert_allclose(ratio, np.round(ratio), atol=1e-9)
        finally:
            env.close()

    def test_deterministic_seeding_identical_replay(self):
        task = InsertionTaskSpec(
            trial_id='test-det', seed=123, shape='d_shape', clearance_m=0.0011,
            bolted=False, misalign_xy=(-0.008, 0.009), mu_peg_hole=0.33,
            mu_table=0.42, socket_mass_kg=0.22, backlash_rad=0.005,
        )
        traces = []
        for _ in range(2):
            env = BlindInsertionEnv(task=task, render=False)
            try:
                obs, info = env.reset()
                ctrl = BlindSpiralController(torque_scale=0.65)
                ctrl.reset(obs)
                ep_log = []
                for _ in range(25):
                    act = ctrl.act(obs, info)
                    obs, rew, term, trunc, info = env.step(act)
                    ep_log.append((
                        obs['q_quantized'].tolist(),
                        obs['actuator_force_noisy'].tolist(),
                        info['privileged']['socket_xy_error'].tolist(),
                        info['privileged']['peg_depth'],
                        rew,
                    ))
                    if term or trunc:
                        break
                traces.append(ep_log)
            finally:
                env.close()
        self.assertEqual(traces[0], traces[1])

    def test_torque_limit_scaling_clamps_actuator_force(self):
        # Push hard downward into the bolted plate under two distinct torque scales and verify
        # MuJoCo physically clamps data.actuator_force inside [-2.94*ts, +2.94*ts].
        task = InsertionTaskSpec(
            trial_id='test-torque', seed=77, shape='square', clearance_m=0.0012,
            bolted=True, misalign_xy=(0.015, 0.0), start_clearance_z_m=0.001,
        )
        env = BlindInsertionEnv(task=task, render=False)
        try:
            for ts_vec in (
                np.array([0.25, 0.25, 0.25, 0.25, 0.25]),
                np.array([0.40, 0.20, 0.30, 0.50, 0.35]),
            ):
                env.reset()
                max_seen = np.zeros(5, dtype=float)
                for _ in range(30):
                    act = np.concatenate([[0.0, 0.0, -0.003], ts_vec])
                    _, _, term, trunc, info = env.step(act)
                    np.maximum(max_seen, info['step_force_abs_max_Nm'][:5], out=max_seen)
                    if term or trunc:
                        break
                expected_lim = STS3215_MAX_TORQUE_NM * ts_vec
                # Every joint's peak actuator force must respect the dynamic torque limit!
                self.assertTrue(
                    np.all(max_seen <= expected_lim + 1e-6),
                    f'Actuator force exceeded scaled limit: max_seen={max_seen}, lim={expected_lim}',
                )
                # Confirm we actually pushed into saturation on at least one load-bearing joint
                self.assertTrue(
                    np.any(max_seen >= 0.90 * expected_lim),
                    f'Expected saturation on load-bearing joints to test clamping: max_seen={max_seen}, lim={expected_lim}',
                )
        finally:
            env.close()

    def test_backlash_deadband_holds_on_micro_reversal(self):
        task = InsertionTaskSpec(trial_id='test-bl', seed=9, shape='round', bolted=True, backlash_rad=0.006)
        env = BlindInsertionEnv(task=task, servo_cfg=STS3215Config(backlash_rad=0.006), render=False)
        try:
            env.reset()
            q0 = env._q_backlash.copy()
            # Move +0.010 rad -> engages upper side of lash (y = q0 + 0.010 - 0.003 = q0 + 0.007)
            y1 = env._apply_backlash(q0 + 0.010)
            np.testing.assert_allclose(y1, q0 + 0.007, atol=1e-12)
            # Reverse by -0.004 rad (inside 0.006 rad deadband!) -> output must NOT move!
            y2 = env._apply_backlash(q0 + 0.006)
            np.testing.assert_allclose(y2, y1, atol=1e-12)
            # Reverse further to q0 + 0.002 -> crosses lower lash edge -> moves to q0 + 0.005
            y3 = env._apply_backlash(q0 + 0.002)
            np.testing.assert_allclose(y3, q0 + 0.005, atol=1e-12)
        finally:
            env.close()

    def test_nominal_fails_and_oracle_succeeds_bolted_and_unbolted(self):
        for bolted in (True, False):
            task = InsertionTaskSpec(
                trial_id=f'test-sanity-{bolted}', seed=55, shape='triangle',
                clearance_m=0.0012, bolted=bolted, misalign_xy=(0.011, -0.005),
                mu_peg_hole=0.30, mu_table=0.45, socket_mass_kg=0.24,
            )
            env = BlindInsertionEnv(task=task, render=False)
            try:
                r_nom = run_episode(env, task, NominalStraightDownController())
                self.assertFalse(r_nom['success'])
                r_orc = run_episode(env, task, PrivilegedOracleController())
                self.assertTrue(r_orc['success'], f'Oracle failed (bolted={bolted}): {r_orc}')
                self.assertEqual(r_orc['self_collision_steps'], 0)
                self.assertEqual(r_orc['joint_limit_steps'], 0)
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
