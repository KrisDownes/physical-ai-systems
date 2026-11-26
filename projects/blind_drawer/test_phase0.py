"""Unit tests for Phase 0 blind drawer opening: strict obs legality, GravComp-TL, backlash, replay, Oracle."""
from __future__ import annotations

from pathlib import Path
import sys
import time
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
from projects.blind_drawer.controllers import (
    NominalStraightBackController,
    PrivilegedOracleController,
)
from projects.blind_drawer.env import (
    ALLOWED_OBS_KEYS,
    FORBIDDEN_PRIVILEGED_KEYS,
    BlindDrawerEnv,
    DrawerTaskSpec,
    STS3215Config,
)
from projects.blind_drawer.gravcomp import (
    STS3215_MAX_TORQUE_NM,
    compute_arm_gravity_torque,
)


class Phase0BlindDrawerTest(unittest.TestCase):
    def test_zero_absolute_ee_pos_zero_privileged_keys_and_disjoint_splits(self):
        env = BlindDrawerEnv(task=DrawerTaskSpec(seed=1, bolted=False, rail_yaw_deg=15.0), render=False)
        try:
            obs, info = env.reset()
            self.assertEqual(set(obs.keys()), ALLOWED_OBS_KEYS)
            self.assertEqual(set(obs.keys()) & FORBIDDEN_PRIVILEGED_KEYS, set())
            self.assertNotIn('ee_pose_fk', obs)
            self.assertNotIn('ee_pos_fk', obs)
            self.assertEqual(obs['ee_rel_start'].shape, (3,))
            np.testing.assert_allclose(obs['ee_rel_start'], 0.0, atol=1e-12)
            for k in ('drawer_travel_mm', 'rail_yaw_rad', 'cabinet_disp_mm', 'cabinet_tilt_deg', 'peak_contact_force_N', 'contact_class'):
                self.assertIn(k, info['privileged'])
        finally:
            env.close()

        expected_n = [
            (build_split_a_bolted_sticky, 30),
            (build_split_b_unbolted_cabinet, 30),
            (build_split_ood_light_height_angle, 20),
        ]
        all_seeds = set()
        for fn, n_exp in expected_n:
            trials = fn()
            self.assertEqual(len(trials), n_exp)
            for t in trials:
                self.assertLess(t.seed, TRAIN_SEED_OFFSET)
                self.assertNotIn(t.seed, all_seeds)
                all_seeds.add(t.seed)

    def test_lean_model_size_and_gravcomp_invariance_under_0_1_ms(self):
        env = BlindDrawerEnv(task=DrawerTaskSpec(seed=2), render=False)
        try:
            self.assertLess(env.model.nbuffer / 1e6, 4.0)
        finally:
            env.close()

        q = np.array([0.12, -0.25, 0.35, 1.15, -0.10, -0.10])
        tau0 = compute_arm_gravity_torque(q)
        self.assertEqual(tau0.shape, (5,))
        self.assertTrue(np.all(np.isfinite(tau0)))

        t0 = time.perf_counter()
        for _ in range(200):
            tau = compute_arm_gravity_torque(q)
            np.testing.assert_array_equal(tau, tau0)
        ms_per_call = (time.perf_counter() - t0) * 1000.0 / 200.0
        self.assertLess(ms_per_call, 0.1)

    def test_backlash_deadband_and_deterministic_replay(self):
        task = DrawerTaskSpec(seed=7, bolted=False, rail_yaw_deg=14.0, detent_force_N=4.5)
        e1 = BlindDrawerEnv(task=task, servo_cfg=STS3215Config(backlash_rad=0.0045), render=False)
        e2 = BlindDrawerEnv(task=task, servo_cfg=STS3215Config(backlash_rad=0.0045), render=False)
        try:
            o1, _ = e1.reset()
            o2, _ = e2.reset()
            c1 = NominalStraightBackController(mode='gravcomp', gravcomp_alpha=0.35)
            c2 = NominalStraightBackController(mode='gravcomp', gravcomp_alpha=0.35)
            c1.reset(o1)
            c2.reset(o2)
            for _ in range(45):
                o1, r1, t1, tr1, i1 = e1.step(c1.act(o1))
                o2, r2, t2, tr2, i2 = e2.step(c2.act(o2))
                np.testing.assert_array_equal(o1['q_quantized'], o2['q_quantized'])
                np.testing.assert_array_equal(o1['ee_rel_start'], o2['ee_rel_start'])
                np.testing.assert_array_equal(o1['tracking_error'], o2['tracking_error'])
                self.assertEqual((r1, t1, tr1), (r2, t2, tr2))
                if t1 or tr1:
                    break
        finally:
            e1.close()
            e2.close()

    def test_gravcomp_tl_clamps_actuator_force(self):
        env = BlindDrawerEnv(task=DrawerTaskSpec(seed=11, bolted=True, rail_yaw_deg=16.0), render=False)
        try:
            obs, _ = env.reset()
            act = {'delta_xyz': np.array([-0.004, 0.0, 0.0]), 'gravcomp_alpha': [0.10, 0.35]}
            obs, _, _, _, _ = env.step(act)
            lim = env.model.actuator_forcerange[:5, 1]
            self.assertTrue(np.all(lim <= STS3215_MAX_TORQUE_NM + 1e-9))
            self.assertLess(float(lim[0]), 0.35)  # shoulder_pan tau_grav=0 -> lim ~ 0.25 Nm
            self.assertTrue(np.all(np.abs(env.data.actuator_force[:5]) <= lim + 1e-3))
        finally:
            env.close()

    def test_oracle_succeeds_across_bolted_unbolted_and_ood_height_angle(self):
        cases = (
            build_split_a_bolted_sticky()[0],
            build_split_b_unbolted_cabinet()[0],
            build_split_ood_light_height_angle()[0],  # +15 mm
            build_split_ood_light_height_angle()[1],  # -15 mm
        )
        for task in cases:
            env = BlindDrawerEnv(task=task, render=False)
            try:
                obs, info = env.reset()
                ctrl = PrivilegedOracleController()
                ctrl.reset(obs)
                done = False
                while not done:
                    obs, _, term, trunc, info = env.step(ctrl.act(obs, info))
                    done = term or trunc
                self.assertTrue(
                    info['success'],
                    f'Oracle failed on {task.trial_id} (yaw={task.rail_yaw_deg:.1f}, dz={task.height_shift_m}): '
                    f'{info["failure_mode"]}, travel={info["privileged"]["drawer_travel_mm"]:.1f}mm, '
                    f'disp={info["episode_max_cab_disp_mm"]:.1f}mm, tilt={info["episode_max_cab_tilt_deg"]:.1f}deg',
                )
                self.assertEqual(info['self_collision_steps'], 0)
                self.assertEqual(info['joint_limit_steps'], 0)
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
