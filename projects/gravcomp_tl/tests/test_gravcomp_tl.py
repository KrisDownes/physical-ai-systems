# SPDX-License-Identifier: MIT
"""Unit & integration tests for the standalone `gravcomp-tl` package."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np

PKG_PARENT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(PKG_PARENT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gravcomp_tl import (
    DryRunSTS3215Bus,
    GravCompPolicyRunner,
    GravCompTL,
    STS3215GravCompController,
    STS3215Spec,
    WatchdogConfig,
    compute_auroc,
    compute_tactile_snr,
    expand_alpha,
    nm_to_sts3215_ticks,
    rad_to_sts3215_ticks,
    sts3215_ticks_to_nm,
    sts3215_ticks_to_rad,
)
from gravcomp_tl.calibrate_so101 import run_calibration
from projects.blind_drawer.gravcomp import compute_arm_gravity_torque as drawer_gravcomp_ref
from projects.blind_drawer.phase1_policy import DrawerResidualGRUPPO


class GravCompTLTests(unittest.TestCase):
    def test_gravcomp_accuracy_and_latency_under_0_05_ms(self):
        gc = GravCompTL()
        q = np.array([0.12, -0.25, 0.48, 1.15, -0.08], dtype=float)
        tau = gc.compute_gravity_torque(q)
        tau_ref = drawer_gravcomp_ref(q)
        np.testing.assert_allclose(tau, tau_ref, atol=1e-12)

        t0 = time.perf_counter()
        for _ in range(1000):
            gc.compute_limits(q, (0.20, 0.45, 0.55, 0.45, 0.30))
        per_call_ms = 1000.0 * (time.perf_counter() - t0) / 1000.0
        self.assertLess(per_call_ms, 0.05)

    def test_sts3215_0x30_tick_conversion_and_alpha_expansion(self):
        np.testing.assert_allclose(expand_alpha(0.35), np.full(5, 0.35))
        np.testing.assert_allclose(expand_alpha((0.20, 0.60)), [0.20, 0.60, 0.60, 0.60, 0.20])
        np.testing.assert_allclose(expand_alpha([0.16, 0.43, 0.58, 0.45, 0.30]), [0.16, 0.43, 0.58, 0.45, 0.30])

        gc = GravCompTL()
        q = np.array([0.0, -0.2, 0.4, 1.2, 0.0], dtype=float)
        lim_nm, reg_0x30, _ = gc.compute_limits(q, (0.16, 0.43, 0.58, 0.45, 0.30))
        self.assertTrue(np.all((reg_0x30 >= 0) & (reg_0x30 <= 1000)))
        self.assertLess(reg_0x30[0], reg_0x30[2])  # shoulder_pan softer than elbow_flex
        np.testing.assert_allclose(sts3215_ticks_to_nm(reg_0x30), lim_nm, atol=0.005)
        np.testing.assert_allclose(sts3215_ticks_to_rad(rad_to_sts3215_ticks(q)), q, atol=STS3215Spec().rad_per_tick)

    def test_safety_watchdog_clamping_slew_overload_nan_and_shutdown(self):
        gc = GravCompTL()
        bus = DryRunSTS3215Bus(external_load_ticks=np.array([5.0, 0.0, 0.0, 0.0, 0.0]), gravity_model=gc)
        wd = WatchdogConfig(joint_limit_margin_rad=0.02, max_torque_slew_nm_s=10.0, overload_err_rad=0.10, overload_soft_alpha=0.12)

        with STS3215GravCompController(bus=bus, gravcomp=gc, dt=0.02, watchdog=wd) as ctrl:
            # 1. Joint-limit clamping
            q_safe, lim1, reg1 = ctrl.command(np.full(5, 99.0), alpha=1.0)
            self.assertTrue(np.all(q_safe <= ctrl.j_hi + 1e-12))
            self.assertTrue(np.all(q_safe >= ctrl.j_lo - 1e-12))

            # 2. Max torque-slew rate limiter: cannot jump from alpha=0.35 init to alpha=1.0 in 1 step
            lim_full, _, _ = gc.compute_limits(np.zeros(5), 1.0)
            self.assertTrue(np.any(lim1 < lim_full - 0.1))

            # 3. Overload auto-soften when tracking_error exceeds overload_err_rad
            bus.present_ticks[0] = np.clip(bus.goal_ticks[0] - 150, 0, 4095)  # ~0.23 rad error > 0.10
            st = ctrl.read_state()
            self.assertTrue(st.overload_triggered)
            ctrl.command(q_safe, alpha=0.80)
            self.assertGreaterEqual(ctrl.overload_count, 1)

            # 4. NaN command triggers emergency_soft_hold
            _, lim_nan, _ = ctrl.command(np.array([np.nan, 0.0, 0.0, 0.0, 0.0]), alpha=0.5)
            self.assertTrue(np.all(np.isfinite(lim_nan)))

        # 5. Context manager exit cleanly releases torque (0x30 = 0)
        np.testing.assert_array_equal(bus.torque_limit_reg, np.zeros(5, dtype=np.int32))

    def test_dry_run_calibration_script_fits_scale_offset_and_loads_into_gravcomp(self):
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / 'so101_gravcomp_calib.json'
            cal = run_calibration(dry_run=True, out_path=out_p, seed=123)
            self.assertTrue(out_p.exists())
            loaded = json.loads(out_p.read_text())
            self.assertEqual(loaded['n_poses'], 12)
            self.assertLess(loaded['max_rmse_nm'], 0.02)
            # Pitch joints (lift=1, elbow=2, wflex=3) recover simulated ~[1.08, 0.94, 1.05] scales accurately
            self.assertAlmostEqual(loaded['grav_scale'][1], 1.08, delta=0.05)
            self.assertAlmostEqual(loaded['grav_scale'][2], 0.94, delta=0.05)

            gc_cal = GravCompTL(calib_path=out_p)
            np.testing.assert_allclose(gc_cal.grav_scale, loaded['grav_scale'], atol=1e-6)

    def test_frozen_checkpoint_runner_matches_drawer_and_insertion(self):
        drw_ckpt = REPO_ROOT / 'projects' / 'blind_drawer' / 'phase2_checkpoints' / 'D2_PerJoint_seed101.npz'
        ins_ckpt = REPO_ROOT / 'projects' / 'blind_insertion' / 'phase5_checkpoints' / 'R2_v2_Res_GravComp_TL_seed101.npz'

        runner = GravCompPolicyRunner(drw_ckpt)
        ref = DrawerResidualGRUPPO.load(drw_ckpt)
        rng = np.random.default_rng(7)
        h_ref = ref.init_hidden()
        runner.reset()
        for _ in range(8):
            x = rng.normal(0.0, 0.5, size=runner.obs_dim)
            base = rng.normal(0.0, 0.002, size=3)
            out = runner.step(x, base)
            _, d_ref, a_ref, _, h_ref, c_ref, y_ref = ref.act_step(x, h_ref, base, deterministic=True)
            np.testing.assert_allclose(out.delta_xyz, d_ref, atol=1e-12)
            np.testing.assert_allclose(out.alpha_5dof, a_ref, atol=1e-12)
            self.assertEqual(out.aux_contact_idx, c_ref)
            self.assertAlmostEqual(float(out.aux_extra[0]), y_ref, places=12)

        runner_ins = GravCompPolicyRunner(ins_ckpt)
        self.assertLess(runner.benchmark_latency_ms(200), 0.5)
        self.assertLess(runner_ins.benchmark_latency_ms(200), 0.5)

    def test_tactile_snr_and_auroc(self):
        rng = np.random.default_rng(0)
        e_s = rng.normal(0.0, 0.002, size=(200, 5))
        snr = compute_tactile_snr(e_s, e_s * 4.0)
        self.assertAlmostEqual(snr['amplification_ratio'], 4.0, places=3)
        self.assertAlmostEqual(compute_auroc(np.array([5.0, 6.0]), np.array([1.0, 2.0])), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
