"""Unit tests for Phase 6 Zero-Shot OOD Stress Suite: 5 suites, bounds, disjointness, frozen checkpoints."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_insertion.phase5_policy import Phase5ResidualPPO
from projects.blind_insertion.phase6_ood_eval import (
    NONCONVEX_POLYGONS,
    OOD_SUITE_ORDER,
    build_phase6_ood_suites,
    verify_phase6_disjointness,
)


class Phase6OODStressTest(unittest.TestCase):
    def test_five_ood_suites_parameters_and_disjointness(self):
        suites = build_phase6_ood_suites()
        self.assertEqual(tuple(suites.keys()), OOD_SUITE_ORDER)
        verify_phase6_disjointness(suites)

        # 1. OOD_Height_Tilt: +-12 mm AND 3-5 deg tilt
        dzs = {round(o.socket_dz_m, 4) for o in suites['OOD_Height_Tilt']}
        self.assertEqual(dzs, {-0.012, 0.012})
        for o in suites['OOD_Height_Tilt']:
            self.assertGreaterEqual(o.socket_tilt_deg, 3.0)
            self.assertLessEqual(o.socket_tilt_deg, 5.0)

        # 2. OOD_Spring_Mount: k_z=800 N/m, k_xy=600 N/m
        for o in suites['OOD_Spring_Mount']:
            self.assertTrue(o.spring_mount)
            self.assertAlmostEqual(o.spring_k_z, 800.0)
            self.assertAlmostEqual(o.spring_k_xy, 600.0)

        # 3. OOD_NonConvex_Asym: l_shape & keyed_notch at 1.0 mm
        nc_shapes = {o.base.shape for o in suites['OOD_NonConvex_Asym']}
        self.assertEqual(nc_shapes, set(NONCONVEX_POLYGONS.keys()))
        for o in suites['OOD_NonConvex_Asym']:
            self.assertAlmostEqual(o.base.clearance_m, 0.0010, places=6)

        # 4. OOD_Grasp_Error: +-3.5 mm offset AND 4-6 deg tilt
        for o in suites['OOD_Grasp_Error']:
            self.assertAlmostEqual(math.hypot(*o.base.grasp_offset_xy), 0.0035, places=5)
            self.assertGreaterEqual(o.grasp_tilt_deg, 4.0)
            self.assertLessEqual(o.grasp_tilt_deg, 6.0)

        # 5. OOD_UltraTight_0.30mm: 0.30 mm clearance
        for o in suites['OOD_UltraTight_0.30mm']:
            self.assertAlmostEqual(o.base.clearance_m, 0.00030, places=6)

    def test_frozen_phase5_checkpoints_exist_and_load_deterministically(self):
        ckpt_dir = Path(__file__).resolve().parent / 'phase5_checkpoints'
        for label in ('R1_v2_Res_PosOnly', 'R2_v2_Res_GravComp_TL'):
            for s in (101, 202, 303):
                p = ckpt_dir / f'{label}_seed{s}.npz'
                self.assertTrue(p.exists(), f'Missing {p}')
                pol = Phase5ResidualPPO.load(p)
                aw = pol.get_actor_weights()
                self.assertTrue(all(np.all(np.isfinite(v)) for v in aw.values()))


if __name__ == '__main__':
    unittest.main()
