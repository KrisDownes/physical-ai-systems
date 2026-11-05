"""Semantic checks for independent pose judgment and temporal support."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import interpolate
from evaluate import compose, inverse, errors


class Contracts(unittest.TestCase):
    def test_body_forward_at_ninety_degrees(self):
        np.testing.assert_allclose(compose(np.array([2,3,np.pi/2]), np.array([1,0,0])),
                                   [2,4,np.pi/2], atol=1e-12)

    def test_omnidirectional_motion_and_inverse(self):
        # Lateral increments matter for Dingo; full SE(2), not unicycle.
        pose = np.array([4.,-3.,1.2])
        delta = np.array([0.,1.,-.3])
        np.testing.assert_allclose(compose(inverse(pose),compose(pose,delta)),delta,atol=1e-12)

    def test_yaw_crosses_pi_on_short_arc(self):
        value, valid = interpolate([0,.1],np.array([[0,0,np.deg2rad(179)],
                                                   [1,0,np.deg2rad(-179)]]),[.05])
        self.assertTrue(valid[0]); self.assertAlmostEqual(abs(value[0,2]),np.pi)

    def test_no_extrapolation_or_bridging_large_gap(self):
        _, valid = interpolate([0,.1,1.],np.zeros((3,3)),[-.01,.05,.5,1.01])
        np.testing.assert_array_equal(valid,[False,True,False,False])

    def test_yaw_error_wrap_and_no_fitted_alignment(self):
        p, a = errors(np.array([[1,0,np.deg2rad(-179)]]),
                      np.array([[0,0,np.deg2rad(179)]]))
        self.assertAlmostEqual(p[0],1); self.assertAlmostEqual(np.rad2deg(a[0]),2)


if __name__ == '__main__': unittest.main()
