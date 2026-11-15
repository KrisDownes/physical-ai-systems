"""Checks the bottle mass model and controller/evaluator boundary."""
import unittest
import numpy as np
from bottle_task import BottleEnvironment,physical_parameters,RADIUS,HEIGHT

class BottleModelTest(unittest.TestCase):
    def test_mass_and_valid_inertia(self):
        p=[physical_parameters(x) for x in ('empty','half','full')]
        self.assertAlmostEqual(p[1]['water_kg'],p[2]['water_kg']/2)
        self.assertEqual(p[0]['water_kg'],0.)
        self.assertTrue(p[0]['mass_kg']<p[1]['mass_kg']<p[2]['mass_kg'])
        for item in p:
            i=item['inertia_kg_m2'];self.assertTrue(all(x>0 for x in i));self.assertLessEqual(max(i),sum(i)-max(i))

    def test_condition_is_not_in_observation(self):
        env=BottleEnvironment('full','wet',render=False)
        try:
            obs=env.reset();self.assertEqual(set(obs),{'joint_rad','time_s'})
            np.testing.assert_allclose(env.model.geom_size[env.geom,:2],[RADIUS,HEIGHT/2])
            self.assertAlmostEqual(env.model.body_mass[env.object_id],physical_parameters('full')['mass_kg'])
        finally:env.close()
