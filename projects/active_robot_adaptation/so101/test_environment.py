"""Meaningful smoke tests for the model and controller observation boundary."""
import unittest
import numpy as np
from environment import Environment, GlobalCorrection, locate_green, pixel_to_plane, recalibrate_translation
from rl_interface import CartesianInterface


class SimTest(unittest.TestCase):
    def test_reset_observation_and_camera_calibration(self):
        env=Environment(camera_shift=(.012,-.008))
        try:
            first=env.reset()
            self.assertEqual(set(first),{'joint_rad','time_s','image'})
            shift=recalibrate_translation(first['image'])
            self.assertIsNotNone(shift)
            self.assertLess(np.linalg.norm(shift-[.012,-.008]),.004)
            estimate=pixel_to_plane(locate_green(first['image']),offset_xy=shift)
            self.assertLess(np.linalg.norm(estimate[:2]-[.22,.024]),.004)
            again=env.reset()
            np.testing.assert_allclose(first['joint_rad'],again['joint_rad'],atol=1e-12)
            np.testing.assert_array_equal(first['image'],again['image'])
        finally:
            env.close()

    def test_joint_command_limit_and_time(self):
        env=Environment(render=False)
        try:
            before=env.reset();after=env.step(np.ones(5),.5)
            self.assertAlmostEqual(after['time_s']-before['time_s'],.1,places=8)
            self.assertEqual(env.saturated_actions,1)
            np.testing.assert_allclose(env.trace[-1]['sent_delta_rad'],.08)
        finally:
            env.close()

    def test_success_requires_grasp_lift_and_settled_destination(self):
        env=Environment(render=False)
        try:
            env._evaluation={'initial_z':.02,
                'after_close':{'contacts':['fixed_jaw_box1','moving_jaw_box2']},
                'after_lift':{'object_xyz':[.22,.0,.06]},
                'after_settle':{'object_xyz':[.22,-.05,.02]}}
            self.assertTrue(env.evaluate()['success'])
            env._evaluation['after_lift']['object_xyz'][2]=.02
            self.assertFalse(env.evaluate()['success'])
            env._evaluation['after_lift']['object_xyz'][2]=.06
            env._evaluation['after_settle']['object_xyz'][0]=.35
            self.assertFalse(env.evaluate()['success'])
        finally:
            env.close()

    def test_global_model_learns_a_reduced_response(self):
        g=GlobalCorrection()
        command=np.array([.015,0.,0.])
        for _ in range(40): g.update(np.zeros(3),command,command*.5)
        self.assertGreater(g.correct(command,np.zeros(3))[0],command[0])

    def test_cartesian_adapter_hides_truth_and_has_no_reward(self):
        task=CartesianInterface()
        try:
            first=task.reset(seed=17)
            second=task.step([.005,0,0,1.2])
            self.assertEqual(set(second),{'joint_rad','time_s','image'})
            self.assertAlmostEqual(second['time_s']-first['time_s'],.1,places=8)
            with self.assertRaises(ValueError): task.step([1,2,3])
        finally: task.close()

    def test_seeded_camera_noise_repeats(self):
        env=Environment(camera_noise_std=2.)
        try:
            a=env.reset(seed=31)['image']
            b=env.reset(seed=31)['image']
            c=env.reset(seed=32)['image']
            np.testing.assert_array_equal(a,b)
            self.assertFalse(np.array_equal(a,c))
        finally: env.close()


if __name__=='__main__': unittest.main()
