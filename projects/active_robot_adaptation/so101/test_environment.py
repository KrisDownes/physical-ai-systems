"""Meaningful smoke tests for the model and controller observation boundary."""
import unittest
import numpy as np
from environment import Environment, GlobalCorrection, locate_green, locate_cyan, pixel_to_plane, recalibrate_translation, task_jacobian
from rl_interface import CartesianInterface
from calibration import select_calibration, CalibrationState, VisualTargets


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

    def test_reference_mask_keeps_object_and_tool_marker(self):
        env=Environment(tool_marker=True,reference_mode='unavailable',camera_shift=(.012,-.008))
        try:
            image=env.reset()['image']
            self.assertIsNotNone(locate_green(image))
            marker=locate_cyan(image)
            self.assertIsNotNone(marker)
            self.assertGreater(np.linalg.norm(marker-locate_green(image)),10)
            self.assertIsNone(recalibrate_translation(image))
        finally: env.close()

    def test_physics_audit_detects_reported_nonadjacent_collision(self):
        import mujoco
        env=Environment(render=False)
        try:
            env.reset()
            env.data.qpos[:6]=0
            env.data.qpos[2]=1.55
            mujoco.mj_forward(env.model,env.data)
            env._audit_physics_contacts()
            self.assertGreater(env.physics_collision_steps,0)
            self.assertTrue(any('shoulder' in pair for pair in env.physics_collision_pairs))
        finally: env.close()

    def test_physics_audit_detects_joint_limit_excursion(self):
        env=Environment(render=False)
        try:
            env.reset()
            before=env.physics_limit_steps
            env.data.qpos[0]=env.model.jnt_range[0,1]+.01
            env._audit_physics_contacts()
            self.assertEqual(env.physics_limit_steps,before+1)
        finally: env.close()

    def test_tool_offset_is_hidden_from_nominal_fk_and_moves_marker_and_tip(self):
        import mujoco
        env=Environment(tool_marker=True,tool_offset=(.008,0,0))
        try:
            obs=env.reset()
            nominal,_=env.controller_fk(obs['joint_rad'])
            actual=env.data.site_xpos[env.site]
            self.assertAlmostEqual(np.linalg.norm(actual-nominal),.008,places=6)
            mid=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_SITE,'tool_marker_site')
            marker_delta=env.data.site_xpos[mid]-env.nominal_data.site_xpos[mid]
            np.testing.assert_allclose(marker_delta,actual-nominal,atol=1e-9)
            _,events=select_calibration(env,obs)
            self.assertTrue(any(x.get('reason')=='inconsistent_with_table' for x in events))
        finally:env.close()

    def test_calibration_change_reprojects_both_targets_and_resets_cartesian_only(self):
        from environment import OnlineJacobian
        joint=OnlineJacobian();joint.b[0,0]=.7
        cart=GlobalCorrection();cart.b[0,0]=.5;cart.data.append((np.zeros(3),np.zeros(3),np.zeros(3)))
        state=CalibrationState();state.apply([.01,-.02],joint,cart)
        self.assertEqual(joint.b[0,0],.7)
        self.assertEqual(cart.b[0,0],1.)
        self.assertFalse(cart.data)
        targets=VisualTargets([160,120],[170,140])
        a,b=targets.reproject([0,0]);c,d=targets.reproject([.01,-.02])
        np.testing.assert_allclose(c[:2]-a[:2],[.01,-.02]);np.testing.assert_allclose(d-b,[.01,-.02])

    def test_inconsistent_table_references_do_not_trigger_marker_fallback(self):
        from unittest.mock import patch
        env=Environment(tool_marker=True)
        try:
            obs=env.reset()
            with patch('calibration.table_offsets',return_value=np.array([[0.,0.],[.02,0.]])):
                offset,events=select_calibration(env,obs)
            self.assertIsNone(offset)
            self.assertTrue(any(e.get('reason')=='table_geometry_invalid' for e in events))
        finally:env.close()

    def test_response_model_applies_to_pitch_and_translation(self):
        j=np.arange(15,dtype=float).reshape(3,5)/10
        b=np.eye(5);b[1,1]=.5;b[2,0]=.2
        expected=np.vstack((j,[0.,.1,.1,.1,0.]))@b
        np.testing.assert_allclose(task_jacobian(j,b),expected[:,:4])


if __name__=='__main__': unittest.main()
