"""Mapping, force-only dynamics and object-state boundaries of the sim servo."""

import unittest

import mujoco
import numpy as np

from joint_mapping import model_action_to_sim
from preview import DEFAULT_MODEL, load_model
from task_dynamics import ServoController, initialize, native_to_public, read_native_state, read_public_state


class TaskDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Arm-only bench isolates numerical control from contacts and a guessed
        # table. The real rollouts keep the scene's collision configuration.
        cls.model = load_model(DEFAULT_MODEL.parent / 'rs_arm.xml')
        cls.model.geom_contype[:] = 0
        cls.model.geom_conaffinity[:] = 0
        cls.model.opt.timestep = 0.001
        cls.pose = np.array([0.1, 1.0, 1.5, -0.3, 0.1, 0.2, 0.02])

    def test_candidate_mapping_roundtrip_includes_gripper_units(self):
        public = np.array([20.0, -90.0, -120.0, 10.0, 15.0, 4.0, -180.0])
        np.testing.assert_allclose(native_to_public(model_action_to_sim(public)), public, atol=1e-12)

    def test_initialization_preserves_free_object_coordinates_and_velocities(self):
        model = load_model(DEFAULT_MODEL)
        data = mujoco.MjData(model)
        free_id = model.joint('red_cube_freejoint').id
        qa, va = model.jnt_qposadr[free_id], model.jnt_dofadr[free_id]
        data.qvel[va:va + 6] = [1, 2, 3, 4, 5, 6]
        before_pose = data.qpos[qa:qa + 7].copy()
        before_velocity = data.qvel[va:va + 6].copy()
        initialize(model, data, self.pose)
        np.testing.assert_array_equal(data.qpos[qa:qa + 7], before_pose)
        np.testing.assert_array_equal(data.qvel[va:va + 6], before_velocity)

    def test_outside_target_rejected_before_controls_change(self):
        data = mujoco.MjData(self.model)
        initialize(self.model, data, self.pose)
        servo = ServoController(self.model)
        original = data.ctrl.copy()
        target = self.pose.copy()
        target[1] = -1
        with self.assertRaisesRegex(ValueError, 'joint2'):
            servo.apply(data, target)
        np.testing.assert_array_equal(data.ctrl, original)

    def test_apply_sets_only_bounded_forces_without_teleport_or_time_step(self):
        data = mujoco.MjData(self.model)
        initialize(self.model, data, self.pose)
        servo = ServoController(self.model)
        target = self.pose.copy()
        target[0] = 2.7
        qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), data.time
        controls = servo.apply(data, target)
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        self.assertEqual(data.time, time)
        self.assertTrue(np.all(controls >= servo.limits[:, 0]))
        self.assertTrue(np.all(controls <= servo.limits[:, 1]))
        self.assertEqual(servo.saturated_steps[0], 1)

    def test_hold_is_stable_and_small_move_occurs_by_physics(self):
        data = mujoco.MjData(self.model)
        initialize(self.model, data, self.pose)
        servo = ServoController(self.model)
        for _ in range(1000):
            servo.step(data, self.pose)
        np.testing.assert_allclose(read_native_state(self.model, data)[:6], self.pose[:6], atol=0.01)
        self.assertTrue(np.isfinite(data.qvel).all())
        target = self.pose.copy()
        target[0] += 0.1
        target[6] += 0.01
        for _ in range(1000):
            servo.step(data, target)
        np.testing.assert_allclose(read_native_state(self.model, data)[:6], target[:6], atol=0.01)
        self.assertAlmostEqual(read_native_state(self.model, data)[6], target[6], delta=0.002)
        self.assertAlmostEqual(data.time, 2.0, places=9)
        np.testing.assert_allclose(model_action_to_sim(read_public_state(self.model, data)),
                                   read_native_state(self.model, data), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
