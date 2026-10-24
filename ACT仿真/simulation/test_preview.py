"""Boundary checks for the simulation-only loader and kinematic pose writer."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

import preview
from preview import DEFAULT_MODEL, load_inputs, load_model, set_pose, trajectory_diagnostics


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(DEFAULT_MODEL)

    def test_official_scene_and_cameras_compile(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (30, 27, 7))
        self.assertEqual([self.model.camera(i).name for i in range(self.model.ncam)],
                         ['overhead_rgb', 'wrist_rgb'])

    def test_chinese_model_path_restores_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='ACT_模型_') as folder:
            path = Path(folder) / 'scene.xml'
            path.write_text('<mujoco><worldbody/></mujoco>', encoding='utf-8')
            self.assertEqual(load_model(path).nq, 0)
        self.assertEqual(Path.cwd(), previous)

    def test_joint_names_and_coupled_fingers_preserve_scene_objects(self):
        data = mujoco.MjData(self.model)
        original = data.qpos.copy()
        pose = np.array([0.1, 0.2, 0.3, -0.1, 0.1, 0.0, 0.02])
        set_pose(self.model, data, pose)
        for name, value in zip([f'joint{i}' for i in range(1, 8)], pose):
            self.assertAlmostEqual(data.qpos[self.model.joint(name).qposadr[0]], value)
        for name in ('joint_left', 'joint_right'):
            self.assertAlmostEqual(data.qpos[self.model.joint(name).qposadr[0]], pose[6])
        arm_addresses = {int(self.model.joint(name).qposadr[0]) for name in
                         [*[f'joint{i}' for i in range(1, 8)], 'joint_left', 'joint_right']}
        objects = [index for index in range(self.model.nq) if index not in arm_addresses]
        np.testing.assert_array_equal(data.qpos[objects], original[objects])

    def test_out_of_model_range_is_reported_without_clamping(self):
        data = mujoco.MjData(self.model)
        poses = np.zeros((2, 7))
        poses[1, 1] = -1.0
        poses[1, 6] = -0.01
        report = trajectory_diagnostics(self.model, data, poses)
        self.assertEqual(report['joint_limits']['joint2']['outside_frames'], 1)
        self.assertEqual(report['joint_limits']['joint7']['outside_frames'], 1)
        self.assertFalse(report['clamping_applied'])
        self.assertEqual(poses[1, 1], -1.0)

    def test_bad_timestamp_shape_or_rate_is_rejected_before_rendering(self):
        for times in (np.zeros(1), np.zeros(2), np.array([0.0, 0.05])):
            with self.subTest(times=times), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'bad.npz'
                np.savez(path, episode0_state=np.zeros((2, 7)),
                         predicted_action_chunk=np.zeros((100, 7)), episode0_timestamp=times)
                with self.assertRaises(ValueError):
                    load_inputs(path)

    def test_failed_rebuild_does_not_leave_a_success_report(self):
        import json

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            np.savez(directory / 'inputs.npz', episode0_state=np.zeros((2, 7)),
                     predicted_action_chunk=np.zeros((100, 7)),
                     episode0_timestamp=np.array([0.0, 1 / 30]))
            (directory / 'index.html').write_text('OLD SUCCESS', encoding='utf-8')
            with patch('sys.argv', ['preview.py', '--run-dir', str(directory)]), \
                 patch.object(preview, 'render_clip', side_effect=RuntimeError('encoder failed')):
                with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
                    preview.main()
            report = json.loads((directory / 'verification.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'rendering_failed')
            self.assertNotIn('OLD SUCCESS', (directory / 'index.html').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
