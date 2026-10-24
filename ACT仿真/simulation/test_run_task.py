"""Checks that the task report cannot award placement for a partial overlap."""
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from run_task import brick_metrics, check_physics, parse_args, write_page


class TaskReportTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
<body name="target_brick" pos="0 0 .014"><freejoint name="target_brick_freejoint"/>
<geom name="target_brick_geom" type="box" size=".016 .012 .01" mass=".02"/></body>
<body name="target_bin"><geom name="target_bin_floor" type="box" pos="0 0 .002" size=".06 .06 .002"/>
<geom name="target_bin_wall_e" type="box" pos=".058 0 .025" size=".002 .06 .025"/>
<geom name="target_bin_wall_w" type="box" pos="-.058 0 .025" size=".002 .06 .025"/>
<geom name="target_bin_wall_n" type="box" pos="0 .058 .025" size=".06 .002 .025"/>
<geom name="target_bin_wall_s" type="box" pos="0 -.058 .025" size=".06 .002 .025"/>
</body></worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def metrics(self, lift=.04):
        return brick_metrics(self.model, self.data, np.array([.2, 0, .01]), {}, lift)

    def test_completed_lift_and_rest_inside_bin(self):
        self.assertTrue(self.metrics()['success'])

    def test_center_inside_but_edge_outside_is_not_placement(self):
        self.data.qpos[0] = .05
        mujoco.mj_forward(self.model, self.data)
        self.assertFalse(self.metrics()['entire_brick_footprint_inside_bin'])
        self.assertFalse(self.metrics()['success'])

    def test_object_must_have_lifted_and_be_resting(self):
        self.assertFalse(self.metrics(lift=0)['success'])
        self.data.qvel[0] = .1
        self.assertFalse(self.metrics()['success'])

    def test_time_reset_and_mujoco_numerical_warning_rejected(self):
        with self.assertRaises(FloatingPointError):
            check_physics(self.data, 0.)
        self.data.time = .1
        check_physics(self.data, 0.)
        self.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number = 1
        with self.assertRaisesRegex(FloatingPointError, 'BADQACC'):
            check_physics(self.data, 0.)


class TaskSelectionTests(unittest.TestCase):
    def test_explicit_model_and_flat_scene_are_parsed(self):
        with patch.object(Path, 'is_file', return_value=True):
            args = parse_args(['--model-dir', 'local/adapted', '--flat-brick', '--action-steps', '10'])
        self.assertEqual(args.model_dir, Path('local/adapted').resolve())
        self.assertTrue(args.flat_brick)
        self.assertTrue(args.square_brick)
        self.assertEqual(args.action_steps, 10)

    def test_flat_scene_rejects_other_sizes_before_creating_output(self):
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            parse_args(['--flat-brick', '--brick-edge-mm', '32'])

    def test_model_selection_is_exclusive(self):
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                parse_args(['--model-dir', 'local/adapted', '--checkpoint', 'published'])

    def test_missing_explicit_weights_fail_before_rollout(self):
        with patch.object(Path, 'is_file', return_value=False), patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                parse_args(['--model-dir', 'local/nonexistent'])

    def test_old_selection_remains_available(self):
        args = parse_args(['--checkpoint', 'finetuned'])
        self.assertEqual(args.checkpoint, 'finetuned')
        self.assertIsNone(args.model_dir)
        self.assertFalse(args.flat_brick)


class CurrentPageTests(unittest.TestCase):
    def report(self, success=False):
        return {'status': 'completed', 'executed_actions_per_chunk': 10,
            'scene': {'brick': {'body_dimensions_m': [.04, .04, .02], 'flat_top': True},
                      'bin': {'outside_dimensions_m': [.12, .12, .05]}},
            'results': {'custom': {'label': '仿真适配 ACT 权重', 'status': 'completed',
                'model_directory': 'D:/ACT/checkpoint_300', 'model_sha256': 'a' * 64,
                'metrics': {'success': success, 'lifted_at_least_3cm': success},
                'actual_inference_queries': 41}}}

    def page(self, report):
        with patch.object(Path, 'write_text') as write_text:
            write_page(Path('report'), report)
        return write_text.call_args.args[0]

    def test_single_model_report_uses_actual_result_and_inputs(self):
        page = self.page(self.report(success=True))
        self.assertIn('1 个策略试次，其中 1 个完成物理抓取入盒', page)
        self.assertIn('40 × 40 × 20 mm（平顶、无凸点）', page)
        self.assertIn('custom/initial_wrist.png', page)
        self.assertIn('D:/ACT/checkpoint_300', page)
        self.assertIn('a' * 64, page)
        self.assertNotIn('published/initial_', page)
        self.assertNotIn('两套权重', page)
        self.assertNotIn('0.26 mm', page)

    def test_failure_is_not_reported_as_grasp_success(self):
        page = self.page(self.report(success=False))
        self.assertIn('1 个策略试次，其中 0 个完成物理抓取入盒', page)

    def test_running_page_does_not_invent_a_result(self):
        report = self.report()
        report['status'], report['results'] = 'running', {}
        page = self.page(report)
        self.assertIn('正在运行，已完成 0 个策略试次', page)
        self.assertIn('等待首个策略试次', page)


if __name__ == '__main__':
    unittest.main()
