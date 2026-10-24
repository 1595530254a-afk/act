"""Pure CPU contract checks; no camera, motor, model or simulator is opened."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import stage2_policy as runtime
import camera_snapshot as camera
import numpy as np


class Stage2Tests(unittest.TestCase):
    def test_ensemble1_keeps_full_future_rows_and_expires_at100(self):
        ensemble = runtime.SparseTemporalEnsembler()
        chunks, actions = [], []
        rng = np.random.default_rng(20261024)
        for frame in range(120):
            chunk = rng.standard_normal((100, 7)).astype(np.float32)
            chunks.append(chunk)
            ensemble.add(frame, chunk)
            action, sources = ensemble.action(frame)
            actions.append(action)
            self.assertEqual(len(sources), min(frame + 1, 100))
            if frame >= 100:
                self.assertEqual(sources[0]['start_frame'], frame - 99)
            self.assertGreaterEqual(sources[0]['normalized_weight'], sources[-1]['normalized_weight'])
        runtime.audit_ensemble(chunks, actions, list(range(120)))

    def test_sparse_offline_fixtures_do_not_pretend_to_be_consecutive(self):
        ensemble = runtime.SparseTemporalEnsembler()
        chunks, actions = [], []
        for frame, value in zip((0, 170, 300), (1., 9., 3.), strict=True):
            chunk = np.full((100, 7), value, dtype=np.float32)
            ensemble.add(frame, chunk)
            action, sources = ensemble.action(frame)
            np.testing.assert_array_equal(action, np.full(7, value))
            self.assertEqual(len(sources), 1)
            chunks.append(chunk)
            actions.append(action)
        runtime.audit_ensemble(chunks, actions, [0, 170, 300])

    def test_bad_weighted_action_is_detected(self):
        with self.assertRaises(ValueError):
            runtime.audit_ensemble([np.zeros((100, 7))], [np.ones(7)], [0])

    def test_shadow_rejects_cached_or_missing_frames_and_extra_fields(self):
        for request in ({'input': 'obs.npz', 'frame_index': 0},
                        {'input': 'obs.npz', 'frame_index': 2},
                        {'input': 'obs.npz', 'frame_index': 1, 'send_action': True}):
            with self.assertRaises(ValueError):
                runtime.validate_request(request, 1)

    def test_payload_path_cannot_escape_package(self):
        for path in ('../elsewhere', str(runtime.ROOT.resolve())):
            with self.assertRaises(ValueError):
                runtime.package_path(path)

    def test_cli_has_no_hardware_or_execution_switch(self):
        args = runtime.parse_args([])
        self.assertEqual(args.mode, 'offline')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runtime.parse_args(['--hardware'])

    def test_state_requires_fresh_actual_feedback_and_real_mapping(self):
        now = time.monotonic_ns()
        value = {'state': [1.] * 7, 'measured': True, 'mapping_id': 'existing_verified_mapping',
                 'state_order': camera.CHANNELS, 'units': 'public_degree', 'receipt_monotonic_ns': now - 1_000_000}
        with patch.object(Path, 'read_text', return_value=json.dumps(value)):
            self.assertEqual(camera.checked_state(Path('unused'), now), value)
        for field, invalid in (('measured', False), ('mapping_id', ''), ('units', 'sim_rad'),
                               ('receipt_monotonic_ns', now - 251_000_000)):
            bad = {**value, field: invalid}
            with patch.object(Path, 'read_text', return_value=json.dumps(bad)), self.assertRaises(ValueError):
                camera.checked_state(Path('unused'), now)


if __name__ == '__main__':
    unittest.main()
