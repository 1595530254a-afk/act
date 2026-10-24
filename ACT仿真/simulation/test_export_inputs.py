"""Narrow input-integrity tests; no model load, camera or motor SDK."""
import json
from pathlib import Path
import tempfile
import unittest

from export_inputs import export_inputs, load_frame0, validate_episode

import numpy as np


class InputIntegrityTests(unittest.TestCase):
    def values(self):
        return [np.zeros((3, 7), np.float32), np.ones((3, 7), np.float32),
                np.arange(3, dtype=np.float32) / 30, np.arange(3), np.arange(3)]

    def validate(self, values):
        return validate_episode(*values, expected_length=3, dataset_from_index=0, fps=30)

    def test_preserves_recorded_values_without_padding_or_mapping(self):
        values = self.values()
        values[0][:, 2] = [-80, -40, -10]
        values[1][:, 6] = [-150, -120, -90]
        arrays = self.validate(values)
        np.testing.assert_array_equal(arrays['episode0_state'], values[0])
        np.testing.assert_array_equal(arrays['episode0_action'], values[1])
        self.assertEqual(arrays['episode0_state'].shape, (3, 7))

    def test_missing_frame_nonfinite_and_bad_clock_are_rejected(self):
        cases = ('short', 'nonfinite', 'frame_gap', 'global_gap', 'wrong_time', 'float_index')
        for case in cases:
            with self.subTest(case=case):
                values = self.values()
                if case == 'short': values[0] = values[0][:2]
                elif case == 'nonfinite': values[1][1, 3] = np.nan
                elif case == 'frame_gap': values[3][1] = 9
                elif case == 'global_gap': values[4][1] = 9
                elif case == 'wrong_time': values[2][1] = 0
                else: values[3] = values[3].astype(np.float32)
                with self.assertRaises(ValueError):
                    self.validate(values)

    def fixture(self, root, state=None, frame_index=0):
        provenance = {'dataset': {'repo_id': 'dataset', 'revision': 'data-rev'},
                      'model': {'repo_id': 'model', 'revision': 'model-rev'}}
        source = {**provenance, 'frame_indices': [frame_index], 'episode_indices': [0],
                  'image_conversion': 'test fixture only'}
        (root / 'provenance.json').write_text(json.dumps(source), encoding='utf-8')
        pixels = np.zeros((1, 3, 480, 640), np.uint8)
        pixels[0, 0, 0, 0] = 17
        np.savez_compressed(root / 'observations.npz', state=np.zeros((1, 7), np.float32)
                            if state is None else state, wrist=pixels, top=pixels)
        return provenance

    def test_fixture_selection_preserves_rgb_channel_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provenance = self.fixture(root)
            frame, source = load_frame0(root, provenance, self.validate(self.values()))
            self.assertEqual(frame['frame0_wrist_rgb'].shape, (480, 640, 3))
            self.assertEqual(frame['frame0_wrist_rgb'][0, 0].tolist(), [17, 0, 0])
            self.assertEqual(source['original_global_frame_index'], 0)

    def test_wrong_revision_or_wrong_original_frame_or_state_is_rejected(self):
        for case in ('revision', 'frame', 'state'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                provenance = self.fixture(root, state=np.ones((1, 7), np.float32) if case == 'state' else None,
                                          frame_index=12 if case == 'frame' else 0)
                if case == 'revision': provenance['dataset'] = {'repo_id': 'dataset', 'revision': 'other'}
                with self.assertRaises(ValueError):
                    load_frame0(root, provenance, self.validate(self.values()))

    def test_existing_export_is_never_overwritten_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / 'inputs.npz'
            artifact.write_bytes(b'existing evidence')
            with self.assertRaises(FileExistsError):
                export_inputs(root)
            self.assertEqual(artifact.read_bytes(), b'existing evidence')


if __name__ == '__main__':
    unittest.main()
