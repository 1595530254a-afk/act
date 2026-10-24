"""Validate the interprocess observation contract without loading ACT weights."""
import unittest
from unittest.mock import patch

import numpy as np

import policy_worker


class ObservationContractTests(unittest.TestCase):
    def observation(self):
        return {'state': np.zeros(7, dtype=np.float32),
            'wrist': np.zeros((480, 640, 3), dtype=np.uint8),
            'top': np.zeros((480, 640, 3), dtype=np.uint8)}

    def test_accepts_camera_pixels_and_finite_float_state(self):
        actual = policy_worker.validate_observation(self.observation())
        self.assertEqual(set(actual), {'state', 'wrist', 'top'})

    def test_rejects_nonfinite_state(self):
        arrays = self.observation()
        arrays['state'][2] = np.nan
        with self.assertRaises(ValueError):
            policy_worker.validate_observation(arrays)

    def test_rejects_missing_camera(self):
        arrays = self.observation()
        del arrays['top']
        with self.assertRaises(ValueError):
            policy_worker.validate_observation(arrays)

    def test_rejects_chw_and_normalized_float_images(self):
        for invalid in (np.zeros((3, 480, 640), dtype=np.uint8),
                        np.zeros((480, 640, 3), dtype=np.float32)):
            arrays = self.observation()
            arrays['wrist'] = invalid
            with self.assertRaises(ValueError):
                policy_worker.validate_observation(arrays)

    def test_rejects_relative_path_and_input_overwrite(self):
        with self.assertRaises(ValueError):
            policy_worker.absolute_npz('relative.npz')
        path = str((policy_worker.ROOT / 'same.npz').resolve())
        with patch.object(policy_worker, 'predict_chunk') as prediction:
            with self.assertRaises(ValueError):
                policy_worker.handle_request(None, {'input': path, 'output': path}, 1)
            prediction.assert_not_called()


if __name__ == '__main__':
    unittest.main()
