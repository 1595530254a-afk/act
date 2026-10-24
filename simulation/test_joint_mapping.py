"""Coordinate math and evidence checks only; no MuJoCo/ROS/CAN imports."""
import math
import unittest

import numpy as np

from joint_mapping import (
    DIRECTIONS, MAPPING_STATUS, dataset_state_to_sim, legacy_plugin105_action_to_sim,
    legacy_plugin105_state_to_sim, mapping_metadata, model_action_to_sim,
)


class JointMappingTests(unittest.TestCase):
    def test_canonical_public_degrees_to_native_rs_radians(self):
        state = [90, -45, -90, 30, -30, 180, 0]
        converted = dataset_state_to_sim(state)
        np.testing.assert_allclose(converted[:6], -np.deg2rad(state[:6]))
        self.assertGreater(converted[1], 0)
        self.assertGreater(converted[2], 0)
        self.assertGreater(converted[4], 0)

    def test_canonical_state_and_action_share_the_same_public_frame(self):
        values = [90, -45, -90, 30, -30, 180, 15]
        state = dataset_state_to_sim(values)
        action = model_action_to_sim(values)
        np.testing.assert_array_equal(action, state)
        np.testing.assert_allclose(action[:6], np.deg2rad([-90, 45, 90, -30, 30, -180]))

    def test_external_plugin_legacy_path_stays_separate_and_explicit(self):
        values = [90, -45, -90, 30, -30, 180, 15]
        state = legacy_plugin105_state_to_sim(values)
        action = legacy_plugin105_action_to_sim(values)
        np.testing.assert_allclose(state[:6], np.deg2rad(values[:6]))
        np.testing.assert_allclose(action[:6], np.deg2rad([90, -45, 90, -30, 30, 180]))
        self.assertAlmostEqual(action[6], state[6] * 6)
        self.assertNotEqual(dataset_state_to_sim(values)[1], state[1])

    def test_gripper_five_motor_radians_to_source_nominal_45mm_slide(self):
        public_open_deg = -math.degrees(5.0)
        self.assertAlmostEqual(dataset_state_to_sim([0] * 6 + [public_open_deg])[6], 0.045)
        self.assertAlmostEqual(model_action_to_sim([0] * 6 + [public_open_deg])[6], 0.045)
        self.assertAlmostEqual(legacy_plugin105_action_to_sim([0] * 6 + [-public_open_deg / 6])[6], 0.05)
        self.assertEqual(dataset_state_to_sim([0] * 7)[6], 0)

    def test_no_silent_clip_or_sign_fit_for_invalid_model_ranges(self):
        state = dataset_state_to_sim([0, 90, -190, 0, 0, 0, -math.degrees(6.0)])
        self.assertAlmostEqual(state[1], -math.pi / 2)
        self.assertAlmostEqual(state[2], math.radians(190))
        self.assertAlmostEqual(state[6], 0.054)
        action = model_action_to_sim([0, 90, -190, 0, 0, 0, 10])
        self.assertLess(action[1], 0)  # Real canonical RS driver would clip public J2 to <=0.
        self.assertGreater(action[2], math.pi)
        self.assertLess(action[6], 0)

    def test_batches_and_noncontiguous_arrays_keep_shape(self):
        values = np.arange(2 * 3 * 14, dtype=np.float32).reshape(2, 3, 14)[..., ::2]
        self.assertEqual(values.shape, (2, 3, 7))
        for convert in (dataset_state_to_sim, model_action_to_sim):
            result = convert(values)
            self.assertEqual(result.shape, values.shape)
            self.assertEqual(result.dtype, np.float64)
            self.assertTrue(np.isfinite(result).all())

    def test_inputs_are_not_mutated_or_shared_with_outputs(self):
        values = np.array([[0, -30, -40, 10, -10, 90, 20]], dtype=np.float64)
        before = values.copy()
        values.setflags(write=False)
        for convert in (dataset_state_to_sim, model_action_to_sim):
            result = convert(values)
            np.testing.assert_array_equal(values, before)
            self.assertFalse(np.shares_memory(result, values))
            result[...] = 0
            np.testing.assert_array_equal(values, before)

    def test_invalid_shape_type_and_nonfinite_inputs_rejected(self):
        invalid = [None, 1, [], [0] * 6, [0] * 8, '1234567', [False] * 7,
                   ['1'] * 7, [1j] * 7, [0] * 6 + [None],
                   [0] * 6 + [math.nan], [0] * 6 + [math.inf],
                   [0] * 6 + [-math.inf], {'q': [0] * 7}]
        for values in invalid:
            for convert in (dataset_state_to_sim, model_action_to_sim,
                            legacy_plugin105_state_to_sim, legacy_plugin105_action_to_sim):
                with self.subTest(values=values, convert=convert.__name__), self.assertRaises(ValueError):
                    convert(values)

    def test_action_scale_overflow_refused(self):
        with self.assertRaises(ValueError):
            legacy_plugin105_action_to_sim([0] * 6 + [np.finfo(np.float64).max])

    def test_metadata_stays_candidate_and_is_independent(self):
        metadata = mapping_metadata()
        self.assertEqual(DIRECTIONS, (-1,) * 7)
        self.assertEqual(MAPPING_STATUS, 'CANDIDATE_UNVERIFIED')
        self.assertEqual(metadata['status'], 'CANDIDATE_UNVERIFIED')
        self.assertFalse(metadata['hardware_execution_permitted'])
        self.assertFalse(metadata['dataset_collection_config_verified'])
        self.assertTrue(metadata['state_action_share_public_coordinate_frame'])
        self.assertFalse(metadata['legacy_plugin105_action_factors_applied_to_state'])
        self.assertFalse(metadata['silent_limit_clipping'])
        metadata['canonical_public_to_rs_native_factors'][0] = 999
        self.assertEqual(mapping_metadata()['canonical_public_to_rs_native_factors'][0], -1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
