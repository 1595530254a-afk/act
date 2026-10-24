"""Pure hard-limit classification checks; no ACT, renderer or physics steps."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from test_evaluate_feedback_20261024 import fixture
import run_long5000_validation_20261024_v2 as runner


class Model:
    def joint(self, name):
        bounds = [-.1, .1] if name == 'joint7' else [-1e9, 1e9]
        return SimpleNamespace(limited=np.array([True]), range=np.asarray(bounds))


def aborted(mode, frame=52):
    arrays, record = fixture(mode)
    query_fields = ('action_chunks', 'chunk_observation_state', 'query_frame_index', 'query_actual_sim_time')
    keep = arrays['query_frame_index'] <= frame
    arrays = {name: value[keep].copy() if name in query_fields else value[:frame].copy()
              for name, value in arrays.items()}
    for index, start in enumerate(arrays['query_frame_index']):
        arrays['action_chunks'][index, frame - start, 6] = 1000
    record['inference'] = [item for item in record['inference'] if item['frame_index'] <= frame]
    record.update({'policy_actions_applied': frame, 'actual_inference_queries': len(record['inference']),
        'actual_neural_forward_calls': len(record['inference'])})
    action = runner.weighted_action(arrays['action_chunks'], arrays['query_frame_index'], frame)[0]
    try:
        runner.validate_target(Model(), runner.model_action_to_sim(action))
    except ValueError as error:
        record['error'] = f'ValueError: {error}'
    return arrays, record


class LimitAbortTests(unittest.TestCase):
    def test_chunk_limit_is_proven_from_raw_chunk_without_accepting_partial_success(self):
        arrays, record = aborted('chunk100')
        proof = runner.prove_limit_rejection(Model(), arrays, record, 'chunk100')
        self.assertEqual(proof['status'], 'policy_limit_abort')
        self.assertEqual(proof['rejected_frame'], 52)
        self.assertEqual(proof['violations'][0]['joint'], 'joint7')
        self.assertIsNone(proof['task_success'])

    def test_ensemble_fresh_query_on_rejected_frame_is_included(self):
        arrays, record = aborted('ensemble1')
        proof = runner.prove_limit_rejection(Model(), arrays, record, 'ensemble1')
        self.assertEqual(len(proof['query_indices']), 53)
        self.assertEqual(proof['query_indices'][-1], 52)

    def test_zero_applied_frames_still_requires_actual_rejected_raw_prediction(self):
        arrays, record = aborted('ensemble1', 0)
        self.assertEqual(runner.prove_limit_rejection(Model(), arrays, record, 'ensemble1')['applied_action_frames'], 0)

    def test_unrelated_error_is_never_ignored_as_a_limit_abort(self):
        arrays, record = aborted('chunk100')
        record['error'] = 'ValueError: unexpected data corruption'
        with self.assertRaisesRegex(ValueError, 'does not explain'):
            runner.prove_limit_rejection(Model(), arrays, record, 'chunk100')

    def test_legal_raw_target_cannot_be_classified_by_a_matching_error_string(self):
        arrays, record = aborted('chunk100')
        arrays['action_chunks'][0, 52, 6] = -10
        with self.assertRaisesRegex(ValueError, 'passes the original hard limits'):
            runner.prove_limit_rejection(Model(), arrays, record, 'chunk100')

    def test_tampered_applied_prefix_or_missing_query_is_rejected(self):
        arrays, record = aborted('ensemble1')
        bad = deepcopy(arrays)
        bad['target_native'][1, 0] += .01
        with self.assertRaises(ValueError):
            runner.prove_limit_rejection(Model(), bad, record, 'ensemble1')
        bad = deepcopy(arrays)
        bad['action_chunks'] = bad['action_chunks'][:-1]
        with self.assertRaises(ValueError):
            runner.prove_limit_rejection(Model(), bad, record, 'ensemble1')


if __name__ == '__main__':
    unittest.main()
