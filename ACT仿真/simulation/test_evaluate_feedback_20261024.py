"""Pure scheduler/provenance checks: no model load, renderer or physics steps."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import unittest

import evaluate_feedback_20261024 as evaluation
import numpy as np


def fixture(mode):
    starts = evaluation.query_frames(mode)
    chunks = np.random.default_rng(20261024).standard_normal((len(starts), 100, 7)).astype(np.float32)
    public = np.tile(np.arange(450, dtype=float)[:, None], (1, 7))
    initial = np.full(7, -1., dtype=float)
    arrays = {'action_chunks': chunks, 'query_frame_index': starts, 'frame_index': np.arange(450),
        'public_state': public, 'chunk_observation_state': np.stack([
            initial if frame == 0 else public[frame - 1] for frame in starts]),
        'query_actual_sim_time': starts / 30, 'sim_time': (np.arange(450) + 1) / 30}
    ensemble = evaluation.SparseTemporalEnsembler()
    targets, actions, indices, rows, weights, counts = [], [], [], [], [], []
    query = 0
    for frame in range(450):
        if frame % evaluation.QUERY_EVERY[mode] == 0:
            ensemble.add(frame, chunks[query])
            query += 1
        action, sources = ensemble.action(frame)
        q = np.full(100, -1, dtype=np.int64)
        r, w = q.copy(), np.zeros(100)
        for index, source in enumerate(sources):
            q[index], r[index], w[index] = source['query_index'], source['chunk_row'], source['normalized_weight']
        actions.append(action)
        targets.append(evaluation.model_action_to_sim(action))
        indices.append(q)
        rows.append(r)
        weights.append(w)
        counts.append(len(sources))
    arrays.update({'target_native': np.asarray(targets), 'ensembled_public_action': np.asarray(actions),
        'active_prediction_count': np.asarray(counts), 'ensemble_query_indices': np.asarray(indices),
        'ensemble_chunk_rows': np.asarray(rows), 'ensemble_weights': np.asarray(weights)})
    result = {'initial_public_state': initial.tolist(),
        'evaluation': {'physics_dt_s': .001},
        'inference': [{'frame_index': int(frame), 'request_index': index + 1} for index, frame in enumerate(starts)]}
    return arrays, result


class FeedbackTests(unittest.TestCase):
    def test_modes_count_actual_neural_queries_instead_of_action_dequeues(self):
        np.testing.assert_array_equal(evaluation.query_frames('chunk100'), [0, 100, 200, 300, 400])
        np.testing.assert_array_equal(evaluation.query_frames('ensemble1'), np.arange(450))

    def test_both_complete_timelines_reconstruct_every_target(self):
        for mode, count in [('chunk100', 5), ('ensemble1', 450)]:
            arrays, result = fixture(mode)
            audit = evaluation.audit_arrays(arrays, result, mode)
            self.assertEqual(audit['query_count'], count)
            self.assertEqual(audit['action_frames'], 450)

    def test_chunk100_exactly_preserves_nonoverlapping_action_sequence(self):
        arrays, _ = fixture('chunk100')
        np.testing.assert_array_equal(arrays['ensembled_public_action'], arrays['action_chunks'].reshape(-1, 7)[:450])
        self.assertTrue((arrays['active_prediction_count'] == 1).all())

    def test_ensemble_uses_100_covering_chunks_after_warmup_and_oldest_has_highest_weight(self):
        arrays, _ = fixture('ensemble1')
        self.assertEqual(arrays['active_prediction_count'][99], 100)
        np.testing.assert_array_equal(arrays['ensemble_query_indices'][100], np.arange(1, 101))
        self.assertGreater(arrays['ensemble_weights'][100, 0], arrays['ensemble_weights'][100, -1])
        self.assertEqual(arrays['ensemble_chunk_rows'][100, 0], 99)
        self.assertEqual(arrays['ensemble_chunk_rows'][100, -1], 0)

    def test_post_action_state_cannot_be_used_as_same_frame_observation(self):
        arrays, result = fixture('ensemble1')
        arrays['chunk_observation_state'][17] = arrays['public_state'][17]
        with self.assertRaisesRegex(ValueError, 'actual state before'):
            evaluation.audit_arrays(arrays, result, 'ensemble1')

    def test_provenance_or_query_tampering_is_rejected(self):
        arrays, result = fixture('ensemble1')
        bad = deepcopy(arrays)
        bad['ensemble_chunk_rows'][101, 0] -= 1
        with self.assertRaises(ValueError):
            evaluation.audit_arrays(bad, result, 'ensemble1')
        bad = deepcopy(result)
        bad['inference'][100]['request_index'] = 100
        with self.assertRaises(ValueError):
            evaluation.audit_arrays(arrays, bad, 'ensemble1')

    def test_wrong_sampling_rate_or_missing_raw_chunks_is_rejected(self):
        arrays, result = fixture('ensemble1')
        with self.assertRaises(ValueError):
            evaluation.audit_arrays(arrays, result, 'chunk100')
        arrays['query_actual_sim_time'][1] += .05
        with self.assertRaisesRegex(ValueError, 'simulation times'):
            evaluation.audit_arrays(arrays, result, 'ensemble1')

    def test_position_selection_only_returns_split_id_and_xy(self):
        plan = {'positions': [{'id': 'baseline', 'split': 'baseline', 'offset_xy_m': [0., 0.],
                               'teacher_action': 'must not enter selected inputs'},
                              {'id': 'newtest_00', 'split': 'test', 'offset_xy_m': [-.03, -.03]}]}
        self.assertEqual(evaluation.select_positions(plan, 'baseline'),
                         [{'id': 'baseline', 'split': 'baseline', 'offset_xy_m': [0., 0.]}])
        with self.assertRaises(ValueError):
            evaluation.select_positions({'positions': [{'id': '../escape', 'split': 'test', 'offset_xy_m': [0., 0.]}]}, 'test')

    def test_cli_does_not_offer_a_fake_20Hz_mode(self):
        base = ['--position-plan', 'unused', '--split', 'validation', '--model-dir', 'unused', '--output-dir', 'unused']
        self.assertEqual(evaluation.parse_args(base + ['--mode', 'ensemble1']).mode, 'ensemble1')
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluation.parse_args(base + ['--mode', 'ensemble20'])


if __name__ == '__main__':
    unittest.main()
