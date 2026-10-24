"""Pure NumPy checks; no policy loading, simulator stepping, or training."""
from contextlib import redirect_stderr
import io
import unittest

import evaluate_act_ensemble as evaluation
import numpy as np


class SparseEnsembleTests(unittest.TestCase):
    def chunk(self, value):
        return np.full((100, 7), value, dtype=np.float32)

    def test_expired_chunks_are_removed_and_oldest_rank_restarts(self):
        ensemble = evaluation.SparseTemporalEnsembler()
        ensemble.add(0, self.chunk(1.))
        ensemble.add(10, self.chunk(9.))
        _, sources_before = ensemble.action(99)
        self.assertEqual([item['query_index'] for item in sources_before], [0, 1])
        action, sources_after = ensemble.action(100)
        np.testing.assert_array_equal(action, np.full(7, 9.))
        self.assertEqual(len(ensemble.chunks), 1)
        self.assertEqual(sources_after, [{'query_index': 1, 'start_frame': 10,
                                        'chunk_row': 90, 'normalized_weight': 1.}])

    def test_absolute_frame_selects_correct_row_from_each_chunk(self):
        ensemble = evaluation.SparseTemporalEnsembler()
        first = np.tile(np.arange(100, dtype=np.float32)[:, None], (1, 7))
        ensemble.add(0, first)
        ensemble.add(10, first + 1000.)
        action, sources = ensemble.action(15)
        old, new = 1., np.exp(-.01)
        expected = (15. * old + 1005. * new) / (old + new)
        np.testing.assert_allclose(action, np.full(7, expected), rtol=0, atol=1e-12)
        self.assertEqual([item['chunk_row'] for item in sources], [15, 5])

    def test_positive_coefficient_weights_oldest_prediction_more(self):
        ensemble = evaluation.SparseTemporalEnsembler()
        ensemble.add(0, self.chunk(0.))
        ensemble.add(10, self.chunk(1000.))
        action, sources = ensemble.action(10)
        self.assertGreater(sources[0]['normalized_weight'], sources[1]['normalized_weight'])
        self.assertLess(action[0], 500.)
        np.testing.assert_allclose(action[0], 1000. * np.exp(-.01) / (1. + np.exp(-.01)),
                                   rtol=0, atol=1e-12)

    def test_constant_predictions_remain_constant_across_overlaps(self):
        ensemble = evaluation.SparseTemporalEnsembler()
        for frame in range(200):
            if frame % 10 == 0:
                ensemble.add(frame, self.chunk(17.5))
            action, sources = ensemble.action(frame)
            np.testing.assert_allclose(action, np.full(7, 17.5), rtol=0, atol=1e-12)
            self.assertAlmostEqual(sum(item['normalized_weight'] for item in sources), 1., places=14)
            self.assertLessEqual(len(sources), 10)

    def test_queries_every_100_frames_reduce_exactly_to_original_chunk_execution(self):
        ensemble = evaluation.SparseTemporalEnsembler()
        rng = np.random.default_rng(20261023)
        chunks = rng.standard_normal((5, 100, 7)).astype(np.float32)
        actions = []
        for frame in range(450):
            if frame % 100 == 0:
                ensemble.add(frame, chunks[frame // 100])
            action, sources = ensemble.action(frame)
            self.assertEqual(len(sources), 1)
            actions.append(action)
        np.testing.assert_array_equal(np.asarray(actions), chunks.reshape(-1, 7)[:450])

    def test_chunk_is_copied_and_external_mutation_cannot_change_actions(self):
        ensemble = evaluation.SparseTemporalEnsembler()
        chunk = self.chunk(3.)
        ensemble.add(0, chunk)
        chunk[:] = 500.
        np.testing.assert_array_equal(ensemble.action(0)[0], np.full(7, 3.))

    def test_future_chunk_is_not_used_before_its_start(self):
        ensemble = evaluation.SparseTemporalEnsembler()
        ensemble.add(10, self.chunk(3.))
        with self.assertRaisesRegex(ValueError, 'No ACT prediction covers'):
            ensemble.action(9)
        np.testing.assert_array_equal(ensemble.action(10)[0], np.full(7, 3.))

    def test_expired_only_prediction_cannot_silently_repeat_last_action(self):
        ensemble = evaluation.SparseTemporalEnsembler()
        ensemble.add(0, self.chunk(3.))
        with self.assertRaisesRegex(ValueError, 'No ACT prediction covers'):
            ensemble.action(100)

    def test_invalid_chunks_and_start_order_are_rejected(self):
        for chunk in (np.zeros((99, 7)), np.full((100, 7), np.nan),
                      np.zeros((100, 7), dtype=complex), np.zeros((100, 7), dtype=bool)):
            with self.assertRaises(ValueError):
                evaluation.SparseTemporalEnsembler().add(0, chunk)
        ensemble = evaluation.SparseTemporalEnsembler()
        ensemble.add(0, self.chunk(3.))
        with self.assertRaises(ValueError):
            ensemble.add(0, self.chunk(3.))
        ensemble.action(9)
        with self.assertRaises(ValueError):
            ensemble.add(5, self.chunk(3.))
        with self.assertRaises(ValueError):
            ensemble.action(8)

    def test_cli_cannot_select_test_or_external_protocol(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluation.parse_args(['--output-dir', 'unused', '--split', 'test'])
            with self.assertRaises(SystemExit):
                evaluation.parse_args(['--output-dir', 'unused', '--teacher-protocol', 'other_protocol.json'])
        args = evaluation.parse_args(['--output-dir', 'unused'])
        self.assertEqual(args.teacher_protocol, evaluation.POSITION_PROTOCOL)


if __name__ == '__main__':
    unittest.main()
