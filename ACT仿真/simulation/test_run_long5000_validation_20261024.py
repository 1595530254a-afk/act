"""Pure ranking and training-completion gate tests; never launch evaluation."""
from pathlib import Path
import unittest
from unittest.mock import patch

import run_long5000_validation_20261024 as runner


def candidate(name, **changes):
    value = {'id': name, 'baseline_success': True, 'validation_successes': 2,
        'validation_sustained_grasps': 2, 'offline_validation_normalized_mae': .08,
        'mode': 'chunk100', 'checkpoint_step': 1000}
    return {**value, **changes}


class RankingTests(unittest.TestCase):
    def first(self, *candidates):
        return runner.ranked_eligible(list(candidates))[0]['id']

    def test_failed_same_mode_baseline_is_ineligible_even_if_validation_is_perfect(self):
        valid = candidate('valid', validation_successes=0)
        invalid = candidate('invalid', baseline_success=False, validation_successes=4,
                            validation_sustained_grasps=4, offline_validation_normalized_mae=.001)
        self.assertEqual(runner.ranked_eligible([invalid, valid]), [valid])

    def test_complete_task_successes_precede_grasp_count_and_offline_loss(self):
        self.assertEqual(self.first(candidate('more_success', validation_successes=3,
                                             offline_validation_normalized_mae=10),
                                    candidate('less_success', validation_sustained_grasps=4,
                                              offline_validation_normalized_mae=.001)), 'more_success')

    def test_sustained_grasps_break_equal_task_successes_before_loss(self):
        self.assertEqual(self.first(candidate('more_grasps', validation_sustained_grasps=3,
                                             offline_validation_normalized_mae=10),
                                    candidate('lower_loss', offline_validation_normalized_mae=.001)), 'more_grasps')

    def test_offline_loss_precedes_execution_mode_tie_break(self):
        self.assertEqual(self.first(candidate('ensemble_better_loss', mode='ensemble1',
                                             offline_validation_normalized_mae=.07),
                                    candidate('chunk')), 'ensemble_better_loss')

    def test_chunk100_is_preferred_before_earlier_checkpoint(self):
        self.assertEqual(self.first(candidate('new_chunk', checkpoint_step=5000),
                                    candidate('early_ensemble', mode='ensemble1', checkpoint_step=600)), 'new_chunk')

    def test_earlier_training_checkpoint_breaks_remaining_tie(self):
        self.assertEqual(self.first(candidate('late', checkpoint_step=5000),
                                    candidate('early', checkpoint_step=600)), 'early')

    def test_empty_eligible_list_does_not_fabricate_fallback_selection(self):
        self.assertEqual(runner.ranked_eligible([candidate('failed', baseline_success=False)]), [])

    def test_running_training_is_rejected_before_loading_checkpoints(self):
        report = {'status': 'running', 'completed_steps': 0}
        with patch.object(runner, 'read_json', return_value=report), patch.object(runner, 'model_hashes') as hashes:
            with self.assertRaisesRegex(ValueError, 'Training must be complete'):
                runner.build_candidates({'training_steps': 5000}, Path('unused'))
            hashes.assert_not_called()


if __name__ == '__main__':
    unittest.main()
