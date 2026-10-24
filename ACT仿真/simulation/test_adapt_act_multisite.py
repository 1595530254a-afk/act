"""CPU-only checks for episode boundaries and independent validation inputs."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json

import adapt_act_multisite as training

training.configure_torch()
import numpy as np
import torch


class FakeEpisode:
    def __init__(self, value, count=125, episode_id='train_a', split='train', offset=(0., 0.)):
        self.entry = {'id': episode_id, 'split': split, 'offset_xy_m': list(offset)}
        self.state = np.full((count, 7), value, dtype=np.float32)
        self.action = self.state.copy()
        self.rgb = np.zeros((3, 480, 640), dtype=np.uint8)

    def __len__(self):
        return len(self.state)

    def images(self, index):
        return {'wrist': self.rgb, 'top': self.rgb}


class MultiEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.first = FakeEpisode(11.)
        self.second = FakeEpisode(99., count=133, episode_id='train_b', offset=(.02, 0.))
        self.dataset = training.MultiEpisodeDemonstrations([self.first, self.second])

    def test_last_action_padding_does_not_enter_next_episode(self):
        batch = self.dataset.batch(np.array([124, 125, 257]), torch)
        self.assertTrue(torch.equal(batch['action'][0], torch.full((100, 7), 11.)))
        self.assertEqual(batch['action_is_pad'][0].tolist(), [False] + [True] * 99)
        self.assertTrue(torch.equal(batch['action'][1], torch.full((100, 7), 99.)))
        self.assertFalse(batch['action_is_pad'][1].any())
        self.assertEqual(batch['action_is_pad'][2].tolist(), [False] + [True] * 99)

    def test_chunk_near_boundary_repeats_own_last_action(self):
        self.first.action[-2] = 12.
        self.first.action[-1] = 13.
        batch = self.dataset.batch(np.array([123]), torch)
        self.assertEqual(batch['action'][0, 0, 0].item(), 12.)
        self.assertTrue(torch.equal(batch['action'][0, 1:], torch.full((99, 7), 13.)))
        self.assertEqual(batch['action_is_pad'][0].tolist(), [False, False] + [True] * 98)

    def test_invalid_indices_rejected(self):
        for index in (-1, len(self.dataset), .5):
            with self.assertRaises(IndexError):
                self.dataset.locate(index)

    def test_validation_predict_receives_only_observations(self):
        episode = FakeEpisode(20., split='validation', episode_id='validation_a', offset=(.01, .01))
        validation = training.MultiEpisodeDemonstrations([episode])
        calls = []
        keys = ('observation.state', 'observation.images.wrist', 'observation.images.top')

        class FakePolicy:
            config = SimpleNamespace(input_features=dict.fromkeys(keys))

            def eval(self):
                return self

            def predict_action_chunk(self, observation):
                if set(observation) != set(keys):
                    raise AssertionError('Inference was given labels or undeclared input')
                calls.append(observation['observation.state'][0].tolist())
                return torch.zeros((1, 100, 7))

        result = training.position_diagnostic(validation, FakePolicy(), lambda value: value,
                                              lambda value: value, torch)
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(state == [20.] * 7 for state in calls))
        self.assertEqual(result['normalized_chunk_mae'], 20.)
        self.assertEqual(result['public_degree_chunk_mae'], 20.)
        self.assertFalse(result['inference_has_action_labels'])
        self.assertFalse(result['physical_grasp_success_evaluated'])
        self.assertEqual(result['per_episode'][0]['valid_target_rows'], 290)

    def test_validation_macro_average_weights_episodes_equally(self):
        episodes = [FakeEpisode(10., count=125, split='validation'),
                    FakeEpisode(30., count=220, split='validation', episode_id='validation_b')]
        keys = ('observation.state', 'observation.images.wrist', 'observation.images.top')
        policy = SimpleNamespace(config=SimpleNamespace(input_features=dict.fromkeys(keys)),
                                 eval=lambda: None,
                                 predict_action_chunk=lambda observation: torch.zeros((1, 100, 7)))
        result = training.position_diagnostic(training.MultiEpisodeDemonstrations(episodes),
                                              policy, lambda value: value, lambda value: value, torch)
        self.assertEqual(result['normalized_chunk_mae'], 20.)


class PartitionTests(unittest.TestCase):
    def entries(self):
        return [
            {'id': 'train_a', 'split': 'train', 'offset_xy_m': [0., 0.], 'manifest': 'train/manifest.json'},
            {'id': 'validation_a', 'split': 'validation', 'offset_xy_m': [.01, .01], 'manifest': 'validation/manifest.json'},
            # This file deliberately does not exist; no test contents may be opened.
            {'id': 'test_a', 'split': 'test', 'offset_xy_m': [-.01, .01], 'manifest': 'absent_test/manifest.json'}]

    def read(self, entries):
        loaded = []

        def fake_loader(entry, path):
            if entry['split'] == 'test':
                raise AssertionError('Test episode content was opened')
            loaded.append(entry['id'])
            return FakeEpisode(11. if entry['split'] == 'train' else 22.,
                               episode_id=entry['id'], split=entry['split'], offset=entry['offset_xy_m'])

        manifest = {'status': 'verified_multisite_rgb_dataset', 'episodes': entries}
        with patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
                patch.object(training, 'MemmapEpisode', side_effect=fake_loader):
            result = training.read_partitions(training.ROOT / 'runs/synthetic_contract_manifest.json')
        return loaded, result

    def test_train_contains_no_validation_or_test_and_test_is_unopened(self):
        loaded, (_, summaries, partitions) = self.read(self.entries())
        self.assertEqual(loaded, ['train_a', 'validation_a'])
        self.assertEqual([episode.entry['id'] for episode in partitions['train'].episodes], ['train_a'])
        self.assertEqual([episode.entry['id'] for episode in partitions['validation'].episodes], ['validation_a'])
        self.assertNotIn('test', partitions)
        self.assertFalse(next(item for item in summaries if item['split'] == 'test')['contents_read'])
        for index in np.random.default_rng(4).choice(len(partitions['train']), 30):
            self.assertEqual(partitions['train'].locate(index)[0].entry['split'], 'train')

    def test_partial_export_rejected_before_episode_load(self):
        manifest = {'status': 'exporting', 'episodes': self.entries()}
        with patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
                patch.object(training, 'MemmapEpisode') as loader:
            with self.assertRaisesRegex(ValueError, 'partial exports are not usable'):
                training.read_partitions(training.ROOT / 'runs/synthetic_contract_manifest.json')
            loader.assert_not_called()

    def test_same_position_cannot_leak_across_splits(self):
        entries = self.entries()
        entries[1]['offset_xy_m'] = entries[0]['offset_xy_m']
        with self.assertRaisesRegex(ValueError, 'complete XY position'):
            self.read(entries)

    def test_same_manifest_cannot_be_reused(self):
        entries = self.entries()
        entries[1]['manifest'] = entries[0]['manifest']
        with self.assertRaisesRegex(ValueError, 'cannot occur more than once'):
            self.read(entries)

    def test_missing_independent_split_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonempty train, validation and test'):
            self.read(self.entries()[:-1])

    def test_path_escape_rejected(self):
        entries = self.entries()
        entries[0]['manifest'] = '../outside/manifest.json'
        with self.assertRaisesRegex(ValueError, 'relative and stay within'):
            self.read(entries)

    def test_duplicate_id_rejected(self):
        entries = self.entries()
        entries[1]['id'] = entries[0]['id']
        with self.assertRaisesRegex(ValueError, 'unique id'):
            self.read(entries)


if __name__ == '__main__':
    unittest.main()
