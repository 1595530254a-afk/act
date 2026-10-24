"""Audit the pinned ACT tabular data, episode boundaries and future-action padding."""

import json

from cpu_budget import CPU_BUDGET

import numpy as np
import torch

from offline_evaluation import ROOT, configure_runtime, episode_split, load_dataset
from short_training import write_json


def main():
    configure_runtime()
    dataset = load_dataset()
    columns = ['index', 'episode_index', 'timestamp', 'action', 'observation.state']
    table = dataset.hf_dataset.select_columns(columns).with_format('numpy')[:]
    count = len(dataset)
    actions = np.asarray(table['action'])
    states = np.asarray(table['observation.state'])
    timestamps = np.asarray(table['timestamp'])
    indices = np.asarray(table['index'])
    episode_ids = np.asarray(table['episode_index'])
    if actions.shape != (count, 7) or states.shape != (count, 7):
        raise RuntimeError('Invalid action/state dimensions.')
    if not np.isfinite(actions).all() or not np.isfinite(states).all() or not np.isfinite(timestamps).all():
        raise RuntimeError('Non-finite tabular values.')
    if not np.array_equal(indices, np.arange(count)):
        raise RuntimeError('Global frame indices are not contiguous.')
    episode_lengths = []
    max_timestamp_step_error = 0.0
    previous_end = 0
    for episode in range(dataset.meta.total_episodes):
        metadata = dataset.meta.episodes[episode]
        start, end = metadata['dataset_from_index'], metadata['dataset_to_index']
        if start != previous_end or end - start != metadata['length']:
            raise RuntimeError(f'Broken episode offsets: {episode}')
        if not np.all(episode_ids[start:end] == episode):
            raise RuntimeError(f'Mixed episode labels: {episode}')
        if not np.all(np.diff(timestamps[start:end]) > 0):
            raise RuntimeError(f'Non-increasing timestamps: {episode}')
        max_timestamp_step_error = max(max_timestamp_step_error,
                                       float(np.max(np.abs(np.diff(timestamps[start:end]) - 1/30))))
        sample = dataset[end - 1]
        torch.testing.assert_close(sample['action_is_pad'], torch.arange(100) > 0)
        torch.testing.assert_close(sample['action'], torch.from_numpy(actions[end - 1]).expand(100, -1))
        episode_lengths.append(end - start)
        previous_end = end
    if previous_end != count or max_timestamp_step_error > 1e-4:
        raise RuntimeError('Dataset coverage or timestamp tolerance failed.')
    split = episode_split(dataset)
    report = {
        'status': 'passed', 'frames': count, 'episodes': dataset.meta.total_episodes,
        'cpu_budget': CPU_BUDGET,
        'finite_action_state_timestamp_values': True, 'global_frame_indices_contiguous': True,
        'episode_offsets_cover_all_frames': True, 'all_episode_end_padding_checks': len(episode_lengths),
        'episode_length_min': min(episode_lengths), 'episode_length_max': max(episode_lengths),
        'episode_length_mean': float(np.mean(episode_lengths)),
        'maximum_timestamp_step_error_s': max_timestamp_step_error,
        'action_min': actions.min(axis=0).tolist(), 'action_max': actions.max(axis=0).tolist(),
        'state_min': states.min(axis=0).tolist(), 'state_max': states.max(axis=0).tolist(),
        'episode_split': split,
        'partition_frame_counts': {key: int(sum(episode_lengths[episode] for episode in values))
                                   for key, values in split.items()},
        'scope': 'Tabular integrity and all episode boundaries; not demonstration task quality or real-robot validation',
        'normalization_and_assessment_caveat': 'Published model and its normalization already used the whole source dataset.',
    }
    write_json(ROOT / 'reports/dataset_audit.json', report)
    print(json.dumps({key: report[key] for key in ['status', 'frames', 'episodes',
                                                 'all_episode_end_padding_checks', 'partition_frame_counts']}, indent=2))


if __name__ == '__main__':
    main()
