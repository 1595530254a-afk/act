"""Evaluate ACT action chunks without teacher forcing on fixed episode-grouped observations."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from cpu_budget import CPU_BUDGET, configure_torch

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.collate import lerobot_collate_fn
from short_training import file_hash, load_policy, write_json


def configure_runtime():
    configure_torch()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_dataset():
    manifest = json.loads((ROOT / 'resources/resource_manifest.json').read_text(encoding='utf-8'))
    return LeRobotDataset(
        manifest['dataset']['repo_id'], root=ROOT / 'data/rebot_cube',
        revision=manifest['dataset']['revision'], video_backend='pyav',
        delta_timestamps={'action': [index / 30 for index in range(100)]},
    )


def episode_split(dataset):
    permutation = np.random.default_rng(20261019).permutation(dataset.meta.total_episodes).tolist()
    return {'train': sorted(permutation[:40]), 'validation': sorted(permutation[40:45]),
            'assessment': sorted(permutation[45:])}


def episode_frames(dataset, episodes, frames_per_episode=10):
    indices = []
    for episode in episodes:
        row = dataset.meta.episodes[episode]
        indices.extend(np.linspace(row['dataset_from_index'], row['dataset_to_index'] - 1,
                                   frames_per_episode, dtype=int).tolist())
    return indices


def evaluate(policy, preprocess, postprocess, dataset, indices, batch_size=4):
    policy.eval()
    normalized_sum = np.zeros(7, dtype=np.float64)
    raw_sum = np.zeros(7, dtype=np.float64)
    first_sum = np.zeros(7, dtype=np.float64)
    delta_error_sum = np.zeros(7, dtype=np.float64)
    outside_count = np.zeros(7, dtype=np.int64)
    valid_count = 0
    delta_count = 0
    stats = dataset.meta.stats['action']
    action_min = np.asarray(stats['min']).reshape(7)
    action_max = np.asarray(stats['max']).reshape(7)
    rows = []
    traces = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset:offset + batch_size]
            raw = lerobot_collate_fn([dataset[index] for index in selected])
            target_raw = raw['action'].clone().cpu().numpy()
            valid = ~raw['action_is_pad'].clone().cpu().numpy()
            episode_ids = raw['episode_index'].tolist()
            processed = preprocess(raw)
            # Future demonstration actions are labels only, excluded from the model inputs.
            observations = {key: processed[key] for key in policy.config.input_features}
            prediction_normalized = policy.predict_action_chunk(observations)
            prediction_raw = postprocess(prediction_normalized.clone()).cpu().numpy()
            prediction_normalized = prediction_normalized.cpu().numpy()
            target_normalized = processed['action'].cpu().numpy()
            if prediction_raw.shape != target_raw.shape or not np.isfinite(prediction_raw).all():
                raise RuntimeError('Invalid evaluation predictions.')
            normalized_error = np.abs(prediction_normalized - target_normalized)
            raw_error = np.abs(prediction_raw - target_raw)
            normalized_sum += (normalized_error * valid[..., None]).sum(axis=(0, 1))
            raw_sum += (raw_error * valid[..., None]).sum(axis=(0, 1))
            first_sum += raw_error[:, 0].sum(axis=0)
            valid_count += int(valid.sum())
            outside = (prediction_raw < action_min) | (prediction_raw > action_max)
            outside_count += (outside * valid[..., None]).sum(axis=(0, 1))
            pair_valid = valid[:, 1:] & valid[:, :-1]
            delta_error = np.abs(np.diff(prediction_normalized, axis=1) - np.diff(target_normalized, axis=1))
            delta_error_sum += (delta_error * pair_valid[..., None]).sum(axis=(0, 1))
            delta_count += int(pair_valid.sum())
            for position, frame_index in enumerate(selected):
                count = int(valid[position].sum())
                rows.append({
                    'frame_index': frame_index, 'episode_index': int(episode_ids[position]),
                    'valid_target_steps': count,
                    'normalized_chunk_mae': float(normalized_error[position][valid[position]].mean()),
                    'raw_chunk_mae': float(raw_error[position][valid[position]].mean()),
                    'raw_first_action_mae': float(raw_error[position, 0].mean()),
                })
                if len(traces) < 5 and count == 100:
                    traces.append({'frame_index': frame_index, 'prediction': prediction_raw[position].tolist(),
                                   'target': target_raw[position].tolist()})
    episode_values = {}
    for row in rows:
        episode_values.setdefault(row['episode_index'], []).append(row['normalized_chunk_mae'])
    summary = {
        'sampled_observations': len(indices), 'episodes': len(episode_values),
        'valid_target_steps': valid_count, 'normalized_chunk_mae': float((normalized_sum / valid_count).mean()),
        'normalized_chunk_mae_by_channel': (normalized_sum / valid_count).tolist(),
        'episode_macro_normalized_mae': float(np.mean([np.mean(values) for values in episode_values.values()])),
        'raw_chunk_mae_by_channel': (raw_sum / valid_count).tolist(),
        'raw_first_action_mae_by_channel': (first_sum / len(indices)).tolist(),
        'normalized_action_delta_error_by_channel': (delta_error_sum / max(delta_count, 1)).tolist(),
        'outside_observed_dataset_range_fraction_by_channel': (outside_count / valid_count).tolist(),
        'wall_s': time.perf_counter() - started, 'all_predictions_finite': True,
    }
    return summary, rows, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', required=True, help='label=local_model_directory')
    parser.add_argument('--report-name', default='extended_offline_evaluation')
    args = parser.parse_args()
    if args.report_name in ['', '.', '..'] or any(char in args.report_name for char in '/\\:'):
        parser.error('report-name must be a simple filename')
    configure_runtime()
    dataset = load_dataset()
    split = episode_split(dataset)
    indices = episode_frames(dataset, list(range(dataset.meta.total_episodes)))
    result = {'status': 'passed', 'scope': 'offline_demonstration_diagnostics_not_robot_success',
              'cpu_budget': CPU_BUDGET,
              'episode_split': split, 'frames_per_episode': 10, 'model_inputs_exclude_actions': True,
              'limitations': ['All episodes were available to the published pretrained model.',
                             'The assessment partition excludes only this project controlled fine-tuning and selection.',
                             'Dataset ranges and units are not verified physical robot limits.',
                             'No closed-loop rollout, camera or robot connection is included.'], 'models': {}}
    artifact_root = ROOT / 'runs' / args.report_name
    artifact_root.mkdir(parents=True, exist_ok=True)
    for specification in args.model:
        label, separator, directory = specification.partition('=')
        if not separator or label in ['', '.', '..'] or any(char in label for char in '/\\:'):
            parser.error('each model must be label=directory, with a simple label')
        model_dir = Path(directory).resolve()
        policy, preprocess, postprocess = load_policy(model_dir)
        summary, rows, traces = evaluate(policy, preprocess, postprocess, dataset, indices)
        for group, episodes in split.items():
            values = [row['normalized_chunk_mae'] for row in rows if row['episode_index'] in episodes]
            summary[f'{group}_sample_macro_normalized_mae'] = float(np.mean(values))
        summary['model_directory'] = str(model_dir)
        summary['model_sha256'] = file_hash(model_dir / 'model.safetensors')
        result['models'][label] = summary
        with (artifact_root / f'{label}_frames.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(artifact_root / f'{label}_traces.json', traces)
        print(f'{label}: normalized chunk MAE={summary["normalized_chunk_mae"]:.6f}; '
              f'500 observations, {summary["wall_s"]:.1f}s', flush=True)
        del policy, preprocess, postprocess
        torch.cuda.empty_cache()
    result['artifacts_directory'] = str(artifact_root.relative_to(ROOT))
    write_json(ROOT / 'reports' / f'{args.report_name}.json', result)


if __name__ == '__main__':
    main()
