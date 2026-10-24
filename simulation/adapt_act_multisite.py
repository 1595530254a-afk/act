"""ACT adaptation with whole-position train/validation/test separation.

Test episode contents are never opened here. RGB arrays are read-only NumPy
memmaps, and future-action chunks are padded within their own episode. The
validation metric selects weights; it is not a physical grasp success claim.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET, configure_torch

# This import sets the existing offline/cache policy before NumPy or Torch.
from adapt_act_sim import (CHUNK_SIZE, MODEL_FILES, compute_gradients, digest,
                           forward_preflight, future_actions, load_policy,
                           save_checkpoint, save_json)
import numpy as np


def child_path(directory, name):
    relative = Path(name)
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory):
        raise ValueError('Dataset paths must be relative and stay within their parent directory')
    return path


def check_scene(scene):
    if (scene.get('fixed_project_brick_dimensions_m') != [.040, .040, .020]
            or scene.get('stud_height_m') != 0.0 or scene.get('flat_top_brick') is not True
            or scene.get('fixed_project_bin_outside_dimensions_m') != [.120, .120, .050]
            or scene.get('fixed_project_bin_wall_m') != .004):
        raise ValueError('Requires the fixed flat 40x40x20 mm block and 120x120x50 mm bin with 4 mm walls')


class MemmapEpisode:
    def __init__(self, entry, manifest_path):
        self.entry, self.manifest_path = entry, manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (self.manifest.get('status') != 'verified_multisite_episode_rgb_export'
                or self.manifest.get('fps') != 30
                or self.manifest.get('task_metrics', {}).get('success') is not True
                or self.manifest.get('hardware_io') is not False
                or self.manifest.get('post_initialization_object_pose_assignment') is not False
                or self.manifest.get('grasp_weld_or_attachment') is not False
                or self.manifest.get('wrist_camera_self_housing_group_4_hidden') is not True):
            raise ValueError('Requires a successful 30 Hz simulation RGB export without hardware, attachment or teleportation')
        check_scene(self.manifest.get('scene', {}))
        self.file_hashes = {'manifest.json': digest(manifest_path)}
        if (entry.get('manifest_sha256') is not None
                and entry['manifest_sha256'] != self.file_hashes['manifest.json']):
            raise ValueError('Episode manifest no longer matches the frozen dataset list')
        trajectory = self.checked_file(self.manifest['trajectory'])
        with np.load(trajectory, allow_pickle=False) as data:
            self.state, self.action = data['state'].copy(), data['action'].copy()
            timestamps = data['timestamp_before'].copy()
        count = self.manifest['frames']
        if (not isinstance(count, int) or count < 120
                or self.state.shape != (count, 7) or self.action.shape != (count, 7)
                or self.state.dtype != np.float32 or self.action.dtype != np.float32
                or not np.isfinite(self.state).all() or not np.isfinite(self.action).all()
                or timestamps.shape != (count,) or not np.isfinite(timestamps).all()
                or not np.allclose(timestamps, np.arange(count) / 30, atol=.0011, rtol=0)):
            raise ValueError('Invalid public float32 state/action or 30 Hz episode timestamp contract')
        self.rgb = {}
        for role in ('wrist', 'top'):
            path = self.checked_file(self.manifest['images'][role])
            value = np.load(path, allow_pickle=False, mmap_mode='r')
            if value.dtype != np.uint8 or value.shape != (count, 3, 480, 640):
                raise ValueError('RGB memmaps must be uint8 N-by-3-by-480-by-640')
            sample = value[::max(1, count // 5), :, ::32, ::32]
            if float(sample.std()) < 2.:
                raise ValueError(f'{role} RGB appears blank or nearly constant')
            self.rgb[role] = value

    def checked_file(self, descriptor):
        path = child_path(self.manifest_path.parent, descriptor['file'])
        actual = digest(path)
        if actual != descriptor['sha256']:
            raise ValueError(f'Dataset hash mismatch: {path}')
        self.file_hashes[descriptor['file']] = actual
        return path

    def __len__(self):
        return len(self.state)

    def images(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return {role: value[index] for role, value in self.rgb.items()}


class MultiEpisodeDemonstrations:
    def __init__(self, episodes):
        if not episodes:
            raise ValueError('Expected at least one episode in this split')
        self.episodes = episodes
        self.offsets = np.cumsum([0] + [len(episode) for episode in episodes])

    def __len__(self):
        return int(self.offsets[-1])

    def locate(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.offsets, index, side='right') - 1)
        return self.episodes[episode_index], int(index - self.offsets[episode_index])

    def batch(self, indices, torch):
        samples = [self.locate(index) for index in indices]
        images = [episode.images(index) for episode, index in samples]
        labels = [future_actions(episode.action, index) for episode, index in samples]
        result = {
            'observation.state': torch.from_numpy(np.stack([episode.state[index] for episode, index in samples])),
            'action': torch.from_numpy(np.stack([label[0] for label in labels])),
            'action_is_pad': torch.from_numpy(np.stack([label[1] for label in labels]))}
        for role in ('wrist', 'top'):
            result['observation.images.' + role] = torch.from_numpy(
                np.stack([sample[role] for sample in images])).float().div_(255)
        return result


def read_partitions(manifest_path):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'verified_multisite_rgb_dataset':
        raise ValueError('Requires a completed verified_multisite_rgb_dataset; partial exports are not usable')
    entries = manifest.get('episodes')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Expected a nonempty episodes list in the multisite manifest')
    seen_ids, seen_paths, position_splits = set(), set(), {}
    summaries, split_entries = [], {split: [] for split in ('train', 'validation', 'test')}
    for entry in entries:
        episode_id, split = entry.get('id'), entry.get('split')
        offset = np.asarray(entry.get('offset_xy_m'), dtype=np.float64)
        if (not isinstance(episode_id, str) or not episode_id or episode_id in seen_ids
                or split not in split_entries or offset.shape != (2,) or not np.isfinite(offset).all()):
            raise ValueError('Each episode needs a unique id, declared split and finite XY offset')
        path = child_path(manifest_path.parent, entry['manifest'])
        if path in seen_paths:
            raise ValueError('An episode manifest cannot occur more than once')
        position = tuple(np.round(offset, decimals=9))
        if position in position_splits and position_splits[position] != split:
            raise ValueError('A complete XY position must belong to only one split')
        seen_ids.add(episode_id)
        seen_paths.add(path)
        position_splits[position] = split
        split_entries[split].append((entry, path))
        summaries.append({'id': episode_id, 'split': split, 'offset_xy_m': offset.tolist(),
                          'manifest': str(path), 'contents_read': split != 'test'})
    if any(not split_entries[split] for split in ('train', 'validation', 'test')):
        raise ValueError('Requires separate, nonempty train, validation and test position sets')
    # In particular, do not open even the per-episode manifest for a test item.
    active = {split: MultiEpisodeDemonstrations([MemmapEpisode(entry, path)
              for entry, path in split_entries[split]]) for split in ('train', 'validation')}
    return manifest, summaries, active


def position_diagnostic(dataset, policy, preprocess, postprocess, torch, anchors_per_episode=5):
    """Inference receives observations only; action labels are compared afterwards."""
    policy.eval()
    per_episode = []
    with torch.inference_mode():
        for episode_index, episode in enumerate(dataset.episodes):
            indices = np.unique(np.linspace(0, len(episode) - 1, anchors_per_episode, dtype=int))
            normalized_sum, public_sum, count = 0., 0., 0
            per_joint = np.zeros(7, dtype=np.float64)
            for index in indices:
                batch = preprocess(dataset.batch(np.asarray([dataset.offsets[episode_index] + index]), torch))
                # Never pass action/action_is_pad into predict_action_chunk. This
                # uses the policy prior at inference, not its action-conditioned encoder.
                observation = {key: batch[key] for key in policy.config.input_features}
                prediction = policy.predict_action_chunk(observation)
                if prediction.shape != (1, CHUNK_SIZE, 7) or not torch.isfinite(prediction).all():
                    raise FloatingPointError('Invalid inference chunk during position validation')
                mask = ~batch['action_is_pad']
                normalized_sum += float((prediction - batch['action']).abs()[mask].sum().item())
                target, pad = future_actions(episode.action, int(index))
                public_prediction = postprocess(prediction).float().cpu().numpy()[0]
                error = np.abs(public_prediction[~pad] - target[~pad])
                public_sum += float(error.sum())
                per_joint += error.sum(axis=0)
                count += int((~pad).sum())
            per_episode.append({
                'id': episode.entry['id'], 'split': episode.entry['split'],
                'offset_xy_m': episode.entry['offset_xy_m'],
                'normalized_chunk_mae': normalized_sum / (count * 7),
                'public_degree_chunk_mae': public_sum / (count * 7),
                'per_joint_public_degree_mae': (per_joint / count).tolist(),
                'observation_anchor_indices': indices.tolist(), 'valid_target_rows': count})
    result = {
        'normalized_chunk_mae': float(np.mean([item['normalized_chunk_mae'] for item in per_episode])),
        'public_degree_chunk_mae': float(np.mean([item['public_degree_chunk_mae'] for item in per_episode])),
        'aggregation': 'equal-weight mean over complete episodes; padding excluded',
        'per_episode': per_episode, 'inference_has_action_labels': False,
        'physical_grasp_success_evaluated': False,
        'scope': 'Offline action prediction error, not a physical rollout or task-success measurement.'}
    if not np.isfinite(result['normalized_chunk_mae']):
        raise FloatingPointError('Non-finite validation metric')
    return result


def train(args, summaries, partitions):
    model_dir, output = args.model_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; earlier experiments are never replaced')
    if output == model_dir or output.is_relative_to(model_dir):
        raise ValueError('Training output must be outside the source checkpoint')
    source_hashes = {name: digest(model_dir / name) for name in MODEL_FILES}
    torch, policy, preprocess, postprocess = load_policy(model_dir)
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32
           for parameter in policy.parameters()):
        raise ValueError('Multisite adaptation requires float32 model parameters')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    training, validation = partitions['train'], partitions['validation']
    familiar_episodes = [episode for episode in training.episodes
                         if np.allclose(episode.entry['offset_xy_m'], [0., 0.], atol=1e-9, rtol=0)]
    familiar = MultiEpisodeDemonstrations(familiar_episodes) if familiar_episodes else None
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'multisite_simulation_imitation_adaptation', 'steps_requested': args.steps,
        'completed_steps': 0, 'batch_size': 2, 'learning_rate': args.learning_rate,
        'optimizer': {'type': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8,
                      'weight_decay': 1e-4, 'state_restored': False},
        'gradient_clip_norm': 10., 'seed': args.seed, 'precision': 'float32',
        'prior_adaptation_steps': args.prior_adaptation_steps,
        'training_script_sha256': digest(Path(__file__).resolve()),
        'shared_training_script_sha256': digest(ROOT / 'simulation/adapt_act_sim.py'),
        'source_model': str(model_dir), 'source_files_sha256': source_hashes,
        'dataset_manifest': str(args.data_manifest.resolve()),
        'dataset_manifest_sha256': digest(args.data_manifest),
        'split_episodes': summaries,
        'dataset_files_sha256': {split: {episode.entry['id']: episode.file_hashes
                                  for episode in dataset.episodes} for split, dataset in partitions.items()},
        'frames': {split: len(dataset) for split, dataset in partitions.items()},
        'sampling': 'Uniform over training frames; future actions remain within their original episode',
        'normalization': 'Original saved model processors and statistics; no refitting',
        'validation_anchors_per_episode': args.validation_anchors,
        'selection_metric': 'validation.normalized_chunk_mae; source weights also eligible',
        'selection_uses_test': False, 'test_episode_contents_opened': 0,
        'cpu_budget': CPU_BUDGET, 'gpu': torch.cuda.get_device_name(),
        'hardware_io': False, 'network_access': False, 'real_robot_qualified': False,
        'physical_grasp_success_evaluated': False, 'checkpoints': [], 'oom_batch_reductions': []}
    save_json(output / 'training_report.json', report)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        report['forward_preflight'] = forward_preflight(training, policy, preprocess, torch, False)
        report['source_validation'] = position_diagnostic(
            validation, policy, preprocess, postprocess, torch, args.validation_anchors)
        if familiar is not None:
            report['source_familiar_training_position'] = position_diagnostic(
                familiar, policy, preprocess, postprocess, torch, args.validation_anchors)
        report['selected_model'] = {'step': 0, 'model_dir': str(model_dir),
            'model_sha256': source_hashes['model.safetensors'],
            'validation_normalized_chunk_mae': report['source_validation']['normalized_chunk_mae'],
            'reason': 'Source baseline remains selected until a checkpoint improves validation'}
        save_json(output / 'training_report.json', report)
        print(json.dumps({'status': 'preflight_and_source_validation_passed',
            'source_validation': report['source_validation']}), flush=True)
        actual_batch_size = 2
        fields = ['step', 'loss', 'l1_loss', 'kld_loss', 'grad_norm_before_clip',
                  'update_wall_s', 'cuda_allocated_mib', 'cuda_peak_allocated_mib',
                  'valid_target_fraction', 'batch_size']
        with (output / 'training_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(1, args.steps + 1):
                update_started = time.perf_counter()
                indices = rng.choice(len(training), size=actual_batch_size, replace=False)
                while True:
                    try:
                        gradients = compute_gradients(training, indices, policy, preprocess,
                                                      optimizer, torch, False)
                        break
                    except torch.cuda.OutOfMemoryError:
                        if actual_batch_size == 1:
                            raise
                        optimizer.zero_grad(set_to_none=True)
                        actual_batch_size, indices = 1, indices[:1]
                        report['oom_batch_reductions'].append({'step': step, 'from': 2, 'to': 1,
                                                               'optimizer_step_started': False})
                    torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.synchronize()
                report['completed_steps'] = step
                row = {'step': step, **gradients, 'batch_size': actual_batch_size,
                       'update_wall_s': time.perf_counter() - update_started,
                       'cuda_allocated_mib': torch.cuda.memory_allocated() / 1024 ** 2,
                       'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2}
                writer.writerow(row)
                stream.flush()
                if step == 1 or step % 10 == 0:
                    print(json.dumps(row), flush=True)
                if step % args.save_every == 0 or step == args.steps:
                    optimizer.zero_grad(set_to_none=True)
                    diagnostic = position_diagnostic(validation, policy, preprocess, postprocess,
                                                     torch, args.validation_anchors)
                    checkpoint = save_checkpoint(output, step, policy, model_dir, source_hashes,
                                                 diagnostic, args.prior_adaptation_steps)
                    if familiar is not None:
                        checkpoint['familiar_training_position'] = position_diagnostic(
                            familiar, policy, preprocess, postprocess, torch, args.validation_anchors)
                    report['checkpoints'].append(checkpoint)
                    metric = diagnostic['normalized_chunk_mae']
                    if metric < report['selected_model']['validation_normalized_chunk_mae']:
                        report['selected_model'] = {key: checkpoint[key]
                                                   for key in ('step', 'model_dir', 'model_sha256')}
                        report['selected_model'].update({'validation_normalized_chunk_mae': metric,
                            'reason': 'Lowest complete-validation-episode normalized chunk MAE'})
                    report['elapsed_wall_s'] = time.perf_counter() - started
                    save_json(output / 'training_report.json', report)
                    print(json.dumps({'status': 'checkpoint_saved', 'step': step,
                                      'selected_model': report['selected_model'],
                                      'validation': diagnostic}), flush=True)
        if any(digest(model_dir / name) != value for name, value in source_hashes.items()):
            raise RuntimeError('Original source model files changed during adaptation')
        report.update({'status': 'completed_multisite_adaptation_not_policy_success',
                       'original_source_files_unchanged': True,
                       'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
                       'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2})
    except Exception as error:
        report.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        report['elapsed_wall_s'] = time.perf_counter() - started
        save_json(output / 'training_report.json', report)
    print(json.dumps({'status': report['status'], 'completed_steps': report['completed_steps'],
        'selected_model': report['selected_model'], 'report': str(output / 'training_report.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-manifest', required=True, type=Path)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/act_rebot_cube')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--validation-anchors', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--prior-adaptation-steps', type=int, default=0,
                        help='Metadata only; optimizer state is not resumed')
    parser.add_argument('--check-data', action='store_true',
                        help='Check train/validation memmaps and batches without loading a model; never open test contents')
    args = parser.parse_args()
    if not 1 <= args.steps <= 2400 or not 1 <= args.save_every <= 2400:
        parser.error('steps and save-every must be within 1..2400')
    if not 0 < args.learning_rate <= 1e-5 or not 4 <= args.validation_anchors <= 6:
        parser.error('learning-rate must be positive and <=1e-5; validation-anchors must be within 4..6')
    if args.prior_adaptation_steps < 0 or not 0 <= args.seed < 2 ** 32:
        parser.error('prior-adaptation-steps must be nonnegative and seed a uint32')
    if not args.check_data and args.output_dir is None:
        parser.error('output-dir is required for training')
    _, summaries, partitions = read_partitions(args.data_manifest)
    if args.check_data:
        configure_torch()
        import torch
        checks = {}
        for split, dataset in partitions.items():
            for episode_index, episode in enumerate(dataset.episodes):
                last = int(dataset.offsets[episode_index + 1] - 1)
                batch = dataset.batch(np.asarray([last]), torch)
                assert batch['action_is_pad'][0].tolist() == [False] + [True] * 99
                assert torch.equal(batch['action'][0], torch.from_numpy(episode.action[-1]).expand(CHUNK_SIZE, -1))
            checks[split] = {'episodes': len(dataset.episodes), 'frames': len(dataset)}
        print(json.dumps({'status': 'whole_episode_split_and_batch_contract_passed', 'splits': checks,
            'split_episodes': summaries, 'test_episode_contents_opened': 0,
            'training_steps_executed': 0, 'model_loaded': False, 'hardware_io': False}), flush=True)
        return
    train(args, summaries, partitions)


if __name__ == '__main__':
    main()
