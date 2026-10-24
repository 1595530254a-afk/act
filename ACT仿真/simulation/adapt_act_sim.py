"""Bounded ACT adaptation to a verified single simulated demonstration.

This is scene-specific imitation, never a hardware policy qualification or an
independent generalization benchmark. Original weights/statistics stay intact.
Images are loaded through a two-shard cache; no video decoding or network is used.
"""
import argparse
from collections import OrderedDict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET, configure_torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache/torch'))

import numpy as np

CHUNK_SIZE = 100
MODEL_FILES = ('config.json', 'model.safetensors', 'policy_preprocessor.json',
               'policy_postprocessor.json',
               'policy_preprocessor_step_3_normalizer_processor.safetensors',
               'policy_postprocessor_step_0_unnormalizer_processor.safetensors')
PROCESSOR_FILES = MODEL_FILES[2:]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def future_actions(actions, index, chunk_size=CHUNK_SIZE):
    """Start at index, repeat the last action past episode end, flag padding."""
    if (not isinstance(actions, np.ndarray) or actions.ndim != 2
            or actions.shape[1] != 7 or len(actions) == 0
            or actions.dtype.kind not in 'fiu' or not np.isfinite(actions).all()
            or not isinstance(index, (int, np.integer)) or not 0 <= index < len(actions)
            or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError('Expected nonempty N-by-7 actions, a valid start and positive chunk size')
    requested = index + np.arange(chunk_size)
    return actions[np.minimum(requested, len(actions) - 1)].copy(), requested >= len(actions)


class SimDemonstration:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if (self.manifest.get('status') != 'verified_single_episode_simulation_teacher_export'
                or not self.manifest.get('task_metrics', {}).get('success')
                or self.manifest.get('episodes') != 1 or self.manifest.get('fps') != 30):
            raise ValueError('Requires a completed, successful, single-episode 30 Hz teacher export')
        scene = self.manifest.get('scene', {})
        if (scene.get('fixed_project_brick_dimensions_m') != [.040, .040, .020]
                or scene.get('stud_height_m') != 0.0 or scene.get('flat_top_brick') is not True
                or scene.get('fixed_project_bin_outside_dimensions_m') != [.120, .120, .050]
                or scene.get('fixed_project_bin_wall_m') != .004):
            raise ValueError('Only the approved flat 40x40x20 mm block and 120x120x50 mm bin with 4 mm walls are accepted')
        if (self.manifest.get('hardware_io') is not False
                or self.manifest.get('post_initialization_object_pose_assignment') is not False
                or self.manifest.get('grasp_weld_or_attachment') is not False
                or self.manifest.get('wrist_camera_self_housing_group_4_hidden') is not True):
            raise ValueError('Requires a simulation-only export without object teleportation/attachment or wrist self-occlusion')
        self.shards = self.manifest['shards']
        if not isinstance(self.shards, list) or not self.shards:
            raise ValueError('Expected at least one teacher image shard')
        self.cache = OrderedDict()
        self.cache_hits = self.cache_misses = 0
        states, actions, offsets = [], [], [0]
        for shard in self.shards:
            path = (self.manifest_path.parent / shard['file']).resolve()
            if not path.is_relative_to(self.manifest_path.parent):
                raise ValueError('Shard path must stay within the export directory')
            if digest(path) != shard['sha256']:
                raise ValueError(f'Shard hash mismatch: {path.name}')
            with np.load(path, allow_pickle=False) as data:
                state, action = data['state'].copy(), data['action'].copy()
                count = len(state)
                if (state.shape != (count, 7) or action.shape != (count, 7)
                        or state.dtype != np.float32 or action.dtype != np.float32
                        or not np.isfinite(state).all() or not np.isfinite(action).all()
                        or not np.array_equal(data['frame_index'], np.arange(offsets[-1], offsets[-1] + count))):
                    raise ValueError(f'Invalid public state/action/frame contract: {path.name}')
                if count != shard['frames'] or not 1 <= count <= 32:
                    raise ValueError('Expected at most 32 images per shard')
                timestamps = data['timestamp']
                expected = np.arange(offsets[-1], offsets[-1] + count) / 30
                if (timestamps.shape != (count,) or not np.isfinite(timestamps).all()
                        or not np.allclose(timestamps, expected, atol=.0011, rtol=0)):
                    raise ValueError('Teacher timestamps do not match the declared 30 Hz observation order')
                states.append(state)
                actions.append(action)
                offsets.append(offsets[-1] + count)
        self.state = np.concatenate(states)
        self.action = np.concatenate(actions)
        self.offsets = np.asarray(offsets)
        if len(self.state) != self.manifest['frames'] or len(self.state) < 120:
            raise ValueError('Manifest frame count mismatch or insufficient demonstration')

    def __len__(self):
        return len(self.state)

    def images(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = int(np.searchsorted(self.offsets, index, side='right') - 1)
        if shard_index not in self.cache:
            self.cache_misses += 1
            # Evict before decompressing the next shard, keeping two decoded
            # camera shards resident rather than three during replacement.
            if len(self.cache) == 2:
                self.cache.popitem(last=False)
            shard = self.shards[shard_index]
            with np.load(self.manifest_path.parent / shard['file'], allow_pickle=False) as source:
                decoded = {role: source[role].copy() for role in ('wrist', 'top')}
            for role, value in decoded.items():
                if value.dtype != np.uint8 or value.shape != (shard['frames'], 3, 480, 640):
                    raise ValueError(f'{role} images must be RGB uint8 N-by-3-by-480-by-640')
                sample = value[::8, :, ::16, ::16]
                if float(sample.std()) < 2.0:
                    raise ValueError(f'{role} images appear blank or nearly constant')
            self.cache[shard_index] = decoded
        else:
            self.cache_hits += 1
            self.cache.move_to_end(shard_index)
        local = index - int(self.offsets[shard_index])
        return {role: value[local] for role, value in self.cache[shard_index].items()}

    def batch(self, indices, torch):
        images = [self.images(int(index)) for index in indices]
        labels = [future_actions(self.action, int(index)) for index in indices]
        result = {'observation.state': torch.from_numpy(self.state[indices].copy()),
                  'action': torch.from_numpy(np.stack([label[0] for label in labels])),
                  'action_is_pad': torch.from_numpy(np.stack([label[1] for label in labels]))}
        for role in ('wrist', 'top'):
            result['observation.images.' + role] = torch.from_numpy(
                np.stack([item[role] for item in images])).float().div_(255)
        return result


def load_policy(directory):
    configure_torch()
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    if not torch.cuda.is_available():
        raise RuntimeError('This bounded adaptation requires CUDA; no CPU training fallback')
    config = ACTConfig.from_pretrained(directory, local_files_only=True)
    config.device, config.pretrained_backbone_weights, config.push_to_hub = 'cuda', None, False
    config.repo_id = None
    if (config.chunk_size != CHUNK_SIZE or list(config.input_features['observation.state'].shape) != [7]
            or set(config.image_features) != {'observation.images.wrist', 'observation.images.top'}
            or any(list(feature.shape) != [3, 480, 640] for feature in config.image_features.values())
            or list(config.output_features['action'].shape) != [7]):
        raise ValueError('Source checkpoint does not match the fixed seven-channel two-camera ACT contract')
    policy = ACTPolicy.from_pretrained(directory, config=config, local_files_only=True, strict=True).to('cuda')
    preprocess, postprocess = make_pre_post_processors(
        config, pretrained_path=str(directory),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    return torch, policy, preprocess, postprocess


def temporal_diagnostic(dataset, policy, preprocess, postprocess, indices, torch):
    """Inference has observations only; future labels are compared afterwards."""
    policy.eval()
    normalized_sum, public_sum, count = 0., 0., 0
    per_joint = np.zeros(7, dtype=np.float64)
    with torch.inference_mode():
        for index in indices:
            batch = preprocess(dataset.batch(np.asarray([index]), torch))
            observation = {key: batch[key] for key in policy.config.input_features}
            prediction = policy.predict_action_chunk(observation)
            if prediction.shape != (1, CHUNK_SIZE, 7) or not torch.isfinite(prediction).all():
                raise FloatingPointError('Invalid action chunk in inference-only diagnostic')
            mask = ~batch['action_is_pad']
            normalized = (prediction - batch['action']).abs()[mask]
            public_prediction = postprocess(prediction).float().cpu().numpy()[0]
            target, pad = future_actions(dataset.action, int(index))
            error = np.abs(public_prediction[~pad] - target[~pad])
            normalized_sum += float(normalized.sum().item())
            public_sum += float(error.sum())
            per_joint += error.sum(axis=0)
            count += int((~pad).sum())
    return {'normalized_chunk_mae': normalized_sum / (count * 7),
            'public_degree_chunk_mae': public_sum / (count * 7),
            'per_joint_public_degree_mae': (per_joint / count).tolist(),
            'observation_anchor_indices': indices.tolist(), 'valid_target_rows': count,
            'scope': 'Held-out observation anchors from one teacher episode; future targets overlap training windows. Not independent generalization or grasp success.'}


def save_checkpoint(output, step, policy, source_dir, source_hashes, diagnostic, prior_steps=0):
    checkpoint = output / 'checkpoints' / f'step_{step:06d}'
    checkpoint.mkdir(parents=True, exist_ok=False)
    model_dir = checkpoint / 'pretrained_model'
    policy.save_pretrained(model_dir)
    # This experiment does not refit processors. Re-saving the runtime pipelines
    # squeezes singleton statistics in this LeRobot version, changing their file
    # hashes despite equivalent values. Preserve the original saved contract.
    for name in PROCESSOR_FILES:
        shutil.copyfile(source_dir / name, model_dir / name)
    hashes = {name: digest(model_dir / name) for name in MODEL_FILES}
    for name in PROCESSOR_FILES:
        if hashes[name] != source_hashes[name]:
            raise RuntimeError('Saved original normalization statistics changed')
    save_json(checkpoint / 'checkpoint_report.json', {
        'status': 'saved', 'training_step': step, 'total_adaptation_steps': prior_steps + step,
        'model_dir': str(model_dir),
        'files_sha256': hashes, 'temporal_diagnostic': diagnostic,
        'original_normalization_statistics_preserved': True,
        'hardware_io': False, 'real_robot_qualified': False})
    return {'step': step, 'total_adaptation_steps': prior_steps + step,
            'model_dir': str(model_dir), 'model_sha256': hashes['model.safetensors'],
            'temporal_diagnostic': diagnostic}


def compute_gradients(dataset, indices, policy, preprocess, optimizer, torch, use_bf16):
    """No optimizer mutation here, so an OOM can retry a smaller minibatch."""
    batch = preprocess(dataset.batch(indices, torch))
    if batch['action'].shape != (len(indices), CHUNK_SIZE, 7) or batch['action_is_pad'].dtype != torch.bool:
        raise ValueError('Invalid action chunk or padding mask after preprocessing')
    for key in (*policy.config.input_features, 'action'):
        if not torch.isfinite(batch[key]).all():
            raise FloatingPointError(f'Non-finite normalized input: {key}')
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    precision_context = torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else nullcontext()
    with precision_context:
        loss, details = policy.forward(batch)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite adaptation loss')
    if details.get('kld_loss', 0.) < -1e-5:
        raise FloatingPointError('ACT KL loss is materially negative; stop and check numerical precision')
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True).item())
    return {'loss': float(loss.item()), 'l1_loss': details['l1_loss'],
            'kld_loss': details.get('kld_loss', 0.), 'grad_norm_before_clip': norm,
            'valid_target_fraction': float((~batch['action_is_pad']).float().mean().item())}


def forward_preflight(dataset, policy, preprocess, torch, use_bf16):
    """Exercise the actual training forward, without gradients or updates."""
    batch = preprocess(dataset.batch(np.array([0, min(190, len(dataset) - 1)]), torch))
    policy.train()
    context = torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else nullcontext()
    with torch.no_grad(), context:
        loss, details = policy.forward(batch)
    if not torch.isfinite(loss) or details.get('kld_loss', 0.) < -1e-5:
        raise FloatingPointError('Preflight loss is non-finite or has a materially negative KL term')
    return {'loss': float(loss.item()), **details, 'optimizer_steps': 0,
            'precision': 'bfloat16_autocast' if use_bf16 else 'float32'}


def train(args, dataset):
    model_dir, output = args.model_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; earlier experiments are never replaced')
    if output == model_dir or output.is_relative_to(model_dir):
        raise ValueError('Training output must be outside the source checkpoint')
    source_hashes = {name: digest(model_dir / name) for name in MODEL_FILES}
    torch, policy, preprocess, postprocess = load_policy(model_dir)
    np.random.seed(20261023)
    torch.manual_seed(20261023)
    torch.cuda.manual_seed_all(20261023)
    rng = np.random.default_rng(20261023)
    validation = np.linspace(0, len(dataset) - 1, 12, dtype=int)
    train_indices = np.setdiff1d(np.arange(len(dataset)), validation)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=1e-4)
    use_bf16 = torch.cuda.is_bf16_supported() and not args.fp32
    precision = 'bfloat16_autocast_with_float32_parameters' if use_bf16 else 'float32'
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'single_scene_simulation_imitation_adaptation', 'steps_requested': args.steps,
              'completed_steps': 0, 'batch_size': 2, 'learning_rate': 1e-5, 'weight_decay': 1e-4,
              'gradient_clip_norm': 10., 'seed': 20261023, 'precision': precision,
              'prior_adaptation_steps': args.prior_adaptation_steps,
              'total_adaptation_steps_requested': args.prior_adaptation_steps + args.steps,
              'continued_from': {'model_dir': str(model_dir),
                  'prior_adaptation_steps': args.prior_adaptation_steps,
                  'optimizer_state_restored': False,
                  'note': 'Continue from saved model weights with a fresh AdamW optimizer; not an exact optimizer resume.'},
              'training_script_sha256': digest(Path(__file__).resolve()),
              'source_model': str(model_dir), 'source_files_sha256': source_hashes,
              'dataset_manifest': str(dataset.manifest_path), 'dataset_manifest_sha256': digest(dataset.manifest_path),
              'frames': len(dataset), 'normalization': 'Original saved model preprocessing and statistics',
              'training_observation_anchors': train_indices.tolist(), 'diagnostic_observation_anchors': validation.tolist(),
              'diagnostic_limit': 'Observation anchors are excluded from minibatches, but future action windows overlap. Same scene and same teacher, not independent generalization.',
              'cpu_budget': CPU_BUDGET, 'gpu': torch.cuda.get_device_name(),
              'hardware_io': False, 'network_access': False, 'real_robot_qualified': False,
              'checkpoints': [], 'oom_batch_reductions': []}
    save_json(output / 'training_report.json', report)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        report['forward_preflight'] = forward_preflight(dataset, policy, preprocess, torch, use_bf16)
        print(json.dumps({'status': 'forward_preflight_passed', **report['forward_preflight']}), flush=True)
        report['baseline_temporal_diagnostic'] = temporal_diagnostic(
            dataset, policy, preprocess, postprocess, validation, torch)
        save_json(output / 'training_report.json', report)
        actual_batch_size = 2
        with (output / 'training_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['step', 'loss', 'l1_loss', 'kld_loss',
                'grad_norm_before_clip', 'update_wall_s', 'cuda_allocated_mib', 'cuda_peak_allocated_mib', 'valid_target_fraction', 'batch_size'])
            writer.writeheader()
            for step in range(1, args.steps + 1):
                update_started = time.perf_counter()
                indices = rng.choice(train_indices, size=actual_batch_size, replace=False)
                while True:
                    try:
                        gradients = compute_gradients(dataset, indices, policy, preprocess,
                                                      optimizer, torch, use_bf16)
                        break
                    except torch.cuda.OutOfMemoryError:
                        if actual_batch_size == 1:
                            raise
                        optimizer.zero_grad(set_to_none=True)
                        actual_batch_size = 1
                        indices = indices[:1]
                        report['oom_batch_reductions'].append({'step': step, 'from': 2, 'to': 1,
                            'optimizer_step_started': False})
                    # The exception frame is gone before dropping cached blocks.
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
                if step % 100 == 0 or step == args.steps:
                    optimizer.zero_grad(set_to_none=True)
                    diagnostic = temporal_diagnostic(dataset, policy, preprocess, postprocess, validation, torch)
                    report['checkpoints'].append(save_checkpoint(
                        output, step, policy, model_dir, source_hashes, diagnostic,
                        args.prior_adaptation_steps))
                    report['elapsed_wall_s'] = time.perf_counter() - started
                    save_json(output / 'training_report.json', report)
                    print(json.dumps({'status': 'checkpoint_saved', 'step': step,
                        'total_adaptation_steps': args.prior_adaptation_steps + step,
                        'diagnostic': diagnostic}), flush=True)
        if any(digest(model_dir / name) != value for name, value in source_hashes.items()):
            raise RuntimeError('Original source model files changed during adaptation')
        report.update({'status': 'completed_simulation_adaptation_not_policy_success',
                       'total_adaptation_steps_completed': args.prior_adaptation_steps + args.steps,
                       'original_source_files_unchanged': True,
                       'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
                       'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2})
    except Exception as error:
        report.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        report['elapsed_wall_s'] = time.perf_counter() - started
        report['image_cache'] = {'capacity_shards': 2, 'hits': dataset.cache_hits, 'misses': dataset.cache_misses}
        save_json(output / 'training_report.json', report)
    print(json.dumps({'status': report['status'], 'completed_steps': report['completed_steps'],
                      'elapsed_wall_s': report['elapsed_wall_s'], 'report': str(output / 'training_report.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-manifest', required=True, type=Path)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/act_rebot_cube')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--prior-adaptation-steps', type=int, default=0,
                        help='Previously completed simulated adaptation steps in the source weights; metadata only')
    parser.add_argument('--fp32', action='store_true', help='Disable autocast and train entirely in float32')
    parser.add_argument('--check-data', action='store_true', help='Validate exported shards and batch contract without loading weights or training')
    args = parser.parse_args()
    if not 1 <= args.steps <= 600:
        parser.error('steps must be within 1..600')
    if args.prior_adaptation_steps < 0:
        parser.error('prior-adaptation-steps must be nonnegative')
    if not args.check_data and args.output_dir is None:
        parser.error('output-dir is required for training')
    dataset = SimDemonstration(args.data_manifest)
    if args.check_data:
        configure_torch()
        import torch
        for index in dataset.offsets[:-1]:
            dataset.images(int(index))
        batch = dataset.batch(np.array([0, len(dataset) - 1]), torch)
        assert batch['action'].shape == (2, CHUNK_SIZE, 7)
        assert batch['action_is_pad'][-1].tolist() == [False] + [True] * 99
        assert torch.equal(batch['action'][-1], batch['action'][-1, :1].expand(CHUNK_SIZE, -1))
        print(json.dumps({'status': 'data_and_batch_contract_passed', 'frames': len(dataset),
            'shards': len(dataset.shards), 'batch_shapes': {key: list(value.shape) for key, value in batch.items()},
            'training_steps_executed': 0, 'model_loaded': False, 'hardware_io': False}), flush=True)
        return
    train(args, dataset)


if __name__ == '__main__':
    main()
