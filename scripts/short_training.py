"""Verify local ACT continued training, checkpoint saving and reloading offline."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache/torch'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.common.train_utils import load_training_state, save_training_state
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.random_utils import set_seed


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_policy(model_dir):
    config = ACTConfig.from_pretrained(model_dir, local_files_only=True)
    config.device = 'cuda'
    config.pretrained_backbone_weights = None
    config.push_to_hub = False
    config.repo_id = None
    policy = ACTPolicy.from_pretrained(model_dir, config=config, local_files_only=True, strict=True)
    policy.to('cuda')
    preprocess, postprocess = make_pre_post_processors(
        config, pretrained_path=str(model_dir),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}},
    )
    return policy, preprocess, postprocess


def predict_probes(policy, preprocess, postprocess, observations):
    policy.eval()
    predictions = []
    with torch.inference_mode():
        for observation in observations:
            # Only observations reach the inference model, never demonstration actions.
            batch = preprocess({key: value.clone() for key, value in observation.items()})
            actions = postprocess(policy.predict_action_chunk(batch)).detach().cpu()
            if list(actions.shape) != [1, policy.config.chunk_size, 7] or not torch.isfinite(actions).all():
                raise RuntimeError(f'Invalid probe output: {actions.shape}')
            predictions.append(actions[0])
    return torch.stack(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error('steps and batch-size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('This training verification requires CUDA.')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed = 1000
    set_seed(seed)
    started_at = datetime.now(timezone(timedelta(hours=8)))
    run_dir = ROOT / 'runs' / f'act_short_{started_at:%Y%m%d_%H%M%S}'
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / 'checkpoint'
    model_dir = ROOT / 'models/act_rebot_cube'
    manifest = json.loads((ROOT / 'resources/resource_manifest.json').read_text(encoding='utf-8'))
    source_hashes = {path.name: file_hash(path) for path in model_dir.iterdir() if path.is_file()}
    policy, preprocess, postprocess = load_policy(model_dir)
    optimizer_config = policy.config.get_optimizer_preset()
    optimizer = optimizer_config.build(policy.get_optim_params())
    fps = json.loads((ROOT / 'data/rebot_cube/meta/info.json').read_text(encoding='utf-8'))['fps']
    dataset = LeRobotDataset(
        manifest['dataset']['repo_id'], root=ROOT / 'data/rebot_cube',
        revision=manifest['dataset']['revision'], video_backend='pyav',
        delta_timestamps={'action': [index / fps for index in policy.config.action_delta_indices]},
    )
    if args.batch_size > len(dataset):
        raise ValueError('batch-size exceeds dataset size')

    # An episode's final frame has one real target and 99 padded future actions.
    final_sample = dataset[len(dataset) - 1]
    expected_pad = torch.arange(policy.config.chunk_size) > 0
    torch.testing.assert_close(final_sample['action_is_pad'], expected_pad)
    torch.testing.assert_close(
        final_sample['action'], final_sample['action'][0].expand(policy.config.chunk_size, -1),
    )
    probe_indices = np.linspace(0, len(dataset) - 1, 5, dtype=int).tolist()
    observations = []
    for index in probe_indices:
        sample = dataset[index]
        observations.append({key: sample[key] for key in policy.config.input_features})
    before_predictions = predict_probes(policy, preprocess, postprocess, observations)
    initial_params = {name: param.detach().cpu().clone() for name, param in policy.named_parameters()}
    run_config = {
        'scope': 'short_continued_training_pipeline_check',
        'started_at': started_at.isoformat(), 'steps': args.steps, 'batch_size': args.batch_size,
        'seed': seed, 'precision': 'float32', 'num_workers': 0, 'shuffle': True,
        'optimizer': {'type': optimizer_config.type, **asdict(optimizer_config)},
        'backbone_lr': policy.config.optimizer_lr_backbone,
        'scheduler': None, 'chunk_size': policy.config.chunk_size,
        'model_source': manifest['model'], 'dataset_source': manifest['dataset'],
        'source_file_sha256': source_hashes, 'dataset_frames': len(dataset),
        'robot_connected': False, 'robot_commands_sent': 0,
    }
    write_json(run_dir / 'run_config.json', run_config)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,
        generator=torch.Generator().manual_seed(seed), collate_fn=lerobot_collate_fn,
    )
    iterator = iter(loader)
    rows = []
    seen_frames = set()
    seen_episodes = set()
    policy.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    print(f'Run directory: {run_dir}', flush=True)
    with (run_dir / 'training_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'step', 'loss', 'l1_loss', 'kld_loss', 'grad_norm_before_clip',
            'learning_rate', 'valid_target_fraction', 'data_read_s', 'update_s',
        ])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            read_started = time.perf_counter()
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
            seen_frames.update(raw_batch['index'].tolist())
            seen_episodes.update(raw_batch['episode_index'].tolist())
            data_read_s = time.perf_counter() - read_started
            update_started = time.perf_counter()
            batch = preprocess(raw_batch)
            if list(batch['action'].shape) != [args.batch_size, policy.config.chunk_size, 7]:
                raise RuntimeError(f'Invalid action label shape: {batch["action"].shape}')
            for key in [*policy.config.input_features, 'action']:
                if not torch.isfinite(batch[key]).all():
                    raise RuntimeError(f'Non-finite training input: {key}')
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.forward(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at step {step}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), optimizer_config.grad_clip_norm, error_if_nonfinite=True,
            ).item()
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError(f'Invalid gradient norm at step {step}: {grad_norm}')
            optimizer.step()
            torch.cuda.synchronize()
            row = {
                'step': step, 'loss': loss.item(), **loss_dict,
                'grad_norm_before_clip': grad_norm, 'learning_rate': optimizer.param_groups[0]['lr'],
                'valid_target_fraction': (~batch['action_is_pad']).float().mean().item(),
                'data_read_s': data_read_s, 'update_s': time.perf_counter() - update_started,
            }
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(f'Step {step}/{args.steps}: loss={row["loss"]:.6f}, '
                      f'grad_norm={grad_norm:.4f}, update={row["update_s"]:.3f}s', flush=True)

    training_s = time.perf_counter() - training_started
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    optimizer.zero_grad(set_to_none=True)
    changed_elements = 0
    changed_tensors = 0
    delta_squared_sum = 0.0
    delta_max = 0.0
    for name, param in policy.named_parameters():
        current = param.detach().cpu()
        if not torch.isfinite(current).all():
            raise RuntimeError(f'Non-finite parameter after training: {name}')
        delta = current - initial_params[name]
        count = torch.count_nonzero(delta).item()
        changed_elements += count
        changed_tensors += int(count > 0)
        delta_squared_sum += delta.double().square().sum().item()
        delta_max = max(delta_max, delta.abs().max().item())
    if changed_elements == 0:
        raise RuntimeError('Optimizer did not change any parameters.')
    after_predictions = predict_probes(policy, preprocess, postprocess, observations)
    pretrained_dir = checkpoint_dir / 'pretrained_model'
    policy.save_pretrained(pretrained_dir)
    preprocess.save_pretrained(pretrained_dir)
    postprocess.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, args.steps, optimizer, num_processes=1, batch_size=args.batch_size)
    reloaded, reloaded_preprocess, reloaded_postprocess = load_policy(pretrained_dir)
    reloaded_predictions = predict_probes(reloaded, reloaded_preprocess, reloaded_postprocess, observations)
    torch.testing.assert_close(reloaded_predictions, after_predictions, rtol=1e-5, atol=1e-5)
    for name, saved_tensor in policy.state_dict().items():
        if not torch.equal(saved_tensor, reloaded.state_dict()[name]):
            raise RuntimeError(f'Reloaded weight mismatch: {name}')
    reloaded_optimizer = reloaded.config.get_optimizer_preset().build(reloaded.get_optim_params())
    restored_step, reloaded_optimizer, _ = load_training_state(checkpoint_dir, reloaded_optimizer, None)
    if restored_step != args.steps or len(reloaded_optimizer.state) != len(optimizer.state):
        raise RuntimeError('Training state reload mismatch.')
    saved_optimizer_states = optimizer.state_dict()['state']
    restored_optimizer_states = reloaded_optimizer.state_dict()['state']
    for parameter_id, old_state in saved_optimizer_states.items():
        new_state = restored_optimizer_states[parameter_id]
        for key, value in old_state.items():
            if not torch.equal(value.cpu(), new_state[key].cpu()):
                raise RuntimeError(f'Reloaded optimizer state mismatch: {key}')
    for filename, digest in source_hashes.items():
        if file_hash(model_dir / filename) != digest:
            raise RuntimeError(f'Original published file was modified: {filename}')
    np.save(run_dir / 'probe_predictions_before.npy', before_predictions.numpy())
    np.save(run_dir / 'probe_predictions_after.npy', after_predictions.numpy())
    report = {
        **run_config, 'status': 'passed', 'completed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'platform': platform.platform(), 'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'lerobot', 'av', 'datasets']},
        'gpu': torch.cuda.get_device_name(), 'torch_cuda_version': torch.version.cuda,
        'training_wall_s': training_s, 'training_updates_per_s': args.steps / training_s,
        'cuda_peak_allocated_mib': peak_allocated, 'cuda_peak_reserved_mib': peak_reserved,
        'loss_first_10_mean': float(np.mean([row['loss'] for row in rows[:10]])),
        'loss_last_10_mean': float(np.mean([row['loss'] for row in rows[-10:]])),
        'loss_min': min(row['loss'] for row in rows), 'loss_max': max(row['loss'] for row in rows),
        'gradient_norm_min': min(row['grad_norm_before_clip'] for row in rows),
        'gradient_norm_max': max(row['grad_norm_before_clip'] for row in rows),
        'clipped_steps': sum(row['grad_norm_before_clip'] > optimizer_config.grad_clip_norm for row in rows),
        'all_losses_and_gradient_norms_finite': True,
        'total_parameter_elements': sum(param.numel() for param in policy.parameters()),
        'changed_parameter_elements': changed_elements, 'changed_parameter_tensors': changed_tensors,
        'parameter_delta_l2': math.sqrt(delta_squared_sum), 'parameter_delta_max_abs': delta_max,
        'unique_training_frames': len(seen_frames), 'unique_training_episodes': len(seen_episodes),
        'episode_end_padding_check': 'passed', 'strict_weights_reload_check': 'passed',
        'optimizer_state_reload_check': 'passed', 'restored_step': restored_step,
        'probe_frames': probe_indices, 'probe_output_shape': list(after_predictions.shape),
        'reload_prediction_max_abs_difference': (after_predictions - reloaded_predictions).abs().max().item(),
        'before_after_prediction_mean_abs_difference': (after_predictions - before_predictions).abs().mean().item(),
        'original_source_files_unchanged': True,
        'checkpoint_directory': str(checkpoint_dir.relative_to(ROOT)),
        'checkpoint_file_sha256': {
            str(path.relative_to(checkpoint_dir)): file_hash(path)
            for path in checkpoint_dir.rglob('*') if path.is_file()
        },
        'limitations': [
            'This is a short local loop using LeRobot ACT, dataset, processors and optimizer presets, not the full LeRobot training CLI.',
            'The published model already trained on these demonstrations; this is not a held-out generalization benchmark.',
            'Random minibatch losses and output changes do not establish improved accuracy or grasp success.',
            'Optimizer and RNG state are saved; this script does not implement exact data-order training resume.',
            'GPU peaks describe PyTorch tensors and allocator, not total driver/display memory.',
            'No robot or camera connection and no laptop performance measurement are included.',
        ],
    }
    write_json(run_dir / 'training_report.json', report)
    report_dir = ROOT / 'reports'
    report_dir.mkdir(exist_ok=True)
    write_json(report_dir / 'short_training_cuda.json', report)
    (report_dir / 'short_training_metrics.csv').write_bytes((run_dir / 'training_metrics.csv').read_bytes())
    print(json.dumps({key: report[key] for key in [
        'status', 'steps', 'batch_size', 'training_wall_s', 'cuda_peak_allocated_mib',
        'changed_parameter_elements', 'strict_weights_reload_check', 'optimizer_state_reload_check',
        'reload_prediction_max_abs_difference', 'checkpoint_directory',
    ]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
