"""Run a local ACT model on real dataset frames without a robot connection."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache/torch'))

from cpu_budget import CPU_BUDGET, configure_torch

import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--model-dir', type=Path, help='Local saved model; defaults to the published policy.')
    parser.add_argument('--run-name', help='Separate output/report name, preserving the published baseline.')
    args = parser.parse_args()
    run_name = args.run_name or f'offline_inference_{"custom_" if args.model_dir else ""}{args.device}'
    if run_name in ['.', '..'] or '/' in run_name or '\\' in run_name or ':' in run_name:
        parser.error('run-name must be a simple filename without path separators')
    configure_torch()
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; the requested GPU verification cannot pass.')
        probe = torch.ones((32, 32), device=device)
        torch.testing.assert_close(probe @ probe, torch.full((32, 32), 32.0, device=device))
        torch.cuda.synchronize()

    manifest = json.loads((ROOT / 'resources/resource_manifest.json').read_text(encoding='utf-8'))
    published_dir = ROOT / 'models/act_rebot_cube'
    model_dir = args.model_dir.resolve() if args.model_dir else published_dir
    dataset_dir = ROOT / 'data/rebot_cube'
    output_dir = ROOT / 'runs' / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / 'reports'
    report_dir.mkdir(exist_ok=True)

    config = ACTConfig.from_pretrained(model_dir, local_files_only=True)
    config.device = device.type
    # The complete ACT checkpoint already contains the vision backbone weights.
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(model_dir, config=config, local_files_only=True, strict=True)
    policy.to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={'device_processor': {'device': device.type}},
    )
    dataset = LeRobotDataset(
        manifest['dataset']['repo_id'], root=dataset_dir,
        revision=manifest['dataset']['revision'], video_backend='pyav',
    )
    expected_frames = json.loads((dataset_dir / 'meta/info.json').read_text(encoding='utf-8'))['total_frames']
    if len(dataset) != expected_frames:
        raise RuntimeError(f'Unexpected dataset size: {len(dataset)} != {expected_frames}')
    stats = json.loads((dataset_dir / 'meta/stats.json').read_text(encoding='utf-8'))['action']
    action_min = np.asarray(stats['min'], dtype=np.float32)
    action_max = np.asarray(stats['max'], dtype=np.float32)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    def predict(observation):
        batch = preprocess({key: value.clone() for key, value in observation.items()})
        actions = postprocess(policy.predict_action_chunk(batch))
        synchronize()
        return actions.detach().cpu()

    rows = []
    timings = []
    chunks = []
    indices = np.linspace(0, len(dataset) - 1, 5, dtype=int).tolist()
    with torch.inference_mode():
        for position, index in enumerate(indices):
            read_start = time.perf_counter()
            sample = dataset[index]
            read_ms = (time.perf_counter() - read_start) * 1000
            observation = {key: sample[key] for key in config.input_features}
            for key, value in observation.items():
                if not torch.isfinite(value).all():
                    raise RuntimeError(f'Non-finite observation: {key}, frame {index}')
                if '.images.' in key:
                    if list(value.shape) != [3, 480, 640] or value.min() < 0 or value.max() > 1:
                        raise RuntimeError(f'Unexpected image shape/range: {key}, frame {index}')
            if position == 0:
                for key in config.image_features:
                    pixels = (observation[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    Image.fromarray(pixels).save(output_dir / f'input_{key.rsplit(".", 1)[-1]}.png')
                for _ in range(3):
                    predict(observation)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

            sample_timings = []
            for _ in range(5):
                synchronize()
                started = time.perf_counter()
                actions = predict(observation)
                sample_timings.append((time.perf_counter() - started) * 1000)
            if list(actions.shape) != [1, config.chunk_size, 7] or not torch.isfinite(actions).all():
                raise RuntimeError(f'Unexpected action output at frame {index}: {actions.shape}')
            predicted = actions[0, 0].numpy()
            target = sample['action'].detach().cpu().numpy()
            row = {
                'frame_index': index,
                'episode_index': int(sample['episode_index'].item()),
                'timestamp_s': float(sample['timestamp'].item()),
                'video_decode_and_read_ms': read_ms,
                'inference_mean_ms': float(np.mean(sample_timings)),
                'state': sample['observation.state'].tolist(),
                'first_predicted_action': predicted.tolist(),
                'dataset_action': target.tolist(),
                'first_action_absolute_error': np.abs(predicted - target).tolist(),
                'first_action_outside_dataset_range': ((predicted < action_min) | (predicted > action_max)).tolist(),
            }
            rows.append(row)
            timings.extend(sample_timings)
            chunks.append(actions[0].numpy())
            print(f'Frame {index}: shape={list(actions.shape)}, mean={row["inference_mean_ms"]:.1f} ms', flush=True)

    np.save(output_dir / 'predicted_action_chunks.npy', np.stack(chunks))
    with (model_dir / 'model.safetensors').open('rb') as model_stream:
        model_sha256 = hashlib.file_digest(model_stream, 'sha256').hexdigest()
    report = {
        'status': 'passed', 'scope': 'offline_inference_only',
        'cpu_budget': CPU_BUDGET,
        'robot_connected': False, 'robot_commands_sent': 0,
        'device': device.type, 'platform': platform.platform(),
        'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'lerobot', 'av', 'datasets']},
        'torch_cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'model_revision': manifest['model']['revision'] if model_dir == published_dir else None,
        'model_directory': str(model_dir),
        'model_sha256': model_sha256,
        'dataset_revision': manifest['dataset']['revision'],
        'dataset_frames': len(dataset), 'sampled_frames': len(rows),
        'warmup_runs': 3, 'timed_runs': len(timings),
        'latency_scope': 'CPU observation tensors -> preprocessing -> full action chunk -> postprocessing -> CPU actions; excludes video decoding',
        'latency_ms': {'mean': float(np.mean(timings)), 'median': float(np.median(timings)), 'p95': float(np.percentile(timings, 95)), 'max': max(timings)},
        'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else None,
        'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2 if device.type == 'cuda' else None,
        'output_shape_per_sample': [1, config.chunk_size, 7],
        'first_action_mae_by_channel': np.mean([row['first_action_absolute_error'] for row in rows], axis=0).tolist(),
        'action_names': json.loads((dataset_dir / 'meta/info.json').read_text(encoding='utf-8'))['features']['action']['names'],
        'samples': rows,
        'limitations': [
            'Frames come from the published training dataset, not a held-out generalization benchmark.',
            'Action errors use dataset coordinate units; these are not verified physical joint limits.',
            'This check establishes loading, decoding, preprocessing and finite action generation, not grasp success or robot safety.',
            'Desktop timings do not establish laptop performance.',
        ],
        'artifacts_directory': str(output_dir.relative_to(ROOT)),
    }
    report_path = report_dir / f'{run_name}.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status', 'gpu', 'latency_ms', 'cuda_peak_allocated_mib', 'output_shape_per_sample']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
