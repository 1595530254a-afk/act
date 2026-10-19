"""Check a bundled ACT model on five bundled observations, without a robot or dataset download."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / 'local/runtime/cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'local/runtime/cache/torch'))

import numpy as np
import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def run(device):
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(0)
    manifest = json.loads((ROOT / 'bundle_manifest.json').read_text(encoding='utf-8'))
    # Check the model/processor/fixture identity even when called without the PowerShell wrapper.
    for entry in manifest['files']:
        if entry['path'].startswith(('models/', 'fixtures/')):
            with (ROOT / entry['path']).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                    raise RuntimeError(f'Payload SHA256 mismatch: {entry["path"]}')
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Record GPU/driver details; CPU checks do not verify GPU readiness.')
        probe = torch.ones((32, 32), device='cuda')
        torch.testing.assert_close(probe @ probe, torch.full((32, 32), 32.0, device='cuda'))
        torch.cuda.synchronize()
    model_dir = ROOT / 'models/act_rebot_cube'
    config = ACTConfig.from_pretrained(model_dir, local_files_only=True)
    config.device = device
    config.pretrained_backbone_weights = None
    config.push_to_hub = False
    policy = ACTPolicy.from_pretrained(model_dir, config=config, local_files_only=True, strict=True)
    policy.to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        config, pretrained_path=str(model_dir),
        preprocessor_overrides={'device_processor': {'device': device}},
    )
    metadata = json.loads((ROOT / 'fixtures/provenance.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'fixtures/observations.npz', allow_pickle=False) as fixture:
        state = fixture['state']
        wrist = fixture['wrist']
        top = fixture['top']
    if state.shape != (5, 7) or wrist.shape != (5, 3, 480, 640) or top.shape != wrist.shape:
        raise RuntimeError('Unexpected bundled observation shapes.')
    if wrist.dtype != np.uint8 or top.dtype != np.uint8 or not np.isfinite(state).all():
        raise RuntimeError('Unexpected bundled observation dtype or values.')
    observations = [{
        'observation.state': torch.from_numpy(state[index].copy()).float(),
        'observation.images.wrist': torch.from_numpy(wrist[index].copy()).float() / 255,
        'observation.images.top': torch.from_numpy(top[index].copy()).float() / 255,
    } for index in range(5)]

    def synchronize():
        if device == 'cuda':
            torch.cuda.synchronize()

    def predict(observation):
        batch = preprocess({key: value.clone() for key, value in observation.items()})
        actions = postprocess(policy.predict_action_chunk(batch)).detach().cpu()
        synchronize()
        if list(actions.shape) != [1, 100, 7] or not torch.isfinite(actions).all():
            raise RuntimeError(f'Invalid model output: {actions.shape}')
        return actions[0]

    timings = []
    predictions = []
    with torch.inference_mode():
        for _ in range(3):
            predict(observations[0])
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for index, observation in enumerate(observations):
            for _ in range(5):
                synchronize()
                started = time.perf_counter()
                actions = predict(observation)
                timings.append((time.perf_counter() - started) * 1000)
            predictions.append(actions.numpy())
            print(f'Frame {metadata["frame_indices"][index]}: finite [1, 100, 7]', flush=True)
    predictions = np.stack(predictions)
    reference = np.load(ROOT / f'fixtures/reference_actions_{device}.npy', allow_pickle=False)
    np.testing.assert_allclose(predictions, reference, rtol=1e-4, atol=0.005)
    return {
        'status': 'passed', 'scope': 'bundled_fixture_offline_inference_only',
        'bundle_id': manifest['bundle_id'], 'device': device, 'platform': platform.platform(),
        'python': platform.python_version(), 'machine': platform.machine(),
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'lerobot', 'numpy']},
        'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
        'gpu_total_memory_mib': torch.cuda.get_device_properties(0).total_memory / 1024**2 if device == 'cuda' else None,
        'torch_cuda_version': torch.version.cuda,
        'model_revision': manifest['model']['revision'], 'model_and_fixture_hashes': 'passed',
        'sampled_frames': 5, 'warmup_runs': 3, 'timed_runs': len(timings),
        'output_shape': list(predictions.shape),
        'reference_check': 'passed', 'reference_max_abs_difference': float(np.max(np.abs(predictions - reference))),
        'reference_device': device,
        'reference_tolerance': {'rtol': 1e-4, 'atol': 0.005},
        'latency_ms': {'mean': float(np.mean(timings)), 'p95': float(np.percentile(timings, 95)), 'max': max(timings)},
        'latency_scope': 'CPU float observations -> processors -> model -> CPU actions; excludes fixture decompression and uint8 conversion',
        'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else None,
        'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2 if device == 'cuda' else None,
        'robot_connected': False, 'robot_commands_sent': 0, 'training_steps': 0,
        'bundle_directory': str(ROOT),
        'limitations': [
            'Five observations are from the published training dataset, not a generalization benchmark.',
            'Reference tolerances allow small float32 CPU/GPU differences; they are not physical action tolerances.',
            'This check does not establish camera integration, CAN/SDK compatibility, closed-loop timing or grasp success.',
            'PyTorch memory metrics exclude driver/display memory; CPU passing does not establish CUDA readiness.',
        ],
    }, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    output_dir = ROOT / 'local/laptop_readiness'
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f'offline_inference_{args.device}.json'
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    try:
        report, predictions = run(args.device)
    except Exception as error:
        report_path.write_text(json.dumps({
            'status': 'failed', 'device': args.device, 'started_at': started,
            'error': f'{type(error).__name__}: {error}',
            'robot_commands_sent': 0, 'training_steps': 0,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise
    report['started_at'] = started
    report['completed_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    np.save(output_dir / f'predicted_actions_{args.device}.npy', predictions)
    print(json.dumps({key: report[key] for key in ['status', 'device', 'gpu', 'latency_ms', 'reference_max_abs_difference']}, ensure_ascii=False, indent=2))
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
