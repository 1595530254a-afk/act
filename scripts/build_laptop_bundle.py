"""Build a standalone laptop ACT inference ZIP from verified local resources."""

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='ACT_laptop_20261019_v2')
    args = parser.parse_args()
    if args.name in ['', '.', '..'] or any(char in args.name for char in '/\\:'):
        parser.error('name must be a simple directory name')
    delivery = ROOT / 'delivery'
    delivery.mkdir(exist_ok=True)
    bundle = delivery / args.name
    archive = delivery / f'{args.name}.zip'
    if bundle.exists() or archive.exists():
        raise FileExistsError('Choose a new --name; existing deliveries are never overwritten.')
    resources = json.loads((ROOT / 'resources/resource_manifest.json').read_text(encoding='utf-8'))
    downloaded = json.loads((ROOT / 'reports/download_manifest.json').read_text(encoding='utf-8'))
    shutil.copytree(ROOT / 'deployment/laptop', bundle, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    model_dir = bundle / 'models/act_rebot_cube'
    model_dir.mkdir(parents=True)
    for entry in downloaded['model']['verified_files']:
        source = ROOT / 'models/act_rebot_cube' / entry['file']
        if sha256(source) != entry['sha256']:
            raise RuntimeError(f'Original model file hash mismatch: {source}')
        shutil.copy2(source, model_dir / entry['file'])
    shutil.copy2(ROOT / 'requirements.lock.txt', bundle / 'requirements.lock.txt')
    uv_root = ROOT / 'tools/uv'
    (bundle / 'tools/uv').mkdir(parents=True)
    shutil.copy2(uv_root / 'bin/uv.exe', bundle / 'tools/uv/uv.exe')
    (bundle / 'licenses').mkdir()
    for filename in ['LICENSE-APACHE', 'LICENSE-MIT']:
        shutil.copy2(uv_root / 'uv-0.13.1.dist-info/licenses' / filename, bundle / 'licenses' / f'uv-{filename}.txt')
    shutil.copy2(uv_root / 'uv-0.13.1.dist-info/sboms/uv.cyclonedx.json', bundle / 'licenses/uv.cyclonedx.json')
    (bundle / 'sources').mkdir()
    shutil.copy2(ROOT / 'data/rebot_cube/README.md', bundle / 'sources/dataset_README.md')
    fixture_dir = bundle / 'fixtures'
    fixture_dir.mkdir()
    dataset = LeRobotDataset(
        resources['dataset']['repo_id'], root=ROOT / 'data/rebot_cube',
        revision=resources['dataset']['revision'], video_backend='pyav',
    )
    indices = np.linspace(0, len(dataset) - 1, 5, dtype=int).tolist()
    states, wrist, top, episodes = [], [], [], []
    for index in indices:
        sample = dataset[index]
        states.append(sample['observation.state'].numpy())
        episodes.append(int(sample['episode_index'].item()))
        for key, destination in [('wrist', wrist), ('top', top)]:
            image = sample[f'observation.images.{key}']
            pixels = (image.numpy() * 255).round().astype(np.uint8)
            torch.testing.assert_close(torch.from_numpy(pixels).float() / 255, image, rtol=0, atol=1e-7)
            destination.append(pixels)
    np.savez_compressed(fixture_dir / 'observations.npz', state=np.stack(states), wrist=np.stack(wrist), top=np.stack(top))
    reference = np.load(ROOT / 'runs/offline_inference_cuda/predicted_action_chunks.npy', allow_pickle=False)
    if reference.shape != (5, 100, 7) or not np.isfinite(reference).all():
        raise RuntimeError('Original desktop reference actions are invalid.')
    baseline = json.loads((ROOT / 'reports/offline_inference_cuda.json').read_text(encoding='utf-8'))
    if [sample['frame_index'] for sample in baseline['samples']] != indices:
        raise RuntimeError('Reference frame selection differs from fixtures.')
    if baseline['model_revision'] != resources['model']['revision']:
        raise RuntimeError('Reference policy revision differs from bundle.')
    np.testing.assert_allclose(np.stack(states), np.asarray([sample['state'] for sample in baseline['samples']]), rtol=0, atol=1e-7)
    np.save(fixture_dir / 'reference_actions_cuda.npy', reference)
    # CPU and CUDA backends can differ (for example CUDA convolution TF32 math).
    # Keep a distinct CPU reference instead of weakening the reproduction tolerance.
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    config = ACTConfig.from_pretrained(model_dir, local_files_only=True)
    config.device = 'cpu'
    config.pretrained_backbone_weights = None
    config.push_to_hub = False
    policy = ACTPolicy.from_pretrained(model_dir, config=config, local_files_only=True, strict=True)
    policy.to('cpu').eval()
    preprocess, postprocess = make_pre_post_processors(
        config, pretrained_path=str(model_dir),
        preprocessor_overrides={'device_processor': {'device': 'cpu'}},
    )
    cpu_reference = []
    with torch.inference_mode():
        for index in range(5):
            observation = {
                'observation.state': torch.from_numpy(states[index].copy()),
                'observation.images.wrist': torch.from_numpy(wrist[index].copy()).float() / 255,
                'observation.images.top': torch.from_numpy(top[index].copy()).float() / 255,
            }
            cpu_reference.append(postprocess(policy.predict_action_chunk(preprocess(observation)))[0].cpu().numpy())
    cpu_reference = np.stack(cpu_reference)
    if cpu_reference.shape != reference.shape or not np.isfinite(cpu_reference).all():
        raise RuntimeError('Generated CPU reference actions are invalid.')
    np.save(fixture_dir / 'reference_actions_cpu.npy', cpu_reference)
    provenance = {
        'dataset': resources['dataset'], 'model': resources['model'], 'frame_indices': indices,
        'episode_indices': episodes, 'rgb_layout': 'NCHW uint8; convert to float32 / 255',
        'states': 'Unmodified float32 dataset values, seven channels in model order',
        'image_conversion': 'Decoded PyAV RGB float01 -> round(*255) uint8; round-trip verified atol 1e-7',
        'reference': 'Original desktop published-policy action chunks; not demonstration labels or grasp scores',
        'reference_devices': ['cpu', 'cuda'],
        'cpu_cuda_reference_max_abs_difference': float(np.max(np.abs(cpu_reference - reference))),
        'action_names': baseline['action_names'],
    }
    (fixture_dir / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest = {
        'bundle_id': args.name, 'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'scope': 'Windows x64 offline inference installation and verification; no robot control',
        'model': resources['model'], 'dataset': resources['dataset'],
        'files': [{'path': path.relative_to(bundle).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                  for path in sorted(bundle.rglob('*')) if path.is_file()],
    }
    (bundle / 'bundle_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                stream.write(path, path.relative_to(delivery).as_posix())
    archive_hash = sha256(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{archive_hash}  {archive.name}\n', encoding='utf-8')
    instruction_copy = delivery / f'{args.name}_给Codex.md'
    shutil.copy2(bundle / '先读我_给笔记本Codex.md', instruction_copy)
    summary = {
        'bundle_directory': str(bundle), 'archive': str(archive), 'instructions': str(instruction_copy),
        'zip_bytes': archive.stat().st_size, 'zip_sha256': archive_hash,
        'payload_file_count': len(manifest['files']),
        'uncompressed_payload_bytes': sum(entry['bytes'] for entry in manifest['files']),
    }
    (delivery / f'{args.name}_build.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
