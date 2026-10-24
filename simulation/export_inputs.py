"""Export recorded data and one original ACT prediction for simulation only.

No motor SDK, camera API, coordinate mapping or training is used here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
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
import pyarrow.parquet as pq


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def validate_episode(state, action, timestamp, frame_index, global_index, *,
                     expected_length, dataset_from_index, fps):
    """Keep the actual recorded 30 Hz sequence, without padding or resampling."""
    if expected_length < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid episode metadata')
    arrays = {'episode0_state': np.asarray(state, dtype=np.float32),
        'episode0_action': np.asarray(action, dtype=np.float32),
        'episode0_timestamp': np.asarray(timestamp, dtype=np.float32),
        'episode0_frame_index': np.asarray(frame_index),
        'episode0_global_index': np.asarray(global_index)}
    for key, values in arrays.items():
        shape = (expected_length, 7) if key.endswith(('_state', '_action')) else (expected_length,)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'Invalid shape or nonfinite values: {key}')
    frames, indices = arrays['episode0_frame_index'], arrays['episode0_global_index']
    if (frames.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu'
            or not np.array_equal(frames, np.arange(expected_length))
            or not np.array_equal(indices, dataset_from_index + np.arange(expected_length))):
        raise ValueError('Episode frame/global indices are incomplete or out of order')
    timestamps = arrays['episode0_timestamp'].astype(np.float64)
    if not np.allclose(timestamps, frames / fps, rtol=0, atol=1e-5):
        raise ValueError('Recorded timestamps do not match the declared frame rate')
    return arrays


def load_episode0(dataset_dir):
    dataset_dir = Path(dataset_dir)
    info_path = dataset_dir / 'meta/info.json'
    info = json.loads(info_path.read_text(encoding='utf-8'))
    metadata_path = dataset_dir / 'meta/episodes/chunk-000/file-000.parquet'
    metadata_columns = ['episode_index', 'length', 'dataset_from_index', 'dataset_to_index',
                        'data/chunk_index', 'data/file_index']
    records = pq.read_table(metadata_path, columns=metadata_columns,
                            filters=[('episode_index', '=', 0)], use_threads=False).to_pylist()
    if len(records) != 1:
        raise ValueError('Exactly one episode 0 metadata row is required')
    record = records[0]
    if record['dataset_to_index'] - record['dataset_from_index'] != record['length']:
        raise ValueError('Episode length disagrees with its global index range')
    data_path = dataset_dir / info['data_path'].format(
        chunk_index=record['data/chunk_index'], file_index=record['data/file_index'])
    columns = ['observation.state', 'action', 'timestamp', 'frame_index', 'index']
    table = pq.read_table(data_path, columns=columns, filters=[('episode_index', '=', 0)],
                          use_threads=False)
    arrays = validate_episode(*(table[name].to_pylist() for name in columns),
        expected_length=record['length'], dataset_from_index=record['dataset_from_index'], fps=info['fps'])
    sources = {str(path): sha256(path) for path in (info_path, metadata_path, data_path)}
    return arrays, info, record, sources


def load_frame0(fixture_dir, provenance, arrays):
    """Use the already verified decoded RGB fixture at global/episode frame 0."""
    fixture_dir = Path(fixture_dir)
    source = json.loads((fixture_dir / 'provenance.json').read_text(encoding='utf-8'))
    for item in ('dataset', 'model'):
        for field in ('repo_id', 'revision'):
            if source[item][field] != provenance[item][field]:
                raise ValueError(f'Fixture {item} {field} differs from the recorded resource identity')
    selected = [index for index, (frame, episode) in enumerate(zip(
        source['frame_indices'], source['episode_indices'], strict=True)) if frame == 0 and episode == 0]
    if len(selected) != 1:
        raise ValueError('Fixture must contain exactly one original global frame 0 in episode 0')
    index = selected[0]
    with np.load(fixture_dir / 'observations.npz', allow_pickle=False) as fixture:
        state = fixture['state'][index].copy()
        images = {role: fixture[role][index].copy() for role in ('wrist', 'top')}
    if state.shape != (7,) or not np.isfinite(state).all() or not np.array_equal(state, arrays['episode0_state'][0]):
        raise ValueError('Fixture state does not exactly match the original Parquet frame 0')
    if any(pixels.shape != (3, 480, 640) or pixels.dtype != np.uint8 for pixels in images.values()):
        raise ValueError('Expected decoded RGB fixture pixels in CHW uint8 format')
    exported = {'frame0_state': state}
    exported.update({f'frame0_{role}_rgb': pixels.transpose(1, 2, 0).copy()
                     for role, pixels in images.items()})
    return exported, {'fixture_row': index, 'original_global_frame_index': 0,
        'original_episode_index': 0, 'original_episode_frame_index': 0,
        'image_conversion': source['image_conversion'],
        'fixture_provenance': source,
        'fixture_files_sha256': {name: sha256(fixture_dir / name)
                                 for name in ('observations.npz', 'provenance.json')}}


def verify_sources(bundle_dir, model_dir, fixture_dir, provenance):
    """Match local original weights/processors and fixture against the frozen bundle."""
    manifest_path = bundle_dir / 'bundle_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    published = manifest['models']['published']
    if published['revision'] != provenance['model']['revision']:
        raise ValueError('Published model revision does not match resource manifest')
    selected = {}
    prefix = published['directory'] + '/'
    for entry in manifest['files']:
        relative = entry['path']
        if relative.startswith(prefix):
            local = model_dir / relative[len(prefix):]
        elif relative in ('fixtures/observations.npz', 'fixtures/provenance.json'):
            local = fixture_dir / Path(relative).name
        else:
            continue
        actual = sha256(local)
        if actual != entry['sha256']:
            raise ValueError(f'Local source SHA256 mismatch: {local}')
        selected[str(local)] = actual
    required = [model_dir / name for name in (
        'config.json', 'model.safetensors', 'policy_preprocessor.json',
        'policy_postprocessor.json', 'policy_preprocessor_step_3_normalizer_processor.safetensors',
        'policy_postprocessor_step_0_unnormalizer_processor.safetensors')]
    if any(str(path) not in selected for path in required):
        raise ValueError('Frozen bundle is missing required original policy/processor identities')
    if selected[str(model_dir / 'model.safetensors')] != published['model_sha256']:
        raise ValueError('Original model identity disagrees with published model record')
    return {'bundle_manifest_sha256': sha256(manifest_path), 'verified_files_sha256': selected}


def predict_frame0(model_dir, frame):
    configure_torch()
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(0)
    config = ACTConfig.from_pretrained(model_dir, local_files_only=True)
    config.device, config.pretrained_backbone_weights, config.push_to_hub = 'cpu', None, False
    if (config.chunk_size != 100 or list(config.input_features['observation.state'].shape) != [7]
            or set(config.image_features) != {'observation.images.wrist', 'observation.images.top'}
            or any(list(value.shape) != [3, 480, 640] for value in config.image_features.values())):
        raise ValueError('Original ACT input/chunk contract differs from the export')
    policy = ACTPolicy.from_pretrained(model_dir, config=config, local_files_only=True, strict=True)
    policy.to('cpu').eval()
    preprocess, postprocess = make_pre_post_processors(config, pretrained_path=str(model_dir),
        preprocessor_overrides={'device_processor': {'device': 'cpu'}})
    observation = {'observation.state': torch.from_numpy(frame['frame0_state'].copy()).float()}
    for role in ('wrist', 'top'):
        observation['observation.images.' + role] = torch.from_numpy(
            frame[f'frame0_{role}_rgb'].copy()).permute(2, 0, 1).float() / 255
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = postprocess(policy.predict_action_chunk(preprocess(observation))).detach().cpu().numpy()
    elapsed_ms = (time.perf_counter() - started) * 1000
    if prediction.shape != (1, 100, 7) or not np.isfinite(prediction).all():
        raise ValueError('ACT prediction must contain exactly 100 finite seven-channel actions')
    return prediction[0], {'device': 'cpu', 'inference_runs': 1, 'warmup_runs': 0,
        'seed': 0, 'inference_ms': elapsed_ms, 'cpu_budget': CPU_BUDGET,
        'model_kind': 'published_original_not_finetuned',
        'preprocessing': 'original policy_preprocessor with CPU device override',
        'postprocessing': 'original policy_postprocessor including action unnormalization',
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'lerobot', 'numpy', 'pyarrow')}}


def export_inputs(output_dir):
    output_dir = Path(output_dir)
    outputs = ('inputs.npz', 'inputs_manifest.json', 'frame0_wrist.png', 'frame0_top.png')
    if any((output_dir / name).exists() for name in outputs):
        raise FileExistsError('Export artifacts already exist; preserve them and choose a new output directory')
    provenance = json.loads((ROOT / 'resources/resource_manifest.json').read_text(encoding='utf-8'))
    dataset_dir, model_dir = ROOT / 'data/rebot_cube', ROOT / 'models/act_rebot_cube'
    bundle_dir = ROOT / 'delivery/ACT_laptop_20261019_v2'
    fixture_dir = bundle_dir / 'fixtures'
    identity = verify_sources(bundle_dir, model_dir, fixture_dir, provenance)
    arrays, info, episode, dataset_hashes = load_episode0(dataset_dir)
    frame, frame_source = load_frame0(fixture_dir, provenance, arrays)
    prediction, inference = predict_frame0(model_dir, frame)
    arrays.update(frame)
    arrays['predicted_action_chunk'] = prediction
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'inputs.npz').open('xb') as target:
        np.savez_compressed(target, **arrays)
    from PIL import Image
    for role in ('wrist', 'top'):
        Image.fromarray(frame[f'frame0_{role}_rgb']).save(output_dir / f'frame0_{role}.png')
    times = arrays['episode0_timestamp'].astype(np.float64)
    manifest = {'schema': 'act_simulation_inputs_v1', 'status': 'exported_original_data_and_prediction',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': provenance['dataset'], 'model': provenance['model'],
        'dataset_local_files_sha256': dataset_hashes, 'source_verification': identity,
        'selected_frame': frame_source, 'episode': {'index': 0, 'frame_count': episode['length'],
            'dataset_from_index': episode['dataset_from_index'], 'dataset_to_index_exclusive': episode['dataset_to_index'],
            'fps': info['fps'], 'first_timestamp_s': float(times[0]), 'last_timestamp_s': float(times[-1]),
            'minimum_recorded_dt_s': float(np.diff(times).min()), 'maximum_recorded_dt_s': float(np.diff(times).max()),
            'resampled': False, 'padded': False},
        'channel_names': info['features']['action']['names'],
        'state_channel_names': info['features']['observation.state']['names'],
        'coordinates': 'Unmodified dataset coordinates. No mapping to URDF, MuJoCo, motors or radians.',
        'inference': inference,
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in arrays.items()},
        'artifacts': {name: {'sha256': sha256(output_dir / name), 'bytes': (output_dir / name).stat().st_size}
                       for name in outputs if name != 'inputs_manifest.json'},
        'robot_connected': False, 'robot_commands_sent': 0, 'camera_io': 0, 'training_steps': 0,
        'simulation_closed_loop': False, 'grasp_success_evaluated': False,
        'limitations': ['Episode 0 is from the published training data, not a generalization test.',
            'The single prediction is conditioned on recorded real RGB/state; it is not conditioned on simulated images.',
            'Only frame 0 RGB is exported; the full episode contains recorded state/action/timestamp values.',
            'Dataset source revision is recorded from the existing download manifest; current local Parquet hashes are included.']}
    with (output_dir / 'inputs_manifest.json').open('x', encoding='utf-8') as target:
        target.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/mujoco_validation_20261023')
    args = parser.parse_args()
    manifest = export_inputs(args.output_dir)
    print(json.dumps({'status': manifest['status'], 'output_directory': str(args.output_dir.resolve()),
        'episode_frames': manifest['episode']['frame_count'], 'prediction_shape': manifest['arrays']['predicted_action_chunk']['shape'],
        'robot_commands_sent': 0, 'training_steps': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
