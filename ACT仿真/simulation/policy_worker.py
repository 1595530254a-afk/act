"""Resident CPU ACT inference for the MuJoCo experiment; no robot I/O.

Run with the ACT virtualenv, pass --model-dir, then send one JSON object per
stdin line. Requests contain absolute input/output NPZ paths. Input arrays are
state (7 public degrees), wrist and top (480x640x3 RGB uint8). The output array
action contains one unnormalized 100x7 public-degree action chunk. {"stop": true}
exits cleanly. Stdout is reserved for the JSON protocol; diagnostics use stderr.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET, configure_torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache/torch'))

import numpy as np


MODEL_FILES = (
    'config.json', 'model.safetensors', 'policy_preprocessor.json',
    'policy_postprocessor.json',
    'policy_preprocessor_step_3_normalizer_processor.safetensors',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors',
)


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def absolute_npz(value):
    if not isinstance(value, str) or not value:
        raise ValueError('NPZ paths must be nonempty absolute strings')
    path = Path(value)
    if not path.is_absolute() or path.suffix.lower() != '.npz':
        raise ValueError('An absolute .npz path is required')
    return path.resolve()


def validate_observation(arrays):
    if set(arrays) != {'state', 'wrist', 'top'}:
        raise ValueError('Input must contain exactly state, wrist and top')
    state = arrays['state']
    if state.shape != (7,) or state.dtype.kind != 'f' or not np.isfinite(state).all():
        raise ValueError('state must be seven finite floating-point public degrees')
    for role in ('wrist', 'top'):
        pixels = arrays[role]
        if pixels.shape != (480, 640, 3) or pixels.dtype != np.uint8:
            raise ValueError(f'{role} must be RGB uint8 with shape (480, 640, 3)')
    return {key: np.ascontiguousarray(value) for key, value in arrays.items()}


def load_observation(path):
    with np.load(path, allow_pickle=False) as source:
        return validate_observation({key: source[key].copy() for key in source.files})


def load_policy(model_dir):
    configure_torch()
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    hashes = {name: sha256(model_dir / name) for name in MODEL_FILES}
    torch.manual_seed(0)
    config = ACTConfig.from_pretrained(model_dir, local_files_only=True)
    config.device, config.pretrained_backbone_weights, config.push_to_hub = 'cpu', None, False
    if (config.chunk_size != 100
            or list(config.input_features['observation.state'].shape) != [7]
            or set(config.image_features) != {'observation.images.wrist', 'observation.images.top'}
            or any(list(feature.shape) != [3, 480, 640] for feature in config.image_features.values())
            or list(config.output_features['action'].shape) != [7]):
        raise ValueError('Policy differs from the seven-channel, two-camera ACT contract')
    policy = ACTPolicy.from_pretrained(model_dir, config=config, local_files_only=True, strict=True)
    policy.to('cpu').eval()
    preprocess, postprocess = make_pre_post_processors(config, pretrained_path=str(model_dir),
        preprocessor_overrides={'device_processor': {'device': 'cpu'}})
    identity = {'directory': str(model_dir), 'files_sha256': hashes,
        'model_sha256': hashes['model.safetensors'], 'device': 'cpu', 'chunk_size': 100,
        'original_n_action_steps': config.n_action_steps, 'seed': 0, 'warmup_runs': 0,
        'preprocessing': 'Saved policy_preprocessor with CPU device override',
        'postprocessing': 'Saved policy_postprocessor with action unnormalization',
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'lerobot', 'numpy')},
        'cpu_budget': CPU_BUDGET, 'training_steps': 0, 'hardware_io': 0}
    return (torch, policy, preprocess, postprocess), identity


def predict_chunk(runtime, arrays):
    torch, policy, preprocess, postprocess = runtime
    observation = {'observation.state': torch.from_numpy(arrays['state'].copy()).float()}
    for role in ('wrist', 'top'):
        observation['observation.images.' + role] = torch.from_numpy(
            arrays[role].copy()).permute(2, 0, 1).float() / 255
    started = time.perf_counter()
    with torch.inference_mode():
        result = postprocess(policy.predict_action_chunk(preprocess(observation)))
        actions = result.detach().cpu().numpy()
    elapsed_ms = (time.perf_counter() - started) * 1000
    if actions.shape != (1, 100, 7) or not np.isfinite(actions).all():
        raise ValueError('Inference must produce exactly 100 finite seven-channel actions')
    return actions[0].astype(np.float32, copy=False), elapsed_ms


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def handle_request(runtime, request, request_index):
    if not isinstance(request, dict) or set(request) != {'input', 'output'}:
        raise ValueError('Request must contain exactly input and output paths')
    input_path, output_path = (absolute_npz(request[key]) for key in ('input', 'output'))
    if input_path == output_path:
        raise ValueError('Input and output must be different files')
    if not output_path.parent.is_dir():
        raise ValueError('Output parent directory does not exist')
    arrays = load_observation(input_path)
    input_hash = sha256(input_path)
    actions, elapsed_ms = predict_chunk(runtime, arrays)
    # Publish only a complete, freshly computed chunk; no partially written NPZ.
    temporary = output_path.with_name(output_path.name + '.' + uuid.uuid4().hex + '.pending')
    with temporary.open('xb') as target:
        np.savez(target, action=actions)
    temporary.replace(output_path)
    return {'status': 'ok', 'request_index': request_index,
        'input': str(input_path), 'input_sha256': input_hash,
        'output': str(output_path), 'output_sha256': sha256(output_path),
        'action_shape': list(actions.shape), 'inference_ms': elapsed_ms,
        'action_coordinates': 'Unnormalized public degrees in saved policy channel order',
        'training_steps': 0, 'hardware_io': 0}


def main():
    # Windows pipe defaults can be GBK; the parent protocol always uses UTF-8.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            runtime, identity = load_policy(args.model_dir.resolve())
        emit({'status': 'ready', 'model_identity': identity})
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({'status': 'startup_error', 'error_type': type(error).__name__, 'error': str(error)})
        return 1
    for index, line in enumerate(sys.stdin, start=1):
        try:
            request = json.loads(line)
            if request == {'stop': True}:
                emit({'status': 'stopped', 'requests_received': index - 1})
                return 0
            with redirect_stdout(sys.stderr):
                response = handle_request(runtime, request, index)
            emit(response)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({'status': 'error', 'request_index': index,
                'error_type': type(error).__name__, 'error': str(error), 'output_written': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
