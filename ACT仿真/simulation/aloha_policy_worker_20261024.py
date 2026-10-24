"""Resident official ALOHA ACT worker, separate from the seven-axis B601 worker.

Requests are JSON lines with absolute input/output NPZ paths. Observation NPZ:
state float[14] in gym-aloha coordinates and top RGB uint8[480,640,3]. Each
request calls predict_action_chunk freshly and returns unnormalized action
float[100,14]. No B601 joint conversion, simulation, training, or hardware I/O.
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

import numpy as np

MODEL_FILES = ('config.json', 'model.safetensors', 'policy_preprocessor.json',
    'policy_postprocessor.json', 'policy_preprocessor_step_3_normalizer_processor.safetensors',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def absolute_npz(value):
    if not isinstance(value, str) or not value:
        raise ValueError('NPZ paths must be nonempty absolute strings')
    path = Path(value)
    if not path.is_absolute() or path.suffix.lower() != '.npz':
        raise ValueError('An absolute .npz path is required')
    return path.resolve()


def validate_observation(arrays):
    if set(arrays) != {'state', 'top'}:
        raise ValueError('ALOHA input must contain exactly state and top')
    state, pixels = arrays['state'], arrays['top']
    if state.shape != (14,) or state.dtype.kind != 'f' or not np.isfinite(state).all():
        raise ValueError('ALOHA state must contain fourteen finite floating-point values')
    if pixels.shape != (480, 640, 3) or pixels.dtype != np.uint8:
        raise ValueError('top must be RGB uint8[480,640,3]')
    return {name: np.ascontiguousarray(value) for name, value in arrays.items()}


def load_policy(directory, device):
    configure_torch()
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is unavailable')
    hashes = {name: digest(directory / name) for name in MODEL_FILES}
    config = ACTConfig.from_pretrained(directory, local_files_only=True)
    config.device, config.pretrained_backbone_weights, config.push_to_hub = device, None, False
    if (config.chunk_size != 100 or config.n_action_steps != 100
            or list(config.input_features['observation.state'].shape) != [14]
            or set(config.image_features) != {'observation.images.top'}
            or list(config.image_features['observation.images.top'].shape) != [3, 480, 640]
            or list(config.output_features['action'].shape) != [14]):
        raise ValueError('Checkpoint differs from the official single-camera 14-channel ALOHA contract')
    torch.manual_seed(0)
    policy = ACTPolicy.from_pretrained(directory, config=config, local_files_only=True, strict=True)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=str(directory),
        preprocessor_overrides={'device_processor': {'device': device},
                                'normalizer_processor': {'device': device}},
        postprocessor_overrides={'unnormalizer_processor': {'device': device},
                                 'device_processor': {'device': 'cpu'}})
    identity = {'directory': str(directory), 'files_sha256': hashes,
        'model_sha256': hashes['model.safetensors'], 'device': device,
        'chunk_size': 100, 'n_action_steps': 100, 'state_shape': [14], 'cameras': ['top'],
        'coordinate_system': 'Native gym-aloha state/action; no B601 angle or gripper mapping',
        'policy_call': 'predict_action_chunk per request, no cached action queue',
        'source_revision': 'ba73b2766f1371cdc133ca4efb97eb090d744625',
        'seed': 0, 'warmup_calls': 0, 'training_steps': 0, 'hardware_io': 0,
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'lerobot', 'numpy')},
        'cpu_budget': CPU_BUDGET}
    return (torch, policy, pre, post), identity


def predict_chunk(runtime, arrays):
    torch, policy, pre, post = runtime
    observation = {'observation.state': torch.from_numpy(arrays['state'].copy()).float(),
        'observation.images.top': torch.from_numpy(arrays['top'].copy()).permute(2, 0, 1).float() / 255}
    started = time.perf_counter()
    with torch.inference_mode():
        actions = post(policy.predict_action_chunk(pre(observation))).detach().cpu().numpy()
    elapsed_ms = (time.perf_counter() - started) * 1000
    if actions.shape != (1, 100, 14) or not np.isfinite(actions).all():
        raise ValueError('Expected one finite ACT chunk[100,14]')
    return actions[0].astype(np.float32, copy=False), elapsed_ms


def handle_request(runtime, request, index):
    if not isinstance(request, dict) or set(request) != {'input', 'output'}:
        raise ValueError('Request must contain exactly input/output paths')
    input_path, output_path = (absolute_npz(request[key]) for key in ('input', 'output'))
    if input_path == output_path or not output_path.parent.is_dir():
        raise ValueError('Output must differ from input and have an existing parent directory')
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = validate_observation({name: archive[name].copy() for name in archive.files})
    action, elapsed_ms = predict_chunk(runtime, arrays)
    temporary = output_path.with_name(output_path.name + '.' + uuid.uuid4().hex + '.pending')
    with temporary.open('xb') as stream:
        np.savez(stream, action=action)
    temporary.replace(output_path)
    return {'status': 'ok', 'request_index': index, 'input': str(input_path),
        'input_sha256': digest(input_path), 'output': str(output_path),
        'output_sha256': digest(output_path), 'action_shape': list(action.shape),
        'inference_ms': elapsed_ms, 'fresh_ACT_forward_calls': 1,
        'coordinate_system': 'Unnormalized native gym-aloha action',
        'training_steps': 0, 'hardware_io': 0}


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            runtime, identity = load_policy(args.model_dir.resolve(), args.device)
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
            emit({'status': 'error', 'request_index': index, 'error_type': type(error).__name__,
                  'error': str(error), 'output_written': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
