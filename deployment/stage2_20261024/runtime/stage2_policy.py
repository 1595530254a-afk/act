"""Portable ACT ensemble1 observation-only handoff; no hardware control path.

Default mode replays bundled observations. Shadow mode accepts one JSON line
per captured state/RGB pair on stdin and writes predictions only. All paths
are resolved from the package root, independent of the desktop checkout.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
from cpu_budget import CPU_BUDGET, configure_torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / 'local/cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'local/cache/torch'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from temporal_ensemble import SparseTemporalEnsembler

MODEL_FILES = ('config.json', 'model.safetensors', 'policy_preprocessor.json',
    'policy_postprocessor.json', 'policy_preprocessor_step_3_normalizer_processor.safetensors',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors')
STATE_CHANNELS = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                  'wrist_flex.pos', 'wrist_yaw.pos', 'wrist_roll.pos', 'gripper.pos')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def package_path(relative):
    value = Path(relative)
    path = (ROOT / value).resolve()
    if value.is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError('Payload paths must be relative to the package root')
    return path


def verify_package():
    manifest = json.loads((ROOT / 'bundle_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('hardware_execution_enabled') is not False:
        raise ValueError('This runtime accepts observation-only packages')
    for item in manifest['files']:
        if digest(package_path(item['path'])) != item['sha256']:
            raise ValueError('Payload hash mismatch: ' + item['path'])
    return manifest


def load_observation(path):
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {'state', 'wrist', 'top'}:
            raise ValueError('NPZ must contain exactly state, wrist and top')
        arrays = {key: source[key].copy() for key in source.files}
    if (arrays['state'].shape != (7,) or arrays['state'].dtype.kind != 'f'
            or not np.isfinite(arrays['state']).all()):
        raise ValueError('state must be seven finite floating-point public coordinates')
    for role in ('wrist', 'top'):
        if arrays[role].shape != (480, 640, 3) or arrays[role].dtype != np.uint8:
            raise ValueError(role + ' must be RGB uint8 HWC [480,640,3]')
    return arrays


def load_policy(directory, requested_device):
    configure_torch()
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if requested_device == 'auto' else requested_device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for observation-only diagnostics')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = ACTConfig.from_pretrained(directory, local_files_only=True)
    config.device, config.pretrained_backbone_weights, config.push_to_hub = device, None, False
    if (config.chunk_size != 100 or list(config.input_features['observation.state'].shape) != [7]
            or set(config.image_features) != {'observation.images.wrist', 'observation.images.top'}
            or any(list(feature.shape) != [3, 480, 640] for feature in config.image_features.values())
            or list(config.output_features['action'].shape) != [7]):
        raise ValueError('Model does not match the two-RGB, seven-state,100x7 action contract')
    policy = ACTPolicy.from_pretrained(directory, config=config, local_files_only=True, strict=True).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(config, pretrained_path=str(directory),
        preprocessor_overrides={'device_processor': {'device': device}})
    identity = {'model_files_sha256': {name: digest(directory / name) for name in MODEL_FILES},
        'device_requested': requested_device, 'device_used': device, 'seed': 0,
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'lerobot', 'numpy', 'psutil')},
        'saved_processors_used': True, 'training_steps': 0, 'hardware_commands_sent': 0,
        'cuda_fallback_to_cpu': requested_device == 'auto' and device == 'cpu'}
    return (torch, policy, preprocess, postprocess), identity


def predict_chunk(runtime, arrays):
    torch, policy, preprocess, postprocess = runtime
    observation = {'observation.state': torch.from_numpy(arrays['state'].copy()).float()}
    for role in ('wrist', 'top'):
        observation['observation.images.' + role] = torch.from_numpy(arrays[role].copy()).permute(2, 0, 1).float() / 255
    started = time.perf_counter_ns()
    with torch.inference_mode():
        action = postprocess(policy.predict_action_chunk(preprocess(observation))).detach().cpu().numpy()
    elapsed = (time.perf_counter_ns() - started) / 1e6
    if action.shape != (1, 100, 7) or not np.isfinite(action).all():
        raise ValueError('ACT did not produce a finite100x7 public action chunk')
    return action[0].astype(np.float32, copy=False), elapsed


def save_npz(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def validate_request(request, next_frame):
    if not isinstance(request, dict) or set(request) - {'input', 'frame_index', 'capture_monotonic_ns', 'source', 'capture'}:
        raise ValueError('Request fields: input, frame_index, capture_monotonic_ns, source and capture')
    frame = request.get('frame_index')
    if type(frame) is not int or frame != next_frame:
        raise ValueError('Fresh observations must have consecutive frame_index starting at0; never duplicate cached frames')
    stamp = request.get('capture_monotonic_ns')
    if stamp is not None and (type(stamp) is not int or stamp <= 0 or stamp > time.monotonic_ns()):
        raise ValueError('capture_monotonic_ns must use this computer monotonic clock')
    path = Path(request['input'])
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def audit_ensemble(chunks, weighted, starts):
    """Independent raw-prediction reconstruction, with no neural forward."""
    starts = np.asarray(starts)
    for frame, actual in zip(starts, weighted, strict=True):
        active = np.flatnonzero((starts <= frame) & (frame < starts + 100))
        rows = frame - starts[active]
        weights = np.exp(-.01 * np.arange(len(active), dtype=np.float64))
        weights /= weights.sum()
        expected = np.sum(np.asarray(chunks)[active, rows].astype(np.float64) * weights[:, None], axis=0)
        if not np.array_equal(actual, expected):
            raise ValueError('Weighted action differs from raw chunks at frame ' + str(frame))


def run(args):
    manifest = verify_package()
    output = (ROOT / args.output_dir).resolve() if args.output_dir else ROOT / 'local' / (
        args.mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    if not output.is_relative_to(ROOT / 'local'):
        raise ValueError('Output must stay under this package local directory')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'mode': args.mode, 'dry_run': True,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
        'package_root': str(ROOT), 'cpu_budget': CPU_BUDGET, 'actual_neural_forwards': 0,
        'hardware_commands_sent': 0, 'camera_or_CAN_opened_by_package': False,
        'driver_or_existing_model_modified': False, 'wallclock_30hz_verified': False,
        'requested_observation_hz': 30, 'ensemble_coeff': .01,
        'source_coordinates': 'Saved model public-degree convention; hardware mapping is external and must not be inferred from simulator',
        'model_state_channels': list(STATE_CHANNELS), 'records': []}
    runtime, identity = None, None
    chunks, weighted, states, query_frames = [], [], [], []
    ensemble = SparseTemporalEnsembler()
    try:
        with redirect_stdout(sys.stderr):
            runtime, identity = load_policy(package_path(manifest['model_directory']), args.device)
        if identity['model_files_sha256'] != manifest['model_files_sha256']:
            raise ValueError('Runtime model/processor identity differs from package manifest')
        report['model_identity'] = identity
        print(json.dumps({'status': 'ready_dry_run_only', 'device': identity['device_used'],
                          'output': str(output), 'hardware_commands_sent': 0}), flush=True)
        if args.mode == 'offline':
            requests = ({'input': item['input'], 'frame_index': item['source_frame_index'], 'source': 'bundled_simulation_fixture'}
                        for index, item in enumerate(manifest['fixtures']))
        else:
            requests = (json.loads(line) for line in sys.stdin if line.strip())
        for request in requests:
            sequence = len(chunks)
            if args.mode == 'shadow' and request == {'stop': True}:
                break
            frame = request.get('frame_index')
            input_path = validate_request(request, sequence if args.mode == 'shadow' else frame)
            if args.mode == 'shadow':
                capture = request.get('capture', {})
                if (set(capture) != {'state', 'wrist', 'top'}
                        or capture['state'].get('measured') is not True
                        or capture['state'].get('state_order') != list(STATE_CHANNELS)
                        or capture['state'].get('units') != 'public_degree'
                        or not capture['state'].get('mapping_id')):
                    raise ValueError('Shadow needs measured-state mapping and both camera timestamp/serial metadata')
                for role in ('state', 'wrist', 'top'):
                    stamp = capture[role].get('receipt_monotonic_ns')
                    if type(stamp) is not int or stamp <= 0 or stamp > time.monotonic_ns():
                        raise ValueError('Missing local receipt_monotonic_ns for ' + role)
                if (not capture['wrist'].get('serial') or not capture['top'].get('serial')
                        or capture['wrist']['serial'] == capture['top']['serial']):
                    raise ValueError('Wrist and top must have distinct actual camera serials')
            started = time.monotonic_ns()
            arrays = load_observation(input_path)
            input_copy = output / f'observation_{sequence:06d}.npz'
            save_npz(input_copy, **arrays)
            raw, forward_ms = predict_chunk(runtime, arrays)
            raw_path = output / f'raw_chunk_{sequence:06d}.npz'
            save_npz(raw_path, action=raw)
            ensemble.add(frame, raw)
            action, provenance = ensemble.action(frame)
            chunks.append(raw)
            weighted.append(action)
            states.append(arrays['state'])
            query_frames.append(frame)
            record = {'frame_index': frame, 'source': request.get('source', 'external_read_only_capture'),
                'input_sha256': digest(input_copy), 'raw_chunk_sha256': digest(raw_path),
                'raw_chunk_file': raw_path.name, 'observation_file': input_copy.name,
                'forward_ms': forward_ms, 'weighted_public_action': action.tolist(),
                'sources': provenance, 'capture_monotonic_ns': request.get('capture_monotonic_ns'),
                'capture_metadata': request.get('capture'),
                'processing_started_monotonic_ns': started, 'hardware_commands_sent': 0}
            if args.mode == 'shadow':
                receipts = [request['capture'][role]['receipt_monotonic_ns'] for role in ('state', 'wrist', 'top')]
                record['input_ages_at_processing_ms'] = dict(zip(('state', 'wrist', 'top'),
                    [(started - stamp) / 1e6 for stamp in receipts], strict=True))
                record['receipt_timestamp_spread_ms'] = (max(receipts) - min(receipts)) / 1e6
                record['all_inputs_within_250ms_at_processing'] = all(0 <= started - stamp <= 250_000_000 for stamp in receipts)
            if args.mode == 'offline':
                # Reference output is read only after a fresh ACT forward.
                with np.load(package_path(manifest['fixtures'][sequence]['reference']), allow_pickle=False) as saved:
                    reference = saved['action'].copy()
                np.testing.assert_allclose(raw, reference, atol=.005, rtol=1e-4)
                record['reference_maximum_difference_degrees'] = float(np.max(np.abs(raw - reference)))
            finished = time.monotonic_ns()
            record.update({'processing_finished_monotonic_ns': finished,
                'total_processing_ms': (finished - started) / 1e6,
                'exceeded_33_333ms_processing_budget': finished - started > 1e9 / 30,
                'capture_to_finish_ms': None if request.get('capture_monotonic_ns') is None else
                    (finished - request['capture_monotonic_ns']) / 1e6})
            report['records'].append(record)
            report['actual_neural_forwards'] = len(chunks)
            with (output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            print(json.dumps({'frame_index': frame, 'forward_ms': forward_ms,
                'total_processing_ms': record['total_processing_ms'], 'dry_run': True,
                'hardware_commands_sent': 0}), flush=True)
        if not chunks:
            raise ValueError('No observations processed')
        audit_ensemble(chunks, weighted, query_frames)
        save_npz(output / 'action_trace.npz', action_chunks=np.stack(chunks),
                 ensembled_public_action=np.stack(weighted), observation_state=np.stack(states),
                 query_frame_index=np.asarray(query_frames))
        report['status'] = 'passed_offline_reproduction' if args.mode == 'offline' else 'completed_observation_only_shadow'
        report['all_weighted_actions_recomputed_from_saved_raw_chunks'] = True
        times = [record['total_processing_ms'] for record in report['records']]
        report['total_processing_ms'] = {'mean': float(np.mean(times)), 'p95': float(np.percentile(times, 95)), 'max': max(times)}
        report['processing_budget_misses'] = sum(record['exceeded_33_333ms_processing_budget'] for record in report['records'])
        report['processing_timer_excludes_jsonl_and_stdout'] = True
        if len(chunks) > 1:
            starts_ns = np.asarray([record['processing_started_monotonic_ns'] for record in report['records']], dtype=np.int64)
            intervals_ms = np.diff(starts_ns) / 1e6
            report['actual_query_start_interval_ms'] = {'mean': float(intervals_ms.mean()),
                'p95': float(np.percentile(intervals_ms, 95)), 'max': float(intervals_ms.max())}
            report['observed_query_start_rate_hz'] = float((len(chunks) - 1) * 1e9 / (starts_ns[-1] - starts_ns[0]))
            report['query_interval_includes_capture_wait_and_logging'] = True
        report['not_a_hardware_grasp_or_30hz_acceptance'] = True
    except Exception as error:
        report.update({'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if identity:
            report['model_files_unchanged'] = {name: digest(package_path(manifest['model_directory']) / name)
                                              for name in MODEL_FILES} == identity['model_files_sha256']
        (output / 'return_report.json').write_text(json.dumps(report, ensure_ascii=False,
            indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('offline', 'shadow'), default='offline')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output-dir', type=Path)
    return parser.parse_args(argv)


if __name__ == '__main__':
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    run(parse_args())
