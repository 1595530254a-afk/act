"""Resident CUDA ACT worker for the 2026-10-24 simulation iteration only.

Reuse the existing strict RGB/state and artifact protocol. Every request makes
a fresh predict_action_chunk call; no robot SDK, action queue, or training.
The earlier CPU worker and its frozen experiment records remain unchanged.
"""
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import policy_worker as cpu_worker


def load_policy(model_dir):
    runtime, identity = cpu_worker.load_policy(model_dir)
    torch, policy, _, _ = runtime
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this worker; no silent CPU fallback')
    from lerobot.policies.factory import make_pre_post_processors

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    policy.config.device = 'cuda'
    policy.to('cuda').eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=str(model_dir),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    torch.cuda.synchronize()
    identity.update({
        'device': 'cuda', 'gpu': torch.cuda.get_device_name(),
        'preprocessing': 'Saved processor statistics with CUDA device override',
        'precision': 'float32; CUDA matmul and cuDNN TF32 disabled',
        'worker_variant': 'cuda_fresh_chunk_20261024',
        'real_time_throughput_measured': False,
        'query_semantics': 'Every request calls predict_action_chunk; no cached action dequeue',
    })
    return (torch, policy, preprocess, postprocess), identity


def main():
    import argparse
    import json

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            runtime, identity = load_policy(args.model_dir.resolve())
        cpu_worker.emit({'status': 'ready', 'model_identity': identity})
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        cpu_worker.emit({'status': 'startup_error', 'error_type': type(error).__name__, 'error': str(error)})
        return 1
    for index, line in enumerate(sys.stdin, start=1):
        try:
            request = json.loads(line)
            if request == {'stop': True}:
                cpu_worker.emit({'status': 'stopped', 'requests_received': index - 1})
                return 0
            with redirect_stdout(sys.stderr):
                response = cpu_worker.handle_request(runtime, request, index)
            cpu_worker.emit(response)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            cpu_worker.emit({'status': 'error', 'request_index': index,
                'error_type': type(error).__name__, 'error': str(error), 'output_written': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
