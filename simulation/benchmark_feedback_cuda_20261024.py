"""Check actual CUDA ACT IPC and timing using an existing saved RGB/state input."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import numpy as np

from policy_process_cuda_20261024 import CudaPolicyProcess
from run_task import digest, save_json


def main():
    model = ROOT / 'runs/mujoco_iteration_20261023/multisite_v1/adapt_v1/checkpoints/step_000600/pretrained_model'
    source = ROOT / 'runs/mujoco_iteration_20261023/multisite_v1/validation_v1_step600/trials/val_-10_-10/custom'
    output = ROOT / 'runs/mujoco_iteration_20261024/cuda_feedback_preflight_v1'
    output.mkdir(parents=True, exist_ok=False)
    input_path, cpu_path = source / 'observation_000.npz', source / 'chunk_000.npz'
    source_hashes = {str(path): digest(path) for path in (input_path, cpu_path, model / 'model.safetensors')}
    with np.load(input_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    with np.load(cpu_path, allow_pickle=False) as saved:
        cpu_action = saved['action'].copy()
    worker = None
    started = time.perf_counter()
    try:
        worker = CudaPolicyProcess(model, output)
        identity = worker.identity
        timings, first, repeat_difference = [], None, 0.
        for index in range(21):
            action, timing = worker.predict(arrays['state'], arrays['wrist'], arrays['top'], index)
            if first is None:
                first = action.copy()
            else:
                repeat_difference = max(repeat_difference, float(np.max(np.abs(action - first))))
            timings.append(timing)
        measured = timings[1:]
        if not np.isfinite(first).all() or repeat_difference != 0:
            raise ValueError('Repeated identical CUDA observations produced unstable action chunks')
        report = {'status': 'actual_cuda_fresh_forward_preflight_passed',
            'model_identity': identity, 'source_files_sha256': source_hashes,
            'actual_ACT_queries': 21, 'warmup_queries_excluded_from_summary': 1,
            'comparison_with_saved_CPU': {
                'maximum_absolute_action_difference_degrees': float(np.max(np.abs(first - cpu_action))),
                'mean_absolute_action_difference_degrees': float(np.mean(np.abs(first - cpu_action))),
                'scope': 'Same old validation input and checkpoint; a CPU/CUDA numerical comparison, not a physics test'},
            'repeated_CUDA_max_difference_degrees': repeat_difference,
            'forward_ms_mean': float(np.mean([row['inference_ms'] for row in measured])),
            'forward_ms_p95': float(np.percentile([row['inference_ms'] for row in measured], 95)),
            'client_roundtrip_ms_mean': float(np.mean([row['client_roundtrip_ms'] for row in measured])),
            'client_roundtrip_ms_p95': float(np.percentile([row['client_roundtrip_ms'] for row in measured], 95)),
            'timings': timings, 'elapsed_wall_s': time.perf_counter() - started,
            'renderer_and_physics_included': False,
            'real_time_simulation_throughput_established': False,
            'training_steps': 0, 'hardware_commands_sent': 0, 'cpu_budget': CPU_BUDGET}
        if any(digest(Path(path)) != expected for path, expected in source_hashes.items()):
            raise ValueError('Frozen source files changed during the preflight')
        save_json(output / 'preflight.json', report)
        print(json.dumps({key: value for key, value in report.items()
                         if key not in ('timings', 'model_identity', 'source_files_sha256')}, ensure_ascii=False))
    finally:
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
