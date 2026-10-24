"""Recompute every recorded CUDA ACT chunk from saved RGB/state, no physics.

Recorded action arrays are opened only after each new neural forward returns.
This verifies output provenance; it is not another physical evaluation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET
from policy_worker import MODEL_FILES, load_observation, predict_chunk, sha256
from policy_worker_cuda_20261024 import load_policy
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    directory = directory.resolve()
    destination = directory / 'fresh_cuda_output_reproduction.json'
    if destination.exists():
        raise FileExistsError('Preserve the previous reproduction report')
    evaluation_path, protocol_path = directory / 'evaluation.json', directory / 'protocol.json'
    report = json.loads(evaluation_path.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    require(report['status'] == 'completed_frozen_feedback_ACT_evaluation'
            and report['protocol'] == protocol, 'Requires a complete frozen feedback evaluation')
    entries, positions = report['results'], protocol['positions']
    require(len(entries) == len(positions) and len(entries) in (1, 4, 8)
            and [row['id'] for row in entries] == [row['id'] for row in positions],
            'Trial IDs differ from the entire declared position list')
    every = {'chunk100': 100, 'ensemble1': 1}[protocol['mode']]
    query_frames = list(range(0, 450, every))
    require(protocol['expected_ACT_queries_per_trial'] == len(query_frames), 'Wrong query count')
    sources = {str(evaluation_path): sha256(evaluation_path), str(protocol_path): sha256(protocol_path),
               str(Path(__file__).resolve()): sha256(__file__)}
    for filename, expected in protocol['source_files_sha256'].items():
        require(sha256(filename) == expected, f'Frozen source changed: {filename}')
        sources[filename] = expected
    for filename, expected in protocol['code_files_sha256'].items():
        path = ROOT / filename
        require(sha256(path) == expected, f'Frozen code changed: {filename}')
        sources[str(path)] = expected
    model_dir = Path(protocol['model_directory'])
    before = {name: sha256(model_dir / name) for name in MODEL_FILES}
    require(before == protocol['model_files_sha256'], 'Frozen weights or processors changed')
    started = time.perf_counter()
    runtime, identity = load_policy(model_dir)
    require(identity['files_sha256'] == before and identity['device'] == 'cuda',
            'Unexpected model or inference device')
    comparisons, artifacts = [], {}
    for entry in entries:
        trial = entry['id']
        require(re.fullmatch(r'[A-Za-z0-9_+\-]+', trial) is not None, 'Invalid trial ID')
        custom = directory / 'trials' / trial / 'custom'
        record_path, identity_path = custom / 'result.json', custom / 'model_identity.json'
        record = json.loads(record_path.read_text(encoding='utf-8'))
        worker_identity = json.loads(identity_path.read_text(encoding='utf-8'))
        require(record == entry['result'] and record['status'] == 'completed'
                and record['policy_actions_applied'] == 450
                and record['actual_neural_forward_calls'] == len(query_frames)
                and len(record['inference']) == len(query_frames), 'Incomplete trial or changed result')
        require(worker_identity['model_identity']['files_sha256'] == before
                and worker_identity['model_identity']['device'] == 'cuda', 'Original worker differs')
        artifacts[str(record_path)], artifacts[str(identity_path)] = sha256(record_path), sha256(identity_path)
        runtime[1].reset()
        runtime[0].manual_seed(0)
        for index, (frame, inference) in enumerate(zip(query_frames, record['inference'])):
            require(inference['frame_index'] == frame and inference['request_index'] == index + 1,
                    'Saved query timeline differs')
            observation_path = custom / f'observation_{index:03d}.npz'
            expected_path = custom / f'chunk_{index:03d}.npz'
            observation_hash = sha256(observation_path)
            require(observation_hash == inference['input_sha256'], 'Input artifact changed')
            observation = load_observation(observation_path)
            recomputed, elapsed_ms = predict_chunk(runtime, observation)
            # The saved expected action is neither read nor hashed until this forward returns.
            expected_hash = sha256(expected_path)
            require(expected_hash == inference['output_sha256'], 'Output artifact changed')
            with np.load(expected_path, allow_pickle=False) as saved:
                require(set(saved.files) == {'action'}, 'Unexpected saved output fields')
                expected = saved['action'].copy()
            require(recomputed.shape == expected.shape == (100, 7)
                    and np.isfinite(expected).all(), 'Invalid output shape or values')
            difference = np.abs(recomputed.astype(np.float64) - expected.astype(np.float64))
            comparisons.append({'trial': trial, 'query_index': index, 'action_frame_index': frame,
                'maximum_absolute_difference_degrees': float(difference.max()),
                'mean_absolute_difference_degrees': float(difference.mean()),
                'exactly_equal': bool(np.array_equal(recomputed, expected)),
                'within_tolerance': bool(np.allclose(recomputed, expected, atol=.005, rtol=1e-4)),
                'source_input_sha256': observation_hash, 'source_output_sha256': expected_hash,
                'policy_input_fields': sorted(observation), 'fresh_inference_ms': elapsed_ms,
                'saved_actions_opened_only_after_fresh_forward': True, 'values_compared': 700})
            artifacts[str(observation_path)], artifacts[str(expected_path)] = observation_hash, expected_hash
        print(json.dumps({'trial': trial, 'fresh_forwards': len(query_frames),
            'maximum_difference_degrees': max(row['maximum_absolute_difference_degrees']
                for row in comparisons if row['trial'] == trial)}, ensure_ascii=False), flush=True)
    after = {name: sha256(model_dir / name) for name in MODEL_FILES}
    require(before == after and all(sha256(path) == value for path, value in sources.items()),
            'Frozen model, processors, protocol, evaluation or code changed during reproduction')
    require(all(sha256(path) == value for path, value in artifacts.items()), 'Trial artifacts changed')
    require(len(comparisons) == len(entries) * len(query_frames), 'Unexpected fresh forward count')
    passed = all(row['within_tolerance'] for row in comparisons)
    result = {'status': 'passed_fresh_CUDA_ACT_output_reproduction' if passed else 'failed_output_reproduction',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'evaluation_directory': str(directory),
        'mode': protocol['mode'], 'split': protocol['split'], 'model_identity': identity,
        'completed_positions': len(entries), 'new_neural_forward_calls': len(comparisons),
        'all_recorded_chunks_recomputed': True, 'action_values_compared': len(comparisons) * 700,
        'all_exactly_equal': all(row['exactly_equal'] for row in comparisons),
        'maximum_absolute_difference_degrees': max(row['maximum_absolute_difference_degrees'] for row in comparisons),
        'tolerance': {'atol_degrees': .005, 'rtol': 1e-4}, 'comparisons': comparisons,
        'model_files_before': before, 'model_files_after': after, 'source_files_sha256': sources,
        'artifact_files_sha256': artifacts, 'source_and_artifacts_unchanged': True,
        'new_rollouts': 0, 'new_physics_steps': 0, 'training_steps': 0, 'hardware_commands': 0,
        'elapsed_wall_s': time.perf_counter() - started, 'cpu_budget': CPU_BUDGET,
        'scope': 'Numerical output provenance only; not another grasp or generalization test.'}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    require(passed, 'At least one recorded ACT chunk differs from its fresh prediction')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    result = verify(parser.parse_args().output_dir)
    print(json.dumps({'status': result['status'], 'calls': result['new_neural_forward_calls'],
        'maximum_difference_degrees': result['maximum_absolute_difference_degrees']}, ensure_ascii=False))
