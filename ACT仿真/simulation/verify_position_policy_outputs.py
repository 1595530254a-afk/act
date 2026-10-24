"""Recompute all saved ACT action chunks without simulation or training.

Uses the ACT virtualenv and the actual saved observation/processor/model files.
Recorded actions are loaded only after each new neural-network prediction, for
comparison; they are never passed to the model as an input.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import numpy as np

from policy_worker import MODEL_FILES, load_observation, load_policy, predict_chunk

DEFAULT_OUTPUT = ROOT / 'runs/mujoco_validation_20261023/position_generalization_20261023'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(output):
    report_path, protocol_path = output / 'generalization_report.json', output / 'protocol.json'
    report, protocol = read_json(report_path), read_json(protocol_path)
    require(report['status'] == 'completed_frozen_position_evaluation', 'Evaluation is not completed')
    source_hashes = {'generalization_report.json': digest(report_path), 'protocol.json': digest(protocol_path)}
    require(source_hashes['protocol.json'] == report['protocol_sha256']
            and protocol == report['protocol'], 'Preregistered protocol changed')
    require(len(report['results']) == 6, 'Expected six recorded trials')
    model_directory = Path(protocol['model_directory'])
    before = {name: digest(model_directory / name) for name in MODEL_FILES}
    require(before == protocol['model_files_sha256'], 'Frozen model files changed')
    require(digest(ROOT / 'simulation/policy_worker.py') ==
            protocol['code_files_sha256']['simulation/policy_worker.py'], 'Policy worker code changed')
    started = time.perf_counter()
    runtime, identity = load_policy(model_directory)
    require(identity['files_sha256'] == before, 'Loaded a different model')
    torch, policy, _, _ = runtime
    comparisons = []
    for entry in report['results']:
        trial = entry['id']
        # predict_action_chunk bypasses the action queue and is stateless. A
        # reset/seed also matches the fresh worker used in each original trial.
        policy.reset()
        torch.manual_seed(0)
        directory = output / 'trials' / trial / 'custom'
        record = read_json(directory / 'result.json')
        require(record == entry['result'], f'{trial}: original result changed')
        require(len(record['inference']) == 5, f'{trial}: expected five observations')
        for index, inference in enumerate(record['inference']):
            observation_path = directory / f'observation_{index:03d}.npz'
            expected_path = directory / f'chunk_{index:03d}.npz'
            observation_hash = digest(observation_path)
            require(observation_hash == inference['input_sha256'], f'{trial}/{index}: input changed')
            observation = load_observation(observation_path)
            # This is a fresh network forward. No recorded action is loaded
            # until after predict_chunk has returned the newly computed result.
            recomputed, elapsed_ms = predict_chunk(runtime, observation)
            output_hash = digest(expected_path)
            require(output_hash == inference['output_sha256'], f'{trial}/{index}: recorded output changed')
            with np.load(expected_path, allow_pickle=False) as saved:
                require(set(saved.files) == {'action'}, f'{trial}/{index}: output contract')
                expected = saved['action'].copy()
            require(recomputed.shape == expected.shape == (100, 7), f'{trial}/{index}: action shape')
            require(np.isfinite(recomputed).all() and np.isfinite(expected).all(), f'{trial}/{index}: nonfinite actions')
            error = np.abs(recomputed.astype(np.float64) - expected.astype(np.float64))
            comparison = {
                'trial': trial, 'observation_index': index,
                'source_input_sha256': observation_hash, 'source_chunk_npz_sha256': output_hash,
                'recomputed_action_array_sha256': array_digest(recomputed),
                'recorded_action_array_sha256': array_digest(expected),
                'maximum_absolute_difference_degrees': float(error.max()),
                'mean_absolute_difference_degrees': float(error.mean()),
                'exactly_equal': bool(np.array_equal(recomputed, expected)),
                'within_tolerance': bool(np.allclose(recomputed, expected, atol=.005, rtol=1e-4)),
                'fresh_inference_ms': elapsed_ms,
                'policy_input_fields': sorted(observation),
                'action_values_compared': int(expected.size),
            }
            comparisons.append(comparison)
            print(json.dumps({'trial': trial, 'observation': index,
                'max_abs_diff_deg': comparison['maximum_absolute_difference_degrees'],
                'exact': comparison['exactly_equal']}, ensure_ascii=False), flush=True)
    after = {name: digest(model_directory / name) for name in MODEL_FILES}
    after_sources = {'generalization_report.json': digest(report_path), 'protocol.json': digest(protocol_path)}
    require(before == after and source_hashes == after_sources, 'Model/protocol/original report changed')
    require(len(comparisons) == 30, 'Expected thirty new neural-network calls')
    passed = all(comparison['within_tolerance'] for comparison in comparisons)
    result = {
        'status': 'passed_fresh_ACT_output_reproduction' if passed else 'failed_ACT_output_reproduction',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'verification_script_sha256': digest(__file__),
        'elapsed_wall_s': time.perf_counter() - started,
        'new_neural_network_forward_calls': len(comparisons),
        'new_rollouts': 0, 'new_physics_steps': 0, 'training_steps': 0, 'hardware_commands_sent': 0,
        'total_action_values_compared': sum(comparison['action_values_compared'] for comparison in comparisons),
        'exact_matches': sum(comparison['exactly_equal'] for comparison in comparisons),
        'maximum_absolute_difference_degrees': max(comparison['maximum_absolute_difference_degrees'] for comparison in comparisons),
        'tolerance': {'atol_degrees': .005, 'rtol': 1e-4},
        'model_identity': identity, 'model_files_sha256_before': before, 'model_files_sha256_after': after,
        'frozen_model_files_unchanged': True,
        'source_hashes_before': source_hashes, 'source_hashes_after': after_sources,
        'protocol_and_original_report_unchanged': True,
        'comparisons': comparisons, 'cpu_budget': CPU_BUDGET,
        'method': 'Reload frozen ACT and saved preprocessing/postprocessing; run a fresh predict_action_chunk on every saved state/wrist/top observation; only then load recorded actions and compare all 100x7 values.',
        'conclusion': ('All thirty recorded action chunks can be reproduced by fresh inference with the frozen ACT model and original observations.' if passed else
                       'At least one saved action chunk could not be reproduced within the specified tolerance; investigate before relying on the execution claim.'),
        'scope': 'Independent numerical evidence of ACT output provenance; not a new grasp trial or proof of successful generalization.',
    }
    target = output / 'policy_output_reproduction.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    require(passed, 'At least one newly computed output differs from the recorded action chunk')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify(args.output_dir.resolve())
    print(json.dumps({'status': result['status'], 'calls': result['new_neural_network_forward_calls'],
        'exact_matches': result['exact_matches'],
        'max_difference_degrees': result['maximum_absolute_difference_degrees']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
