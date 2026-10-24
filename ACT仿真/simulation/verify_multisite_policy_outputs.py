"""Reproduce completed multisite ACT outputs with fresh CPU neural forwards.

Only recorded state/wrist/top observations are provided to ACT. Each saved
action chunk is opened for comparison only after its new prediction returns.
This performs no MuJoCo import/step, new rollout, training, or robot command.
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

from verify_position_policy_outputs import (MODEL_FILES, array_digest, digest,
    load_observation, load_policy, predict_chunk, read_json, require)
import numpy as np


def verify(output):
    output = output.resolve()
    target = output / 'policy_output_reproduction.json'
    if target.exists():
        raise FileExistsError('Preserve the existing reproduction report; overwriting is not allowed')
    report_path, protocol_path = output / 'evaluation.json', output / 'protocol.json'
    report, protocol = read_json(report_path), read_json(protocol_path)
    require(report['status'] == 'completed_frozen_multisite_ACT_evaluation',
            'Requires a completed frozen multisite evaluation')
    require(report['protocol'] == protocol, 'Saved evaluation and preregistered protocol differ')
    positions, entries = protocol['positions'], report['results']
    require(len(entries) in (4, 5, 6) and len(entries) == len(positions),
            'Expected a complete four-, five-, or six-position evaluation')
    require([entry['id'] for entry in entries] == [point['id'] for point in positions]
            and len({entry['id'] for entry in entries}) == len(entries),
            'Recorded trial IDs differ from the complete predeclared position list')
    require((protocol['duration_s'], protocol['policy_hz'], protocol['executed_actions_per_chunk']) == (15, 30, 100),
            'Requires five nonoverlapping ACT chunks per 450-action trial')
    source_hashes = {str(report_path): digest(report_path), str(protocol_path): digest(protocol_path)}
    for filename, expected in protocol['source_files_sha256'].items():
        require(digest(filename) == expected, f'Frozen evaluation source changed: {filename}')
        source_hashes[filename] = expected
    for filename, expected in protocol['code_files_sha256'].items():
        path = ROOT / filename
        require(digest(path) == expected, f'Frozen evaluation code changed: {filename}')
        source_hashes[str(path)] = expected
    model_directory = Path(protocol['model_directory'])
    before = {name: digest(model_directory / name) for name in MODEL_FILES}
    require(before == protocol['model_files_sha256'], 'Frozen ACT weights or saved processors changed')
    source_hashes[str(Path(__file__).resolve())] = digest(__file__)
    source_hashes[str(ROOT / 'simulation/verify_position_policy_outputs.py')] = digest(
        ROOT / 'simulation/verify_position_policy_outputs.py')
    started = time.perf_counter()
    runtime, identity = load_policy(model_directory)
    require(identity['files_sha256'] == before, 'Loaded an unexpected ACT model or processors')
    torch, policy, _, _ = runtime
    comparisons, artifact_hashes = [], {}
    for entry in entries:
        trial = entry['id']
        require(isinstance(trial, str) and re.fullmatch(r'[A-Za-z0-9_+\-]+', trial) is not None,
                'Invalid trial directory ID')
        directory = output / 'trials' / trial / 'custom'
        record_path = directory / 'result.json'
        record = read_json(record_path)
        require(record == entry['result'], f'{trial}: original result changed')
        require(record['status'] == 'completed' and record['actual_inference_queries'] == 5
                and record['policy_actions_applied'] == 450 and len(record['inference']) == 5,
                f'{trial}: incomplete recorded ACT execution')
        identity_path = directory / 'model_identity.json'
        worker_identity = read_json(identity_path)
        require(worker_identity['model_identity']['files_sha256'] == before,
                f'{trial}: original worker used a different ACT model or processors')
        artifact_hashes[str(record_path)] = digest(record_path)
        artifact_hashes[str(identity_path)] = digest(identity_path)
        policy.reset()
        torch.manual_seed(0)
        for index, inference in enumerate(record['inference']):
            observation_path = directory / f'observation_{index:03d}.npz'
            expected_path = directory / f'chunk_{index:03d}.npz'
            observation_hash = digest(observation_path)
            require(observation_hash == inference['input_sha256'], f'{trial}/{index}: input artifact changed')
            observation = load_observation(observation_path)
            require(set(observation) == {'state', 'wrist', 'top'}, f'{trial}/{index}: unexpected policy input')
            # Do not load or even hash the saved chunk before this fresh forward.
            recomputed, elapsed_ms = predict_chunk(runtime, observation)
            output_hash = digest(expected_path)
            require(output_hash == inference['output_sha256'], f'{trial}/{index}: saved chunk artifact changed')
            with np.load(expected_path, allow_pickle=False) as saved:
                require(set(saved.files) == {'action'}, f'{trial}/{index}: output fields differ')
                expected = saved['action'].copy()
            require(recomputed.shape == expected.shape == (100, 7), f'{trial}/{index}: action shape differs')
            require(np.isfinite(recomputed).all() and np.isfinite(expected).all(), f'{trial}/{index}: nonfinite action')
            difference = np.abs(recomputed.astype(np.float64) - expected.astype(np.float64))
            comparison = {
                'trial': trial, 'split': entry['split'], 'observation_index': index,
                'action_frame_index': index * 100,
                'source_input_sha256': observation_hash, 'source_chunk_npz_sha256': output_hash,
                'recomputed_action_array_sha256': array_digest(recomputed),
                'recorded_action_array_sha256': array_digest(expected),
                'maximum_absolute_difference_degrees': float(difference.max()),
                'mean_absolute_difference_degrees': float(difference.mean()),
                'exactly_equal': bool(np.array_equal(recomputed, expected)),
                'within_tolerance': bool(np.allclose(recomputed, expected, atol=.005, rtol=1e-4)),
                'fresh_inference_ms': elapsed_ms, 'policy_input_fields': sorted(observation),
                'action_values_compared': int(expected.size),
                'saved_actions_opened_only_after_fresh_forward': True}
            comparisons.append(comparison)
            artifact_hashes[str(observation_path)] = observation_hash
            artifact_hashes[str(expected_path)] = output_hash
            print(json.dumps({'trial': trial, 'observation': index,
                'max_abs_diff_deg': comparison['maximum_absolute_difference_degrees'],
                'exact': comparison['exactly_equal']}, ensure_ascii=False), flush=True)
    after = {name: digest(model_directory / name) for name in MODEL_FILES}
    after_sources = {filename: digest(filename) for filename in source_hashes}
    require(before == after and source_hashes == after_sources,
            'Frozen model/processor/protocol/evaluation/source code changed during reproduction')
    require(all(digest(filename) == expected for filename, expected in artifact_hashes.items()),
            'A saved trial/input/output artifact changed during reproduction')
    expected_calls = len(entries) * 5
    require(len(comparisons) == expected_calls, 'Unexpected number of fresh neural-network calls')
    passed = all(comparison['within_tolerance'] for comparison in comparisons)
    result = {
        'status': 'passed_fresh_ACT_output_reproduction' if passed else 'failed_ACT_output_reproduction',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_directory': str(output), 'split': protocol['split'], 'subset': protocol.get('subset'),
        'completed_positions': len(entries), 'verification_script_sha256': digest(__file__),
        'elapsed_wall_s': time.perf_counter() - started,
        'new_neural_network_forward_calls': len(comparisons),
        'new_rollouts': 0, 'new_physics_steps': 0, 'training_steps': 0, 'hardware_commands_sent': 0,
        'total_action_values_compared': sum(item['action_values_compared'] for item in comparisons),
        'exact_matches': sum(item['exactly_equal'] for item in comparisons),
        'all_chunks_exactly_equal': all(item['exactly_equal'] for item in comparisons),
        'maximum_absolute_difference_degrees': max(item['maximum_absolute_difference_degrees'] for item in comparisons),
        'tolerance': {'atol_degrees': .005, 'rtol': 1e-4},
        'model_identity': identity, 'model_files_sha256_before': before, 'model_files_sha256_after': after,
        'frozen_model_files_unchanged': True,
        'source_hashes_before': source_hashes, 'source_hashes_after': after_sources,
        'protocol_and_original_report_unchanged': True,
        'trial_input_output_files_unchanged': True, 'trial_artifact_files_sha256': artifact_hashes,
        'comparisons': comparisons, 'cpu_budget': CPU_BUDGET,
        'method': 'Fresh CPU ACT predict_action_chunk from only saved state/wrist/top; open each recorded action chunk only after its new forward returns; compare every unnormalized 100x7 value.',
        'conclusion': (f'All {expected_calls} recorded action chunks were reproduced within tolerance by fresh inference from the frozen ACT and observations.' if passed else
            'At least one recorded action chunk could not be reproduced within tolerance; investigate its provenance.'),
        'scope': 'Numerical ACT output provenance check only; not a new physical grasp or independent generalization evaluation.'}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    require(passed, 'At least one newly computed chunk differs from its recorded actions')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Existing completed evaluate_multisite_act.py result directory')
    args = parser.parse_args()
    result = verify(args.output_dir)
    print(json.dumps({'status': result['status'], 'calls': result['new_neural_network_forward_calls'],
        'exact_matches': result['exact_matches'],
        'max_difference_degrees': result['maximum_absolute_difference_degrees']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
