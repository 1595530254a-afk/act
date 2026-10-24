"""Resume the frozen matrix, distinguishing proven policy-limit aborts.

The original protocol, interrupted summary and all trial directories remain
unchanged. Only an independently reconstructed hard-limit rejection permits
continuing to another candidate; all other incomplete output still stops.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import numpy as np

from evaluate_feedback_20261024 import audit_arrays, audit_rollout, query_frames, select_positions, weighted_action
from joint_mapping import model_action_to_sim
from preview import load_model
from run_checkpoint_comparison import SOURCE
from run_long5000_validation_20261024 import BASE, build_candidates, ranked_eligible, read_json, validate_child
from run_position_generalization import model_hashes
from run_task import digest, save_json
from task_dynamics import validate_target


def prove_limit_rejection(model, arrays, record, mode):
    """Check the complete applied prefix and reconstruct its first rejected action."""
    frame = len(arrays['frame_index'])
    if not 0 <= frame < 450 or record['policy_actions_applied'] != frame:
        raise ValueError('Not an incomplete action prefix')
    starts = arrays['query_frame_index']
    expected = query_frames(mode, frame + 1)
    if (not np.array_equal(starts, expected)
            or arrays['action_chunks'].shape != (len(expected), 100, 7)
            or len(record['inference']) != len(expected)
            or record['actual_inference_queries'] != len(expected)
            or record['actual_neural_forward_calls'] != len(expected)):
        raise ValueError('Aborted query schedule differs from raw chunks')
    # A fresh query can occur on the rejected frame. It is audited separately;
    # the ordinary prefix audit must see only queries used by applied frames.
    used = starts < frame
    prefix = dict(arrays)
    for name in ('action_chunks', 'chunk_observation_state', 'query_frame_index', 'query_actual_sim_time'):
        prefix[name] = arrays[name][used]
    prefix_record = {**record, 'inference': [row for row in record['inference'] if row['frame_index'] < frame]}
    if frame:
        audit_arrays(prefix, prefix_record, mode)
    for index, start in enumerate(starts):
        state = np.asarray(record['initial_public_state']) if start == 0 else arrays['public_state'][start - 1]
        inference = record['inference'][index]
        if (not np.array_equal(state, arrays['chunk_observation_state'][index])
                or inference['frame_index'] != int(start) or inference['request_index'] != index + 1
                or abs(arrays['query_actual_sim_time'][index] - start / 30) > .00100001):
            raise ValueError('Aborted query did not use its actual pre-action state')
    for target in arrays['target_native']:
        validate_target(model, target)
    action, indices, rows, weights = weighted_action(arrays['action_chunks'], starts, frame)
    target = model_action_to_sim(action)
    try:
        validate_target(model, target)
    except ValueError as error:
        # Require the exact original exception, not a substring/status guess.
        original = f'ValueError: {error}'
        if record['error'] != original or ' outside [' not in str(error):
            raise ValueError('Raw rejected target does not explain the original error') from error
    else:
        raise ValueError('The purported rejected target passes the original hard limits')
    violations = []
    for index in range(7):
        joint = model.joint(f'joint{index + 1}')
        if bool(joint.limited[0]):
            lower, upper = joint.range
            if not lower <= target[index] <= upper:
                violations.append({'joint': f'joint{index + 1}', 'target': float(target[index]),
                    'lower': float(lower), 'upper': float(upper)})
    if not violations:
        raise ValueError('No independently verified hard-limit violation')
    return {'status': 'policy_limit_abort', 'rejected_frame': frame,
        'applied_action_frames': frame, 'raw_public_action': action.tolist(),
        'unapplied_native_target': target.tolist(), 'violations': violations,
        'query_indices': indices.tolist(), 'chunk_rows': rows.tolist(), 'weights': weights.tolist(),
        'all_earlier_applied_targets_pass_original_limits': True,
        'exact_original_validate_target_exception_reproduced': original,
        'task_success': None, 'partial_final_metrics_not_used_for_selection': True}


def validate_abort(directory, candidate, mode, split, plan, position_path, sources, code):
    """Accept only a source-compatible partial split caused by policy limits."""
    report = read_json(directory / 'evaluation.json')
    protocol, points = report['protocol'], select_positions(plan, split)
    frozen = read_json(SOURCE / 'protocol.json')
    expected_sources = {str(path): sources[str(path)] for path in (position_path, SOURCE / 'protocol.json',
        Path(frozen['camera_fit_path']), Path(frozen['successful_reference_scene']))}
    if (report['status'] != 'evaluation_error' or not report['model_files_unchanged']
            or protocol['mode'] != mode or protocol['split'] != split or protocol['positions'] != points
            or protocol['model_files_sha256'] != candidate['model_files_sha256']
            or Path(protocol['model_directory']).resolve() != Path(candidate['model_directory'])
            or Path(protocol['position_plan']).resolve() != position_path
            or protocol['code_files_sha256'] != code or protocol['source_files_sha256'] != expected_sources
            or read_json(directory / 'protocol.json') != protocol
            or not 1 <= len(report['results']) <= len(points)):
        raise ValueError('Incomplete child is not compatible with the frozen matrix')
    artifacts = {'evaluation.json': digest(directory / 'evaluation.json'),
        'protocol.json': digest(directory / 'protocol.json')}
    complete, abort, queries = [], None, 0
    for index, entry in enumerate(report['results']):
        point = points[index]
        trial, record = directory / 'trials' / point['id'], entry['result']
        custom = trial / 'custom'
        identity = read_json(custom / 'model_identity.json')['model_identity']
        if (entry['id'] != point['id'] or entry['offset_xy_m'] != point['offset_xy_m']
                or digest(trial / 'scene/task_scene.xml') != entry['scene_sha256']
                or read_json(custom / 'result.json') != record
                or identity['files_sha256'] != candidate['model_files_sha256'] or identity['device'] != 'cuda'
                or record['hardware_io'] or record['training_steps'] != 0):
            raise ValueError('Partial trial identity or saved result differs')
        queries += record['actual_inference_queries']
        for filename in ('custom/result.json', 'custom/model_identity.json', 'custom/rollout_arrays.npz', 'scene/task_scene.xml'):
            artifacts[f'trials/{point["id"]}/{filename}'] = digest(trial / filename)
        if record['status'] == 'completed':
            if record['policy_actions_applied'] != 450 or record['actual_inference_queries'] != len(query_frames(mode)):
                raise ValueError('Earlier completed trial is incomplete')
            audit = audit_rollout(trial / 'scene/task_scene.xml', custom, record, mode)
            if audit != entry['contacts']:
                raise ValueError('Earlier completed trial audit differs')
            complete.append({'id': point['id'], 'success': record['metrics']['success'], 'contacts': audit})
            continue
        if record['status'] != 'failed' or index != len(report['results']) - 1:
            raise ValueError('Unknown failure or trials after an abort')
        with np.load(custom / 'rollout_arrays.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        for query, inference in enumerate(record['inference']):
            observation, raw = custom / f'observation_{query:03d}.npz', custom / f'chunk_{query:03d}.npz'
            if digest(observation) != inference['input_sha256'] or digest(raw) != inference['output_sha256']:
                raise ValueError('Aborted worker artifact hashes differ')
            with np.load(observation, allow_pickle=False) as value:
                if (set(value.files) != {'state', 'wrist', 'top'}
                        or not np.array_equal(value['state'], arrays['chunk_observation_state'][query])
                        or any(value[role].shape != (480, 640, 3) or value[role].dtype != np.uint8 for role in ('wrist', 'top'))):
                    raise ValueError('Unexpected aborted policy inputs')
            with np.load(raw, allow_pickle=False) as value:
                if set(value.files) != {'action'} or not np.array_equal(value['action'], arrays['action_chunks'][query]):
                    raise ValueError('Aborted raw chunks differ from the worker output')
        abort = {'id': point['id'], **prove_limit_rejection(load_model(trial / 'scene/task_scene.xml'), arrays, record, mode)}
        if report['error'] != f'RuntimeError: Incomplete feedback rollout: {record["error"]}':
            raise ValueError('Top-level failure has a different cause')
    if abort is None or queries != report['actual_ACT_queries']:
        raise ValueError('No proven policy-limit abort or query count mismatch')
    return {'status': 'policy_limit_abort', 'directory': str(directory), 'split': split, 'mode': mode,
        'successes': None, 'sustained_grasps': None, 'trials': len(complete),
        'completed_trials_before_abort': complete, 'aborted_trial': abort,
        'not_attempted_point_ids': [point['id'] for point in points[len(report['results']):]],
        'ACT_queries': queries, 'artifacts_sha256': artifacts,
        'all_completed_trial_audits_recomputed_without_ACT_or_physics_steps': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'validation_matrix_v1')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    protocol_path, original_summary_path = output / 'matrix_protocol.json', output / 'running_summary.json'
    config = read_json(protocol_path)['configuration']
    position_path = Path(config['position_plan'])
    plan = read_json(position_path)
    sources, code = config['source_files_sha256'], config['evaluation_code_files_sha256']
    training_path = BASE / 'long5000_v1/training_report.json'
    candidates, _ = build_candidates(plan, training_path)
    if candidates != config['candidates'] or digest(ROOT / 'simulation/run_long5000_validation_20261024.py') != config['runner_sha256']:
        raise ValueError('Original registered candidates or runner changed')
    interrupted = read_json(original_summary_path)
    original_log = Path(interrupted['active_evaluation']['log'])
    amendment = {'matrix_protocol_sha256': digest(protocol_path),
        'original_interrupted_summary_sha256': digest(original_summary_path),
        'original_failed_child_log': str(original_log), 'original_failed_child_log_sha256': digest(original_log),
        'new_runner_sha256': digest(Path(__file__)), 'original_runner_sha256': config['runner_sha256'],
        'reason': 'Independently proven policy hard-limit aborts make only that candidate-mode ineligible; never clip or relax limits.',
        'baseline_abort_validation': 'not_attempted_due_policy_limit_abort; successes/grasp counts null',
        'eval_physics_or_position_plan_changed': False, 'old_results_reused_without_new_ACT_calls': True,
        'unknown_exceptions_still_stop': True, 'test_results_read': False, 'final_test_auto_started': False}
    amendment_path = output / 'recovery_amendment_v2.json'
    if amendment_path.exists():
        if read_json(amendment_path) != amendment:
            raise ValueError('Recovery amendment changed')
    else:
        save_json(amendment_path, amendment)
    summary_path = output / 'running_summary_v2.json'
    previous = read_json(summary_path) if summary_path.exists() else read_json(original_summary_path)
    summary = {'status': 'running', 'matrix_protocol_sha256': digest(protocol_path),
        'amendment_sha256': digest(amendment_path), 'completed_evaluations': [], 'candidate_results': [],
        'test_results_read': False, 'final_test_auto_started': False, 'cpu_budget': CPU_BUDGET}

    def check_sources(candidate):
        if model_hashes(Path(candidate['model_directory'])) != candidate['model_files_sha256']:
            raise ValueError('Registered model changed')
        for name, sha in code.items():
            if digest(ROOT / name) != sha:
                raise ValueError(f'Frozen evaluation code changed: {name}')
        for name, sha in sources.items():
            if digest(Path(name)) != sha:
                raise ValueError(f'Frozen source changed: {name}')

    try:
        for candidate in candidates:
            for mode in plan['modes']:
                splits = {}
                for split in ('baseline', 'validation'):
                    if split == 'validation' and splits['baseline']['status'] == 'policy_limit_abort':
                        splits[split] = {'status': 'not_attempted_due_policy_limit_abort', 'successes': None,
                            'sustained_grasps': None, 'directory': None, 'trials': 0, 'ACT_queries': 0}
                        break
                    check_sources(candidate)
                    directory = output / candidate['id'] / mode / split
                    log_path = directory.parent / f'{split}.log'
                    existed = directory.exists()
                    summary['active_evaluation'] = {'candidate': candidate['id'], 'mode': mode, 'split': split,
                        'directory': str(directory), 'log': str(log_path)}
                    save_json(summary_path, summary)
                    if not existed:
                        if log_path.exists():
                            raise RuntimeError(f'Prior launch without output requires review: {log_path}')
                        directory.parent.mkdir(parents=True, exist_ok=True)
                        print(json.dumps({'stage': 'starting_split', 'candidate': candidate['id'], 'mode': mode, 'split': split}), flush=True)
                        with log_path.open('xb') as log:
                            process = subprocess.run([str(ROOT / 'local/mujoco-env/Scripts/python.exe'), '-u',
                                str(ROOT / 'simulation/evaluate_feedback_20261024.py'), '--position-plan', str(position_path),
                                '--split', split, '--mode', mode, '--model-dir', candidate['model_directory'], '--output-dir', str(directory)],
                                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                        if process.returncode and not (directory / 'evaluation.json').is_file():
                            raise RuntimeError(f'Unknown evaluation failure; preserve {log_path}')
                    report = read_json(directory / 'evaluation.json')
                    if report['status'] == 'completed_frozen_feedback_ACT_evaluation':
                        checked = {'status': 'completed', **validate_child(directory, candidate, mode, split, plan, position_path, sources, code)}
                    else:
                        checked = validate_abort(directory, candidate, mode, split, plan, position_path, sources, code)
                    old = next((item for item in previous['completed_evaluations'] if item['directory'] == str(directory)), None)
                    if old and old['artifacts_sha256'] != checked['artifacts_sha256']:
                        raise ValueError('Previously registered child changed')
                    checked['reused_existing_output'] = existed
                    summary['completed_evaluations'].append(checked)
                    summary['active_evaluation'] = None
                    splits[split] = checked
                    save_json(summary_path, summary)
                    print(json.dumps({'stage': 'checked_split', 'candidate': candidate['id'], 'mode': mode, 'split': split,
                        'status': checked['status'], 'successes': checked['successes'], 'completed_trials': checked['trials'],
                        'ACT_queries': checked['ACT_queries'], 'reused': existed}), flush=True)
                aborted = any(row['status'] == 'policy_limit_abort' for row in splits.values())
                baseline_pass = splits['baseline']['status'] == 'completed' and splits['baseline']['successes'] == 1
                summary['candidate_results'].append({**candidate, 'mode': mode,
                    'eligible_for_selection': baseline_pass and not aborted,
                    'eligibility_status': 'ineligible_policy_limit_abort' if aborted else ('eligible_by_baseline' if baseline_pass else 'ineligible_baseline_task_failure'),
                    'baseline_status': splits['baseline']['status'], 'baseline_success': baseline_pass,
                    'validation_status': splits['validation']['status'], 'validation_successes': splits['validation']['successes'],
                    'validation_sustained_grasps': splits['validation']['sustained_grasps'],
                    'baseline_result': splits['baseline']['directory'], 'validation_result': splits['validation']['directory']})
                save_json(summary_path, summary)
        for candidate in candidates:
            check_sources(candidate)
        ranked = ranked_eligible([row for row in summary['candidate_results'] if row['eligible_for_selection']])
        selection = {'status': 'selection_proposed_requires_root_review' if ranked else 'no_candidate_passed_original_position',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'matrix_protocol_sha256': digest(protocol_path),
            'amendment_sha256': digest(amendment_path), 'position_plan': str(position_path), 'position_plan_sha256': digest(position_path),
            'selection_order': plan['selection_order'], 'ranked_eligible_candidates': ranked,
            'all_candidate_results': summary['candidate_results'], 'selected_candidate': ranked[0] if ranked else None,
            'test_results_read': False, 'final_test_auto_started': False, 'task_generalization_passed': False,
            'training_steps': 0, 'hardware_commands_sent': 0}
        selection_path = output / 'selection_proposal.json'
        if selection_path.exists():
            raise FileExistsError('Preserve the existing selection proposal for review')
        save_json(selection_path, selection)
        summary.update({'status': 'completed_validation_matrix_no_final_test', 'selection_proposal': str(selection_path)})
    except (Exception, KeyboardInterrupt) as error:
        summary.update({'status': 'matrix_execution_error_not_task_failure', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
        summary['completed_physical_trials'] = sum(row['trials'] for row in summary['completed_evaluations'])
        summary['policy_limit_aborted_trials'] = sum(row['status'] == 'policy_limit_abort' for row in summary['completed_evaluations'])
        summary['not_attempted_trials_due_policy_limit_abort'] = (
            sum(4 for row in summary['candidate_results'] if row['validation_status'] == 'not_attempted_due_policy_limit_abort')
            + sum(len(row.get('not_attempted_point_ids', [])) for row in summary['completed_evaluations']))
        summary['actual_ACT_queries'] = sum(row['ACT_queries'] for row in summary['completed_evaluations'])
        save_json(summary_path, summary)


if __name__ == '__main__':
    main()
