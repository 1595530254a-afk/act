"""Sequential, resumable validation matrix; never launch the final test.

Run only after long5000 training is complete. A partial/failed child directory
or a previous launch log without completed results stops the matrix; it is
never silently rerun. Complete children are source/hash checked and audited
without new physics before reuse. Checkpoint selection never reads test data.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

from evaluate_feedback_20261024 import audit_rollout, query_frames, select_positions
from run_checkpoint_comparison import SOURCE
from run_position_generalization import model_hashes
from run_task import digest, save_json

BASE = ROOT / 'runs/mujoco_iteration_20261024'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ranking_key(candidate):
    return (-candidate['validation_successes'], -candidate['validation_sustained_grasps'],
        candidate['offline_validation_normalized_mae'], 0 if candidate['mode'] == 'chunk100' else 1,
        candidate['checkpoint_step'], candidate['id'])


def ranked_eligible(candidates):
    return sorted([candidate for candidate in candidates if candidate['baseline_success']], key=ranking_key)


def checkpoint_candidate(report, model_dir, candidate_id, step):
    rows = [row for row in report['checkpoints'] if int(row['step']) == step]
    if len(rows) != 1 or Path(rows[0]['model_dir']).resolve() != model_dir.resolve():
        raise ValueError(f'Checkpoint/report mismatch: {candidate_id}')
    row = rows[0]
    mae = float(row['temporal_diagnostic']['normalized_chunk_mae'])
    if not math.isfinite(mae) or mae < 0:
        raise ValueError('Offline validation MAE must be finite and nonnegative')
    hashes = model_hashes(model_dir)
    if hashes['model.safetensors'] != row['model_sha256']:
        raise ValueError(f'Checkpoint weights differ from the training report: {candidate_id}')
    return {'id': candidate_id, 'checkpoint_step': step,
        'total_adaptation_steps': row.get('total_adaptation_steps'),
        'model_directory': str(model_dir.resolve()), 'model_files_sha256': hashes,
        'offline_validation_normalized_mae': mae}


def build_candidates(plan, training_path):
    report = read_json(training_path)
    if (not report['status'].startswith('completed_') or report['completed_steps'] != plan['training_steps']
            or report['selection_uses_test'] or report['test_episode_contents_opened'] != 0):
        raise ValueError('Training must be complete with no test-based selection before matrix evaluation')
    retained_dir = Path(plan['retained_previous_model_also_eligible']).resolve()
    retained_report = retained_dir.parents[2] / 'training_report.json'
    candidates = [checkpoint_candidate(read_json(retained_report), retained_dir, 'retained_v1', 600)]
    if plan['checkpoints_for_validation'] != [1000, 2000, 3000, 4000, 5000]:
        raise ValueError('This matrix requires the five predeclared long-training checkpoints')
    for step in plan['checkpoints_for_validation']:
        model_dir = training_path.parent / f'checkpoints/step_{step:06d}/pretrained_model'
        candidates.append(checkpoint_candidate(report, model_dir, f'step_{step:06d}', step))
    return candidates, retained_report


def validate_child(directory, candidate, mode, split, plan, position_path, source_hashes, code_hashes):
    """Read-only artifact audit. An incomplete child raises, never scores zero."""
    report_path = directory / 'evaluation.json'
    if not report_path.is_file():
        raise RuntimeError(f'Partial child has no complete evaluation: {directory}')
    report = read_json(report_path)
    protocol = report['protocol']
    points = select_positions(plan, split)
    if (report['status'] != 'completed_frozen_feedback_ACT_evaluation'
            or report['completed_trials'] != len(points) or len(report['results']) != len(points)
            or not report['model_files_unchanged'] or protocol['mode'] != mode or protocol['split'] != split
            or protocol['model_files_sha256'] != candidate['model_files_sha256']
            or Path(protocol['model_directory']).resolve() != Path(candidate['model_directory'])
            or Path(protocol['position_plan']).resolve() != position_path
            or protocol['positions'] != points or protocol['code_files_sha256'] != code_hashes
            or any(protocol['source_files_sha256'].get(name) != expected
                   for name, expected in source_hashes.items() if name in protocol['source_files_sha256'])):
        raise RuntimeError(f'Incomplete or incompatible child; do not rerun automatically: {directory}')
    expected_sources = {str(path): digest(path) for path in (position_path, SOURCE / 'protocol.json',
        Path(read_json(SOURCE / 'protocol.json')['camera_fit_path']),
        Path(read_json(SOURCE / 'protocol.json')['successful_reference_scene']))}
    if protocol['source_files_sha256'] != expected_sources or read_json(directory / 'protocol.json') != protocol:
        raise ValueError('Child protocol/source hashes differ')
    if report['actual_ACT_queries'] != len(points) * len(query_frames(mode)):
        raise ValueError('Child neural query count differs from its declared mode')
    artifacts = {'evaluation.json': digest(report_path), 'protocol.json': digest(directory / 'protocol.json')}
    for point, entry in zip(points, report['results'], strict=True):
        trial = directory / 'trials' / point['id']
        record = entry['result']
        identity = read_json(trial / 'custom/model_identity.json')['model_identity']
        if (entry['id'] != point['id'] or entry['offset_xy_m'] != point['offset_xy_m']
                or record['status'] != 'completed' or record['policy_actions_applied'] != 450
                or record['actual_inference_queries'] != len(query_frames(mode))
                or record['hardware_io'] or record['training_steps'] != 0
                or read_json(trial / 'custom/result.json') != record
                or digest(trial / 'scene/task_scene.xml') != entry['scene_sha256']
                or identity['files_sha256'] != candidate['model_files_sha256']
                or identity['device'] != 'cuda'):
            raise ValueError(f'Invalid completed trial: {trial}')
        # This independently rechecks all raw chunks, state timing, artifact
        # hashes, contacts and score; it never calls ACT or mj_step.
        audit = audit_rollout(trial / 'scene/task_scene.xml', trial / 'custom', record, mode)
        if audit != entry.get('contacts'):
            raise ValueError('Child audit is missing or differs from independent recomputation')
        for filename in ('custom/result.json', 'custom/model_identity.json', 'custom/rollout_arrays.npz', 'scene/task_scene.xml'):
            artifacts[f'trials/{point["id"]}/{filename}'] = digest(trial / filename)
    successes = sum(bool(entry['result']['metrics']['success']) for entry in report['results'])
    if successes != report['successes']:
        raise ValueError('Child task-success summary differs from its trials')
    return {'directory': str(directory), 'split': split, 'mode': mode, 'successes': successes,
        'trials': len(points), 'sustained_grasps': sum(bool(entry['contacts']['physical_grasp_confirmed']) for entry in report['results']),
        'ACT_queries': report['actual_ACT_queries'], 'artifacts_sha256': artifacts,
        'all_trial_audits_recomputed_without_ACT_or_physics_steps': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--position-plan', type=Path, default=BASE / 'position_plan.json')
    parser.add_argument('--training-report', type=Path, default=BASE / 'long5000_v1/training_report.json')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'validation_matrix_v1')
    args = parser.parse_args(argv)
    position_path, training_path, output = args.position_plan.resolve(), args.training_report.resolve(), args.output_dir.resolve()
    plan = read_json(position_path)
    if (plan['action_hz'], plan['duration_s'], plan['chunk_size'], plan['modes']) != (30, 15, 100, ['chunk100', 'ensemble1']):
        raise ValueError('Unexpected predeclared execution plan')
    candidates, retained_report = build_candidates(plan, training_path)
    frozen = read_json(SOURCE / 'protocol.json')
    code = dict(frozen['code_files_sha256'])
    for name in ('simulation/evaluate_feedback_20261024.py', 'simulation/evaluate_act_ensemble.py',
                 'simulation/run_checkpoint_comparison.py', 'simulation/preview.py',
                 'simulation/analyze_position_generalization.py', 'simulation/policy_process_cuda_20261024.py',
                 'simulation/policy_worker_cuda_20261024.py'):
        code[name] = digest(ROOT / name)
    sources = {str(path): digest(path) for path in (position_path, training_path, retained_report,
        SOURCE / 'protocol.json', Path(frozen['camera_fit_path']), Path(frozen['successful_reference_scene']))}
    config = {'position_plan': str(position_path), 'source_files_sha256': sources,
        'evaluation_code_files_sha256': code, 'runner_sha256': digest(Path(__file__)),
        'candidates': candidates, 'modes': plan['modes'], 'splits': ['baseline', 'validation'],
        'planned_trials': 60, 'selection_order': plan['selection_order'],
        'test_results_read': False, 'final_test_auto_started': False, 'cpu_budget': CPU_BUDGET}
    protocol_path = output / 'matrix_protocol.json'
    if output.exists() and not protocol_path.exists() and any(output.iterdir()):
        raise FileExistsError('Existing matrix directory lacks a registered protocol')
    output.mkdir(parents=True, exist_ok=True)
    if protocol_path.exists():
        if read_json(protocol_path)['configuration'] != config:
            raise ValueError('Resume source/code/model hashes differ from the registered matrix')
    else:
        save_json(protocol_path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'configuration': config})
    summary_path = output / 'running_summary.json'
    summary = {'status': 'running', 'matrix_protocol_sha256': digest(protocol_path), 'completed_evaluations': [],
        'candidate_results': [], 'test_results_read': False, 'final_test_auto_started': False}
    previous = read_json(summary_path) if summary_path.exists() else None

    def check_sources(candidate):
        if model_hashes(Path(candidate['model_directory'])) != candidate['model_files_sha256']:
            raise ValueError('Registered model files changed')
        for name, expected in code.items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'Evaluation code changed: {name}')
        for name, expected in sources.items():
            if digest(Path(name)) != expected:
                raise ValueError(f'Registered source changed: {name}')

    try:
        for candidate in candidates:
            for mode in plan['modes']:
                splits = {}
                for split in ('baseline', 'validation'):
                    check_sources(candidate)
                    directory = output / candidate['id'] / mode / split
                    log_path = directory.parent / f'{split}.log'
                    existed = directory.exists()
                    summary['active_evaluation'] = {'candidate': candidate['id'], 'mode': mode,
                        'split': split, 'directory': str(directory), 'log': str(log_path),
                        'phase': 'checking_completed_output' if existed else 'starting_subprocess',
                        'started_utc': datetime.now(timezone.utc).isoformat()}
                    save_json(summary_path, summary)
                    if not existed:
                        if log_path.exists():
                            raise RuntimeError(f'Prior launch without completed output requires review: {log_path}')
                        directory.parent.mkdir(parents=True, exist_ok=True)
                        print(json.dumps({'stage': 'starting_split', 'candidate': candidate['id'], 'mode': mode,
                            'split': split, 'log': str(log_path)}, ensure_ascii=False), flush=True)
                        with log_path.open('xb') as log:
                            result = subprocess.run([str(ROOT / 'local/mujoco-env/Scripts/python.exe'), '-u',
                                str(ROOT / 'simulation/evaluate_feedback_20261024.py'), '--position-plan', str(position_path),
                                '--split', split, '--mode', mode, '--model-dir', candidate['model_directory'],
                                '--output-dir', str(directory)], stdout=log, stderr=subprocess.STDOUT,
                                cwd=ROOT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                        if result.returncode:
                            raise RuntimeError(f'Evaluation process exited {result.returncode}; preserve partial output and inspect {log_path}')
                    checked = validate_child(directory, candidate, mode, split, plan, position_path, sources, code)
                    if previous:
                        saved = next((item for item in previous['completed_evaluations'] if item['directory'] == str(directory)), None)
                        if saved and saved['artifacts_sha256'] != checked['artifacts_sha256']:
                            raise ValueError('A previously accepted completed child changed')
                    checked['reused_complete_output'] = existed
                    summary['completed_evaluations'].append(checked)
                    summary['active_evaluation'] = None
                    splits[split] = checked
                    save_json(summary_path, summary)
                    print(json.dumps({'stage': 'completed_split', 'candidate': candidate['id'], 'mode': mode,
                        'split': split, 'successes': checked['successes'], 'trials': checked['trials'],
                        'ACT_queries': checked['ACT_queries'], 'reused': existed}, ensure_ascii=False), flush=True)
                summary['candidate_results'].append({**candidate, 'mode': mode,
                    'baseline_success': splits['baseline']['successes'] == 1,
                    'validation_successes': splits['validation']['successes'],
                    'validation_sustained_grasps': splits['validation']['sustained_grasps'],
                    'baseline_result': splits['baseline']['directory'], 'validation_result': splits['validation']['directory']})
                save_json(summary_path, summary)
        for candidate in candidates:
            check_sources(candidate)
        ranked = ranked_eligible(summary['candidate_results'])
        selection = {'status': 'selection_proposed_requires_root_review' if ranked else 'no_candidate_passed_original_position',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'matrix_protocol_sha256': digest(protocol_path),
            'position_plan': str(position_path), 'position_plan_sha256': digest(position_path),
            'selection_order': plan['selection_order'], 'ranked_eligible_candidates': ranked,
            'all_candidate_results': summary['candidate_results'], 'selected_candidate': ranked[0] if ranked else None,
            'test_results_read': False, 'final_test_auto_started': False, 'task_generalization_passed': False,
            'training_steps': 0, 'hardware_commands_sent': 0}
        selection_path = output / 'selection_proposal.json'
        if selection_path.exists():
            old = read_json(selection_path)
            if old['selected_candidate'] != selection['selected_candidate'] or old['all_candidate_results'] != selection['all_candidate_results']:
                raise ValueError('Completed matrix selection differs on resume')
        else:
            save_json(selection_path, selection)
        summary['status'] = 'completed_validation_matrix_no_final_test'
        summary['selection_proposal'] = str(selection_path)
    except (Exception, KeyboardInterrupt) as error:
        summary.update({'status': 'matrix_execution_error_not_task_failure', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
        summary['completed_physical_trials'] = sum(item['trials'] for item in summary['completed_evaluations'])
        summary['actual_ACT_queries'] = sum(item['ACT_queries'] for item in summary['completed_evaluations'])
        save_json(summary_path, summary)


if __name__ == '__main__':
    main()
