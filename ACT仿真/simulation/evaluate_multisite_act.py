"""Evaluate a frozen ACT on predeclared positions, never on teacher actions.

Only point IDs, split labels and initial XY offsets are extracted from the
teacher protocol. Initial robot state, cameras, props and dynamics remain the
old six-position protocol. All commands are simulation-only ACT predictions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np

from analyze_position_generalization import longest_run
from joint_mapping import dataset_state_to_sim, model_action_to_sim
from preview import load_model
from run_checkpoint_comparison import create_scene, run_without_video, SOURCE
from run_position_generalization import model_hashes, normalized_scene_tree
from run_task import digest, save_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--teacher-protocol', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('validation', 'test'))
    parser.add_argument('--subset', choices=('previous_diagnostic', 'final_held_out'))
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.split == 'validation' and args.subset is not None:
        parser.error('--subset is defined only for test positions')
    return args


def select_positions(protocol, split, subset=None):
    """Pure extraction: no teacher frames, state sequences or actions."""
    selected = []
    for point in protocol['positions']:
        if point['split'] != split or (subset is not None and point.get('subset') != subset):
            continue
        if not re.fullmatch(r'[A-Za-z0-9_+\-]+', point['id']):
            raise ValueError('Invalid point ID')
        offset = np.asarray(point['offset_xy_m'], dtype=float)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError('Expected two finite initial XY offsets')
        selected.append({'id': point['id'], 'split': split,
            'subset': point.get('subset'), 'offset_xy_m': offset.tolist()})
    if not selected or len({point['id'] for point in selected}) != len(selected):
        raise ValueError('Expected nonempty positions with unique IDs')
    return selected


def audit_contacts(scene, directory, result):
    """Audit only recorded qpos; no ACT calls and no mj_step."""
    with np.load(directory / 'rollout_arrays.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected = np.stack([model_action_to_sim(arrays['action_chunks'][i // 100, i % 100])
                         for i in range(len(arrays['target_native']))])
    if not np.array_equal(expected, arrays['target_native']):
        raise ValueError('An applied target differs from ACT output conversion')
    for index, inference in enumerate(result['inference']):
        if (digest(directory / f'observation_{index:03d}.npz') != inference['input_sha256']
                or digest(directory / f'chunk_{index:03d}.npz') != inference['output_sha256']):
            raise ValueError('Inference input or output artifact changed')
    model, counts = load_model(scene), {'left': [], 'right': [], 'bilateral': []}
    data = mujoco.MjData(model)
    brick_id = model.body('target_brick').id
    left_id, right_id = (model.body(name).id for name in ('gripper_left', 'gripper_right'))
    for pose in arrays['qpos']:
        data.qpos[:] = pose
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        fingers = set()
        for contact in data.contact[:data.ncon]:
            if contact.efc_address < 0 or contact.dist > contact.includemargin:
                continue
            first, second = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if first == brick_id:
                fingers.add(second)
            elif second == brick_id:
                fingers.add(first)
        counts['left'].append(left_id in fingers)
        counts['right'].append(right_id in fingers)
        counts['bilateral'].append(left_id in fingers and right_id in fingers)
    bilateral = np.asarray(counts['bilateral'], dtype=bool)
    lift = arrays['brick_position'][:, 2] - result['metrics']['initial_position_m'][2]
    elevated = bilateral & (lift >= .03)
    confirmed = longest_run(elevated) >= 3
    if result['metrics']['success'] and not confirmed:
        raise ValueError('Apparent success lacks sustained elevated bilateral contact')
    return {'left_finger_contact_frames': int(np.count_nonzero(counts['left'])),
        'right_finger_contact_frames': int(np.count_nonzero(counts['right'])),
        'bilateral_contact_frames': int(bilateral.sum()),
        'first_bilateral_frame': int(np.flatnonzero(bilateral)[0]) if bilateral.any() else None,
        'longest_elevated_bilateral_frames': longest_run(elevated),
        'physical_grasp_confirmed': confirmed,
        'all_applied_targets_exact_ACT_conversion': True,
        'all_input_output_hashes_match_worker_records': True,
        'contact_rule': 'Active efc_address >= 0 and dist <= includemargin, both fingers on target brick.',
        'grasp_rule': 'At least 3 consecutive recorded 30 Hz frames with bilateral contact and >=30 mm lift.',
        'audit_new_physics_steps': 0, 'audit_new_ACT_calls': 0}


def evaluate(args):
    model_dir = args.model_dir.resolve()
    teacher_path = args.teacher_protocol.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Preserve earlier evaluation; use a fresh output directory')
    teacher = json.loads(teacher_path.read_text(encoding='utf-8'))
    points = select_positions(teacher, args.split, args.subset)
    frozen_path = SOURCE / 'protocol.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if (frozen['duration_s'], frozen['policy_hz'], frozen['executed_actions_per_chunk']) != (15, 30, 100):
        raise ValueError('Frozen evaluation settings differ')
    if (digest(Path(frozen['camera_fit_path'])) != frozen['camera_fit_sha256']
            or digest(Path(frozen['successful_reference_scene'])) != frozen['successful_reference_scene_sha256']):
        raise ValueError('Camera or reference scene differs from the original frozen protocol')
    model_before = model_hashes(model_dir)
    code = dict(frozen['code_files_sha256'])
    for name in ('simulation/run_checkpoint_comparison.py', 'simulation/evaluate_multisite_act.py',
                 'simulation/analyze_position_generalization.py', 'simulation/preview.py'):
        code[name] = digest(ROOT / name)
    sources = {str(path): digest(path) for path in (teacher_path, frozen_path,
        Path(frozen['camera_fit_path']), Path(frozen['successful_reference_scene']))}

    def check_frozen():
        if model_hashes(model_dir) != model_before:
            raise ValueError('Checkpoint or saved processors changed')
        for filename, expected in code.items():
            if digest(ROOT / filename) != expected:
                raise ValueError(f'Evaluation code changed: {filename}')
        for filename, expected in sources.items():
            if digest(Path(filename)) != expected:
                raise ValueError(f'Frozen protocol or scene changed: {filename}')

    check_frozen()
    output.mkdir(parents=True, exist_ok=False)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_directory': str(model_dir), 'model_files_sha256': model_before,
        'teacher_protocol': str(teacher_path), 'teacher_protocol_sha256': digest(teacher_path),
        'split': args.split, 'subset': args.subset, 'positions': points,
        'fixed_initial_state_protocol': str(frozen_path), 'source_files_sha256': sources,
        'code_files_sha256': code, 'duration_s': 15, 'policy_hz': 30,
        'executed_actions_per_chunk': 100,
        'teacher_arrays_read': False, 'teacher_actions_read': False,
        'object_truth_usage': 'Only initial scene XY placement and physical scoring; never policy input',
        'test_scope': teacher.get('test_limit'), 'training_steps_allowed': 0,
        'hardware_commands_allowed': 0, 'cpu_budget': CPU_BUDGET}
    save_json(output / 'protocol.json', plan)
    report = {'status': 'running', 'protocol': plan, 'results': [],
        'training_steps': 0, 'hardware_commands_sent': 0, 'actual_ACT_queries': 0}
    save_json(output / 'evaluation.json', report)
    initial = dataset_state_to_sim(frozen['initial_public_joint_target_degrees'])
    fit = json.loads(Path(frozen['camera_fit_path']).read_text(encoding='utf-8'))
    normalized = normalized_scene_tree(ET.parse(frozen['successful_reference_scene']).getroot())
    try:
        for point in points:
            check_frozen()
            directory = output / 'trials' / point['id']
            scene, metadata = create_scene(directory, point, frozen, fit, normalized)
            result = run_without_video(scene, metadata, directory / 'custom', model_dir, initial, frozen)
            report['actual_ACT_queries'] += result['actual_inference_queries']
            entry = {'id': point['id'], 'split': args.split, 'subset': point.get('subset'),
                'offset_xy_m': point['offset_xy_m'], 'scene_sha256': digest(scene), 'result': result}
            report['results'].append(entry)
            save_json(output / 'evaluation.json', report)
            if result['status'] != 'completed' or result['actual_inference_queries'] != 5 or result['policy_actions_applied'] != 450:
                raise RuntimeError(f'Incomplete ACT rollout: {result["error"]}')
            identity = json.loads((directory / 'custom/model_identity.json').read_text(encoding='utf-8'))
            if identity['model_identity']['files_sha256'] != model_before:
                raise ValueError('Worker loaded a different checkpoint or processors')
            entry['contacts'] = audit_contacts(scene, directory / 'custom', result)
            save_json(output / 'evaluation.json', report)
            print(json.dumps({'point': point['id'], 'success': result['metrics']['success'],
                'lift_mm': result['metrics']['maximum_lift_m'] * 1000,
                'bilateral_frames': entry['contacts']['bilateral_contact_frames']}, ensure_ascii=False), flush=True)
        report['status'] = 'completed_frozen_multisite_ACT_evaluation'
    except Exception as error:
        report['status'] = 'evaluation_error'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        report['model_files_unchanged'] = model_hashes(model_dir) == model_before
        report['completed_trials'] = len(report['results'])
        report['successes'] = sum(bool(entry['result']['metrics']['success']) for entry in report['results'])
        save_json(output / 'evaluation.json', report)
    check_frozen()


if __name__ == '__main__':
    evaluate(parse_args())
