"""Audit saved three-checkpoint traces and classify physical failure stage.

No ACT calls or simulation steps: disposable qpos + mj_forward only. The palm
tcp is not used as a claimed exact pinch point. Original reports are read only.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np

from analyze_position_generalization import longest_run
from joint_mapping import model_action_to_sim
from preview import load_model
from run_checkpoint_comparison import check_frozen, model_hashes, SOURCE
from run_task import digest, save_json


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(output):
    target = output / 'comparison_diagnostics.json'
    if target.exists():
        raise FileExistsError('Preserve earlier diagnostics')
    report = read_json(output / 'comparison.json')
    if report['status'] != 'completed_same_conditions_checkpoint_comparison':
        raise ValueError('Comparison not complete')
    frozen = read_json(SOURCE / 'protocol.json')
    check_frozen(frozen, report['protocol']['source_files_sha256'])
    if digest(ROOT / 'simulation/run_checkpoint_comparison.py') != report['protocol']['script_sha256']:
        raise ValueError('Comparison script changed since registration')
    for model in report['protocol']['models'].values():
        if model_hashes(Path(model['directory'])) != model['files_sha256']:
            raise ValueError('Model files changed')
    initial_reference, results, initial_image_differences = {}, {}, []
    for checkpoint, entries in report['results'].items():
        results[checkpoint] = []
        for entry in entries:
            directory = output / checkpoint / entry['id']
            record = entry['result']
            if record != read_json(directory / 'custom/result.json'):
                raise ValueError('Report and trial record differ')
            if record['status'] != 'completed' or record['policy_actions_applied'] != 450 or record['actual_inference_queries'] != 5:
                raise ValueError('Incomplete physical trial')
            with np.load(directory / 'custom/observation_000.npz', allow_pickle=False) as archive:
                initial = {name: archive[name].copy() for name in archive.files}
            if entry['id'] not in initial_reference:
                initial_reference[entry['id']] = initial
            else:
                reference = initial_reference[entry['id']]
                if not np.array_equal(initial['state'], reference['state']):
                    raise ValueError(f'Initial robot state differs across checkpoints: {entry["id"]}')
                for role in ('wrist', 'top'):
                    difference = np.abs(initial[role].astype(float) - reference[role].astype(float))
                    initial_image_differences.append({'checkpoint': checkpoint, 'id': entry['id'],
                        'role': role, 'maximum_absolute_rgb_difference': float(difference.max()),
                        'unequal_rgb_values': int(np.count_nonzero(difference)),
                        'total_rgb_values': int(difference.size)})
                    if difference.max() > 1:
                        raise ValueError('Initial RGB differs by more than the observed one-level rasterization noise')
            with np.load(directory / 'custom/rollout_arrays.npz', allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            expected_targets = np.stack([model_action_to_sim(arrays['action_chunks'][i // 100, i % 100]) for i in range(450)])
            if not np.array_equal(expected_targets, arrays['target_native']):
                raise ValueError('Physical target was not exactly the ACT output conversion')
            for index, inference in enumerate(record['inference']):
                for role, field in [('observation', 'input_sha256'), ('chunk', 'output_sha256')]:
                    if digest(directory / f'custom/{role}_{index:03d}.npz') != inference[field]:
                        raise ValueError('Inference input/output changed')
            scene = directory / 'scene/task_scene.xml'
            if digest(scene) != entry['scene_sha256']:
                raise ValueError('Scene changed')
            model = load_model(scene)
            data = mujoco.MjData(model)
            brick_id = model.body('target_brick').id
            left_id, right_id = (model.body(name).id for name in ('gripper_left', 'gripper_right'))
            left, right, bilateral, brick = [], [], [], []
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
                left.append(left_id in fingers)
                right.append(right_id in fingers)
                bilateral.append(left_id in fingers and right_id in fingers)
                brick.append(data.body('target_brick').xpos.copy())
            if not np.allclose(brick, arrays['brick_position'], atol=1e-12, rtol=0):
                raise ValueError('Geometry reconstruction differs from physical trace')
            lift = np.asarray(brick)[:, 2] - record['metrics']['initial_position_m'][2]
            bilateral = np.asarray(bilateral, dtype=bool)
            elevated = bilateral & (lift >= .03)
            confirmed_grasp = longest_run(elevated) >= 3
            success = bool(record['metrics']['success'])
            if success and not confirmed_grasp:
                raise ValueError('Success lacks sustained elevated bilateral contact')
            stage = ('completed_pick_place' if success else
                'lifted_grasp_but_not_completed_place' if confirmed_grasp else
                'bilateral_contact_without_confirmed_lift' if bilateral.any() else
                'no_bilateral_grasp_contact')
            results[checkpoint].append({'id': entry['id'], 'success': success, 'stage': stage,
                'maximum_lift_mm': float(lift.max() * 1000),
                'left_finger_contact_frames': int(np.count_nonzero(left)),
                'right_finger_contact_frames': int(np.count_nonzero(right)),
                'bilateral_contact_frames': int(bilateral.sum()),
                'first_bilateral_frame': int(np.flatnonzero(bilateral)[0]) if bilateral.any() else None,
                'longest_elevated_bilateral_frames': longest_run(elevated),
                'physical_grasp_confirmed': confirmed_grasp,
                'scene_sha256': digest(scene), 'trace_sha256': digest(directory / 'custom/rollout_arrays.npz'),
                'all_450_targets_exact_ACT_conversion': True})
    result = {'status': 'passed_saved_trace_audit', 'source_comparison_sha256': digest(output / 'comparison.json'),
        'same_position_initial_joint_state_exact_across_three_models': True,
        'same_scene_camera_and_render_loop_across_three_models': True,
        'initial_RGB_arrays_bitwise_identical': all(row['unequal_rgb_values'] == 0 for row in initial_image_differences),
        'initial_RGB_maximum_absolute_difference': max(row['maximum_absolute_rgb_difference'] for row in initial_image_differences),
        'initial_RGB_rasterization_comparison': initial_image_differences,
        'rasterization_limit': 'Same XML, state and render sequence do not guarantee identical pixels: initial views differ in at most 14/921600 RGB values per camera by 1 grey level. Closed-loop numerical trajectories can amplify these differences.',
        'source_protocol_and_models_and_execution_code_unchanged': True, 'results': results,
        'new_neural_forward_calls': 0, 'new_physics_steps': 0, 'training_steps': 0, 'hardware_commands_sent': 0,
        'contact_rule': 'Active efc_address >= 0 and distance <= includemargin; simultaneous target-brick contact with gripper_left and gripper_right.',
        'grasp_rule': 'At least 3 consecutive saved 30 Hz frames with bilateral contact and >=30 mm lift.',
        'method': 'Assign recorded qpos into disposable MjData and call mj_forward, never mj_step.',
        'cpu_budget': CPU_BUDGET}
    save_json(target, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    analyze(parser.parse_args().output_dir.resolve())
