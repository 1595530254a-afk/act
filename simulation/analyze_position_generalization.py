"""Independently audit saved position trials; never step physics or call ACT.

Only the analysis report is written. Saved qpos frames are forwarded to recover
geometric contacts and the upstream tcp site, not replayed as a new experiment.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np

from joint_mapping import model_action_to_sim
from preview import load_model
from task_dynamics import native_to_public

DEFAULT_OUTPUT = ROOT / 'runs/mujoco_validation_20261023/position_generalization_20261023'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_scene(path):
    tree = deepcopy(ET.parse(path).getroot())
    tree.find('compiler').set('meshdir', 'FROZEN_UPSTREAM_MESHES')
    tree.find(".//body[@name='target_brick']").set('pos', 'REFERENCE_BRICK_POSITION')
    return ET.tostring(tree)


def max_abs(array):
    return float(np.max(np.abs(array)))


def longest_run(mask):
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def differences(values):
    return {'max_abs_degrees': max_abs(values),
            'rms_degrees': float(np.sqrt(np.mean(values ** 2))),
            'mean_abs_degrees': float(np.mean(np.abs(values)))}


def audit_trial(output, entry, trial, protocol, baseline_state):
    name = trial['id']
    directory = output / 'trials' / name
    record = read_json(directory / 'custom/result.json')
    require(record == entry['result'], f'{name}: original report and per-trial record disagree')
    require(record['status'] == 'completed', f'{name}: rollout aborted')
    require(record['training_steps'] == 0 and not record['hardware_io'], f'{name}: unexpected I/O/training')
    require(not record['object_pose_assigned_during_rollout']
            and not record['grasp_weld_or_object_attach_used'], f'{name}: attachment or teleport flag')
    require(record['actual_inference_queries'] == 5 and record['policy_actions_applied'] == 450,
            f'{name}: unexpected action/query count')
    require(record['evaluation']['executed_actions_per_chunk'] == 100,
            f'{name}: action horizon changed')
    identity = read_json(directory / 'custom/model_identity.json')['model_identity']
    require(identity['files_sha256'] == protocol['model_files_sha256'], f'{name}: model files differ')
    scene_path = directory / 'scene/task_scene.xml'
    require(digest(scene_path) == entry['scene_sha256'], f'{name}: scene hash changed')
    require(normalized_scene(scene_path) == normalized_scene(protocol['successful_reference_scene']),
            f'{name}: scene properties other than position/path differ')
    xml_position = np.fromstring(ET.parse(scene_path).find(".//body[@name='target_brick']").get('pos'), sep=' ')
    expected_position = np.array(protocol['reference_brick_center_m'], dtype=np.float64)
    expected_position[:2] += trial['offset_xy_m']
    require(np.allclose(xml_position, expected_position, atol=1e-12, rtol=0), f'{name}: unexpected placement')
    settled_initial = np.array(record['metrics']['initial_position_m'])
    offset = settled_initial[:2] - np.array(protocol['reference_brick_center_m'])[:2]
    require(np.allclose(offset, trial['offset_xy_m'], atol=1e-7, rtol=0), f'{name}: settled XY mismatch')
    offset_length = float(np.linalg.norm(offset))
    require((offset_length < 1e-7 if name == 'baseline' else .03 - 1e-7 <= offset_length <= .05 + 1e-7),
            f'{name}: offset outside preregistered 30--50 mm')
    arrays_path = directory / 'custom/rollout_arrays.npz'
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    qpos, chunks = arrays['qpos'], arrays['action_chunks']
    require(qpos.shape == (450, 16) and chunks.shape == (5, 100, 7), f'{name}: unexpected trace shape')
    require(all(np.isfinite(value).all() for value in arrays.values()), f'{name}: nonfinite trace')
    expected_targets = np.stack([model_action_to_sim(chunks[index // 100, index % 100])
                                 for index in range(450)])
    require(np.array_equal(expected_targets, arrays['target_native']),
            f'{name}: physical target differs from model output conversion')
    state_difference = max_abs(np.array(record['initial_public_state']) - baseline_state)
    require(state_difference <= 1e-7, f'{name}: initial state differs from reference')
    input_hashes, output_hashes, observed_keys = [], [], []
    first_state = None
    for index, inference in enumerate(record['inference']):
        observation = directory / f'custom/observation_{index:03d}.npz'
        action_path = directory / f'custom/chunk_{index:03d}.npz'
        require(digest(observation) == inference['input_sha256']
                and digest(action_path) == inference['output_sha256'], f'{name}: inference artifact hash mismatch')
        with np.load(observation, allow_pickle=False) as inputs:
            require(set(inputs.files) == {'state', 'wrist', 'top'}, f'{name}: extra policy input field')
            require(inputs['state'].shape == (7,) and inputs['state'].dtype.kind == 'f', f'{name}: state shape')
            require(np.array_equal(inputs['state'], arrays['chunk_observation_state'][index]),
                    f'{name}: observation differs from saved policy state')
            if index == 0:
                first_state = inputs['state'].copy()
            for role in ('wrist', 'top'):
                require(inputs[role].shape == (480, 640, 3) and inputs[role].dtype == np.uint8,
                        f'{name}: image contract {role}')
            observed_keys.append(sorted(inputs.files))
        with np.load(action_path, allow_pickle=False) as saved:
            require(set(saved.files) == {'action'} and np.array_equal(saved['action'], chunks[index]),
                    f'{name}: chunk differs from worker output')
        input_hashes.append(digest(observation))
        output_hashes.append(digest(action_path))
    require(np.array_equal(first_state, baseline_state), f'{name}: initial policy state differs')
    model = load_model(scene_path)
    data = mujoco.MjData(model)
    joint_addresses = np.array([int(model.joint(f'joint{i}').qposadr[0]) for i in range(1, 8)])
    for index in range(1, 5):
        actual_previous_state = native_to_public(qpos[index * 100 - 1, joint_addresses])
        require(np.allclose(actual_previous_state, arrays['chunk_observation_state'][index], atol=1e-10, rtol=0),
                f'{name}: policy state does not match preceding physical frame')
    brick_id = model.body('target_brick').id
    left_id, right_id = model.body('gripper_left').id, model.body('gripper_right').id
    tcp, brick, bilateral, left_contact, right_contact = [], [], [], [], []
    for pose in qpos:
        data.qpos[:] = pose
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        fingers = set()
        for contact in data.contact[:data.ncon]:
            # The upstream finger meshes use a 0.15 mm contact margin.
            # Positive geometric distance can therefore be an active physical
            # constraint; requiring penetration would miss the true grasp.
            if contact.efc_address < 0 or contact.dist > contact.includemargin:
                continue
            first, second = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if first == brick_id:
                fingers.add(second)
            if second == brick_id:
                fingers.add(first)
        left_contact.append(left_id in fingers)
        right_contact.append(right_id in fingers)
        bilateral.append(left_id in fingers and right_id in fingers)
        tcp.append(data.site('tcp').xpos.copy())
        brick.append(data.body('target_brick').xpos.copy())
    tcp, brick, bilateral = np.asarray(tcp), np.asarray(brick), np.asarray(bilateral, dtype=bool)
    require(np.allclose(brick, arrays['brick_position'], atol=1e-12, rtol=0),
            f'{name}: reconstructed brick poses differ from recorded physics')
    lift = brick[:, 2] - settled_initial[2]
    lifted_grasp = bilateral & (lift >= .03)
    grasp_confirmed = longest_run(lifted_grasp) >= 3
    original_success = bool(record['metrics']['success'])
    if original_success:
        require(grasp_confirmed, f'{name}: apparent success lacks sustained elevated two-finger contacts')
    first_bilateral = int(np.flatnonzero(bilateral)[0]) if bilateral.any() else None
    result = {
        'id': name, 'original_success': original_success,
        'physical_grasp_confirmed': grasp_confirmed,
        'verified_pick_place_success': bool(original_success and grasp_confirmed),
        'offset_xy_m': offset.tolist(), 'offset_length_m': offset_length,
        'initial_policy_state_max_difference_degrees': state_difference,
        'maximum_lift_m': float(lift.max()),
        'mujoco_warning_counts': record['mujoco_warning_counts'],
        'contacts': {'left_finger_frames': int(np.count_nonzero(left_contact)),
            'right_finger_frames': int(np.count_nonzero(right_contact)),
            'bilateral_frames': int(bilateral.sum()),
            'first_bilateral_frame': first_bilateral,
            'longest_bilateral_run_frames': longest_run(bilateral),
            'maximum_lift_with_bilateral_contacts_m': float(lift[bilateral].max()) if bilateral.any() else None,
            'bilateral_frames_at_least_3cm_lift': int(lifted_grasp.sum()),
            'longest_elevated_bilateral_run_frames': longest_run(lifted_grasp),
            'contact_rule': 'MuJoCo active contact efc_address >= 0 and dist <= includemargin; target_brick simultaneously contacts gripper_left and gripper_right. Upstream finger collision margin is 0.15 mm.',
            'grasp_confirmation_rule': 'At least three consecutive saved 30 Hz frames with both finger contacts and >=30 mm lift'},
        'tcp_site_distances': {
            'minimum_to_initial_brick_center_xyz_m': float(np.linalg.norm(tcp - settled_initial, axis=1).min()),
            'minimum_to_initial_brick_center_xy_m': float(np.linalg.norm(tcp[:, :2] - settled_initial[:2], axis=1).min()),
            'minimum_to_current_brick_center_xyz_m': float(np.linalg.norm(tcp - brick, axis=1).min()),
            'site_definition': 'Upstream site tcp at gripper_end origin; palm reference, not the exact fingertip pinch point'},
        'evidence': {'scene_sha256': digest(scene_path), 'trace_sha256': digest(arrays_path),
            'input_artifact_sha256': input_hashes, 'output_artifact_sha256': output_hashes,
            'all_policy_input_keys': observed_keys, 'all_targets_exactly_from_saved_ACT_chunks': True,
            'all_subsequent_policy_states_match_previous_physics_frame': True,
            'loaded_six_model_hashes_match_protocol': True,
            'only_initial_object_position_differs_from_reference_scene': True},
    }
    return result, tcp, chunks


def analyze(output):
    report_path, protocol_path = output / 'generalization_report.json', output / 'protocol.json'
    report, protocol = read_json(report_path), read_json(protocol_path)
    require(report['status'] == 'completed_frozen_position_evaluation', 'Wait for all preregistered trials to complete')
    protocol_hash, source_report_hash = digest(protocol_path), digest(report_path)
    require(protocol_hash == report['protocol_sha256'] and protocol == report['protocol'], 'Protocol changed')
    require(report['training_steps'] == 0 and not report['teacher_action_sequence_read']
            and not report['object_truth_passed_to_policy'], 'Unexpected evaluator input/training declaration')
    frozen_hashes = {name: digest(Path(protocol['model_directory']) / name)
                     for name in protocol['model_files_sha256']}
    require(frozen_hashes == protocol['model_files_sha256'] == report['model_files_sha256_after'],
            'Checkpoint changed')
    current_code_hashes = {name: digest(ROOT / name) for name in protocol['code_files_sha256']}
    require(current_code_hashes == protocol['code_files_sha256'], 'Evaluation code changed')
    require(digest(protocol['successful_reference_scene']) == protocol['successful_reference_scene_sha256'],
            'Prior successful scene changed')
    config = read_json(Path(protocol['model_directory']) / 'config.json')
    require(set(config['input_features']) == {'observation.state', 'observation.images.wrist', 'observation.images.top'},
            'Model config contains unexpected inputs')
    baseline_state = np.array(protocol['successful_reference_initial_public_state'])
    by_id = {entry['id']: entry for entry in report['results']}
    require(set(by_id) == {trial['id'] for trial in protocol['trials']}, 'Missing/extra trial')
    diagnostics = []
    baseline_tcp = baseline_chunks = baseline_grasp_index = None
    for trial in protocol['trials']:
        result, tcp, chunks = audit_trial(output, by_id[trial['id']], trial, protocol, baseline_state)
        if trial['id'] == 'baseline':
            baseline_tcp, baseline_chunks = tcp.copy(), chunks.copy()
            baseline_grasp_index = result['contacts']['first_bilateral_frame']
        delta = chunks[0].astype(np.float64) - baseline_chunks[0]
        result['first_chunk_difference_from_baseline'] = {
            'six_arm_channels': differences(delta[:, :6]), 'gripper_channel': differences(delta[:, 6]),
            'per_channel_max_abs_degrees': np.max(np.abs(delta), axis=0).tolist(),
            'per_channel_mean_abs_degrees': np.mean(np.abs(delta), axis=0).tolist(),
        }
        tcp_difference = tcp - baseline_tcp
        result['tcp_trajectory_difference_from_baseline'] = {
            'rms_xyz_m': float(np.sqrt(np.mean(np.sum(tcp_difference ** 2, axis=1)))),
            'max_xyz_m': float(np.linalg.norm(tcp_difference, axis=1).max()),
            'rms_first200_frames_xyz_m': float(np.sqrt(np.mean(np.sum(tcp_difference[:200] ** 2, axis=1)))),
            'max_first200_frames_xyz_m': float(np.linalg.norm(tcp_difference[:200], axis=1).max()),
        }
        if baseline_grasp_index is not None:
            actual_delta = tcp_difference[baseline_grasp_index, :2]
            offset = np.asarray(trial['offset_xy_m'])
            projection = float(np.dot(actual_delta, offset) / np.dot(offset, offset)) if np.any(offset) else None
            result['tcp_trajectory_difference_from_baseline'].update({
                'at_baseline_first_bilateral_frame': baseline_grasp_index,
                'tcp_xy_shift_at_that_frame_m': actual_delta.tolist(),
                'requested_object_xy_shift_m': offset.tolist(),
                'projection_onto_requested_shift_fraction': projection,
                'xy_shift_error_m': float(np.linalg.norm(actual_delta - offset)),
                'interpretation_limit': 'Same-time palm-site displacement is a path response diagnostic, not a grasp target or control input',
            })
        diagnostics.append(result)
        print(json.dumps({'trial': trial['id'], 'success': result['verified_pick_place_success'],
            'bilateral_frames': result['contacts']['bilateral_frames'],
            'lift_with_contacts_m': result['contacts']['maximum_lift_with_bilateral_contacts_m'],
            'arm_chunk_max_deg': result['first_chunk_difference_from_baseline']['six_arm_channels']['max_abs_degrees'],
            'gripper_chunk_max_deg': result['first_chunk_difference_from_baseline']['gripper_channel']['max_abs_degrees']},
            ensure_ascii=False), flush=True)
    shifted = [result for result in diagnostics if result['id'] != 'baseline']
    require(digest(protocol_path) == protocol_hash and digest(report_path) == source_report_hash,
            'Original protocol/report changed during post-hoc analysis')
    after_model_hashes = {name: digest(Path(protocol['model_directory']) / name) for name in frozen_hashes}
    after_code_hashes = {name: digest(ROOT / name) for name in current_code_hashes}
    require(after_model_hashes == frozen_hashes and after_code_hashes == current_code_hashes,
            'Frozen model or evaluation code changed during analysis')
    verified_successes = sum(result['verified_pick_place_success'] for result in shifted)
    visual_response = all(result['first_chunk_difference_from_baseline']['six_arm_channels']['max_abs_degrees'] > 1e-6
                          for result in shifted)
    outcome = {'status': 'completed_independent_posthoc_audit',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'analysis_script_sha256': digest(__file__),
        'source_report_sha256': source_report_hash, 'protocol_sha256': protocol_hash,
        'protocol_and_original_report_unmodified': True,
        'model_files_sha256_after_analysis': after_model_hashes,
        'frozen_code_files_sha256_after_analysis': after_code_hashes,
        'baseline_grasp_confirmed': diagnostics[0]['physical_grasp_confirmed'],
        'baseline_pick_place_success': diagnostics[0]['verified_pick_place_success'],
        'new_position_trials': len(shifted),
        'new_position_verified_pick_place_successes': verified_successes,
        'independent_conclusion': {
            'genuine_frozen_ACT_inference_path_verified': True,
            'visual_response_with_identical_initial_robot_state': visual_response,
            'new_position_grasp_generalization_passed': verified_successes == len(shifted),
            'finding': (f'Baseline has sustained elevated two-finger contact and completed pick/place. '
                        f'{verified_successes}/{len(shifted)} shifted positions completed verified pick/place. '
                        'Initial robot states and frozen model files match. First predicted arm action chunks '
                        'change with the position-dependent images, so this is not a saved command-file replay. '
                        'Visual response alone does not demonstrate correct localization or reach correction.'),
            'failure_pattern': ('All shifted trials have zero sampled bilateral finger-contact frames; '
                               'one finger contacts the object, but the changed path does not place it reliably between both fingers.'
                               if all(result['contacts']['bilateral_frames'] == 0 for result in shifted) else
                               'Inspect per-trial contact, lift, and final-placement metrics.'),
        },
        'policy_input_contract': {'actual_npz_fields': ['state', 'wrist', 'top'],
            'robot_state_dimension': 7, 'camera_shapes_HWC': [480, 640, 3],
            'ground_truth_coordinates_present': False, 'teacher_actions_present': False,
            'path_audit': 'Frozen policy_worker validate_observation permits exactly these three fields; predict_chunk calls ACTPolicy.predict_action_chunk with saved pre/postprocessors. run_checkpoint derives each target only from predicted chunks. No teacher loader is on this path.',
            'scope': 'Code-path and artifact audit, not an operating-system file-access sandbox trace'},
        'trials': diagnostics, 'cpu_budget': CPU_BUDGET,
        'new_ACT_calls': 0, 'new_physics_steps': 0, 'training_steps': 0, 'hardware_io': False,
        'method': 'Only assign already-recorded qpos into disposable MjData and call mj_forward to recover geometry/contacts; zero velocity for geometry-only inspection. No mj_step, rollout, or policy invocation.',
        'limitations': ['Contacts sampled at 30 Hz and reconstructed geometrically; unsaved force histories are not inferred.',
            'TCP is the upstream palm reference site, not the exact pinch point.',
            'Five hand-selected positions with one trial each; not a statistical population success rate.',
            'New only relative to this one-position simulation adaptation; original real-data pretraining may include nearby positions.'],
    }
    target = output / 'position_diagnostics.json'
    with target.open('w', encoding='utf-8') as stream:
        json.dump(outcome, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.output_dir.resolve())
    print(json.dumps({'status': result['status'], 'baseline': result['baseline_pick_place_success'],
        'new_position_successes': result['new_position_verified_pick_place_successes'],
        'new_position_trials': result['new_position_trials']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
