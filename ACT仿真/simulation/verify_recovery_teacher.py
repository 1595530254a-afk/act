"""Audit recovery dataset provenance and saved physics, without new rollouts."""
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
from generate_multisite_teacher import active_contacts
from generate_recovery_teacher import geometry_contract
from joint_mapping import model_action_to_sim
from preview import load_model
from run_position_generalization import model_hashes
from run_task import brick_metrics, digest, save_json
from task_dynamics import read_public_state


def bilateral(model, poses):
    data = mujoco.MjData(model)
    values = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        values.append(bool(active_contacts(model, data)[0]))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-dir', type=Path, required=True)
    args = parser.parse_args()
    base = args.teacher_dir.resolve()
    manifest_path, protocol_path = base / 'manifest.json', base / 'protocol.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    manifest_hash = digest(manifest_path)
    assert digest(protocol_path) == manifest['protocol_sha256']
    assert protocol['student_actions_before_branch'] == 200 and protocol['recovery_time_scale'] == .4
    assert model_hashes(Path(protocol['student_model'])) == protocol['student_model_files_sha256']
    assert {item['id'] for item in protocol['points']} == {item['id'] for item in manifest['attempts']}
    assert len(manifest['attempts']) == 25
    assert all(item['split'] == 'train' for item in manifest['attempts'])
    records = []
    for item in manifest['attempts']:
        directory = base / item['id']
        if item['status'] == 'skipped_already_bilateral_grasp':
            source = Path(item['student_rollout_path'])
            assert digest(source) == item['student_rollout_sha256']
            with np.load(source, allow_pickle=False) as arrays:
                last = arrays['qpos'][-3:].copy()
            model = load_model(directory / 'scene/task_scene.xml')
            observed = bilateral(model, last)
            assert observed == [True, True, True]
            assert item['recovery_physics_executed'] is False and not (directory / 'expert').exists()
            records.append({'id': item['id'], 'status': 'verified_skip_already_bilateral',
                            'last_three_bilateral': observed})
            continue
        episode_path = base / item['manifest']
        assert digest(episode_path) == item['manifest_sha256']
        episode = json.loads(episode_path.read_text(encoding='utf-8'))
        assert episode['status'] == 'physical_teacher_probe_passed'
        assert episode['privileged_training_expert'] and episode['expert_NOT_ACT_success']
        scene, source = Path(episode['scene']['path']), Path(episode['student_prefix']['rollout_path'])
        source_result_path = Path(episode['student_prefix']['result_path'])
        assert digest(scene) == episode['scene']['sha256']
        assert digest(source) == episode['student_prefix']['rollout_sha256']
        assert digest(source_result_path) == episode['student_prefix']['result_sha256']
        source_result = json.loads(source_result_path.read_text(encoding='utf-8'))
        assert source_result['policy_actions_applied'] == 200 and source_result['actual_inference_queries'] == 2
        for index, inference in enumerate(source_result['inference']):
            assert digest(source.parent / f'observation_{index:03d}.npz') == inference['input_sha256']
            assert digest(source.parent / f'chunk_{index:03d}.npz') == inference['output_sha256']
        with np.load(source, allow_pickle=False) as arrays:
            prefix = {key: arrays[key].copy() for key in arrays.files}
        expected = np.asarray([model_action_to_sim(prefix['action_chunks'][i // 100, i % 100]) for i in range(200)])
        assert np.array_equal(expected, prefix['target_native'])
        model = load_model(scene)
        verified_geometry = geometry_contract(model)
        for key, value in verified_geometry.items():
            assert episode['scene'][key] == value
        assert not all(bilateral(model, prefix['qpos'][-3:]))
        snapshot = Path(episode['student_prefix']['snapshot_path'])
        assert digest(snapshot) == episode['student_prefix']['snapshot_sha256']
        with np.load(snapshot, allow_pickle=False) as arrays:
            assert np.array_equal(arrays['qpos'], prefix['qpos'][-1]) and not arrays['qvel'].any()
        trajectory_path = episode_path.parent / episode['trajectory']['file']
        assert digest(trajectory_path) == episode['trajectory']['sha256']
        with np.load(trajectory_path, allow_pickle=False) as arrays:
            trajectory = {key: arrays[key].copy() for key in arrays.files}
        state, action = trajectory['state'], trajectory['action']
        before, after = trajectory['qpos_before'], trajectory['qpos_after']
        assert state.shape == action.shape == (210, 7) and state.dtype == action.dtype == np.float32
        assert np.array_equal(before[0], prefix['qpos'][-1])
        assert np.array_equal(before[1:], after[:-1])
        assert np.max(np.abs(trajectory['timestamp_before'] - np.arange(210) / 30)) <= .00101
        assert episode['retiming']['student_plus_recovery_frames'] == 410 <= 450
        assert np.max(np.abs(np.array([model_action_to_sim(a) for a in action]) - trajectory['target_native'])) < 1e-6
        data = mujoco.MjData(model)
        contact, elevated, state_error, deep_count = [], [], 0., 0
        for frame in range(210):
            data.qpos[:] = before[frame]
            mujoco.mj_forward(model, data)
            state_error = max(state_error, float(np.max(np.abs(read_public_state(model, data).astype(np.float32) - state[frame]))))
            data.qpos[:] = after[frame]
            mujoco.mj_forward(model, data)
            contact_now, deep = active_contacts(model, data)
            contact.append(contact_now)
            lift = data.body('target_brick').xpos[2] - episode['task_metrics']['initial_position_m'][2]
            elevated.append(contact_now and lift >= .03)
            deep_count += len(deep)
        assert state_error == 0. and deep_count == 0
        assert np.flatnonzero(contact).tolist() == episode['bilateral_contact_frames']
        assert np.flatnonzero(elevated).tolist() == episode['elevated_bilateral_contact_frames']
        assert longest_run(np.asarray(elevated)) >= 3
        data.qvel[:] = trajectory['qvel_after'][-1]
        mujoco.mj_forward(model, data)
        metrics = brick_metrics(model, data, np.array(episode['task_metrics']['initial_position_m']), {},
                                episode['task_metrics']['maximum_lift_m'])
        assert metrics['success'] and metrics['final_finger_contacts'] == 0
        records.append({'id': item['id'], 'status': 'verified_physical_recovery',
                        'student_source_ACT_hashes_and_applied_actions_verified': True,
                        'fresh_branch_initial_qpos_exact_match': True, 'initial_qvel_reset_explicitly_zero': True,
                        'state_vs_qpos_before_error_deg': state_error, 'trajectory_continuity_exact': True,
                        'recovery_frames': 210, 'student_plus_recovery_frames': 410,
                        'elevated_bilateral_frames': int(np.count_nonzero(elevated)),
                        'sampled_penetration_pairs_over_1mm': deep_count,
                        'recomputed_final_success': metrics['success']})
        print(json.dumps(records[-1]), flush=True)
    assert digest(manifest_path) == manifest_hash
    report = {'status': 'passed_independent_recovery_dataset_audit', 'records': records,
              'source_manifest_sha256': manifest_hash, 'protocol_sha256': digest(protocol_path),
              'valid_recovery_episodes': len(manifest['episodes']),
              'skipped_already_bilateral_grasp': manifest['skipped_already_bilateral_grasp'],
              'snapshot_action_count': 200, 'retime_scale': .4, 'student_plus_recovery_actions': 410,
              'all_positions_train_only': True, 'new_physics_steps': 0, 'new_ACT_queries': 0,
              'training_steps': 0, 'hardware_io': False, 'cpu_budget': CPU_BUDGET,
              'limitation': 'Source snapshots omit actual velocities; branch explicitly starts at zero velocity. Contact depths checked at saved 30 Hz states.'}
    save_json(base / 'independent_trace_verification.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}), flush=True)


if __name__ == '__main__':
    main()
