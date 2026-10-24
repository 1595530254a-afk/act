"""Read-only phase/contact diagnosis of validation rollouts against teachers.

No test position, policy forward, training, or new physics step is used. The
teacher is privileged post-hoc diagnostic evidence, never an evaluation input.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np

from preview import load_model

BASE = ROOT / 'runs/mujoco_iteration_20261023/multisite_v1'
MODELS = ('validation_v1_step600', 'validation_v2_step600', 'validation_v2_step1200')
VALIDATION = ('val_-10_-10', 'val_-10_+10', 'val_+10_-10', 'val_+10_+10')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rotation_error(actual, desired):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(actual.T @ desired) - 1.) / 2., -1., 1.))))


def kinematics(model, poses):
    data = mujoco.MjData(model)
    tcp, rotation, left, right = [], [], [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        tcp.append(data.site('tcp').xpos.copy())
        rotation.append(data.body('gripper_end').xmat.reshape(3, 3).copy())
        contact_sides = set()
        for contact in data.contact[:data.ncon]:
            if contact.efc_address < 0 or contact.dist > contact.includemargin + 1e-9:
                continue
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ''
                     for g in (contact.geom1, contact.geom2)]
            if 'target_brick_geom' in names:
                contact_sides.update(side for side in ('left', 'right')
                                     if any(name.startswith(f'gripper_{side}_') for name in names))
        left.append('left' in contact_sides)
        right.append('right' in contact_sides)
    return np.asarray(tcp), np.asarray(rotation), np.asarray(left), np.asarray(right)


def monotonic_phase(states, teacher_states):
    # Compare only six arm joints, avoiding the contact-dependent gripper
    # blocking at the object width. No TCP assumption enters this phase fit.
    costs = ((states[:, None, :6] - teacher_states[None, :, :6]) ** 2).mean(axis=2)
    count, reference_count = costs.shape
    previous = np.full(reference_count, np.inf)
    previous[0] = costs[0, 0]
    back = np.zeros((count, reference_count), dtype=np.int16)
    for frame in range(1, count):
        options = np.full((4, reference_count), np.inf)
        for advance in range(4):
            options[advance, advance:] = previous[:reference_count - advance]
        step = options.argmin(axis=0)
        current = options[step, np.arange(reference_count)] + costs[frame]
        back[frame] = np.arange(reference_count) - step
        previous = current
    phase = np.empty(count, dtype=np.int64)
    phase[-1] = int(previous.argmin())
    for frame in range(count - 1, 0, -1):
        phase[frame - 1] = back[frame, phase[frame]]
    return phase, np.sqrt(costs[np.arange(count), phase])


def closure_event(public_gripper):
    opening_degrees = -np.asarray(public_gripper)
    # A full opening is followed by the first grasp closure; this excludes the
    # initially closed fingers and subsequent release/recovery attempts.
    peak = int(np.argmax(opening_degrees[110:210]) + 110)
    begin = next((frame for frame in range(peak + 1, min(250, len(opening_degrees)))
                  if opening_degrees[peak] - opening_degrees[frame] >= 5.), None)
    crossing = next((frame for frame in range(peak + 1, min(260, len(opening_degrees)))
                     if 2 * np.deg2rad(opening_degrees[frame]) * .009 - .000147 <= .040), None)
    return {'maximum_opening_frame': peak, 'closure_5deg_after_peak_frame': begin,
            'first_gap_at_most_40mm_after_peak_frame': crossing,
            'maximum_nominal_gap_mm': float((2 * np.deg2rad(opening_degrees[peak]) * .009 - .000147) * 1000)}


def at_event(frame, phase, state_error, tcp, rotations, teacher_tcp, teacher_rotations,
             state, teacher_state, target, teacher_action, actual_brick, teacher_initial_brick):
    if frame is None:
        return None
    matched = int(phase[frame])
    return {'rollout_frame_after_action': frame, 'teacher_phase_after_action': matched,
            'phase_lead_frames': matched - frame,
            'arm_joint_RMS_error_to_matched_teacher_deg': float(state_error[frame]),
            'tcp_residual_to_matched_teacher_xyz_mm': ((tcp[frame] - teacher_tcp[matched]) * 1000).tolist(),
            'tcp_relative_to_current_brick_residual_xyz_mm': (
                (tcp[frame] - actual_brick[frame]) - (teacher_tcp[matched] - teacher_initial_brick)).__mul__(1000).tolist(),
            'gripper_orientation_error_to_matched_teacher_deg': rotation_error(rotations[frame], teacher_rotations[matched]),
            'actual_gripper_public_deg': float(state[frame, 6]),
            'teacher_actual_gripper_at_matched_phase_deg': float(teacher_state[matched, 6]),
            'target_gripper_public_deg': float(target[frame, 6]),
            'teacher_target_gripper_at_matched_phase_deg': float(teacher_action[matched, 6])}


def main():
    teachers = {}
    for name in VALIDATION:
        path = BASE / 'teacher_v2' / name / 'trajectory.npz'
        manifest = json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8'))
        with np.load(path, allow_pickle=False) as arrays:
            saved = {key: arrays[key].copy() for key in arrays.files}
        model = load_model(Path(manifest['scene']['path']))
        tcp, rotations, left, right = kinematics(model, saved['qpos_after'])
        teachers[name] = (manifest, saved, model, tcp, rotations, left & right)
    results = []
    for model_name in MODELS:
        for name in VALIDATION:
            path = BASE / model_name / 'trials' / name / 'custom/rollout_arrays.npz'
            if not path.exists():
                continue
            result = json.loads((path.parent / 'result.json').read_text(encoding='utf-8'))
            with np.load(path, allow_pickle=False) as arrays:
                saved = {key: arrays[key].copy() for key in arrays.files}
            manifest, teacher, model, teacher_tcp, teacher_rotation, teacher_bilateral = teachers[name]
            state = saved['public_state']
            chunks = saved['action_chunks']
            target = chunks.reshape(-1, 7)[:len(state)]
            # Both state sequences used here are after action i, avoiding the
            # one-frame offset between teacher state (before) and rollout state.
            teacher_after = np.array(teacher['state'], np.float64)
            for index, joint in enumerate(range(1, 8)):
                address = model.joint(f'joint{joint}').qposadr[0]
                teacher_after[:, index] = -np.rad2deg(teacher['qpos_after'][:, address])
            teacher_after[:, 6] /= .009
            phase, state_error = monotonic_phase(state, teacher_after)
            tcp, rotations, left, right = kinematics(model, saved['qpos'])
            target_closure = closure_event(target[:, 6])
            actual_closure = closure_event(state[:, 6])
            teacher_target_closure = closure_event(teacher['action'][:, 6])
            teacher_actual_closure = closure_event(teacher_after[:, 6])
            teacher_initial_brick = np.asarray(manifest['task_metrics']['initial_position_m'])
            events = {}
            for event_name, frame in [('target_40mm', target_closure['first_gap_at_most_40mm_after_peak_frame']),
                                      ('actual_40mm', actual_closure['first_gap_at_most_40mm_after_peak_frame']),
                                      ('target_closure_start', target_closure['closure_5deg_after_peak_frame']),
                                      ('first_bilateral', int(np.flatnonzero(left & right)[0]) if (left & right).any() else None)]:
                events[event_name] = at_event(frame, phase, state_error, tcp, rotations,
                                              teacher_tcp, teacher_rotation, state, teacher_after,
                                              target, teacher['action'], saved['brick_position'], teacher_initial_brick)
            boundaries = []
            for query in range(len(chunks)):
                actual_state = saved['chunk_observation_state'][query]
                teacher_distances = np.sqrt(((teacher['state'][:, :6] - actual_state[:6]) ** 2).mean(axis=1))
                matched = int(teacher_distances.argmin())
                first = chunks[query, 0]
                entry = {'query_index': query, 'rollout_frame_before_action': query * 100,
                         'nearest_teacher_state_frame': matched,
                         'six_arm_joint_RMS_error_deg': float(teacher_distances[matched]),
                         'gripper_state_error_from_that_teacher_deg': float(actual_state[6] - teacher['state'][matched, 6]),
                         'next_first_target_minus_actual_state_public_deg': (first - actual_state).tolist(),
                         'next_first_target_minus_teacher_action_public_deg': (first - teacher['action'][matched]).tolist(),
                         'first_target_arm_RMS_jump_from_actual_deg': float(np.sqrt(((first[:6] - actual_state[:6]) ** 2).mean())),
                         'first_target_gripper_change_nominal_gap_mm': float(-2 * np.deg2rad(first[6] - actual_state[6]) * .009 * 1000)}
                if query:
                    entry['first_target_minus_last_target_public_deg'] = (first - chunks[query - 1, 99]).tolist()
                boundaries.append(entry)
            # Pregrasp interpolation residual is evaluated at matched phases,
            # before a successful lift can make trajectories diverge legitimately.
            pregrasp = (phase >= 155) & (phase <= 205)
            matched_error = (tcp - teacher_tcp[phase]) * 1000
            records = {
                'model': model_name, 'position': name, 'success': result['metrics']['success'],
                'trace_sha256': digest(path), 'teacher_trace_sha256': digest(BASE / 'teacher_v2' / name / 'trajectory.npz'),
                'phase_fitting': 'Monotonic least-squared six-arm-joint alignment, 0..3 teacher frames advanced per rollout frame; gripper excluded; both states after action.',
                'phase_at_frames_100_150_175_200_225_250_300': {str(i): int(phase[i]) for i in (100, 150, 175, 200, 225, 250, 300)},
                'phase_matched_pregrasp_tcp_xyz_mean_mm': matched_error[pregrasp].mean(axis=0).tolist(),
                'phase_matched_pregrasp_tcp_xyz_RMS_mm': np.sqrt((matched_error[pregrasp] ** 2).mean(axis=0)).tolist(),
                'phase_matched_pregrasp_arm_joint_RMS_mean_deg': float(state_error[pregrasp].mean()),
                'target_closure': target_closure, 'actual_closure': actual_closure,
                'teacher_target_closure': teacher_target_closure, 'teacher_actual_closure': teacher_actual_closure,
                'events': events, 'chunk_boundaries': boundaries,
                'teacher_first_bilateral_frame': int(np.flatnonzero(teacher_bilateral)[0]),
                'rollout_bilateral_frames': int((left & right).sum()),
            }
            results.append(records)
            print(json.dumps({'model': model_name, 'position': name, 'success': records['success'],
                              'phase': records['phase_at_frames_100_150_175_200_225_250_300'],
                              'target_40mm': events['target_40mm'],
                              'pregrasp_mean_residual_mm': records['phase_matched_pregrasp_tcp_xyz_mean_mm']}), flush=True)
    report = {'status': 'completed_read_only_validation_phase_diagnostics', 'entries': results,
              'test_positions_read': 0, 'new_physics_steps': 0, 'new_policy_forward_calls': 0,
              'training_steps': 0, 'hardware_io': False, 'cpu_budget': CPU_BUDGET,
              'limitations': ['Post-hoc phase correspondence is approximate and does not establish causal attribution.',
                              'Six-arm-state matching excludes gripper intentionally, because contact width changes its state.',
                              'No modified-execution or recovery-policy experiment has been run.']}
    (BASE / 'validation_phase_diagnostics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
