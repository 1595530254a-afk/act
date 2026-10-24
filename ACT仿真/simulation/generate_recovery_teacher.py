"""Generate privileged recovery experts from frozen student training snapshots.

Student actions are genuine frozen ACT outputs. Recovery is a NEW independent
physics branch initialized from saved full qpos, with qvel explicitly zeroed
because the source rollout does not save velocities. No active rollout object
is teleported, welded or attached. These are synthetic TRAINING experts, never
ACT success claims or validation/test data.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np

from analyze_position_generalization import longest_run
from export_sim_demonstration import public_to_native
from generate_multisite_teacher import active_contacts, smoothstep
from joint_mapping import dataset_state_to_sim
from preview import load_model
from run_checkpoint_comparison import create_scene, run_without_video, SOURCE
from run_position_generalization import model_hashes, normalized_scene_tree
from run_task import brick_metrics, check_physics, digest, save_json
from task_dynamics import native_to_public, read_native_state, ServoController

BASE = ROOT / 'runs/mujoco_iteration_20261023/multisite_v1'
MODEL = BASE / 'adapt_v1/checkpoints/step_000600/pretrained_model'
REFERENCE = ROOT / 'runs/mujoco_iteration_20261023/synthetic_demo_flat40_v1'
POINTS = (
    {'id': 'train_-40_-40', 'split': 'train', 'offset_xy_m': [-.040, -.040]},
    {'id': 'train_+00_+00', 'split': 'train', 'offset_xy_m': [0., 0.]},
    {'id': 'train_+40_+40', 'split': 'train', 'offset_xy_m': [.040, .040]},
)


class PoseIK:
    def __init__(self, model, initial_qpos):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = initial_qpos
        self.qadr = np.array([model.joint(f'joint{i}').qposadr[0] for i in range(1, 7)])
        self.dofs = np.array([model.joint(f'joint{i}').dofadr[0] for i in range(1, 7)])
        self.lower = np.array([model.joint(f'joint{i}').range[0] for i in range(1, 7)])
        self.upper = np.array([model.joint(f'joint{i}').range[1] for i in range(1, 7)])
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros_like(self.jp)
        self.tcp = model.site('tcp').id
        mujoco.mj_forward(model, self.data)

    def solve(self, position, rotation, gripper):
        for _ in range(60):
            dp = position - self.data.site('tcp').xpos
            dr = np.cross(self.data.site('tcp').xmat.reshape(3, 3).T, rotation.T).sum(axis=0) * .5
            if np.linalg.norm(dp) < 2e-6 and np.linalg.norm(dr) < 2e-5:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.tcp)
            jac = np.vstack((self.jp[:, self.dofs], self.jr[:, self.dofs] * .15))
            change = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-8, np.r_[dp, dr * .15])
            change *= min(1., .08 / max(float(np.max(np.abs(change))), 1e-12))
            self.data.qpos[self.qadr] = np.clip(self.data.qpos[self.qadr] + change, self.lower, self.upper)
            mujoco.mj_forward(self.model, self.data)
        error = [float(np.linalg.norm(position - self.data.site('tcp').xpos)),
                 float(np.linalg.norm(np.cross(self.data.site('tcp').xmat.reshape(3, 3).T,
                                               rotation.T).sum(axis=0) * .5))]
        if error[0] > .00005 or error[1] > .0005:
            raise RuntimeError(f'Recovery IK failed: {error}')
        return np.r_[self.data.qpos[self.qadr], gripper], error


def blend_rotation(first, second, fraction):
    qa, qb, matrix = np.empty(4), np.empty(4), np.empty(9)
    mujoco.mju_mat2Quat(qa, first.reshape(-1))
    mujoco.mju_mat2Quat(qb, second.reshape(-1))
    if np.dot(qa, qb) < 0:
        qb *= -1
    quaternion = qa * (1. - fraction) + qb * fraction
    quaternion /= np.linalg.norm(quaternion)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def geometry_contract(model):
    brick_dimensions = (model.geom('target_brick_geom').size[:3] * 2).tolist()
    east, west = model.geom('target_bin_wall_e'), model.geom('target_bin_wall_w')
    north, south = model.geom('target_bin_wall_n'), model.geom('target_bin_wall_s')
    outside = [east.pos[0] + east.size[0] - west.pos[0] + west.size[0],
               north.pos[1] + north.size[1] - south.pos[1] + south.size[1],
               north.pos[2] + north.size[2]]
    walls = [east.size[0] * 2, west.size[0] * 2, north.size[1] * 2, south.size[1] * 2,
             model.geom('target_bin_floor').size[2] * 2]
    if (not np.allclose(brick_dimensions, [.04, .04, .02], atol=1e-12, rtol=0)
            or not np.allclose(outside, [.12, .12, .05], atol=1e-12, rtol=0)
            or not np.allclose(walls, [.004] * 5, atol=1e-12, rtol=0)):
        raise ValueError('Actual compiled scene geometry violates approved fixed props')
    if any((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or '').startswith('target_brick_stud_')
           for index in range(model.ngeom)):
        raise ValueError('Expected flat block without studs')
    for name in ('d405_wrist_mount', 'd405_camera_body', 'd405_camera_front',
                 'd405_left_imager', 'd405_right_imager'):
        if model.geom(name).group[0] != 4:
            raise ValueError('Expected unchanged wrist camera own-housing group 4')
    return {'fixed_project_brick_dimensions_m': np.round(brick_dimensions, 9).tolist(),
            'fixed_project_bin_outside_dimensions_m': np.round(outside, 9).tolist(),
            'fixed_project_bin_wall_m': round(float(walls[0]), 9),
            'flat_top_brick': True, 'stud_height_m': 0.,
            'geometry_verified_from_compiled_model': True}


def recovery_targets(model, snapshot, teacher, skip_initial_clearance=False):
    ik = PoseIK(model, snapshot)
    current_position = ik.data.site('tcp').xpos.copy()
    current_rotation = ik.data.site('tcp').xmat.reshape(3, 3).copy()
    current_gripper = read_native_state(model, ik.data)[6]
    current_brick = ik.data.body('target_brick').xpos.copy()
    current_brick_rotation = ik.data.body('target_brick').xmat.reshape(3, 3).copy()
    reference_data = mujoco.MjData(model)
    reference_data.qpos[:] = teacher['qpos_after'][195]
    mujoco.mj_forward(model, reference_data)
    source_brick_rotation = reference_data.body('target_brick').xmat.reshape(3, 3)
    yaw_difference = np.arctan2(current_brick_rotation[1, 0], current_brick_rotation[0, 0])
    yaw_difference -= np.arctan2(source_brick_rotation[1, 0], source_brick_rotation[0, 0])
    c, s = np.cos(yaw_difference), np.sin(yaw_difference)
    yaw_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    relative_pinch = reference_data.site('tcp').xpos - reference_data.body('target_brick').xpos
    grasp_position = current_brick + yaw_rotation @ relative_pinch
    grasp_rotation = yaw_rotation @ reference_data.site('tcp').xmat.reshape(3, 3)
    open_gripper = float(public_to_native([0, 0, 0, 0, 0, 0, -170.])[6])
    closed_gripper = float(public_to_native([0, 0, 0, 0, 0, 0, -98.8])[6])
    raised = current_position.copy()
    raised[2] = max(raised[2] + .06, grasp_position[2] + .08)
    approach = grasp_position + [0, 0, .07]
    lifted = grasp_position + [0, 0, .08]
    tail_target = public_to_native(teacher['action'][250])
    reference_data.qpos[:] = teacher['qpos_after'][0]
    for index, name in enumerate(f'joint{i}' for i in range(1, 8)):
        reference_data.qpos[model.joint(name).qposadr[0]] = tail_target[index]
    mujoco.mj_forward(model, reference_data)
    tail_position = reference_data.site('tcp').xpos.copy()
    tail_rotation = reference_data.site('tcp').xmat.reshape(3, 3).copy()
    steps = [
        ('open_at_actual_pose', 30, current_position, current_rotation, open_gripper),
        ('raise_clearance', 45, raised, current_rotation, open_gripper),
        ('align_above_current_actual_block', 75, approach, grasp_rotation, open_gripper),
        ('descend_open', 45, grasp_position, grasp_rotation, open_gripper),
        ('close_at_aligned_pose', 40, grasp_position, grasp_rotation, closed_gripper),
        ('lift_closed', 45, lifted, grasp_rotation, closed_gripper),
        ('transport_to_original_bin_path', 90, tail_position, tail_rotation, float(tail_target[6])),
    ]
    if skip_initial_clearance:
        steps = steps[2:]
    targets, errors, phases = [], [], []
    phase_metadata = []
    previous_p, previous_r, previous_g = current_position, current_rotation, current_gripper
    for name, count, position, rotation, gripper in steps:
        start = len(targets)
        for index in range(count):
            fraction = float(smoothstep((index + 1) / count))
            p = previous_p * (1. - fraction) + position * fraction
            r = blend_rotation(previous_r, rotation, fraction)
            g = previous_g * (1. - fraction) + gripper * fraction
            target, error = ik.solve(p, r, g)
            targets.append(target)
            errors.append(error)
            phases.append(name)
        phase_metadata.append({'name': name, 'first_frame': start, 'last_frame': len(targets) - 1,
                               'target_tcp_xyz_m': position.tolist(), 'target_gripper_native_m': gripper})
        previous_p, previous_r, previous_g = position, rotation, gripper
    # Source tail is a previously verified transport/release, not ACT. The
    # preceding IK path ends at its first exact target TCP pose and orientation.
    tail = public_to_native(teacher['action'][250:])
    phase_metadata.append({'name': 'original_teacher_bin_release_tail', 'source_frames': [250, 402],
                           'first_frame': len(targets), 'last_frame': len(targets) + len(tail) - 1})
    targets.extend(tail)
    errors.extend([[0., 0.]] * len(tail))
    phases.extend(['original_teacher_bin_release_tail'] * len(tail))
    return np.asarray(targets), np.asarray(errors), phases, phase_metadata


def run_recovery(scene, snapshot, teacher, output, time_scale=1., student_actions=150):
    output.mkdir(exist_ok=False)
    model = load_model(scene)
    verified_geometry = geometry_contract(model)
    targets, ik_errors, _, phase_metadata = recovery_targets(model, snapshot, teacher)
    original_frame_count = len(targets)
    if time_scale != 1.:
        sample = np.linspace(0., len(targets) - 1., int(np.ceil(len(targets) * time_scale)))
        targets = np.stack([np.interp(sample, np.arange(len(targets)), targets[:, channel])
                            for channel in range(7)], axis=1)
        for phase in phase_metadata:
            first, last = phase['first_frame'], phase['last_frame']
            phase['original_first_frame'], phase['original_last_frame'] = first, last
            phase['first_frame'] = int(np.searchsorted(sample, first, side='left'))
            phase['last_frame'] = int(np.searchsorted(sample, last, side='right') - 1)
        if student_actions + len(targets) > 450:
            raise ValueError('Retimed student plus recovery exceeds fixed 15 second action budget')
    clearance_last_frame = next(phase['last_frame'] for phase in phase_metadata if phase['name'] == 'raise_clearance')
    expert_replan = None
    data = mujoco.MjData(model)
    # Fresh branch reset only. All 16 source coordinates, including the object's
    # free joint, are copied exactly before the first new physics step.
    data.qpos[:] = snapshot
    data.qvel[:] = 0.
    data.ctrl[:] = 0.
    mujoco.mj_forward(model, data)
    if not np.array_equal(data.qpos, snapshot):
        raise ValueError('Fresh branch failed to preserve saved full qpos')
    initial = data.body('target_brick').xpos.copy()
    controller = ServoController(model)
    states, before, after, position, velocities, times = [], [], [], [], [], []
    contact_frames, elevated_frames = [], []
    penetrations = Counter()
    minimum_distance, peak = 0., 0.
    for frame, target in enumerate(targets):
        states.append(native_to_public(read_native_state(model, data)).astype(np.float32))
        before.append(data.qpos.copy())
        times.append(float(data.time))
        end = (frame + 1) / 30
        while data.time < end - 1e-10:
            previous = data.time
            controller.step(data, target)
            check_physics(data, previous)
        mujoco.mj_forward(model, data)
        current_position = data.body('target_brick').xpos.copy()
        position.append(current_position)
        after.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        lift = float(current_position[2] - initial[2])
        peak = max(peak, lift)
        bilateral, deep = active_contacts(model, data)
        if bilateral:
            contact_frames.append(frame)
            if lift >= .03:
                elevated_frames.append(frame)
        for pair, distance in deep:
            penetrations[pair] += 1
            minimum_distance = min(minimum_distance, distance)
        if data.warning.number.sum():
            raise RuntimeError(f'Physics warning at recovery frame {frame}')
        if frame == clearance_last_frame:
            # Opening a previously held block may let it settle to the table.
            # Query its NEW actual pose only for this privileged training
            # expert, and plan from the actual raised robot rather than using
            # the old pre-release block height or teleporting it.
            replanned, next_ik, _, next_phases = recovery_targets(
                model, data.qpos.copy(), teacher, skip_initial_clearance=True)
            first = frame + 1
            remaining = len(targets) - first
            sample = np.linspace(0., len(replanned) - 1., remaining)
            targets[first:] = np.stack([np.interp(sample, np.arange(len(replanned)), replanned[:, channel])
                                        for channel in range(7)], axis=1)
            for phase in next_phases:
                lower, upper = phase['first_frame'], phase['last_frame']
                phase['original_first_frame'], phase['original_last_frame'] = lower, upper
                phase['first_frame'] = first + int(np.searchsorted(sample, lower, side='left'))
                phase['last_frame'] = first + int(np.searchsorted(sample, upper, side='right') - 1)
            phase_metadata = [phase for phase in phase_metadata if phase['last_frame'] <= frame] + next_phases
            ik_errors = np.concatenate((ik_errors, next_ik))
            expert_replan = {'after_recovery_frame': frame, 'reason': 'Use actual settled block pose after opening and clearance',
                             'actual_block_center_m': data.body('target_brick').xpos.tolist(),
                             'actual_full_qpos': data.qpos.tolist(), 'policy_received_object_truth': False}
    metrics = brick_metrics(model, data, initial, {}, peak)
    elevated_mask = np.zeros(len(targets), dtype=bool)
    elevated_mask[elevated_frames] = True
    sustained = longest_run(elevated_mask)
    passed = bool(metrics['success'] and sustained >= 3 and not penetrations)
    trajectory = output / 'trajectory.npz'
    public_action = np.asarray([native_to_public(target) for target in targets], np.float32)
    np.savez_compressed(trajectory, state=states, action=public_action, target_native=targets,
                        qpos_before=before, qpos_after=after, qvel_after=velocities,
                        brick_position_after=position, timestamp_before=times,
                        pre_retime_IK_residual_position_rotation=ik_errors)
    report = {'status': 'physical_teacher_probe_passed' if passed else 'physical_recovery_failed_NOT_usable',
              'expert_NOT_ACT_success': True, 'privileged_training_expert': True,
              'expert_kind': 'recovery_from_student_snapshot',
              'scene': {'path': str(scene), 'sha256': digest(scene), **verified_geometry},
              'trajectory': {'file': trajectory.name, 'sha256': digest(trajectory)},
              'frames': len(targets), 'fps': 30, 'phases': phase_metadata,
              'retiming': {'time_scale': time_scale, 'original_plan_frames': original_frame_count,
                           'retimed_frames': len(targets), 'fps_unchanged': 30,
                           'method': 'Linearly resample native joint targets over the entire plan including original release tail, then rerun physical servo; NOT video acceleration.',
                           'student_plus_recovery_frames': student_actions + len(targets),
                           'student_plus_recovery_s': (student_actions + len(targets)) / 30,
                           'fixed_evaluation_budget_frames': 450},
              'privileged_expert_replan_after_clearance': expert_replan,
              'reset_semantics': 'Independent fresh MjData reset from saved full qpos snapshot; qvel set to zero because student recorder omits velocities.',
              'branch_initial_full_qpos_exact_match': True, 'branch_initial_qvel': 'all_zero_not_saved_actual_velocity',
              'post_initialization_object_pose_assignment': False, 'grasp_weld_or_attachment': False,
              'wrist_camera_self_housing_group_4_hidden': True,
              'top_camera_self_housing_group_4_hidden': False,
              'task_metrics': metrics, 'bilateral_contact_frames': contact_frames,
              'elevated_bilateral_contact_frames': elevated_frames,
              'longest_elevated_bilateral_run': sustained,
              'penetrations_over_1mm_frame_pair_counts': penetrations.most_common(),
              'minimum_reported_contact_distance_m': minimum_distance,
              'mujoco_warning_counts': data.warning.number.tolist(), 'controller': controller.metadata(),
              'IK_max_error_position_orientation': ik_errors.max(axis=0).tolist(),
              'ACT_queries_in_expert': 0, 'training_steps': 0, 'hardware_io': False,
              'cpu_budget': CPU_BUDGET,
              'limitations': ['Velocity-reset branch is not uninterrupted continuation of the live student physics.',
                              'Privileged object state is allowed for training expert generation only.',
                              'Contact penetration sampled at recorded 30 Hz states.']}
    save_json(output / 'manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'recovery_probe')
    parser.add_argument('--all-training-positions', action='store_true')
    parser.add_argument('--recovery-time-scale', type=float, choices=(.4, .45, .5, 1.), default=1.)
    parser.add_argument('--student-prefix-actions', type=int, choices=(150, 200), default=150)
    parser.add_argument('--reuse-student-prefix-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    frozen['duration_s'] = args.student_prefix_actions / 30
    fit = json.loads(Path(frozen['camera_fit_path']).read_text(encoding='utf-8'))
    normalized = normalized_scene_tree(ET.parse(frozen['successful_reference_scene']).getroot())
    initial = dataset_state_to_sim(frozen['initial_public_joint_target_degrees'])
    identity = model_hashes(MODEL)
    points = list(POINTS)
    if args.all_training_positions:
        points = [{'id': f'recovery_train_{x:+03d}_{y:+03d}',
                   'source_train_position_id': f'train_{x:+03d}_{y:+03d}',
                   'split': 'train', 'offset_xy_m': [x / 1000., y / 1000.]}
                  for x in (-40, -20, 0, 20, 40) for y in (-40, -20, 0, 20, 40)]
    protocol = {'student_model': str(MODEL), 'student_model_files_sha256': identity,
                'points': points, 'student_actions_before_branch': args.student_prefix_actions,
                'snapshot_frame_action_count': args.student_prefix_actions,
                'admission_rule': 'If all last three recorded student frames have active bilateral target-brick finger contact, skip recovery as skipped_already_bilateral_grasp; never deliberately release an already stable grasp.',
                'student_executed_actions_per_chunk': 100,
                'recovery_time_scale': args.recovery_time_scale,
                'reuse_student_prefix_roots': [str(path.resolve()) for path in args.reuse_student_prefix_root],
                'generator_script_sha256': digest(Path(__file__)),
                'branch_qvel_reset_to_zero': True, 'split': 'train',
                'validation_or_test_positions_used': False,
                'expert_is_privileged_training_data_NOT_ACT_success': True,
                'training_steps': 0, 'hardware_io': False}
    save_json(output / 'protocol.json', protocol)
    with np.load(REFERENCE / 'trajectory.npz', allow_pickle=False) as arrays:
        teacher = {key: arrays[key].copy() for key in arrays.files}
    reports = []
    for point in points:
        directory = output / point['id']
        scene, metadata = create_scene(directory, point, frozen, fit, normalized)
        student_directory = None
        for reuse_root in args.reuse_student_prefix_root:
            for name in (point['id'], point.get('source_train_position_id'), point['id'].removeprefix('recovery_')):
                if name is None:
                    continue
                candidate = reuse_root.resolve() / name / 'student'
                if (candidate / 'result.json').exists() and (candidate / 'rollout_arrays.npz').exists():
                    candidate_result = json.loads((candidate / 'result.json').read_text(encoding='utf-8'))
                    if (candidate_result['status'] == 'completed'
                            and candidate_result['policy_actions_applied'] == args.student_prefix_actions
                            and candidate_result['model_sha256'] == identity['model.safetensors']):
                        old_scene = candidate.parent / 'scene/task_scene.xml'
                        expected_xy = np.asarray(frozen['reference_brick_center_m'])[:2] + point['offset_xy_m']
                        old_position = np.fromstring(ET.parse(old_scene).getroot().find(
                            ".//body[@name='target_brick']").get('pos'), sep=' ')
                        if not np.allclose(old_position[:2], expected_xy, atol=1e-10, rtol=0):
                            raise ValueError('Reusable student prefix came from a different object position')
                        student_directory = candidate
                        break
            if student_directory is not None:
                break
        if student_directory is None:
            student_directory = directory / 'student'
            student = run_without_video(scene, metadata, student_directory, MODEL, initial, frozen)
        else:
            student = json.loads((student_directory / 'result.json').read_text(encoding='utf-8'))
        if student['status'] != 'completed' or student['policy_actions_applied'] != args.student_prefix_actions:
            raise RuntimeError(f'Student prefix failed: {point["id"]}')
        with np.load(student_directory / 'rollout_arrays.npz', allow_pickle=False) as arrays:
            snapshot = arrays['qpos'][-1].copy()
            last_three_poses = arrays['qpos'][-3:].copy()
        eligibility_model = load_model(scene)
        eligibility_data = mujoco.MjData(eligibility_model)
        last_three_bilateral = []
        for pose in last_three_poses:
            eligibility_data.qpos[:] = pose
            mujoco.mj_forward(eligibility_model, eligibility_data)
            last_three_bilateral.append(bool(active_contacts(eligibility_model, eligibility_data)[0]))
        np.savez_compressed(directory / 'branch_initial_snapshot.npz', qpos=snapshot,
                            qvel=np.zeros(eligibility_model.nv), source_student_frame=np.int64(args.student_prefix_actions - 1))
        if all(last_three_bilateral):
            report = {**point, 'status': 'skipped_already_bilateral_grasp', 'passed': False,
                      'last_three_student_frames_bilateral': last_three_bilateral,
                      'student_rollout_path': str(student_directory / 'rollout_arrays.npz'),
                      'student_rollout_sha256': digest(student_directory / 'rollout_arrays.npz'),
                      'recovery_physics_executed': False}
            reports.append(report)
            progress = {'status': 'recovery_probe_in_progress', 'attempts': reports,
                        'episodes': [item for item in reports if item.get('passed')],
                        'protocol_sha256': digest(output / 'protocol.json')}
            save_json(output / 'manifest.json', progress)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            continue
        try:
            expert = run_recovery(scene, snapshot, teacher, directory / 'expert',
                                  args.recovery_time_scale, args.student_prefix_actions)
            expert.update({'episode_id': point['id'], 'split': 'train', 'offset_xy_m': point['offset_xy_m'],
                           'episodes': 1, 'student_prefix': {
                               'model_directory': str(MODEL), 'model_files_sha256': identity,
                               'actions_executed': args.student_prefix_actions,
                               'last_frame_index': args.student_prefix_actions - 1,
                               'rollout_path': str(student_directory / 'rollout_arrays.npz'),
                               'rollout_sha256': digest(student_directory / 'rollout_arrays.npz'),
                               'result_path': str(student_directory / 'result.json'),
                               'result_sha256': digest(student_directory / 'result.json'),
                               'reused_recorded_ACT_prefix': student_directory != directory / 'student',
                               'snapshot_path': str(directory / 'branch_initial_snapshot.npz'),
                               'snapshot_sha256': digest(directory / 'branch_initial_snapshot.npz')},
                           'admission': {'last_three_student_frames_bilateral': last_three_bilateral,
                                         'eligible': True, 'stable_existing_grasp_would_be_skipped': True},
                           'object_truth_usage': 'Only privileged training expert labels from current actual object pose; never sent to student ACT.',
                           'observation_timing': 'Actual state and eventual rendered images BEFORE each expert target.',
                           'action_semantics': 'New privileged recovery IK targets and explicit original-teacher transport/release tail, all public degrees; NOT ACT predictions.',
                           'shards': [], 'physics_only_probe': True,
                           'teacher_tail_source': {'path': str(REFERENCE / 'trajectory.npz'),
                                                   'sha256': digest(REFERENCE / 'trajectory.npz'),
                                                   'frames_inclusive': [250, 402]}})
            expert_manifest = directory / 'expert/manifest.json'
            save_json(expert_manifest, expert)
            report = {**point, 'manifest': str(expert_manifest.relative_to(output)),
                      'manifest_sha256': digest(expert_manifest),
                      'status': expert['status'], 'task_metrics': expert['task_metrics'],
                      'passed': expert['status'] == 'physical_teacher_probe_passed',
                      'elevated_bilateral_run_frames': expert['longest_elevated_bilateral_run'],
                      'penetration_pairs_over_1mm': expert['penetrations_over_1mm_frame_pair_counts']}
        except Exception as error:
            report = {**point, 'status': 'recovery_generation_failed', 'error': f'{type(error).__name__}: {error}'}
        reports.append(report)
        save_json(output / 'manifest.json', {'status': 'recovery_probe_in_progress', 'attempts': reports,
                                           'episodes': [item for item in reports if item.get('passed')],
                                           'protocol_sha256': digest(output / 'protocol.json')})
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if model_hashes(MODEL) != identity:
        raise ValueError('Student checkpoint or processors changed')
    summary = {'status': 'recovery_probe_completed', 'attempts': reports,
               'episodes': [item for item in reports if item.get('passed')],
               'protocol_sha256': digest(output / 'protocol.json'),
               'strict_successes': sum(item['status'] == 'physical_teacher_probe_passed' for item in reports),
               'skipped_already_bilateral_grasp': sum(item['status'] == 'skipped_already_bilateral_grasp' for item in reports),
               'total_probes': len(reports),
               'student_ACT_queries_in_source_prefixes': len(reports) * 2,
               'actual_student_ACT_queries_this_run': sum(
                   2 for point in points if (output / point['id'] / 'student/result.json').exists()),
               'expert_ACT_queries': 0, 'training_steps': 0, 'hardware_io': False,
               'validation_or_test_positions_used': False, 'cpu_budget': CPU_BUDGET}
    save_json(output / 'manifest.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'episodes'}), flush=True)


if __name__ == '__main__':
    main()
