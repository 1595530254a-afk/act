"""Make position-correct synthetic teachers, validated by independent physics.

The source is the already verified 403-frame flat-block teacher. A six-joint
Jacobian IK translation offsets its approach, grasp and lift; transport returns
smoothly to the original release path. Only initial XML block XY is changed.
This is privileged simulation expert generation, never ACT evaluation. Images
can subsequently be rendered from qpos_before, without replaying the physics.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np

from export_sim_demonstration import public_to_native
from preview import load_model
from run_task import brick_metrics, check_physics
from task_dynamics import initialize, native_to_public, read_native_state, ServoController

SOURCE = ROOT / 'runs/mujoco_iteration_20261023/synthetic_demo_flat40_v1/manifest.json'
OUTPUT = ROOT / 'runs/mujoco_iteration_20261023/multisite_v1/teacher_v2'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def smoothstep(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def offset_fraction(frame):
    # Identical initial state, then displaced approach/grasp/lift. The block is
    # transported back above the same bin before the existing wide release.
    return float(smoothstep((frame - 90.) / 50.) * (1. - smoothstep((frame - 220.) / 55.)))


def positions():
    def item(name, split, x, y):
        return {'id': name, 'split': split, 'offset_xy_m': [x / 1000., y / 1000.]}
    train = [item(f'train_{x:+03d}_{y:+03d}', 'train', x, y)
             for x in (-40, -20, 0, 20, 40) for y in (-40, -20, 0, 20, 40)]
    validation = [item(f'val_{x:+03d}_{y:+03d}', 'validation', x, y)
                  for x in (-10, 10) for y in (-10, 10)]
    test = [{**item(name, 'test', x, y), 'subset': 'previous_diagnostic'} for name, x, y in (
        ('test_x_plus30', 30, 0), ('test_x_minus30', -30, 0),
        ('test_y_plus30', 0, 30), ('test_y_minus30', 0, -30),
        ('test_xy_plus30_minus40', 30, -40))]
    final = [{**item(f'final_{index:02d}', 'test', x, y), 'subset': 'final_held_out'}
             for index, (x, y) in enumerate(((-35, 15), (15, 35), (-25, -35),
                                             (35, 25), (-15, 25), (25, -15)))]
    return train + validation + test + final


def derive_scene(source_scene, output, offset):
    tree = ET.parse(source_scene)
    root = tree.getroot()
    compiler = root.find('compiler')
    meshes = (source_scene.parent / compiler.get('meshdir')).resolve()
    compiler.set('meshdir', os.path.relpath(meshes, output).replace('\\', '/'))
    for texture in root.findall('asset/texture'):
        if texture.get('file'):
            path = (source_scene.parent / texture.get('file')).resolve()
            texture.set('file', os.path.relpath(path, output).replace('\\', '/'))
    brick = root.find(".//body[@name='target_brick']")
    position = np.fromstring(brick.get('pos'), sep=' ')
    position[:2] += np.asarray(offset)
    brick.set('pos', ' '.join(f'{value:.12g}' for value in position))
    scene = output / 'task_scene.xml'
    ET.indent(tree, space='  ')
    tree.write(scene, encoding='utf-8', xml_declaration=True)
    return scene


def shifted_targets(model, source_targets, offset):
    """IK uses separate scratch data; it cannot move the physical rollout."""
    scratch = mujoco.MjData(model)
    qadr = np.array([model.joint(f'joint{i}').qposadr[0] for i in range(1, 7)])
    dofs = np.array([model.joint(f'joint{i}').dofadr[0] for i in range(1, 7)])
    lower = np.array([model.joint(f'joint{i}').range[0] for i in range(1, 7)])
    upper = np.array([model.joint(f'joint{i}').range[1] for i in range(1, 7)])
    tcp = model.site('tcp').id
    jac_position = np.zeros((3, model.nv))
    jac_rotation = np.zeros((3, model.nv))
    targets, residuals = [], []
    for frame, source in enumerate(source_targets):
        initialize(model, scratch, source)
        desired_position = scratch.site('tcp').xpos.copy()
        desired_position[:2] += np.asarray(offset) * offset_fraction(frame)
        desired_rotation = scratch.site('tcp').xmat.reshape(3, 3).copy()
        target = source.copy()
        for iteration in range(40):
            current_rotation = scratch.site('tcp').xmat.reshape(3, 3)
            positional_error = desired_position - scratch.site('tcp').xpos
            rotational_error = np.cross(current_rotation.T, desired_rotation.T).sum(axis=0) * .5
            if np.linalg.norm(positional_error) < 2e-6 and np.linalg.norm(rotational_error) < 2e-5:
                break
            mujoco.mj_jacSite(model, scratch, jac_position, jac_rotation, tcp)
            jacobian = np.vstack((jac_position[:, dofs], jac_rotation[:, dofs] * .15))
            error = np.r_[positional_error, rotational_error * .15]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 1e-8, error)
            # Small iterative step, preserving the source IK branch and limits.
            delta *= min(1., .08 / max(float(np.max(np.abs(delta))), 1e-12))
            target[:6] = np.clip(target[:6] + delta, lower, upper)
            scratch.qpos[qadr] = target[:6]
            mujoco.mj_forward(model, scratch)
        positional_error = np.linalg.norm(desired_position - scratch.site('tcp').xpos)
        rotational_error = np.linalg.norm(np.cross(scratch.site('tcp').xmat.reshape(3, 3).T,
                                                   desired_rotation.T).sum(axis=0) * .5)
        residuals.append([float(positional_error), float(rotational_error)])
        if positional_error > .00005 or rotational_error > .0005:
            raise RuntimeError(f'IK failed at frame {frame}: {residuals[-1]}')
        targets.append(target)
    return np.asarray(targets), np.asarray(residuals)


def active_contacts(model, data):
    sides, penetrations = set(), []
    for contact in data.contact[:data.ncon]:
        if contact.efc_address < 0 or contact.dist > contact.includemargin + 1e-9:
            continue
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ''
                 for geom in (contact.geom1, contact.geom2)]
        if 'target_brick_geom' in names:
            sides.update(side for side in ('left', 'right')
                         if any(name.startswith(f'gripper_{side}_') for name in names))
        if contact.dist < -.001:
            penetrations.append((' / '.join(names), float(contact.dist)))
    return len(sides) == 2, penetrations


def make_episode(output, item, source_manifest, source_targets):
    episode = output / item['id']
    episode.mkdir(exist_ok=False)
    source_scene = Path(source_manifest['scene']['path'])
    scene = derive_scene(source_scene, episode, item['offset_xy_m'])
    model = load_model(scene)
    targets, ik_residuals = shifted_targets(model, source_targets, item['offset_xy_m'])
    data = mujoco.MjData(model)
    initialize(model, data, source_targets[0])
    controller = ServoController(model)
    for _ in range(round(.5 / model.opt.timestep)):
        previous = data.time
        controller.step(data, source_targets[0])
        check_physics(data, previous)
    initial = data.body('target_brick').xpos.copy()
    start = data.time
    states, before, after, positions_after, errors, timestamps = [], [], [], [], [], []
    bilateral_frames, elevated_frames = [], []
    penetration_counts = Counter()
    minimum_distance = 0.
    peak_lift = 0.
    for frame, target in enumerate(targets):
        mujoco.mj_forward(model, data)
        states.append(native_to_public(read_native_state(model, data)).astype(np.float32))
        before.append(data.qpos.copy())
        timestamps.append(float(data.time - start))
        end = start + (frame + 1) / 30
        while data.time < end - 1e-10:
            previous = data.time
            controller.step(data, target)
            check_physics(data, previous)
        mujoco.mj_forward(model, data)
        position = data.body('target_brick').xpos.copy()
        lift = float(position[2] - initial[2])
        peak_lift = max(peak_lift, lift)
        positions_after.append(position)
        after.append(data.qpos.copy())
        errors.append(read_native_state(model, data) - target)
        bilateral, penetrations = active_contacts(model, data)
        if bilateral:
            bilateral_frames.append(frame)
            if lift >= .03:
                elevated_frames.append(frame)
        for name, distance in penetrations:
            penetration_counts[name] += 1
            minimum_distance = min(minimum_distance, distance)
        if data.warning.number.sum() or not np.isfinite(data.qpos).all():
            raise RuntimeError(f'Physics warning/nonfinite state: {item["id"]}, frame={frame}')
    metrics = brick_metrics(model, data, initial, {}, peak_lift)
    passed = metrics['success'] and bool(elevated_frames) and not penetration_counts
    public = np.array([native_to_public(target) for target in targets], dtype=np.float32)
    trajectory = episode / 'trajectory.npz'
    np.savez_compressed(trajectory, state=np.asarray(states, np.float32), action=public,
                        target_native=targets, qpos_before=before, qpos_after=after,
                        brick_position_after=positions_after, tracking_error_native=errors,
                        timestamp_before=timestamps, ik_residual_position_rotation=ik_residuals)
    report = deepcopy(source_manifest)
    report.update({
        'status': 'physical_teacher_probe_passed' if passed else 'physical_teacher_failed_NOT_usable',
        'dataset_id': 'b601_rs_multisite_flat40_synthetic_v1',
        'episode_id': item['id'], 'split': item['split'], 'offset_xy_m': item['offset_xy_m'],
        'subset': item.get('subset'),
        'frames': len(targets), 'expected_frames': len(targets), 'episodes': 1, 'fps': 30,
        'shards': [], 'total_shard_bytes': 0, 'physics_only_probe': True,
        'source_teacher': {'path': str(SOURCE), 'sha256': digest(SOURCE)},
        'teacher_method': 'source teacher TCP translation using six-joint Jacobian IK, unchanged TCP orientation and gripper schedule; translation fades during transport',
        'action_semantics': 'NEW simulation-only teacher: position-correct six-arm IK targets plus the source synthetic teacher gripper schedule; observation before control target. NOT unchanged human data.',
        'teacher_privileged_object_initial_xy_used': True,
        'teacher_NOT_ACT_or_independent_human_demonstration': True,
        'ACT_inference_queries': 0, 'policy_model_weights_used': False,
        'scene': {**source_manifest['scene'], 'path': str(scene), 'sha256': digest(scene)},
        'trajectory': {'file': trajectory.name, 'sha256': digest(trajectory)},
        'task_metrics': metrics, 'bilateral_contact_frames': bilateral_frames,
        'bilateral_frames_at_least_3cm_lift': elevated_frames,
        'penetrations_over_1mm_frame_pair_counts': penetration_counts.most_common(),
        'minimum_reported_penetration_distance_m': minimum_distance,
        'IK_max_position_error_m': float(ik_residuals[:, 0].max()),
        'IK_max_orientation_error_rad_approx': float(ik_residuals[:, 1].max()),
        'max_arm_target_difference_from_source_deg': np.rad2deg(np.abs(targets[:, :6] - source_targets[:, :6]).max(axis=0)).tolist(),
        'teacher_action_array_sha256': hashlib.sha256(public.tobytes()).hexdigest(),
        'max_tracking_error_native': np.max(np.abs(errors), axis=0).tolist(),
        'controller': controller.metadata(), 'warmup_s': start,
        'trajectory_simulation_s': float(data.time - start),
        'mujoco_warning_counts': data.warning.number.tolist(), 'cpu_budget': CPU_BUDGET,
        'contact_sheet': None, 'frame195_camera_red_pixel_counts': {},
        'limitations': ['Synthetic IK-derived variants of a single source episode, not independent human demonstrations.',
                        'Only approved approximate simulator; not real hardware validation.',
                        'RGB not exported yet; cannot train from this physics-only manifest.'],
    })
    report['teacher_release_override']['six_arm_target_values_unchanged'] = False
    save_json(episode / 'manifest.json', report)
    return {'id': item['id'], 'split': item['split'], 'offset_xy_m': item['offset_xy_m'],
            'subset': item.get('subset'),
            'manifest': str((episode / 'manifest.json').relative_to(output)),
            'manifest_sha256': digest(episode / 'manifest.json'), 'passed': bool(passed),
            'task_success': metrics['success'], 'maximum_lift_m': peak_lift,
            'bilateral_frames': len(bilateral_frames), 'elevated_bilateral_frames': len(elevated_frames),
            'penetration_pairs_over_1mm': sum(penetration_counts.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=('probe', 'all'), default='probe')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    trajectory = SOURCE.parent / source['trajectory']['file']
    if (source['status'] != 'verified_single_episode_simulation_teacher_export'
            or digest(trajectory) != source['trajectory']['sha256']
            or digest(Path(source['scene']['path'])) != source['scene']['sha256']):
        raise ValueError('Verified source teacher is missing or changed')
    with np.load(trajectory, allow_pickle=False) as arrays:
        source_targets = public_to_native(arrays['action'])
    candidates = positions()
    protocol = {
        'id': 'multisite_flat40_v1', 'positions': candidates,
        'source_teacher_manifest_sha256': digest(SOURCE),
        'frames_per_episode': 403, 'fps': 30,
        'train_positions': 25, 'validation_positions': 4, 'test_positions': 11,
        'previous_diagnostic_positions': 5, 'final_held_out_positions': 6,
        'split_by': 'whole initial block XY positions and episodes, never random video frames',
        'test_limit': 'Five positions were already inspected in previous failed frozen-policy evaluation; held out from this adaptation, not a previously untouched benchmark.',
        'only_successful_physical_experts_can_be_training_data': True,
        'offset_schedule': {'ramp_in_frames': [90, 140], 'full_shift_frames': [140, 220],
                            'ramp_out_frames': [220, 275], 'shape': 'cubic smoothstep'},
        'unchanged': ['flat40 dimensions', 'bin geometry/placement', 'camera geometry',
                      'initial robot state', 'servo gains', 'gripper action sequence'],
        'policy_input_contract': 'Only two rendered RGB cameras and seven actual public joint values; no block coordinates go to ACT.',
        'privileged_expert_generation': True, 'hardware_io': False, 'ACT_queries': 0,
    }
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Existing protocol differs; use a fresh output directory')
    else:
        save_json(protocol_path, protocol)
    probe_ids = {'train_+00_+00', 'test_x_plus30', 'test_x_minus30',
                 'test_y_plus30', 'test_y_minus30', 'test_xy_plus30_minus40'}
    selected = candidates if args.stage == 'all' else [item for item in candidates if item['id'] in probe_ids]
    results = []
    summary_path = output / 'physics_summary.json'
    if summary_path.exists():
        results = json.loads(summary_path.read_text(encoding='utf-8'))['episodes']
    completed = {result['id'] for result in results}
    for item in selected:
        if item['id'] in completed:
            continue
        try:
            result = make_episode(output, item, source, source_targets)
        except Exception as error:
            result = {**item, 'passed': False, 'error': f'{type(error).__name__}: {error}'}
        results.append(result)
        progress = {'status': 'physics_validation_in_progress',
                    'protocol_sha256': digest(protocol_path), 'episodes': results}
        save_json(summary_path, progress)
        save_json(output / 'manifest.json', progress)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {'status': 'physics_validation_completed', 'stage': args.stage,
               'protocol_sha256': digest(protocol_path), 'episodes': results,
               'passed_count': sum(item['passed'] for item in results),
               'total_count': len(results), 'hardware_io': False, 'ACT_queries': 0,
               'RGB_exported': False, 'training_started': False}
    save_json(summary_path, summary)
    save_json(output / 'manifest.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'episodes'}), flush=True)


if __name__ == '__main__':
    main()
