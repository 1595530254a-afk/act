"""Drive recorded demonstration states through the derived scene as a reference.

This is a physics alignment diagnostic, not an ACT rollout or a policy score.
The script does not change scene geometry, mapping, gains or object positions.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from joint_mapping import dataset_state_to_sim, mapping_metadata
from preview import font, load_model, overview_camera
from task_dynamics import initialize, read_native_state, ServoController

INPUTS = ROOT / 'runs/mujoco_validation_20261023/inputs.npz'
DEFAULT_OUTPUT = ROOT / 'runs/mujoco_task_20261023/reference_check'
FINGER_GEOMS = tuple(f'gripper_{side}_{part}_collision' for side in ('left', 'right')
                     for part in ('front', 'mid', 'rear'))
SNAPSHOTS = (0, 170, 180, 190, 195, 200, 205, 207, 210, 220, 240, 280, 300, 402)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def vertices_world(model, data, name):
    """Collision mesh/box vertices in world coordinates, after MuJoCo compile."""
    geom = model.geom(name)
    if geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(geom.dataid[0])
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start:start + count].astype(np.float64)
    elif geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
        half = geom.size
        vertices = np.array([[x, y, z] for x in (-half[0], half[0])
                             for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    else:
        raise ValueError(f'Unsupported diagnostic geometry: {name}')
    return vertices @ data.geom(name).xmat.reshape(3, 3).T + data.geom(name).xpos


def contact_pairs(model, data):
    pairs = []
    for contact in data.contact[:data.ncon]:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g)
                 for g in (contact.geom1, contact.geom2)]
        if any(name.startswith('target_brick') for name in names):
            pairs.append({'geoms': names, 'signed_distance_m': float(contact.dist)})
    return pairs


def grasp_geometry(model, data):
    origin = data.site('tcp').xpos.copy()
    rotation = data.body('gripper_end').xmat.reshape(3, 3)
    brick_id = model.geom('target_brick_geom').id
    brick_local = (vertices_world(model, data, 'target_brick_geom') - origin) @ rotation
    finger_bounds, distances = {}, {}
    for name in FINGER_GEOMS:
        local = (vertices_world(model, data, name) - origin) @ rotation
        finger_bounds[name] = {'min': local.min(axis=0).tolist(), 'max': local.max(axis=0).tolist()}
        distances[name] = float(mujoco.mj_geomDistance(model, data, model.geom(name).id,
                                                     brick_id, 1.0, None))
    left = finger_bounds['gripper_left_front_collision']
    right = finger_bounds['gripper_right_front_collision']
    # XML's fingers slide along +/- local Y. Use the nearest surfaces in that
    # known closing coordinate; the front meshes occupy local X[-30,0] mm.
    left_inner, right_inner = left['max'][1], right['min'][1]
    return {
        'tcp_world_m': origin.tolist(),
        'brick_centre_in_tcp_frame_m': ((data.body('target_brick').xpos - origin) @ rotation).tolist(),
        'brick_main_box_bounds_in_tcp_frame_m': {'min': brick_local.min(axis=0).tolist(),
                                                 'max': brick_local.max(axis=0).tolist()},
        'front_finger_inner_gap_m': float(right_inner - left_inner),
        'front_finger_inner_surfaces_tcp_y_m': [left_inner, right_inner],
        'brick_main_box_projected_width_tcp_y_m': float(np.ptp(brick_local[:, 1])),
        'finger_collision_mesh_bounds_in_tcp_frame_m': finger_bounds,
        'finger_to_main_brick_signed_distances_m': distances,
        'distance_cap_m': 1.0,
        'note': 'Collision-mesh geometry only; TCP is the source site at the finger tips. No grasp transform is fitted.'}


def save_snapshot(renderer, model, data, frame, output):
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    camera = overview_camera()
    camera.lookat[:] = [.29, .04, .08]
    camera.azimuth, camera.elevation, camera.distance = 115, -40, .90
    canvas = Image.new('RGB', (960, 560), '#101820')
    renderer.update_scene(data, camera=camera, scene_option=options)
    canvas.paste(Image.fromarray(renderer.render()), (0, 38))
    for name, offset in [('overhead_rgb', 38), ('wrist_rgb', 278)]:
        renderer.update_scene(data, camera=name, scene_option=options)
        canvas.paste(Image.fromarray(renderer.render()).resize((320, 240)), (640, offset))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'示范轨迹物理重放 · 第 {frame} 帧 · 不调用 ACT', fill='white', font=font(19))
    draw.text((12, 526), '用于检查场景和夹爪几何是否对齐；没有修改物体位置或绑定夹爪', fill='#ffcc72', font=font(15))
    path = output / f'grasp_frame{frame}.png'
    canvas.save(path)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = args.scene.resolve()
    with np.load(INPUTS, allow_pickle=False) as source:
        poses = dataset_state_to_sim(source['episode0_state'])
        timestamps = source['episode0_timestamp'].copy()
    if poses.shape != (403, 7) or not np.allclose(np.diff(timestamps), 1 / 30, atol=2e-6):
        raise ValueError('Expected the verified 403-frame, 30 Hz episode 0')
    model, data = load_model(scene), None
    data = mujoco.MjData(model)
    initialize(model, data, poses[0])
    controller = ServoController(model)
    for _ in range(round(.5 / model.opt.timestep)):
        controller.step(data, poses[0])
    if data.warning.number.sum():
        raise RuntimeError('MuJoCo warning during initial settling')
    initial = data.body('target_brick').xpos.copy()
    start = data.time
    renderer = mujoco.Renderer(model, height=480, width=640)
    records, full_states, targets, errors, positions, tcp_positions = [], [], [], [], [], []
    contacts, frame_indices = Counter(), []
    peak = float(initial[2])
    try:
        for frame, target in enumerate(poses):
            end_time = start + (frame + 1) / 30
            while data.time < end_time - 1e-10:
                previous_time = data.time
                controller.step(data, target)
                if data.time <= previous_time or data.warning.number.sum():
                    raise RuntimeError('MuJoCo warning or automatic reset during reference replay')
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError('Non-finite reference physics state')
            # Refresh geometry at the final integrated qpos before measuring it.
            mujoco.mj_forward(model, data)
            actual = read_native_state(model, data)
            error = actual - target
            brick_position = data.body('target_brick').xpos.copy()
            peak = max(peak, float(brick_position[2]))
            pairs = contact_pairs(model, data)
            finger_pairs = [pair for pair in pairs if any(name.startswith('gripper_') for name in pair['geoms'])]
            if finger_pairs:
                frame_indices.append(frame)
            contacts.update(' / '.join(pair['geoms']) for pair in finger_pairs)
            errors.append(error)
            targets.append(target)
            full_states.append(data.qpos.copy())
            positions.append(brick_position)
            tcp_positions.append(data.site('tcp').xpos.copy())
            if frame in SNAPSHOTS:
                record = {'frame': frame, 'source_timestamp_s': float(timestamps[frame]),
                          'simulation_time_s': float(data.time - start),
                          'target_native': target.tolist(), 'actual_native': actual.tolist(),
                          'tracking_error_native': error.tolist(),
                          'brick_position_world_m': brick_position.tolist(),
                          'brick_contacts': pairs, 'grasp_geometry': grasp_geometry(model, data)}
                if frame in (190, 200, 207):
                    record['image'] = save_snapshot(renderer, model, data, frame, output)
                records.append(record)
    finally:
        renderer.close()
    errors = np.asarray(errors)
    np.savez_compressed(output / 'reference_states.npz', target_native=np.asarray(targets),
                        qpos=np.asarray(full_states), tracking_error_native=errors,
                        brick_position=np.asarray(positions), tcp_position=np.asarray(tcp_positions))
    report = {
        'status': 'reference_replay_completed',
        'kind': 'recorded_episode0_state_targets_through_physics_NOT_ACT',
        'scene_path': str(scene), 'scene_sha256': digest(scene),
        'inputs_path': str(INPUTS), 'inputs_sha256': digest(INPUTS),
        'frames': len(poses), 'source_hz': 30, 'warmup_s': float(start),
        'simulated_trajectory_s': float(data.time - start),
        'mapping': mapping_metadata(), 'controller': controller.metadata(),
        'maximum_abs_tracking_error_native': np.abs(errors).max(axis=0).tolist(),
        'median_abs_tracking_error_native': np.median(np.abs(errors), axis=0).tolist(),
        'tracking_error_units': ['rad'] * 6 + ['m'],
        'initial_brick_position_m': initial.tolist(),
        'final_brick_position_m': positions[-1].tolist(),
        'maximum_brick_lift_m': float(peak - initial[2]),
        'brick_lifted_3cm': bool(peak - initial[2] >= .03),
        'brick_finger_contact_frame_count': len(frame_indices),
        'brick_finger_contact_frame_indices': frame_indices,
        'brick_finger_contact_pair_counts': contacts.most_common(),
        'mujoco_warning_counts': data.warning.number.tolist(),
        'representative_frames': records,
        'interpretation': [
            'This diagnostic replays measured demonstration states as targets; it does not run an ACT model.',
            'No pickup here means the reconstructed scene and candidate coordinate/zero/gripper convention are not validated by the reference motion.',
            'The source TCP site is at the front-finger tips; using it as an object centre is only an unverified placement assumption.',
            'Finger gaps, projected brick widths and signed distances quantify possible one-sided or absent contact; this script does not tune them to force a grasp.',
            'A failed ACT rollout in this unvalidated reconstruction cannot by itself establish that the model weights are defective.'
        ],
        'scene_or_mapping_modified': False, 'object_pose_assigned_during_replay': False,
        'grasp_weld_or_attachment_used': False, 'actual_ACT_inference_queries': 0,
        'hardware_io': False, 'cpu_budget': CPU_BUDGET}
    (output / 'reference_tracking.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: report[name] for name in ('status', 'frames', 'maximum_brick_lift_m',
        'brick_finger_contact_frame_count', 'maximum_abs_tracking_error_native',
        'mujoco_warning_counts')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
