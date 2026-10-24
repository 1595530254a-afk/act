"""Measure the recorded grasp against fixed approximate scene reconstructions.

This diagnostic never runs ACT, changes a recorded target or attaches an object.
Only initial conditions are varied; assumptions are retained in the output.
"""

import argparse
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

from check_task_reference import vertices_world, grasp_geometry
from joint_mapping import dataset_state_to_sim, GRIPPER_METRES_PER_MOTOR_RAD, mapping_metadata
from preview import load_model, set_pose
from task_dynamics import initialize, read_native_state, ServoController, KP, KD
from run_task import brick_metrics


OUTPUT = ROOT / 'runs/mujoco_iteration_20261023/alignment'
SCENE = ROOT / 'runs/mujoco_task_20261023/scene/task_scene.xml'


class ReferenceServo(ServoController):
    def __init__(self, model, mode):
        super().__init__(model)
        self.mode = mode
        self.reference_velocity = np.zeros(7)

    def apply(self, data, target_native):
        if self.mode == 'baseline':
            return super().apply(data, target_native)
        bias = data.qfrc_bias[self.dof_addresses].copy()
        bias[6] += data.qfrc_bias[self.finger_dofs].sum()
        kp = KP * (4 if self.mode == 'gains4x' else 1)
        kd = KD * (2 if self.mode == 'gains4x' else 1)
        reference_velocity = (self.reference_velocity if self.mode.startswith('velocity') else 0)
        requested = kp * (target_native - data.qpos[self.qpos_addresses])
        requested += kd * (reference_velocity - data.qvel[self.dof_addresses]) + bias
        if self.mode == 'velocity_feedforward_passive_compensation':
            passive = data.qfrc_passive[self.dof_addresses].copy()
            passive[6] += data.qfrc_passive[self.finger_dofs].sum()
            requested -= passive
        if not np.isfinite(requested).all():
            raise FloatingPointError('Nonfinite servo force')
        limited = np.clip(requested, self.limits[:, 0], self.limits[:, 1])
        self.saturated_steps += (limited != requested)
        self.steps_applied += 1
        data.ctrl[self.actuator_ids] = limited
        return limited.copy()


def benchmark_scene(scene, poses, output):
    results = []
    for mode in ('baseline', 'gains4x', 'velocity_feedforward',
                 'velocity_feedforward_passive_compensation'):
        model = load_model(scene)
        data = mujoco.MjData(model)
        initialize(model, data, poses[0])
        servo = ReferenceServo(model, mode)
        for _ in range(round(.5 / model.opt.timestep)):
            servo.step(data, poses[0])
        initial = data.body('target_brick').xpos.copy()
        start = data.time
        qpos, positions, errors, bilateral, records = [], [], [], [], []
        for frame in range(403):
            end = start + (frame + 1) / 30
            next_frame = min(frame + 1, 402)
            while data.time < end - 1e-10:
                if mode.startswith('velocity'):
                    fraction = np.clip((data.time - start) * 30 - frame, 0, 1)
                    target = poses[frame] + fraction * (poses[next_frame] - poses[frame])
                    servo.reference_velocity = (poses[next_frame] - poses[frame]) * 30
                else:
                    target = poses[frame]
                servo.step(data, target)
            mujoco.mj_forward(model, data)
            if data.warning.number.sum() or not np.isfinite(data.qpos).all():
                raise RuntimeError(f'MuJoCo warning in {mode}')
            compared_target = poses[next_frame] if mode.startswith('velocity') else poses[frame]
            errors.append(read_native_state(model, data) - compared_target)
            qpos.append(data.qpos.copy())
            positions.append(data.body('target_brick').xpos.copy())
            sides = set()
            for contact in data.contact[:data.ncon]:
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ''
                         for g in (contact.geom1, contact.geom2)]
                if any(name.startswith('target_brick') for name in names):
                    for side in ('left', 'right'):
                        if any(name.startswith(f'gripper_{side}_') for name in names):
                            sides.add(side)
            if len(sides) == 2:
                bilateral.append(frame)
            if frame in (185, 190, 195, 200, 205, 207, 210, 220, 250, 280, 300, 402):
                record = geometry_record(model, data, frame)
                record['tracking_error_native'] = errors[-1].tolist()
                records.append(record)
        np.savez_compressed(output / f'{mode}.npz', qpos=qpos, brick_position=positions,
                            tracking_error_native=errors)
        results.append({
            'mode': mode, 'scene': str(scene), 'source': 'all7_original_episode0_state',
            'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
            'control_mode': 'linear_target_interpolation_plus_analytic_velocity' if mode.startswith('velocity') else '30Hz_zero_order_hold',
            'kp': (KP * (4 if mode == 'gains4x' else 1)).tolist(),
            'kd': (KD * (2 if mode == 'gains4x' else 1)).tolist(),
            'max_lift_m': float(np.max(np.asarray(positions)[:, 2]) - initial[2]),
            'bilateral_contact_frames': bilateral,
            'max_tracking_error_native': np.max(np.abs(errors), axis=0).tolist(),
            'grasp_window_rms_tracking_error_native': np.sqrt(np.mean(np.asarray(errors)[185:211] ** 2, axis=0)).tolist(),
            'saturated_steps_per_joint': servo.saturated_steps.tolist(),
            'final_brick_position_m': positions[-1].tolist(),
            'task_metrics': brick_metrics(model, data, initial, {}, float(np.max(np.asarray(positions)[:, 2]) - initial[2])),
            'frames': records, 'warnings': data.warning.number.tolist(),
            'position_assignments_after_initialization': False,
            'grasp_weld_or_attachment': False, 'hardware_io': False,
        })
        (output / 'controller_benchmark.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({key: results[-1][key] for key in ('mode', 'max_lift_m', 'bilateral_contact_frames', 'grasp_window_rms_tracking_error_native')}), flush=True)
    return results


def geometry_record(model, data, frame):
    record = grasp_geometry(model, data)
    record['frame'] = frame
    record['gripper_world_rotation'] = data.body('gripper_end').xmat.reshape(3, 3).tolist()
    record['finger_world_bounds'] = {}
    record['contacts'] = [
        {'names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                   for g in (contact.geom1, contact.geom2)],
         'distance_m': float(contact.dist), 'position_m': contact.pos.tolist()}
        for contact in data.contact[:data.ncon]]
    brick_geom_id = model.geom('target_brick_geom').id
    for side in ('left', 'right'):
        for part in ('front', 'mid', 'rear'):
            name = f'gripper_{side}_{part}_collision'
            vertices = vertices_world(model, data, name)
            segment = np.empty(6)
            distance = mujoco.mj_geomDistance(model, data, model.geom(name).id,
                                             brick_geom_id, 1.0, segment)
            origin = data.site('tcp').xpos
            rotation = data.body('gripper_end').xmat.reshape(3, 3)
            record['finger_world_bounds'][name] = {
                'min_m': vertices.min(axis=0).tolist(),
                'max_m': vertices.max(axis=0).tolist(),
                'closest_distance_m': float(distance),
                'closest_segment_world_m': segment.reshape(2, 3).tolist(),
                'closest_segment_tcp_m': ((segment.reshape(2, 3) - origin) @ rotation).tolist(),
            }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--scene', type=Path, default=SCENE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--square', action='store_true', help='Recompile approximate 32 x 32 x 20 mm square brick')
    parser.add_argument('--gripper-scale', type=float, default=1.0,
                        help='Explicit diagnostic scale relative to existing candidate; never hardware')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = load_model(SCENE)
    with np.load(ROOT / 'runs/mujoco_validation_20261023/inputs.npz', allow_pickle=False) as source:
        poses = dataset_state_to_sim(source['episode0_state'])
    poses[:, 6] *= args.gripper_scale
    if args.benchmark:
        args.output.mkdir(parents=True, exist_ok=True)
        scene = args.scene.resolve()
        if args.square:
            tree = ET.parse(scene)
            root = tree.getroot()
            meshdir = (scene.parent / root.find('compiler').get('meshdir')).resolve()
            root.find('compiler').set('meshdir', os.path.relpath(meshdir, args.output).replace('\\', '/'))
            for element in root.findall('asset/texture'):
                if element.get('file'):
                    filename = (scene.parent / element.get('file')).resolve()
                    element.set('file', os.path.relpath(filename, args.output).replace('\\', '/'))
            root.find(".//geom[@name='target_brick_geom']").set('size', '0.016 0.016 0.010')
            scene = args.output.resolve() / 'square_scene.xml'
            tree.write(scene, encoding='utf-8', xml_declaration=True)
        (args.output / 'assumptions.json').write_text(json.dumps({
            'all7_recorded_targets_unchanged_before_coordinate_conversion': True,
            'gripper_scale_vs_existing_candidate': args.gripper_scale,
            'existing_gripper_conversion_m_per_motor_rad': GRIPPER_METRES_PER_MOTOR_RAD,
            'effective_gripper_conversion_m_per_motor_rad': GRIPPER_METRES_PER_MOTOR_RAD * args.gripper_scale,
            'mapping_metadata': mapping_metadata(),
            'gripper_conversion_officially_verified_for_collection': False,
            'square_body_assumption': args.square,
            'square_height_20mm_measured_from_real_object': False,
            'scene_recompiled_before_any_physics': True,
            'object_pose_assignments_after_initialization': False,
            'hardware_io': False,
        }, indent=2) + '\n', encoding='utf-8')
        benchmark_scene(scene, poses, args.output)
        return
    data = mujoco.MjData(model)
    records = []
    for frame in (170, 180, 185, 190, 195, 200, 205, 207, 210, 215, 220):
        set_pose(model, data, poses[frame])
        record = geometry_record(model, data, frame)
        record['target_native'] = poses[frame].tolist()
        records.append(record)
    report = {
        'kind': 'unchanged_recorded_state_geometry_diagnostic_NOT_ACT',
        'scene': str(SCENE), 'frame_geometry': records,
        'retired_trial_warning': 'Earlier grasp_reference_v2 cases resized compiled model.geom_size in place; those collision results are not accepted as validation of a recompiled scene.',
        'object_pose_assigned_during_replay': False,
        'grasp_weld_or_attachment_used': False,
        'cpu_budget': CPU_BUDGET,
    }
    (OUTPUT / 'geometry.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'frames': len(records),
                      'output': str(OUTPUT / 'geometry.json')}))


if __name__ == '__main__':
    main()
