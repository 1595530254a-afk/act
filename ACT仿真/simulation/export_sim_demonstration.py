"""Export one verified physical simulation demonstration for visual adaptation.

Teacher actions derive from the public episode 0 measured STATE sequence,
replayed as 30 Hz position targets. An explicit flag adds a wider release for
the fixed 40 mm brick, producing a NEW synthetic teacher. Actions are not the
public dataset's original action column. Observations precede each target.
No policy, hardware interface, object attachment or post-reset teleport is used.
"""

import argparse
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

from joint_mapping import DIRECTIONS, GRIPPER_METRES_PER_MOTOR_RAD, mapping_metadata
from preview import load_model
from run_task import brick_metrics, check_physics
from task_dynamics import initialize, read_native_state, ServoController
from task_scene import build_scene


INPUTS = ROOT / 'runs/mujoco_validation_20261023/inputs.npz'
CAMERA_FIT = ROOT / 'runs/mujoco_iteration_20261023/camera_fit/camera_fit.json'
DEFAULT_OUTPUT = ROOT / 'runs/mujoco_iteration_20261023/synthetic_demo_flat40_v1'
GRIPPER_METRES_PER_RAD = 0.009
SHARD_FRAMES = 32
CAMERAS = {'wrist': 'wrist_rgb', 'top': 'overhead_rgb'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def public_to_native(public):
    native = -np.deg2rad(np.asarray(public, np.float64))
    native[..., 6] *= GRIPPER_METRES_PER_RAD
    return native


def read_public(model, data):
    native = read_native_state(model, data)
    native[6] /= GRIPPER_METRES_PER_RAD
    return -np.rad2deg(native)


def write_shard(output, index, cache):
    filename = f'episode000_shard{index:03d}.npz'
    path = output / filename
    arrays = {key: np.asarray(values) for key, values in cache.items()}
    np.savez_compressed(path, **arrays)
    return {'file': filename, 'sha256': digest(path), 'size_bytes': path.stat().st_size,
            'first_frame': int(arrays['frame_index'][0]),
            'last_frame': int(arrays['frame_index'][-1]),
            'frames': len(arrays['frame_index']),
            'shapes': {key: list(value.shape) for key, value in arrays.items()},
            'dtypes': {key: str(value.dtype) for key, value in arrays.items()}}


def contact_sheet(output, success):
    sheet = Image.new('RGB', (1280, 4 * 510 + 45), '#111820')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f'FLAT 40 mm teacher | genuine contact physics | success={success}', fill='white')
    for row, frame in enumerate((0, 195, 240, 300)):
        for column, camera in enumerate(CAMERAS):
            y = 45 + row * 510
            sheet.paste(Image.open(output / f'frame{frame:03d}_{camera}.png'), (column * 640, y))
            draw.text((column * 640 + 10, y + 485), f'{camera} | frame {frame} | observation BEFORE target', fill='white')
    sheet.save(output / 'contact_sheet.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--physics-only', action='store_true', help='Verify teacher physics before rendering images')
    parser.add_argument('--flat40-release-teacher', action='store_true',
                        help='Explicitly create a new synthetic teacher with a wider gripper release over the bin')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists() or list(output.glob('episode000_shard*.npz')):
        raise FileExistsError('Use a fresh output directory; existing exports are not overwritten or mixed')
    if tuple(DIRECTIONS) != (-1.0,) * 7:
        raise ValueError('This export requires the documented all-negative canonical RS mapping')
    fit = json.loads(CAMERA_FIT.read_text(encoding='utf-8'))
    if fit['status'] != 'inferred_attached_camera_candidate':
        raise ValueError('No accepted inferred virtual wrist camera candidate')
    scene, scene_metadata = build_scene(output / 'scene', square_brick=True,
                                        square_edge_m=.040, flat_brick=True, wrist_camera=fit)
    save_json(output / 'scene_metadata.json', scene_metadata)
    model = load_model(scene)
    data = mujoco.MjData(model)
    with np.load(INPUTS, allow_pickle=False) as source:
        source_public = source['episode0_state'].astype(np.float32)
        source_timestamps = source['episode0_timestamp'].copy()
    if source_public.shape != (403, 7) or not np.allclose(np.diff(source_timestamps), 1 / 30, atol=2e-6):
        raise ValueError('Expected precisely one verified 403-frame 30 Hz source state episode')
    teacher_public = source_public.copy()
    if args.flat40_release_teacher:
        # Source episode released a smaller object at a nominal gap ~36 mm.
        # The fixed 40 mm project brick needs a NEW synthetic teacher release.
        # Keep all six arm targets and every earlier grasp target unchanged.
        teacher_public[278:291, 6] = np.linspace(source_public[278, 6], -170., 13)
        teacher_public[291:311, 6] = -170.
    targets = public_to_native(teacher_public)
    if not np.allclose(-np.rad2deg(targets[:, :6]), teacher_public[:, :6]):
        raise ValueError('Arm coordinate round trip failed')
    initialize(model, data, targets[0])
    controller = ServoController(model)
    for _ in range(round(.5 / model.opt.timestep)):
        previous = data.time
        controller.step(data, targets[0])
        check_physics(data, previous)
    initial_position = data.body('target_brick').xpos.copy()
    start_time = data.time
    report = {
        'status': 'physical_probe_in_progress' if args.physics_only else 'export_in_progress_not_yet_usable',
        'dataset_id': 'b601_rs_synthetic_demo_flat40mm_episode0_replay_v1',
        'episodes': 1, 'expected_frames': 403, 'frames': 0, 'fps': 30,
        'observation_timing': 'render and read actual robot state BEFORE applying teacher target i',
        'action_semantics': 'new teacher action derives from public episode0 observation.state[i], used as a 30 Hz controller position target; NOT the original dataset action column. Any gripper release override is explicitly recorded below.',
        'new_human_demonstration': False, 'independent_demonstrations': 0,
        'source_dataset': {'repo_id': 'nikodembartnik/rebot_cube',
                           'revision': 'c5cdae89d2e1f474a8e2f1ac5f16302126d0c33f',
                           'episode': 0, 'column_used_as_teacher': 'observation.state'},
        'source_inputs': {'path': str(INPUTS), 'sha256': digest(INPUTS)},
        'source_camera_fit': {'path': str(CAMERA_FIT), 'sha256': digest(CAMERA_FIT),
                              'hardware_hand_eye_calibration': False},
        'scene': {'path': str(scene), 'sha256': digest(scene),
                  'metadata_sha256': digest(output / 'scene_metadata.json'),
                  'scene_id': scene_metadata['scene_id'],
                  'dimensions_measured_from_real_object': False,
                  'fixed_project_brick_dimensions_m': [.040, .040, .020],
                  'flat_top_brick': True, 'fixed_project_stud_grid': [0, 0], 'stud_height_m': 0.0,
                  'fixed_project_bin_outside_dimensions_m': [.120, .120, .050],
                  'fixed_project_bin_wall_m': .004,
                  'fixed_project_bin_inside_dimensions_m': [.112, .112, .046]},
        'physics_model': {'mujoco_version': mujoco.__version__,
                          'arm_commit': scene_metadata['upstream_arm_commit'],
                          'arm_xml_sha256': scene_metadata['upstream_arm_sha256'],
                          'upstream_manifest_sha256': digest(ROOT / 'simulation/upstream_manifest.json')},
        'policy_model_weights_used': False, 'ACT_inference_queries': 0,
        'mapping': {'directions': list(DIRECTIONS),
                    'effective_gripper_metres_per_motor_rad': GRIPPER_METRES_PER_RAD,
                    'imported_default_gripper_metres_per_motor_rad': GRIPPER_METRES_PER_MOTOR_RAD,
                    'scale_relative_to_imported_default': GRIPPER_METRES_PER_RAD / GRIPPER_METRES_PER_MOTOR_RAD,
                    'hardware_calibration_verified': False,
                    'imported_candidate_metadata': mapping_metadata()},
        'format': {'wrist': 'uint8 RGB [N,3,480,640]', 'top': 'uint8 RGB [N,3,480,640]',
                   'state': 'float32 [N,7] actual canonical public joint degrees before teacher target',
                   'action': 'float32 [N,7] new teacher position target in same public degrees',
                   'qpos': 'float64 [N,nq] actual simulation coordinates before action',
                   'frame_index': 'int64 [N]', 'timestamp': 'float64 [N] seconds since episode observation 0'},
        'shards': [], 'geometry_group_3_hidden': True, 'all_site_groups_hidden': True,
        'wrist_camera_self_housing_group_4_hidden': True,
        'top_camera_self_housing_group_4_hidden': False,
        'frame195_camera_red_pixel_counts': {},
        'post_initialization_object_pose_assignment': False,
        'grasp_weld_or_attachment': False, 'recorded_public_target_overrides': bool(args.flat40_release_teacher),
        'teacher_release_override': ({
            'channel': 'gripper.pos', 'frames_inclusive': [278, 310],
            'ramp_frames_inclusive': [278, 290], 'hold_frames_inclusive': [291, 310],
            'open_public_deg': -170., 'nominal_inner_gap_m': float(2 * np.deg2rad(170.) * GRIPPER_METRES_PER_RAD - .000147),
            'basis': 'Original episode release gap about 36 mm is narrower than the project-fixed flat 40 mm brick. Widen only while the unchanged arm path is over the bin.',
            'six_arm_target_values_unchanged': True, 'grasp_targets_before_frame278_unchanged': True,
            'NOT_original_unmodified_human_demonstration': True,
        } if args.flat40_release_teacher else None),
        'teacher_action_array_sha256': hashlib.sha256(teacher_public.tobytes()).hexdigest(),
        'physics_only_probe': bool(args.physics_only),
        'hardware_io': False, 'cpu_budget': CPU_BUDGET,
        'limitations': ['One deterministic replay in one approximate reconstructed scene; no variation or generalization evidence.',
                       'Visual observations are simulated; teacher targets derive from the same author episode used for reconstruction.',
                       'Usable only for a bounded visual-adaptation/overfit diagnostic after physical success passes; not an independent benchmark.'],
    }
    save_json(output / 'manifest.json', report)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    renderer = None if args.physics_only else mujoco.Renderer(model, height=480, width=640)
    cache = {key: [] for key in ('wrist', 'top', 'state', 'action', 'qpos', 'frame_index', 'timestamp')}
    states, qpos_before, qpos_after, positions, errors, times = [], [], [], [], [], []
    peak_lift = 0.0
    bilateral_frames = []
    try:
        for frame, target in enumerate(targets):
            mujoco.mj_forward(model, data)
            actual_public = read_public(model, data).astype(np.float32)
            if not args.physics_only:
                cache['state'].append(actual_public)
                cache['action'].append(teacher_public[frame])
                cache['qpos'].append(data.qpos.copy())
                cache['frame_index'].append(np.int64(frame))
                cache['timestamp'].append(np.float64(data.time - start_time))
            for key, camera in (() if args.physics_only else CAMERAS.items()):
                # The fitted virtual optical centre falls inside the approximate
                # D405 visual housing. Hide only that sensor's own shell from
                # its wrist image; retain it in the separate overhead view.
                options.geomgroup[4] = 0 if key == 'wrist' else 1
                renderer.update_scene(data, camera=camera, scene_option=options)
                rgb = renderer.render()
                if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                    raise ValueError('Unexpected camera RGB shape/dtype')
                cache[key].append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
                if frame == 195:
                    channels = rgb.astype(np.float32)
                    red = ((channels[..., 0] > 80)
                           & (channels[..., 0] > channels[..., 1] * 1.6)
                           & (channels[..., 0] > channels[..., 2] * 1.6))
                    report['frame195_camera_red_pixel_counts'][key] = int(red.sum())
                    if red.sum() < 80:
                        raise ValueError(f'{key} camera failed pregrasp red-pixel visibility check')
                if frame in (0, 195, 240, 300):
                    Image.fromarray(rgb).save(output / f'frame{frame:03d}_{key}.png')
            states.append(actual_public)
            qpos_before.append(data.qpos.copy())
            times.append(data.time - start_time)
            end_time = start_time + (frame + 1) / 30
            while data.time < end_time - 1e-10:
                previous = data.time
                controller.step(data, target)
                check_physics(data, previous)
            mujoco.mj_forward(model, data)
            if data.warning.number.sum() or not np.isfinite(data.qpos).all():
                raise RuntimeError(f'Invalid physics at frame {frame}')
            position = data.body('target_brick').xpos.copy()
            peak_lift = max(peak_lift, float(position[2] - initial_position[2]))
            positions.append(position)
            qpos_after.append(data.qpos.copy())
            errors.append(read_native_state(model, data) - target)
            contact_sides = set()
            for contact in data.contact[:data.ncon]:
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ''
                         for geom in (contact.geom1, contact.geom2)]
                if any(name.startswith('target_brick') for name in names):
                    contact_sides.update(side for side in ('left', 'right')
                                         if any(name.startswith(f'gripper_{side}_') for name in names))
            if len(contact_sides) == 2:
                bilateral_frames.append(frame)
            if not args.physics_only and (len(cache['frame_index']) == SHARD_FRAMES or frame == len(targets) - 1):
                shard = write_shard(output, len(report['shards']), cache)
                report['shards'].append(shard)
                report['frames'] += shard['frames']
                save_json(output / 'manifest.json', report)
                cache = {key: [] for key in cache}
                print(json.dumps({'status': 'shard_saved', 'frames': report['frames'],
                                  'total_frames': len(targets), 'file': shard['file']}), flush=True)
        report['task_metrics'] = brick_metrics(model, data, initial_position, scene_metadata, peak_lift)
        report['bilateral_contact_frames'] = bilateral_frames
        report['controller'] = controller.metadata()
        report['warmup_s'] = float(start_time)
        report['trajectory_simulation_s'] = float(data.time - start_time)
        report['max_tracking_error_native'] = np.max(np.abs(errors), axis=0).tolist()
        report['mujoco_warning_counts'] = data.warning.number.tolist()
        report['total_shard_bytes'] = sum(item['size_bytes'] for item in report['shards'])
        report['frames'] = len(targets)
        passed_status = ('physical_teacher_probe_passed' if args.physics_only
                         else 'verified_single_episode_simulation_teacher_export')
        report['status'] = (passed_status
                            if report['task_metrics']['success'] and len(bilateral_frames) > 0
                            else 'physical_reference_failed_NOT_usable_for_adaptation')
        np.savez_compressed(output / 'trajectory.npz', state=states, action=teacher_public,
                            qpos_before=qpos_before, qpos_after=qpos_after,
                            brick_position_after=positions, tracking_error_native=errors,
                            timestamp_before=times)
        report['trajectory'] = {'file': 'trajectory.npz', 'sha256': digest(output / 'trajectory.npz')}
        if not args.physics_only:
            contact_sheet(output, report['task_metrics']['success'])
            report['contact_sheet'] = {'file': 'contact_sheet.png', 'sha256': digest(output / 'contact_sheet.png')}
        save_json(output / 'manifest.json', report)
        print(json.dumps({'status': report['status'], 'frames': report['frames'],
                          'shards': len(report['shards']), 'total_bytes': report['total_shard_bytes'],
                          'task_metrics': report['task_metrics']}), flush=True)
        if not report['task_metrics']['success']:
            raise RuntimeError('Physical teacher replay did not pass; export is not usable')
    except Exception as error:
        report['status'] = 'export_failed_NOT_usable'
        report['error'] = f'{type(error).__name__}: {error}'
        save_json(output / 'manifest.json', report)
        raise
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == '__main__':
    main()
