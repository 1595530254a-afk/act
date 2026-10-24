"""Fit an attached virtual camera to pregrasp red centroids from real video.

This is an inferred reconstruction, not measured hand-eye calibration. The
world brick point and robot coordinates are fixed candidate assumptions.
Run in the ACT Python environment; MuJoCo FK uses its separate environment.
No simulator asset, policy weight or hardware configuration is changed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import numpy as np

DEFAULT_OUTPUT = ROOT / 'runs/mujoco_iteration_20261023/camera_fit'
INPUTS = ROOT / 'runs/mujoco_validation_20261023/inputs.npz'
VIDEO = ROOT / 'data/rebot_cube/videos/observation.images.wrist/chunk-000/file-000.mp4'
SCENE = ROOT / 'runs/mujoco_validation_20261023/task_camera_check/scene/task_scene.xml'
FRAME_INDICES = tuple(range(60, 186, 5))
FULLY_VISIBLE_FIRST_FRAME = 110
BRICK_WORLD = np.array([.340, -.025, .010])
FOVY_DEG = 62.82
CV_FROM_MUJOCO = np.diag([1., -1., -1.])


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def forward_kinematics(output):
    import mujoco
    from joint_mapping import dataset_state_to_sim
    from preview import load_model, set_pose
    model = load_model(SCENE)
    data = mujoco.MjData(model)
    with np.load(INPUTS, allow_pickle=False) as source:
        targets = dataset_state_to_sim(source['episode0_state'])
    points, camera_pos, camera_mat = [], [], []
    for index in FRAME_INDICES:
        set_pose(model, data, targets[index])
        rotation = data.body('gripper_end').xmat.reshape(3, 3)
        origin = data.body('gripper_end').xpos
        points.append((BRICK_WORLD - origin) @ rotation)
        camera_pos.append((data.cam_xpos[model.camera('wrist_rgb').id] - origin) @ rotation)
        camera_mat.append(rotation.T @ data.cam_xmat[model.camera('wrist_rgb').id].reshape(3, 3))
    np.savez_compressed(output / 'fk_inputs.npz', frame_indices=FRAME_INDICES,
                        object_points_gripper=points, source_camera_pos=camera_pos[0],
                        source_camera_mat=camera_mat[0], brick_world=BRICK_WORLD)


def extract_video_frames():
    import av
    wanted, found = set(FRAME_INDICES), {}
    with av.open(str(VIDEO)) as video:
        video.streams.video[0].codec_context.thread_count = 2
        for index, frame in enumerate(video.decode(video=0)):
            if index in wanted:
                found[index] = frame.to_ndarray(format='rgb24')
            if index >= max(wanted):
                break
    if set(found) != wanted:
        raise ValueError('Video did not contain every requested pregrasp frame')
    return found


def red_centroid(rgb):
    import cv2
    red, green, blue = (rgb[..., i].astype(np.float32) for i in range(3))
    mask = ((red > 90) & (red > green * 1.6) & (red > blue * 1.6)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[selected, cv2.CC_STAT_AREA])
    if not 80 <= area <= 20000:
        return None
    return centroids[selected], area, stats[selected, :4]


def rotation_angle_degrees(left, right):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(left.T @ right) - 1) / 2, -1, 1))))


def fit_camera(output):
    import cv2
    from PIL import Image, ImageDraw
    subprocess.run([str(ROOT / 'local/mujoco-env/Scripts/python.exe'), str(Path(__file__).resolve()),
                    '--fk', '--output-dir', str(output)], check=True, cwd=ROOT)
    with np.load(output / 'fk_inputs.npz', allow_pickle=False) as source:
        indices = source['frame_indices']
        points = source['object_points_gripper']
        source_position = source['source_camera_pos']
        source_rotation = source['source_camera_mat']
    frames = extract_video_frames()
    observations, excluded = [], []
    for row, index in enumerate(indices):
        detection = red_centroid(frames[int(index)])
        if detection is not None:
            centroid, area, rectangle = detection
            if index < FULLY_VISIBLE_FIRST_FRAME:
                excluded.append({'frame': int(index), 'centroid_px': centroid.tolist(),
                                 'reason': 'Brick visibly occluded by the fingers/camera lower border in inspected real frames; partial-red centroid is not the whole brick centre.'})
                continue
            observations.append({'frame': int(index), 'row': row, 'centroid_px': centroid.tolist(),
                                 'area_px': area, 'bbox_xywh_px': rectangle.tolist()})
    if len(observations) < 12:
        raise ValueError(f'Only {len(observations)} usable red detections; no fit attempted')
    object_points = np.asarray([points[item['row']] for item in observations], np.float64)
    image_points = np.asarray([item['centroid_px'] for item in observations], np.float64)
    focal = 480 / (2 * np.tan(np.deg2rad(FOVY_DEG) / 2))
    intrinsics = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1.]])
    source_rvec = cv2.Rodrigues(CV_FROM_MUJOCO @ source_rotation.T)[0]
    source_tvec = -(CV_FROM_MUJOCO @ source_rotation.T) @ source_position
    # Every fourth detection is held out; the remaining moving wrist poses fit
    # one constant attached camera. This does not tune any policy outcome.
    validation = np.arange(len(observations)) % 4 == 0
    train = ~validation
    ok, rvec, tvec = cv2.solvePnP(object_points[train], image_points[train], intrinsics,
                                None, source_rvec.copy(), source_tvec.copy(), True,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError('OpenCV attached camera fit failed')
    rvec, tvec = cv2.solvePnPRefineLM(object_points[train], image_points[train], intrinsics,
                                   None, rvec, tvec)
    fitted_cv_rotation = cv2.Rodrigues(rvec)[0]
    fitted_position = -(fitted_cv_rotation.T @ tvec.reshape(3))
    fitted_rotation = fitted_cv_rotation.T @ CV_FROM_MUJOCO
    before = cv2.projectPoints(object_points, source_rvec, source_tvec, intrinsics, None)[0].reshape(-1, 2)
    after = cv2.projectPoints(object_points, rvec, tvec, intrinsics, None)[0].reshape(-1, 2)
    errors_before, errors_after = np.linalg.norm(before - image_points, axis=1), np.linalg.norm(after - image_points, axis=1)
    depths = (object_points @ fitted_cv_rotation.T + tvec.reshape(3))[:, 2]
    offset_change = float(np.linalg.norm(fitted_position - source_position))
    angle_change = rotation_angle_degrees(source_rotation, fitted_rotation)
    plausible = bool(offset_change < .12 and angle_change < 60 and (depths > .02).all())
    image_rows = []
    for i, item in enumerate(observations):
        item.update({'held_out': bool(validation[i]), 'before_px': before[i].tolist(),
                     'after_px': after[i].tolist(), 'before_error_px': float(errors_before[i]),
                     'after_error_px': float(errors_after[i])})
        if i % 3 == 0:
            canvas = Image.fromarray(frames[item['frame']])
            draw = ImageDraw.Draw(canvas)
            for position, colour in [(image_points[i], '#29ff44'), (before[i], '#ffff00'), (after[i], '#00ffff')]:
                x, y = position
                draw.line((x - 7, y, x + 7, y), fill=colour, width=3)
                draw.line((x, y - 7, x, y + 7), fill=colour, width=3)
            draw.rectangle((0, 0, 640, 29), fill='#101820')
            draw.text((8, 8), f"Frame {item['frame']}: green=real, yellow=old, cyan=fitted", fill='white')
            image_rows.append(canvas)
    sheet = Image.new('RGB', (1280, 480 * ((len(image_rows) + 1) // 2)), '#101820')
    for i, frame in enumerate(image_rows):
        sheet.paste(frame, ((i % 2) * 640, (i // 2) * 480))
    sheet.save(output / 'centroid_projection_contact_sheet.png')
    metrics = lambda values, mask: {'rmse_px': float(np.sqrt(np.mean(values[mask] ** 2))),
                                    'median_px': float(np.median(values[mask])), 'max_px': float(np.max(values[mask]))}
    report = {
        'status': 'inferred_attached_camera_candidate' if plausible else 'fit_outside_source_mount_prior_not_accepted',
        'method': 'OpenCV iterative PnP; fixed candidate brick point transformed into moving gripper frame; one attached six-DOF camera fit',
        'hardware_calibration': False, 'hardware_io': False, 'source_assets_modified': False,
        'fovy_fixed_deg': FOVY_DEG, 'resolution': [640, 480], 'intrinsics_assumed': intrinsics.tolist(),
        'fixed_brick_point_world_m': BRICK_WORLD.tolist(),
        'source_camera_local_position_m': source_position.tolist(),
        'fitted_camera_local_position_m': fitted_position.tolist(),
        'fitted_camera_local_rotation_columns': fitted_rotation.tolist(),
        'xml_xyaxes': np.r_[fitted_rotation[:, 0], fitted_rotation[:, 1]].tolist(),
        'translation_change_m': offset_change, 'rotation_change_deg': angle_change,
        'within_loose_source_mount_prior': plausible, 'positive_depth_all_frames': bool((depths > .02).all()),
        'before_training': metrics(errors_before, train), 'after_training': metrics(errors_after, train),
        'before_held_out': metrics(errors_before, validation), 'after_held_out': metrics(errors_after, validation),
        'point_cloud_singular_values_m': np.linalg.svd(object_points - object_points.mean(axis=0), compute_uv=False).tolist(),
        'detections': observations, 'excluded_partial_detections': excluded,
        'cpu_budget': CPU_BUDGET,
        'limitations': [
            'Brick location, candidate robot zero/mapping, pixel centroid-to-3D-centre correspondence and camera focal length are assumptions.',
            'This fit can absorb errors in assumed brick world coordinates and robot kinematics; it does not establish real hand-eye extrinsics.',
            'Only red centroids are fitted, not robot landmarks, full-image texture or grasp success. Held-out samples come from the same episode.',
            'A small residual is diagnostic evidence for a more similar virtual view, not independent camera calibration or policy success.'
        ]}
    save_json(output / 'camera_fit.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'translation_change_m', 'rotation_change_deg',
                      'before_held_out', 'after_held_out', 'fitted_camera_local_position_m', 'xml_xyaxes')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--fk', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.fk:
        forward_kinematics(output)
    else:
        fit_camera(output)
