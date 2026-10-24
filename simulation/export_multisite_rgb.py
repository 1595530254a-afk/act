"""Render matched RGB observations from successful multisite teacher physics.

This records no new policy result: images are rendered from saved pre-action
qpos, paired with the actual saved state and matching teacher action. Test
episodes are not rendered or made available to training.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np
from PIL import Image

from preview import load_model
from export_sim_demonstration import read_public


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def export_episode(source_manifest_path, output):
    source = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    if (source['status'] not in ('physical_teacher_probe_passed', 'physical_recovery_expert_passed')
            or not source['task_metrics']['success']
            or source.get('hardware_io') is not False
            or source.get('grasp_weld_or_attachment') is not False
            or source.get('post_initialization_object_pose_assignment') is not False):
        raise ValueError('Requires successful simulation physics without attachment or teleportation')
    trajectory_path = source_manifest_path.parent / source['trajectory']['file']
    scene_path = Path(source['scene']['path'])
    if (digest(trajectory_path) != source['trajectory']['sha256']
            or digest(scene_path) != source['scene']['sha256']):
        raise ValueError('Teacher physics source changed')
    if output.exists():
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding='utf-8'))
            if (existing.get('status') == 'verified_multisite_episode_rgb_export'
                    and existing.get('source_manifest_sha256') == digest(source_manifest_path)
                    and all(digest(output / item['file']) == item['sha256']
                            for item in existing['images'].values())):
                return manifest_path
        raise FileExistsError(f'Preserve incomplete or different export: {output}')
    with np.load(trajectory_path, allow_pickle=False) as arrays:
        state = arrays['state'].astype(np.float32)
        action = arrays['action'].astype(np.float32)
        qpos = arrays['qpos_before'].copy()
        timestamp = arrays['timestamp_before'].copy()
    count = len(state)
    if (state.shape != (count, 7) or action.shape != (count, 7)
            or count < 120 or not np.isfinite(state).all()
            or not np.isfinite(action).all()
            or not np.allclose(np.diff(timestamp), 1 / 30, atol=.0011)):
        raise ValueError('Invalid pre-action state/action sequence or 30 Hz timing')
    output.mkdir(parents=True, exist_ok=False)
    # Keep compact matched targets locally; RGB is memory mapped to prevent
    # repeatedly decompressing many neighboring images for random minibatches.
    np.savez_compressed(output / 'trajectory.npz', state=state, action=action,
                        timestamp_before=timestamp, qpos_before=qpos)
    report = deepcopy(source)
    report.update({'status': 'rgb_export_in_progress_NOT_usable',
                   'source_manifest': str(source_manifest_path),
                   'source_manifest_sha256': digest(source_manifest_path),
                   'source_trajectory': str(trajectory_path),
                   'source_trajectory_sha256': digest(trajectory_path),
                   'frames': count, 'fps': 30,
                   'trajectory': {'file': 'trajectory.npz',
                                  'sha256': digest(output / 'trajectory.npz')},
                   'images': {}, 'rendered_frames': 0,
                   'observation_timing': 'Saved actual qpos/state BEFORE matching teacher action',
                   'image_storage': 'uint8 RGB NCHW npy memmap',
                   'export_script_sha256': digest(Path(__file__).resolve()),
                   'new_physics_steps': 0, 'new_ACT_queries': 0,
                   'cpu_budget': CPU_BUDGET})
    save_json(output / 'manifest.json', report)
    model = load_model(scene_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    images = {role: np.lib.format.open_memmap(output / f'{role}_rgb.npy',
              mode='w+', dtype=np.uint8, shape=(count, 3, 480, 640))
              for role in ('wrist', 'top')}
    maximum_state_error = 0.
    try:
        for frame in range(count):
            data.qpos[:] = qpos[frame]
            data.qvel[:] = 0
            data.time = float(timestamp[frame])
            mujoco.mj_forward(model, data)
            maximum_state_error = max(maximum_state_error,
                float(np.max(np.abs(read_public(model, data) - state[frame]))))
            if maximum_state_error > 1e-4:
                raise ValueError('Rendered qpos does not match the saved policy observation state')
            for role, camera in (('wrist', 'wrist_rgb'), ('top', 'overhead_rgb')):
                option.geomgroup[4] = int(role != 'wrist')
                renderer.update_scene(data, camera=camera, scene_option=option)
                rgb = renderer.render().copy()
                if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                    raise ValueError('Camera contract changed')
                images[role][frame] = rgb.transpose(2, 0, 1)
                if frame in (0, 160, 195, 240, 300):
                    Image.fromarray(rgb).save(output / f'frame{frame:03d}_{role}.png')
            if frame % 100 == 0:
                print(json.dumps({'episode': output.name, 'rendered': frame + 1,
                                  'total': count}), flush=True)
        for role, array in images.items():
            array.flush()
            path = output / f'{role}_rgb.npy'
            report['images'][role] = {'file': path.name, 'sha256': digest(path),
                'shape': list(array.shape), 'dtype': 'uint8',
                'size_bytes': path.stat().st_size}
        report.update({'status': 'verified_multisite_episode_rgb_export',
                       'rendered_frames': count,
                       'maximum_saved_vs_rendered_state_error_degrees': maximum_state_error,
                       'source_files_unchanged': digest(trajectory_path) == source['trajectory']['sha256']
                           and digest(scene_path) == source['scene']['sha256']})
        if not report['source_files_unchanged']:
            raise ValueError('Teacher source changed while rendering')
        save_json(output / 'manifest.json', report)
        return output / 'manifest.json'
    except Exception as error:
        report.update({'status': 'failed_NOT_usable',
                       'error': f'{type(error).__name__}: {error}'})
        save_json(output / 'manifest.json', report)
        raise
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    source_path = args.teacher_manifest.resolve()
    source = json.loads(source_path.read_text(encoding='utf-8'))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'exporting', 'source_manifest': str(source_path),
              'source_manifest_sha256': digest(source_path), 'episodes': [],
              'test_images_exported': False, 'training_steps': 0,
              'hardware_io': False, 'cpu_budget': CPU_BUDGET}
    started = time.perf_counter()
    save_json(output / 'manifest.json', report)
    for entry in source['episodes']:
        item = deepcopy(entry)
        item['source_episode_manifest_sha256'] = item.pop('manifest_sha256', None)
        if item['split'] == 'test':
            # Metadata is retained for split checks; neither trajectory nor RGB
            # from test episodes is read by this exporter or training.
            item['source_episode_manifest'] = str((source_path.parent / item['manifest']).resolve())
            item['manifest'] = f"not_exported_test/{item['id']}.json"
            item['rgb_exported'] = False
        else:
            if item['split'] not in ('train', 'validation'):
                raise ValueError('Unexpected position split')
            episode_manifest = export_episode(
                (source_path.parent / item['manifest']).resolve(), output / item['id'])
            item['manifest'] = str(episode_manifest.relative_to(output))
            item['manifest_sha256'] = digest(episode_manifest)
            item['rgb_exported'] = True
        report['episodes'].append(item)
        report['elapsed_wall_s'] = time.perf_counter() - started
        save_json(output / 'manifest.json', report)
    if digest(source_path) != report['source_manifest_sha256']:
        raise ValueError('Source episode list changed during export')
    report['status'] = 'verified_multisite_rgb_dataset'
    save_json(output / 'manifest.json', report)
    print(json.dumps({'status': report['status'], 'episodes': len(report['episodes']),
                      'elapsed_wall_s': report['elapsed_wall_s']}), flush=True)


if __name__ == '__main__':
    main()
