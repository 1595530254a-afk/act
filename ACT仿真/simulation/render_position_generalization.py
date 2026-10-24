"""Replay all six frozen ACT position trials as one synchronized comparison video.

Rendering restores saved robot/object qpos and calls mj_forward only. It never
loads ACT, runs mj_step, changes trial outcomes, or modifies evaluation sources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from preview import font, load_model, overview_camera
from render_policy_replay import Encoder

SOURCE = ROOT / 'runs/mujoco_validation_20261023/position_generalization_20261023'
LABELS = {
    'baseline': '原位置复核',
    'x_plus30': 'X +30 mm',
    'x_minus30': 'X -30 mm',
    'y_plus30': 'Y +30 mm',
    'y_minus30': 'Y -30 mm',
    'xy_plus30_minus40': 'X +30 / Y -40 mm',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_source(source):
    report_path, protocol_path = source / 'generalization_report.json', source / 'protocol.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if (report['status'] != 'completed_frozen_position_evaluation'
            or report['frozen_model_unchanged'] is not True
            or report['protocol_sha256'] != digest(protocol_path)
            or report['protocol'] != protocol
            or report['training_steps'] != 0 or report['hardware_commands_sent'] != 0
            or report['teacher_action_sequence_read'] is not False
            or report['object_truth_passed_to_policy'] is not False
            or len(report['results']) != 6
            or [trial['id'] for trial in protocol['trials']] != list(LABELS)
            or [trial['id'] for trial in report['results']] != list(LABELS)
            or protocol['executed_actions_per_chunk'] != 100):
        raise ValueError('Source is not the complete expected frozen six-position evaluation')
    model_directory = Path(protocol['model_directory'])
    model_hashes = protocol['model_files_sha256']
    if model_hashes != report['model_files_sha256_after']:
        raise ValueError('Frozen model metadata differs before and after evaluation')
    for name, expected in model_hashes.items():
        if digest(model_directory / name) != expected:
            raise ValueError(f'Frozen model file changed: {name}')
    source_hashes = {
        str(report_path): digest(report_path), str(protocol_path): digest(protocol_path),
    }
    verified = []
    for trial in report['results']:
        trial_directory = source / 'trials' / trial['id']
        result_path = trial_directory / 'custom/result.json'
        scene_path = trial_directory / 'scene/task_scene.xml'
        arrays_path = trial_directory / 'custom/rollout_arrays.npz'
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if (result != trial['result'] or result['status'] != 'completed'
                or result['actual_inference_queries'] != 5 or len(result['inference']) != 5
                or result['policy_actions_applied'] != 450
                or result['model_sha256'] != model_hashes['model.safetensors']
                or result['training_steps'] != 0 or result['hardware_io'] is not False
                or result['object_pose_assigned_during_rollout'] is not False
                or result['grasp_weld_or_object_attach_used'] is not False
                or digest(scene_path) != trial['scene_sha256']):
            raise ValueError(f'Trial source verification failed: {trial["id"]}')
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if (arrays['qpos'].shape != (450, 16)
                or arrays['sim_time'].shape != (450,)
                or arrays['action_chunks'].shape != (5, 100, 7)
                or not all(np.isfinite(array).all() for array in arrays.values())
                or not np.allclose(np.diff(arrays['sim_time']), 1 / 30, atol=.0007, rtol=0)
                or not np.array_equal(arrays['qpos'][0], np.asarray(result['first_recorded_full_qpos']))):
            raise ValueError(f'Unexpected actual saved trajectory: {trial["id"]}')
        for index, entry in enumerate(result['inference']):
            for key in ('input', 'output'):
                artifact = Path(entry[key])
                if digest(artifact) != entry[key + '_sha256']:
                    raise ValueError(f'Changed inference artifact: {artifact}')
                source_hashes[str(artifact)] = entry[key + '_sha256']
            with np.load(entry['output'], allow_pickle=False) as archive:
                if not np.array_equal(archive['action'], arrays['action_chunks'][index]):
                    raise ValueError(f'ACT output differs from saved chunk: {trial["id"]}/{index}')
        for path in (result_path, scene_path, arrays_path):
            source_hashes[str(path)] = digest(path)
        model = load_model(scene_path)
        if model.nq != arrays['qpos'].shape[1]:
            raise ValueError('Scene qpos size differs from saved trajectory')
        verified.append((trial, arrays, model, mujoco.MjData(model)))
    return report, source_hashes, verified


def compose(images, trials, index, times):
    canvas = Image.new('RGB', (1920, 1120), '#101820')
    draw = ImageDraw.Draw(canvas)
    for slot, (image, trial, time) in enumerate(zip(images, trials, times)):
        left, top = slot % 3 * 640, slot // 3 * 560
        result = trial['result']
        success = result['metrics']['success']
        color = '#79e6a2' if success else '#ffb193'
        draw.text((left + 12, top + 4), LABELS[trial['id']], font=font(25), fill='white')
        status = '抓放成功' if success else '抓放失败'
        lift_mm = result['metrics']['maximum_lift_m'] * 1000
        draw.text((left + 310, top + 8), f'{status} · 抬升 {lift_mm:.1f} mm', font=font(19), fill=color)
        canvas.paste(Image.fromarray(image), (left, top + 40))
        query = min(index // 100 + 1, 5)
        draw.text((left + 12, top + 522),
                  f't={time:.2f}s · 连续轨迹回看 30fps · 原运行推理 {query}/5',
                  font=font(16), fill='#d3dce6')
        draw.text((left + 12, top + 542), 'ACT 每 100 步看图一次；测试期间不训练',
                  font=font(14), fill='#ffcc72')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '视频')
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = 'ACT_位置泛化_六组同步回看_20261023'
    video = output / (stem + '.mp4')
    keyframe = output / (stem + '_frame221.png')
    provenance_path = output / (stem + '_来源.json')
    for path in (video, keyframe, provenance_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite: {path}')
    report, source_hashes, verified = verify_source(source)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required')
    camera = overview_camera()
    camera.lookat[:] = [.23, 0, .14]
    camera.azimuth, camera.elevation, camera.distance = 125, -40, 1.12
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    renderers = []
    encoder = None
    try:
        renderers = [mujoco.Renderer(model, height=480, width=640)
                     for _, _, model, _ in verified]
        encoder = Encoder(ffmpeg, video, 1920, 1120)
        for index in range(450):
            images, times = [], []
            for renderer, (trial, arrays, model, data) in zip(renderers, verified):
                data.qpos[:] = arrays['qpos'][index]
                data.qvel[:] = 0
                data.time = float(arrays['sim_time'][index])
                # Restore exact saved robot AND object pose, then visualize only.
                mujoco.mj_forward(model, data)
                if not np.allclose(data.body('target_brick').xpos, arrays['brick_position'][index],
                                   atol=1e-12, rtol=0):
                    raise ValueError(f'Saved object position mismatch: {trial["id"]}/{index}')
                renderer.update_scene(data, camera=camera, scene_option=options)
                images.append(renderer.render().copy())
                times.append(float(arrays['sim_time'][index]))
            canvas = compose(images, report['results'], index, times)
            encoder.write(np.asarray(canvas))
            if index == 221:
                canvas.save(keyframe)
            if index % 90 == 0:
                print(f'Six-trial synchronized saved trajectory replay: {index}/450', flush=True)
        encoder.finish()
    finally:
        for renderer in renderers:
            renderer.close()
        if encoder is not None:
            encoder.stop()
    # Evaluation sources must remain byte-for-byte intact during visualization.
    if any(digest(path) != expected for path, expected in source_hashes.items()):
        raise ValueError('Source evidence changed while rendering')
    probe = shutil.which('ffprobe')
    if not probe:
        raise RuntimeError('ffprobe is required to verify the delivered video')
    probe_result = json.loads(subprocess.check_output([
        probe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,nb_read_frames,duration',
        '-of', 'json', str(video)], text=True))
    stream = probe_result['streams'][0]
    if (stream['width'], stream['height'], stream['r_frame_rate'], stream['nb_read_frames']) != (
            1920, 1120, '30/1', '450') or abs(float(stream['duration']) - 15) > .001:
        raise ValueError(f'Delivered video does not match requested timeline: {stream}')
    provenance = {
        'status': 'verified_synchronized_replay_of_six_completed_frozen_ACT_trials',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_directory': str(source), 'source_files_sha256': source_hashes,
        'source_model_files_sha256': report['model_files_sha256_after'],
        'frozen_model_unchanged': report['frozen_model_unchanged'],
        'source_ACT_inference_queries_per_trial': 5, 'source_actions_per_trial': 450,
        'source_executed_actions_per_chunk': 100, 'policy_image_refresh_period_s': 100 / 30,
        'new_ACT_inference_queries': 0, 'new_physics_steps': 0, 'training_steps': 0,
        'hardware_io': False, 'fps': 30, 'frames': 450, 'duration_s': 15,
        'dimensions': [1920, 1120], 'layout': '3 columns by 2 rows, each 640x560',
        'camera': {'lookat': camera.lookat.tolist(), 'azimuth': 125, 'elevation': -40,
                   'distance': 1.12, 'shared_across_trials': True},
        'render_method': 'Restore saved full qpos, mj_forward only; no mj_step or policy invocation',
        'input_semantics': 'Continuous trajectory replay, not additional per-frame ACT observations',
        'trials': [{'id': trial['id'], 'label': LABELS[trial['id']],
                    'offset_xy_m': trial['offset_xy_m'],
                    'success': trial['result']['metrics']['success'],
                    'maximum_lift_m': trial['result']['metrics']['maximum_lift_m']}
                   for trial in report['results']],
        'outputs': {kind: {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
                    for kind, path in [('video', video), ('keyframe', keyframe)]},
        'ffprobe': probe_result, 'cpu_budget': CPU_BUDGET, 'ffmpeg_threads': 2,
    }
    with provenance_path.open('x', encoding='utf-8') as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': provenance['status'], 'video': str(video),
                      'keyframe': str(keyframe), 'provenance': str(provenance_path)},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
