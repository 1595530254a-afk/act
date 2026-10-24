"""Render continuous views from the verified saved ACT physics trace, without inference.

The recorded model only received five observations. This script visualizes all
450 saved post-step states with mj_forward; it does not run physics or ACT.
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

SOURCE = ROOT / 'runs/mujoco_validation_20261023/fixed_flat40_20261023/policy_step300_progress_fix'
MODEL_HASH = 'e82c8794497ab288c6c58fd16008b78770eddb39bc5f5faab3143b1a530c6c76'
SOURCE_HASHES = {
    'scene/task_scene.xml': '68706e05518c369e512510a0637e025e5ae6de47a580991afd5ef5818ed0b256',
    'scene/scene_metadata.json': '9106c8a33e2f5f3d5df7fe2ba37bbc739d45044432c9935fac98cacda736ae97',
    'custom/result.json': 'd2b96ccf01acdf39e89b3808fe1fb412cbfc9ed0cf5c4187b4d30fd46fcb83c2',
    'custom/model_identity.json': '64ba9e4ea84c66f26ac744d786cf071d56b43f60518da833bba37dc96f5abac5',
    'custom/rollout_arrays.npz': 'ce1a1b6f3ea1c782e8604ac3f0694201e4d6393d344b505af56a5b90a5c30a4f',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_source():
    for relative, expected in SOURCE_HASHES.items():
        if digest(SOURCE / relative) != expected:
            raise ValueError(f'Verified trace source changed: {relative}')
    result = json.loads((SOURCE / 'custom/result.json').read_text(encoding='utf-8'))
    identity = json.loads((SOURCE / 'custom/model_identity.json').read_text(encoding='utf-8'))['model_identity']
    if (result['status'] != 'completed' or result['metrics']['success'] is not True
            or result['model_sha256'] != MODEL_HASH or identity['model_sha256'] != MODEL_HASH
            or result['actual_inference_queries'] != 5 or result['policy_actions_applied'] != 450
            or result['hardware_io'] is not False or result['object_pose_assigned_during_rollout'] is not False
            or result['grasp_weld_or_object_attach_used'] is not False):
        raise ValueError('Source is not the expected successful ACT rollout')
    model_dir = Path(result['model_directory'])
    if model_dir != Path(identity['directory']):
        raise ValueError('Result and worker model directories disagree')
    for filename, expected in identity['files_sha256'].items():
        if digest(model_dir / filename) != expected:
            raise ValueError(f'Selected ACT model file changed: {filename}')
    with np.load(SOURCE / 'custom/rollout_arrays.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if (arrays['qpos'].shape != (450, 16) or arrays['sim_time'].shape != (450,)
            or arrays['action_chunks'].shape != (5, 100, 7)
            or not all(np.isfinite(array).all() for array in arrays.values())
            or not np.allclose(np.diff(arrays['sim_time']), 1 / 30, atol=.0007, rtol=0)):
        raise ValueError('Unexpected saved ACT trace shape or timing')
    if not np.array_equal(arrays['qpos'][0], np.asarray(result['first_recorded_full_qpos'])):
        raise ValueError('First saved pose differs from source result')
    for index, entry in enumerate(result['inference']):
        input_path = SOURCE / f'custom/observation_{index:03d}.npz'
        output_path = SOURCE / f'custom/chunk_{index:03d}.npz'
        if (digest(input_path) != entry['input_sha256']
                or digest(output_path) != entry['output_sha256']):
            raise ValueError(f'Inference artifacts changed: query {index}')
        with np.load(output_path, allow_pickle=False) as saved:
            if not np.array_equal(saved['action'], arrays['action_chunks'][index]):
                raise ValueError(f'Chunk differs from saved trace: query {index}')
    return result, identity, arrays


class Encoder:
    def __init__(self, ffmpeg, path, width, height):
        self.path = path
        self.pending = path.with_name(path.stem + '.pending.mp4')
        if self.path.exists() or self.pending.exists():
            raise FileExistsError(f'Refusing to overwrite: {path}')
        self.process = subprocess.Popen([
            ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '30', '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.pending)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, pixels):
        self.process.stdin.write(pixels.tobytes())

    def finish(self):
        self.process.stdin.close()
        self.process.stdin = None
        _, error = self.process.communicate(timeout=60)
        if self.process.returncode:
            raise RuntimeError(error.decode('utf-8', errors='replace'))
        # Windows rename refuses to replace a file created during encoding.
        self.pending.rename(self.path)

    def stop(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.process.stderr.close()


def compose(overview, top, wrist, index, simulated_time):
    canvas = Image.new('RGB', (1280, 800), '#101820')
    canvas.paste(Image.fromarray(overview).resize((768, 576), Image.Resampling.LANCZOS), (0, 48))
    canvas.paste(Image.fromarray(top).resize((512, 384), Image.Resampling.LANCZOS), (768, 16))
    canvas.paste(Image.fromarray(wrist).resize((512, 384), Image.Resampling.LANCZOS), (768, 416))
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 9), 'ACT 自主抓放 · 同一次成功运行的连续回看', font=font(24), fill='white')
    draw.text((776, 0), '连续仿真视角 · 顶部（30fps）', font=font(13), fill='#c9e8ff')
    draw.text((776, 400), '连续仿真视角 · 腕部（30fps）', font=font(13), fill='#c9e8ff')
    query = min(index // 100 + 1, 5)
    draw.text((18, 642), f't={simulated_time:.2f}s  ·  已执行动作 {index + 1}/450', font=font(22), fill='white')
    draw.text((18, 681), f'原运行实际推理：第 {query}/5 次；每 3.33 秒重新输入图像', font=font(20), fill='#ffcc72')
    draw.text((18, 720), '右侧为连续回看画面，不代表模型逐帧接收了这些图像', font=font(20), fill='#d3dce6')
    draw.text((18, 759), '仅重绘已保存物理轨迹 · 不重新推理／训练／连接真机', font=font(20), fill='#aebfce')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '视频')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {
        'composite': output / 'ACT_自主抓放_连续视角_20261023.mp4',
        'top': output / 'ACT_顶部视角_20261023.mp4',
        'wrist': output / 'ACT_腕部视角_20261023.mp4',
        'frame221': output / 'ACT_自主抓放_连续视角_20261023_frame221.png',
        'provenance': output / 'ACT_自主抓放_连续视角_20261023_来源.json',
    }
    for path in files.values():
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite: {path}')
    result, identity, arrays = verify_source()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required')
    model = load_model(SOURCE / 'scene/task_scene.xml')
    if model.nq != arrays['qpos'].shape[1]:
        raise ValueError('Scene state size differs from saved trace')
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    camera = overview_camera()
    camera.lookat[:] = [.23, 0, .14]
    camera.azimuth, camera.elevation, camera.distance = 125, -40, 1.12
    renderer = mujoco.Renderer(model, height=480, width=640)
    encoders = []
    try:
        composite = Encoder(ffmpeg, files['composite'], 1280, 800)
        encoders.append(composite)
        top_encoder = Encoder(ffmpeg, files['top'], 640, 480)
        encoders.append(top_encoder)
        wrist_encoder = Encoder(ffmpeg, files['wrist'], 640, 480)
        encoders.append(wrist_encoder)
        for index, pose in enumerate(arrays['qpos']):
            data.qpos[:] = pose
            data.qvel[:] = 0
            data.time = float(arrays['sim_time'][index])
            # Visualization only: restore exact saved robot AND object pose.
            # No mj_step, controller, ACT invocation or new physics success claim.
            mujoco.mj_forward(model, data)
            if not np.allclose(data.body('target_brick').xpos, arrays['brick_position'][index], atol=1e-12, rtol=0):
                raise ValueError(f'Scene and saved object position disagree at frame {index}')
            options.geomgroup[4] = 1
            renderer.update_scene(data, camera=camera, scene_option=options)
            overview = renderer.render().copy()
            renderer.update_scene(data, camera='overhead_rgb', scene_option=options)
            top = renderer.render().copy()
            options.geomgroup[4] = 0
            renderer.update_scene(data, camera='wrist_rgb', scene_option=options)
            wrist = renderer.render().copy()
            canvas = compose(overview, top, wrist, index, float(arrays['sim_time'][index]))
            composite.write(np.asarray(canvas))
            top_encoder.write(top)
            wrist_encoder.write(wrist)
            if index == 221:
                canvas.save(files['frame221'])
            if index % 90 == 0:
                print(f'Continuous saved-trace views: {index}/450', flush=True)
        for encoder in encoders:
            encoder.finish()
    finally:
        renderer.close()
        for encoder in encoders:
            encoder.stop()
    provenance = {
        'status': 'continuous_visualization_of_verified_saved_ACT_rollout',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_directory': str(SOURCE), 'source_files_sha256': SOURCE_HASHES,
        'source_model_sha256': MODEL_HASH, 'source_model_files_sha256': identity['files_sha256'],
        'source_task_success': result['metrics']['success'],
        'source_actual_ACT_inference_queries': 5, 'source_observation_period_s': 100 / 30,
        'source_physics_steps': result['physics_steps'],
        'new_ACT_inference_queries': 0, 'new_physics_steps': 0, 'training_steps': 0,
        'hardware_io': False, 'real_robot_qualified': False,
        'render_method': 'Exact saved full qpos at 450 successive states, mj_forward for visualization only; no mj_step',
        'fps': 30, 'frames': 450, 'video_duration_s': 15,
        'composite_dimensions': [1280, 800], 'single_view_dimensions': [640, 480],
        'input_semantics': 'Continuous top/wrist views are replay visualizations, not additional ACT inputs. The source policy received only five recorded observation pairs.',
        'source_result': str(SOURCE / 'custom/result.json'),
        'outputs': {key: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
                    for key, path in files.items() if key != 'provenance'},
        'cpu_budget': CPU_BUDGET, 'ffmpeg_threads_per_encoder': 2,
    }
    with files['provenance'].open('x', encoding='utf-8') as destination:
        json.dump(provenance, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write('\n')
    print(json.dumps({'status': provenance['status'], 'outputs': {key: str(path) for key, path in files.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
