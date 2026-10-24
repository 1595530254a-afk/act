"""Render the saved physical teacher trajectory, explicitly separate from ACT.

Setting qpos here displays recorded physics states only. This renderer does not
generate controls, run an evaluation, or certify a grasp from an animation.
"""
import argparse
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if (manifest['status'] != 'verified_single_episode_simulation_teacher_export'
            or not manifest['task_metrics']['success']):
        raise ValueError('Only a completed, physically successful teacher can be rendered here')
    trajectory = args.manifest.parent / manifest['trajectory']['file']
    scene = Path(manifest['scene']['path'])
    if digest(trajectory) != manifest['trajectory']['sha256'] or digest(scene) != manifest['scene']['sha256']:
        raise ValueError('Teacher physics trace or scene hash changed')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'teacher.mp4').exists():
        raise FileExistsError('Existing teacher video is preserved; use a fresh output')
    with np.load(trajectory, allow_pickle=False) as arrays:
        qpos = arrays['qpos_after'].copy()
    model = load_model(scene)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    camera = overview_camera()
    camera.lookat[:] = [.29, .04, .09]
    camera.azimuth, camera.elevation, camera.distance = 115, -40, .95
    encoder = subprocess.Popen([shutil.which('ffmpeg'), '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '960x560', '-r', '15', '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'teacher.mp4')],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    snapshots = []
    try:
        for index in range(len(qpos)):
            if index % 2 and index not in (195, 240, 300):
                continue
            data.qpos[:] = qpos[index]
            mujoco.mj_forward(model, data)
            canvas = Image.new('RGB', (960, 560), '#101820')
            options.geomgroup[4] = 1
            renderer.update_scene(data, camera=camera, scene_option=options)
            canvas.paste(Image.fromarray(renderer.render()), (0, 38))
            for role, cam, y in [('top', 'overhead_rgb', 38), ('wrist', 'wrist_rgb', 278)]:
                options.geomgroup[4] = int(role != 'wrist')
                renderer.update_scene(data, camera=cam, scene_option=options)
                canvas.paste(Image.fromarray(renderer.render()).resize((320, 240)), (640, y))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), '仿真示范 · 已保存的接触物理轨迹 · 不是 ACT', fill='white', font=font(19))
            draw.text((12, 526), f't={index / 30:.2f}s · 平顶红块 40×40×20 mm · 只修正松爪目标',
                      fill='#ffcc72', font=font(16))
            if index in (0, 195, 240, 300, 402):
                canvas.save(output / f'teacher_frame{index:03d}.png')
                snapshots.append((index, canvas.copy()))
            if index % 2 == 0:
                encoder.stdin.write(canvas.tobytes())
    finally:
        renderer.close()
        encoder.stdin.close()
        encoder.stdin = None
        _, error = encoder.communicate(timeout=60)
        if encoder.returncode:
            raise RuntimeError(error.decode('utf-8', errors='replace'))
    sheet = Image.new('RGB', (960, 560 * 2), '#101820')
    for n, (_, frame) in enumerate(snapshots[1:]):
        sheet.paste(frame.resize((480, 280)), ((n % 2) * 480, (n // 2) * 280))
    # Four inspection frames, each at half size.
    sheet.crop((0, 0, 960, 560)).save(output / 'teacher_contact_sheet.png')
    report = {'status': 'rendered_recorded_physics_teacher_NOT_ACT',
              'source_manifest': str(args.manifest.resolve()),
              'source_manifest_sha256': digest(args.manifest),
              'video_sha256': digest(output / 'teacher.mp4'),
              'physics_success_from_source': manifest['task_metrics'],
              'actual_ACT_queries': 0, 'training': False, 'hardware_io': False,
              'cpu_budget': CPU_BUDGET}
    (output / 'teacher_render.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
