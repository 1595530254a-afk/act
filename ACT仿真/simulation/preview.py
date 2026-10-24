"""Kinematic MuJoCo replay of saved ACT observations and predictions; no robot I/O."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from joint_mapping import dataset_state_to_sim, model_action_to_sim, mapping_metadata

DEFAULT_MODEL = ROOT / 'local/mujoco_upstream/rebotarm_mujoco_rs/models/rs_grasp_scene.xml'
DEFAULT_RUN = ROOT / 'runs/mujoco_validation_20261023'
JOINTS = [f'joint{i}' for i in range(1, 8)]


def load_model(path):
    # Native MuJoCo's Windows filename API rejects the Chinese workspace path.
    # Resolve the directory in Python and pass an ASCII relative filename.
    path = path.resolve()
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        return mujoco.MjModel.from_xml_path(path.name)
    finally:
        os.chdir(previous)


def load_inputs(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    states = arrays['episode0_state']
    predicted = arrays['predicted_action_chunk']
    if states.ndim != 2 or states.shape[1] != 7 or len(states) < 2:
        raise ValueError('Expected an episode with shape [N,7]')
    if predicted.shape != (100, 7):
        raise ValueError('Expected one ACT action chunk with shape [100,7]')
    for name in ('episode0_state', 'predicted_action_chunk', 'episode0_timestamp'):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f'Non-finite input: {name}')
    timestamps = arrays['episode0_timestamp']
    if timestamps.shape != (len(states),):
        raise ValueError('Episode timestamps must have shape [N] matching states')
    if not np.allclose(np.diff(timestamps), 1 / 30, atol=2e-6, rtol=1e-4):
        raise ValueError('Expected strictly increasing 30 Hz episode timestamps')
    return arrays


def set_pose(model, data, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('Expected seven finite simulator coordinates')
    for name, value in zip(JOINTS, values):
        data.qpos[model.joint(name).qposadr[0]] = value
    # Upstream rs_arm.xml explicitly couples these three slides 1:1.
    for name in ('joint_left', 'joint_right'):
        data.qpos[model.joint(name).qposadr[0]] = values[6]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def trajectory_diagnostics(model, data, poses):
    violations = {}
    for index, name in enumerate(JOINTS):
        joint = model.joint(name)
        lower, upper = joint.range
        outside = (poses[:, index] < lower - 1e-6) | (poses[:, index] > upper + 1e-6)
        violations[name] = {
            'unit': 'rad' if index < 6 else 'm',
            'model_range': [float(lower), float(upper)],
            'trajectory_range': [float(poses[:, index].min()), float(poses[:, index].max())],
            'outside_frames': int(outside.sum()),
        }
    penetration_pairs = Counter()
    penetrated_frames = 0
    for pose in poses:
        set_pose(model, data, pose)
        found = False
        for contact in data.contact[:data.ncon]:
            if contact.dist < -0.001:
                first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
                second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
                penetration_pairs[f'{first or contact.geom1} / {second or contact.geom2}'] += 1
                found = True
        penetrated_frames += int(found)
    return {
        'frames': len(poses), 'joint_limits': violations,
        'frames_with_model_penetration_over_1mm': penetrated_frames,
        'most_common_model_penetrations': penetration_pairs.most_common(10),
        'scope': 'Kinematic geometry in the supplied scene; not physical tracking or real-robot safety certification.',
        'clamping_applied': False,
    }


def overview_camera():
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.20, 0.0, 0.23]
    camera.distance = 1.25
    camera.azimuth = 130
    camera.elevation = -24
    return camera


def font(size):
    path = Path('C:/Windows/Fonts/msyh.ttc')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def render_clip(model, data, poses, timestamps, output, title):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required to save the preview video')
    fps = 15
    indices = np.arange(0, len(poses), 2)
    temporary = output.with_name(output.stem + '.pending.mp4')
    temporary_poster = output.with_name(output.stem + '.pending.png')
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '960x560', '-r', str(fps), '-i', 'pipe:0',
               '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2',
               '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = overview_camera()
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    normal_font, small_font = font(19), font(15)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for position, index in enumerate(indices):
            set_pose(model, data, poses[index])
            canvas = Image.new('RGB', (960, 560), '#101820')
            renderer.update_scene(data, camera=camera, scene_option=options)
            canvas.paste(Image.fromarray(renderer.render()), (0, 38))
            for name, top, label in [('overhead_rgb', 38, '虚拟顶部相机'), ('wrist_rgb', 278, '虚拟腕部相机')]:
                renderer.update_scene(data, camera=name, scene_option=options)
                view = Image.fromarray(renderer.render()).resize((320, 240))
                canvas.paste(view, (640, top))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((646, top + 5, 802, top + 29), fill='#101820')
                draw.text((652, top + 7), label, font=small_font, fill='white')
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), title, font=normal_font, fill='white')
            draw.text((12, 526), f't={timestamps[index]:.2f}s · 帧 {index} · 候选坐标转换 / 运动学预览 / 不连接真机',
                      font=small_font, fill='#ffcc72')
            if position == 0:
                canvas.save(temporary_poster)
            process.stdin.write(canvas.tobytes())
            if position % 45 == 0:
                print(f'{output.name}: {position}/{len(indices)} frames', flush=True)
        process.stdin.close()
        process.stdin = None
        _, error = process.communicate(timeout=60)
        if process.returncode:
            raise RuntimeError(f'ffmpeg failed ({process.returncode}): {error.decode("utf-8", errors="replace")}')
        temporary.replace(output)
        temporary_poster.replace(output.with_suffix('.png'))
    finally:
        renderer.close()
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        process.stderr.close()


def write_html(output, report):
    limits = report['dataset_replay']['joint_limits']
    action_limits = report['act_prediction']['joint_limits']
    rows = ''.join(f'<tr><td>{name}</td><td>{item["outside_frames"]}</td><td>{action_limits[name]["outside_frames"]}</td></tr>'
                   for name, item in limits.items())
    source_images = ''.join(f'<figure><img src="frame0_{name}.png"><figcaption>模型本次输入：真实{name}画面，固定为第0帧</figcaption></figure>'
                            for name in ('wrist', 'top'))
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B601-RS · ACT 仿真预览</title><style>
body{margin:0;background:#101820;color:#e6edf3;font:16px/1.65 'Microsoft YaHei',sans-serif}main{max-width:1100px;margin:auto;padding:28px}
h1{font-size:28px;margin:0}h2{font-size:21px}p{color:#c1ccd6}.note{border-left:4px solid #ffcc72;padding:12px 18px;background:#202c36}
video{width:100%;background:#05090d;border-radius:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}img{width:100%;border-radius:8px}
figcaption{font-size:14px;color:#b4c2ce}table{border-collapse:collapse}td,th{border-bottom:1px solid #43515c;padding:6px 24px;text-align:left}a{color:#83cbff}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style><main><h1>B601-RS · ACT 仿真预览</h1>
<p>官方 Seeed 机械臂模型 · 原始公开 ACT 权重 · 单场景回放</p>
<p class="note">本页是运动学回放与预测动作预览。采用有源码依据的 LeRobot canonical RS 公共坐标转换，但作者实际采集分支和本机零位仍未验证。没有闭环抓取、没有动力学跟踪，也没有连接真实机械臂。红蓝黄物体来自官方仿真示例，并非训练数据里的灰盒场景；视频中的物体不会随手臂被抓起。</p>
<h2>1．公开数据集的示范状态回放</h2><p>完整第 0 段示范，共 FRAMES 帧、DURATION 秒。可暂停、拖动时间轴和全屏。不会为“好看”静默裁剪越界角度。</p>
<video controls preload="metadata" poster="dataset_replay.png" src="dataset_replay.mp4"></video>
<h2>2．已有 ACT 权重输出的动作预览</h2><p>输入第 0 帧真实腕部图像、顶部图像和七轴状态，单次推理生成 100×7 动作，再转换到虚拟关节显示。这不是由虚拟相机连续反馈驱动的策略闭环。</p>
<video controls preload="metadata" poster="act_prediction.png" src="act_prediction.mp4"></video>
<h2>本次推理使用的真实画面</h2><div class="grid">SOURCE_IMAGES</div>
<h2>候选转换后的模型范围诊断</h2><p>以下只反映该仿真模型与候选转换是否相容，不是现场机械限位认证。</p>
<table><thead><tr><th>关节</th><th>示范越界帧数 / 403</th><th>ACT 预测越界帧数 / 100</th></tr></thead><tbody>LIMIT_ROWS</tbody></table>
<h2>场景接触诊断</h2><p>按当前示例场景检测到超过 1 mm 几何穿透：示范 CONTACT_DATASET 帧，ACT 预测 CONTACT_ACT 帧。桌面高度、基座和物体布置尚未按原始采集场景对齐，因此不能把这些计数当成原作者轨迹或真机策略的碰撞结论；当前也不能宣布场景碰撞检查通过。</p>
<p><a href="verification.json">完整验证记录</a> · <a href="inputs_manifest.json">模型与输入来源</a></p>
<p>下一步需要把采集坐标和本机零位对应清楚，再评估短段真机动作。仿真通过不能替代真实装配、承重与停机验证。</p></main></html>'''
    html = html.replace('FRAMES', str(report['dataset_replay']['frames']))
    html = html.replace('DURATION', f'{report["episode_duration_s"]:.2f}')
    html = html.replace('SOURCE_IMAGES', source_images).replace('LIMIT_ROWS', rows)
    html = html.replace('CONTACT_DATASET', str(report['dataset_replay']['frames_with_model_penetration_over_1mm']))
    html = html.replace('CONTACT_ACT', str(report['act_prediction']['frames_with_model_penetration_over_1mm']))
    (output / 'index.html').write_text(html, encoding='utf-8')


def interactive(model, data, trajectories):
    import mujoco.viewer

    state = {'mode': 'dataset', 'paused': True, 'index': 0}
    def on_key(key):
        if key == 32:
            state['paused'] = not state['paused']
        elif key in (49, 50):
            state['mode'] = 'dataset' if key == 49 else 'act'
            state['index'] = 0
        elif key in (82, 114):
            state['index'] = 0
    print('Simulation only. Space=pause/play; 1=dataset; 2=ACT; R=restart; close window to stop.', flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = overview_camera().lookat
            viewer.cam.distance = 1.25
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -24
        while viewer.is_running():
            started = time.perf_counter()
            poses = trajectories[state['mode']]
            index = state['index'] % len(poses)
            with viewer.lock():
                set_pose(model, data, poses[index])
            viewer.sync()
            if not state['paused']:
                state['index'] += 1
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args()
    arrays = load_inputs(args.run_dir / 'inputs.npz')
    model = load_model(args.model)
    data = mujoco.MjData(model)
    dataset_poses = dataset_state_to_sim(arrays['episode0_state'])
    action_poses = model_action_to_sim(arrays['predicted_action_chunk'])
    if args.viewer:
        interactive(model, data, {'dataset': dataset_poses, 'act': action_poses})
        return
    report = {
        'status': 'rendering', 'scope': 'candidate_mapping_replay_only',
        'model_path': str(args.model), 'mujoco_version': importlib.metadata.version('mujoco'),
        'mapping': mapping_metadata(),
        'cpu_budget': CPU_BUDGET, 'robot_commands_sent': 0, 'hardware_modules_imported': False,
        'physics_steps_executed': 0, 'act_closed_loop_tested': False,
        'real_robot_mapping_verified': False, 'grasp_success_verified': False,
        'training_performed': False, 'scene_matches_training_dataset': False,
        'collision_validation_passed': False,
        'model_nq': model.nq, 'model_nv': model.nv, 'model_nu': model.nu,
        'episode_duration_s': float(arrays['episode0_timestamp'][-1]),
        'dataset_replay': trajectory_diagnostics(model, data, dataset_poses),
        'act_prediction': trajectory_diagnostics(model, data, action_poses),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.run_dir / 'verification.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.run_dir / 'index.html').write_text('<meta charset="utf-8"><p>正在生成本轮仿真预览，请稍后刷新。本轮尚未完成。</p>', encoding='utf-8')
    try:
        render_clip(model, data, dataset_poses, arrays['episode0_timestamp'],
                    args.run_dir / 'dataset_replay.mp4', '公开示范状态回放 · episode 0')
        render_clip(model, data, action_poses, np.arange(len(action_poses)) / 30,
                    args.run_dir / 'act_prediction.mp4', '原始公开 ACT 权重 · 单次预测动作预览')
    except Exception as error:
        report['status'] = 'rendering_failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.run_dir / 'index.html').write_text('<meta charset="utf-8"><p>本轮生成失败，不能沿用旧预览作为本轮结果。请查看 verification.json。</p>', encoding='utf-8')
        raise
    report['status'] = 'kinematic_preview_generated'
    (args.run_dir / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_html(args.run_dir, report)
    print(json.dumps({'report': str(args.run_dir / 'index.html'), 'status': report['status']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
