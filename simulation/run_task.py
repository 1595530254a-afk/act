"""Evaluate local ACT checkpoints in one reconstructed physical scene.

Simulation owns MuJoCo; a separate, persistent ACT Python process owns Torch.
No robot, CAN, camera SDK, training, or network access is used.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from joint_mapping import dataset_state_to_sim, model_action_to_sim, mapping_metadata
from preview import font, load_model, overview_camera
from task_scene import build_scene
from task_dynamics import initialize, ServoController, read_public_state

BASE = ROOT / 'runs/mujoco_validation_20261023'
DEFAULT_OUTPUT = BASE / 'task_comparison'
CHECKPOINTS = {
    'published': ROOT / 'models/act_rebot_cube',
    'finetuned': ROOT / 'delivery/ACT_laptop_20261019_v2/models/act_controlled_step1000',
}
LABELS = {'published': '原始公开权重', 'finetuned': '本地续训 1000 步权重'}


def save_json(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_physics(data, previous_time):
    if data.time <= previous_time:
        raise FloatingPointError('MuJoCo time reset or failed to advance')
    for warning in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL):
        if data.warning[int(warning)].number:
            raise FloatingPointError(f'MuJoCo numerical warning: {warning.name}')


class PolicyProcess:
    def __init__(self, model_dir, output):
        self.output = output
        self.log = (output / 'policy_stderr.log').open('w', encoding='utf-8')
        self.process = subprocess.Popen([
            str(ROOT / '.venv/Scripts/python.exe'), '-u',
            str(ROOT / 'simulation/policy_worker.py'), '--model-dir', str(model_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding='utf-8', bufsize=1)
        self.messages = queue.Queue()
        def receive():
            for line in self.process.stdout:
                self.messages.put(line)
            self.messages.put(None)
        self.reader = threading.Thread(target=receive, daemon=True)
        self.reader.start()
        try:
            self.identity = self.read(timeout=120)
            if self.identity.get('status') != 'ready':
                raise RuntimeError(f'ACT worker did not initialize: {self.identity}')
        except Exception:
            self.close()
            raise

    def read(self, timeout=60):
        try:
            line = self.messages.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError('ACT worker timed out; inspect policy_stderr.log') from error
        if line is None:
            raise RuntimeError(f'ACT worker exited ({self.process.poll()}); inspect policy_stderr.log')
        return json.loads(line)

    def predict(self, state, wrist, top, index):
        input_path = self.output / f'observation_{index:03d}.npz'
        output_path = self.output / f'chunk_{index:03d}.npz'
        np.savez_compressed(input_path, state=state, wrist=wrist, top=top)
        self.process.stdin.write(json.dumps({'input': str(input_path), 'output': str(output_path)}) + '\n')
        self.process.stdin.flush()
        response = self.read()
        if response.get('status') != 'ok':
            raise RuntimeError(f'ACT prediction failed: {response}')
        if response['input_sha256'] != digest(input_path) or response['output_sha256'] != digest(output_path):
            raise ValueError('ACT worker artifact hash mismatch')
        with np.load(output_path, allow_pickle=False) as result:
            action = result['action'].copy()
        if action.shape != (100, 7) or not np.isfinite(action).all():
            raise ValueError('Invalid ACT output chunk')
        return action, response

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop":true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=15)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=10)
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def brick_metrics(model, data, initial_position, metadata, lift_max):
    """Conservative final footprint check; no object position is modified."""
    position = data.body('target_brick').xpos.copy()
    rotation = data.body('target_brick').xmat.reshape(3, 3)
    geom = model.geom('target_brick_geom')
    half = geom.size[:3]
    corners = np.array([[x, y, z] for x in (-half[0], half[0])
                        for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    corners = corners @ rotation.T + position
    bin_pos = data.body('target_bin').xpos.copy()
    bin_rotation = data.body('target_bin').xmat.reshape(3, 3)
    local = (corners - bin_pos) @ bin_rotation
    floor = model.geom('target_bin_floor')
    # Geom sizes and body-relative positions are read from the actual scene.
    floor_top = floor.pos[2] + floor.size[2]
    east = model.geom('target_bin_wall_e')
    west = model.geom('target_bin_wall_w')
    north = model.geom('target_bin_wall_n')
    south = model.geom('target_bin_wall_s')
    bounds = [west.pos[0] + west.size[0], east.pos[0] - east.size[0],
              south.pos[1] + south.size[1], north.pos[1] - north.size[1]]
    inside = bool((local[:, 0] > bounds[0]).all() and (local[:, 0] < bounds[1]).all()
                  and (local[:, 1] > bounds[2]).all() and (local[:, 1] < bounds[3]).all())
    resting_height = bool(local[:, 2].min() >= floor_top - .003
                          and local[:, 2].min() <= floor_top + .012)
    joint = model.joint('target_brick_freejoint')
    speed = float(np.linalg.norm(data.qvel[joint.dofadr[0]:joint.dofadr[0] + 3]))
    contacts_with_fingers = 0
    for contact in data.contact[:data.ncon]:
        bodies = [int(model.geom_bodyid[int(g)]) for g in (contact.geom1, contact.geom2)]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or '' for body in bodies]
        if 'target_brick' in names and any('gripper' in name for name in names):
            contacts_with_fingers += 1
    lifted = lift_max >= .03
    success = bool(lifted and inside and resting_height and speed < .05 and contacts_with_fingers == 0)
    return {'success': success, 'lifted_at_least_3cm': bool(lifted),
            'maximum_lift_m': float(lift_max), 'entire_brick_footprint_inside_bin': inside,
            'resting_on_bin_floor': resting_height, 'final_speed_m_s': speed,
            'final_finger_contacts': contacts_with_fingers,
            'initial_position_m': initial_position.tolist(), 'final_position_m': position.tolist(),
            'bin_interior_bounds_local_xy_m': bounds,
            'success_rule': 'lift >= 0.03m, all brick box corners inside bin XY, rests at bin floor, speed <0.05m/s, no final finger contact',
            'scope': 'One reconstructed scene trial, not a benchmark success rate or hardware result.'}


def run_checkpoint(key, scene, scene_metadata, output, initial, duration, action_steps,
                   model_dir=None, label=None):
    model_dir = Path(model_dir if model_dir is not None else CHECKPOINTS[key]).resolve()
    label = label if label is not None else LABELS.get(key, key)
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(scene)
    data = mujoco.MjData(model)
    initialize(model, data, initial)
    controller = ServoController(model)
    # Let the free object settle while the same controller holds the robot.
    for _ in range(500):
        controller.step(data, initial)
    actual = read_public_state(model, data)
    if np.max(np.abs(dataset_state_to_sim(actual)[:6] - initial[:6])) > .05:
        raise RuntimeError('Robot did not hold its initial pose within 0.05 rad in the reconstructed scene')
    initial_position = data.body('target_brick').xpos.copy()
    initial_public = read_public_state(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    camera = overview_camera()
    camera.lookat[:] = [.23, 0, .14]
    camera.azimuth, camera.elevation, camera.distance = 125, -40, 1.12
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required')
    temporary_video = output / 'rollout.pending.mp4'
    encoder = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '960x560', '-r', '15', '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary_video)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    worker = None
    saved = {'public_state': [], 'target_native': [], 'brick_position': [],
             'qpos': [], 'sim_time': [], 'action_chunks': [], 'chunk_observation_state': []}
    inference = []
    penetration = Counter()
    lift_max = 0.
    start_wall, sim_start = time.perf_counter(), data.time
    status, error_text = 'completed', None
    metrics = None
    try:
        worker = PolicyProcess(model_dir, output)
        save_json(output / 'model_identity.json', worker.identity)
        total_steps = round(duration * 30)
        for index in range(total_steps):
            if index % action_steps == 0:
                observations = {}
                for role, camera_name in [('wrist', 'wrist_rgb'), ('top', 'overhead_rgb')]:
                    options.geomgroup[4] = int(role != 'wrist')
                    renderer.update_scene(data, camera=camera_name, scene_option=options)
                    observations[role] = renderer.render().copy()
                    if index == 0:
                        Image.fromarray(observations[role]).save(output / f'initial_{role}.png')
                public = read_public_state(model, data)
                chunk, timing = worker.predict(public, observations['wrist'], observations['top'], len(inference))
                saved['action_chunks'].append(chunk)
                saved['chunk_observation_state'].append(public)
                inference.append(timing)
                print(f'{key}: t={index / 30:.2f}/{duration:.2f}s, actual ACT query {len(inference)}', flush=True)
            target = model_action_to_sim(chunk[index % action_steps])
            end_time = sim_start + (index + 1) / 30
            while data.time < end_time - 1e-10:
                previous_time = data.time
                controller.step(data, target)
                check_physics(data, previous_time)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError('MuJoCo state became non-finite')
            # mj_step integrates qpos after its last geometry update. Measure
            # and render the integrated state, rather than a one-step-old pose.
            mujoco.mj_forward(model, data)
            position = data.body('target_brick').xpos.copy()
            lift_max = max(lift_max, float(position[2] - initial_position[2]))
            saved['public_state'].append(read_public_state(model, data))
            saved['target_native'].append(target)
            saved['brick_position'].append(position)
            saved['qpos'].append(data.qpos.copy())
            saved['sim_time'].append(data.time - sim_start)
            for contact in data.contact[:data.ncon]:
                if contact.dist < -.001:
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g)
                             for g in (contact.geom1, contact.geom2)]
                    penetration[' / '.join(names)] += 1
            if index % 2 == 0:
                options.geomgroup[4] = 1
                renderer.update_scene(data, camera=camera, scene_option=options)
                canvas = Image.new('RGB', (960, 560), '#101820')
                canvas.paste(Image.fromarray(renderer.render()), (0, 38))
                for role, offset, label in [('top', 38, '连续顶部视角'), ('wrist', 278, '连续腕部视角')]:
                    options.geomgroup[4] = int(role != 'wrist')
                    renderer.update_scene(data, camera='overhead_rgb' if role == 'top' else 'wrist_rgb',
                                          scene_option=options)
                    current_view = renderer.render().copy()
                    canvas.paste(Image.fromarray(current_view).resize((320, 240)), (640, offset))
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((645, offset + 5, 811, offset + 29), fill='#101820')
                    draw.text((651, offset + 7), label, fill='white', font=font(15))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 8), label + ' · 物理仿真闭环', fill='white', font=font(19))
                draw.text((12, 526), f't={index / 30:.2f}s · 推理 {len(inference)} 次 · ACT采样每 {action_steps / 30:.2f}s · 右侧连续视角',
                          fill='#ffcc72', font=font(15))
                if index == 0:
                    canvas.save(output / 'rollout.png')
                encoder.stdin.write(canvas.tobytes())
        metrics = brick_metrics(model, data, initial_position, scene_metadata, lift_max)
    except Exception as error:
        status, error_text = 'failed', f'{type(error).__name__}: {error}'
        metrics = brick_metrics(model, data, initial_position, scene_metadata, lift_max)
    finally:
        if worker:
            try:
                worker.close()
            except Exception as error:
                status = 'failed'
                error_text = f'{error_text or ""}; ACT cleanup: {type(error).__name__}: {error}'
        renderer.close()
        encoder.stdin.close()
        encoder.stdin = None
        _, encoder_error = encoder.communicate(timeout=60)
        if encoder.returncode:
            status = 'failed'
            error_text = f'ffmpeg failed: {encoder_error.decode("utf-8", errors="replace")}'
        elif temporary_video.stat().st_size > 1000:
            temporary_video.replace(output / 'rollout.mp4')
        arrays = {name: np.asarray(values) for name, values in saved.items()}
        np.savez_compressed(output / 'rollout_arrays.npz', **arrays)
        record = {'status': status, 'error': error_text, 'checkpoint': key, 'label': label,
            'model_directory': str(model_dir),
            'model_sha256': worker.identity.get('model_identity', {}).get('model_sha256') if worker else None,
            'elapsed_wall_s': time.perf_counter() - start_wall,
            'simulated_duration_s': float(data.time - sim_start), 'policy_actions_applied': len(saved['target_native']),
            'actual_inference_queries': len(inference), 'inference': inference,
            'physics_steps': int(round(data.time / model.opt.timestep)),
            'initial_public_state': initial_public.tolist(),
            'first_recorded_full_qpos': arrays['qpos'][0].tolist() if len(arrays['qpos']) else None,
            'mapping': mapping_metadata(), 'simulation_controller': controller.metadata(),
            'mujoco_warning_counts': [int(warning.number) for warning in data.warning],
            'metrics': metrics, 'penetrations_over_1mm_frame_pair_counts': penetration.most_common(12),
            'object_pose_assigned_during_rollout': False, 'grasp_weld_or_object_attach_used': False,
            'hardware_io': False, 'training_steps': 0, 'closed_loop': len(inference) > 1,
            'closed_loop_attempted': True, 'cpu_budget': CPU_BUDGET,
            'evaluation': {'policy_hz': 30, 'chunk_size': 100, 'executed_actions_per_chunk': action_steps,
                'replanning_period_s': action_steps / 30, 'physics_dt_s': float(model.opt.timestep),
                'setting_note': ('Saved model default: execute all 100 actions before fresh camera feedback.' if action_steps == 100
                                 else 'Receding horizon: execute first 10 predicted actions before fresh camera feedback.')}}
        save_json(output / 'result.json', record)
    return record


def write_page(output, report):
    items = []
    results = report.get('results', {})
    successful = sum(bool(value.get('metrics', {}).get('success')) for value in results.values())
    if report['status'] == 'running':
        conclusion = f'正在运行，已完成 {len(results)} 个策略试次。'
    elif results:
        conclusion = f'已记录 {len(results)} 个策略试次，其中 {successful} 个完成物理抓取入盒。'
    else:
        conclusion = '本轮尚无完成的策略试次。'
    if report.get('error'):
        conclusion += ' ' + report['error']
    for key, result in results.items():
        metrics = result['metrics']
        label = escape(result.get('label', LABELS.get(key, key)))
        model_directory = escape(result.get('model_directory', '未记录'))
        model_hash = escape(result.get('model_sha256') or '请查看权重身份文件')
        items.append(f'''<section><h2>{label}</h2>
<p>运行状态：{result['status']} · 抓起 ≥3cm：{'是' if metrics['lifted_at_least_3cm'] else '否'} · 最终入盒：{'是' if metrics['success'] else '否'} · 实际重新推理 {result['actual_inference_queries']} 次</p>
<video controls preload="metadata" poster="{key}/rollout.png" src="{key}/rollout.mp4"></video>
<p>本次加载：<code>{model_directory}</code><br>模型 SHA256：<code>{model_hash}</code></p>
<p><a href="{key}/result.json">完整数值结果</a> · <a href="{key}/model_identity.json">权重与处理器身份</a></p></section>''')
    first_key = next(iter(results), None)
    initial_views = ''.join(f'<figure><img src="{first_key}/initial_{role}.png"><figcaption>实际送入 ACT 的{label}画面</figcaption></figure>'
                            for role, label in [('top', '顶部'), ('wrist', '腕部')]) if first_key else '<p>等待首个策略试次。</p>'
    scene = report.get('scene', {})
    brick, container = scene.get('brick', {}), scene.get('bin', {})
    dimensions = brick.get('body_dimensions_m')
    bin_dimensions = container.get('outside_dimensions_m')
    scene_description = '本轮使用记录文件中的固定场景。'
    if dimensions and bin_dimensions:
        brick_size = ' × '.join(f'{value * 1000:g}' for value in dimensions)
        bin_size = ' × '.join(f'{value * 1000:g}' for value in bin_dimensions)
        top_description = '平顶、无凸点' if brick.get('flat_top') else '有凸点'
        scene_description = f'红色方块 {brick_size} mm（{top_description}）；灰色开口盒外尺寸 {bin_size} mm。'
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>红色方块 → 灰盒 · ACT 物理仿真</title><style>
body{margin:0;background:#101820;color:#e6edf3;font:16px/1.65 'Microsoft YaHei',sans-serif}main{max-width:1100px;margin:auto;padding:28px}h1{font-size:29px}h2{font-size:23px}p{color:#c1ccd6}.note{border-left:4px solid #ffcc72;padding:12px 18px;background:#202c36}video{width:100%;border-radius:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}img{width:100%}a{color:#83cbff}figcaption{color:#aabac8}code{overflow-wrap:anywhere}
</style><main><h1>红色块体 → 灰色收纳盒</h1>
<p>运行状态：STATUS</p>
<p><strong>CONCLUSION</strong></p>
<p class="note">本轮使用虚拟腕部/顶部 RGB 和仿真关节反馈，ACT 生成动作，经仿真 PD 力矩控制推动动力学。每次执行 ACTION_STEPS 步，再读取新画面。视频右侧显示连续相机视角，模型实际输入按上述周期采样；不能把每一帧连续画面都当成模型已读取。物体通过真实接触运动，没有绑定夹爪、传送物体或播放预设抓取动画。</p>
<p>SCENE_DESCRIPTION 尺寸为本项目设定，相机位姿和材质仍是重建近似。一次试次不能衡量整体成功率，也不是实机验收。</p>
RESULTS
REFERENCE
<h2>实际送入 ACT 的初始虚拟画面</h2><div class="grid">INITIAL</div>
<p><a href="comparison.json">本轮完整记录</a> · <a href="scene/scene_metadata.json">场景假设与尺寸</a></p>
<p>本次评估过程没有更新权重、没有真实机械臂指令。模型来源以本次加载路径和 SHA256 为准。CPU 进程限制为 4 个逻辑核，ACT 推理 2 线程。</p></main></html>'''
    page = page.replace('STATUS', report['status']).replace('RESULTS', '\n'.join(items)).replace('INITIAL', initial_views)
    page = page.replace('ACTION_STEPS', str(report.get('executed_actions_per_chunk', 10)))
    page = page.replace('CONCLUSION', escape(conclusion)).replace('SCENE_DESCRIPTION', escape(scene_description))
    reference = '<p><a href="reference_tracking.json">本轮示范轨迹诊断记录</a>（结果以记录文件为准）</p>'
    page = page.replace('REFERENCE', reference if report.get('reference_check') else '')
    (output / 'index.html').write_text(page, encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--duration', type=float, default=15.)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--checkpoint', choices=['both', 'published', 'finetuned'], default='both')
    selection.add_argument('--model-dir', type=Path, help='Evaluate only this local model directory, including its saved processors')
    parser.add_argument('--action-steps', choices=[10, 100], type=int, default=100)
    parser.add_argument('--square-brick', action='store_true', help='Use the source-video 4x4 stud layout; dimensions remain assumed')
    parser.add_argument('--flat-brick', action='store_true', help='Use the fixed 40 x 40 x 20 mm flat square prop without studs')
    parser.add_argument('--brick-edge-mm', type=int, choices=[32, 40, 48], default=40)
    parser.add_argument('--wrist-camera-fit', type=Path, help='Inferred camera candidate JSON, never hardware calibration')
    args = parser.parse_args(argv)
    if args.flat_brick and args.brick_edge_mm != 40:
        parser.error('--flat-brick requires the fixed 40 mm project prop')
    if args.flat_brick:
        args.square_brick = True
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be finite and greater than zero')
    if args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()
        for name in ('config.json', 'model.safetensors'):
            if not (args.model_dir / name).is_file():
                parser.error(f'--model-dir is missing {name}: {args.model_dir}')
    return args


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    wrist_camera = (json.loads(args.wrist_camera_fit.read_text(encoding='utf-8'))
                    if args.wrist_camera_fit else None)
    scene, scene_metadata = build_scene(output / 'scene', square_brick=args.square_brick,
        square_edge_m=args.brick_edge_mm / 1000, wrist_camera=wrist_camera, flat_brick=args.flat_brick)
    save_json(output / 'scene/scene_metadata.json', scene_metadata)
    with np.load(BASE / 'inputs.npz', allow_pickle=False) as source:
        initial = dataset_state_to_sim(source['frame0_state'])
    checkpoints = ({'custom': args.model_dir} if args.model_dir is not None else
                   {key: CHECKPOINTS[key] for key in (CHECKPOINTS if args.checkpoint == 'both' else [args.checkpoint])})
    report = {'status': 'running', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scene': scene_metadata, 'scene_sha256': digest(scene), 'results': {},
        'executed_actions_per_chunk': args.action_steps,
        'simulation_only': True, 'hardware_commands_sent': 0, 'training_steps': 0,
        'model_directories': {key: str(path.resolve()) for key, path in checkpoints.items()},
        'same_initial_state_and_scene_for_both': len(checkpoints) > 1, 'single_trial_per_checkpoint': True}
    save_json(output / 'comparison.json', report)
    write_page(output, report)
    try:
        for key, model_dir in checkpoints.items():
            label = '仿真适配 ACT 权重' if key == 'custom' else LABELS[key]
            report['results'][key] = run_checkpoint(key, scene, scene_metadata, output / key,
                initial, args.duration, args.action_steps, model_dir=model_dir, label=label)
            video = output / key / 'rollout.mp4'
            if video.is_file():
                # Keep raw run evidence for the webpage, and deliver videos to
                # the user-selected folder without replacing earlier exports.
                video_directory = ROOT / '视频'
                video_directory.mkdir(parents=True, exist_ok=True)
                outcome = '成功' if report['results'][key]['metrics']['success'] else '未通过'
                video_hash = digest(video)
                delivered = video_directory / f'ACT_{output.name}_{key}_{outcome}_{video_hash[:8]}.mp4'
                if not delivered.exists():
                    shutil.copy2(video, delivered)
                if digest(delivered) != video_hash:
                    raise ValueError('Delivered video hash differs from the recorded rollout')
                report['results'][key]['delivered_video'] = str(delivered)
                save_json(output / key / 'result.json', report['results'][key])
            save_json(output / 'comparison.json', report)
            write_page(output, report)
        report['status'] = ('completed' if all(value['status'] == 'completed'
                             for value in report['results'].values()) else 'completed_with_aborted_trial')
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save_json(output / 'comparison.json', report)
        write_page(output, report)
    print(json.dumps({'status': report['status'], 'output': str(output),
        'results': {key: value['metrics'] for key, value in report['results'].items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
