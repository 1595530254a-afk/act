"""Render completed feedback trials, including failures and actual policy RGB.

Only restore saved qpos/qvel and call mj_forward; no new ACT or physics steps.
"""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from evaluate_feedback_20261024 import audit_rollout, query_frames
from preview import font, load_model, overview_camera
from render_multisite_evaluation import verify_video
from render_policy_replay import Encoder
from run_position_generalization import model_hashes
from run_task import digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_source(path):
    report = read_json(path)
    plan_path = path.parent / 'protocol.json'
    plan = read_json(plan_path)
    mode = plan['mode']
    expected_queries = len(query_frames(mode))
    if (report['status'] != 'completed_frozen_feedback_ACT_evaluation'
            or report['protocol'] != plan or report['model_files_unchanged'] is not True
            or report['training_steps'] != 0 or report['hardware_commands_sent'] != 0
            or not 1 <= len(report['results']) <= 8
            or report['completed_trials'] != len(plan['positions'])
            or len(report['results']) != len(plan['positions'])
            or report['actual_ACT_queries'] != expected_queries * len(report['results'])
            or [row['id'] for row in report['results']] != [row['id'] for row in plan['positions']]
            or model_hashes(Path(plan['model_directory'])) != plan['model_files_sha256']):
        raise ValueError('Expected a complete frozen feedback evaluation')
    hashes = {str(path): digest(path), str(plan_path): digest(plan_path)}
    for name, expected in plan['source_files_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Frozen evaluation source changed: {name}')
        hashes[name] = expected
    for name, expected in plan['code_files_sha256'].items():
        file = ROOT / name
        if digest(file) != expected:
            raise ValueError(f'Frozen evaluation code changed: {name}')
        hashes[str(file)] = expected
    verified = []
    for trial in report['results']:
        directory = path.parent / 'trials' / trial['id']
        scene = directory / 'scene/task_scene.xml'
        custom = directory / 'custom'
        result = read_json(custom / 'result.json')
        identity = read_json(custom / 'model_identity.json')['model_identity']
        if (result != trial['result'] or result['status'] != 'completed'
                or result['actual_inference_queries'] != expected_queries
                or result['policy_actions_applied'] != 450
                or result['object_truth_passed_to_policy'] is not False
                or result['policy_inputs'] != ['state', 'wrist', 'top']
                or result['grasp_weld_or_object_attach_used'] is not False
                or result['object_pose_assigned_during_rollout'] is not False
                or result['hardware_io'] is not False
                or identity['device'] != 'cuda'
                or identity['files_sha256'] != plan['model_files_sha256']
                or digest(scene) != trial['scene_sha256']
                or audit_rollout(scene, custom, result, mode) != trial['contacts']):
            raise ValueError(f'Incomplete feedback source: {trial["id"]}')
        with np.load(custom / 'rollout_arrays.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if arrays['qpos'].shape != (450, 16) or not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError('Unexpected or nonfinite physical trajectory')
        for filename in ('rollout_arrays.npz', 'result.json', 'model_identity.json'):
            hashes[str(custom / filename)] = digest(custom / filename)
        hashes[str(scene)] = trial['scene_sha256']
        for entry in result['inference']:
            for key in ('input', 'output'):
                hashes[entry[key]] = entry[key + '_sha256']
        model = load_model(scene)
        verified.append((trial, arrays, model, mujoco.MjData(model), custom))
    if report['successes'] != sum(bool(row['result']['metrics']['success']) for row in report['results']):
        raise ValueError('Summary success count differs from individual trials')
    return report, hashes, verified


def label(trial):
    x, y = (value * 1000 for value in trial['offset_xy_m'])
    return f'X {x:+g} / Y {y:+g} mm'


def outcome(trial):
    return ('成功', '#80e7a3') if trial['result']['metrics']['success'] else ('失败', '#ffad91')


def compose_trial(overview, top, wrist, trial, frame, query, query_frame, mode):
    canvas = Image.new('RGB', (1280, 800), '#101820')
    canvas.paste(Image.fromarray(overview).resize((768, 576), Image.Resampling.LANCZOS), (0, 48))
    canvas.paste(Image.fromarray(top).resize((512, 384), Image.Resampling.LANCZOS), (768, 16))
    canvas.paste(Image.fromarray(wrist).resize((512, 384), Image.Resampling.LANCZOS), (768, 416))
    draw = ImageDraw.Draw(canvas)
    status, color = outcome(trial)
    draw.text((16, 8), f'ACT · {label(trial)} · 抓放{status}', font=font(22), fill=color)
    draw.text((775, 0), '顶部：模型实际收到的RGB', font=font(13), fill='#b7dbff')
    draw.text((775, 400), '腕部：模型实际收到的RGB', font=font(13), fill='#b7dbff')
    draw.text((16, 641), f'动作后 t={(frame + 1) / 30:.2f}s · {frame + 1}/450步', font=font(23), fill='white')
    draw.text((16, 682), f'累计真实推理 {query + 1}次 · 最新观测 t={query_frame / 30:.2f}s', font=font(19), fill='#ffd17c')
    draw.text((16, 722), '每步视觉＋时间集成' if mode == 'ensemble1' else '每100步重新看图；右侧保留实际缓存图像', font=font(20), fill='#d4e0ea')
    draw.text((16, 762), '30fps保存轨迹回看 · 不代表墙钟实时运行', font=font(20), fill='#b0bfce')
    return canvas


def compose_montage(images, trials, frame, query_count, mode):
    columns = 4 if len(trials) > 6 else (3 if len(trials) > 4 else min(2, len(trials)))
    width, height = (480, 416) if len(trials) > 6 else (640, 560)
    canvas = Image.new('RGB', (columns * width, math.ceil(len(trials) / columns) * height), '#101820')
    draw = ImageDraw.Draw(canvas)
    image_height = int(width * .75)
    for index, (pixels, trial) in enumerate(zip(images, trials)):
        left, top = index % columns * width, index // columns * height
        status, color = outcome(trial)
        draw.text((left + 10, top + 4), label(trial), font=font(20), fill='white')
        draw.text((left + width - 94, top + 5), f'抓放{status}', font=font(19), fill=color)
        canvas.paste(Image.fromarray(pixels).resize((width, image_height), Image.Resampling.LANCZOS), (left, top + 34))
        draw.text((left + 10, top + height - 35), f't={(frame + 1) / 30:.2f}s · 真实推理 {query_count}次', font=font(15), fill='#d5e3ef')
        draw.text((left + 10, top + height - 18), f'{mode} · 仅仿真时间频率 · 所有试次保留', font=font(13), fill='#ffd17c')
    return canvas


def write_page(directory, evaluation_path, report, videos, poster, provenance_path):
    directory.mkdir(parents=True, exist_ok=False)
    for path in [*videos.values(), poster, provenance_path]:
        try:
            os.link(path, directory / path.name)
        except OSError:
            shutil.copyfile(path, directory / path.name)
    shutil.copyfile(evaluation_path, directory / 'evaluation.json')
    shutil.copyfile(evaluation_path.parent / 'protocol.json', directory / 'protocol.json')
    plan, rows, clips = report['protocol'], [], []
    for trial in report['results']:
        status, _ = outcome(trial)
        metric, contacts = trial['result']['metrics'], trial['contacts']
        rows.append('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in (
            label(trial), status, f'{metric["maximum_lift_m"] * 1000:.2f} mm',
            contacts['bilateral_contact_frames'], trial['result']['actual_inference_queries'])) + '</tr>')
        clips.append(f'<section><h2>{escape(label(trial))} · {status}</h2><video controls preload="metadata" src="{quote(videos[trial["id"]].name)}"></video></section>')
    title = {'test': '最终保留位置', 'validation': '模型选择验证位置', 'baseline': '原位置复核'}[plan['split']]
    frequency = '每动作步重新观察，30 Hz视觉查询＋时间集成' if plan['mode'] == 'ensemble1' else '每100步重新观察，约3.33秒一次'
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ACT 5000步迭代结果</title>
<style>body{{margin:0;background:#101820;color:#e6edf3;font:16px/1.75 "Microsoft YaHei",sans-serif}}main{{max-width:1200px;margin:auto;padding:26px}}video{{width:100%}}a{{color:#82caff}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #384b59}}.note{{padding:15px;background:#23313d;border-left:4px solid #ffd17c}}code{{overflow-wrap:anywhere}}</style><main>
<h1>ACT连续5000步迭代 · {title}</h1>
<p class="note"><strong>本组完成抓放 {report['successes']}/{report['completed_trials']}，全部成功与失败均展示。</strong> 这只是当前本体、物体、相机和位置范围的仿真结果。</p>
<p>所用checkpoint：<code>{escape(plan['model_directory'])}</code><br>模型SHA256：<code>{escape(plan['model_files_sha256']['model.safetensors'])}</code><br>执行方式：{frequency}；动作30 Hz、物理子步1 ms。这里均指仿真时间，未声称端到端墙钟实时运行。</p>
<p>左侧是动作后的保存物理轨迹；单点视频右侧是该动作前模型真正收到的顶部/腕部RGB。chunk100模式中的图像保持，是原策略实际使用缓存的观察，不是视频掉帧。所有视频30fps、15秒，纯回看不再调用ACT或步进物理。</p>
<p>红块平顶40×40×20毫米；灰盒外尺寸120×120×50毫米。腕部橙色件是D405相机支架。测试只改变红块初始XY位置；策略只输入两路RGB及7维机器人状态，不读取物体真值或示范动作。</p>
<p>最终位置仅对本轮本地训练留出，不能保证原始公开真实示范没有覆盖相近位置；均在原±40毫米区域内。验证位置用于挑模型，须与最终位置分开解读。</p>
<video controls preload="metadata" poster="{quote(poster.name)}" src="{quote(videos['montage'].name)}"></video>
<table><tr><th>初始偏移</th><th>抓放</th><th>最大抬升</th><th>双指接触帧</th><th>真实推理次数</th></tr>{''.join(rows)}</table>
<p>完整成功要求：抬升至少30毫米、持续双指夹持、物体全部进入盒内并在底部静止、最终松爪。没有物体绑定或瞬移。</p>
<p><a href="evaluation.json">实际评估</a> · <a href="protocol.json">冻结协议</a> · <a href="{quote(provenance_path.name)}">视频来源与哈希</a></p>
{''.join(clips)}</main></html>'''
    (directory / 'index.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    args = parser.parse_args()
    path, report_dir = args.evaluation.resolve(), args.report_dir.resolve()
    if report_dir.exists():
        raise FileExistsError('Preserve earlier pages; choose a fresh report directory')
    report, hashes, verified = verify_source(path)
    mode = report['protocol']['mode']
    output = ROOT / '视频'
    output.mkdir(parents=True, exist_ok=True)
    stem = 'ACT_5000_' + path.parent.name + '_' + mode + '_' + digest(path)[:8]
    videos = {'montage': output / (stem + '_同步回看.mp4')}
    videos.update({row['id']: output / (stem + '_' + row['id'] + '_实际输入.mp4') for row in report['results']})
    poster, provenance_path = output / (stem + '_frame221.png'), output / (stem + '_来源.json')
    for destination in [*videos.values(), poster, provenance_path]:
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite: {destination}')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required')
    camera = overview_camera()
    camera.lookat[:] = [.23, 0, .14]
    camera.azimuth, camera.elevation, camera.distance = 125, -40, 1.12
    options = mujoco.MjvOption()
    options.geomgroup[3], options.sitegroup[:] = 0, 0
    renderers, encoders, latest = [], {}, {}
    montage_size = None
    try:
        for trial, _, model, _, _ in verified:
            renderers.append(mujoco.Renderer(model, height=480, width=640))
            encoders[trial['id']] = Encoder(ffmpeg, videos[trial['id']], 1280, 800)
        for frame in range(450):
            overviews = []
            for renderer, (trial, arrays, model, data, custom) in zip(renderers, verified):
                data.qpos[:], data.qvel[:] = arrays['qpos'][frame], arrays['qvel'][frame]
                data.time = float(arrays['sim_time'][frame])
                mujoco.mj_forward(model, data)
                options.geomgroup[4] = 1
                renderer.update_scene(data, camera=camera, scene_option=options)
                overview = renderer.render().copy()
                query = int(np.searchsorted(arrays['query_frame_index'], frame, side='right') - 1)
                if trial['id'] not in latest or latest[trial['id']][0] != query:
                    with np.load(custom / f'observation_{query:03d}.npz', allow_pickle=False) as saved:
                        latest[trial['id']] = query, saved['top'].copy(), saved['wrist'].copy()
                _, top, wrist = latest[trial['id']]
                encoders[trial['id']].write(np.asarray(compose_trial(
                    overview, top, wrist, trial, frame, query, int(arrays['query_frame_index'][query]), mode)))
                overviews.append(overview)
            query_count = int(np.searchsorted(verified[0][1]['query_frame_index'], frame, side='right'))
            canvas = compose_montage(overviews, report['results'], frame, query_count, mode)
            if montage_size is None:
                montage_size = canvas.size
                encoders['montage'] = Encoder(ffmpeg, videos['montage'], *montage_size)
            encoders['montage'].write(np.asarray(canvas))
            if frame == 221:
                canvas.save(poster)
            if frame % 90 == 0:
                print(f'Completed ACT feedback replay {frame}/450', flush=True)
        for encoder in encoders.values():
            encoder.finish()
    finally:
        for renderer in renderers:
            renderer.close()
        for encoder in encoders.values():
            encoder.stop()
    if any(digest(Path(name)) != value for name, value in hashes.items()):
        raise ValueError('Evaluation evidence changed while rendering')
    probes = {name: verify_video(file, *(montage_size if name == 'montage' else (1280, 800)))
              for name, file in videos.items()}
    provenance = {'status': 'verified_feedback_trajectory_and_actual_RGB_replay',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_evaluation': str(path),
        'source_files_sha256': hashes, 'mode': mode, 'split': report['protocol']['split'],
        'model_files_sha256': report['protocol']['model_files_sha256'],
        'all_trials_included': True, 'successes': report['successes'],
        'actual_ACT_queries_in_source': report['actual_ACT_queries'],
        'new_ACT_queries': 0, 'new_physics_steps': 0, 'training_steps': 0, 'hardware_commands': 0,
        'right_panels': 'Exact saved pre-action model RGB; held until the next actual query',
        'fps': 30, 'frames': 450, 'duration_s': 15, 'cpu_budget': CPU_BUDGET,
        'outputs': {name: {'path': str(file), 'sha256': digest(file), 'bytes': file.stat().st_size}
                    for name, file in {**videos, 'frame221': poster}.items()}, 'ffprobe': probes}
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_page(report_dir, path, report, videos, poster, provenance_path)
    print(json.dumps({'page': str(report_dir / 'index.html'), 'videos': {k: str(v) for k, v in videos.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
