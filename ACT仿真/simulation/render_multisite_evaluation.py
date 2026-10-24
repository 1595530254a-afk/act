"""Render a completed multisite ACT evaluation; never execute a policy or physics.

Inputs are evaluate_multisite_act.py's evaluation.json and its saved artifacts.
Every trial, including failures, appears in the synchronized overview and report.
Individual videos also show continuous wrist/top views; those replay views are
not additional observations received by the ACT policy.
"""
import argparse
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from joint_mapping import model_action_to_sim
from preview import font, load_model, overview_camera
from render_policy_replay import Encoder


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_source(evaluation_path):
    """Check complete ACT artifacts before reading them for visualization."""
    source = evaluation_path.parent
    report = read_json(evaluation_path)
    plan_path = source / 'protocol.json'
    plan = read_json(plan_path)
    if (report['status'] != 'completed_frozen_multisite_ACT_evaluation'
            or report['protocol'] != plan or report['model_files_unchanged'] is not True
            or report['training_steps'] != 0 or report['hardware_commands_sent'] != 0
            or plan['teacher_arrays_read'] is not False
            or plan['teacher_actions_read'] is not False
            or plan['training_steps_allowed'] != 0 or plan['hardware_commands_allowed'] != 0
            or (plan['duration_s'], plan['policy_hz'], plan['executed_actions_per_chunk']) != (15, 30, 100)
            or not 1 <= len(report['results']) <= 6
            or report['completed_trials'] != len(plan['positions'])
            or len(report['results']) != len(plan['positions'])
            or report['actual_ACT_queries'] != 5 * len(report['results'])
            or [trial['id'] for trial in report['results']] != [p['id'] for p in plan['positions']]):
        raise ValueError('Expected a complete frozen ACT evaluation with all planned trials')
    model_directory = Path(plan['model_directory'])
    for name, expected in plan['model_files_sha256'].items():
        if digest(model_directory / name) != expected:
            raise ValueError(f'Evaluated ACT model file changed: {name}')
    hashes = {str(evaluation_path): digest(evaluation_path), str(plan_path): digest(plan_path)}
    for path, expected in plan['source_files_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen evaluation source changed: {path}')
        hashes[path] = expected
    verified = []
    for trial, position in zip(report['results'], plan['positions']):
        if (not re.fullmatch(r'[A-Za-z0-9_+\-]+', trial['id'])
                or trial['offset_xy_m'] != position['offset_xy_m']
                or trial['split'] != plan['split'] or trial.get('subset') != plan.get('subset')):
            raise ValueError('Trial identity or initial position differs from the frozen plan')
        directory = source / 'trials' / trial['id']
        paths = {'result': directory / 'custom/result.json',
                 'identity': directory / 'custom/model_identity.json',
                 'scene': directory / 'scene/task_scene.xml',
                 'arrays': directory / 'custom/rollout_arrays.npz'}
        result = read_json(paths['result'])
        identity = read_json(paths['identity'])['model_identity']
        if (result != trial['result'] or result['status'] != 'completed'
                or result['actual_inference_queries'] != 5 or len(result['inference']) != 5
                or result['policy_actions_applied'] != 450 or result['training_steps'] != 0
                or result['hardware_io'] is not False
                or result['object_pose_assigned_during_rollout'] is not False
                or result['grasp_weld_or_object_attach_used'] is not False
                or result['object_truth_passed_to_policy'] is not False
                or result['policy_inputs'] != ['state', 'wrist', 'top']
                or result['model_sha256'] != plan['model_files_sha256']['model.safetensors']
                or identity['files_sha256'] != plan['model_files_sha256']
                or digest(paths['scene']) != trial['scene_sha256']
                or trial['contacts']['all_applied_targets_exact_ACT_conversion'] is not True
                or trial['contacts']['all_input_output_hashes_match_worker_records'] is not True
                or (result['metrics']['success'] and trial['contacts']['physical_grasp_confirmed'] is not True)):
            raise ValueError(f'Incomplete or inconsistent ACT trial: {trial["id"]}')
        with np.load(paths['arrays'], allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        if (arrays['qpos'].shape != (450, 16) or arrays['sim_time'].shape != (450,)
                or arrays['brick_position'].shape != (450, 3)
                or arrays['target_native'].shape != (450, 7)
                or arrays['action_chunks'].shape != (5, 100, 7)
                or not all(np.isfinite(array).all() for array in arrays.values())
                or not np.allclose(np.diff(arrays['sim_time']), 1 / 30, atol=.0007, rtol=0)):
            raise ValueError(f'Unexpected saved trajectory shapes/timing: {trial["id"]}')
        expected_targets = np.stack([model_action_to_sim(arrays['action_chunks'][i // 100, i % 100])
                                     for i in range(450)])
        if not np.array_equal(expected_targets, arrays['target_native']):
            raise ValueError(f'Applied command differs from ACT conversion: {trial["id"]}')
        for index, inference in enumerate(result['inference']):
            for key in ('input', 'output'):
                path = Path(inference[key])
                if digest(path) != inference[key + '_sha256']:
                    raise ValueError(f'ACT inference artifact changed: {path}')
                hashes[str(path)] = inference[key + '_sha256']
            with np.load(inference['output'], allow_pickle=False) as saved:
                if not np.array_equal(saved['action'], arrays['action_chunks'][index]):
                    raise ValueError(f'ACT chunk differs from trace: {trial["id"]}/{index}')
        for path in paths.values():
            hashes[str(path)] = digest(path)
        model = load_model(paths['scene'])
        if model.nq != arrays['qpos'].shape[1]:
            raise ValueError('Saved scene and full qpos dimensions differ')
        verified.append((trial, arrays, model, mujoco.MjData(model)))
    if report['successes'] != sum(bool(trial['result']['metrics']['success']) for trial in report['results']):
        raise ValueError('Summary success count differs from individual results')
    return report, hashes, verified


def trial_label(trial):
    dx, dy = (value * 1000 for value in trial['offset_xy_m'])
    return f'X {dx:+g} / Y {dy:+g} mm'


def outcome(trial):
    success = bool(trial['result']['metrics']['success'])
    return ('抓放成功', '#79e6a2') if success else ('抓放失败', '#ffb193')


def scope_label(trial):
    return '最终全新位置' if trial.get('subset') == 'final_held_out' else (
        '旧诊断位置' if trial.get('subset') == 'previous_diagnostic' else '验证集位置')


def cross_group_summary(evaluation_path, model_sha):
    """Keep old diagnostic results separate from the untouched final positions."""
    directory = evaluation_path.parent.parent
    paths = {'baseline': directory / 'baseline_v1_step600/custom/result.json',
             'validation': directory / 'validation_v1_step600/evaluation.json',
             'final_held_out': directory / 'final_heldout_v1_step600/evaluation.json',
             'previous_diagnostic': directory / 'previous_diagnostic_v1_step600/evaluation.json'}
    summaries = {}
    for group, path in paths.items():
        source = read_json(path)
        if group == 'baseline':
            if source['status'] != 'completed' or source['model_sha256'] != model_sha:
                raise ValueError('Baseline result is not from the selected frozen checkpoint')
            successes, trials = int(bool(source['metrics']['success'])), 1
        else:
            if (source['status'] != 'completed_frozen_multisite_ACT_evaluation'
                    or source['model_files_unchanged'] is not True
                    or source['protocol']['model_files_sha256']['model.safetensors'] != model_sha):
                raise ValueError('Cross-group result is incomplete or from another checkpoint')
            successes, trials = source['successes'], source['completed_trials']
            if (trials != len(source['protocol']['positions'])
                    or trials != len(source['results'])
                    or successes != sum(bool(t['result']['metrics']['success']) for t in source['results'])):
                raise ValueError('Cross-group totals differ from individual results')
        summaries[group] = {'successes': successes, 'trials': trials,
                            'source': str(path), 'sha256': digest(path)}
    return summaries


def compose_trial(overview, top, wrist, trial, index, time):
    canvas = Image.new('RGB', (1280, 800), '#101820')
    canvas.paste(Image.fromarray(overview).resize((768, 576), Image.Resampling.LANCZOS), (0, 48))
    canvas.paste(Image.fromarray(top).resize((512, 384), Image.Resampling.LANCZOS), (768, 16))
    canvas.paste(Image.fromarray(wrist).resize((512, 384), Image.Resampling.LANCZOS), (768, 416))
    draw = ImageDraw.Draw(canvas)
    status, color = outcome(trial)
    draw.text((15, 9), f'ACT {scope_label(trial)} · {trial_label(trial)} · {status}', font=font(20), fill=color)
    draw.text((776, 0), '顶部连续回看 · 30fps', font=font(13), fill='#c9e8ff')
    draw.text((776, 400), '腕部连续回看 · 30fps', font=font(13), fill='#c9e8ff')
    draw.text((18, 642), f't={time:.2f}s · 已执行动作 {index + 1}/450', font=font(22), fill='white')
    draw.text((18, 681), f'原运行推理 {min(index // 100 + 1, 5)}/5；每 100 步（3.33秒）看图一次',
              font=font(19), fill='#ffcc72')
    draw.text((18, 720), '连续回看画面，不代表模型以视频帧率接收图像', font=font(20), fill='#d3dce6')
    draw.text((18, 759), '仅重绘已保存物理轨迹 · 不重新推理／训练／连接真机', font=font(20), fill='#aebfce')
    return canvas


def compose_montage(images, trials, index, times):
    columns = 3 if len(trials) > 4 else 2
    canvas = Image.new('RGB', (columns * 640, math.ceil(len(trials) / columns) * 560), '#101820')
    draw = ImageDraw.Draw(canvas)
    for slot, (image, trial, time) in enumerate(zip(images, trials, times)):
        left, top = slot % columns * 640, slot // columns * 560
        status, color = outcome(trial)
        lift = trial['result']['metrics']['maximum_lift_m'] * 1000
        draw.text((left + 12, top + 5), trial_label(trial), font=font(23), fill='white')
        draw.text((left + 327, top + 9), f'{status} · 抬升 {lift:.1f} mm', font=font(17), fill=color)
        canvas.paste(Image.fromarray(image), (left, top + 40))
        draw.text((left + 12, top + 522),
                  f't={time:.2f}s · 连续轨迹回看 30fps · 原运行推理 {min(index // 100 + 1, 5)}/5',
                  font=font(16), fill='#d3dce6')
        draw.text((left + 12, top + 542), f'{scope_label(trial)} · ACT 每100步看图；测试时不训练',
                  font=font(14), fill='#ffcc72')
    return canvas


def verify_video(path, width, height):
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        raise RuntimeError('ffprobe is required')
    result = json.loads(subprocess.check_output([
        ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,nb_read_frames,duration',
        '-of', 'json', str(path)], text=True))
    stream = result['streams'][0]
    if ((stream['width'], stream['height'], stream['r_frame_rate'], stream['nb_read_frames'])
            != (width, height, '30/1', '450') or abs(float(stream['duration']) - 15) > .001):
        raise ValueError(f'Delivered video metadata mismatch: {path}')
    return result


def reproduction_link(report_dir, groups, model_sha):
    path = Path(groups['final_held_out']['source']).parent / 'policy_output_reproduction.json'
    if not path.exists():
        return ''
    evidence = read_json(path)
    if (evidence['status'] != 'passed_fresh_ACT_output_reproduction'
            or evidence['all_chunks_exactly_equal'] is not True
            or evidence['new_neural_network_forward_calls'] != 30
            or evidence['total_action_values_compared'] != 21000
            or evidence['maximum_absolute_difference_degrees'] != 0
            or evidence['model_identity']['model_sha256'] != model_sha):
        raise ValueError('Reproduction evidence differs from the selected evaluation checkpoint')
    shutil.copyfile(path, report_dir / path.name)
    return ('<p>调用链数值复算：对最终六点保存的30组输入重新运行ACT，21000个动作值与原运行逐值一致，最大差0°。'
            '这证明动作可由冻结模型复现，不代表物理重测或泛化通过。'
            '<a href="policy_output_reproduction.json">完整数值复算记录</a>；'
            f'SHA256：<code>{digest(path)}</code>。</p>')


def write_page(report_dir, evaluation_path, report, videos, poster, provenance_path, groups):
    """Create a fresh report without editing any previous simulation pages."""
    report_dir.mkdir(parents=True, exist_ok=False)
    for path in [*videos.values(), poster, provenance_path]:
        destination = report_dir / path.name
        try:
            os.link(path, destination)
        except OSError:
            shutil.copyfile(path, destination)
    shutil.copyfile(evaluation_path, report_dir / 'evaluation.json')
    shutil.copyfile(evaluation_path.parent / 'protocol.json', report_dir / 'protocol.json')
    for group, summary in groups.items():
        shutil.copyfile(summary['source'], report_dir / (group + '_evidence.json'))
    reproduction = reproduction_link(report_dir, groups,
                                    report['protocol']['model_files_sha256']['model.safetensors'])
    rows, clips = [], []
    for trial in report['results']:
        status, _ = outcome(trial)
        metric = trial['result']['metrics']
        contact = trial['contacts']
        rows.append('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in (
            trial_label(trial), status, f'{metric["maximum_lift_m"] * 1000:.2f} mm',
            contact['bilateral_contact_frames'], contact['longest_elevated_bilateral_frames'],
            trial['result']['actual_inference_queries'])) + '</tr>')
        clips.append(f'<section><h2>{escape(trial_label(trial))} · {status}</h2>'
                     f'<video controls preload="metadata" src="{quote(videos[trial["id"]].name)}"></video></section>')
    plan = report['protocol']
    split_label = '最终留出测试（六个全新位置）' if plan.get('subset') == 'final_held_out' else (
        '旧诊断位置复查（与最终六点分开）' if plan.get('subset') == 'previous_diagnostic' else '验证集')
    group_labels = {'baseline': '原位置复核', 'validation': '模型选择用验证集',
                    'final_held_out': '最终六个全新位置', 'previous_diagnostic': '旧诊断位置'}
    group_rows = ''.join(f'<tr><td>{group_labels[group]}</td><td>{s["successes"]}/{s["trials"]}</td>'
                         f'<td><a href="{group}_evidence.json">原始记录</a></td></tr>'
                         for group, s in groups.items())
    final = groups['final_held_out']
    conclusion = ('最终六个全新位置未完成抓放，当前位置泛化尚未通过。' if final['successes'] == 0
                  else f'最终全新位置完成 {final["successes"]}/{final["trials"]}，仍仅代表本组小范围仿真测试。')
    sibling = ('multisite_previous_diagnostic_20261023' if plan.get('subset') == 'final_held_out'
               else 'multisite_iteration_20261023')
    sibling_label = '另看旧诊断五点（含一次成功，不能代替最终结果）' if plan.get('subset') == 'final_held_out' else '返回最终六个全新位置结果（主结论）'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ACT 多位置冻结评估</title>
<style>body{{margin:0;background:#101820;color:#e6edf3;font:16px/1.7 "Microsoft YaHei",sans-serif}}main{{max-width:1200px;margin:auto;padding:28px}}h1{{font-size:28px}}h2{{font-size:23px}}p{{color:#c1ccd6}}a{{color:#83cbff}}video{{width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #3b4c59;text-align:left}}.note{{padding:15px;background:#202c36;border-left:4px solid #ffcc72}}code{{overflow-wrap:anywhere}}</style><main>
<h1>ACT 多位置训练后：{split_label}</h1>
<p class="note"><strong>{conclusion}</strong> 原位置通过、验证集改善、旧诊断点的一次成功，均不代替最终留出测试。</p>
<p>留出范围：最终六点仅对本轮本地仿真训练保留，不能保证原始50条真实数据未曾覆盖这些位置。所有测试位置都在 X/Y ±40毫米的训练区域内；这检验区域内的位置适应性，不是区域外泛化。</p>
<table><thead><tr><th>评估组（同一冻结权重）</th><th>抓放成功／总数</th><th>证据</th></tr></thead><tbody>{group_rows}</tbody></table>
<p><a href="../{sibling}/index.html">{sibling_label}</a></p>
<p><strong>本组完成抓放 {report['successes']}/{report['completed_trials']}；所有预定试次均展示。</strong></p>
<p>本页展示模型实际输出驱动的物理运行。评估时权重冻结；只改变红块初始位置，不继续训练、不读取示范动作、不向策略提供方块真值坐标。物体尺寸、盒子和控制器保持固定。</p>
<p class="note">视频为保存的完整关节及物体轨迹以 30fps 连续重绘。ACT 实际每 100 步（约 3.33 秒）重新接收图像，15 秒内共推理 5 次；连续视频不代表逐帧模型输入。回看不调用 ACT、不运行新的物理步。</p>
<p>画面腕部的橙色带孔件是 D405 相机安装支架，属于机器人结构件，会随腕部运动。任务目标仍是桌面上的平顶红色40×40×20毫米积木。</p>
<video controls preload="metadata" poster="{quote(poster.name)}" src="{quote(videos['montage'].name)}"></video>
<table><thead><tr><th>初始偏移</th><th>抓放结果</th><th>最大抬升</th><th>双指接触帧</th><th>抬升后连续双指接触</th><th>ACT推理次数</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>成功规则：物体抬升至少30毫米、全部落入盒内并在盒底静止、最终松爪；成功还需至少3个连续30Hz记录帧同时满足双指接触与30毫米抬升。失败试次保留。</p>
<p>这是固定仿真条件下的小范围位置测试，不能代表任意场景或实机性能。模型 SHA256：<code>{escape(plan['model_files_sha256']['model.safetensors'])}</code></p>
<p><a href="evaluation.json">原始结果</a> · <a href="protocol.json">冻结评估协议</a> · <a href="{quote(provenance_path.name)}">视频来源与完整性校验</a></p>
{reproduction}
<h2>各位置连续视角</h2><p>左侧总览；右上顶部，右下腕部。可放大播放器查看。</p>
{''.join(clips)}</main></html>'''
    (report_dir / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '视频')
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    evaluation_path = args.evaluation.resolve()
    output, report_dir = args.output_dir.resolve(), args.report_dir.resolve()
    if report_dir.exists():
        raise FileExistsError(f'Use a fresh report directory: {report_dir}')
    output.mkdir(parents=True, exist_ok=True)
    report, source_hashes, verified = verify_source(evaluation_path)
    groups = cross_group_summary(evaluation_path, report['protocol']['model_files_sha256']['model.safetensors'])
    source_hashes.update({s['source']: s['sha256'] for s in groups.values()})
    stem = 'ACT_多位置_' + evaluation_path.parent.name
    videos = {'montage': output / (stem + '_同步回看.mp4')}
    videos.update({trial['id']: output / (stem + '_' + trial['id'] + '_连续视角.mp4')
                   for trial in report['results']})
    keyframe = output / (stem + '_frame221.png')
    provenance_path = output / (stem + '_来源.json')
    for path in [*videos.values(), keyframe, provenance_path]:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite: {path}')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required')
    camera = overview_camera()
    camera.lookat[:] = [.23, 0, .14]
    camera.azimuth, camera.elevation, camera.distance = 125, -40, 1.12
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    columns = 3 if len(verified) > 4 else 2
    montage_size = (columns * 640, math.ceil(len(verified) / columns) * 560)
    renderers, encoders = [], {}
    try:
        for trial, _, model, _ in verified:
            renderers.append(mujoco.Renderer(model, height=480, width=640))
            encoders[trial['id']] = Encoder(ffmpeg, videos[trial['id']], 1280, 800)
        encoders['montage'] = Encoder(ffmpeg, videos['montage'], *montage_size)
        for index in range(450):
            overviews, times = [], []
            for renderer, (trial, arrays, model, data) in zip(renderers, verified):
                data.qpos[:] = arrays['qpos'][index]
                data.qvel[:] = 0
                data.time = float(arrays['sim_time'][index])
                mujoco.mj_forward(model, data)
                if not np.allclose(data.body('target_brick').xpos, arrays['brick_position'][index],
                                   atol=1e-12, rtol=0):
                    raise ValueError(f'Saved object position mismatch: {trial["id"]}/{index}')
                options.geomgroup[4] = 1
                renderer.update_scene(data, camera=camera, scene_option=options)
                overview = renderer.render().copy()
                renderer.update_scene(data, camera='overhead_rgb', scene_option=options)
                top = renderer.render().copy()
                options.geomgroup[4] = 0
                renderer.update_scene(data, camera='wrist_rgb', scene_option=options)
                wrist = renderer.render().copy()
                encoders[trial['id']].write(np.asarray(compose_trial(
                    overview, top, wrist, trial, index, data.time)))
                overviews.append(overview)
                times.append(data.time)
            canvas = compose_montage(overviews, report['results'], index, times)
            encoders['montage'].write(np.asarray(canvas))
            if index == 221:
                canvas.save(keyframe)
            if index % 90 == 0:
                print(f'Completed ACT trial replay: {index}/450', flush=True)
        for encoder in encoders.values():
            encoder.finish()
    finally:
        for renderer in renderers:
            renderer.close()
        for encoder in encoders.values():
            encoder.stop()
    if any(digest(path) != expected for path, expected in source_hashes.items()):
        raise ValueError('Evaluation evidence changed during rendering')
    probes = {key: verify_video(path, *(montage_size if key == 'montage' else (1280, 800)))
              for key, path in videos.items()}
    provenance = {
        'status': 'verified_continuous_replay_of_completed_frozen_multisite_ACT_evaluation',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_evaluation': str(evaluation_path), 'source_files_sha256': source_hashes,
        'source_model_files_sha256': report['protocol']['model_files_sha256'],
        'source_split': report['protocol']['split'], 'source_subset': report['protocol'].get('subset'),
        'source_ACT_queries_per_trial': 5, 'source_actions_per_trial': 450,
        'source_executed_actions_per_chunk': 100, 'source_image_refresh_period_s': 100 / 30,
        'new_ACT_queries': 0, 'new_physics_steps': 0, 'training_steps': 0, 'hardware_io': False,
        'render_method': 'Restore exact saved full qpos including object; mj_forward only',
        'input_semantics': 'Continuous replay camera views are not per-frame ACT observations',
        'frames': 450, 'fps': 30, 'duration_s': 15, 'montage_dimensions': list(montage_size),
        'single_trial_dimensions': [1280, 800],
        'camera': {'lookat': camera.lookat.tolist(), 'azimuth': 125, 'elevation': -40,
                   'distance': 1.12, 'same_for_all_trials': True},
        'all_planned_trials_included': True, 'successes': report['successes'],
        'cross_group_summary_same_checkpoint': groups,
        'trials': [{'id': trial['id'], 'offset_xy_m': trial['offset_xy_m'],
                    'metrics': trial['result']['metrics'], 'contacts': trial['contacts']}
                   for trial in report['results']],
        'outputs': {key: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
                    for key, path in {**videos, 'frame221': keyframe}.items()},
        'ffprobe': probes, 'cpu_budget': CPU_BUDGET, 'ffmpeg_threads_per_encoder': 2,
    }
    with provenance_path.open('x', encoding='utf-8') as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    write_page(report_dir, evaluation_path, report, videos, keyframe, provenance_path, groups)
    print(json.dumps({'status': provenance['status'], 'videos': {k: str(v) for k, v in videos.items()},
                      'page': str(report_dir / 'index.html'), 'provenance': str(provenance_path)},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
