"""Frozen ACT position evaluation. Read only a preregistered protocol, no teacher.

Only the brick's initial XML body XY changes, before model compilation. ACT runs
in a separate process with exactly state/wrist/top arrays. No training or robot
I/O occurs. Object truth is used for simulation initialization and scoring only.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import numpy as np

from joint_mapping import dataset_state_to_sim
from run_task import digest, run_checkpoint, save_json
from task_scene import build_scene


def model_hashes(directory):
    names = ('config.json', 'model.safetensors', 'policy_preprocessor.json',
             'policy_postprocessor.json',
             'policy_preprocessor_step_3_normalizer_processor.safetensors',
             'policy_postprocessor_step_0_unnormalizer_processor.safetensors')
    return {name: digest(directory / name) for name in names}


def normalized_scene_tree(root):
    normalized = deepcopy(root)
    normalized.find('compiler').set('meshdir', 'FROZEN_UPSTREAM_MESHES')
    normalized.find(".//body[@name='target_brick']").set('pos', 'REFERENCE_BRICK_POSITION')
    return ET.tostring(normalized)


def write_page(output, report):
    results = report['results']
    heldout = [r for r in results if r['id'] != 'baseline']
    count = sum(r['result']['metrics']['success'] for r in heldout)
    baseline = next((r for r in results if r['id'] == 'baseline'), None)
    baseline_text = ('原位置复核通过' if baseline and baseline['result']['metrics']['success']
                     else '原位置未通过' if baseline else '等待原位置复核')
    rows, clips = [], []
    for trial in report['protocol']['trials']:
        entry = next((r for r in results if r['id'] == trial['id']), None)
        shift = ' / '.join(f'{v * 1000:+g}' for v in trial['offset_xy_m'])
        if entry:
            result = entry['result']
            outcome = '抓放成功' if result['metrics']['success'] else '未完成抓放'
            movement = f"{result['metrics']['maximum_lift_m'] * 1000:.2f} mm"
            prediction = f"{entry['first_chunk_comparison']['max_abs_action_difference_degrees']:.3f}°"
            clips.append(f'''<section><h2>{escape(trial['label'])} · {outcome}</h2>
<p>初始偏移 X/Y：{shift} mm；抬升：{movement}；ACT实际推理：{result['actual_inference_queries']}次。</p>
<video controls preload="metadata" poster="trials/{trial['id']}/custom/rollout.png" src="trials/{trial['id']}/custom/rollout.mp4"></video>
<p><a href="trials/{trial['id']}/custom/result.json">物理结果</a> · <a href="trials/{trial['id']}/custom/model_identity.json">模型身份</a> · <a href="trials/{trial['id']}/scene/scene_metadata.json">初始场景</a></p></section>''')
        else:
            outcome, movement, prediction = '待运行', '—', '—'
        rows.append(f'<tr><td>{escape(trial["label"])}</td><td>{shift}</td><td>{outcome}</td><td>{movement}</td><td>{prediction}</td></tr>')
    summary = f'{baseline_text}；新位置已测{len(heldout)}/5，完成抓放{count}/{len(heldout)}。'
    if not heldout:
        summary = f'{baseline_text}；尚无新位置结果。'
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACT 冻结权重 · 位置泛化验证</title><style>body{margin:0;background:#101820;color:#e6edf3;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1100px;margin:auto;padding:28px}h1{font-size:29px}h2{font-size:23px}p{color:#c1ccd6}a{color:#83cbff}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #3b4c59;text-align:left}.note{background:#202c36;padding:15px;border-left:4px solid #ffcc72}video{width:100%}code{overflow-wrap:anywhere}</style><main>
<h1>ACT 冻结权重：方块换位置后还能抓吗？</h1><p><strong>SUMMARY</strong></p>
<p>状态：STATUS。模型与相机、机械臂起点、盒子、物体尺寸、控制器均固定；只改变开局红块XY位置。每个新位置仅运行一次，15秒，30Hz执行，每次100步后重新读取视觉。测试期间不训练、不修正轨迹。</p>
<table><thead><tr><th>试次</th><th>X/Y偏移 mm</th><th>结果</th><th>最大抬升</th><th>首段动作相对原点差异</th></tr></thead><tbody>ROWS</tbody></table>
<p class="note">动作随画面变化，说明模型使用了视觉信息；是否能完成新位置抓放，要看物理结果。5个位置是小范围位置适应性检查，不能代表ACT算法的普遍性能，也不是随机总体成功率。程序失败与物理任务失败分别记录。</p>
<p>策略输入严格只有腕部RGB、顶部RGB和7维关节状态；物体真值仅用于仿真初始化及评分。每段视频来自本次新调用ACT的物理运行；右侧连续视角与模型实际采样频率分开标注。</p>
<p><a href="protocol.json">测试前冻结的完整协议</a> · <a href="generalization_report.json">原始结果与权重前后哈希</a> · <a href="../fixed_flat40_20261023/index.html">之前的固定场景结果</a></p>
CLIPS</main></html>'''
    page = page.replace('SUMMARY', summary).replace('STATUS', report['status'])
    page = page.replace('ROWS', '\n'.join(rows)).replace('CLIPS', '\n'.join(clips))
    (output / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output = protocol_path.parent
    if (output / 'generalization_report.json').exists():
        raise FileExistsError('Earlier evaluation is preserved; preregister a fresh protocol directory')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol['training_steps_allowed'] != 0 or protocol['executed_actions_per_chunk'] != 100:
        raise ValueError('Unexpected frozen evaluation protocol')
    if len(protocol['trials']) != 6 or protocol['trials'][0]['id'] != 'baseline':
        raise ValueError('Expected one baseline and five preregistered new positions')
    model_dir = Path(protocol['model_directory'])
    expected = protocol['model_files_sha256']
    if model_hashes(model_dir) != expected:
        raise ValueError('Frozen checkpoint changed before evaluation')
    fit_path = Path(protocol['camera_fit_path'])
    if digest(fit_path) != protocol['camera_fit_sha256']:
        raise ValueError('Frozen camera fit changed')
    for name, expected_code in protocol['code_files_sha256'].items():
        if digest(ROOT / name) != expected_code:
            raise ValueError(f'Evaluation code changed after preregistration: {name}')
    initial = dataset_state_to_sim(protocol['initial_public_joint_target_degrees'])
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': digest(protocol_path), 'protocol': protocol,
              'results': [], 'training_steps': 0, 'hardware_commands_sent': 0,
              'teacher_action_sequence_read': False,
              'policy_input_fields': ['state', 'wrist', 'top'],
              'object_truth_passed_to_policy': False, 'cpu_budget': CPU_BUDGET}
    save_json(output / 'generalization_report.json', report)
    write_page(output, report)
    reference_scene = Path(protocol['successful_reference_scene'])
    if digest(reference_scene) != protocol['successful_reference_scene_sha256']:
        raise ValueError('Successful baseline scene changed')
    normalized_scene = normalized_scene_tree(ET.parse(reference_scene).getroot())
    baseline_state = baseline_action = None
    try:
        for trial in protocol['trials']:
            trial_dir = output / 'trials' / trial['id']
            scene, metadata = build_scene(trial_dir / 'scene', square_brick=True,
                square_edge_m=.040, flat_brick=True, wrist_camera=fit)
            tree = ET.parse(scene)
            body = tree.find(".//body[@name='target_brick']")
            position = np.asarray(protocol['reference_brick_center_m']).copy()
            position[:2] += trial['offset_xy_m']
            body.set('pos', ' '.join(f'{value:.9g}' for value in position))
            # Normalizing only path spelling and initial brick position must
            # leave exactly the same scene tree for all six trials.
            structural_bytes = normalized_scene_tree(tree.getroot())
            if structural_bytes != normalized_scene:
                raise ValueError('A scene property other than initial brick position changed')
            tree.write(scene, encoding='utf-8', xml_declaration=True)
            metadata['brick']['centre_m'] = position.tolist()
            metadata['placement_basis'] = 'Preregistered held-out XY evaluation, fixed before any trial outcome'
            metadata['position_evaluation'] = trial
            save_json(trial_dir / 'scene/scene_metadata.json', metadata)
            if model_hashes(model_dir) != expected:
                raise ValueError('Frozen checkpoint changed between trials')
            result = run_checkpoint('custom', scene, metadata, trial_dir / 'custom', initial,
                protocol['duration_s'], 100, model_dir=model_dir, label=f"ACT · {trial['label']}")
            identity = json.loads((trial_dir / 'custom/model_identity.json').read_text(encoding='utf-8'))['model_identity']
            if identity['files_sha256'] != expected or result['model_sha256'] != expected['model.safetensors']:
                raise ValueError('Worker loaded a different model from the frozen checkpoint')
            original_state_difference = float(np.max(np.abs(np.asarray(result['initial_public_state'])
                - np.asarray(protocol['successful_reference_initial_public_state']))))
            if original_state_difference > 1e-7:
                raise ValueError('Initial settled robot state differs from the successful reference')
            with np.load(trial_dir / 'custom/observation_000.npz', allow_pickle=False) as saved:
                state = saved['state'].copy()
            with np.load(trial_dir / 'custom/chunk_000.npz', allow_pickle=False) as saved:
                action = saved['action'].copy()
            if baseline_state is None:
                baseline_state, baseline_action = state, action
            state_difference = float(np.max(np.abs(state - baseline_state)))
            if state_difference > 1e-7:
                raise ValueError('Initial robot state differs: cannot isolate visual position change')
            entry = {'id': trial['id'], 'offset_xy_m': trial['offset_xy_m'],
                     'scene_sha256': digest(scene), 'result': result,
                     'first_chunk_comparison': {
                         'initial_state_max_difference_from_previous_success_degrees': original_state_difference,
                         'initial_joint_state_max_difference_degrees': state_difference,
                         'max_abs_action_difference_degrees': float(np.max(np.abs(action - baseline_action))),
                         'mean_abs_action_difference_degrees': float(np.mean(np.abs(action - baseline_action)))}}
            video = trial_dir / 'custom/rollout.mp4'
            if video.is_file():
                destination = ROOT / '视频' / f"ACT_位置泛化_{trial['id']}_20261023.mp4"
                if destination.exists():
                    raise FileExistsError(f'Preserve existing video: {destination}')
                shutil.copy2(video, destination)
                if digest(destination) != digest(video):
                    raise ValueError('Delivered video differs from original')
                entry['delivered_video'] = str(destination)
            report['results'].append(entry)
            save_json(output / 'generalization_report.json', report)
            write_page(output, report)
            print(json.dumps({'trial': trial['id'], 'success': result['metrics']['success'],
                              'lift_mm': result['metrics']['maximum_lift_m'] * 1000,
                              'first_chunk': entry['first_chunk_comparison']}, ensure_ascii=False), flush=True)
            if result['status'] != 'completed':
                raise RuntimeError('Trial aborted; keep execution error separate from grasp failure')
            if trial['id'] == 'baseline' and not result['metrics']['success']:
                raise RuntimeError('Baseline replication failed; shifted-position results would be inconclusive')
        report['status'] = 'completed_frozen_position_evaluation'
    except Exception as error:
        report['status'] = 'evaluation_error'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['model_files_sha256_after'] = model_hashes(model_dir)
        report['frozen_model_unchanged'] = report['model_files_sha256_after'] == expected
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        heldout = [entry for entry in report['results'] if entry['id'] != 'baseline']
        report['new_position_trials_completed'] = len(heldout)
        report['new_position_successes'] = sum(bool(e['result']['metrics']['success']) for e in heldout)
        save_json(output / 'generalization_report.json', report)
        write_page(output, report)
    if not report['frozen_model_unchanged']:
        raise RuntimeError('Model changed during evaluation')


if __name__ == '__main__':
    main()
