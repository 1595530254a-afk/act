"""Plot saved desktop ACT diagnostics; requires Matplotlib, no model execution."""

import csv
import json
from pathlib import Path

from cpu_budget import CPU_BUDGET

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]


def main():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    training = json.loads((ROOT / 'reports/controlled_training.json').read_text(encoding='utf-8'))
    assessment = json.loads((ROOT / 'reports/desktop_final_evaluation.json').read_text(encoding='utf-8'))
    with (ROOT / 'reports/controlled_training_metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row['step']) for row in rows]
    losses = np.array([float(row['loss']) for row in rows])
    history = training['validation_history']
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(steps, losses, color='#a5b9ce', linewidth=0.6, alpha=0.65, label='每步损失')
    axes[0].plot(steps[24:], np.convolve(losses, np.ones(25) / 25, mode='valid'),
                 color='#25659b', linewidth=1.6, label='25 步移动平均')
    axes[0].set(xlabel='训练步数', ylabel='L1 + 10 × KL', title='训练损失（批次不同，会波动）')
    axes[0].legend(frameon=False)
    axes[1].plot([row['step'] for row in history], [row['normalized_chunk_mae'] for row in history],
                 'o-', color='#2a8062', linewidth=1.8)
    axes[1].set(xlabel='训练步数', ylabel='归一化动作段 MAE', title='固定 5 条验证示范 / 50 帧观察')
    axes[1].axvline(training['best_step'], color='#9a6c2f', linestyle='--', alpha=0.7, label='选中检查点')
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis='y', alpha=0.2)
    figure.savefig(ROOT / 'reports/desktop_training_curves.png', dpi=170)
    plt.close(figure)

    labels = ['公开原模型', '原 100 步短训', '本轮 1,000 步候选']
    keys = ['published', 'short100', 'selected1000']
    values = [assessment['models'][key]['normalized_chunk_mae'] for key in keys]
    figure, axis = plt.subplots(figsize=(7.6, 4.1), constrained_layout=True)
    bars = axis.bar(labels, values, color=['#6583a4', '#ba8d66', '#398368'], width=0.6)
    axis.bar_label(bars, fmt='%.5f', padding=4)
    axis.set(ylabel='归一化动作段 MAE（越低越好）', ylim=(0, max(values) * 1.2),
             title='50 条示范 / 500 帧离线诊断\n公开预训练已见过这些数据，不代表真机成功率')
    axis.grid(axis='y', alpha=0.2)
    figure.savefig(ROOT / 'reports/desktop_model_comparison.png', dpi=170)
    plt.close(figure)

    artifact_root = ROOT / assessment['artifacts_directory']
    source_trace = json.loads((artifact_root / 'published_traces.json').read_text(encoding='utf-8'))[0]
    selected_trace = json.loads((artifact_root / 'selected1000_traces.json').read_text(encoding='utf-8'))[0]
    if source_trace['frame_index'] != selected_trace['frame_index'] or source_trace['target'] != selected_trace['target']:
        raise RuntimeError('Traces must compare the same observation and demonstration actions.')
    names = ['肩部水平', '肩部俯仰', '肘部', '腕部俯仰', '腕部偏航', '腕部旋转', '夹爪']
    figure, axes = plt.subplots(4, 2, figsize=(11, 10), constrained_layout=True)
    time_s = np.arange(100) / 30
    for channel, axis in enumerate(axes.flat[:7]):
        axis.plot(time_s, np.array(source_trace['target'])[:, channel], color='#303030', label='示范标签')
        axis.plot(time_s, np.array(source_trace['prediction'])[:, channel], color='#6583a4', label='公开原模型')
        axis.plot(time_s, np.array(selected_trace['prediction'])[:, channel], color='#398368', label='本轮候选')
        axis.set(title=names[channel], xlabel='预测时域 / 秒', ylabel='数据集坐标值')
        axis.grid(alpha=0.15)
    axes.flat[7].axis('off')
    axes.flat[7].legend(*axes.flat[0].get_legend_handles_labels(), loc='center', frameon=False)
    figure.suptitle(f'帧 {source_trace["frame_index"]}：一次观察预测的 100 步动作段\n不是闭环轨迹；坐标单位与真机标定尚未核对', fontsize=13)
    figure.savefig(ROOT / 'reports/desktop_action_chunk_example.png', dpi=150)
    plt.close(figure)
    print(f'Saved three diagnostic figures. CPU budget: {CPU_BUDGET}')


if __name__ == '__main__':
    main()
