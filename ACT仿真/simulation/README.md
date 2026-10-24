# B601-RS 的 ACT 仿真源码

作者：**请输入名字QaQ**。当前第一阶段结果为多位置新分支 **step3000＋ensemble1**：原位1/1、四验证位置4/4，冻结后八个新位置8/8，用户随后追加一点1/1。结果限于固定物体、相机和动力学下的局部XY变化。

较早的单点sim300在五个新位置0/5，是后续改进的起点；它不是当前最终策略。完整失败与迭代过程见[教学复盘](../docs/stage1_20261024/ACT仿真训练复现与调参复盘.md)，实际命令、数据条件与环境安装见[训练复现步骤](../docs/stage1_20261024/训练复现步骤.md)。

## 按数据流找代码

| 环节 | 文件 |
| --- | --- |
| 固定官方模型资产 | [fetch_model.py](fetch_model.py)、[upstream_manifest.json](upstream_manifest.json) |
| 公开关节角度到仿真坐标 | [joint_mapping.py](joint_mapping.py)、[MAPPING.md](MAPPING.md) |
| 场景与接触动力学 | [task_scene.py](task_scene.py)、[task_dynamics.py](task_dynamics.py) |
| 单位置合成示范和适配 | [export_sim_demonstration.py](export_sim_demonstration.py)、[adapt_act_sim.py](adapt_act_sim.py) |
| 多位置教师和RGB | [generate_multisite_teacher.py](generate_multisite_teacher.py)、[export_multisite_rgb.py](export_multisite_rgb.py) |
| 连续5000步训练 | [adapt_act_multisite_20261024.py](adapt_act_multisite_20261024.py) |
| CUDA策略输入与输出 | [policy_worker_cuda_20261024.py](policy_worker_cuda_20261024.py)、[policy_process_cuda_20261024.py](policy_process_cuda_20261024.py) |
| chunk100与逐帧融合评估 | [evaluate_feedback_20261024.py](evaluate_feedback_20261024.py)、[evaluate_act_ensemble.py](evaluate_act_ensemble.py) |
| 选模和留出位置验证 | [run_long5000_validation_20261024_v2.py](run_long5000_validation_20261024_v2.py)、[run_position_generalization.py](run_position_generalization.py) |

## 固定条件和验收

红块40×40×20 mm、平顶；灰盒外尺寸120×120×50 mm，壁和底厚4 mm。这是本项目建立的仿真几何，相机参数同样是虚拟重建参数。训练输入为顶部与腕部RGB及七维关节状态，输出100×7动作块；最终执行方式每个仿真步重新推理并做时间融合。

任务评分要求红块抬升至少3 cm、外轮廓完整进入盒内、落在盒底、速度低于0.05 m/s且最终夹爪脱离；另外记录双指持续接触及抬起帧。没有用绑定或传送物体代替物理抓取。30 Hz指仿真控制时间，不代表电脑实际推理速度达到30 Hz。

## 运行位置

先在克隆仓库根进入 `ACT仿真`，后续命令从这个阶段目录执行。ACT使用 `.venv`，MuJoCo使用 `local/mujoco-env`，依赖分别锁定；首次运行还需恢复最小运行输入、下载固定官方资产并取得完整checkpoint。仓库不包含20 GiB RGB缓存或Python环境。请按复现步骤选择直接评估或重新训练路线，避免把历史单点启动脚本当成最终模型入口。

源码中保留了早期运动学回放和固定单点实验工具，供理解诊断过程；它们的运行条件、策略版本与最终评估不同。当前训练日志与结果入口统一在[阶段首页](../README.md)。

上游算法、模型、数据与资产归属见[作者和上游说明](../../AUTHORS.md)。
