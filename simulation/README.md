# B601-RS 与 ACT：MuJoCo 仿真验证

## 最新：冻结权重的位置泛化检查（2026-10-23）

`run_position_generalization.py` 按测试前冻结的协议复核原位置并测试五个初始XY偏移。原位置1/1通过；X±30mm、Y±30mm及X+30/Y−40mm五个新位置**0/5完成抓放**。全部程序正常完成，无MuJoCo数值警告，模型六文件前后哈希不变。相同初始七维状态下首段输出随图像变化，但当前策略未通过3–5cm位置变化检查。不能将固定起点成功等同于位置泛化，也不能据此推广ACT算法的整体表现。

结果页：`runs/mujoco_validation_20261023/position_generalization_20261023/index.html`。协议、全部原始输入/输出/轨迹/失败视频同目录保存；视频亦交付到用户指定的 `D:\外置大脑\ACT\视频`。详见 `reports/ACT位置泛化测试报告_20261023.md`。本轮不训练、不修正失败点动作、不更新笔记本默认模型。

## 固定任务基线：平顶红色方块 → 灰色开口盒（2026-10-23）

固定红块 **40×40×20 mm、无凸点**；灰盒外尺寸 **120×120×50 mm**，壁／底厚 **4 mm**，内部 **112×112×46 mm**。编译后的 MuJoCo 几何已与 `cad/pick_place_props_20261023` 交付 STL 的尺寸和哈希核对，不再进行物体尺寸对照。

当前页面：`runs/mujoco_validation_20261023/fixed_flat40_20261023/index.html`。旧 `task_default100/index.html` 已转到当前页面，原页面保存在同目录 `index_before_flat40.html`。

已通过接触物理示范：`runs/mujoco_iteration_20261023/synthetic_demo_flat40_v1/manifest.json`，403帧、30Hz。红块抬升99.04mm，最后落在盒底、无夹指接触、无MuJoCo警告。该示范采用公开episode0的**实测状态作为新控制目标**，并把夹爪278–310帧的松爪目标扩大至−170°以适应40mm红块，六轴路径保持原值。它是新的合成教师，不是原始action列的直接重放，也不是ACT自主成功。

`export_sim_demonstration.py` 导出真实仿真图像、实际关节状态和新教师动作；`adapt_act_sim.py` 在此前本地续训1000步的模型基础上，新适配300步。选定权重为 `runs/mujoco_iteration_20261023/adapt_flat40_v1/checkpoints/step_000300/pretrained_model`，SHA256为 `e82c8794497ab288c6c58fd16008b78770eddb39bc5f5faab3143b1a530c6c76`。训练约93秒，显存峰值1465MiB，模型处理器与统计原样保留。

**ACT物理闭环已通过一次固定初始条件试次**：`fixed_flat40_20261023/policy_step300_progress_fix/custom/result.json`，运行15秒、5次真实推理、30Hz动作执行、抬升94.69mm，完整入盒并停在盒底，末尾无夹指接触；0求解警告，未记录到超过1mm的几何穿透，无物体绑定或传送。

当前按保存模型原定设置**每次执行100步**（约3.33秒后读新画面）。原先人为改成10步重规划时，闭爪发生在预测块第25～37步，反复重算会丢弃后续闭爪，导致相位停滞。恢复100步后同模型同场景完成任务，分析保存在 `policy_step300_v1/custom/analysis/failure_analysis.json`。不要将这项结论推广为所有ACT都必须执行100步。

运行 `simulation/RUN_FIXED_FLAT40.ps1` 可用已核对哈希的权重重新复现；每次创建新的结果目录。它调用 `run_task.py --model-dir PATH --flat-brick --brick-edge-mm 40 --action-steps 100 --duration 15 --wrist-camera-fit runs/mujoco_iteration_20261023/camera_fit/camera_fit.json --output-dir NEW_DIRECTORY`。这是单示范、单场景适配，只证明该固定任务可运行，未验证随机摆放泛化或实机。训练损失和示范成功均不代替自主抓放验收。

相机位姿是从示范画面推定的虚拟参数，不能当作D405实机手眼标定。腕部输入排除group4的自身相机壳遮挡，顶部／总览仍显示相机壳。state/action夹爪比例采用 `0.009 m/rad` 名义90mm总开度，见 `MAPPING.md`。

所有进程保持4个逻辑CPU、低优先级、Torch2线程；训练使用本机CUDA。源权重、源统计和冻结上游模型保持原值，未更改笔记本部署包，未连接真机。后续内容保留为历史实验记录。

## 历史：红色矩形积木 → 灰色开口盒

用户要求修正第一轮的物体与仿真条件后，新增了 `task_scene.py`、`task_dynamics.py`、`policy_worker.py`、`run_task.py`。保留原来的运动学回放作为第一轮记录。

- 新结果：`runs/mujoco_validation_20261023/task_default100/index.html`；原始公开权重与本地 `act_controlled_step1000` 在相同初始状态和场景各运行 15 秒。
- 本轮是物理闭环：虚拟腕部/顶部 RGB ＋仿真关节状态 → ACT 的 100×7 公共角度 → 仿真 PD＋重力补偿力矩 → MuJoCo 接触动力学 → 新画面。使用模型保存的默认执行长度 100 步，30 Hz，每约 3.33 秒重新推理一次。
- 另保留每 10 步重新推理的诊断对照：`runs/mujoco_validation_20261023/task_comparison/index.html`。这是不同的推理执行设置，不能与默认设置混为同一次实验。
- 红积木约 32×24×20 mm，灰盒外尺寸约 120×120×50 mm。**这些是画面重建假设，不是厂家提供的尺寸。** 位置依据候选坐标下示范的抓取/投放位置；桌面为基座平面附近，移除了不相符的 0.10 m 示例桌面。
- 顶部相机近似重建；腕部相机沿原方向前移约 9.4 mm，修复被自身不透明镜头几何遮挡的问题。没有测得作者的真实相机外参或本机零位。
- 不通过绑定、焊接、传送物体制造抓取效果；每次 ACT 输入、完整原始动作块、实际执行轨迹、权重与处理器哈希均保存。

不训练，不改模型权重，不修改上游缓存或笔记本部署包。进程沿用 4 个逻辑 CPU、低优先级、Torch 2 线程预算。重新实验须选择不存在的输出目录：

```powershell
& 'D:\外置大脑\ACT\local\mujoco-env\Scripts\python.exe' 'D:\外置大脑\ACT\simulation\run_task.py' --output-dir 'D:\外置大脑\ACT\runs\mujoco_red_brick_new_trial' --duration 15 --action-steps 100
```

抓取验收要求：积木抬升至少 3 cm、完整外轮廓进入盒内、落在盒底、速度低于 0.05 m/s 且夹爪已脱离。程序完成和模型成功分开记录；失败结果同样保留。重建场景尚非标准评测环境，结果不能替代真实机械臂验证。

## 第一轮保留记录

本目录用于台式机上的纯仿真，不含真实机械臂控制入口。运行环境和输出都在 `D:\外置大脑\ACT` 内。

## 直接查看

- 结果页面：`D:\外置大脑\ACT\runs\mujoco_validation_20261023\index.html`。
- 在 PowerShell 运行 `D:\外置大脑\ACT\simulation\OPEN_REPORT.ps1`，浏览器打开本地结果页。
- 需要可旋转、缩放的三维窗口时，运行 `D:\外置大脑\ACT\simulation\OPEN_VIEWER.ps1`。窗口初始暂停：空格播放/暂停，`1` 切示范，`2` 切 ACT，`R` 重放；关闭窗口退出。鼠标操作使用 MuJoCo 原生界面。

结果页提供两段视频，各有整体、虚拟顶部和虚拟腕部视角：

1. 公开数据集第 0 段示范的 **403 帧状态回放**，源时间戳 0–13.4 秒、30 Hz。
2. **作者原始 ACT 权重**读取第 0 帧真实腕部/顶部图像与七轴状态后，生成 **100×7** 动作的单次预览。不是本地续训候选，也不是连续读取仿真相机的策略闭环。

视频按隔帧采样为 15 FPS；原始数据与模型输出保存在 `inputs.npz` 中，未修改或裁剪。视频中的物体不会被运动学回放抓起。

## 本轮实际发现

不能把笔记本的第三方插件 1.0.5 与 LeRobot 原生 `rebot_b601_follower` 当成同一接口：

- 旧插件观测保持原生电机角度，动作另乘 `[1,1,-1,-1,-1,1,6]`。初版按这个假设解释数据，J2、J3 和夹爪大量超出模型范围；数值诊断保留在 `initial_plugin105_diagnostics.json`。
- 原生 LeRobot 的 RS profile 与 follower 源码明确规定：公开 state/action 同处一种坐标，七轴均以 `-1` 转到 RS 原生电机角度，夹爪没有 `×6`。本轮改为有源码依据的 canonical 候选，**403 帧示范和 100 步预测均没有模型关节范围越界**。
- 这仍不是作者实际采集分支、校准零位或本机动作映射的确认。原生驱动 RS 支持合入时间晚于模型训练记录，作者可能用了未合并分支；作者确实将该模型归入其 RS 集合，但具体采集 CLI 和校准文件未公开。
- 当前官方示例桌面与训练现场未对齐。示范回放有 184 帧、ACT 预测有 6 帧检测到超过 1 mm 的模型几何穿透，主要涉及桌面/夹爪；**不能判定原作者示范有碰撞，也不能宣布碰撞验证通过**。不通过改桌高、删碰撞或裁剪数据来消除这些记录。

坐标证据、公式和边界见 [MAPPING.md](MAPPING.md)，完整本轮诊断见结果目录的 `verification.json`。

## 已完成与边界

已完成：独立 MuJoCo 环境、冻结官方模型资源、原始公开模型 CPU 推理、示范与预测的可视化、关节范围及静态几何接触诊断。

未完成：原训练场景的桌面/盒子/相机位姿对齐、动力学控制跟踪、虚拟相机输入的 ACT 连续闭环、物体抓取验收、本机零位对齐、真实机械臂执行。

本轮只通过 `qpos + mj_forward` 做运动学回放。上游七个 actuator 是力/力矩 motor，不能把关节角度直接写入 `data.ctrl` 当作位置控制。仿真不会验证电机接线、J3 实体装配、机械支撑或失能防坠。

## 复现与验证

依赖在 `local\mujoco-env`，与 ACT 的 `.venv` 分开。仿真、导出和渲染进程复用 `scripts\cpu_budget.py`，限制至 4 个逻辑 CPU、低优先级；ACT CPU 推理为 2 线程，视频编码为 2 线程。这是进程占用限制，不是温度保护。

首次重建环境：

```powershell
& 'D:\外置大脑\ACT\tools\uv\bin\uv.exe' venv --python 'D:\外置大脑\ACT\.python\cpython-3.12.15-windows-x86_64-none\python.exe' 'D:\外置大脑\ACT\local\mujoco-env'
& 'D:\外置大脑\ACT\tools\uv\bin\uv.exe' pip install --python 'D:\外置大脑\ACT\local\mujoco-env\Scripts\python.exe' --cache-dir 'D:\外置大脑\ACT\cache\uv' -r 'D:\外置大脑\ACT\simulation\requirements.lock.txt'
```

原生 MuJoCo 的 Windows 文件接口在本机不能读取绝对中文路径。`preview.py` 在 Python 中定位模型目录，再以 ASCII 相对文件名加载，结束后恢复工作目录；未搬动或修改原资源。

资源校验及重建预览（已有输入时不再推理）：

```powershell
& 'D:\外置大脑\ACT\.python\cpython-3.12.15-windows-x86_64-none\python.exe' 'D:\外置大脑\ACT\simulation\fetch_model.py' --verify-only
& 'D:\外置大脑\ACT\local\mujoco-env\Scripts\python.exe' 'D:\外置大脑\ACT\simulation\preview.py'
```

输入导出器 `export_inputs.py` 由现有 ACT `.venv` 运行，已有目标文件会拒绝覆盖，避免误把不同模型/输入混为一次结果。该导出器依赖 ACT 环境中的 PyArrow 和 PyTorch；仿真环境不需要再安装它们。

窄验证命令分别使用对应环境：

```powershell
& 'D:\外置大脑\ACT\.venv\Scripts\python.exe' -m unittest discover -s 'D:\外置大脑\ACT\simulation' -p 'test_export_inputs.py'
& 'D:\外置大脑\ACT\.venv\Scripts\python.exe' -m unittest discover -s 'D:\外置大脑\ACT\simulation' -p 'test_joint_mapping.py'
& 'D:\外置大脑\ACT\local\mujoco-env\Scripts\python.exe' -m unittest discover -s 'D:\外置大脑\ACT\simulation' -p 'test_preview.py'
```

## 来源

- 模型资源：[Seeed-Projects/reBot_RS_Sim](https://github.com/Seeed-Projects/reBot_RS_Sim/tree/438f7713d83fb8cd0d0cb05e8cc335264dd3997a)，44 文件约 62.1 MiB。逐文件尺寸、Git blob SHA1、SHA256 及归属说明保存在 `upstream_manifest.json`，缓存原字节保存于 `local\mujoco_upstream`。
- ACT：[nikodembartnik/act_rebot_cube](https://huggingface.co/nikodembartnik/act_rebot_cube)，修订 `409b639946cc4140b3934a6e3d2c255f4335ee9d`。
- 数据：[nikodembartnik/rebot_cube](https://huggingface.co/datasets/nikodembartnik/rebot_cube)，修订 `c5cdae89d2e1f474a8e2f1ac5f16302126d0c33f`。
- 作者 [B601-RS 集合](https://huggingface.co/collections/nikodembartnik/seeed-studio-rebot-arm-b601-rs)。
- 原生驱动：[LeRobot RS profile](https://github.com/huggingface/lerobot/blob/e595b7902714ba51f91e47523f66f89c5181b649/src/lerobot/robots/rebot_b601_follower/motor_family.py)及 [follower](https://github.com/huggingface/lerobot/blob/e595b7902714ba51f91e47523f66f89c5181b649/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py)。

第一轮不训练，不安装 ROS 2，不接 CAN，也不修改已有笔记本部署包。
