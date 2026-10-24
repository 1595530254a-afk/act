# 第一阶段源码与仿真重建说明

归档日期：2026-10-24。选定本地 ACT 为 **step3000 + ensemble1**，模型 SHA256 为 `c25ffa08fd2b67343f37eab06bb948642e2a6d2578e387a563ba81105c506c2c`。原先登记的8点测试8/8；随后追加 (+10,0) 毫米位置单次1/1，分组记录，不能合并成未经限定的总体成功率。

本说明核对源码、依赖和必要输入，没有执行新的安装、训练、ACT或物理回合。仿真通过不是B601真机验证。

## 1. Git保存什么

`simulation/` 是任务场景、物理控制、训练适配、评估、原始预测复算和视频回看的源码。`scripts/cpu_budget.py` 为共享依赖。57个仿真Python文件已通过AST解析，内部Python导入均可解析到仿真目录或共享脚本；它不代替运行测试。

`source_manifest.json` 记录74个源码/公共配置文件的大小与SHA256、实际环境版本、选定模型六文件身份以及11段精选视频。Git不保存环境、原始示范数据、完整运行缓存或模型大权重。

选定 `model.safetensors` 为206,707,932字节，超过100MiB。实际六文件由 **stage2本地包** 提供：解压后的 `models/act_long5000_step3000/`，不是原始公开checkpoint，也不是调用旧笔记本包里的模型。

## 2. 恢复最小冻结输入

以下副本随源码提供，原件没有改动。`runtime_inputs/path_mapping.json` 列出原SHA、公开副本SHA及恢复目标；含项目路径的JSON用 `<ACT_ROOT>` 表示新机器项目根。占位替换会改变文件字节与SHA，不得声称它们仍是原始冻结文件。

|公开副本（runtime_inputs下）|恢复到新项目根下的目标|
|---|---|
|source_protocol.json|runs/mujoco_validation_20261023/position_generalization_20261023/protocol.json|
|camera_fit.json|runs/mujoco_iteration_20261023/camera_fit/camera_fit.json|
|successful_reference_scene.xml|runs/mujoco_validation_20261023/fixed_flat40_20261023/policy_step300_progress_fix/scene/task_scene.xml|
|position_plan.json|runs/mujoco_iteration_20261024/position_plan.json|
|final_selection.json|runs/mujoco_iteration_20261024/final_selection.json（归档说明，不是评估必需输入）|
|upstream_manifest.json|simulation/upstream_manifest.json（现有下载清单优先核对，勿盲目覆盖）|

`compact_results.json` 是精简结果证据，不作为策略输入。参考XML在评估器中用于场景结构核对；新回合场景和网格纹理由代码生成。参考XML恢复至表中原有深度，其mesh相对路径才与原布局对应。

新机器按以下顺序操作，选择未存放旧冻结结果的工作副本：

1. 在新项目根创建表中所需目录，把副本分别放到对应目标。对JSON中的 `<ACT_ROOT>` 做JSON安全的字符串替换；Windows反斜杠须正确转义。XML无需修改时保留原字节。
2. 用新项目根的实际绝对路径填写恢复后source protocol的 `camera_fit_path` 与 `successful_reference_scene`；其他含 `<ACT_ROOT>` 的路径也改为实际路径。协议内旧模型路径是历史身份记录，实际重跑模型由 `--model-dir` 明确指定。
3. 根据实际恢复文件重新计算source protocol的 `camera_fit_sha256` 和 `successful_reference_scene_sha256`。例如可用 `Get-FileHash -Algorithm SHA256 -LiteralPath ...`；若重新格式化过JSON，也必须使用格式化后实际文件的hash。
4. 保存恢复后的protocol，再记录它、position_plan、相机fit与参考XML的**新SHA256**；保留path_mapping中的原SHA作来源证据。原文件和修改后的重建文件分别保存，不修改原始冻结报告中的身份。
5. 核对 `simulation/upstream_manifest.json`，用 `simulation/fetch_model.py` 下载固定commit的44项官方资产到 `local/mujoco_upstream`；之后 `--verify-only` 只核验已有资产。缓存不能从空目录直接用verify-only代替下载。新回合会在protocol中记录本次实际源码和输入hash。

这补齐了评估器读取的最小场景、相机、初态和位置输入，不包含原始全部3600张双相机观察及动作块。完整原运行视频/网络复算证据须使用本地原始归档，不能只凭这些模板宣称原运行已被重新执行。

## 3. 使用分离环境

- ACT `.venv`：Python3.12、Torch2.10.0+cu130、torchvision0.25.0+cu130、LeRobot0.6.1、NumPy2.2.6；完整已用版本见根 `requirements.lock.txt`。CUDA版Torch需要匹配发行索引和驱动。
- 仿真 `local/mujoco-env`：MuJoCo3.17.0、NumPy2.2.6、Pillow13.0.0、psutil7.2.2；锁定文件为 `simulation/requirements.lock.txt`。无须在这里再安装Torch。
- 可选ALOHA校验为独立环境gym-aloha0.1.5、MuJoCo3.2.7、dm-control1.0.27，是双臂递交方块，不是B601放盒子任务，不能混装其MuJoCo版本。

各环境实际版本已写入 `source_manifest.json`。CUDA worker目前使用 `<ACT_ROOT>/.venv/Scripts/python.exe`；新机器须采用该Windows布局，或另行进行明确的代码移植。不要把源码存档描述为跨平台一键安装包。已有环境无需再次安装。

## 4. 重跑评估

在项目根运行；`<STAGE2_PACKAGE>` 替换为stage2包实际解压目录。输出目录必须不存在，保留旧结果。

```powershell
& 'local\mujoco-env\Scripts\python.exe' -B 'simulation\evaluate_feedback_20261024.py' `
  --position-plan 'runs\mujoco_iteration_20261024\position_plan.json' `
  --split test --mode ensemble1 `
  --model-dir '<STAGE2_PACKAGE>\models\act_long5000_step3000' `
  --output-dir 'runs\stage1_reconstructed_test_new'
```

先核对stage2包的六文件模型hash与 `source_manifest.json`，尤其上述权重hash。该命令实际启动新的CUDA网络和物理回合；需要相应硬件和环境，不能当成播放命令。物理模型1ms子步，动作30Hz，每回合450步；策略只读取7维状态和顶部/腕部RGB，不读取物体真值、教师动作或IK轨迹。

窄测试入口可分别运行：

```powershell
& '.venv\Scripts\python.exe' -m unittest discover -s simulation -p 'test_joint_mapping.py'
& 'local\mujoco-env\Scripts\python.exe' -m unittest discover -s simulation -p 'test_evaluate_feedback_20261024.py'
```

这两个命令在本次静态审计中没有重新执行。先完成依赖及恢复输入核对，再运行新回合。

## 5. 回看与复算

完整评估完成后，`simulation/render_feedback_20261024.py --evaluation NEW/evaluation.json --report-dir NEW_PAGE` 仅恢复保存qpos/qvel、计算姿态并编码视频，不新增网络调用或物理子步。单点视频右侧直接读保存的真实策略RGB；30fps与30Hz仿真查询频率都不代表墙钟实时吞吐。可直接打开本地HTML或MP4，不要求启动HTTP服务。

`simulation/verify_feedback_outputs_20261024.py` 对保存输入做新的网络forward，并核对原始100×7动作块；它不是重新运行物理。原8点3600块/2,520,000动作值已逐值一致，追加点证据分开保存。

时间集成公式为 `w_i = exp(-0.01*i) / Σ_j exp(-0.01*j)`，按最旧到最新排序。稳态100份预测中最新权重约0.585%，观测来源平均年龄57.70帧/1.923秒。因此每帧新forward与完全执行最新预测是不同语义；本轮没有单独隔离其因果效应。

## 来源与许可

公开ACT来自 `nikodembartnik/act_rebot_cube` revision `409b639946cc4140b3934a6e3d2c255f4335ee9d`；数据 `nikodembartnik/rebot_cube` revision `c5cdae89d2e1f474a8e2f1ac5f16302126d0c33f`。本地固定版本模型卡、数据卡均声明Apache-2.0。

官方资产来自 `Seeed-Projects/reBot_RS_Sim` commit `438f7713d83fb8cd0d0cb05e8cc335264dd3997a`；逐文件来源/hash见upstream manifest。此次Git不重新发布官方mesh。缓存 `THIRD_PARTY_MODELS.md` 明确D405支架来自 [Bowen Zhu](https://github.com/bowenszhu/rebot-b601-rs-d405-wrist-mount)，v1.0.0、commit `ca379b9a4e19096730cd255dfc35ec419a7ad144`，CERN-OHL-W-2.0；不能将其推及所有STL。缓存未包含整套上游仓库的根LICENSE，整套资产许可未完整核对。

当前项目没有根LICENSE；本次发布是用户授权的源码阶段存档，不代表替第三方内容重新授权。后续如宣布统一开源许可，应由仓库所有者明确选择，并保留各自归属。
