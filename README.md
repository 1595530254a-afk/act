# ACT 抓放项目入口

更新日期：2026-10-19（Asia/Shanghai）

本项目的资料、配置、数据、模型、训练输出和部署说明统一保存在 `D:\外置大脑\ACT`。继续本项目时先读本文件，再按需查看 `resources/resource_manifest.json` 和相应配置。

## 已确定的路线

**台式机负责模型加载、训练和离线验证；笔记本负责推理，并连接 B601-RS 与相机。**

第一阶段优先复用公开数据和现成 ACT 权重，暂不把自行采集示范或购买示教主臂作为前置条件。能否直接迁移到本机，需要验证后判断。

“加载现成权重”和“训练”是两件事：

- 加载权重：使用别人已经训练好的策略，先验证模型能否运行、接口是否符合要求。
- 继续训练：从现成权重开始，用公开数据更新参数。是否需要训练以及训练多久，以实验目的和验证结果决定。
- 重新训练：使用公开示范重新训练 ACT 任务策略，用于学习和复现实验。此阶段不涉及大型通用 VLA 的预训练。

## 设备与当前状态

| 项目 | 已知情况 | 验证边界 |
| --- | --- | --- |
| 台式机 | i5-13600KF、32GB 内存、RTX 5060 Ti 8GB、Windows 11 | 本机已读取硬件信息；尚未运行 ACT |
| Python | `D:\Python\python.exe`，3.10.10 | 此解释器内未发现 PyTorch、LeRobot、huggingface_hub；不代表其他环境均未安装 |
| 磁盘 | D 盘检查时约剩余 122GiB | 瞬时值，下载和训练前重新确认 |
| 笔记本 | 用户报告为联想拯救者，约 2021 年购买，R9000P 或 Y9000P，Windows，疑似 NVIDIA 30 系显卡 | CPU、GPU、显存、内存、驱动和推理性能均待读取，不能按系列名称推定 |
| 机械臂 | B601-RS，已装配；上位机可以驱动机械臂 | 用户报告；尚未验证 Python SDK / LeRobot 的状态读取和动作接口 |
| 相机 | 用户报告已有 D405 与 D435i，已安装腕部相机 | 当前安装位置、序列号、画面和数据流待检查 |
| 示教设备 | 只有 B601，无示教主臂 | 优先复用公开资源；是否需要补采本机数据由迁移表现决定 |
| Git | 已在 ACT 目录内建立独立仓库，当前分支为 `master` | 用户指定远端为 https://github.com/1595530254a-afk/act.git；只归档脚本、配置和小型报告 |

## 首个公开候选

- [数据集：nikodembartnik/rebot_cube](https://huggingface.co/datasets/nikodembartnik/rebot_cube)
- [模型：nikodembartnik/act_rebot_cube](https://huggingface.co/nikodembartnik/act_rebot_cube)

已核实：数据包含 50 条示范、20,948 帧、30 FPS；模型任务描述为“把红色方块放入灰色盒子”。输入为腕部和俯视两路 640×480 彩色图像，以及 7 维关节/夹爪状态；输出为 7 维动作。发布者未提供评测结果。两者许可标注为 Apache-2.0。

作者将候选收录在 B601-RS 合集中，但元数据使用通用的 `rebot_b601_follower`。实际电机版本、关节标定与本机的一致性仍需核对。公开可下载不等于已经验证了本机抓放成功率。

| 契约 | 当前候选的定义 |
| --- | --- |
| 状态输入 | `observation.state`，7 维 |
| 腕部图像 | `observation.images.wrist`，模型形状 `[3, 480, 640]` |
| 俯视图像 | `observation.images.top`，模型形状 `[3, 480, 640]` |
| 动作输出 | `action`，7 维 |
| 顺序 | shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_yaw、wrist_roll、gripper |
| 预测段长度 | `chunk_size=100`；实际消费方式和观察更新频率需要部署时配置、验证 |

**复用该双视角权重时，需要提供对应的两个真实视角。**建议检查 D405 腕部与 D435i 俯视布置，并尽量接近原数据中的画面。若采用单腕部方案，可从公开数据筛选腕部输入重新训练相应模型；不能直接省掉现有权重要求的俯视输入。

单纯修改相机键名不会消除视角差异；单纯让关节维数相同也不会消除零位、方向、单位和夹爪定义的差异。归一化与反归一化信息必须和模型配套使用。

## 当前已完成的工作

- 确认项目目录，并记录台式机训练、笔记本推理的分工。
- 读取台式机 GPU、Python 和磁盘状态。
- 核实公开数据集和模型的输入输出，固定仓库 revision。
- 保存配置快照及其本地 SHA-256，详见 `resources/resource_manifest.json`。
- 建立项目内的 Python 3.12.15 和虚拟环境；CUDA 版 PyTorch 与 LeRobot 正在安装。
- 按固定 revision 下载最终模型、预处理/后处理与归一化文件，以及完整数据集。
- 校验 8 个模型文件、11 个数据文件的上游内容哈希；记录见 `reports/download_manifest.json`。

当前模型文件约 197MiB，数据文件约 829MiB，均已下载并校验。**尚未开始训练或真机测试；离线推理等待环境安装完成后执行。**

安装、缓存和下载均位于本目录。`scripts/setup_environment.ps1` 用于建立环境；`scripts/download_resources.py` 用于固定版本下载及哈希核对。Python 推理脚本和 PowerShell 启动器在验证通过后记录正式运行结果。

## 下一步与验收

1. **建立独立训练环境。**选择与候选模型和 RTX 50 系相兼容的 Python、PyTorch、LeRobot 版本，记录实际版本；验证 CUDA 张量运算。现有 Python 3.10 不直接当作已就绪的训练环境。
2. **准备完整模型与公开数据。**按固定 revision 获取最终模型及配套预处理、后处理、归一化文件；数据、模型和缓存放在 ACT 目录内。部署不需要下载全部历史训练检查点及优化器状态。
3. **台式机离线推理。**从公开示范读取真实图像和状态，检查完整处理流程、动作形状、有限值和目标范围，不向真机发送动作。
4. **小规模训练验证。**先用保守 batch 测量显存与每步耗时，保存并重新加载检查点。是否继续完整训练，依据学习目的和验证结果决定；不直接照抄发布者的 batch 16 与 100,000 步。
5. **笔记本离线测速。**读取具体硬件，迁移完整模型部署包和兼容环境，测量持续推理的延迟、抖动与内存占用。模型能加载不等于已经满足动作供给和闭环反应需求。
6. **核对真机接口后低速测试。**先检查本机标定、关节单位和方向、夹爪范围、两个视角和可达区域。模型输出先检查，再通过限位和速度限制执行；记录抓空、滑落、放错及超时。根据表现决定是否补采现场数据。

笔记本的 Windows 真机控制路线需要单独核实 USB-CAN 型号及驱动。上位机能动不代表 LeRobot 已经接通。LeRobot 当前主线文档将 Windows native-CAN 标为实验性，要求相应的 PEAK PCAN 和 PCAN-Basic 支持；固定版本的厂商路线也需要实测。系统选择在接口核实后决定。

## 文件说明

- `README.md`：项目入口、决策和当前状态。
- `resources/resource_manifest.json`：源地址、固定 revision、配置快照哈希和下载状态。
- `resources/model_config.json`：公开模型接口配置快照。
- `resources/model_train_config.json`：发布者训练配置快照；作为来源记录，不直接作为本机执行命令。
- `resources/model_preprocessor.json`、`resources/model_postprocessor.json`：处理流程配置快照；其引用的张量文件还未下载。
- `resources/dataset_info.json`：数据结构快照。

后续按实际需要在本目录内建立独立环境、`data`、`models`、`runs` 和部署包目录，避免将 ACT 项目输出散落到其他工作区。

## 官方参考

- [LeRobot ACT](https://huggingface.co/docs/lerobot/en/act)
- [B601 当前主线接入说明](https://huggingface.co/docs/lerobot/main/rebot_b601)
- [矽递 B601-RS LeRobot 教程](https://wiki.seeedstudio.com/rebot_arm_b601_rs_lerobot/)

上述文档可能随版本变化；后续执行以固定代码版本、实际帮助信息和已验证接口为准。
