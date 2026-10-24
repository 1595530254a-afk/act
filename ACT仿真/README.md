# ACT仿真 第一阶段

作者：**请输入名字QaQ** · [1595530254a-afk](https://github.com/1595530254a-afk)

本阶段复现B601机械臂将红块放入灰盒的MuJoCo任务，记录从公开权重迁移失败，到单点适配、多位置训练、执行方式改进与留出位置验证的过程。

## 从这里阅读

1. [教学复盘](docs/stage1_20261024/ACT仿真训练复现与调参复盘.md)：数据、loss、学习率、batch、训练步数、失败迭代与选模依据。
2. [训练复现步骤](docs/stage1_20261024/训练复现步骤.md)：材料条件、数据生成、检查、训练与评估命令。
3. [PDF报告](reports/ACT仿真训练复现与调参复盘.pdf)与[可编辑Word](reports/ACT仿真训练复现与调参复盘_修订版.docx)：按起步、失败、原因、修改与结果讲清训练过程；统一字体与图表，不含视频附件或视频链接。

## 本阶段结果

| 项目 | 保存结果 |
| --- | --- |
| 最终策略 | 新5000步分支的step3000＋ensemble1 |
| 训练数据 | 25个合成位置、10,075帧；4个验证位置、1,612帧 |
| 验证 | 原位1/1；4验证位置4/4 |
| 冻结后的新位置 | 预先保留8点8/8；用户随后追加1点1/1，分开统计 |
| 范围 | 固定物体、相机与动力学下的局部XY插值，不是任意场景或真机成绩 |

## 源码和成果

- [simulation](simulation/)：场景、教师、训练、反馈评估、物理评分与复算源码。
- [依赖与最小输入恢复](docs/stage1_20261024/public_reproduction.md)：环境分离、路径及冻结输入。
- [训练记录](results/stage1_20261024/training_record.json)与[完整5000步loss日志](results/stage1_20261024/training_metrics.csv)：实际超参数、五个检查点、选模身份和逐步指标。
- [验证结果](docs/stage1_20261024/runtime_inputs/compact_results.json)与[发布清单](results/stage1_20261024/release_manifest.json)：八点与追加点分开统计，文件可按SHA256核对。
- [署名成果视频目录](results/stage1_20261024/videos/)：包含成功、失败以及最终验证的展示副本。
- [八点同步总览](results/stage1_20261024/videos/v04_final8_overview.mp4) · [实际双相机输入](results/stage1_20261024/videos/v05_final8_test00.mp4) · [追加位置](results/stage1_20261024/videos/v13_extra_p10_0.mp4)

所有命令从本目录运行，即克隆仓库后先进入 `ACT仿真`。源码中的项目根指本目录。Git不包含20 GiB RGB缓存和大模型文件；初始化与数据生成条件见复现步骤，不宣称clone之后就能立刻训练。

原ACT与LeRobot、公开模型/数据和Seeed资产来源见[作者与上游说明](../AUTHORS.md)。
