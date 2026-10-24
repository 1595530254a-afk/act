# 第一阶段依赖与最小输入恢复

作者：请输入名字QaQ。克隆后先进入仓库的 `ACT仿真`，下文的项目根、`<ACT_ROOT>` 都指这个阶段目录。完整命令见[训练复现步骤](训练复现步骤.md)。

## 环境与资产

- ACT环境：Python3.12、Torch2.10.0+cu130、torchvision0.25.0+cu130、LeRobot0.6.1；依赖锁定在根 `requirements.lock.txt`。
- 仿真环境：MuJoCo3.17.0、NumPy2.2.6、Pillow13.0.0、psutil7.2.2；锁定在 `simulation/requirements.lock.txt`，与Torch环境分开。
- 运行 `simulation/fetch_model.py` 按固定清单下载官方44项资产至 `local/mujoco_upstream`；首次下载后才可用 `--verify-only` 核验。

本目录的 `runtime_inputs` 是最小冻结输入副本，不是完整训练集。`path_mapping.json` 记录原路径、原SHA与公开副本SHA；原件没有被本次整理覆盖。

| 公开副本 | 恢复到阶段根下 |
| --- | --- |
| source_protocol.json | runs/mujoco_validation_20261023/position_generalization_20261023/protocol.json |
| camera_fit.json | runs/mujoco_iteration_20261023/camera_fit/camera_fit.json |
| successful_reference_scene.xml | runs/mujoco_validation_20261023/fixed_flat40_20261023/policy_step300_progress_fix/scene/task_scene.xml |
| position_plan.json | runs/mujoco_iteration_20261024/position_plan.json |

按表创建目录并复制。JSON中的 `<ACT_ROOT>` 须用实际阶段根进行JSON安全替换。源protocol的 `camera_fit_path`、`successful_reference_scene` 必须指向恢复后的文件，并重算对应 `camera_fit_sha256`、`successful_reference_scene_sha256`。更改路径或JSON格式会改变SHA，保留原SHA作来源，不冒充逐字节相同。

参考XML恢复到表中层级，其mesh相对路径才与原布局一致。其他场景/纹理由源码生成。`compact_results.json` 是结果摘要，不作为ACT观察输入；`final_selection.json` 是历史选模身份记录，不要求当前机器存在其中原电脑目录。

## 权重与数据条件

直接评估需取得完整六文件checkpoint，显式传 `--model-dir`。原选定step3000权重SHA为 `c25ffa08fd2b67343f37eab06bb948642e2a6d2578e387a563ba81105c506c2c`；普通Git不包含206.7 MB模型文件。

重做历史长训则需另一个初始化sim300（权重SHA `e82c8794497ab288c6c58fd16008b78770eddb39bc5f5faab3143b1a530c6c76`）与25训练/4验证回合。58个双路RGB数组约20.06 GiB，不在Git里。若从公开模型、公开episode0重新生成教师并适配，应记为新的实验分支，不承诺复现相同权重或同一成功率。具体生成链、边界与命令已写入[训练复现步骤](训练复现步骤.md)。

## 来源

ACT与LeRobot、公开数据/模型、Seeed资产的作者归属见仓库根[作者说明](../../../AUTHORS.md)。相机外观拟合参数属于本仿真，不是实机标定。官方mesh不在本仓库重新分发，按固定下载清单保留上游来源和各自许可。
