# ACT 第二阶段：按顺序执行，只观察并回传

本包为独立增量。不要覆盖旧部署目录、旧默认模型、驱动、标定、CAN配置或已验证的停机代码。模型是单分支5000步实验中按validation选出的step3000，SHA256为`c25ffa08fd2b67343f37eab06bb948642e2a6d2578e387a563ba81105c506c2c`；不能称为厂家原版或实机已验收模型。

本批完成“文件验收→真实双相机/真实状态快照→只预测shadow→回传”，**没有执行电机动作的入口**。它使用LeRobot ACT与原保存处理器，不依赖MuJoCo、仿真关节映射或仿真PD。

## 1. 接收与环境

单独解压到笔记本内置盘固定目录，和旧包并列。命令中的`<ACT_PYTHON>`替换为此前ACT检查通过的解释器，工作目录为本包根。不要复制台式机venv，不运行pip修复硬件venv，不重装驱动。已有依赖为Python3.12、PyTorch2.10、LeRobot0.6.1、NumPy、psutil；缺项原样回报，在独立推理环境处理。

```powershell
& '<ACT_PYTHON>' runtime/test_stage2.py
& '<ACT_PYTHON>' runtime/stage2_policy.py --device cpu
```

默认只跑三组包内仿真观察，不打开相机或CAN。依次核对保存前、接近抓取和运输阶段的网络输出；三帧来自源frame0/170/300，不假装连续30Hz。重叠时间集成由纯函数单测检查。

读取新`local/offline_*/return_report.json`：`status=passed_offline_reproduction`、`actual_neural_forwards=3`、模型SHA正确、`model_files_unchanged=true`、硬件命令0。失败时保留原错误；不更改阈值。脚本启动先校验包内载荷SHA。

GPU可用时再用`--device cuda`验收。`--device auto`会记录实际选择；无GPU时CPU仍可完成dry_run。CPU通过不能宣称30Hz实时，台式机速度也不能代替笔记本实测。

## 2. 沿用已经通过的硬件链，不从头重复

用户已确认笔记本的保持、回零、人工确认承重、失能实际成功。保留现有MotorBridge/RS驱动、PCANBasic、MIT控制进程及其日志和已验证回零路径。不要再次要求重复电机型号、保持/归零/失能测试；硬件、零位、负载或驱动改变时，仅记录受影响项。

正常退出仍走：**保持→既有已验证回零路径→现场人工确认到位和承重→失能**。本包退出只关闭推理或相机，不能接管或触发该链。不要因为推理报错、窗口关闭或finally而直接失能，更不能把“全零目标”当已验证的回零路径。独立紧急处置沿用现场既有方案。

本包不导入机器人SDK，不写电机参数、零位或使能状态。不要把仿真比例、正负号、PD增益或位置范围写入真机。

## 3. 双相机只读检查及真实状态合同

先用已有相机环境执行，不自动安装依赖：

```powershell
& '<CAMERA_PYTHON>' runtime/camera_snapshot.py
```

记录D405腕部与D435i顶部的实际序列号；不要按枚举顺序猜角色。缺`pyrealsense2`等依赖会返回`dependency_missing`与具体列表，不必破坏现有环境。

由笔记本AI在**现有已验证控制器**旁增加只读导出：持续更新JSON，含7维实际反馈、原有映射标识和本机`time.monotonic_ns()`接收时间。不得填零、拿目标角冒充实测角、复制旧状态或自行猜测方向。不得为了读数据另启动第二个CAN写控制器。字段见`snapshot_contract.json`。若映射尚缺一项，先回报具体缺项，继续相机工作；不自行代入仿真映射。

```powershell
& '<CAMERA_PYTHON>' runtime/camera_snapshot.py --capture-once `
  --wrist-serial '<D405_SERIAL>' --top-serial '<TOP_SERIAL>' `
  --state-json '<既有只读控制器持续写出的JSON>' --output-dir local/capture_001
```

此命令仅开相机，获取640×480 RGB两路，读取已存在状态JSON；状态大于250ms会拒绝保存有效快照。两设备时钟不能直接假定同步，记录各自设备时间域、帧号、接收时间和时间差。单张快照不代表双流持续30Hz通过。

## 4. 只观察shadow：模型输出只落盘

```powershell
Get-Content local/capture_001/shadow_request.jsonl | `
  & '<ACT_PYTHON>' runtime/stage2_policy.py --mode shadow --device auto
```

上面是单快照接线诊断，冷启动可能使快照变旧，报告会显示年龄；不能用它证明实时性。

持续shadow时，先启动驻留推理进程等待`ready_dry_run_only`，再由既有只读采集器通过stdin逐行送`snapshot_contract.json`的request。连续编号从0开始，每一帧必须新采两张RGB和实际状态，不能把同一观察重复450次。需要发送`{"stop":true}`结束只预测进程。该接口只接受文件与元信息，不接受电机动作或执行开关。

每次调用真正生成100×7 ACT输出，再将覆盖当前帧的预测按`exp(-0.01*i)`加权，i按最旧预测到最新预测排序；每帧查询一次称ensemble1。保存每段rawchunk、实际加权public action、来源行和权重、输入SHA、观测状态与时间戳，可独立复算。**没有单位/方向转接及电机发送函数。**

持续采集30秒左右，核对序列号、RGB格式/画面、反馈新鲜度、输入时间差，记录`actual_query_start_interval_ms`和端到端延迟。报告里的模型耗时、文件IO耗时和真实采样间隔分开看；超时只记为dry_run性能缺项，不能静默重复旧动作或声称30Hz控制通过。

## 5. 回传，停止在第二阶段边界

把最新`return_report.json`、`capture_report.json`、已有真实映射/状态读取说明及`RETURN_stage2.json`发回；大原始图像与录像可留笔记本，先发统计与少量真实样图。使用`return_template.json`填写，未做项明确写未做。

本批不切换真机默认模型，不执行抓放，不改既有停机链。完成后下一步由本机已有控制器接收经过真实映射和限位验证的候选动作；该接线和首次真实策略动作另行确认，不能因本包离线成功就自动发送。

依赖与日志都相对本包根，核心模型/当前日志放内置盘；旧包及旧模型原位保留，回退只需继续使用原入口。本包没有自动安装或覆盖步骤。
