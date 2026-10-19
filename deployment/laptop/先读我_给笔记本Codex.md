# ACT 笔记本部署包：给本机 Codex 的操作说明

版本日期：2026-10-19。本包用于 **Windows x64 笔记本的离线推理安装与验收**。模型和五帧测试输入已经包含在包内；首次安装 Python 与依赖仍需要联网。

用户可以直接把这个 ZIP 发给笔记本 Codex，并附一句：

> 请先读包内《先读我_给笔记本Codex.md》，按实际本地硬盘和 U 盘路径安装并完成离线验收，输出报告。不要训练或驱动机械臂。

## 1. 本轮任务与文件

执行顺序：**解压到内置盘 → 读取硬件与空间 → 校验文件 → 使用 U 盘缓存安装 → 离线推理 → 返回报告**。

- `models/act_rebot_cube/`：原始公开 ACT 模型、配置、处理器与归一化张量；没有用台式机的 100 步短训模型替换它。
- `fixtures/`：五帧真实双视角观察、对应状态与桌面参考输出；不需要下载完整 829MiB 数据集。
- `tools/uv/uv.exe`：Windows x64 的 uv 0.13.1，支持无全局 Python 启动安装。
- `requirements.lock.txt`：台式机已验证的完整 77 包版本锁定，含 PyTorch 2.10.0+cu130、LeRobot 0.6.1。
- `scripts/`：校验、安装、环境配置与离线检查入口；不包含机械臂控制程序。
- `bundle_manifest.json`：来源、版本及全部分发文件的 SHA-256。
- `NOTICE.md`、`licenses/`：来源与第三方许可文件。

这不是整个 ACT Git 仓库，无需先配置 Git、GitHub 登录或克隆仓库。不要运行模型目录中的发布者训练配置。

## 2. 确认笔记本配置，选择存放位置

用户记得是约 2021 年购买的联想拯救者 R9000P 或 Y9000P，Windows、疑似 NVIDIA 30 系显卡。请实际读取型号、Windows 版本/架构、CPU、内存、GPU 精确型号、显存、驱动，以及内置盘和 U 盘余量。不能按系列或购买价格推断。

可使用 `Get-CimInstance Win32_ComputerSystem`、`Win32_Processor`、`Win32_OperatingSystem`、`Get-PSDrive -PSProvider FileSystem`，并执行：

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

`nvidia-smi` 失败时检查设备管理器等实际情况，记录原因，不直接断言没有独显。记录笔记本接电与电源模式，避免把节能模式下的单次结果当成性能上限。

| 保存位置 | 内容 |
| --- | --- |
| 笔记本内置盘：解压后的包目录 | Python、`.venv`、模型、脚本、少量测试输入和当前报告；当前版本核心约 3.5GiB，建议预留 6–8GiB |
| 用户 U 盘：选定的缓存目录 | 安装/下载缓存与下载临时文件，建议预留 10GiB 或更多；已完成的日志/备份也可复制过去 |

这些是准备余量，不是固定安装峰值，更不要求内置盘空出 50GB。系统盘仍要保留 Windows 和驱动正常工作所需空间。

不要直接在 ZIP 预览窗口运行。把包解压到笔记本内置盘的固定目录，例如 `D:\ACT_laptop`；不能假定 D 盘存在，可改用其他内置盘。下面所有命令都在**实际解压目录中，能看到本说明、scripts、models 的那一层**执行。

U 盘路径示例为 `E:\ACT-cache`，必须替换成已确认的真实盘符。不要格式化 U 盘或清理用户其他文件。安装脚本会把大缓存和临时下载写到指定位置，本地环境与模型不会被移动到 U 盘。

## 3. 校验并安装

先执行全文件哈希校验，不需要 Python：

```powershell
.\scripts\verify_bundle.ps1
```

通过后，使用实际 U 盘缓存路径安装：

```powershell
.\scripts\setup_laptop.ps1 -BulkRoot 'E:\ACT-cache'
```

脚本安装项目内 Python 3.12.15，建立本地 `.venv`，按锁文件安装依赖并执行 `uv pip check`。模型与测试输入不再联网下载；Python 和依赖需要访问 GitHub 的 Python 分发、PyPI 和 PyTorch 官方下载站。

如果已有可用且版本完全匹配的项目环境，可先核对；不能复制台式机 `.venv`，也不能把未核对的全局 Python 当成包内环境。正常路线优先使用脚本建立独立环境。

如果 PowerShell 执行策略阻止脚本，仅调整本次进程或单次子进程的策略，不永久修改整机策略。每一步实际检查退出码，失败时记录具体错误，不继续宣称安装成功。

本包使用 `cu130`。CUDA 13.x 的驱动兼容基线为 R580 或更新，仍需根据实际笔记本型号与 Windows 核对驱动支持，并以 CUDA 张量检查和模型运行结果验收。[NVIDIA 官方说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)

若驱动旧或 CUDA 不可用，先记录具体型号和版本，提供官方更新方案，不自动重启。可以先完成 CPU 验证，但必须标明 GPU 尚未验收。不要把安装整套 CUDA Toolkit 当成更新显卡驱动；本轮不安排 WSL、Ubuntu、ROS 或 Docker。

网络失败时分辨具体站点和错误，按笔记本已有代理设置处理；不要照抄台式机端口，不关闭 TLS 校验。这些公开依赖不要求登录 Hugging Face。

## 4. 实际运行离线检查

```powershell
.\scripts\run_laptop_check.ps1 -Device cuda
```

如果 CUDA 暂时不通过，另跑 CPU 诊断：

```powershell
.\scripts\run_laptop_check.ps1 -Device cpu
```

运行时重设为本地小型缓存/临时目录，只读包内模型和测试输入，**不依赖安装用 U 盘是否在线**。没有训练、下载数据集或发送机械臂动作。

验收应同时满足：进程退出成功、本次 JSON 的 `status=passed`、五帧完整动作段为 `[5,100,7]` 且全部有限、参考输出检查通过、实际设备与报告一致。GPU 模式还会执行 CUDA 张量运算检查。

CPU、CUDA 分别与包内相应设备的参考输出比较，避免将不同计算后端的差异误判为文件或处理错误。float32 容差为 `rtol=1e-4, atol=0.005`，仅用于数值复现诊断，不是物理关节允许误差。异常时记录差异与依赖版本，不擅自放宽阈值掩盖问题。

脚本会生成 3 次预热、25 次计时，记录平均/P95/最大延迟与 PyTorch 显存峰值。计时不含测试文件解压和 uint8 图像转换，不代表相机采集、控制循环或抓放成功率；不能将桌面参考速度当成笔记本合格线。

## 5. 把结果发回台式机

在 `local/laptop_readiness/笔记本准备结果.md` 写明：

- 本次测试时间、实际解压路径、安装缓存 U 盘路径、两处剩余空间。
- 电脑型号、Windows/CPU/内存、GPU/显存/驱动、电源模式。
- Python、PyTorch、CUDA runtime、LeRobot 版本；依赖检查和包哈希检查结果。
- GPU/CPU 各自是否通过；参考输出最大差异；平均/P95/最大延迟与显存峰值。
- 实际问题、未完成项和下一步；训练步数 0、机械臂动作命令数 0。

把这份 Markdown 与本次生成的 `offline_inference_cuda.json` 发回。若只有 CPU 通过，发送 `offline_inference_cpu.json` 并明确 GPU 未通过。判断以本次退出码和时间为准，不引用其他电脑的旧报告。

后续才进行相机与机械臂接口联调。当前模型需要 wrist、top 两路 640×480 RGB 和 7 维状态；单位、零位、方向、夹爪范围与 CAN/SDK 兼容性尚未在本机确认。本包通过不等于真机部署完成。
