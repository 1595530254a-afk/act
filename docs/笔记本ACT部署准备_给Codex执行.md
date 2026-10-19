# 笔记本 ACT 部署准备：给本机 Codex 执行

日期：2026-10-19。请把整份文档交给笔记本上的 Codex；它应读取实际配置并完成下面的离线准备，最后给出本机验证报告。

## 用户需要做什么

1. 笔记本接电源、联网，插上准备存放缓存与数据的 U 盘，打开本机 Codex，让它访问项目文件夹与该 U 盘。
2. 把本文发给它，并说：“请按这份文档完成本机 ACT 离线部署准备，实际执行和验证，不训练、不驱动机械臂，完成后给我验收报告。”
3. 完成后把 `local/laptop_readiness/笔记本准备结果.md` 和本次推理 JSON 发回台式机这边。

第一轮不需要连接机械臂或相机，使用公开数据中的真实图像即可。主要耗时通常在下载环境和数据；笔记本上的驱动与网络情况需现场确认。

## 给笔记本 Codex 的任务与背景

用户希望“台式机训练，笔记本推理并最终接机械臂”。本次只完成**硬件识别 → 独立环境 → 公开模型离线推理 → 本机报告**。暂不执行训练、反向传播、参数更新或机械臂运动；也不执行发布者的 `train_config.json`。

- 笔记本：用户记得是约 2021 年购买的联想拯救者 R9000P 或 Y9000P，Windows，疑似 NVIDIA 30 系显卡。型号、GPU 和显存均未确认，必须在本机读取，不能按购买价格推断。
- 机械臂：矽递 reBot Arm B601-RS（RobStride 电机）。上位机可以让关节与夹爪运动；Python SDK、LeRobot 和本机标定尚未验证。
- 相机：用户已有 Intel RealSense D405 与 D435i，已装腕部相机；实际安装视角待核对。
- 台式机：RTX 5060 Ti 8GB，已通过公开 ACT 权重的离线推理，**尚未训练**。
- 代码仓库：[1595530254a-afk/act](https://github.com/1595530254a-afk/act)，当前共享分支 `master`。

允许在项目目录内安装运行环境、下载公开资源、执行离线验证并写报告。优先复用现有脚本，不必重新设计部署系统。已有项目先检查工作区和分支，保留本地修改；本次结果放在 `local/laptop_readiness/`，无需先配置 GitHub 推送权限。

## 1. 读取本机硬件和工具

先记录 Windows 版本与架构、电脑型号、CPU、内存、GPU 精确型号、显存、驱动、磁盘余量，以及 Git、Python、PowerShell 是否可用。可以使用：

```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free
Get-Command git, py, python, pwsh -ErrorAction SilentlyContinue | Select-Object Name, Source
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

若 `nvidia-smi` 不存在或失败，另查 `Win32_VideoController` 与设备管理器，记录真实原因，不能直接判断“没有独显”。不要用 WMI 的 `AdapterRAM` 数值替代 NVIDIA 工具读取的大容量显存。

**本次采用笔记本“内置盘放运行核心，U 盘放缓存与数据”的分工。**此前整套项目预留 20GB 的建议包含缓存和安装余量，不代表笔记本内置盘必须空出 20GB，更不要求 50GB。

2026-10-19 台式机按文件逻辑大小实测：

| 项目 | 约占空间 |
| --- | --- |
| ACT 虚拟环境 `.venv` | 3.26GiB |
| Python 与安装工具 | 0.09GiB |
| 安装及下载缓存 `cache` | 3.27GiB |
| 模型与配套文件 | 0.19GiB |
| 公开验证数据 | 0.81GiB |
| 上述合计 | 7.62GiB，约 8GB |

这是台式机相同版本的参考实测，不是笔记本已测结果或安装峰值；文件系统分配开销和后续版本也会影响空间。

### 笔记本的存储分工

| 放哪里 | 保存内容 | 当前版本参考 |
| --- | --- | --- |
| 笔记本内置盘的项目目录 | 代码、`.python`、`.venv`、`tools`、完整 `models/act_rebot_cube`、小型报告和本次运行日志 | 核心环境与模型约 3.5GiB |
| U 盘的独立目录，例如 `E:\ACT-storage` | 安装/下载缓存、项目临时下载文件、公开验证数据、已完成的日志与其他归档副本 | 现有缓存与公开数据约 4.1GiB，另留安装余量 |

建议内置盘为项目留约 **6–8GiB**，U 盘留约 **10GiB 或更多**可用空间，这是按当前版本估计的准备余量，不是硬性最低配置。Windows、驱动和系统自身仍要有正常工作空间。Codex 应先读取实际余量，再判断是否够用，不自动清理用户其他文件。

模型约 200MB，放本地不会带来很大的空间压力。本项目 CUDA 推理启动时将模型加载到显存后计算，通常不在每个动作步骤重新读取权重；本地保存环境和模型便于启动与维护。[PyTorch 模型加载与设备说明](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

普通 U 盘可以先承担缓存、数据和归档角色，不要求把运行环境装到 U 盘。下载、安装和从公开视频读取测试帧可能变慢，离线验证时测量并记录即可。以后相机实时输入直接进入本机内存与显存，不需要从公开数据集持续读取图像；不要把每帧图像同步写到 U 盘作为控制的前置步骤。

### 给笔记本 Codex：安装前先落实路径分工

**共享仓库当前的脚本默认全部放在同一个项目目录，还没有自动完成上述拆分。**只在笔记本副本做以下必要路径调整，再运行后续安装命令；本轮不迁移或修改台式机环境。

1. 内置盘保留项目根目录；确认用户 U 盘的身份、实际盘符、文件系统和余量，不假定 E 盘存在，不格式化已有盘。为低频文件选定目录，如实际的 `E:\ACT-storage`。
2. 用本机配置 `ACT_BULK_ROOT` 指向该目录；在笔记本的 `act_env.ps1` 中把 `HF_HOME`、Hugging Face 缓存、`TORCH_HOME`、`XDG_CACHE_HOME`、`UV_CACHE_DIR`、`PIP_CACHE_DIR` 和项目 `TEMP`/`TMP` 指向它下面的 `cache` 或 `tmp`。Python 安装目录、`.venv`、`tools` 和最终模型仍在内置盘。不要把环境变量写成台式机路径。
3. 在笔记本的 `download_resources.py`、`offline_inference.py` 中将数据目录统一指向 `ACT_BULK_ROOT/data/rebot_cube`，模型目录继续使用本地项目的 `models/act_rebot_cube`。下载报告必须记录实际数据路径。`runs` 和 `reports` 先留本地，运行完成后可复制归档到 U 盘。没有设置 `ACT_BULK_ROOT` 时保留原有项目内路径，改动仅涉及存储位置，不改模型、预处理、数据内容或计时口径。
4. 将笔记本的存储设置保存到项目 `local/` 下的本机配置文件，并让 `act_env.ps1` 在每次设置路径前读取它，保证安装、下载与推理入口均使用同一配置。遇到 U 盘未连接时给出明确提示，不能悄悄改回内置盘继续下载。
5. 核对代码语法、目录和环境变量的实际值，再依次运行依赖检查、资源哈希验证和离线推理。报告列出本机调整了哪些文件；无需先推送这些笔记本本地配置。

先选定内置盘项目路径，再在笔记本建立环境。不要直接复制台式机 `.venv`；Python 环境含路径信息，移动目录后可能需要重建。[Python 虚拟环境说明](https://docs.python.org/3.12/library/venv.html)

最终报告补充本地项目路径、U 盘数据/缓存路径、两处安装后的剩余空间和读盘表现。外接盘增加的是磁盘空间，不增加内存、显存或算力。

此阶段沿用 Windows，不安排 Ubuntu、WSL、ROS、Docker、独立 CUDA Toolkit 或编译器安装。先运行现有预编译依赖；遇到具体缺项再处理。

## 2. 获取项目

项目目录放在笔记本内置盘，可选择 `D:\外置大脑\ACT`；没有 D 盘时也可使用用户目录下的 `ACT`。U 盘另选数据/缓存目录，例如 `E:\ACT-storage`。先确认实际磁盘与目标目录，避免覆盖同名文件。后续命令假设已经进入内置盘的项目根目录。

目标目录不存在且 Git 可用时：

```powershell
$ActProjectPath = 'D:\外置大脑\ACT' # 替换为笔记本内置盘上已确认的项目路径
git clone https://github.com/1595530254a-afk/act.git $ActProjectPath
Set-Location -LiteralPath $ActProjectPath
git status --short
git branch --show-current
git remote -v
```

如果目录已经存在，先检查它，不重复克隆，不强制覆盖或重置。若没有 Git，也可从仓库网页下载 ZIP，解压后进入含 `README.md`、`scripts` 和 `requirements.lock.txt` 的目录；报告说明采用 ZIP，不能假装完成了 Git 检查。公开仓库的下载不需要登录 GitHub。

先读 `README.md`，核对这些入口存在：

- `scripts/act_env.ps1`
- `scripts/setup_environment.ps1`
- `scripts/download_resources.py`
- `scripts/run_offline.ps1`
- `scripts/offline_inference.py`
- `requirements.lock.txt`
- `resources/resource_manifest.json`

仓库里的推理报告是**台式机参考记录**，不能作为笔记本已通过的证据。

## 3. 建立项目内环境

第一选择是复用台式机已验证的版本组合：Python 3.12.15、PyTorch 2.10.0+cu130、torchvision 0.25.0+cu130、LeRobot 0.6.1、PyAV 15.1.0。完整依赖以仓库 `requirements.lock.txt` 为准，不自动升级到最新版本。

项目安装脚本的 `BootstrapPython` 默认值为台式机的 `D:\Python\python.exe`，**笔记本不能假定存在这个路径**。按实际情况选下面一种启动方式。

### 已有能运行 pip 的 Python

先实际执行该 Python 的 `--version` 和 `-m pip --version`，然后把它的绝对路径传给脚本。例如下面的占位路径必须替换：

```powershell
.\scripts\setup_environment.ps1 -BootstrapPython '替换为本机实际的python.exe绝对路径'
```

这个 Python 只负责把 uv 装到项目内；运行 ACT 的 Python 和依赖仍由脚本独立建立，不向已有全局 Python 安装 PyTorch。

### 没有可用的 Python

可以先将固定版本 uv 安装到脚本预期的项目位置，再运行同一个安装脚本，无需先安装全局 Python。下面命令在项目根目录执行：

```powershell
. .\scripts\act_env.ps1
$env:UV_UNMANAGED_INSTALL = Join-Path $ActRoot 'tools\uv\bin'
Invoke-WebRequest -Uri 'https://astral.sh/uv/0.13.1/install.ps1' -OutFile (Join-Path $ActRoot 'tmp\install_uv_0.13.1.ps1')
```

Codex 先读取下载的官方安装脚本，确认版本和安装位置，再执行：

```powershell
& (Join-Path $ActRoot 'tmp\install_uv_0.13.1.ps1')
& (Join-Path $ActRoot 'tools\uv\bin\uv.exe') --version
.\scripts\setup_environment.ps1
```

确认 uv 为 0.13.1、安装脚本退出码为 0，且依赖检查通过。项目已经有此版本 uv 时直接复用。此启动方式依据 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)及[项目内安装选项](https://docs.astral.sh/uv/reference/installer/)。

若 PowerShell 执行策略阻止脚本，只在当前进程或单次子进程范围处理，不永久修改整机执行策略。脚本每次执行后检查退出码，失败时保留实际错误，不能继续报“安装成功”。

每个新的 PowerShell 会话，在下载或 Python 操作前执行：

```powershell
. .\scripts\act_env.ps1
```

仓库默认将这些文件放在项目内；笔记本按前述要求调整后，应确认 Python、虚拟环境和最终模型在内置盘，缓存与临时下载文件在 U 盘。不要把台式机的 `.venv` 原样复制过来作为笔记本环境。

### 驱动兼容性

`+cu130` 表示使用 CUDA 13.0 的 PyTorch 构建。NVIDIA 官方给出的 CUDA 13.x 驱动兼容基线为 R580 或更新；仍需确认该笔记本 GPU、Windows 和具体驱动的实际支持情况，并通过下面的张量运算验证。[NVIDIA CUDA 13.0 官方说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)

如果驱动太旧，记录型号、当前版本与所需官方驱动方案；不要把安装 CUDA Toolkit 当作驱动更新，也不要自动重启。可以先完成环境、资源下载和 CPU 离线检查，GPU 项标为待解决。若必须使用其他 PyTorch CUDA 构建，在笔记本单独记录版本和理由，不直接改写共享锁文件。

```powershell
& .\.venv\Scripts\python.exe -c "import torch; print('torch=', torch.__version__); print('runtime_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable'; print('gpu=', torch.cuda.get_device_name()); print('vram_bytes=', torch.cuda.get_device_properties(0).total_memory); x=torch.ones((32,32),device='cuda'); torch.testing.assert_close(x@x,torch.full((32,32),32.,device='cuda')); torch.cuda.synchronize(); print('CUDA tensor check passed')"
```

## 4. 下载模型及离线验证用数据

先确保本会话没有遗留的 Hugging Face 离线模式，再运行已有下载校验脚本：

```powershell
. .\scripts\act_env.ps1
$env:HF_HUB_OFFLINE = '0'
$env:HF_DATASETS_OFFLINE = '0'
& .\.venv\Scripts\python.exe .\scripts\download_resources.py
```

固定源已经写在 `resources/resource_manifest.json`：

| 资源 | 仓库 | 固定 revision |
| --- | --- | --- |
| ACT 模型 | `nikodembartnik/act_rebot_cube` | `409b639946cc4140b3934a6e3d2c255f4335ee9d` |
| 公开数据 | `nikodembartnik/rebot_cube` | `c5cdae89d2e1f474a8e2f1ac5f16302126d0c33f` |

脚本下载最终模型与配套预处理/后处理、归一化张量，以及完整数据集；逐文件校验上游哈希，成功后写 `reports/download_manifest.json`。当前固定版本应校验 8 个模型文件、11 个数据文件。

也可用 U 盘转移已有完整资源：模型放在笔记本本地项目的 `models/act_rebot_cube/`，数据放在其 U 盘配置的 `data/rebot_cube/`，再用已调整路径的下载脚本联网核验、补齐。模型必须连同配置及归一化文件一起转移，仅拷贝 `model.safetensors` 不够。

下载公开数据是为了复现离线测试，不是在笔记本训练；以后只接真机推理时不必携带完整训练数据。

如果网络失败，分别记录 GitHub、PyPI、PyTorch 下载站或 Hugging Face 的具体失败。检查笔记本本机已有代理，再按需设置当前进程的 `HTTP_PROXY` / `HTTPS_PROXY`；不要照抄台式机的代理端口，不关闭 TLS 校验，不索要或写入个人令牌。这些公开资源不要求先登录 Hugging Face。

## 5. 实际跑一次笔记本离线推理

先在 `local/laptop_readiness/desktop_reference/` 保存仓库自带的 `reports/offline_inference_cuda.json` 参考副本，避免与本次结果混淆。若该备份已存在，保留原件。

CUDA 检查通过后执行：

```powershell
.\scripts\run_offline.ps1 -Device cuda
```

如果 CUDA 暂未通过，可执行 CPU 诊断，报告清楚标注 CPU：

```powershell
.\scripts\run_offline.ps1 -Device cpu
```

现有脚本严格加载模型及配套处理器，使用 PyAV 读取真实双视角图像，抽取 5 帧，完成 3 次预热和 25 次计时，并检查动作形状和数值。它不连接机械臂。保留脚本的 `video_backend='pyav'`，不要因 TorchCodec/FFmpeg DLL 问题随意替换整套依赖。

验收必须来自本次实际运行：

- 进程退出码为 0，生成时间与本次运行对应，JSON 中 `status` 为 `passed`。
- 5 个输入样本均成功，单次动作输出为 `[1, 100, 7]`，没有 NaN 或 Inf。
- 报告设备与实测一致，CUDA 测试确实使用笔记本独显。
- 保存平均、P95、最大延迟和 PyTorch 峰值显存；CPU 测试如实记录无 CUDA 显存指标。

把本次生成的 `reports/offline_inference_cuda.json` 或 `reports/offline_inference_cpu.json` 复制到 `local/laptop_readiness/`，并写明本机测试时间。仓库中的同名报告会被现有脚本更新，属于预期行为；不要将其误提交为台式机新的测量。

台式机的参考结果是平均约 20.4ms、P95 约 22.4ms、活跃张量峰值约 271.8MiB。只作对照，不是笔记本验收阈值。计时包括预处理、模型前向、反归一化与动作回到 CPU，不含视频解码；显存不含驱动、桌面显示等全部占用。短基准也不代表持续运行或闭环抓放性能。

## 6. 真机相关内容先记录，留待下一阶段

离线准备通过后，记录已有上位机/SDK 的名称、版本和本机位置，以及 USB-CAN 转接器的实际型号和驱动。已接入设备可以只读枚举；设备未带到笔记本时写“未连接，待核对”，不影响本轮离线验收。

本轮不要求购买示教主臂、采集新数据或安装通用 CAN 驱动。上位机能动，不等于当前固定版 LeRobot 已接通；先识别实际转接器，再确定厂家 SDK 或 LeRobot 的接入路线。

相机联调留到设备在场时进行：确认 D405、D435i 能通过 USB 3 数据线同时提供稳定 RGB 图像，记录哪路是 wrist、哪路是 top。当前公开权重要求两路真实画面，各为 640×480；它目前不能直接删掉一路，或把同一路复制两份当作等效部署。深度图不是这份权重的必需输入。

以后接真机前还需要核对 7 维关节/夹爪顺序、单位、方向、零位与范围，以及部署端的限位和速度约束；这部分不属于“装好推理环境”，不能因离线通过就直接发送模型动作。

## 最终交付

在项目内写 `local/laptop_readiness/笔记本准备结果.md`，把本次推理 JSON 一起放在这个目录。报告使用下面的简短结构：

```text
测试时间：
项目绝对路径 / Git 提交或 ZIP 来源：
电脑型号 / Windows 版本与架构：
CPU / 内存：
GPU / 显存 / 驱动：
Python / PyTorch / CUDA runtime / LeRobot 版本：
依赖检查：通过 / 失败及具体原因
资源哈希：通过 / 失败及具体文件
CUDA 张量检查：通过 / 未通过及原因
离线推理：CUDA 或 CPU，是否通过，输出形状
延迟：平均 / P95 / 最大值；PyTorch 峰值显存：
机械臂 / 相机 / USB-CAN 信息：已知内容与待核对项
本次训练步数：0；机械臂动作命令数：0
仍需解决的具体事项：
结果 JSON 绝对路径：
```

结论区分“GPU 离线推理已通过”“仅 CPU 已通过”“环境仍有阻塞”；不要把它们统称为“真机部署成功”。完成本轮后等待台式机端核对结果，再继续真机接口与相机联调。
