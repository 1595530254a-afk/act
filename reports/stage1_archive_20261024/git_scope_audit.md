# 第一阶段 Git 发布范围审计

当前仓库为 `https://github.com/1595530254a-afk/act.git`，分支 **master**。远程HEAD读取成功，Git Credential Manager已缓存账号 `1595530254a-afk`；Git姓名与邮箱已配置，未请求或输出token。审计shell没有gh，但这不阻止用户授权的普通Git提交与推送，也不要求创建PR或切换分支。实际写权限由root的正常push验证。

本轮只新增审计与公开复现说明，没有stage、commit、push、安装、审批、模型前向或物理步进，没有修改现有源码、README或ignore。

## 建议提交范围

同名JSON的 `recommended_git_add_whitelist` 是可逐项选择的清单，不执行 `git add .`：

- 74个仿真源码、共享CPU预算脚本、公开资源元数据与锁定清单。57个仿真Python文件AST解析通过，内部模块导入闭合；外部依赖与三套环境实际版本已记录在 `docs/stage1_20261024/source_manifest.json`。
- 本轮相关的两份仿真报告、本审计、公开复现说明、源码manifest及10份公开复现目录文件。root另行制作的Word、阶段包说明或新脚本，应加入root自己的明确清单后核对。
- 可选11段最终8点/追加1点视频，合计29,271,305字节，每段不足3MiB；逐文件大小、SHA256及450帧/30fps/15秒的ffprobe结果已列入JSON。无需提交全部历史视频。

现有ignore排除了models/data/runs/local/delivery及各环境缓存，但没有排除output、视频和cad目录。工作树另有大量之前的笔记本/CAN/MIT/手眼标定改动，须按明确文件名选择，不能整目录夹带。

## 大文件与私密信息

选定step3000 `model.safetensors` 为 **206,707,932字节**，超过100MiB，不能普通Git提交。Git保存六文件hash，本地stage2包保存实际模型，目录为 `models/act_long5000_step3000`。模型/原始示范数据/环境/完整输出缓存不加入Git。

Git可见UTF-8文本的已知凭据格式扫描没有发现选定源码或新复现说明内的命中。旧review脚本、旧J4报告和CAD独立核验文件共8个文件包含微信或本机用户目录信息，文件/行号在JSON中列出，没有复制私有路径值；这些文件不属于本轮发布范围。模式扫描不能保证发现所有秘密，新增Word和其他未纳入清单的文件仍须由其制作方核对。

## 必要复现输入已补齐

`docs/stage1_20261024/runtime_inputs/` 现包含原source protocol、相机fit、成功参考场景XML、position plan、最终选择、上游下载清单的公开副本，以及精简结果和path mapping。原件未改，mapping逐项记录原SHA、公开副本SHA与应恢复的项目相对位置。

公开JSON用 `<ACT_ROOT>` 占位。`public_reproduction.md` 明确恢复步骤、重新填写protocol路径及重算相机/参考场景hash的方法；新机器恢复后的protocol/plan必须记录新hash，不能宣称仍与原件逐字节相同。实际模型通过stage2包目录和 `--model-dir` 指定。源码仍采用Windows `.venv/Scripts/python.exe` 布局，不把本轮归档冒称为跨平台一键安装器。

## 来源与许可边界

固定版本B601模型卡与数据卡均声明Apache-2.0。官方缓存的D405腕部支架归属文件明确CERN-OHL-W-2.0；本次没有完整复核整套Seeed资产的根LICENSE，不能据此宣称全部STL统一采用某许可证。建议Git发布自身集成源码、下载清单、来源和归属，不重新发布官方mesh二进制。

当前项目没有根LICENSE；源码阶段存档不自动为所有第三方内容重新授权。后续如要声明统一开源许可，由仓库所有者明确选择并保留已有归属。本次不擅自添加许可证。

源文件hash、最小输入原件/公开副本hash均已重新核对。更详细的导入图、环境版本、文件清单及边界见同名JSON和公开复现目录。
