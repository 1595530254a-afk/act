# 第一阶段ACT仿真复盘：历史证据提纲

归档日期：2026-10-24。该文件为Word复盘的事实底稿；所有结论来自当前保存记录，只读检查，没有重新运行ACT/物理/训练或Git写操作。

当前阶段完成的是固定仿真任务的本地位置验证；不是B601真机部署通过。基础数据和权重来自社区 nikodembartnik/rebot_cube 与 nikodembartnik/act_rebot_cube，而机械臂资产来自Seeed的固定开源版本。原始数据审计为50回合、20948帧，数据文件本身不是预训练完成的模型。

## 建议纳入Word的10条“问题—措施—结果”

### 1. 先区分运动学预览与接触物理闭环

日期：2026-10-23（first_pass.date）。

**问题：** 最初网页能播放机械臂运动，但不能由动画判定策略抓放成立。

**采取措施：** 接入Seeed reBot_RS_Sim固定commit438f7713d83fb8cd0d0cb05e8cc335264dd3997a的44个模型资产；先检查坐标映射和输入导出，再新增力矩伺服与ACT闭环。

**观测结果：** first_pass记录MuJoCo3.17.0、21项测试通过；physics_control_tested、act_closed_loop_tested、grasp_success_verified均为false，真机命令0。初轮403帧示范状态回放和100×7单次预测仅是预览。

**因果边界：** 运动学qpos回放不能代替受力、接触或自主抓取；当前源码行只用于说明实现，不声称它与第一轮当时文件逐字相同。

证据：[first_pass](D:/外置大脑/ACT/reports/mujoco_first_pass_20261023.json)、[upstream](D:/外置大脑/ACT/simulation/upstream_manifest.json)。

实际代码：[simulation/preview.py:61](D:/外置大脑/ACT/simulation/preview.py:61)（set_pose，L61–71）；[simulation/task_dynamics.py:80](D:/外置大脑/ACT/simulation/task_dynamics.py:80)（ServoController，L80–138）。

### 2. 公开ACT失败与合成专家成功必须分开记

日期：2026-10-23（task与teacher工件批次）。

**问题：** 修正成红块入灰盒后，公开权重及本机真实数据续训1000步权重在重建仿真中仍未抓起；原始记录示范也未验证对准。

**采取措施：** 加入真实接触物理评分；先构造可成立的合成教师，用公开episode0的实测state作为新目标，明确记录夹爪松爪覆盖，而非把teacher动作交给学生冒充ACT。

**观测结果：** task_default100两模型各15s、5次ACT查询均失败；参考演示最大抬升0.258mm。后来的平顶40mm教师403帧/30Hz，抬升99.040mm并完整入盒。

**因果边界：** 专家拥有明确的轨迹目标，成功属于专家，不是学生ACT。真实数据到近似仿真的失败不能唯一归咎于权重；相机、单位、映射和动力学迁移仍有不确定性。

证据：[task_failures](D:/外置大脑/ACT/reports/mujoco_task_comparison_20261023.json)、[teacher_flat](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/synthetic_demo_flat40_v1/manifest.json)、[download](D:/外置大脑/ACT/reports/download_manifest.json)、[real_data_audit](D:/外置大脑/ACT/reports/dataset_audit.json)。

实际代码：[simulation/run_task.py:126](D:/外置大脑/ACT/simulation/run_task.py:126)（brick_metrics，L126–168）；[simulation/policy_worker.py:105](D:/外置大脑/ACT/simulation/policy_worker.py:105)（predict_chunk，L105–118）；[simulation/export_sim_demonstration.py:86](D:/外置大脑/ACT/simulation/export_sim_demonstration.py:86)（main，L86–296）。

### 3. 道具尺寸与顶面形状先冻结

日期：2026-10-23（CAD与fixed_flat40工件）。

**问题：** 早期重建的积木外形、尺寸和凸点与用户希望制作的道具不一致，任务条件反复变化会混淆算法结果。

**采取措施：** 按用户确认的项目尺寸固定平顶方块和开口方盒，去顶部凸点；导出STL与STEP并核对实体尺寸，后续仿真不再用尺寸对照挑成功。

**观测结果：** 红块40×40×20mm、实心无凸点；灰盒外120×120×50mm，壁和底各4mm，内腔112×112×46mm。STL按毫米导入，颜色由材料/显示决定；另有STEP实体。

**因果边界：** 这些是本项目约定，不是Seeed或数据作者实物测量规格；STL不含标准单位/颜色，STEP也不是原生SLDPRT特征树。

证据：[props](D:/外置大脑/ACT/cad/pick_place_props_20261023/尺寸说明.md)、[props_verification](D:/外置大脑/ACT/cad/pick_place_props_20261023/independent_verification.json)。

实际代码：[simulation/task_scene.py:50](D:/外置大脑/ACT/simulation/task_scene.py:50)（build_scene，L50–241）。

### 4. 固定单示范适配成功仍只是局部成立

日期：2026-10-23；sim300训练开始2026-10-23T17:04:55.668688+08:00。

**问题：** 单场景适配后，每次仅执行10步就重算会持续丢掉预测块后面的闭爪动作，仿真在抓取前相位停滞。

**采取措施：** 用物理成立的403帧合成示范继续学习300次更新；读取保存预测做相位诊断，在同checkpoint、同scene下恢复原定100动作执行长度。

**观测结果：** batch2、lr1e-5、FP32、92.930s、峰值allocated1465.059MiB。sim300 SHA e82c8794497ab288c6c58fd16008b78770eddb39bc5f5faab3143b1a530c6c76。固定点15s/5次真实ACT推理，抬升94.691mm并入盒。

**因果边界：** 这次执行长度修复有同模型同场景的直接后续证据；不能推广成所有ACT都必须执行100步。单示范的重叠窗口误差不是独立泛化验证。

证据：[sim300_train](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/adapt_flat40_v1/training_report.json)、[prefix_failure](D:/外置大脑/ACT/runs/mujoco_validation_20261023/fixed_flat40_20261023/policy_step300_v1/custom/analysis/failure_analysis.json)、[sim300_success](D:/外置大脑/ACT/runs/mujoco_validation_20261023/fixed_flat40_20261023/policy_step300_progress_fix/custom/result.json)。

实际代码：[simulation/adapt_act_sim.py:269](D:/外置大脑/ACT/simulation/adapt_act_sim.py:269)（train，L269–375）；[simulation/run_task.py:171](D:/外置大脑/ACT/simulation/run_task.py:171)（run_checkpoint，L171–318）。

### 5. 换位置暴露覆盖不足，而非证明模型完全不看图

日期：2026-10-23（冻结五点测试与三模型比较）。

**问题：** 用户质疑固定轨迹回放：固定起点成功是否只是记住路径。

**采取措施：** 先冻结模型/相机/场景/初始状态，再仅改变方块初始XY；保留原始双RGB、state、100×7预测和接触轨迹；另外将三阶段权重置于同一新循环比较。

**观测结果：** sim300原位1/1、五新位0/5，最高抬升2.087mm，五新点无双侧共同夹持。三模型同条件18试次/90次查询：公开模型0/1+0/5，desktop1000为0/1+0/5，sim300为1/1+0/5。

**因果边界：** 初始状态相同时预测随图像变化，不能说“完全没识别红块”；但反应不足以形成抓持。结果与局部拟合/覆盖不足相符，不能证明灾难性遗忘是唯一根因，更不能概括ACT算法全部泛化能力。

证据：[generalization_protocol](D:/外置大脑/ACT/runs/mujoco_validation_20261023/position_generalization_20261023/protocol.json)、[generalization](D:/外置大脑/ACT/runs/mujoco_validation_20261023/position_generalization_20261023/generalization_report.json)、[generalization_diagnostics](D:/外置大脑/ACT/runs/mujoco_validation_20261023/position_generalization_20261023/position_diagnostics.json)、[checkpoint_comparison](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/checkpoint_comparison_v3/comparison.json)。

实际代码：[simulation/policy_worker.py:105](D:/外置大脑/ACT/simulation/policy_worker.py:105)（predict_chunk，L105–118）；[simulation/evaluate_feedback_20261024.py:81](D:/外置大脑/ACT/simulation/evaluate_feedback_20261024.py:81)（audit_arrays，L81–118）。

### 6. 增加25位置和恢复示范后，低离线MAE仍不能代替任务验收

日期：2026-10-23 22:33至23:30（报告UTC转换为UTC+8）。

**问题：** 单轨迹覆盖太窄；多训练一些是否就会自动纠错，需用物理结果检验。

**采取措施：** 基于同一403帧路径，通过六关节Jacobian IK制成25网格位置变体；4位置独立验证、旧5诊断+新6最终保留。新增clean和恢复分支，原位准入后按完整抓放选模型。

**观测结果：** 训练10075帧/25回合，验证1612帧/4回合。teacher严格通过39/40（一个旧诊断点1.020mm穿透保留不通过）。v1分支step600验证2/4；从v1step600出发的v2分支step600为2/4但原位失败，同一v2分支step1200的MAE降至1.535°而验证0/4（step600是该1200步分支中的中间检查点，不能额外相加）。clean v3选step1800为0/4。22条恢复专家成功、合并47条训练回合/14695帧后，恢复step1800和2400均0/4且原位失败。保留v1step600最终新6为0/6，旧5为1/5。

**因果边界：** 25条是合成IK变体，不是25次独立人工操作；恢复成功属于专家。恢复聚合只受DAgger启发，不是完整逐步交互DAgger。各分支不能相加成同一条连续训练；未评估checkpoint结果未知。

证据：[teacher_positions](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/teacher_v2/protocol.json)、[teacher_positions_physics](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/teacher_v2/manifest.json)、[teacher_trace_audit](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/teacher_v2/independent_trace_verification.json)、[multisite_rgb](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/rgb_v2/manifest.json)、[adapt_v1](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/adapt_v1/training_report.json)、[adapt_v2](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/adapt_v2/training_report.json)、[adapt_v3](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/adapt_v3/training_report.json)、[adapt_recovery_v1](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/adapt_recovery_v1/training_report.json)、[recovery_experts](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/recovery200_train25/manifest.json)、[teacher_old_diag_penetration](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/teacher_v2/test_xy_plus30_minus40/manifest.json)、[old_selection](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/final_selection.json)、[old_final6](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/final_heldout_v1_step600/evaluation.json)、[old_diagnostic5](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/previous_diagnostic_v1_step600/evaluation.json)。

实际代码：[simulation/generate_multisite_teacher.py:57](D:/外置大脑/ACT/simulation/generate_multisite_teacher.py:57)（positions，L57–71）；[simulation/generate_multisite_teacher.py:94](D:/外置大脑/ACT/simulation/generate_multisite_teacher.py:94)（shifted_targets，L94–133）；[simulation/adapt_act_multisite.py:209](D:/外置大脑/ACT/simulation/adapt_act_multisite.py:209)（train，L209–339）。

### 7. 改为明确的5000步预算与每帧新视觉预测

日期：2026-10-24；训练开始2026-10-24T10:27:01.937775+08:00。

**问题：** 之前短分支和低频重规划不能回答足量训练及连续视觉反馈的效果；动作队列出队频率也不能冒称网络forward频率。

**采取措施：** 从sim300新开同一条5000更新分支，保留25/4数据分割；实际batch8、lr1e-5、FP32/AdamW，保留归一化。独立CUDA worker每请求真正predict_action_chunk；比较chunk100和ensemble1，后者每30Hz动作帧读取双RGB+state并做原版时间集成。

**观测结果：** 5000步全部batch8，40000个观察窗口，OOM0；训练循环含验证/保存1201.800s，峰值allocated3766.476MiB/reserved4118MiB。chunk100每15s真实5次forward，ensemble1真实450次。CUDA预检forward均19.153ms、往返均48.200ms，不含渲染和物理。

**因果边界：** 来源是sim300，不是前轮v1step600；选中step3000的累计仿真适配为3300。30Hz是仿真时间，不是墙钟实时。5000预算、batch、lr、CUDA和执行模式的组合没有全因子消融，不能单独给某项归功。

证据：[new_plan](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/position_plan.json)、[train_preflight](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/long5000_preflight_v1/preflight.json)、[train5000](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/long5000_v1/training_report.json)、[cuda_preflight](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/cuda_feedback_preflight_v1/preflight.json)。

实际代码：[simulation/adapt_act_multisite_20261024.py:222](D:/外置大脑/ACT/simulation/adapt_act_multisite_20261024.py:222)（train，L222–362）；[simulation/adapt_act_sim.py:232](D:/外置大脑/ACT/simulation/adapt_act_sim.py:232)（compute_gradients，L232–253）；[simulation/policy_worker_cuda_20261024.py:18](D:/外置大脑/ACT/simulation/policy_worker_cuda_20261024.py:18)（load_policy，L18–42）；[simulation/evaluate_feedback_20261024.py:121](D:/外置大脑/ACT/simulation/evaluate_feedback_20261024.py:121)（run_feedback，L121–237）；[simulation/evaluate_act_ensemble.py:29](D:/外置大脑/ACT/simulation/evaluate_act_ensemble.py:29)（SparseTemporalEnsembler，L29–64）。

### 8. 将策略越界与程序异常分开，保留全部未完成试次

日期：2026-10-24（validation_matrix_v1）。

**问题：** step2000的夹爪原始预测轻微小于0，旧严格guard使矩阵停下；片段中已入盒也不能算450帧完整通过。

**采取措施：** 保留原runner、冻结协议和原中断目录；新v2 runner仅对raw chunk独立重建首次未施加目标、逐条确认此前目标合法、用原validate_target精确复现同一异常的候选分类policy_limit_abort。不clip、不放宽边界；未知错误仍停。

**观测结果：** planned60=完整39+越界partial5+因对应baseline中断未尝试16，真实ACT查询11665。5次均为joint7负下限：step2000 chunk原位frame352；step3000 chunk第4val frame360；step4000 chunk原位352；step5000 chunk原位355；step5000 ensemble原位328。partial指标不计成功、未跑组计null。

**因果边界：** 这是模型/执行组合的越界行为，不伪装成完整任务失败，也不自动重试。界限近0的微小预测偏差暴露部署边界问题，但本轮保持原硬边界，未验证真实夹爪保护。

证据：[matrix_protocol](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/validation_matrix_v1/matrix_protocol.json)、[matrix_original_stop](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/validation_matrix_v1/running_summary.json)、[matrix_amendment](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/validation_matrix_v1/recovery_amendment_v2.json)、[matrix_completion](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/validation_matrix_v1/validation_completion.json)、[matrix_summary](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/validation_matrix_v1/running_summary_v2.json)。

实际代码：[simulation/task_dynamics.py:35](D:/外置大脑/ACT/simulation/task_dynamics.py:35)（validate_target，L35–44）；[simulation/run_long5000_validation_20261024_v2.py:31](D:/外置大脑/ACT/simulation/run_long5000_validation_20261024_v2.py:31)（prove_limit_rejection，L31–89）；[simulation/run_long5000_validation_20261024_v2.py:162](D:/外置大脑/ACT/simulation/run_long5000_validation_20261024_v2.py:162)（main，L162–290）。

### 9. 冻结模型与模式，再做8点以及用户追加1点

日期：2026-10-24；选择冻结2026-10-24T11:13:21.458082+08:00。

**问题：** 需要证明位置改动后由ACT重新预测，并避免从测试结果反向挑模型或只展示成功视频。

**采取措施：** 按同模式原位准入→验证完整成功数→稳态抓持数→离线MAE→chunk100→较早step排序；在测试前冻结step3000+ensemble1，最终8点各一次。用户随后另要求一点，先登记(+10,0)mm再单次测试；两组统计分开。另对保存输入做全量fresh网络复算及动作/接触/评分独立审计。

**观测结果：** step3000与step2000 ensemble都val4/4，按较低归一化MAE .0354018胜出；原位1/1。新8点8/8、各450frame，抬升96.78–105.94mm、最长连续高举双指68–73frame，3600份chunk/2520000值fresh复算maxdiff0。追加点1/1、抬升101.374mm、双指97frame，450份chunk/315000值复算maxdiff0；距旧8最近25.495mm且不重合48已见本地偏移。

**因果边界：** 只证明固定本体、物体、盒子、相机、光照下的本地XY插值留出。8点是提前保留测试，追加点是在看过8点后由用户提出，不能混为预注册9点benchmark。视觉响应不是语义识别证明；D405真实标定、sim2real和任意场景泛化未验证。

证据：[proposal](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/validation_matrix_v1/selection_proposal.json)、[final_selection](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/final_selection.json)、[final_launch](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/final_test_launch.json)、[final8](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/final8_selected_v1/evaluation.json)、[final8_fresh](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/final8_selected_v1/fresh_cuda_output_reproduction.json)、[initial_visual](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/final8_selected_v1/initial_visual_response_analysis.json)、[final_completion](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/completion_check.json)、[extra_registration](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/additional_point_p10_0_v1/test_registration.json)、[extra_evaluation](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/additional_point_p10_0_v1/extra_p10_0_rollout/evaluation.json)、[extra_fresh](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/additional_point_p10_0_v1/extra_p10_0_rollout/fresh_cuda_output_reproduction.json)、[extra_summary](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/additional_point_p10_0_v1/test_summary.json)、[novelty](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/singlepoint_novelty_audit.json)。

实际代码：[simulation/evaluate_feedback_20261024.py:121](D:/外置大脑/ACT/simulation/evaluate_feedback_20261024.py:121)（run_feedback，L121–237）；[simulation/evaluate_feedback_20261024.py:81](D:/外置大脑/ACT/simulation/evaluate_feedback_20261024.py:81)（audit_arrays，L81–118）；[simulation/evaluate_feedback_20261024.py:240](D:/外置大脑/ACT/simulation/evaluate_feedback_20261024.py:240)（audit_rollout，L240–284）；[simulation/verify_feedback_outputs_20261024.py:27](D:/外置大脑/ACT/simulation/verify_feedback_outputs_20261024.py:27)（verify，L27–130）。

### 10. 用ALOHA配套公开任务单独校验标准链路

日期：2026-10-24 10:37（protocol/evaluation UTC+8）。

**问题：** B601近似场景的失败不能单独判断通用ACT实现是否正常，需要模型与环境配套的独立链路检查。

**采取措施：** 加载LeRobot公开ALOHA transfer cube human ACT配套gym-aloha环境，冻结五个seed，CPU推理，14维状态和单顶部RGB，使用官方环境成功信号，保留失败及全部输入/预测。

**观测结果：** gym_aloha/AlohaTransferCube-v0、gym-aloha0.1.5、MuJoCo3.2.7；5回合4成功1失败(seed20261025)，共17次真实CPU forward。控制50Hz、每100动作一次网络查询；无训练、真机和动作裁剪。

**因果边界：** 这是独立5-seed烟测，不能复现发布者500回合83%指标，也不是B601结果；14通道/单相机/50Hz不能直接替换B601七通道/双相机/30Hz权重。

证据：[aloha_protocol](D:/外置大脑/ACT/runs/aloha_baseline_20261024/smoke_5seeds/protocol.json)、[aloha](D:/外置大脑/ACT/runs/aloha_baseline_20261024/smoke_5seeds/evaluation.json)、[aloha_audit](D:/外置大脑/ACT/runs/aloha_baseline_20261024/smoke_5seeds/artifact_audit.json)。

实际代码：[simulation/aloha_baseline_20261024.py:194](D:/外置大脑/ACT/simulation/aloha_baseline_20261024.py:194)（smoke，L194–417）。

## 保留失败结果，避免只讲最终成功

| 分支/检查点 | 实际四点验证 | 解释 |
|---|---:|---|
| validation_sim300 | 0/4 | 20次ACT查询；保存原结果 |
| validation_v1_step600 | 2/4 | 20次ACT查询；保存原结果 |
| validation_v2_step600 | 2/4 | 20次ACT查询；保存原结果 |
| validation_v2_step1200 | 0/4 | 20次ACT查询；保存原结果 |
| validation_v3_step1800 | 0/4 | 20次ACT查询；保存原结果 |
| validation_recovery_step1800 | 0/4 | 20次ACT查询；保存原结果 |
| validation_recovery_step2400 | 0/4 | 20次ACT查询；保存原结果 |
| validation_v1_step600_ensemble10 | 1/4 | 180次ACT查询；保存原结果 |

上述分支不能相加为同一次连续训练。10/24新5000步分支从sim300重新出发；最后使用其step3000，而不是把五个checkpoint全部训练步数累加。

## 当前最终8点（与追加1点分开）

| ID | 初始XY偏移mm | 完整成功 | 最大抬升mm | 最长连续高举双指帧 |
|---|---|---:|---:|---:|
| newtest_00 | (-30,-30) | 1 | 99.623133 | 73 |
| newtest_01 | (-30,+30) | 1 | 104.708925 | 68 |
| newtest_02 | (+30,-30) | 1 | 96.782185 | 72 |
| newtest_03 | (+30,+30) | 1 | 102.871590 | 68 |
| newtest_04 | (-35,+5) | 1 | 105.937317 | 69 |
| newtest_05 | (+35,-5) | 1 | 97.539866 | 71 |
| newtest_06 | (-5,-35) | 1 | 97.176510 | 71 |
| newtest_07 | (+5,+35) | 1 | 102.295691 | 68 |

追加点(+10,0)mm是用户在8点完成后另行提出的单次插值检查，1/1成功，不能事后合并成预注册9点测试。

## 核心证据文件的SHA256

全部来源的完整SHA256、路径、JSON指针及代码行号保存在同目录history_evidence.json。这里列核心归档入口：

- [first_pass](D:/外置大脑/ACT/reports/mujoco_first_pass_20261023.json)：`596a86227759a8823b4d0f156d07869b1a26401acaa978823ba47ce18549f733`

- [task_failures](D:/外置大脑/ACT/reports/mujoco_task_comparison_20261023.json)：`c516457804159104ea6ca57d5b9a928ee264a11718f372882003bb00f1d56c7e`

- [props](D:/外置大脑/ACT/cad/pick_place_props_20261023/尺寸说明.md)：`f2c8055d20cedf4a7426159086f0c4ceed3fb8a50c03cc52d38d51402ce51195`

- [teacher_flat](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/synthetic_demo_flat40_v1/manifest.json)：`2140dc85a83fa1890e479f2225578be8bf27c4599f2f22d4b8e595f9f59d736f`

- [sim300_train](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/adapt_flat40_v1/training_report.json)：`4264c62618569f16f2d70cd93e0981c11160a4c9e43e0e77d852b0e0b7293d3e`

- [sim300_success](D:/外置大脑/ACT/runs/mujoco_validation_20261023/fixed_flat40_20261023/policy_step300_progress_fix/custom/result.json)：`d2b96ccf01acdf39e89b3808fe1fb412cbfc9ed0cf5c4187b4d30fd46fcb83c2`

- [generalization](D:/外置大脑/ACT/runs/mujoco_validation_20261023/position_generalization_20261023/generalization_report.json)：`dac13858ff878d37cd4165acdeceebfa95e3704c7f9610daafa83e61499cefae`

- [checkpoint_comparison](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/checkpoint_comparison_v3/comparison.json)：`b26f3fb402827de5ab9dd710397a04797bab77fa042888cc894c64f140a9f988`

- [teacher_positions](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/teacher_v2/protocol.json)：`d1ef6b5da7e308991bf39b215bdd67b6014b605d33d5190387b29139f64eb444`

- [old_final6](D:/外置大脑/ACT/runs/mujoco_iteration_20261023/multisite_v1/final_heldout_v1_step600/evaluation.json)：`5579a870aa1155dd4ad186793f0f6864128abcfdc2fda10666eb015c45bb4798`

- [train5000](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/long5000_v1/training_report.json)：`9824d723727e42e6bf85587f80440cf1e4ab4b06efa5d00a47790a5a5ce31fb0`

- [matrix_completion](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/validation_matrix_v1/validation_completion.json)：`16a884f123d294fbac813e9b7a4ee973cbeb80c3e9163ac70d2410a065a16651`

- [matrix_amendment](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/validation_matrix_v1/recovery_amendment_v2.json)：`51f118a39f7c1a3bbfb8bdc14e26eedf30647a7ce7297cc5b9453a06e577e4ad`

- [final_selection](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/final_selection.json)：`e0a4f7db7498c53534558f2099513554d99ccc3bd3266087b4e5fcfd30bd3343`

- [final8](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/final8_selected_v1/evaluation.json)：`6ef257e217f20b51412dd82fbb80a590103cf97b08c47694b6656dc271ba8c3b`

- [final8_fresh](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/final8_selected_v1/fresh_cuda_output_reproduction.json)：`93e9fd3977a72dd8e2200c9f459ef719fe432b7976c836c2ef30fce56fb945c2`

- [extra_summary](D:/外置大脑/ACT/runs/mujoco_iteration_20261024/additional_point_p10_0_v1/test_summary.json)：`e48ef5e100c48196a0cff303f443ab6a353c69f3e02a78a6d0ff8fa3fc28b333`

- [aloha](D:/外置大脑/ACT/runs/aloha_baseline_20261024/smoke_5seeds/evaluation.json)：`8ccf1853aede0cd7707bf53e10db4f0280d17a8cf5074ce154ce1189bf3e5e18`

## 仍未知/未验证

- 真实B601七轴物理零位、关节符号/单位及夹爪开度映射独立标定文件，本轮仿真证据不能替代。
- 真实D405手眼外参/安装刚性、相机曝光和真实观察域，未在本轮完成验证。
- 5000步、batch8、lr1e-5、CUDA和ensemble1各自因果贡献未被全因子消融分离。
- 未接受物体/盒位置外推、尺寸颜色变化、相机变化、光照变化或实机安全成功验收。
- 用户原始遥操作现场的全部具体问题不由本仿真归档重新判定；不要把台式机仿真成功写成真机已完成。
- 没有独立实物测量证据支持把项目40mm道具尺寸说成原数据作者统一标准。

代码行号与SHA为本次归档时版本。早期报告记录的当时源码hash若与当前文件不同，保留历史记录，不据当前文件冒称逐字复现旧版。所有本次读取来源在归档前再次检查SHA未变。
