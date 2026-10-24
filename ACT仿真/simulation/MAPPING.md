# ACT 到 RS MuJoCo：候选坐标转换

本模块用于仿真坐标转换、姿态回放和动作诊断，没有硬件、ROS 或 CAN 入口。**当前实物映射状态仍是 `CANDIDATE_UNVERIFIED`；仿真中的抓取评分独立进行，不代表允许按此映射下发真机。**

模型七维顺序来自本地数据集：`shoulder_pan.pos`、`shoulder_lift.pos`、`elbow_flex.pos`、`wrist_flex.pos`、`wrist_yaw.pos`、`wrist_roll.pos`、`gripper.pos`。数据集声明 `robot_type=rebot_b601_follower`；作者[模型/数据集合](https://huggingface.co/collections/nikodembartnik/seeed-studio-rebot-arm-b601-rs)将其归为 B601-RS。实际采集 CLI、分支和零位未公开，不能直接将另一个外部插件的坐标约定套在数据上。

当前默认采用**原生 LeRobot canonical 公开关节坐标 → RS 原生电机坐标 → 官方 RS 仿真坐标**的候选链。它有公开源码依据，但不是对作者实际采集配置的确认。

## 两个独立函数

输入支持 NumPy 数组或数值列表，形状为 `(..., 7)`；返回同形状的新 `float64` 数组。前六维写入 `joint1`～`joint6` 的旋转弧度，第七维是虚拟滑动关节 `joint7` 的米值。

|入口|前六维|第七维|
|---|---|---|
|`dataset_state_to_sim(state)`|canonical 公开度数乘 RS profile 的 −1，再转弧度|`radians(-state[6]) × 0.045 / 5` 米|
|`model_action_to_sim(action)`|与状态相同：公开度数乘 −1，再转弧度|`radians(-action[6]) × 0.045 / 5` 米|

**新版默认 state/action 同属公开坐标，转换相同，夹爪没有 ×6 或 ÷6。**原生 LeRobot 冻结版本中，RS 的七维 `joint_directions` 全为 −1：观测把原生电机度数除以方向得到公开度数；动作按相同公开度数表达，乘方向后送给原生电机。公开 J2/J3 区间为负值，而 RS 原生模型对应正值，所以负公开观测转换成正仿真角有源码依据，并非为了绕开限位而拟合符号。

动作函数特意返回 **PRECLIP 候选值**，没有重现公开限位或相对步幅裁剪。根程序应按实际加载的 XML 检查每轴越限、保存原值和统计；公开正 J2、J3 超范围或异常夹爪值仍然会产生可见越界，不裁成合法值。模型当前 J2/J3 为 `0..3.14 rad`；源码公开软限位与模型区间本身也不是完全相同的范围。

## 旧外部插件只保留显式诊断路径

之前拿到的外部插件1.0.5采用另一套约定：观测直接读取原生度数；动作乘 `[1,1,-1,-1,-1,1,6]`。它与原生 `rebot_b601_follower` 的 canonical 语义不同。初版诊断使用此候选得到了大量 J2/J3/夹爪越界，这说明需要查清接口来源，不能把那些结果直接判为 ACT 模型坏了或实物关节坏了。

两个旧假设保留为 `legacy_plugin105_state_to_sim()` 和 `legacy_plugin105_action_to_sim()`，用于复核旧结果，不是默认映射，不自动探测或混合两套约定。新版代码的 `DIRECTIONS=(-1,)*7` 仅代表 canonical 到 RS native 的源码方向；`LEGACY_PLUGIN105_ACTION_FACTORS` 是另一条诊断路径。两者不能混用。

## 官方桥接证据与夹爪显示

完成 canonical→native 转换后，官方控制栈把前六个硬件关节弧度直接发布；仿真桥按名称复制给 `target_arm`。这条 native→仿真的链没有再增加 J2/J3 翻号或零偏，也不证明数据集与当前实物的零位一致。

夹爪源约定是：电机0～5 rad → ROS 单指0～0.045 m，名义总开度90 mm。原显示桥又将单指45 mm放大到MuJoCo滑块50 mm，组合显示比例为 `0.01 m/rad`。当前 v3 接触仿真采用名义物理比例 **`0.009 m/rad`**，不应用显示放大；否则物理夹指会多张开约11.1%。旧插件诊断函数仍保留旧显示比例以复核历史结果。它不是实测夹爪毫米开度，也不是夹力标定。

在仅设置 `qpos` 后调用 `mj_forward` 的运动学回放中，应同时写：`joint_left = joint_right = joint7`；两指向相反方向移动由 XML 自身的轴和装配方向表达，不给右指再手动取负。仅写虚拟关节并期待 equality 在 `mj_forward` 中自动移动手指，会造成显示不一致。动力学步进则由所加载模型的约束求解，不能把运动学回放当作动力学抓放成功。

## 冻结来源

- 原生 LeRobot，冻结 `e595b7902714ba51f91e47523f66f89c5181b649`：[DM/RS 公开坐标与电机方向](https://github.com/huggingface/lerobot/blob/e595b7902714ba51f91e47523f66f89c5181b649/src/lerobot/robots/rebot_b601_follower/motor_family.py#L73-L167)、[公开状态/动作转换](https://github.com/huggingface/lerobot/blob/e595b7902714ba51f91e47523f66f89c5181b649/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py#L244-L316)。RS profile 的七维方向均为 −1，没有外部插件的夹爪 ×6；`homing_offset=0` 的默认记录不等于作者采集时的实物零位已知。
- 数据元信息的公开快照见 [dataset_info.json](../resources/dataset_info.json)。外部插件1.0.5的观测/动作与默认系数源码属于原本地诊断证据，未随本阶段公开包提供；文件SHA256及历史路径保留在 `joint_mapping.py` 的 `mapping_metadata()` 中。这些历史路径是来源记录，不是可点击下载入口，也不是运行默认映射所需的插件。
- 官方 Wiki 指向的作者控制栈，冻结 `aa807cf2affe85b13b86221159871aae7742c01b`：[硬件状态直接复制](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_RS/blob/aa807cf2affe85b13b86221159871aae7742c01b/rebotarm_ros2_RS/src/rebotarmcontroller/rebotarmcontroller/hardware_manager.py#L333-L362)、[ROS 关节与夹爪发布](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_RS/blob/aa807cf2affe85b13b86221159871aae7742c01b/rebotarm_ros2_RS/src/rebotarmcontroller/rebotarmcontroller/ros_publishers.py#L85-L151)。
- Seeed RS 仿真仓库，冻结 `438f7713d83fb8cd0d0cb05e8cc335264dd3997a`：[驱动约定](https://github.com/Seeed-Projects/reBot_RS_Sim/blob/438f7713d83fb8cd0d0cb05e8cc335264dd3997a/rebotarm_mujoco_rs/rebotarm_mujoco_rs/rs_driver.py#L19-L29)、[状态输入与夹爪缩放](https://github.com/Seeed-Projects/reBot_RS_Sim/blob/438f7713d83fb8cd0d0cb05e8cc335264dd3997a/rebotarm_mujoco_rs/rebotarm_mujoco_rs/mujoco_sync.py)、[RS MJCF 模型](https://github.com/Seeed-Projects/reBot_RS_Sim/blob/438f7713d83fb8cd0d0cb05e8cc335264dd3997a/rebotarm_mujoco_rs/models/rs_arm.xml)。

单测覆盖单位、状态/动作符号区别、夹爪比例、批量形状、非有限数值、无静默裁剪和输入不变性。它们验证转换实现，不验证采集配置或真机对应。
