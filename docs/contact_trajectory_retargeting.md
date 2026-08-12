# 几何轨迹修正与新 mesh Stage1 流程

该流程属于 **retarget 后处理阶段**，位于原始 `mano2dexhand.py` retarget 与 Stage1 之间。它只求解右手相对工具的平移，支持原有三维自由平移和语义中轴单自由度平移：

- 保持 `opt_wrist_rot` 完全不变；
- 保持全部 `opt_dof_pos` 完全不变；
- 将右腕、右手全部关节以及 Stage1 的 MANO 跟踪目标施加同一个刚体位移；
- 位移在工具局部坐标系中求解，所以工具运动和旋转时，手与工具的相对关系保持一致；
- 通过五次 smoothstep 在抓握区间前后渐入、渐出，避免轨迹速度突变。

正式输出采用三级串行门槛：

1. **语义几何优化**：红区接触、黑区避让、刚体平移和姿态不变量；
2. **精确几何验证**：高密度整手碰撞点、collision URDF 网格穿透和位移连续性；
3. **Isaac Gym/PhysX 验证**：在隔离进程中加载真实手/工具碰撞资产，对代表帧执行固定姿态接触回放，检查指定末端接触覆盖、接触包络、求解器力和有限性。

任一级失败都不会替换正式 PKL。PhysX 固定姿态回放验证碰撞响应是否合理；双手、动态工具、重力和任务控制共同作用下的抓握稳定性仍需由 Stage1 冒烟测试验证。

修正结果默认镜像保存到 `data/retargeting_contact`。例如：

```text
data/retargeting/OakInk-v2/mano2artimano_rh/<sequence>@0.pkl
    -> data/retargeting_contact/OakInk-v2/mano2artimano_rh/<sequence>@0.pkl
```

## 1. 资产准备

Stage1 实际加载的视觉 mesh 必须包含逐顶点语义颜色：红色表示允许接触区域，黑色表示禁止接触区域。对应的 `coacd_object_preview` URDF 及其引用的碰撞 mesh 也必须是新工具的物理资产。

推荐先把新资产安装到该 OakInk 对象本来的位置，再运行修正。也可以用 `--mesh` 和 `--collision-urdf` 指定候选资产进行诊断；但 Stage1 启动时会校验文件哈希，如果数据集实际加载的资产不一致会立即报错。

## 2. 生成修正 PKL

先完成普通 retarget，再运行：

```bash
python main/dataset/contact_trajectory_retarget.py correct \
  --data-idx 083f7@0 \
  --dexhand artimano \
  --side right \
  --translation-mode semantic-axis \
  --headless
```

工具会自动完成以下步骤：

1. 按 OakInk 数据集的相同帧范围和 `skip=2` 重建工具轨迹；
2. 读取 dex hand URDF 的碰撞表面并进行离线前向运动学；
3. 自动检测主要抓握区间；
4. 在工具局部坐标系中估计由邻近黑区指向红区的 mesh 语义中轴，并只优化沿该轴的非负平移距离，使指定接触 link 落入红区，同时要求包括未接触关节在内的整手避开黑区；
5. 用更密的手部碰撞表面和真实 collision URDF 检查穿透、黑区避让及轨迹连续性；
6. 在独立 Isaac Gym 进程中进行 PhysX 接触回放；
7. 写出报告、预览图，并仅在三级门槛全部通过时原子保存 PKL。

省略 `--headless` 时，第三级先完成代表帧 PhysX 验收，再创建单个 Isaac Gym 环境循环播放完整轨迹。Viewer 中只有一个工具 Actor 和一个修正后的绿色右手 Actor，不加载原始白手。手动关闭 Viewer 后才保存结果并退出命令。加入 `--headless` 时执行相同的 PhysX 验收但不开窗口。Isaac Gym 不参与第一级优化，因此不会把不稳定的仿真梯度引入平移求解。

Viewer 回放是纯运动学显示，不执行会改变 PKL 关节姿态的物理模拟。正式物理指标由 Viewer 打开前的独立 PhysX 代表帧环境产生。命令行不再提供 `--show-original-hand` 参数。

Viewer 依赖可用的 X11 图形会话。运行前可检查：

```bash
echo $DISPLAY
```

若为空，需要从图形桌面终端运行或配置 SSH X11 转发；程序会明确报告 `DISPLAY is empty`，不会静默退化成无窗口模式。

终端不再在完成时打印整份 JSON，而是在执行过程中输出 `[1/3]`、`[2/3]`、`[3/3]`、产物保存位置和最终状态。完整数值仍保存在 `contact_report.json`，供程序化复验和 Stage1 门槛检查使用。

`semantic-axis` 首先选取红区附近（默认 3 cm）的局部黑色表面点，对红区和局部黑区进行平衡 PCA，并使用“局部黑区中心 → 红区中心”确定轴的正方向。最终位移满足
`delta_tool = distance * direction_tool`，其中 `0 <= distance <= max_shift`。因此
`max_shift` 在该模式下是实际欧氏位移上限，而不是三个坐标分量各自的上限。

相关参数：

- `--axis-roi-distance 0.03`：选择红区附近局部黑点的半径；
- `--axis-grid-samples 201`：轴向距离粗搜索采样数；
- `--max-shift 0.05`：沿中轴的最大平移距离；
- `--translation-mode free3d`：需要复现实验时退回原三维自由搜索。

若已经知道稳定抓握区间，可显式指定：

```bash
python main/dataset/contact_trajectory_retarget.py correct \
  --data-idx 083f7@0 \
  --dexhand artimano \
  --side right \
  --translation-mode semantic-axis \
  --start-frame 180 \
  --end-frame 520
```

常用输出为：

```text
data/retargeting_contact/.../<sequence>@0.pkl
data/retargeting_contact/.../<sequence>@0/<sequence>@0.contact_report.json
data/retargeting_contact/.../<sequence>@0/<sequence>@0.contact_preview.png
data/retargeting_contact/.../<sequence>@0/<sequence>@0.verify_report.json
```

修正 PKL 继续保存在原路径。其余三个检查文件统一放在与 PKL 同级、以
`<sequence>@0` 命名的目录中；`verify_report.json` 在执行独立复验命令后生成。

不要把 `--allow-infeasible` 生成的诊断 PKL 用于 Stage1。正常模式下约束未通过时不会写出 PKL。
每次 `correct` 都从原始 `data/retargeting` PKL 重新计算，不再提前检查目标 PKL 是否存在；三级验证通过后直接原子覆盖已有修正 PKL。保留的 `--overwrite` 仅用于兼容旧命令，不再影响行为。

## 3. 修正结果验证

### 3.1 自动单元测试

```bash
python -m unittest tests/test_contact_trajectory_retarget.py -v
```

测试覆盖平滑位移曲线、黑区到红区的优化方向、Artimano URDF 碰撞链、腕姿/手指姿势不变量，以及相同位移是否同步进入 Stage1 的 MANO 目标。

### 3.2 报告和预览检查

`contact_report.json` 中至少确认：

- `passed: true`；
- `geometry_feasible: true`；
- `translation_mode: semantic-axis`；
- `axis_constraint.passed: true`，且 `perpendicular_norm` 接近零；
- `after_hold.red_frame_pass_ratio >= settings.min_red_frame_ratio`；
- `after_hold.black_violation_fraction <= settings.max_black_violation_fraction`；
- `invariants.passed: true`；
- `validation_tiers.semantic_geometry.passed: true`；
- `validation_tiers.precise_geometry.passed: true`；
- `validation_tiers.isaac_gym_physics.passed: true`；
- `rotation_max_abs_error` 和 `dof_max_abs_error` 为零。

精确几何层默认使用每 link 64 个碰撞点和 24 个时间采样帧，限制最大碰撞网格穿透为 6 mm、穿透点比例为 1%。PhysX 层把每个 shape 的 contact offset 显式设为 3 mm，避免 Isaac Gym 默认每侧 20 mm 的接触包络被误判为约 40 mm 穿透。

打开 `contact_preview.png`，逐项检查紫色箭头由黑区指向红区、绿色接触 link 位于红区，手掌、手指及无实际接触的关节没有落在黑区，渐入/渐出曲线连续。

### 3.3 独立复验

```bash
python main/dataset/contact_trajectory_retarget.py verify \
  --input-pkl data/retargeting_contact/OakInk-v2/mano2artimano_rh/<sequence>@0.pkl
```

该命令从磁盘重新加载源 PKL、新 mesh、碰撞 URDF 和修正 PKL，重新计算几何指标、不变量和碰撞资产 bundle 哈希。退出码为 `0` 且 `passed: true` 才可进入 Stage1。

### 3.4 Stage1/PhysX 冒烟检查

先用少量环境和一个 iteration 检查完整数据加载、碰撞资产和仿真是否正常：

```bash
python main/rl/train.py \
  task=ResDexHand dexhand=artimano side=BiH \
  headless=true num_envs=4 max_iterations=1 \
  dataIndices=[083f7@0] \
  experiment=083f7_contact_smoke
```

`correct` 已完成固定姿态 PhysX 接触验证；本冒烟测试进一步验证双手 PD 跟踪、动态工具、重力、摩擦和任务奖励共同作用下的完整抓握稳定性。碰撞资产哈希校验能防止加载错文件，但不能代替这一整任务动力学验证。

## 4. 使用修正轨迹训练 Stage1

冒烟检查通过后，执行正常 Stage1 命令：

```bash
python main/rl/train.py \
  task=ResDexHand dexhand=artimano side=BiH \
  headless=true num_envs=4096 learning_rate=2e-4 \
  test=false randomStateInit=true dataIndices=[083f7@0] \
  early_stop_epochs=10000 actionsMovingAverage=0.4 \
  experiment=083f7_contact_artimano
```

加载修正 PKL 时，数据集会自动把同一个工具局部平移同步到：

- `opt_wrist_pos`（PKL 中已修正）；
- Stage1 的 `wrist_pos`；
- Stage1 的全部 `mano_joints`；
- 相应线速度、关节速度、`tips_distance` 和 `joints_distance`。

因此 Stage1 的复位目标、观测目标和奖励目标不会互相冲突。左手仍从 `data/retargeting` 读取，姿态和轨迹不变。

左右手路径已经直接固定在对应 OakInk 数据集实现中：右手读取
`data/retargeting_contact`，左手读取 `data/retargeting`，训练命令不再需要传入路径参数。
如果右手对应的修正 PKL 不存在，Stage1 会直接报错并停止，不会退回未经修正的轨迹或零 DOF 初值继续训练。

该项目的完整执行流程到 Stage1 训练结束，不再包含后续残差修正阶段。
