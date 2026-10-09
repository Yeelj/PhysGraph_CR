# 几何轨迹修正与新 mesh Stage1 流程

该流程属于 **retarget 后处理阶段**，位于原始 `mano2dexhand.py` retarget 与 Stage1 之间。默认先提取源轨迹逐指接触意图，并在源红区中心与源接触中心之间按 `alpha=[1,0.75,0.5,0.25,0]` 插值，生成多个映射到目标安全优选带中心的初始候选。候选按可行性、全接触、保留数量、最差手指保持率、接触均衡性、帧覆盖、黑区违规和平移量排序；存在可行候选时直接采用最优项。候选失败后，后续优化恢复旧的红区中心映射，并完全移除优选带损失和验收约束，只保留接触保持、红区接触与黑区避让。优选方向不可靠时静默回退到目标红区中位数中心。`contact-se3` 模式先尝试纯平移，未达到全接触保留时再允许最多 5° 的腕部小旋转。原有三维自由平移和语义中轴单自由度平移仍作为兼容模式保留：

- `contact-translation` 保持 `opt_wrist_rot` 完全不变；`contact-se3` 仅允许受限腕部旋转；
- 保持全部 `opt_dof_pos` 完全不变；
- 将右腕、右手全部关节以及 Stage1 的 MANO 跟踪目标施加同一个逐帧刚体平移；
- 平移在工具局部坐标系中求解，所以工具运动和旋转时，目标抓握与接触区域的相对关系保持一致；
- 通过五次 smoothstep 在抓握区间前后渐入、渐出，避免轨迹速度突变。

正式输出采用一层必选几何门槛和一层可选物理门槛：

1. **语义几何优化**：固定 contact-link ratio、黑区避让、工具局部刚体平移和腕姿/手指 DOF 不变量；
2. **Isaac Gym/PhysX 验证（命令添加 `--physx` 时启用）**：在隔离进程中加载真实手/工具碰撞资产，对代表帧执行固定姿态接触回放，检查指定末端接触覆盖、接触包络、求解器力和有限性。

语义几何通过后始终原子保存正式 PKL。默认不运行 PhysX；添加 `--physx` 后，即使 PhysX 失败也保留已通过几何验收的 PKL 和失败报告，但其元数据仍把 PhysX 标记为必选失败，因此 Stage1 会拒绝读取。只有语义几何失败时才默认不保存，除非显式使用 `--allow-infeasible`。PhysX 固定姿态回放验证碰撞响应是否合理；双手、动态工具、重力和任务控制共同作用下的抓握稳定性仍需由 Stage1 冒烟测试验证。

运行任何优化之前，程序会分别检查旧 mesh 与目标新 mesh。两个 mesh 都必须具有
与顶点一一对应的颜色，且每个顶点必须能被严格分类为红色或黑色；缺少颜色、缺少
任一区域或存在未分类颜色都会立即报错退出。该预检结果写入
`semantic_precheck`，不会在失败后继续使用错误 mesh 优化。

红区验收按序列动态生成：先用源 mesh 和 `contact_distance` 识别原接触手指；优化和 refinement 优先要求全部原接触手指保留。全部保留仍无法通过时，最终验收按 `floor(min_contact_retention_ratio × 原接触手指数)` 放松，且至少保留一根；例如比例设为 0.8 时，4 根要求保留 3 根、5 根要求保留 4 根。每根手指的保持率按“源轨迹中该手指接触的帧里，修正后仍接触的比例”计算，不使用脱离源时序的绝对 80% 帧门槛。`min_red_link_ratio` 自动设置为“本序列所需保留手指数 / 配置 contact link 总数”，不再对所有序列使用同一固定比例；抓握区间内仍要求至少 80% 帧达到该动态门槛。

修正结果按 `data_id` 和 `new_obj` 分目录保存到 `data/retargeting_contact`。例如：

```text
data/retargeting/OakInk-v2/mano2artimano_rh/<sequence>@0.pkl
    -> data/retargeting_contact/OakInk-v2/mano2artimano_rh/<data_id>/<new_obj>/<sequence>@0.pkl
```

## 1. 资产准备

普通 retarget 始终继续从 OakInk 原有的 `object_preview` 和
`coacd_object_preview` 目录读取旧 mesh/URDF。新资产单独放在固定目录：

```text
/home/ps/data/ylj/PhysGraph/data/OakInk-v2/new_obj
```

新资产保持与 OakInk 原资产相同的用途分离：

```text
new_obj/object_preview/<编号>/<编号>.ply
new_obj/coacd_object_preview/<编号>/<编号>.ply
new_obj/coacd_object_preview/<编号>/<编号>.urdf
```

`object_preview` 中的表面mesh必须包含逐顶点红/黑语义颜色；
`coacd_object_preview` 中放凸分解后的碰撞mesh及其URDF。只有命令显式传入
`--new-obj <编号>` 时，几何修正和Stage1才读取这组新资产；不传参数时仍读取旧路径。

## 2. 生成修正 PKL

先完成普通 retarget，再运行：

```bash
python main/dataset/contact_trajectory_retarget.py correct \
  --data-idx 083f7@0 \
  --dexhand artimano \
  --side right \
  --new-obj spoon_01 \
  --translation-mode contact-translation
```

需要优先复用同一 `data_id` 下已验收的 mesh 修正残差时添加 `--bank`：

```bash
python main/dataset/contact_trajectory_retarget.py correct \
  --data-idx 083f7@0 \
  --dexhand artimano \
  --side right \
  --new-obj spoon_01 \
  --translation-mode contact-translation \
  --bank
```

工具会自动完成以下步骤：

1. 按 OakInk 数据集的相同帧范围和 `skip=2` 重建工具轨迹；
2. 读取 dex hand URDF 的碰撞表面并进行离线前向运动学；
3. 使用旧 mesh 自动检测主要抓握区间，并记录配置 contact link 在旧红区附近的接触摘要；
4. 从目标红区生成靠黑区一端的安全优选带，评估多个源中心插值候选；若存在直接可行候选则按多指接触均衡性选择，全部失败时将排名最高的两个候选分别送入无优选带的黑区 backoff；
5. 若命令包含 `--physx`，在独立 Isaac Gym 进程中进行 PhysX 接触回放；
6. 写出报告、预览图；语义几何通过即原子保存 PKL，PhysX 失败状态同时写入元数据并阻止 Stage1 直接使用。

需要 PhysX 验收时，在命令中添加 `--physx`。同时添加 `--headless` 会执行无窗口验收：

```bash
python main/dataset/contact_trajectory_retarget.py correct \
  --data-idx 083f7@0 \
  --dexhand artimano \
  --side right \
  --new-obj spoon_01 \
  --translation-mode contact-translation \
  --physx \
  --headless
```

添加 `--physx` 但省略 `--headless` 时，程序先完成代表帧 PhysX 验收，再创建单个 Isaac Gym 环境循环播放完整轨迹。Viewer 中只有一个工具 Actor 和一个修正后的绿色右手 Actor，不加载原始白手。手动关闭 Viewer 后才保存结果并退出命令。未添加 `--physx` 时不会启动 Isaac Gym，`--headless` 也不会产生额外作用。Isaac Gym 不参与第一级优化，因此不会把不稳定的仿真梯度引入平移求解。

Viewer 回放是纯运动学显示，不执行会改变 PKL 关节姿态的物理模拟。正式物理指标由 Viewer 打开前的独立 PhysX 代表帧环境产生。命令行不再提供 `--show-original-hand` 参数。

Viewer 依赖可用的 X11 图形会话。运行前可检查：

```bash
echo $DISPLAY
```

若为空，需要从图形桌面终端运行或配置 SSH X11 转发；程序会明确报告 `DISPLAY is empty`，不会静默退化成无窗口模式。

终端不再在完成时打印整份 JSON，而是在执行过程中输出 `[1/2]`、`[2/2]`、产物保存位置和最终状态。完整数值仍保存在 `contact_report.json`，供程序化复验和 Stage1 门槛检查使用。

默认 `contact-translation` 同时验收源接触中心、源红区中心及三种插值中心到目标优选带中心的候选。候选去重后按最差手指保持率和均衡性优先排序；全部直接候选失败后，只对排名最高的若干候选执行无优选带 backoff，仍失败才进入局部平移。`contact-se3` 仅在无优选带的回退平移仍未达到全接触保留时优化受限腕部旋转，手指 DOF 始终不变。即使旧红区在上、新红区在下，也会直接产生向下的初始平移，而不是从零位移寻找少量指尖接触解。

每轮 optimization/refinement 使用的帧数始终不超过 `max_optimization_frames`；即使完整抓握区间的全部帧都失败，也只对均匀采样的代表帧优化，最终验收仍在完整抓握区间上执行。几何目标中的红/黑最近邻只查询一次，避免此前 SE(3) 方案中重复 KD-tree 查询和失败帧无限扩张造成的长时间运行。

相关参数：

- `--center-candidate-alphas 1.0,0.75,0.5,0.25,0.0`：源接触中心到源红区中心的插值候选；
- `--center-candidate-dedup-distance 0.001`：平移差小于 1 mm 的候选合并；
- `--center-backoff-candidates 2`：直接候选全部失败后进入黑区 backoff 的候选数；
- `--max-residual-shift 0.01`：中心候选与 backoff 均失败后局部平移的每轴残差上限；
- `--max-optimization-frames 48`：初始优化与每轮 refinement 的硬帧数上限；
- `--contact-distance 0.006`：接触 link 的采样点到红区的最大距离；
- `--min-contact-retention-ratio 0.4`：当前默认的源接触手指数量系数；实际数量向下取整且至少为一，`min_red_link_ratio` 由每条序列自动计算；需要 4→3 时应设为 0.8；
- `--min-red-frame-ratio 0.8`：抓握区间内必须合格的帧比例；
- `--black-clearance 0.006`：整手到黑区的最小安全距离；
- `--max-black-violation-fraction 0.00001`：当前允许的黑区违规采样点比例，即 0.001%；
- `--front-band-min 0.15 --front-band-max 0.35`：红区沿黑到红语义轴的安全前端带；
- `--front-semantic-margin 0.002`：优选带到黑区的最小语义余量；
- `--translation-mode contact-se3 --max-rotation-degrees 5`：启用平移优先、必要时小旋转。

兼容模式 `semantic-axis` 首先选取红区附近（默认 3 cm）的局部黑色表面点，对红区和局部黑区进行平衡 PCA，并使用“局部黑区中心 → 红区中心”确定轴的正方向。最终位移满足
`delta_tool = distance * direction_tool`，其中 `0 <= distance <= max_shift`。因此
`max_shift` 在该模式下是实际欧氏位移上限，而不是三个坐标分量各自的上限。

兼容模式参数：

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
  --translation-mode contact-translation \
  --start-frame 180 \
  --end-frame 520
```

常用输出为：

```text
data/retargeting_contact/.../<data_id>/<new_obj>/<sequence>@0.pkl
data/retargeting_contact/.../<data_id>/<new_obj>/<sequence>@0/<sequence>@0.contact_report.json
data/retargeting_contact/.../<data_id>/<new_obj>/<sequence>@0/<sequence>@0.contact_preview.png
data/retargeting_contact/.../<data_id>/<new_obj>/<sequence>@0/<sequence>@0.verify_report.json
```

修正 PKL 继续保存在原路径。其余三个检查文件统一放在与 PKL 同级、以
`<sequence>@0` 命名的目录中；`verify_report.json` 在执行独立复验命令后生成。

不要把 `--allow-infeasible` 生成的诊断 PKL 用于 Stage1。正常模式下约束未通过时不会写出 PKL。
默认每次 `correct` 都从原始 `data/retargeting` PKL 重新计算。添加 `--bank` 后，程序从
`data/retargeting_bank/<sequence-id>` 检索最相似的已验收 tool mesh：先尝试迁移其
工具局部修正残差并直接执行完整语义几何验收；候选失败时再将最佳残差作为现有优化器
的初值。全部已配置验证通过后，修正 PKL 和对应 bank entry 都采用原子写入。bank 只保存
通过验收的结果，不保存算法或 schema 版本字段。

## 3. 修正结果验证

### 3.1 自动单元测试

```bash
python -m unittest tests/test_contact_trajectory_retarget.py -v
```

测试覆盖平滑位移轨迹、上下红区中心平移、refinement 帧数硬上限、兼容平移优化、Artimano URDF 碰撞链、腕姿/手指 DOF 不变量，以及相同位移是否同步进入 Stage1 的 MANO 目标。旧 schema-v6 SE(3) PKL 的读取兼容也保留测试。

### 3.2 报告和预览检查

`contact_report.json` 中至少确认：

- `passed: true`；
- `geometry_feasible: true`；
- `translation_mode: contact-translation`；
- `center_candidate_search` 包含全部 alpha、去重关系、逐候选指标、排名与 `selected_candidate`；
- `contact_transfer.method` 记录直接中心候选或 backoff/红区中心回退路径，同时检查 `preferred_initial_validation` 和 `fallback_used`；
- `preferred_region.method` 为 `semantic_axis_front_band` 或方向不可靠时的 `red_center_fallback`；
- `delta_tool` 和 `optimizer.final_delta_tool` 有限且一致；
- `semantic_precheck.passed: true`，且新旧 mesh 未分类顶点数均为零；
- 帧比例采用 OR 验收：`after_hold.source_red_frame_pass_ratio` 与 `after_hold.red_frame_pass_ratio` 任意一个达到 `settings.min_red_frame_ratio` 即通过；二者都低于门槛时该项才失败；
- `validation_tiers.semantic_geometry.red_frame_acceptance.passed_by` 记录实际由 `source_contact_frames`、`red_frames`、`both` 或 `neither` 哪一路判定；
- `after_hold.contact_retention_level` 优先为 `full`，无法全保留但满足动态 80% 门槛时为 `relaxed_80`；
- `after_hold.source_link_retention_ratio >= settings.min_contact_retention_ratio`；
- `after_hold.red_link_ratio_mean` 用于观察全部配置接触 link 的平均红区接触比例；
- `after_hold.black_violation_fraction <= settings.max_black_violation_fraction`；
- `invariants.passed: true`；
- `validation_tiers.semantic_geometry.passed: true`；
- `physics_validation_required` 与本次命令配置一致：仅添加 `--physx` 时为 `true`；
- 若 `physics_validation_required: true`，则 `validation_tiers.isaac_gym_physics.passed: true`；否则该层应为 `status: not_requested`、`required: false`；
- `rotation_max_abs_error` 和 `dof_max_abs_error` 为零，`wrist_translation_max_abs_error` 接近零。

添加 `--physx` 后还要求：工具发生接触的代表帧比例不低于 50%；对源接触手指分别统计 `per_link_contact_frame_ratio`，达到 50% 的源接触手指数不少于本序列动态 80% 数量；最大初始接触包络不超过 10 mm、最大接触力不超过 50000，且所有接触数值有限。PhysX 失败仍保存几何有效 PKL，但 `passed=false` 且 Stage1 验收门会拒绝读取。PhysX 层把每个 shape 的 contact offset 显式设为 3 mm，避免 Isaac Gym 默认每侧 20 mm 的接触包络被误判为约 40 mm 穿透。

打开 `contact_preview.png`，逐项检查紫色箭头由旧红区中心指向新红区中心、绿色接触 link 位于红区，手掌和手指没有落在黑区，渐入/渐出曲线连续。兼容 `semantic-axis` 模式下紫色箭头仍表示黑区到红区的语义轴。

### 3.3 独立复验

```bash
python main/dataset/contact_trajectory_retarget.py verify \
  --input-pkl data/retargeting_contact/OakInk-v2/mano2artimano_rh/<data_id>/<new_obj>/<sequence>@0.pkl \
  --new-obj spoon_01
```

该命令从磁盘重新加载源 PKL、新 mesh、碰撞 URDF 和修正 PKL，重新计算几何指标、不变量和碰撞资产 bundle 哈希。退出码为 `0` 且 `passed: true` 才可进入 Stage1。

### 3.4 Stage1/PhysX 冒烟检查

先用少量环境和一个 iteration 检查完整数据加载、碰撞资产和仿真是否正常：

```bash
python main/rl/train.py \
  task=ResDexHand dexhand=artimano side=BiH \
  new_obj=spoon_01 correct=true \
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
  new_obj=spoon_01 correct=true \
  headless=true num_envs=4096 learning_rate=2e-4 \
  test=false randomStateInit=true dataIndices=[083f7@0] \
  early_stop_epochs=10000 actionsMovingAverage=0.4 \
  experiment=083f7_contact_artimano
```

加载修正 PKL 时，数据集会自动把同一个逐帧工具局部平移同步到：

- `opt_wrist_pos`（PKL 中已修正，`opt_wrist_rot` 保持不变）；
- Stage1 的 `wrist_pos`；
- Stage1 的全部 `mano_joints`；
- 相应线速度、关节速度、`tips_distance` 和 `joints_distance`。

因此 Stage1 的复位目标、观测目标和奖励目标不会互相冲突。左手仍从 `data/retargeting` 读取，姿态和轨迹不变。

左手始终从 `data/retargeting` 读取原始 PKL、object mesh 和 URDF。右手读取由 Hydra
原生参数 `correct` 控制：`correct=false` 从 `data/retargeting` 读取原始右手 PKL；
`correct=true` 按 `<data_id>/<new_obj>` 从 `data/retargeting_contact` 读取修正 PKL。
`correct=true` 必须同时给出 `new_obj`，且目标修正 PKL 不存在时直接报错，不静默回退。

Stage1 的训练和测试使用同一入口。`new_obj=spoon_01` 选择右手新 tool mesh/URDF；
省略时读取 data_id 对应的原始 tool。`correct` 与 mesh 选择相互独立，因此
`correct=false new_obj=spoon_01` 会使用原始右手轨迹配合新 tool，而
`correct=true new_obj=spoon_01` 会读取该 tool 对应的修正右手轨迹。

该项目的完整执行流程到 Stage1 训练结束，不再包含后续残差修正阶段。

## 5. TACO 数据

TACO 使用现有三位 object ID 作为 `new_obj`。表面语义 mesh 和物理资产分别解析为：

```text
data/TACO/object_models/<id>/<id>.ply
data/TACO/coacd_object_models/<id>/<id>.ply
data/TACO/coacd_object_models/<id>/<id>.urdf
```

右手 tool 从标注 mesh 独立读取红/黑逐顶点语义，并向 Stage1 提供与 OakInk 相同的
`tool_red_points`、`tool_black_points` 和计数字段；TACO mesh 已经是米制，语义 scale
固定为 1。左手 target、原始 tool trajectory 和原始序列归一化不因 `new_obj` 改变。

生成 TACO 修正轨迹：

```bash
python main/dataset/contact_trajectory_retarget.py correct \
  --dataset taco \
  --data-idx bb_bowl_0919036 \
  --dexhand artimano \
  --new-obj 058 \
  --headless
```

输出路径为：

```text
data/retargeting_contact/TACO/mano2artimano_rh/
  bb_bowl_0919036/058/bb_bowl_0919036.pkl
```

Stage1 使用相同参数选择替换资产和修正 PKL：

```bash
python main/rl/train.py \
  task=ResDexHand dexhand=artimano side=BiH \
  dataIndices=[bb_bowl_0919036] new_obj=058 correct=true \
  headless=true num_envs=4 max_iterations=1
```

`correct=false` 时仍替换右手资产，但读取原始 TACO retarget PKL；`correct=true` 时读取
对应 `<sequence-id>/<new_obj>` 下的 contact-corrected PKL。两种情况下左手始终读取原始
target 资产和原始左手 PKL。Memory bank 不按数据集分层，只使用全局唯一的 sequence ID：

```text
data/retargeting_bank/<sequence-id>/<tool-id>/
```
