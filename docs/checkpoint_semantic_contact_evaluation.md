# 已保存 checkpoint 的完整操作片段语义接触评估

`ResDexHand` 双手环境在 `test=true` 时默认启用四项独立评估指标。
无需重新训练；继续使用 checkpoint 训练时的网络、动作处理、归一化和参考轨迹配置。
`correct=true` 使用相应新工具的修正轨迹，`correct=false` 使用原始轨迹。
评估不改变策略观测、reward、成功/失败判定或训练路径。

## 测试方式

例如，测试使用修正轨迹训练的 artimano 模型：

```bash
python main/rl/train.py \
  task=ResDexHand dexhand=artimano side=BiH \
  test=true randomStateInit=false rolloutBegin=null rolloutLen=null \
  dataIndices=[9fc3e@0] new_obj=kni@1 correct=true \
  checkpoint=/path/to/checkpoint.pth headless=true num_envs=4
```

模型本身使用原始轨迹时保留 `correct=false`，不要为了评估改用修正轨迹。
其余参数（例如 `actionsMovingAverage`、PID 模式及策略语义开关）沿用该模型的训练配置。
这里只提供评估统计，不依据 checkpoint 文件名猜测或覆盖模型配置。

四项指标都统计 `dataIndices` 选择并实际加载的完整操作片段。由于原代码的
正常终止条件为 `t + 1 + 3 >= seq_len`，完整区间为从帧 0 到正常终止帧，
不把未来观测保留的末尾三帧算成未执行。提前失败仍统计失败帧的接触。
随机/非零起点及 `rolloutLen` 裁剪的测试可以计算接触比例，但不能把一个
未完整执行的操作片段算成语义成功。若需完整片段结果，使用上述三个起点参数。

## 接触检测及指标定义

启动日志明确打印 `backend=collision_surface_proxy`。
当前 GPU 管线采用确定性的 URDF 碰撞表面采样，覆盖全部右手碰撞 link
（包括手掌和指节），而非仅五个末端 link 或 link 原点。
每步按实时刚体位姿将样本变换到当前工具局部坐标系，再计算到红、黑区域的距离。
只统计所属 link 净接触力达到阈值且样本位于工具表面阈值范围内的点。
最近红区归为红接触；最近黑区归为黑接触；等距边界保守归为黑区。
一个事件指一个有效碰撞表面采样点在一个控制步的状态。

这是统一的接触代理指标，**不代表精确 PhysX 接触点或接触面积**。
净接触力也可能包含其他物体的作用，距离筛选降低但不能完全排除此误差。
不同方法必须使用相同机器人、采样数、物理配置、语义 mesh 和阈值；
正式实验应抽样验证检测结果。同一工具上先汇总，再做跨工具比较。

日志的四个主指标为：

- `operation_completion_ratio`：实际执行的片段帧数 / 完整片段计划帧数。
- `planned_semantic_contact_coverage`：应接触且实际完成合规红区接触的帧数 /
  **整个计划**中全部应接触帧数。合规帧要求至少一个红接触且无黑接触。
- `black_violation_frame_ratio`：至少一个黑接触的帧数 / 实际执行的片段帧数。
- `semantic_task_success_rate`：当前结束 episode 中，原任务成功且完整片段语义合规的比例。

红黑接触事件占比仍保留在评估快照及批量汇总 API 中作为辅助统计，
不再列为主日志字段。它们的分母仅包含已发生的接触事件，无法单独体现操作完成程度。

完整操作要求从帧 0 连续执行到当前正常终止帧。提前失败不会缩短两个计划分母；
剩余的应接触帧算作未完成计划，不伪造为实际黑接触或已观测到的漏抓。
正常接近、释放帧若未标记为应接触，不降低计划接触覆盖率。

无工具接触时，存在应接触计划则覆盖率为 0；原红黑事件占比为 `NA`，
违规帧比例为 0，语义成功为失败。没有任何应接触标记时，覆盖率为 `NA`，
语义成功也为失败，需要检查任务标注，而不是把空分母当成成功。
语义合规保留红区占比下限、两个黑区违规上限及最短有效接触帧数条件，
并新增计划接触覆盖率下限（默认 0.80）。一次短暂触碰不能掩盖持续漏抓。

## 共同的应接触时间标记

标记在开始 rollout 前固定，之后不会依据策略实际接触或失败位置改变。
所有方法都应使用同一份任务标记；策略仍保留各自训练时的参考轨迹。
加载优先级如下：

1. 显式配置的 `contactRequiredMaskFile` 共享 JSON。
2. 对应修正 PKL 中的 `source_contact_intent.contact_frame_mask`。
   该标记在几何修正**之前**由原工具接触意图提取，属于 `grasp_interval` 局部时间轴；
   评估器按区间起点映射回完整操作时间轴，并对配置 contact link 取任一接触。
   `correct=true` 从自己加载的修正 PKL 读取；`correct=false new_obj=...` 在同一项目
   `data/retargeting_contact/<dataset>/<hand>/<data_id>/<new_obj>/<filename>` 查找同一份标注。
   此读取只提供评估标记，不会替换基线的策略参考轨迹。
3. 旧版修正 PKL 如果只保存接触摘要而没有逐帧标记，则将共享的 `grasp_interval`
   作为连续应接触区间，来源打印为 `source_grasp_interval`。这是较粗的源抓握时间标记，
   可以通过共享 JSON 细化，无需重做轨迹或重新训练。
4. 缺少共享标记及区间时，从 checkpoint 参考轨迹的 `tips_distance` 推断任一指尖
   距工具不超过 `expectedContactDistance` 的帧，默认 0.03 m。
   此回退的来源会明确打印；不同参考可能生成不同标记，正式跨方法比较应补充共享 JSON。

JSON 以操作 ID 为键，可以是逐帧布尔数组，也可以是包含首尾的应接触区间。
例如（ID 和区间仅为格式示意，实际值须由任务标注确定）：

```json
{
  "sequence@0": {"required_intervals": [[100, 299], [350, 399]]},
  "another@0": [false, true, true, false]
}
```

逐帧数组必须覆盖该操作的完整有效帧数。所有索引都相对于选中的操作片段，
不是原始 mocap 文件索引，也不是随机起点之后的 episode 局部计数。
同一标记可以复制到其他方法的评估项目并配置：
`task.env.semanticContactEval.contactRequiredMaskFile=/path/to/shared_masks.json`。
空标记可以明确表示没有应接触帧；错误的 ID、非二值数组或不匹配的 PKL 区间会报错，
不静默改用其他标记。

## 日志与并行统计

示例（数值仅用于说明）：

```text
reward: 454.82 steps: 36.0 operation_completion_ratio: 0.1200 planned_semantic_contact_coverage: 0.1500 black_violation_frame_ratio: 0.0000 semantic_task_success_rate: 0.0000 semantic_sr_total: 0.0000 eval_frames: 36 complete_episodes: 0/1 no_contact_episodes: 0 planned_frames: 300 required_frames: 200 compliant_required_frames: 30 missing_contact_frames: 0 unexecuted_required_frames: 170
```

所有比例范围为 0–1。`semantic_sr_total` 为本次 Player 运行以来全部结束
episode 的累计语义成功率，包含失败和无接触 episode。
多个环境同一步结束时，保持旧日志的批量输出形式：reward 和 steps 为均值，
完成比例为执行帧总数 / 计划帧总数；覆盖率为合规应接触帧总数 / 计划应接触帧总数；
违规帧比例按实际帧总数汇总，
语义成功率按结束 episode 数汇总。未结束的环境不会混入本条日志。
`eval_frames` 为实际统计的控制步总数，不含随机起点偏移；
`complete_episodes` 与 `no_contact_episodes` 帮助解释提前失败和无接触结果。
`planned_frames`、`required_frames` 和 `compliant_required_frames` 显示两个计划分母与
覆盖率分子。`missing_contact_frames` 仅指已执行且应接触、但红黑区都没有接触的帧；
`unexecuted_required_frames` 指剩余未执行的应接触帧。快照另存 `black_required_frames`，
计划应接触帧由合规、黑区违规、已执行漏抓和未执行四类完整划分。
采集发生在计数递增和复位前，复位仅清空对应环境的统计。

## 可配置阈值与公平性边界

参数位于 `task.env.semanticContactEval`，独立于 `task.env.semanticContact.enabled`：

```yaml
enabled: true
forceThreshold: 1.0
contactDistance: 0.01
pointsPerLink: 32
envChunkSize: 16
minRedContactRatio: 0.90
maxBlackContactRatio: 0.05
maxBlackViolationFrameRatio: 0.05
minContactFrames: 3
minPlannedSemanticContactCoverage: 0.80
expectedContactDistance: 0.03
contactRequiredMaskFile: null
```

阈值是可配置的初始评估标准，应在正式比较前确定并对所有方法保持一致。
关闭附加评估可用 `task.env.semanticContactEval.enabled=false`。
它不会关闭策略本身的语义观测；关闭策略语义观测也不会关闭附加评估。

按本项目的当前实验设定，各方法继续使用自己训练时的参考轨迹和原任务判定。
因此这里的语义成功率是 **各自原任务成功与共同语义区域约束的联合结果**。
若不同参考导致失败/成功条件不同，该指标不能单独解释为完全统一的功能任务成功率。
计划覆盖率同时受操作完成程度和接触质量影响，因此与完成比例同时报告。
黑区违规帧比例仍只描述已执行期间的违规频率；提前失败造成暴露时长差异，
不能仅凭违规为零认定完整操作中的接触质量更优。

## 验证

```bash
python -m unittest discover -s tests -p 'test_semantic_contact_eval.py' -v
```

纯 CPU 测试覆盖表面坐标变换、缩放、颜色边界、阈值、失败及终止帧统计、
无接触、最短接触时长、随机起点、并行复位、计划分母及真实 Player 日志方法；
另外验证共享 JSON 优先级、原始/修正轨迹共用源标记、局部区间到完整时间轴映射，
以及漏抓、黑接触和未执行应接触帧的分离。
实际 checkpoint 仿真还需要可用的 Isaac Gym 运行环境和 NVIDIA 驱动。
