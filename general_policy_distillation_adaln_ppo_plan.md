# 通用左右手策略蒸馏预训练与 PPO 阶段 AdaLN-Zero 语义调制实现方案

日期：2026-10-26。项目根目录：`/Users/ylj/CV/A-PhysGraph`。

**状态：待实现的交接方案。本轮仅新增本文档，没有修改训练代码、配置、数据或 checkpoint，也没有启动训练。**

本文档汇总了本次讨论最终选定的方案；此前 `.review` 中涉及的任务 checkpoint、残差策略、教师 token、PPO 蒸馏项等备选不属于本次实现范围。

## 1. 已确定的目标与边界

利用通用左右手策略，先让当前双手 Transformer 学习基础运动和轨迹跟随，再按当前新工具 mesh 语义引导方法进行 PPO 训练。

本方案固定以下选择，后续实现窗口按此执行：

| 项目 | 阶段 A：动作蒸馏预训练 | 阶段 B：PPO 微调 |
|---|---|---|
| 初始化来源 | 随机初始化学生；加载两个通用 MLP 教师 | 加载阶段 A 的学生 actor 权重 |
| 右手参考 | 当前新工具对应的几何修正轨迹 | 同一几何修正轨迹 |
| 左手参考 | 当前流程的左手参考 | 当前流程的左手参考 |
| AdaLN-Zero 语义调制 | 明确旁路；语义专属模块冻结 | 启用当前语义课程；语义专属模块解冻 |
| 物理接触边、拓扑和几何结构 | 保留当前结构及实际物理输入 | 保留当前结构及实际物理输入 |
| 网络更新 | actor 路径：编码器、Transformer、POL、动作头 | 按当前 PPO 联合优化 actor/critic |
| 教师参与 | 冻结教师提供动作均值标签 | 不创建、不调用教师 |
| 训练目标 | 动作蒸馏损失 | 当前 PPO 损失；不加入蒸馏项 |
| 优化器与课程计数 | 预训练独立维护 | 新建优化器、环境和训练计数 |

只使用以下两个教师：

```text
左手：/Users/ylj/Desktop/run_data/imitator_lh_artimano.pth
右手：/Users/ylj/Desktop/run_data/imitator_rh_artimano.pth
```

本方案不使用任务专属 `9fc3e@0...pth`；不把 MLP 参数压缩成 POL；不增加教师 token、残差动作底座、PPO 蒸馏损失或额外语义奖励。POL 随完整学生 actor 预训练，并与配套网络共同迁移。

预训练改变策略初值。新工具语义几何修正与 PPO 阶段的 AdaLN-Zero 调制仍是现有方法的两个环节。不能在未实验前承诺加速比例或最终成功率提高。

## 2. 已核实的代码与 checkpoint 事实

以下结论来自当前本地源码和 checkpoint 参数元数据；尚未在 Isaac Gym/PyTorch 训练环境中验证前向行为。

- 两个教师均为单手 MLP，输出 28 维动作，观测统计量维度为 `privileged=22`、`proprioception=79`、`target=266`。
- 当前学生默认 `d_model=64`、4 层 Transformer、8 heads；POL 为静态 64 维参数；动作头为 `64→1024→512→56`。
- 当前后两层 Transformer 带 `SemanticAdaLNZero`；语义条件仅散布到五个右手接触 token。
- AdaLN 投影输出 `delta_gamma, beta, delta_gate`，投影权重和偏置均为零初始化。当前残差倍率为 `1+delta_gate`，初始值为 1；这点保持原实现。
- 当前左右手 base checkpoint 配置虽然存在，但 Transformer 构造和 forward 没有加载它们。不能靠填写这些旧字段完成本方案。
- 当前实时语义特征来自新工具红黑点集、参考接触意图及实际物理状态。接触语义分数仅记录，加到总 reward 的语句处于注释状态。
- 当前模型输入归一化函数直接返回原始观测。另已找到本机原始 Maniptrans 的教师实现和配置，其 `normalize_input=true` 并应用 RMS；教师应按原实现还原，不能复用当前项目的空操作预处理。checkpoint 与其原始运行版本的对应关系仍需前向一致性验证。

主要源码入口：

| 用途 | 当前文件与位置 |
|---|---|
| 学生结构、AdaLN-Zero、POL、动作头 | [network_builder_transformer_bih_graph_improve_correct.py](/Users/ylj/CV/A-PhysGraph/lib/rl/network_builder_transformer_bih_graph_improve_correct.py:121) |
| 五个接触 token 的语义条件和课程 mask | [同文件 build_semantic_token_condition](/Users/ylj/CV/A-PhysGraph/lib/rl/network_builder_transformer_bih_graph_improve_correct.py:604) |
| 旧策略输入预处理 | [models.py](/Users/ylj/CV/A-PhysGraph/lib/rl/models.py:69) |
| 原始教师的 RMS 预处理（只读参考） | [Maniptrans models.py](/Users/ylj/CV/Maniptrans/lib/rl/models.py:68)、[moving_avg.py](/Users/ylj/CV/Maniptrans/lib/rl/moving_avg.py:77) |
| 原始教师 MLP 实现（只读参考） | [Maniptrans network_builder.py](/Users/ylj/CV/Maniptrans/lib/rl/network_builder.py:122) |
| 融合输入按 key 排序后拼接 | [fusion.py](/Users/ylj/CV/A-PhysGraph/lib/nn/features/fusion.py:47) |
| 新工具与修正轨迹开关 | [config.yaml](/Users/ylj/CV/A-PhysGraph/main/cfg/config.yaml:33) |
| 原始/修正轨迹选择 | [contact_retarget_memory.py](/Users/ylj/CV/A-PhysGraph/main/contact_retarget_memory.py:49) |
| 修正跟踪目标并重算速度 | [dataset/base.py](/Users/ylj/CV/A-PhysGraph/main/dataset/base.py:203) |
| 双手观测拼接与实时语义 | [dexhandmanip_bih.py](/Users/ylj/CV/A-PhysGraph/physgraph_envs/lib/envs/tasks/dexhandmanip_bih.py:1376) |
| 当前 PPO 优化 | [agent.py](/Users/ylj/CV/A-PhysGraph/lib/rl/agent.py:105) |
| 创建训练实例与断点恢复 | [runner.py](/Users/ylj/CV/A-PhysGraph/lib/rl/runner.py:25) |
| 现有恢复会加载 optimizer | [rl/base.py](/Users/ylj/CV/A-PhysGraph/lib/rl/base.py:789) |

实现前重新核对以上符号，行号可能随后续编辑变化。

## 3. 数据与教师接口契约

### 3.1 几何修正轨迹必须真正进入两个阶段

两个阶段均设置 `new_obj=<目标工具>`、`correct=true`，检查右手数据日志为 `retargeting_kind=contact_corrected`，记录实际 PKL 路径。仅设置 `new_obj` 不会自动加载修正轨迹；当前 `correct` 默认是 `false`。

沿用当前数据加载器，使 dex-hand 初始化目标、MANO 手腕/关键点跟踪目标、速度/角速度及手物距离均来自一致的修正结果。不要只替换可视化姿态或仅更新 `opt_wrist_pos`。

教师权重冻结，但教师接收的目标是修正后的参考。阶段 A 关闭神经语义调制时，几何修正依然生效。保留现有 mesh/碰撞资源一致性和修正可行性校验。

### 3.2 当前 artimano 布局下的投影

适用前提：每侧 22 个手指关节，单手动作 28 维，`obsFutureLength=1`。必须通过运行时 shape 断言确认，不能静默适用于其他手型。

双手输入：`proprioception=[B,158]`、`privileged=[B,118]`、`target=[B,900]`。各字段都是右手在前、左手在后，每侧长度分别为 79、59、450。

| 教师字段 | 右手投影 | 左手投影 |
|---|---|---|
| proprioception | `obs["proprioception"][:, :79]` | `obs["proprioception"][:, 79:158]` |
| privileged，仅 dq | `obs["privileged"][:, :22]` | `obs["privileged"][:, 59:81]` |
| target，仅手部目标 | `obs["target"][:, :266]` | `obs["target"][:, 450:716]` |

投影后的每侧总输入为 367 维。只向教师传这三个字段。学生继续接收完整双手观测，以及 `phys_node_pos`、`physical_contact`、`semantic_contact`。

教师融合顺序按当前 `SimpleFeatureFusion` 的排序应为 `[privileged, proprioception, target]`。不要按配置文件显示顺序或 Python 字典插入顺序自行拼接。

### 3.3 参考时刻对齐：先验证，再大规模蒸馏

当前源码存在差异：单手 imitator 使用 `progress_buf+1`，双手操作环境使用 `progress_buf-1`。该差异不能通过参数形状检查解决。

本方案第一版默认教师使用与学生同一逻辑参考时刻，即上述当前双手目标投影。这样教师动作与学生当前目标对应。保留学生现有参考构造，不全局修改其时间索引，也不自动给教师目标加两帧。

实现者必须在少量环境中验证冻结教师对修正参考的跟踪。如果表现异常，先排查参考相位、手型/关节顺序、预处理与动作接口。若确认必须使用不同的教师前瞻，应显式记录目标帧，并检查学生是否具备相应目标信息；作为后续设计处理，不能在本方案中悄悄引入学生不可见的未来监督。

### 3.4 教师网络重建与预处理

已知 checkpoint 形状：融合 MLP `367→512→512→512→256`，actor MLP `256→256→512→128→64`，动作均值层 `64→28`。

优先使用生成这些 checkpoint 的原始教师实现及配置。当前配置给出的候选激活为融合层 SiLU/swish、actor 层 ELU；激活与预处理不能单从权重形状证明。若需要手工重建，必须核对原配置或通过原教师前向/回放建立等价性证据。

教师加载应明确匹配 feature encoder、actor MLP、mu 的键和形状，并单独完整加载所需输入 RMS 统计量。value、sigma 等不用的键可按已记录白名单排除；不得用无检查的 `strict=False` 掩盖 actor 或 RMS 漏载。

已读原始参考项目 `/Users/ylj/CV/Maniptrans`：其 `main/cfg/rl_train/DexHandImitatorPPO.yaml` 设置 `normalize_input=true`；`lib/rl/models.py` 应用 RMS；`lib/rl/moving_avg.py` 使用 epsilon=`1e-5`，标准化后裁剪到 `[-20,20]`。因此本方案默认教师采用 checkpoint 的冻结 RMS，按字段准确加载并只应用一次：

```text
x_normalized = clamp((x - running_mean) / sqrt(running_var + 1e-5), -20, 20)
```

遵循原实现的 dtype 转换，mean/var 在公式中转换为 float；保存的统计量本身维持原值。教师 RMS 处于 eval 状态，mean/var/count 始终不更新。**学生仍沿用当前项目的预处理，不把教师 RMS 迁移到学生。**

教师的原始运行版本若有更直接的配置证据，按该版本复现并记录；不得为了获得某个 rollout 表现而静默切换 raw/RMS 模式。用固定原始输入比较原始教师和重建教师的 raw mean，是开始蒸馏前的必要验收。原始参考仓库仅供读取，不在本方案中修改。

教师 `.eval()`、`requires_grad_(False)`，前向使用 `no_grad()`。监督读取确定性动作均值，避免使用随机采样的 `actions`、value 或隐藏特征平均值。

## 4. 阶段 A：只进行动作蒸馏

### 4.1 保持同构网络，明确旁路 AdaLN-Zero

新增一个模型级开关，建议名 `semantic_adaln_enabled`，默认 `true`。阶段 A 设置为 `false`；阶段 B 和未使用预训练的现有训练默认仍为 `true`。

建议保持环境 `semanticContact.enabled=true` 和完整观测结构，以便两个阶段结构一致、便于记录语义指标。模型开关为 `false` 时，在语义条件构造之前直接返回 `(None, None)`，不调用语义编码器，也不执行 AdaLN 调制。

不要删除 `SemanticConditionEncoder` 或最后两层的 `semantic_adaln` 模块，不修改 token 数量/位置，不截断共享 Transformer。两个阶段保留相同的参数名称和形状。开关是阶段配置，不写成可训练参数。

阶段 A 同时冻结 `semantic_condition_encoder` 与各层 `semantic_adaln` 参数，并将其排除出预训练 optimizer。AdaLN 的 modulation weight/bias 保持精确零值；普通注意力 residual 路径照常运行。

`physical_contact` 独立于语义调制开关，继续进入当前物理边构造；拓扑、几何和静态结构路径保留。不要通过关掉全部接触输入来近似“关闭 AdaLN”。

开关应在模型构造/编译前固定。当前构造器会 `torch.compile(latent_encoder)`，跨阶段采用新进程，避免复用按旧开关编译的图。未指定新字段时，应与现有训练行为一致。

### 4.2 数据采集和标签

在目标新工具环境中，由两个冻结教师查询当前双手状态，生成：

```text
mu_teacher = concat([mu_right, mu_left], dim=-1)  # [B,56]
```

动作排列为 `[右手手腕6, 右手手指22, 左手手腕6, 左手手指22]`。本方案第一版保持当前非 PID 控制模式。

阶段 A 用教师动作驱动环境采集观测。环境执行动作沿用阶段 B 的现有 soft-clamp/裁剪、rescale、平滑和控制流程；当前 `PPO.preprocess_actions` 中 soft-clamp 在 `[-1,1]` 上等价于 `tanh(raw_mean)`，后续按动作上下界 rescale。此处理只执行一次，标签始终另存 raw mean；不能先处理动作，调用环境封装时又处理一次。固定 dt、控制频率、关节限位及 translation/orientation scale。蒸馏标签与学生输出均使用相同的策略动作坐标，不把物理力矩或关节角直接作为 56 维标签。

第一版以教师原始动作均值监督学生原始均值，同时记录按环境裁剪后的动作误差。若后续选择裁剪后的监督标签，需要单独记录并保持两侧定义一致，不能混用。

覆盖多条训练轨迹和不同参考阶段，按轨迹或片段划分验证集。通用教师是否能完成整个工具任务需要先检查；不能假定两只手独立拼接就具备双手任务专家能力。记录失败和提前终止分布，避免数据只覆盖初始姿态。

从会原地更新的 GPU 环境张量存入 buffer 时，使用独立拷贝，例如 `detach().clone()`，避免下一次仿真覆盖旧样本。无需在第一版引入教师动作混合的 PPO rollout。

### 4.3 损失、优化对象与产物

学生从完整观测输出 `mu_student=[B,56]`。按左右手分别计算手腕与手指动作均方误差，采用组内均值后加权：

\[
L_A=\frac12\sum_{h\in\{R,L\}}
\left[\lambda_w\operatorname{MSE}(\mu^w_{S,h},\mu^w_{T,h})
+\lambda_j\operatorname{MSE}(\mu^j_{S,h},\mu^j_{T,h})\right].
\]

第一版 `lambda_w=lambda_j=1`，分别记录四项指标。组内均值避免手指项仅因 22 维大于手腕 6 维而占主导。损失不包含语义奖励、critic 回归或教师分布方差匹配。

训练学生的 latent encoder 中实际参与 actor 前向的参数、POL 和动作头。价值头、状态无关 logstd、语义专属模块不在预训练 optimizer 中。允许未使用的 tokenizer 等已有模块没有梯度，不为此增加额外损失。

训练预算、batch 大小和学习率作为显式配置与日志保存，先用小规模运行检查误差下降，再扩展。现阶段不声称某个固定 epoch 数或学习率最优。选择 checkpoint 时同时参考留出数据动作误差、学生独立 rollout 跟踪表现及接触诊断。

## 5. 阶段 A → B 的初始化 checkpoint 契约

新建一个专门的 actor 初始化格式，区别于现有 PPO 完整续训 checkpoint。建议结构：

```text
format_version: 1
stage: distill_actor
actor_state: canonical student state
metadata:
  dexhand / action layout / d_model / num_layers / obs layout version
  teacher paths and SHA256 / teacher normalization mode
  reference timing mode / dataIndices / new_obj / correct
  corrected reference PKL paths and identifiers / mesh identifiers
  stage-A semantic_adaln_enabled: false
  training config / seed / validation metrics / source version identifier if available
```

`actor_state` 明确包含两个前缀下的完整参数和持久 buffer：

```text
latent_encoder.*
action_miu_head.*
```

包括 latent encoder 内冻结的语义模块初始值与零初始化 AdaLN 投影。导出时统一去除 wrapper 的 `a2c_network.` 前缀，并以未编译模块的键导出，避免 `_orig_mod.` 路径差异。规范化只处理完整模块路径片段，检查规范化后没有键冲突，不做任意字符串替换。使用 CPU tensor 的独立副本。

不导出 value head、sigma、输入/价值归一化统计量、教师模型、optimizer、PPO epoch/frame 或环境状态。预训练的训练指标与配置放 metadata，不解释为 PPO 进度。

阶段 B 新建与当前配置一致的模型后，只加载上述 actor 前缀。加载器应检查 key 集合和 shape；不能仅对全模型调用 `strict=False` 后忽略返回值。显式报告有意保留初始化的 value head、sigma 和归一化状态。

阶段 B 使用新配置 `semantic_adaln_enabled=true`；不把 metadata 中阶段 A 的 false 自动应用到新模型。检查 AdaLN modulation 仍为零、语义专属模块恢复当前 PPO 模型默认可训练状态。普通 state_dict 不保存 `requires_grad`，跨阶段通过新建模型恢复默认定义，不对全模型执行 `requires_grad_(True)`：当前 `bias_gen.semantic_relation_scale` 是保留但不使用的永久冻结旧参数，必须继续冻结。

## 6. 阶段 B：现有语义 PPO 微调

建议以独立进程启动，创建新的仿真环境、PPO agent、optimizer、学习率调度器和统计状态。只加载 actor 初始化 checkpoint，不使用完整 PPO `restore()`。

保留当前设置：新工具 mesh、修正右手参考、左手参考、实时红黑语义计算、AdaLN 位置与范围、接触 token mask、物理图、PPO 损失和奖励定义、成功/失败与接触评估。

显式确保：

1. `semantic_adaln_enabled=true`，恢复当前新建模型的默认 trainability：语义模块与有效共享 actor 参数可训练，critic 和 sigma 按当前 PPO 正常训练；永久冻结的 legacy 参数继续冻结。
2. 不创建教师，不加载教师路径，不计算蒸馏损失；两个通用 checkpoint 仅作为预训练 metadata 留痕。
3. PPO rollout 由学生动作采样与学生 log probability 构成。
4. 训练 epoch/frame、环境 `total_train_env_frames` 和语义课程从新运行起点开始，不带入阶段 A 交互步数。
5. 新建仿真避免预训练累计的 `gym.get_frame_count` 提前推进误差收紧课程。
6. 沿用当前 `curriculumSteps=320000` 对应课程：前 5% 为 0，5–30% 从 0 到 0.25，30–60% 从 0.25 到 1，之后为 1。这里的进度依据 `total_train_env_frames/num_envs`，不能直接当成总 batch 交互量。
7. 不启用当前被注释的语义 reward 加法；语义指标继续按现有逻辑记录和评估。

零初始化的语义分支可以在 PPO 中逐步学习对已预训练控制行为的调节。初期课程为零时，语义梯度为零属于预期；投影权重为零时，语义编码器也可能在最初更新没有梯度，不能据此误判为断图。验收需在课程非零且经过有效更新后检查。

## 7. 建议的最小文件改动范围（供下一窗口实现）

以下均是未来实现位置，本轮不创建对应代码或修改配置。

| 文件 | 建议职责 |
|---|---|
| 新增 `/Users/ylj/CV/A-PhysGraph/lib/rl/imitator_teacher.py` | 重建/加载两个冻结教师，验证键、形状和预处理，提供动作均值 |
| 新增 `/Users/ylj/CV/A-PhysGraph/lib/rl/teacher_obs_adapter.py` | 单手字段投影、shape/动作顺序断言、目标帧 provenance |
| 新增 `/Users/ylj/CV/A-PhysGraph/main/rl/pretrain_distill.py` | 独立阶段 A 入口、环境/数据采集、BC optimizer、验证和导出 |
| 新增 `/Users/ylj/CV/A-PhysGraph/lib/rl/actor_initialization.py` | actor-only 格式、canonical key、显式加载校验 |
| 修改 `/Users/ylj/CV/A-PhysGraph/lib/rl/network_builder_transformer_bih_graph_improve_correct.py` | 透传模型级 AdaLN 开关，保留模块但在阶段 A 旁路条件分支 |
| 修改 `/Users/ylj/CV/A-PhysGraph/lib/rl/runner.py` | 在创建新 agent 后、训练前执行 model-only 初始化，保留现有续训路径 |
| 修改 `/Users/ylj/CV/A-PhysGraph/main/rl/train.py` | 将初始化配置传递到 runner；必要时抽取小型环境构造 helper 供两个入口复用 |
| 修改 `/Users/ylj/CV/A-PhysGraph/main/cfg/config.yaml` 与 `/Users/ylj/CV/A-PhysGraph/main/cfg/rl_train/ResDexHandPPO.yaml` | 声明新字段并正确透传；不改变未使用本方案的默认行为 |
| 新增对应测试文件 | 验证适配器、AdaLN 阶段开关、actor 初始化边界和小规模训练行为 |

当前 `PPOAgent.calc_gradients` 无需新增蒸馏项，优先保持 PPO loss 实现不变。无需改写几何修正算法、轨迹缓存选择规则或语义评估算法。

## 8. 配置接口建议

以下字段是待实现的建议接口，当前仓库尚不支持，不是现在可直接执行的命令：

```yaml
# 顶层独立字段；实际 Hydra 透传层级在实现中统一声明
actor_init_checkpoint: null  # 阶段 B 专用；与 checkpoint 完整续训互斥
semantic_adaln_enabled: true # 默认维持当前行为；阶段 A 显式 false
distillation:
  rh_teacher_checkpoint: /Users/ylj/Desktop/run_data/imitator_rh_artimano.pth
  lh_teacher_checkpoint: /Users/ylj/Desktop/run_data/imitator_lh_artimano.pth
  teacher_reference_mode: student_aligned
  teacher_normalization: frozen_rms # 按原始 Maniptrans 还原；前向一致性验证后固定
  wrist_loss_weight: 1.0
  joint_loss_weight: 1.0
```

教师路径也可复用已有 rh/lh base 字段，但必须显式限定在阶段 A 使用并记录语义；不要使这些遗留字段在普通 PPO 中突然触发教师底座行为。

`checkpoint` 继续表示原有完整 PPO 续训；`actor_init_checkpoint` 表示新运行的模型初始化。两者同时设置时应报清楚的配置错误，避免隐含加载优先级。

当前 `from_ckpt_epoch=false` 仍会恢复旧 optimizer，所以不能把它当作本方案的 model-only 初始化替代品。

## 9. 实现顺序与必要验收

按以下次序实现，前一项失败时先修复，不启动长时间训练。

### 第一步：教师与参考输入

- 校验两个 checkpoint 的 actor 参数与 RMS 完整加载，固定输入上的重建结果与原始教师 raw mean 一致；教师输出 `[B,28]` 且有限，参数和统计量始终不更新。
- 验证三个观测字段的具体值与左右手顺序，包含非对称测试数据，避免两个教师实际都读到右手。
- 记录修正参考路径和目标帧；在起点、中段、终点验证教师跟踪行为，排查当前单手/双手参考相位差。失败时停止大规模阶段 A，报告接口/时序问题，禁止静默加两帧。
- 两阶段实际右手数据来源均为 `contact_corrected`，学生和教师均使用这套参考。

### 第二步：阶段 A 的语义旁路

- 使用非零语义课程值和不同红黑语义输入测试：固定其他观测时，关闭 AdaLN 的学生动作应不随仅语义字段变化；保留 `physical_contact` 不变，避免混淆。
- 确认实际物理接触仍进入图边路径，不因 AdaLN 开关被移除。
- 在蒸馏更新前后比较：教师及冻结语义参数不变；AdaLN projection 精确为零；POL 和有效 actor 路径至少在若干 batch 中获得梯度并更新；value head 和 sigma 不变。
- 小规模验证集动作误差下降，学生独立 rollout 能跟随修正参考。未使用模块可以无梯度。

### 第三步：初始化导出与加载

- 编译/未编译情况下导出同一 canonical key 集合，并在另一包装模式下正确加载。
- 在相同观测、eval 模式和语义旁路下，导出前后动作均值一致至合理浮点容差。
- 阶段 B 语义分支为零初始化调制时，在相同权重/观测下与阶段 A 旁路输出一致。
- value、sigma、RMS、optimizer、训练计数和环境状态没有从阶段 A 继承；语义模块的默认可训练状态已恢复，legacy semantic scale 仍为冻结状态。
- 错误手型、缺失关键 actor 键或 shape 不一致时清楚报错，不能静默部分加载。

### 第四步：PPO 与默认行为回归

- 不设置 `actor_init_checkpoint` 时，原有训练与完整续训接口正常；新开关默认 true 保持原网络路径。
- 设置初始化时，PPO 新运行从独立课程起点开始，教师推理计数为零，PPO loss 中无蒸馏项。
- 非零课程下 AdaLN 能获得梯度；经过有效更新后语义编码器能参与学习，掩码仍仅作用于指定接触 token。
- 原有语义 reward 记录、修正轨迹校验和全操作接触评估保持原定义。
- 运行现有相关测试：`test_semantic_contact.py`、`test_semantic_contact_eval.py`，若触及数据加载再运行 `test_contact_trajectory_retarget.py`。在具备 Isaac Gym 的训练环境完成教师及学生小规模 rollout 和短 PPO 验证。

## 10. 实验记录与完成交付

用相同任务、修正轨迹、语义配置和预算比较“随机初始化＋当前语义 PPO”与“蒸馏初始化＋当前语义 PPO”。记录达到固定成功率的 PPO 环境步数，以及含阶段 A 数据采集/教师推理/训练在内的总耗时；同时报告最终任务与语义接触指标。多个随机种子验证结果稳定性。

如果比较 AdaLN-Zero 的贡献，各组使用同一蒸馏初始化权重，只改变语义设置，避免把预训练贡献归入语义模块。

下一窗口完成实现后应交付：新增/修改文件列表、可执行的两阶段命令、配置示例、actor 初始化产物格式、教师接口验证结果、短训练结果及剩余限制。未完成运行验证时明确报告，不把文档设计视为已证实的训练收益。

## 11. 可直接复制到另一窗口的任务说明

> 请读取 `/Users/ylj/CV/A-PhysGraph/docs/general_policy_distillation_adaln_ppo_plan.md`，按其中固定方案实现。只使用通用左右手 imitator checkpoint：阶段 A 用几何修正后的右手参考进行动作蒸馏，保留物理接触边，旁路并冻结 AdaLN-Zero 语义分支；阶段 B 以 actor-only 权重初始化全新的 PPO 运行，启用现有 mesh 语义课程和 AdaLN-Zero，教师与蒸馏损失均不进入 PPO。先验证教师观测、归一化和参考时刻，再实现预训练、导出、加载与必要测试。保留原有默认训练、完整续训、几何修正与语义评估行为。本文档是方案，当前尚无对应实现；先检查文件现状，避免覆盖其他窗口已有改动。
