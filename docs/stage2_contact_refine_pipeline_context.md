# Stage 2 接触区域残差策略：现状、推荐 Pipeline 与后续讨论上下文

> 文档状态：方案讨论稿，尚未修改模型实现  
> 最后更新：2026-07-28  
> 适用代码：当前 `PhysGraph_CR` 工作区  
> 目的：让新的 Codex 对话窗口只阅读本文件，就能延续 Stage 2 接触区域残差方案的讨论、设计和后续实现。

## 1. 新对话如何继续

在新对话中可以直接输入：

> 请先阅读 `docs/stage2_contact_refine_pipeline_context.md`，基于其中已经确认的当前实现、推荐 Pipeline 和待决策项继续讨论。先不要修改代码，除非我明确要求实现。

如果后续决定实施，还应要求 Codex 再次核对本文引用的代码，因为仓库可能已经发生变化。

## 2. 当前讨论的目标与边界

目标是改进 Stage 2，使其更符合“在冻结 Stage 1 动作策略基础上，根据工具红/黑接触区域进行小幅、稳定、可解释的动作修正”。

重点问题：

1. 当前 Stage 2 为什么运行一套冻结 Base PhysGraph 和一套从头训练的 Residual PhysGraph。
2. 接触区域信息当前如何进入动作分布。
3. 是否应直接复用 Stage 1 的 policy token 和基础动作，而不是重新训练完整 Residual PhysGraph。
4. 是否应引入类似 AHA-WAM 的 cross-attention、gated residual 或 layer-wise context routing。
5. 如何继续保留现有动作空间 residual mask 带来的安全性。

当前约束：

- Stage 1 checkpoint 必须保持冻结，Stage 2 不应破坏原有轨迹策略。
- 当前只允许修正右手腕 6 维动作：3 维平移和 3 维旋转。
- 右手手指和整个左手的动作均值应默认保持 Stage 1 输出。
- 本文只记录方案；截至本文创建时，模型代码尚未按推荐 Pipeline 修改。

## 3. 已确认的当前 Stage 2 实现

### 3.1 当前总体结构

当前 Stage 2 不是在 Stage 1 Transformer 内插入一个小型 contact adapter，而是每次前向运行两套独立的完整 PhysGraph：

```text
                    ┌─ Frozen Stage-1 Base Policy ─→ μ_base
常规观测 obs ───────┤
                    └─ Trainable Residual PhysGraph ─→ z_state ─┐
                                                               ├─ Contact Fuser ─→ z_refine
contact_refine ─→ Contact Region Encoder ─→ contact_z ─────────┘
                                                               │
                                                               ▼
                                                     Residual Actor Head
                                                               │
                                                               ▼
                                                         Δμ_contact

μ_final = μ_base + 0.25 × M ⊙ Δμ_contact
```

其中：

- `Base Policy` 加载 Stage 1 checkpoint，完全冻结并在 `torch.no_grad()` 中运行。
- `Residual PhysGraph` 是另一个独立实例，不与 Base Policy 共享参数。
- 当前 runner 在 Stage 2 训练时跳过外层 checkpoint restore，因此 trainable Residual PhysGraph 没有从 Stage 1 checkpoint 初始化，而是从随机初始化开始学习。
- 两个 PhysGraph 在数据依赖上相互独立，只是代码中顺序执行；交换执行顺序不会改变结果。
- Base Policy 的 policy token 不会传给 residual branch。
- Base Policy 的动作均值也不会输入 residual branch。
- 两个分支直到动作空间才发生联系。

主要代码：

- `lib/rl/network_builder_transformer_bih_graph_improve_correct.py`
  - `BimanualHandToolPolicy.__init__`
  - `BimanualHandToolPolicy.forward`
  - `_load_base_policy`
- `lib/rl/runner.py`
  - `_restore`

### 3.2 常规观测与 `z_state`

当前 trainable Residual PhysGraph 接收与 Stage 1 相同的常规观测：

- `proprioception`：双手关节、三角编码和 wrist base state。
- `privileged`：速度、工具/物体状态、tip force、COM 和重量等。
- `target`：轨迹目标、link 位置/速度误差、工具/物体目标、完整 BPS 等。

该观测经过 tokenization 和 4 层 PhysGraph Transformer：

```text
常规观测
  ↓
61 × 64D tokens
  ↓
4 × Graph Transformer Layer
  ↓
Final LayerNorm
  ↓
读取第 0 个 policy token
  ↓
z_state ∈ R^(B×64)
```

因此：

```text
z_state = E_residual_graph(obs)
```

`z_state` 是 trainable Residual PhysGraph 对当前双手—工具—物体状态和轨迹目标的全局 64 维表示。它不是 Stage 1 的 `z_base`，也不包含 `contact_refine` 红黑区域信息。

当前每个 PhysGraph Encoder 的主要结构：

```text
Tokenizers:
input → Linear(input,256) → SiLU
      → Linear(256,128) → SiLU
      → Linear(128,64)

Transformer Layer × 4:
X → LayerNorm
  → 8-head self-attention, d_model=64, head_dim=8
  → residual add
  → LayerNorm
  → Linear(64,512) → ReLU → Linear(512,64)
  → residual add

Final LayerNorm → policy token [B,64]
```

需要注意的代码现状：

- Stage 1 pipeline 图中描述的 dynamic contact bias，在当前 forward 中没有传入 `contact_mask`，因此未实际启用。
- geometric bias 没有传入 `x_pos`，因此未实际启用。
- type/side/node-ID embeddings 虽然定义了，但当前 forward 没有实际添加。
- 红黑接触区域完全不进入这 4 层图注意力。

### 3.3 `contact_refine` 与 `contact_z`

工具可视化 PLY 通过顶点颜色划分：

- 红区：期望接触区域。
- 黑区：不期望或禁止接触区域。

红区和黑区各采样 512 个点，然后分别编码为 128 维 BPS 距离。最终：

```text
contact_refine =
[
  red_bps,       # 128
  black_bps,     # 128
  red_valid,     # 1
  black_valid    # 1
]
→ [B,258]
```

Contact Region Encoder：

```text
258
 ↓ Linear(258,256)
SiLU
 ↓ Linear(256,128)
SiLU
 ↓ Linear(128,64)
contact_z ∈ R^(B×64)
```

因此：

```text
contact_z = E_contact(contact_refine)
```

`contact_z` 是红黑区域静态几何的全局 64 维表示。它本身不包含：

- 当前哪个手指接近红区；
- 当前哪个手指接触黑区；
- 红黑区域相对当前 wrist/fingertip 的方向；
- 当前应向哪个方向修正动作。

这些动态因素只能由 `z_state` 间接提供。

`contact_refine` 在一个 episode 内是静态的。如果 Stage 2 只训练单一任务/单一工具，它在所有环境中也可能基本相同；此时 `contact_z` 更接近一个固定的工具语义条件或网络偏置，而不是随控制步变化的 contact feedback。

### 3.4 red/black 有效位

`red_valid` 和 `black_valid` 用于区分：

- 真实存在的语义区域；
- 为保持固定输入尺寸而生成的 fallback/padding。

它们有三条作用路径：

1. 作为 `contact_refine` 最后两维输入 Contact Region Encoder。
2. reward 代码直接读取原始 valid flag，无效时将对应 reward/penalty 置零。
3. Stage 2 初始化时检查红区和黑区是否都有效；当前实现如果任一无效会直接报错。

因此在正常 Stage 2 训练中，两个值通常恒为 `[1,1]`，对神经网络的动态区分能力很弱，主要保留数据 schema 和异常处理意义。

### 3.5 Contact Fuser 与 `z_refine`

Contact Region Encoder 和 Contact Fuser 是两个不同模块。

先拼接：

```text
z_state   [B,64]
contact_z [B,64]
----------------
concat    [B,128]
```

Contact Fuser：

```text
128
 ↓ Linear(128,64)
SiLU
 ↓ Linear(64,64)
z_refine ∈ R^(B×64)
```

公式：

```text
z_refine = F_contact([z_state, contact_z])
```

`z_refine` 是 residual branch 的接触条件化 latent，随后同时用于：

- Residual Actor Head，产生动作均值修正量；
- Stage 2 Value Head，估计 PPO value。

当前没有：

```text
z_refine = z_state + Δz_contact
```

也没有：

```text
z_refine = z_base + Δz_contact
```

当前 Contact Fuser 是 concat 后重新映射，不是 gated residual。

### 3.6 动作均值与分布

Frozen Base Policy：

```text
z_base → Frozen Base Actor Head → μ_base ∈ R^(B×56)
```

Residual branch：

```text
z_refine
  ↓ Linear(64,1024)
SiLU
  ↓ Linear(1024,512)
SiLU
  ↓ Linear(512,56)
Δμ_contact ∈ R^(B×56)
```

最后一层 `Linear(512,56)` 的 weight 和 bias 零初始化，所以 Stage 2 初始时：

```text
Δμ_contact = 0
μ_final = μ_base
```

最终动作均值：

```text
μ_final = μ_base + 0.25 × M ⊙ Δμ_contact
```

其中：

```text
M[0:6]  = 1   # 右手腕 3D translation + 3D rotation
M[6:56] = 0   # 右手手指和整个左手保持 Base mean
```

最终只有一个动作分布：

```text
a ~ Normal(μ_final, diag(σ²))
```

这不是两个高斯分布的融合。Base Policy 的方差不会参与组合。当前 logstd：

- 右手腕前 6 维：`-5`，标准差约 `0.0067`。
- 其余维度：forward 中固定为 `-8`，标准差约 `0.000335`。

接触信息只影响最终均值，不影响状态条件化方差。

### 3.7 PPO 如何让 contact 信息影响动作

梯度链路：

```text
contact_refine
  → Contact Region Encoder
  → contact_z
  → Contact Fuser
  → z_refine
  → Residual Actor Head
  → Δμ_contact
  → μ_final
  → sampled action
  → simulation/contact result
  → imitation + contact reward
  → PPO actor loss
```

Base mean 使用 `torch.no_grad()`：

```text
μ_final = stopgrad(μ_base) + 0.25 × M ⊙ Δμ_contact
```

所以 PPO actor gradient 只能更新：

- trainable Residual PhysGraph；
- Contact Region Encoder；
- Contact Fuser；
- Residual Actor Head；
- 有效的 wrist logstd。

PPO critic 还通过：

```text
z_refine + privileged obs → Value Head → V(s)
```

训练 Residual PhysGraph、Contact Region Encoder 和 Contact Fuser。

Residual Actor Head 最后一层零初始化带来一个细节：

- 第一次 actor backward 主要更新零初始化的最后一层；
- 此时 actor gradient 不能通过零权重继续传到 `z_refine`；
- Value Head 未零初始化，因此 critic gradient 可以从一开始训练 Fuser、Contact Encoder 和 Residual PhysGraph；
- Actor Head 最后一层变为非零后，actor gradient 才继续传入这些模块。

### 3.8 当前 contact reward

总 reward：

```text
r_total = r_imitation_RH + r_imitation_LH + r_contact_RH
```

右手 contact reward：

```text
r_contact = 1.0 × r_red - 2.0 × penalty_black
```

关键定义：

- active contact：接触力范数大于 `0.1`。
- 红区阈值：`0.02 m`。
- 红区 expected-contact 辅助范围：`0.04 m`。
- 黑区阈值：`0.015 m`。
- `r_red` 使用到红区最近距离的 `exp(-50d)`。
- `penalty_black` 只对真实 active contact 生效。

reward 使用当前 contact body 世界位置和变换到世界坐标的红/黑点集，因此 reward 是动态的；但这套动态的 finger-to-region 距离没有作为显式 observation 输入 Contact Region Encoder。

## 4. 对当前实现的判断

### 4.1 为什么不共享参数有合理性

如果 Base Policy 和 Residual Policy 共享同一个可训练 Encoder，PPO 更新 residual 时也会改变 Base Policy 表示，使 `μ_base` 不再保持 Stage 1 行为。独立参数可以确保 Base Policy 完全冻结。

### 4.2 为什么“从头训练完整 Residual PhysGraph”值得质疑

不共享参数不等于必须随机初始化第二套完整 PhysGraph。当前方案的主要代价：

- 每次前向运行两个完整 PhysGraph，计算和显存开销接近翻倍。
- Stage 2 需要重新学习 Stage 1 已掌握的手—工具—物体状态表示。
- 接触区域依然在 Transformer 后晚融合，第二套完整图网络没有让红黑区域参与 token-level reasoning。
- 在有限的 Stage 2 PPO 预算下，随机初始化完整 backbone 的样本效率可能较低。
- Residual Network 没有显式看到 Base Policy 当前准备执行的动作。

### 4.3 当前 residual 的准确含义

当前 residual 是：

```text
Δμ_contact = Rθ(obs, contact_refine)
```

不是：

```text
Δμ_contact = Rθ(obs, contact_refine, μ_base)
```

也不是：

```text
Δμ_contact = Rθ(z_base, contact_refine)
```

因此它通过最终 PPO reward 间接学会补偿 Base Policy，但没有显式获得“Base Policy 此刻准备做什么”。

## 5. 推荐 Pipeline

### 5.1 推荐原则

推荐方案应满足：

1. Stage 1 backbone 和 Base Actor 永久冻结。
2. 只运行一次 Stage 1 PhysGraph，复用其 policy token 或 layerwise tokens。
3. Residual Network 显式获得 Base Policy 当前动作，至少获得可修正的 `μ_base[0:6]`。
4. 接触区域不能只作为一个静态全局向量；应逐步引入当前 hand/tool 与 red/black region 的动态关系。
5. 最后继续使用动作空间 masked residual，保证未授权动作维度不变。
6. 新模块应零初始化或 gate 零初始化，确保 Stage 2 初始行为严格复现 Stage 1。

### 5.2 推荐总体结构

```text
常规观测 obs
   │
   ▼
Frozen Stage-1 PhysGraph（只运行一次）
   │
   ├──────────────→ z_base [B,64]
   │                   │
   │                   ├───────────────┐
   │                   │               │
   ▼                   │               │
Frozen Base Actor      │               │
   │                   │               │
   ▼                   │               │
μ_base [B,56] ─────────┤               │
                       │               │
μ_base[0:6] ───────────┤               │
                       │               │
red/black region       │               │
   │                   │               │
   ▼                   │               │
Region Encoder/Tokens ─┤               │
                       │               │
dynamic fingertip-     │               │
region geometry ───────┘               │
                                       ▼
                              Trainable Residual Adapter
                                       │
                                       ▼
                              Δμ_contact [B,6 or 56]
                                       │
                                       ▼
                 μ_final = μ_base + s × M ⊙ Δμ_contact
                                       │
                                       ▼
                             Normal(μ_final, σ)
```

推荐 residual 输入：

```text
Δμ = Rθ(
    stopgrad(z_base),
    stopgrad(μ_base[0:6]),
    contact representation,
    dynamic contact geometry
)
```

只输入 `μ_base + contact_z` 不够，因为不同状态可能产生相似基础动作，但需要不同接触修正。`z_base` 或动态状态信息仍然必要。

### 5.3 推荐的分阶段方案

#### Phase A：最小可行版本，优先验证

目标：先验证“复用 Stage 1 表示 + 显式输入 Base action”是否优于当前双 Encoder。

```text
z_base = FrozenPhysGraph(obs)
μ_base = FrozenActor(z_base)
contact_z = ContactEncoder(contact_refine)

adapter_input = [
    stopgrad(z_base),        # 64
    stopgrad(μ_base[0:6]),   # 6
    contact_z               # 64
]

Δμ_wrist = ResidualMLP(adapter_input)  # 6
μ_final[0:6] = μ_base[0:6] + s × Δμ_wrist
μ_final[6:56] = μ_base[6:56]
```

建议：

- Residual Head 直接输出 6 维，而不是先输出 56 维再 mask。
- 输出层零初始化。
- `s` 初始保留 `0.25`，后续再消融。
- Base `z_base` 和 `μ_base` 均 `detach/stopgrad`。
- Value Head 可以使用 `[z_base.detach(), contact_z, dynamic_contact]`，但 actor/critic 是否共享 contact adapter 需要单独决定。

优点：

- 只运行一个 PhysGraph。
- 复用 Stage 1 已学表示。
- Residual 明确知道当前基础 wrist action。
- 实现和调试风险最低。

局限：

- `contact_z` 仍是静态全局向量。
- 没有 token-level red/black 选择。
- 不能充分表达“哪个手指对应哪个区域”。

#### Phase B：加入动态 contact geometry

把 reward 中已经计算但尚未输入 policy 的动态几何关系显式提供给 residual adapter，例如：

- 每个右手 contact body 到红区最近距离；
- 每个 contact body 到黑区最近距离；
- 最近红/黑点相对 contact body 的 3D 向量；
- active-contact mask 和 contact force；
- wrist/tool frame 中的相对位置；
- red/black valid mask。

示例：

```text
dynamic_contact_i =
[
  d_red_i,
  d_black_i,
  vector_to_red_i,
  vector_to_black_i,
  contact_force_i,
  active_i
]
```

这一步很重要，因为当前静态 `contact_z` 只描述区域形状，并不告诉网络此刻手指和区域的相对关系。

#### Phase C：Cross-Attention Residual Adapter

将 red/black region 表示为多个 token，而不是单个 `contact_z`：

```text
C_region ∈ R^(B×Nc×64)
```

可选 tokenization：

- 将 128 个 BPS anchors 池化为 16–32 个区域 token；
- 每个 token 包含 red distance、black distance、anchor position/index、region type 和 valid mask；
- 或直接将红/黑点云分别编码为少量 point tokens。

Query 建议来自：

- Stage 1 的 RH hand/palm token；
- RH contact-link/fingertip tokens；
- tool token；
- policy token；
- `μ_base[0:6]` 对应的 action query。

单层推荐形式：

```text
U = CrossAttn(
    Q = current RH/tool/base-policy tokens,
    K = region tokens,
    V = region tokens
)

H_adapt = H_base + α × Wo(U)
```

其中 gate `α` 零初始化。然后：

```text
Δμ_wrist = ResidualHead(Pool(H_adapt), μ_base[0:6])
μ_final = μ_base + s × M ⊙ Δμ_wrist
```

重要警告：

- 如果把当前单个 `contact_z [B,1,64]` 作为唯一 K/V，cross-attention 的 softmax 只有一个 key，权重恒为 1，会退化为近似线性变换，不能实现有意义的区域选择。
- 因此 cross-attention 需要多个 region tokens。

#### Phase D：AHA-WAM 风格 Layer-wise Routing（高风险备选）

AHA-WAM 中图示的“cross-attention 后 residual update”严格来说是 OVCR：

```text
latest observation query
  → cross-attend cached planner K/V
  → predict ΔK/ΔV
  → gated residual update planner K/V
  → Action DiT layer-wise joint attention
```

映射到 PhysGraph 可以设计为：

```text
Stage-1 layerwise tokens/KV          ← planner context
current contact-aware queries        ← observation-guided routing queries
updated Stage-1 KV                   ← contact-conditioned context
lightweight residual executor        ← action expert
```

但当前 PhysGraph 没有 AHA-WAM 的异步慢 planner/快 executor，也没有 stale cached context 问题。因此完整照搬 OVCR 的理论动机不完全成立，代价和风险较高。

更合理的顺序是：

1. Phase A 建立低成本强基线。
2. Phase B 证明动态 contact geometry 有效。
3. Phase C 验证 cross-attention 的 token-level 区域选择。
4. 只有前三步显示单层 adapter 容量不足时，才考虑 Phase D 的 layer-wise K/V routing。

## 6. 推荐 Pipeline 的训练与冻结关系

推荐的参数状态：

| 模块 | 参数状态 |
|---|---|
| Stage 1 tokenizers | Frozen |
| Stage 1 PhysGraph layers | Frozen |
| Stage 1 Base Actor | Frozen |
| Base logstd | 不参与 Stage 2 合成或保持 Frozen |
| Region Encoder/Tokenizer | Trainable |
| Dynamic Contact Encoder | Trainable |
| Cross-Attention/Residual Adapter | Trainable |
| Residual Wrist Head | Trainable |
| Stage 2 Value Head | Trainable |
| Final wrist logstd | Trainable或保持当前固定初始化策略 |

最终 actor：

```text
μ_final = stopgrad(μ_base) + s × M ⊙ Δμθ
```

这样 PPO actor gradient 只流入 contact/residual 模块。

初始化要求：

- Residual Head 最后一层零初始化；或
- residual gate `α=0`；最好同时避免多个非零入口。
- 初始化后进行单元检查：

```text
max_abs(μ_final - μ_base) == 0
```

非 mask 维度始终检查：

```text
max_abs(μ_final[:,6:56] - μ_base[:,6:56]) == 0
```

## 7. 为什么推荐显式输入 `μ_base`

当前 residual：

```text
Δμ = Rθ(obs, contact)
```

推荐 residual：

```text
Δμ = Rθ(z_base, μ_base[0:6], contact, dynamic_contact)
```

显式输入 `μ_base[0:6]` 的意义：

- 网络知道 Base Policy 当前准备向哪个方向平移和旋转。
- 网络可以学习“在现有动作上增加、减小或反向修正多少”。
- 更符合 residual control 的语义。
- 避免 residual branch 重新根据同一观测推断 Base Policy 的行为。

但仅输入 `μ_base` 不足：

- 相同基础动作在不同手指接触状态下可能需要不同修正。
- 精细接触仍依赖当前状态和动态区域几何。
- 所以 `z_base`/current state 和 dynamic contact features 必须保留。

## 8. 关键设计选择与待决策项

后续讨论需要逐项决定：

### 8.1 Residual 输入

候选：

1. `[z_base, contact_z]`
2. `[z_base, μ_base[0:6], contact_z]`，当前推荐的 Phase A。
3. `[z_base, μ_base[0:6], contact_z, dynamic_contact]`，当前推荐的 Phase B。
4. Stage 1 layerwise tokens + region tokens，Phase C/D。

### 8.2 Contact 表示

候选：

- 保留当前 258D BPS → 单个 `contact_z`。
- 128 BPS anchor tokens。
- 池化为 16–32 个 region tokens。
- 红/黑点云 point tokens。
- 直接使用每个 fingertip/contact-body 到红黑区域的动态关系。

当前建议：Phase A 保留现有 BPS encoder以减少变量；Phase B/C 再引入动态、多 token 表示。

### 8.3 Residual 输出维度

候选：

- 输出 56 维再 mask；
- 直接输出 6 维 wrist residual。

当前建议：直接输出 6 维，更符合可控范围，也减少无效参数。

### 8.4 是否修正手指动作

当前仅修正 wrist 6 维。如果接触错误主要来自手指关节姿态，cross-attention 再强也受 action mask 限制。

是否扩展到右手手指必须作为独立消融，不能与 backbone/contact 表示同时变化，否则无法判断收益来源。

### 8.5 Critic 输入与共享

候选：

- Actor/Critic 共用 contact adapter。
- Actor 使用轻量 contact adapter，Critic 使用独立 value features。
- Critic 使用更多 privileged dynamic contact features，但 actor 保持部署可用观测。

需要特别防止 critic-only privileged 信息泄漏到 actor。

### 8.6 Cross-attention 层数

候选：

- 1 个 late cross-attention block；
- 2 个 adapter blocks；
- 4 层 layer-wise routing。

当前建议从 1 层开始，先控制计算量和 PPO 稳定性。

## 9. 建议实验与消融

至少比较：

1. `Current`: 当前双 PhysGraph + late concat MLP。
2. `Reuse-Z`: Frozen `z_base` + contact MLP + residual head。
3. `Reuse-Z+A`: `z_base + μ_base[0:6] + contact_z`。
4. `Dynamic`: 在 3 的基础上加入动态 fingertip-region geometry。
5. `CrossAttn`: 多 region token + single cross-attention adapter。
6. `LayerWise`: 仅在前述方案仍不足时尝试。

控制变量：

- 相同 Stage 1 checkpoint。
- 相同 contact reward。
- 相同 action mask。
- 相同 residual scale。
- 相同 PPO 预算、随机种子和环境数量。
- 尽量控制 trainable parameter 数量或至少报告差异。

主要指标：

- `rh_contact_good_ratio`
- `rh_contact_bad_ratio`
- `rh_contact_active_ratio`
- `rh_contact_red_dist`
- `rh_contact_black_dist`
- `rh_contact_region_reward`
- `rh_contact_nonregion_penalty`
- `rh_contact_residual_translation_norm`
- `rh_contact_residual_rotation_norm`
- imitation success/failure rate
- object pose/rotation error
- fingertip/joint tracking error
- action clipping比例
- wrist residual smoothness
- 推理耗时和显存

必须增加的 sanity checks：

- 初始化时 `μ_final == μ_base`。
- 非 mask 维度均值始终完全等于 Base mean。
- shuffle red/black label 后性能应下降，否则网络可能没有使用语义区域。
- 将 contact 输入置零或替换为其他工具区域后性能应有可解释变化。
- 记录 residual gate、attention map 和 residual norm，检查是否塌缩或饱和。

## 10. 代码位置索引

后续讨论/实现应优先检查：

- `lib/rl/network_builder_transformer_bih_graph_improve_correct.py`
  - `CustomMultiheadAttention`
  - `TransformerLayer`
  - `PhysicallyGroundedBias`
  - `MLP_new`
  - `MLP_action`
  - `MLP_value`
  - `LATENT_ENCODER`
  - `BimanualHandToolPolicy`
- `lib/rl/res_models.py`
  - 最终高斯分布、采样和 `base_actions` 输出
- `lib/rl/runner.py`
  - Stage 2 checkpoint restore 逻辑
- `lib/rl/agent.py`
  - PPO actor/critic/bounds loss
- `lib/rl/base.py`
  - rollout、contact metrics、Stage 2 早停
- `physgraph_envs/lib/envs/tasks/dexhandmanip_bih.py`
  - observation
  - BPS
  - contact reward
  - action execution
- `main/dataset/oakink2_dataset_utils.py`
  - 红黑顶点颜色解析和固定采样
- `main/cfg/rl_train/ResDexHandPPO.yaml`
  - contact residual scale、mask、logstd、PPO
- `main/cfg/task/ResDexHand.yaml`
  - contact reward 阈值和动作配置
- `main/cfg/config.yaml`
  - Stage 2 训练预算和早停阈值

参考材料：

- `AHA.pdf`
  - Layer-wise Joint Attention：公式 3。
  - Observation-Guided Video-Context Routing：公式 8–10。
- `crossAttn.png`
  - 图中 cross-attention 后 residual update 的局部示意。

## 11. 当前推荐结论

当前最推荐先实现和验证的不是完整 AHA-WAM layer-wise K/V routing，而是：

```text
Frozen Stage-1 PhysGraph（一次前向）
    ├→ z_base
    └→ μ_base

contact_refine → Contact Encoder → contact_z

[stopgrad(z_base), stopgrad(μ_base[0:6]), contact_z]
    → small Residual Adapter
    → Δμ_wrist [B,6]

μ_final[0:6]  = μ_base[0:6] + 0.25 × Δμ_wrist
μ_final[6:56] = μ_base[6:56]
```

随后再逐步增加：

1. 动态 fingertip/contact-body 到红黑区域的几何关系；
2. 多 region token；
3. 单层 cross-attention + zero-init gated residual；
4. 必要时才研究 layer-wise routing。

该顺序能够把收益来源拆开验证，并避免在 PPO 中同时引入“新输入、新 tokenization、新 attention、layer-wise 双流和 action mask 扩展”等过多变量。

## 12. 尚未执行的工作

截至本文创建时：

- 没有修改 Stage 2 网络结构。
- 没有让 Base Policy 返回 `z_base` 或 layerwise tokens。
- 没有将 `μ_base` 输入 residual branch。
- 没有增加 dynamic contact observation。
- 没有加入 cross-attention。
- 没有运行新方案实验。

后续只有在用户明确要求实现后，才应开始修改代码。
