# Stage-1 红黑接触 reward 与 relation bias

本方案只扩展现有 Stage-1，不引入 Stage2、残差动作或第二策略。完整路径为：MANO retarget →
几何轨迹修正 → Stage-1 imitation + semantic-contact 联合训练。

## 数据与观测

右手工具网格必须同时具有纯红和纯黑顶点。加载时分别抽取并确定性采样 512 个局部点，运行时按
OakInk2 物体缩放及 PhysX 当前物体位姿变换到世界坐标。每个环境、每个接触手指生成 8 维观测：

`[actual, expected, red_proximity, red_contact, black_proximity, black_violation, force_norm, curriculum]`

红接触优先：当手指满足红区有效接触时，即使位于颜色边界附近也不计为黑区违规。

此外始终生成独立的 5 维 `physical_contact` 观测，其值与上述 `actual` 通道使用同一接触力公式。
它只驱动 contact-link ↔ tool 的 dynamic physical edge，不受 `semanticContact.enabled` 控制；该开关
关闭时仍保留 dynamic physical edge，但不生成 8 维 semantic 观测、semantic AdaLN condition 或
semantic 指标。

## Reward

- `actual = sigmoid((F - 1.0) / 0.25)`
- `expected = sigmoid((0.03 - d_demo) / 0.003)`，其中 `d_demo` 来自几何修正后的 Stage-1 轨迹
- `red_quality = exp(-(d_red / 0.01)^2)`
- `raw = R_red - 2 P_black - 0.2 P_missing - 0.05 P_force`
- 当前实现：`reward = imitation_reward`，`curriculum * raw` 只记录为指标，不加入总 reward

`R_red`、`P_missing` 按 expected contact 归一化；`P_black`、`P_force` 按 active contact 归一化。

课程阶段（默认 320000 control steps）为：0–5% 纯 imitation；5–30% semantic 由 0 平滑升到
0.25；30–60% 由 0.25 平滑升到 1；60–100% 保持 1。因此前 30% 明确为 imitation-dominant。

## A1：Semantic AdaLN-Zero + geometric attention bias

原 spatial、edge、geometric 和 anatomical bias 均保留。dynamic physical contact edge 独立于
semantic 开关；geometric bias 只作用于双向的 RH contact-link ↔ tool 和 LH contact-link ↔ object
节点对，不再作用于 finger ↔ finger、左右手或 tool ↔ object。A1 不改变 geometric bias 原有的
head 分配，因此它仍作用于全部 attention heads。

Semantic 信息不再以 standalone bias、semantic-geometric fused bias 或 contact-link ↔ policy bias
加入 attention logits。每个 RH contact-link 使用六个保持独立、未求和的关系特征：

`[expected, expected*red_proximity, red_contact, black_proximity, black_violation, w]`

其中 `w` 为 curriculum。共享 condition encoder 将六维关系与 thumb/index/middle/ring/pinky 的
finger-ID embedding 编码为 64 维条件，并以 `w` 作为对应 contact token 的最终 AdaLN mask 强度。
因此 curriculum 为零时，即使 projection bias 在训练后非零，semantic 调制仍严格为零，且其他关系
不会被重复乘两次 curriculum。条件只散射到五个 RH contact-link token；policy、palm、tool、object、
LH 和其他 link 的 condition 与 mask 均为零。

四层 Transformer 中只有后两层在 attention 前使用 baseline-preserving AdaLN-Zero：

`u = (1 + w*0.1*tanh(delta_gamma)) * LayerNorm(x) + w*0.1*tanh(beta)`

`x = x + (1 + w*0.1*tanh(delta_gate)) * Attention(u)`

每层 modulation projection 的 weight 和 bias 均严格零初始化，因此初始时与原来的
`x + Attention(LayerNorm(x))` 完全一致。显式 token mask 在训练后仍阻止 projection bias 修改非接触
token。semantic 观测缺失时，两层自动退回普通 LayerNorm 路径。旧 semantic relation scale 仅作为
冻结的 state-dict 兼容参数保留，不再参与 forward；旧模型权重缺少 condition encoder/AdaLN 参数时，
严格加载逻辑会注入当前 condition encoder 参数和零初始化的 AdaLN projection；由于 projection 为零，
初始 forward 仍与旧模型一致。
