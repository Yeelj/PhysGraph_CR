# Stage-1 红黑接触 reward 与 relation bias

本方案只扩展现有 Stage-1，不引入 Stage2、残差动作或第二策略。完整路径为：MANO retarget →
几何轨迹修正 → Stage-1 imitation + semantic-contact 联合训练。

## 数据与观测

右手工具网格必须同时具有纯红和纯黑顶点。加载时分别抽取并确定性采样 512 个局部点，运行时按
OakInk2 物体缩放及 PhysX 当前物体位姿变换到世界坐标。每个环境、每个接触手指生成 8 维观测：

`[actual, expected, red_proximity, red_contact, black_proximity, black_violation, force_norm, curriculum]`

红接触优先：当手指满足红区有效接触时，即使位于颜色边界附近也不计为黑区违规。

## Reward

- `actual = sigmoid((F - 1.0) / 0.25)`
- `expected = sigmoid((0.03 - d_demo) / 0.003)`，其中 `d_demo` 来自几何修正后的 Stage-1 轨迹
- `red_quality = exp(-(d_red / 0.01)^2)`
- `raw = R_red - 2 P_black - 0.2 P_missing - 0.05 P_force`
- `reward = imitation_reward + curriculum * raw`

`R_red`、`P_missing` 按 expected contact 归一化；`P_black`、`P_force` 按 active contact 归一化。

课程阶段（默认 320000 control steps）为：0–5% 纯 imitation；5–30% semantic 由 0 平滑升到
0.25；30–60% 由 0.25 平滑升到 1；60–100% 保持 1。因此前 30% 明确为 imitation-dominant。

## 多关系 attention bias

原 edge 参数和拓扑、几何、解剖 bias 均保留，但 edge 构造从单一整数覆盖改为 batch-aware multi-hot：

- 基础关系：kinematic、physical contact、self；
- 原解剖关系：serial、synergy；
- 新语义关系：expected contact、red proximity、red contact、black proximity、black violation。

同一节点对可以同时具有多个关系。例如 contact-link ↔ tool 可以同时保留 physical、expected 和
red-contact，所有关系以 `einsum` 相加，不再使用 contact edge 覆盖原 edge。语义关系还同时注入
contact-link ↔ policy token。head 0–1 保持 serial，2–3 保持 synergy，4–5 服务红区语义，6–7
服务黑区语义。语义 bias 经 `2*tanh(...)` 限制在 `[-2, 2]`；红区 scale 从 `+1e-3`、黑区 scale
从 `-1e-3` 近零初始化，既保证旧 Stage-1 checkpoint 初始行为近似不变，也让语义关系从首个优化步
即可获得方向正确的梯度。physical relation 不乘 curriculum，始终保留；旧 edge embedding 和 scale
参数名不变，历史 checkpoint 可严格恢复。
