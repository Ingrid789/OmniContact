# V5 搬箱训练：软终止与 critic 价值上限

V3 第 1650 轮的确定性验证中，10 个演示参考回合和 10 个 CF-Gen 参考回合均以机器人跌倒结束。V4 的想象回合用 0.5 m 的骨盆高度偏差作为硬阈值，未比较参考与预测的骨盆倾斜。V5 沿用 [V4 的采集和训练配置](carry-wm-v4.md)，修改 32 步想象回报和 critic 目标。该改动的检验指标是同一留出集、相同种子下的物理跌倒率和搬箱成功率。

## 想象中的一步如何计算

执行动作后，取 WM 预测的下一状态和对应下一参考帧。演示与 CF-Gen 都提供世界坐标系的骨盆高度和单位四元数。用骨盆坐标系中重力方向的 z 分量度量倾斜。对于 wxyz 四元数，单独改变偏航角不会改变这个分量。

~~~text
g_z(q) = 2(q_x² + q_y²) − 1
e_h    = |z_pred − z_ref|                       （米）
e_tilt = |g_z(q_pred) − g_z(q_ref)|              （无量纲）
p      = sigmoid((max(e_h / 0.25, e_tilt / 0.8) − 1) / 0.05)
~~~

沿用 SuperMimic 的两个阈值和 sigmoid 宽度。举例来说，骨盆高度差为 25 cm、倾斜误差为零时，p 为 0.5。V5 保留动作当步的奖励。若下一步终止概率也为 0.5，前两步之后的奖励只剩 0.25 的累计存活权重，另乘时间折扣。物理采集与验证继续按 V4 的摔倒、掉箱、超时及放置成功条件重置。软概率只作用于学习用的想象轨迹。[原版终止实现](https://github.com/HybridRobotics/supermimic/blob/abc124c/src/supermimic/envs/world_model_env.py#L200-L219) · [论文附录](https://arxiv.org/abs/2610.28480)

| 使用位置 | V5 的计算 |
|---|---|
| Actor 的 32 步回报 | 令 S₀=1，先加 γᵗ Sₜ rₜ，再更新 Sₜ₊₁=Sₜ cₜ。常规步 cₜ=1−pₜ，确定掉箱、箱子严重偏离或参考结束时 cₜ=0。梯度可以从存活权重经 WM 的预测状态回到 actor。 |
| Actor 的末端估值 | 末项为 γ³² S₃₂ min(V(s₃₂), V_max)。日志中 `actor/alive32` 表示这批想象轨迹的平均 S₃₂。 |
| Critic 的 λ-return | 使用 γ=0.99、λ=0.95 和当步的 cₜ 向后递推。递推结束后截断每个监督目标的上界。每个硬边界之前的有效时间步参与拟合，包括终止动作当步。 |
| 价值上限 | 搬箱任务所有正奖励项的权重和为 5.4，控制周期为 0.02 s。单步奖励上界为 0.108，理论折扣回报上界为 0.108/(1−0.99)=10.8。代码从奖励权重计算这个值，保留负回报。 |

Actor 在冻结的 WM 和 critic 中求导，critic 使用已分离的样本和固定的 λ-return 更新。[原版 actor、bootstrap 和 critic 更新](https://github.com/HybridRobotics/supermimic/blob/abc124c/src/supermimic/learners/policy_learner.py#L98-L198)

## 仍需用实验区分的差异

* **重置动作帧。** SuperMimic 可按各动作时间段的回报采样起始帧。V5 的主回合从选中搬箱片段的第 0 帧开始。如果机器人在前段经常跌倒，后段持箱和放置的真实样本会偏少。要做中段重置，需要同步机器人、箱子、速度和支撑状态。[原版重置采样](https://github.com/HybridRobotics/supermimic/blob/abc124c/src/supermimic/runners/reset_sampling.py)
* **WM 根部预测。** 上游与本项目的根部监督误差尺度及损失分组不同，本项目还监督箱子和接触。在上述 V3 验证的可用 32 步窗口中，以实际动作为输入开环预测，根部三维位置平均误差为 0.341 m，完整姿态误差为 0.484 rad。这两个量与软终止使用的高度差和重力投影差分别统计。[原版 WM 损失](https://github.com/HybridRobotics/supermimic/blob/abc124c/src/supermimic/learners/wm_learner.py)
* **优化与观测。** 上游将部分隐藏层权重交给 Muon，其他参数使用 Adam，并对 actor 施加逐元素梯度裁剪。在真实与想象的本体感觉观测上也有对应的噪声机制。V5 继续沿用 V4 的优化器、动作探索和无噪声 CF-Gen 参考配置。[原版优化器](https://github.com/HybridRobotics/supermimic/blob/abc124c/src/supermimic/learners/learner_utils.py) · [原版采集器](https://github.com/HybridRobotics/supermimic/blob/abc124c/src/supermimic/runners/supermimic_runner.py)

V5 的软阈值相对参考姿态，物理跌倒判据使用绝对姿态。参考直立且高度跟踪准确时，物理倾斜分支在骨盆旋转矩阵的 zz 元素低于 0.35 时触发。刚到此边界时重力投影差约 0.65，单步软终止概率约 2.3%。训练效果需要看实际姿态轨迹和成功率。

训练入口为 `python -m carrywm.train_v5`，使用 V4 的参数及独立输出目录。配置和检查点写入版本号 5 及终止参数，用 V5 入口恢复对应的 V5 检查点。
