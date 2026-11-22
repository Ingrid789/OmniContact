# OmniContact 原版与 V5 的完整改动对照：Unitree G1 搬箱

原版 OmniContact 将搬箱分成两层。CF-Gen 根据机器人、箱子和目标桌的位置规划接触流，CF-Track 每 20 ms 根据参考和观测给 G1 的 29 个关节发指令。**V5 沿用官方 CF-Gen、搬箱示范、G1 模型和关节控制接口，重新训练低层 CF-Track。** 原版 actor 通过 Isaac Lab 中实际执行过的动作学习；V5 先让可微世界模型（WM）从 MuJoCo 物理交互学会预测，再让 actor 在 WM 里连续尝试 32 步，通过未来搬箱回报的梯度更新动作策略。

全文以官方训练任务 `OmniContact` 的搬箱 MLP 为基线，并对照官方 sim2sim 的 CF-Gen 和推理接口。原项目还提供其他物体任务、AMP 和 Transformer 配置。本分支的 V5 训练入口为 `carrywm.train_v5`，覆盖 G1 搬箱。原训练任务仍可在仓库中找到。[官方任务注册](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/__init__.py) · [官方训练配置](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/agents/rsl_rl_ppo_cfg.py) · [V5 入口](carrywm/train_v5.py)

## 1. 整个搬箱系统改了哪些部分？

| 模块 | 原版 OmniContact 搬箱 MLP | V5 搬箱 MLP |
|---|---|---|
| 参考从哪里来 | 训练从示范提取接触流；官方 sim2sim 可在执行时调用 CF-Gen 生成参考 | 训练主回合按 90% 示范、10% 官方 CF-Gen 选源。两路参考均包含接触流和全身姿态 |
| 策略看到什么 | 11 帧 × 49 维接触流，加 5 帧 × 141 维观测，合计 1244 维 | 11 帧 × 113 维参考先编码成 32 维，再与 10 帧 × 141 维观测拼成 1442 维 decoder 输入 |
| 谁产生下一状态 | Isaac Lab 并行执行动作，物理引擎给出下一状态及奖励 | 1024 个 MuJoCo Warp 环境提供监督数据；训练 actor 时由新增的 WM 预测机器人、自由箱子及接触 |
| 一轮采多少 | 官方默认 4096 个物理环境各采 24 步，共 98304 个物理转移 | 1024 个物理环境各采 5 步，共 5120 个新转移；另在 WM 中展开 2048 × 32 步 |
| Actor 怎么更新 | 对真实 rollout 算优势，用新旧动作概率比和 PPO 裁剪目标更新 | 闭环展开中冻结 WM 和 critic，沿后续状态、奖励以及存活权重反传，更新 actor 一次 |
| Critic 学什么 | 在真实 PPO rollout 上拟合回报，为优势估计提供基线 | 在想象 rollout 上拟合带软终止的 λ-return，并给 actor 提供有上界的末端价值 |
| 怎样处理失败 | 超时、躯干锚点高度、指定末端高度或箱高跟踪偏差可触发物理终止 | 物理环境记录跌倒、掉箱、稳定放置和超时；想象训练单独计算骨盆软终止概率，降低失衡后轨迹的回报权重 |
| 执行策略 | 官方 sim2sim 使用其 MLP 权重、观测适配器和 PD 控制 | V5 使用新的参考与历史适配器，导出 TorchScript actor。WM 和 critic 仅在训练时使用 |

两边一次控制都输出 29 个关节指令。V5 的 32 步是训练时把同一个 actor 连续调用 32 次，每次都读取刚预测出来的新状态。上表中的物理转移数描述各自的一轮预算；衡量训练提速要比较在相同搬箱评估下达到同一成功率所需的时间。[官方物理环境](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/env_cfg.py) · [官方 PPO](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/agents/rsl_rl_ppo_cfg.py) · [V5 物理采集](carrywm/collector_v4.py) · [V5 策略更新](carrywm/v3.py)

## 2. 同样在第 100 帧，为什么两套 actor 看到的输入不同？

**原版与 V5 看同一个第 100 帧。** 原版在 11 个时刻读取五个跟踪点的位姿和四位接触，组成 539 维参考，同时读取第 96 至 100 帧共 705 维实际观测。V5 保留这 11 个时刻，增加骨盆和全身关节参考，并将历史延伸到第 91 帧。官方将位姿序列和接触序列分别展平；V5 按帧打包，并统一为 MuJoCo 的关节顺序。跟踪点从官方橡胶手和踝 roll 连杆换成手掌和踝 pitch 连杆，使示范与 CF-Gen 输出能落在同一组连杆上。[官方参考构造](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/mdp/observations.py) · [V5 参考适配器](carrywm/data.py)

回合开始时，90% 的主回合读取一条搬箱 NPZ 示范，10% 用官方 CF-Gen 根据当前骨盆、箱体位姿、箱体尺寸和目标位置做运动学规划。V5 的生成目标水平位置扰动标准差为 **0 m**。回合内通过同一路参考训练，掉箱恢复单独计数。[参考读取与生成](carrywm/data.py) · [物理重置与恢复](carrywm/sim.py)

1. 在第 100 个控制时刻，读取第 **100、101、102、103、104、108、112、116、124、132、150** 帧。这个窗口最远看未来 1 秒。每帧含下表的 113 个数。
2. Encoder 将 `11 × 113 = 1243` 个数压缩为 32 维 $z_{100}$。Decoder 把它与第 **91 至 100 帧**的实际观测拼接，计算当步的 29 个关节指令。
3. 物理采集时给动作加入标准差 **0.1** 的高斯探索噪声，按默认角度和各关节缩放得到 PD 目标，依次执行四个 5 ms 仿真子步。到第 101 帧，重新计算参考和动作。想象训练时由 WM 预测下一状态，actor 用新状态继续决策。评估使用确定性动作。

| 单帧参考 | 维数 | 具体取值 |
|---|---:|---|
| 接触流 | 49 | 左右手掌、躯干及左右踝的相对位置和 6D 朝向共 45 维，外加左脚、右脚、左手、右手对箱体的四位期望接触。位姿在当前躯干航向坐标系中表达。 |
| 全身参考 | 64 | 29 个参考关节角、29 个参考关节速度、6 维参考骨盆相对当前骨盆的朝向。NPZ 直接读取关节速度，CF-Gen 对连续 50 Hz 的关节角序列差分，再取稀疏帧。 |

每帧 141 维观测包括五个身体测点位置（15）、根部角速度与投影重力（6）、关节角与速度（58）、上一动作（29）、箱子相对位置与朝向（9）及八个箱角（24）。期望接触告诉策略什么时候应该抓箱，观测历史里的箱子实际运动则让策略判断动作执行到了哪里。[观测与坐标变换](carrywm/model.py)

## 3. 从单个策略 MLP 到参考编码器、动作解码器和可微 WM

**原版**的 actor 和 critic 都用 `512 → 256 → 128` 的 ELU 隐藏层。Actor 把参考和历史一起输入 MLP，输出高斯策略的关节动作均值；critic 还可读取仿真器提供的特权观测。**V5** 把 actor 拆成参考 encoder 和动作 decoder，加入单独学习物理转移的网络，并让 critic 读取 WM 可以提供的状态、任务参考和物理参数。三个网络从头训练；32 维编码由搬箱回报端到端学出。[官方网络设置](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/agents/rsl_rl_ppo_cfg.py) · [V5 网络](carrywm/model.py)

| 模块 | 默认网络与输出 |
|---|---|
| Actor encoder | `1243 → 2048 → 1024 → 512 → 256 → 32 → tanh`，用搬箱回报端到端学习参考编码。 |
| Actor decoder | `1442 → 2048 → 1024 → 512 → 256 → 29`，输入为 32 维编码和展平的 `10 × 141` 维观测。一次前向输出一步控制。 |
| WM | `129 → 2048 → 2048 → 2048` 共用主干。三个输出头分别预测机器人 35 维速度增量、箱子 6 维速度增量和 6 个接触 logits。 |
| Critic | `1343 → 512 → 256 → 1`。用当前人箱状态（84）、参考窗口（1243）和场景参数（16）估计折扣回报。 |

WM 的输入 129 维由当前状态（84）、实际执行的动作（29）、物理参数（16）拼成。状态包含机器人根部、29 个关节和箱子的位姿与速度。参数包括箱子半尺寸、质量、主惯量、摩擦，以及两张桌子的位置。WM 将速度增量积分为新的位姿，再由 FK 求得身体位置和朝向。箱子有独立的运动状态，即使双手松开，模型也能表达箱子继续下落。六个接触输出分别监督四个肢体与箱子的接触、两脚的支撑。[网络、积分与损失](carrywm/model.py)

## 4. 从 Isaac Lab 的 PPO rollout 到 MuJoCo 采集和 WM 内策略更新

**原版的一轮**先用当前策略在 Isaac Lab 收一批动作、奖励和终止标记，用 GAE 计算优势，再将数据分 4 批训练 5 遍。PPO 裁剪概率比（参数 0.2），含熵奖励，初始学习率为 0.001。**V5 的一轮**增加了可跨轮复用的物理回放池。WM 学的是实际执行动作后的状态误差；actor 学的是改变当步动作将怎样改变后续回报；critic 学的是从想象状态出发的剩余回报。三个网络各用一个 Adam，学习率为 0.0002。[官方优化参数](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/agents/rsl_rl_ppo_cfg.py) · [V5 训练损失](carrywm/train.py)

| 环节 | 数据和更新方式 |
|---|---|
| 预热 | 初始策略带动作探索在 MuJoCo Warp 中采集至少 **8192** 条真实转移。回放池需要出现同一环境、同一回合内连续 8 步的窗口。预热阶段只积累数据和归一化统计。 |
| 物理采样 | 1024 个独立环境各走 5 个控制步，得到 **5120** 条新物理转移。终止步保留实际下一状态，各环境按自身进度重置。 |
| WM 更新 | 独立抽 10 批，每批 1024 个连续 8 步窗口。从实测初态出发，依次输入实采动作，连续预测 8 步。将预测状态和接触与物理真值逐步比较，每批更新一次。 |
| Actor 更新 | 从回放抽 2048 个起点，在冻结参数的 WM 中各做 **32 步**闭环决策。损失沿 0.64 s 的状态和动作链反传，actor 更新一次。默认在反向计算时重算 encoder 激活。 |
| Critic 更新 | 复用 actor 展开的轨迹及本轮固定的价值目标。每遍打乱硬边界内的有效时间步，分 4 批，训练 5 遍，共 **20 次**更新。满长度一批为 16384 个时间步。 |

replay ratio 衡量 WM 每轮抽用了多少转移步。V5 每轮抽用 $10 \times 1024 \times 8 = 81920$ 步，除以当轮新增的 5120 步，正好为 **16**。训练时钟在预热后每个并行控制步累计一轮更新的 0.2，第 5 步触发完整更新。WM 的状态损失按各物理量的尺度计算位置、速度和旋转误差，接触头以物理标签计算二元交叉熵。[并行训练入口](carrywm/train_v4.py) · [按环境分段的回放](carrywm/replay_v4.py) · [Actor 和 critic 的批次更新](carrywm/v3.py)

物理条件也随采集器改变。官方 Isaac Lab 训练配置包含摩擦、关节增益、质心、外推力与观测噪声等随机化。V5 从片段读取箱子尺寸和两张桌子的位置，在 Warp 中设置碰撞几何和惯量，并给实际执行的动作加噪声。比较泛化性时，需要明确评估所用的扰动范围和物理后端。[官方环境事件](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/env_cfg.py#L284-L385) · [V5 场景参数](carrywm/warp_sim.py)

例如第 100 帧在桌边抬箱，监督 WM 时给它第 100 帧的实测初态和连续八个**实际执行的动作**，将预测的第 101 至 108 帧逐一与 MuJoCo 真值比较。训练 actor 时则由策略在每个预测状态上**重新选动作**，实采动作不再规定它下一步应该怎么做。姿态通过可微积分和 G1 正运动学得到，因而第 102 帧拿到的奖励能影响第 100 帧的策略参数。[WM 监督与想象展开](carrywm/train.py)

## 5. 从物理奖励和硬终止，到可微奖励与软终止

原版用 Isaac Lab 中的身体、箱体和接触传感器读数计算跟踪奖励。V5 沿用这些任务目标和权重，把奖励重写为可在预测状态上求导的 PyTorch 运算。两种实现里同名的指标，其测量方式也可能不同。[官方奖励配置](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/env_cfg.py#L388-L487) · [官方奖励实现](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/mdp/rewards.py) · [V5 奖励实现](carrywm/model.py)

| 搬箱信号 | V5 相对原版怎样实现 |
|---|---|
| 躯干、全身、双手和箱子跟踪 | 沿用位置、朝向、手箱相对位姿等奖励。用预测人箱状态和可微 FK 求这些量，在每个想象步和参考对齐 |
| 手脚接触 | 原版使用 Isaac Lab 过滤后的接触传感器。V5 按 MuJoCo 碰撞对汇总接触力，用四个肢体与箱子接触、两脚受力共六个标签监督 WM；想象时用预测接触概率计算接触匹配及脚滑项 |
| 动作平滑、力矩与脚滑 | 保留动作变化、关节限位等扣分。想象中的手臂力矩按 PD 近似，脚的平移速度由相邻两步 FK 位置差分求得 |
| 回合结束 | 原版在物理 rollout 里处理超时和跟踪失败；其躯干锚点、指定末端及箱体高度偏差阈值均为 0.5 m。V5 区分用于实际重置的物理事件和用于想象回报的连续存活权重。想象中的掉箱还会产生单独惩罚 |

每次 WM 算完下一状态，V5 取**同一时刻**的预测骨盆和参考骨盆。高度误差 $e_h$ 以米计。倾斜误差 $e_g$ 是重力在两个骨盆坐标系中的 z 分量之差的绝对值，绕竖直轴转身不会改变这个分量。对于单位四元数 $q=(w,x,y,z)$：

$$
e_h=|z_{\rm pred}-z_{\rm ref}|,\quad g_z(q)=2(x^2+y^2)-1,\quad
e_g=|g_z(q_{\rm pred})-g_z(q_{\rm ref})|,\quad
p_t=\sigma\!\left(\frac{\max(e_h/0.25,\ e_g/0.8)-1}{0.05}\right).
$$

**例子。** 如果第 100 步执行后骨盆比对应参考低 25 cm，倾斜误差很小，这步的终止概率为 0.5。把这一步作为展开起点，当前动作产生的奖励仍取完整权重，下一步奖励的存活权重降到 0.5。如果下一步的 $p$ 仍为 0.5，再下一步的存活权重就降为 0.25。时间折扣另外计算。Actor 因而能从后续回报的变化得到关于当前姿态的梯度。[SuperMimic 终止公式](https://github.com/HybridRobotics/supermimic/blob/abc124c/src/supermimic/envs/world_model_env.py#L200-L219) · [V5 实现](carrywm/v5.py)

**Actor 损失。** 常规步取 $c_t=1-p_t$，确定掉箱、箱高严重偏离或走到参考末帧时取 $c_t=0$。从 $S_0=1$ 开始累乘 $S_{t+1}=S_tc_t$，设 $V_t$ 是 critic 对第 $t$ 步的估值，则

$$
L_\pi=-\mathbb E\!\left[\sum_{t=0}^{31}\gamma^t S_t r_t+
\gamma^{32} S_{32}\min(V_{32},V_{\max})\right],\qquad \gamma=0.99.
$$

**Critic 损失。** 从 $G_{32}=V_{32}$ 出发，由第 31 步反推至第 0 步，$\lambda=0.95$。当步的 $c_t$ 控制未来价值占比。完成整段递推后才对监督目标施加上界，使用有效帧上的均方误差更新 critic。

$$
G_t=r_t+\gamma c_t\big[(1-\lambda)V_{t+1}+\lambda G_{t+1}\big],\qquad
L_V=\mathbb E\!\left[(V_t-\operatorname{stopgrad}(\min(G_t,V_{\max})))^2\right].
$$

**为什么上限是 10.8？** 躯干与身体跟踪、手部与箱体跟踪、手箱相对位姿及接触匹配的正奖励权重合计为 5.4，每项匹配因子至多为 1。动作变化、力矩、关节限位、脚滑和掉箱均可扣分。20 ms 内正奖励至多为 $5.4\times0.02=0.108$，因而 $V_{\max}=0.108/(1-0.99)=10.8$。Actor 的末端 bootstrap 和 critic 的最终监督目标只做上截断，负回报保留。[奖励项与权重](carrywm/model.py) · [SuperMimic 策略与价值更新](https://github.com/HybridRobotics/supermimic/blob/abc124c/src/supermimic/learners/policy_learner.py#L98-L198)

## 6. 从跟踪参考到验证搬箱结果

官方配置的终止重点是身体和物体是否偏离参考。V5 的物理评估另外检查完整的任务事件链，包括抓离支撑面、搬至目标、松手及稳定放置。物理跌倒按骨盆的绝对高度与倾斜判定；前节的软终止用骨盆**相对参考**的误差，服务于 WM 内的策略学习。[官方终止条件](https://github.com/Ingrid789/OmniContact/blob/19cde38364a5b48e47348b270c895eabe2ba3ad9/source/omnicontact/omnicontact/tasks/omnicontact/env_cfg.py#L489-L516) · [V5 物理事件](carrywm/sim.py)

MuJoCo 采集和验证使用实际仿真状态。骨盆高度低于 0.3 m 或骨盆旋转矩阵的 zz 分量低于 0.35 时记录跌倒。搬箱成功要求箱子曾由手抓住并离开支撑面，随后在目标支撑面上松手，距目标小于 0.2 m，线速度小于 0.1 m/s、角速度小于 0.2 rad/s，并连续稳定 5 步。实际掉箱结束原搬运段，符合条件时从当前人箱位姿重新做一次 IK 规划，恢复段独立记账。[物理终止与恢复](carrywm/sim.py)

评估关闭动作探索，分别统计示范和生成参考下的成功、跌倒、掉箱和超时。先比较同一批场景上的**成功数／总回合数**，再比较达到相同成功率所用的真实交互数和总用时。日志里的 `actor/alive32` 是 WM 想象至第 32 步时的平均存活权重。若此值上升而物理跌倒依旧，值得同时检查 WM 使用实采动作开环预测的 1、8、32 步骨盆和箱体误差。要比较原版与 V5，应固定场景、参考来源、控制频率及成功判据，并报告两套策略在相同评估条件下的结果。V3 的中间检查点和 V5 的想象回报都无法给出这一对比的结论。[物理验证](carrywm/train.py) · [动力学误差](carrywm/validation.py) · [V5 回归测试](tests/test_v5.py)

### 在仓库里去哪里看这些改动

| 阅读顺序 | V5 实现 |
|---|---|
| 参考和物理环境 | [数据划分、参考字段、CF-Gen 接口](carrywm/data.py) → [官方场景、PD、重置及掉箱恢复](carrywm/sim.py) → [批量 Warp 物理与接触](carrywm/warp_sim.py) |
| 从物理采样到学习 | [批量 actor 采集](carrywm/collector_v4.py) → [按环境和回合保存的回放池](carrywm/replay_v4.py) → [预热与 replay ratio 调度](carrywm/train_v4.py) |
| 从输入到梯度 | [网络、损失与可微 FK](carrywm/model.py) → [32 步展开与回报](carrywm/train.py) → [actor 和 critic 的更新顺序](carrywm/v3.py) → [V5 软终止与价值上限](carrywm/v5.py) |
| 训练、验证与部署 | [V5 入口](carrywm/train_v5.py) · [物理轨迹上的动力学验证](carrywm/validation.py) · [策略导出](carrywm/export.py) |

此前的 [V2 设计](docs/carry-wm-v2-design.md)、[V3 改动记录](docs/omnicontact-to-v3.md) 和 [V4 并行采集说明](docs/carry-wm-v4.md) 保留了各阶段实验配置。

## 7. 复现 V5

在本分支仓库根目录操作。正式训练使用带 NVIDIA GPU 的 Linux 环境，训练栈包含 Python 3.11、CUDA 版 PyTorch 2.7、NumPy、SciPy、PyYAML，以及下列与 SuperMimic 对齐的物理后端。准备官方 sim2sim 仓库中的 G1 XML、网格和 CF-Gen，并从数据集下载 `npz_clips/box/case1_carry`。数据按原始动作分组，同一动作的切片及尺寸变体归入同一分区。已核对的 623 个片段分为训练 512 个、验证 53 个和测试 58 个。[官方 sim2sim 修订](https://github.com/Ingrid789/OmniContact_sim2sim/tree/1cf9ddd4067cbe5710b1f475e96f9c69f88055e2) · [官方搬箱数据](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset) · [SuperMimic 环境版本](https://github.com/HybridRobotics/supermimic/blob/abc124c/uv.lock)

在上述 Python 环境中执行，替换两个输入路径，并为新实验指定独立的输出目录。

```bash
python -m pip install 'mujoco==3.10.0' 'mujoco-warp==3.10.0.1' 'warp-lang==1.14.0'
export CARRY_SIM_ROOT=/path/to/OmniContact_sim2sim
export CARRY_MOTION_DIR=/path/to/OmniContact-Dataset/npz_clips/box/case1_carry
export V5_RUN=outputs/carry-v5-seed42

python -m unittest discover -s tests -p 'test_v5.py'
CARRY_WARP_DEVICE=cuda:0 python -m unittest discover -s tests -p 'test_v4_warp.py'

python -m carrywm.monitor --gpu 0 --output-dir "$V5_RUN" --interval 2 -- \
  python -u -m carrywm.train_v5 --device cuda:0 --precision bf16 \
  --sim-root "$CARRY_SIM_ROOT" --motion-dir "$CARRY_MOTION_DIR" --output-dir "$V5_RUN" \
  --iterations 6000 --seed 42 --save-every 50 --log-every 5 --eval-every 250 --eval-episodes 3
```

| 输出 | 读法 |
|---|---|
| `config.json`、`metrics.jsonl` | 核对训练版本、终止参数、预热窗口、每轮真实转移数、replay ratio、优化器更新次数和各阶段耗时。上述命令每 250 轮分别验证 3 个示范与 3 个生成参考主回合。 |
| `gpu-*.csv`、`gpu-*.json` | 分别记录显存采样与汇总，包括训练进程占用、整卡占用、整卡忙碌率和本次启动经过时间。 |
| `latest.pt` | 包含模型、优化器、回放及随机状态。续训时在同一条 V5 命令末尾加 `--resume "$V5_RUN/latest.pt"`。恢复后新开物理回合并补足连续窗口。 |
| `summary.json` | 完成指定训练预算后写入，总结轮数、真实转移、累计墙钟时间和 PyTorch 显存峰值。 |

导出训练好的 actor 可执行 `python -m carrywm.export "$V5_RUN/latest.pt" "$V5_RUN/actor.pt"`。导出的 TorchScript actor 接收 `[B,11,113]` 参考和 `[B,10,141]` 历史，返回 `[B,29]` 归一化 PD 指令。用于其他控制端时应保持关节顺序、坐标系和动作缩放一致。[导出与一致性核验](carrywm/export.py)

## 上游项目

搬箱场景、接触流生成器和数据来自 Runyi Yu 等人的 *OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation*。原项目还包含多任务 MLP、AMP 和 Transformer 配方及预训练策略。[论文](https://arxiv.org/abs/2606.26201) · [项目主页](https://omnicontact.github.io/) · [原始训练仓库](https://github.com/Ingrid789/OmniContact) · [预训练权重](https://drive.google.com/drive/folders/1hN6eStd3jRfQTDbMaXl4DrwxZwcdA1Jk?usp=drive_link) · [策略演示](https://omnicontact.github.io/policy-viewer.html?v=policy-hide-push-ghostbox-20260604a)

可微 WM、闭环 BPTT、存活加权及价值上限参考 *SuperMimic* 的实现。[论文](https://arxiv.org/abs/2610.28480) · [官方代码](https://github.com/HybridRobotics/supermimic/tree/abc124c)
