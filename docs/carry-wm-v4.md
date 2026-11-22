# V4 搬箱训练

V4 位于 `codex/v3-without-cfgen-noise` 分支。新入口 `carrywm.train_v4` 从零训练；同分支的 `carrywm.train_v3` 可复现原有的无噪声 V3。网络、奖励、数据划分和搬箱终止判据沿用 [V3 实现说明](omnicontact-to-v3.md)。

## 一轮如何运行

| 环节 | 实际执行 |
|---|---|
| 采样 | 在同一 GPU 上用 MuJoCo Warp 推进 1024 个独立物理环境。每个控制步为 20 ms，内部重新计算 4 次 5 ms 的 PD 力矩。每个环境独立采样参考、重置和处理掉箱恢复。 |
| 预热 | 初始 actor 使用标准差 0.1 的动作探索噪声。至少收集 8192 条真实转移，并检查回放里已存在同一环境、同一回合内的连续 8 步，在此期间只更新观测和动作统计量。 |
| 学习时钟 | 预热结束后，每个并行控制步新增 1024 条真实转移。调度每步累计 `1024 × 16`，到 `10 × 1024 × 8` 时触发一轮更新。等价于每步加 0.2，每 5 步更新一轮。 |
| WM | 从回放独立采样 10 批，每批 1024 个连续 8 步窗口，每批更新一次。每轮抽用 81920 个转移步，新增 5120 个真实转移，`replay ratio = 16`。 |
| Actor | 从回放采 2048 个起点，用 WM 各展开 32 步。冻结 WM 和 critic 的权重，沿展开轨迹反传并更新 actor 一次。默认一次处理 2048 条轨迹并重算 encoder 激活。 |
| Critic | 复用 actor 的展开结果，固定本轮更新前算出的 λ-return。每遍将有效时间步重新打乱，均分 4 批，重复 5 遍，共更新 20 次。每批最多 16384 个有效时间步。 |

训练的主回合仍以 1:9 的概率选择 CF-Gen 和演示参考。CF-Gen 的放置目标 XY 扰动标准差为 0 m。训练时的动作探索与这一开关独立。CPU MuJoCo 继续用于留出集的确定性物理评估，沿用 V3 的成功、跌倒、掉箱和超时判据。正式对比两个检查点时应固定原生 MuJoCo 版本、数据划分和种子。

`carrywm/warp_sim.py` 批量推进机器人和箱子，按环境归集脚、手腕与箱子的接触力及地面和桌面支撑。训练场景省去仅用于显示的影子机器人和参考标记。`carrywm/collector_v4.py` 对所有活跃环境做一次批量策略前向；`carrywm/replay_v4.py` 按环境和时间存储数据，保留终止步的真实下一状态。物理容量溢出和非有限状态会显式报错。

## 启动与评估

在现有 V3 的 Python 环境中另建 V4 依赖目录。版本对齐 [SuperMimic 的依赖锁文件](https://github.com/HybridRobotics/supermimic/blob/abc124c/uv.lock)。以下变量分别指向官方 sim2sim 项目、搬箱 NPZ 目录和本次运行的独立输出目录。

```bash
python -m pip install --target "$V4_DEPS" mujoco==3.10.0 mujoco-warp==3.10.0.1 warp-lang==1.14.0
export PYTHONPATH="$V4_DEPS${PYTHONPATH:+:$PYTHONPATH}"
CARRY_SIM_ROOT="$CARRY_SIM_ROOT" CARRY_MOTION_DIR="$CARRY_DATA" CARRY_WARP_DEVICE=cuda:0 \
  python -m unittest discover -s tests -p 'test_v4*.py'
python -m carrywm.monitor --gpu 0 --output-dir "$V4_RUN" -- \
  python -u -m carrywm.train_v4 --device cuda:0 --precision bf16 \
  --sim-root "$CARRY_SIM_ROOT" --motion-dir "$CARRY_DATA" --output-dir "$V4_RUN" \
  --iterations 6000 --eval-every 250 --eval-episodes 3 --save-every 50 --log-every 5 --seed 42
```

`metrics.jsonl` 记录真实转移数量、回放比例、实际优化器更新次数和分阶段用时。`gpu-*.json` 和 `gpu-*.csv` 记录整卡与训练进程的显存采样，包括 Warp 在 PyTorch 之外申请的显存。`latest.pt` 保存 V4 模型、优化器、回放和计数。断点恢复使用相同配置并追加 `--resume "$V4_RUN/latest.pt"`，物理环境会开启新回合，先重新采集至少 8 个并行控制步。旧回放不会与新回合拼接。
