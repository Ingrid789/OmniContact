# V5 GPU 加速分支说明（`v5-GPU-acceleration`）

本分支基于 V5（`codex/v5-soft-termination`）做 **A 组严格数学等价的系统级加速**，保持 V5 的软终止概率、Critic 价值上限 `VALUE_MAX = 10.8`、奖励、数据划分、动作探索噪声及零噪声 CF-Gen 规则不变，重点消除原版因老旧低频 CPU 单核阻塞导致的 RTX 3090 闲置问题。

## 加速改动清单（A1–A5）

| 模块 | 原版瓶颈 | 本分支修改 | 文件 |
|---|---|---|---|
| **A1. 全 GPU Replay 与 ClipBank** | 65536 条回放与 87 MB Actor 参考切片全在 CPU NumPy 拼接并同步 H2D 拷贝 | 新增连续张量存储 `ClipBank` 与全 GPU `VectorReplay`，静态 clip 永久驻留 GPU、动态 recovery clip 自动回收压缩；兼容旧版检查点平滑迁移（`from_legacy`） | [`carrywm/replay_v4.py`](../carrywm/replay_v4.py) |
| **A2. 并行与缓存 CF-Gen** | 启动预热及后期实时掉箱恢复在单核阻塞计算 ~2 s/次 | 启动时用 CPU 进程池并行预计算 512 组零噪声 CF-Gen 计划并写入 `~/.cache/carrywm` 磁盘缓存；多环境同时掉箱时并行求解 Recovery IK | [`carrywm/warp_sim.py`](../carrywm/warp_sim.py) |
| **A3. GPU 物理闭环与 CUDA Graph** | 每 tick 5120 次 Python `_finish_step` 循环 + 全量 `qpos/qvel` 回传；逐核发射 Warp | 新增纯 Warp `_pd_ctrl` 核函数与 `wp.ScopedCapture` 物理 CUDA Graph；`finish_step_gpu` 在 GPU 上向量化判定终止，每 tick 仅回传 5 KB 事件标志；`VectorCollector` 全程在 GPU 推进 | [`carrywm/warp_sim.py`](../carrywm/warp_sim.py), [`carrywm/collector_v4.py`](../carrywm/collector_v4.py) |
| **A4. 算子融合、去冗余重算与 Fused Adam** | 32 步想象展开小核函数 Launch Bound；显存空余一半仍开启 `--checkpoint-encoder`；每轮 4 次硬同步 | V4/V5 默认关闭 `--checkpoint-encoder`（OOM 时自动回退开启）；`torch.compile` 编译 WM 8 步更新、单步想象（0-D Tensor 步索引避免重编译）、动态 Critic 与采集前向；CUDA 下启用 `Adam(..., fused=True)`；仅日志轮次同步分段计时 | [`carrywm/train.py`](../carrywm/train.py), [`carrywm/v3.py`](../carrywm/v3.py), [`carrywm/train_v4.py`](../carrywm/train_v4.py) |
| **A5. 异步保存与并行 CPU Eval** | 每 50 轮同步写 683 MB 检查点；每 250 轮单核串行评估 6 个 MuJoCo 回合 | `AsyncCheckpointer` 后台线程异步写盘；`validate()` 按确定性抽样用多核并行跑 CPU MuJoCo 评估；主进程 `--cpu-threads` 默认放开至 `min(4, cpus)` | [`carrywm/train_v4.py`](../carrywm/train_v4.py), [`carrywm/train.py`](../carrywm/train.py) |

## 新增运行参数（均不影响断点版本兼容性）

* `--compile` / `--no-compile`（V4/V5 默认开启）
* `--cuda-graph` / `--no-cuda-graph`（V4/V5 默认开启）
* `--async-save` / `--no-async-save`（V4/V5 默认开启）
* `--checkpoint-encoder` / `--no-checkpoint-encoder`（V4/V5 默认关闭；遇 CUDA OOM 自动回退开启）
* `--cfgen-cache-dir`（默认 `$CARRYWM_CACHE_DIR` 或 `~/.cache/carrywm`）
* `--eval-workers`（默认 `min(6, max(1, cpus // 4))`）
* `--cpu-threads`（默认 `min(4, cpus)`）

## 启动训练（待当前 V5 训练结束后运行）

```bash
python -m carrywm.monitor --output-dir "$V5_ACCEL_RUN" --gpu 0 --interval 2 -- \
  python -u -m carrywm.train_v5 \
  --sim-root "$CARRY_SIM_ROOT" --motion-dir "$CARRY_DATA" \
  --output-dir "$V5_ACCEL_RUN" --device cuda:0 --precision bf16 \
  --iterations 6000 --eval-every 250 --eval-episodes 3 \
  --save-every 50 --log-every 5 --seed 42
```

如需单独排查某个系统加速开关（例如 CUDA Graph 或 `torch.compile`），可追加 `--no-cuda-graph` 或 `--no-compile`。
