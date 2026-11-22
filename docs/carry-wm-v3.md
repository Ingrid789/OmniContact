# Carry-box V3 training schedule

Each round collects the same 32 real transitions as the V2 server configuration.
The modules share V2's state and reference representations, architecture, 8-step
dynamics loss, 32-step policy objective, rewards, discount and termination rules.

| Module | Data per round | Adam steps |
|---|---|---:|
| World model | Ten independently sampled replay batches. Each contains 1024 windows of eight consecutive transitions. Backpropagate, clip, and apply Adam before drawing the next batch. | 10 |
| Actor | 4096 imagined trajectories of length 32. Split the input into two groups of 2048, accumulate each group's summed loss divided by 4096, then clip the aggregate gradient. WM and critic weights stay fixed throughout this update. | 1 |
| Critic | Cache detached states, goals and contexts from those actor rollouts. Compute the lambda-return targets using the pre-update critic. Shuffle valid transitions once per epoch, split into four minibatches, and visit them for five epochs. | 20 |

Input normalization for the actor is updated once per logical actor batch.
Up to 131,072 imagined transitions are available for critic fitting before
termination masking. By default a critic backward pass can use a full minibatch
of up to 32,768 valid transitions. A smaller physical chunk accumulates
gradients for the same optimizer step. On GPU memory pressure the trainer can
reduce the physical chunk size while preserving logical sample and update counts.

## Start a new V3 run

Start in an empty output directory. V3 initializes the actor, critic, WM,
optimizers, normalization statistics and replay from scratch. Each warmup action
comes from the current untrained actor with Gaussian exploration noise (standard
deviation 0.1 in normalized action coordinates). Real observations and executed
actions initialize the normalizers. After warmup the replay must contain an
eight-transition window; the first regular iteration fits the WM before the
first actor or critic step.

The official SuperMimic runner collects from its own policy and gates learning
on replay-window availability. It uses many parallel simulator environments.
For this single-collector carry-box experiment we explicitly collect **4096
real transitions** first. We use zero teacher actions and zero separate WM-only
pretraining updates. The real-transition count is a local adaptation.
[Upstream collector and training gate](https://github.com/HybridRobotics/supermimic/blob/abc124c/src/supermimic/runners/supermimic_runner.py#L598-L697)

```bash
python -m carrywm.monitor --output-dir "$V3_RUN" --gpu 0 -- \
  python -u -m carrywm.train_v3 --sim-root "$CARRY_SIM_ROOT" --motion-dir "$CARRY_DATA" \
  --output-dir "$V3_RUN" --iterations 6000 \
  --collect-steps 32 --wm-batch-size 1024 --batch-size 4096 --actor-microbatch 2048 \
  --critic-chunk 32768 --precision bf16 --save-every 50 --log-every 5 \
  --eval-every 250 --eval-episodes 3 --seed 42
```

The first V3 round writes `latest.pt`; subsequent saves follow the configured
interval. The warmup line in `metrics.jsonl` records real transitions, valid
WM windows, teacher use, and optimizer counts. Later lines report logical
optimizer steps, actual actor microbatch size, time and simulation outcomes.
The monitor records device/process memory and device busy percentage in
`gpu-*.csv` and `gpu-*.json`. On completion `summary.json` includes the time
spent in both warmup and training. For an interrupted V3 run, use the same
options and add `--resume "$V3_RUN/latest.pt"`. The V3 entry point accepts
V3 checkpoints for recovery; the scratch experiment uses no resume flag.
