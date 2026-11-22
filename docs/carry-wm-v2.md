# Training the carry-box world model and policy

Run from the repository root with Python 3.11. The tested configuration uses
PyTorch 2.7 with CUDA, MuJoCo 3.4, NumPy, SciPy and PyYAML. Use ONNX Runtime for
the released collection teacher and pytest for the tests. Provide the official
sim2sim checkout with its G1 meshes, planner, YAML and teacher ONNX weights, and
the directory of 50 Hz training NPZs. The training data and weights are external.
[Sim2sim revision used for testing](https://github.com/Ingrid789/OmniContact_sim2sim/tree/1cf9ddd4067cbe5710b1f475e96f9c69f88055e2)

```bash
export CARRY_SIM_ROOT=/path/to/OmniContact_sim2sim
export SIM2SIM_ROOT="$CARRY_SIM_ROOT"
export CARRY_DATA=/path/to/assets/data/box/case1_carry
python -m pytest -q tests/test_data_carry.py tests/test_train_v2.py tests/test_model_v2.py

RUN=outputs/carry-v2
python -m carrywm.monitor --gpu 0 --output-dir "$RUN" -- \
  python -u -m carrywm.train --sim-root "$CARRY_SIM_ROOT" --motion-dir "$CARRY_DATA" \
  --output-dir "$RUN" --iterations 6000 --warmup-steps 4096 --wm-pretrain 128 \
  --collect-steps 32 --wm-batch-size 1024 --batch-size 2560 --auto-batch \
  --target-memory-fraction .84 --max-batch-size 3000 --precision bf16
```

This batch configuration targets a 24 GiB RTX 3090. Start with a smaller batch
on a different GPU. The tuner uses measured training allocations and available
memory; it retries policy updates with a smaller batch after a CUDA OOM.
The 10% generator probability applies per primary episode and recovery attempts
are tracked separately. The optional frozen teacher supplies initial replay
actions; the actor loss uses returns through the frozen world model. Pass
`--no-warmup-teacher` to warm up using the current policy.

The MLP actor reads reference tensors of shape `[B,11,113]` and observations of
shape `[B,10,141]`. The 32-dimensional encoder output conditions one 29-joint
action. The 129-dimensional dynamics input concatenates 84 state values, 29
action channels and 16 physical context values. The context contains box
half-sizes, mass, diagonal inertia, three friction coefficients and two table
positions. All joint vectors passed to the V2 actor use MuJoCo joint order.
See the [design and dimensions](carry-wm-v2-design.md) for field definitions.

## Outputs and comparisons

- `metrics.jsonl` contains actual collection steps, measured per-stage times, losses,
  real MuJoCo evaluation outcomes and held-out 1/8/32-step dynamics errors.
  Counts are reported for each reference source and observed evaluation phase.
  With the default one episode per source, success rates describe those two trials.
  Increase `--eval-episodes` for policy comparisons.
- `gpu-*.csv` records timestamp offsets, training-process memory, total device memory
  and GPU busy percentage. `gpu-*.json` reports sampled peaks, memory percentages,
  mean device busy percentage, exit status and end-to-end elapsed seconds for that
  invocation. Device-wide measurements include other users' jobs.
- `latest.pt` is an atomic training checkpoint. Resume with the same training
  options and `--resume "$RUN/latest.pt"`. A new physical episode starts on
  resume. `summary.json` records the completed training budget, total steps,
  cumulative trainer wall time and PyTorch peak allocation/reservation.
- Export only the deployable actor with
  `python -m carrywm.export "$RUN/latest.pt" "$RUN/actor.pt"`.
  The exporter reloads the TorchScript file and compares actions against
  the PyTorch policy. The exported output is a normalized PD command.
- For the input ablation, train another run with `--goal-mode flow`. It masks the
  64 whole-body targets and keeps parameter count fixed. Use `--encoder-width`
  to compare 1024, 2048 and 3072 under the same seed, evaluation scenes and
  real-collection budget. Log time to the same measured success rate.
