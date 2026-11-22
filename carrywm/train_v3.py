"""V3 training schedule for the MuJoCo carry-box world model and policy."""
import argparse

from . import train as base
from .sim import DEFAULT_CFGEN_GOAL_NOISE_STD_M
from .v3 import LearnerV3


WM_UPDATES = 10
WM_BATCH = 1024
ACTOR_BATCH = 4096


def wm_iteration(learner, replay, batch_size=WM_BATCH):
    """Each call to wm_update independently samples windows and steps Adam."""
    totals = {}
    for _ in range(WM_UPDATES):
        metrics = learner.wm_update(replay, batch_size)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.) + value
    return {**{name: value / WM_UPDATES for name, value in totals.items()},
            "wm/optimizer_steps": WM_UPDATES, "wm/windows_per_update": batch_size}


def configure(parser):
    parser.description = __doc__
    parser.set_defaults(batch_size=ACTOR_BATCH, wm_batch_size=WM_BATCH,
                        warmup_steps=4096, warmup_teacher=False, wm_pretrain=0)
    parser.add_argument("--cfgen-goal-noise-std-m", type=float, default=DEFAULT_CFGEN_GOAL_NOISE_STD_M,
                        help="Std dev in metres of the training CF-Gen placement goal's XY noise; "
                             "0 disables this noise while retaining the table-support constraint")
    parser.add_argument("--actor-microbatch", type=int, default=ACTOR_BATCH,
                        help="Maximum simultaneous actor graphs within the 4096-rollout update")
    parser.add_argument("--checkpoint-encoder", action=argparse.BooleanOptionalAction, default=True,
                        help="Recompute actor encoder activations during rollout backpropagation")
    parser.add_argument("--critic-chunk", type=int, default=32768,
                        help="Maximum valid transitions in one critic backward pass")


def args_from_cli(argv=None):
    args = base.args_from_cli(argv, configure=configure)
    if args.warmup_teacher or args.wm_pretrain != 0:
        raise ValueError("V3 scratch warmup collects noisy actions from its own initial policy; "
                         "leave the teacher off and --wm-pretrain at zero.")
    if (args.wm_batch_size != WM_BATCH or args.batch_size != ACTOR_BATCH
            or not 1 <= args.actor_microbatch <= ACTOR_BATCH
            or args.critic_chunk < 1 or args.auto_batch):
        raise ValueError("V3 uses 1024 WM windows, 4096 actor rollouts, and a positive "
                         "actor microbatch up to 4096. Omit --auto-batch so the "
                         "effective sample count remains fixed.")
    return args


if __name__ == "__main__":
    base.train(args_from_cli(), version=3, learner_type=LearnerV3, wm_iteration=wm_iteration)
