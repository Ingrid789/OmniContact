"""Train the V5 carry-box policy with soft pelvis termination and bounded critic returns."""
import math

from . import train_v4
from .v5 import GRAVITY_Z_THRESHOLD, HEIGHT_THRESHOLD_M, TERMINATION_SOFTNESS, VALUE_MAX, LearnerV5


def configure(parser):
    train_v4.configure(parser)
    parser.description = __doc__
    parser.set_defaults(termination_height_m=HEIGHT_THRESHOLD_M, termination_gravity_z=GRAVITY_Z_THRESHOLD,
                        termination_softness=TERMINATION_SOFTNESS, critic_value_max=VALUE_MAX)


def args_from_cli(argv=None):
    args = train_v4.args_from_cli(argv, configure_parser=configure)
    if not math.isfinite(args.drop_penalty) or args.drop_penalty < 0:
        raise ValueError('V5 requires a finite nonnegative drop penalty for its reward upper bound')
    return args


def train(args):
    train_v4.train(args, version=5, learner_type=LearnerV5)


if __name__ == '__main__':
    train(args_from_cli())
