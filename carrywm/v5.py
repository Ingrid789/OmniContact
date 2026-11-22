"""V5 survival-weighted policy objective for the V4 carry-box trainer."""
import torch

from .model import DT, TRACKING_REWARD_WEIGHTS
from .train import GAMMA
from .v3 import LearnerV3


HEIGHT_THRESHOLD_M = .25
GRAVITY_Z_THRESHOLD = .8
TERMINATION_SOFTNESS = .05
VALUE_MAX = DT * sum(TRACKING_REWARD_WEIGHTS.values()) / (1. - GAMMA)


def soft_termination_probability(after, target):
    """Score the post-step pelvis against its reference. Both quaternions are unit wxyz."""
    height_error = (after[..., 2] - target[..., 99]).abs()
    # Local gravity z is 2*(x²+y²)-1. Yaw alone leaves it unchanged.
    gravity_z = 2 * after[..., 4:6].square().sum(-1) - 1
    ref_gravity_z = 2 * target[..., 101:103].square().sum(-1) - 1
    tilt_error = (gravity_z - ref_gravity_z).abs()
    normalized = torch.maximum(height_error / HEIGHT_THRESHOLD_M, tilt_error / GRAVITY_Z_THRESHOLD)
    return torch.sigmoid((normalized - 1.) / TERMINATION_SOFTNESS)


class LearnerV5(LearnerV3):
    def _continuation(self, after, target, task_alive):
        p = soft_termination_probability(after, target)
        # Only a hard carry-task boundary invalidates subsequent critic examples.
        return (1. - p) * task_alive, task_alive

    def _limit_return(self, value):
        return value.clamp(max=VALUE_MAX)
