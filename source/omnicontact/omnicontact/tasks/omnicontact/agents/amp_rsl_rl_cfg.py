"""Native RSL-RL PPO + AMP configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from omnicontact.tasks.omnicontact.amp_motion_dataset import AMPCarryBoxMotionDatasetCfg
from .rsl_rl_ppo_cfg import CarryBoxPPORunnerCfg, RslRlTransformerActorCriticCfg


AMP_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_rubber_hand",
]
AMP_ANCHOR_NAME = "torso_link"
AMP_OBS_TERMS = [
    "base_height",
    "joint_pos",
    # "ee_pos",
    # "base_lin_vel",
    # "base_ang_vel",
    "projected_gravity",
    "box_pos_local",
    "contact_info",
]


@configclass
class AMPPPOCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "AMPPPO"
    amp_replay_buffer_size: int = 100000
    amp_grad_pen_lambda: float = 1.0
    amp_loss_coef: float = 1.0
    amp_grad_pen_coef: float = 1.0
    amp_normalize_obs: bool = True
    amp_discriminator_lr: float = 1.0e-3


@configclass
class AMPCarryBoxRunnerCfg(CarryBoxPPORunnerCfg):
    experiment_name = "OmniContact_AMP"
    max_iterations = 50000
    amp_data = AMPCarryBoxMotionDatasetCfg(
        motion_files=[], body_names=AMP_BODY_NAMES, anchor_name=AMP_ANCHOR_NAME,
        root_body_name="pelvis", amp_obs_terms=AMP_OBS_TERMS, history_length=10,
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 0.5
    amp_tracking_weight = 0.9
    algorithm = AMPPPOCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class AMPCarryBoxTransformerRunnerCfg(AMPCarryBoxRunnerCfg):
    experiment_name = "OmniContact_AMP_Transformer"
    policy = RslRlTransformerActorCriticCfg()
