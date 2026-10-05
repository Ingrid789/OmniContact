from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

"""Native RSL-RL policies for OmniContact's split observation groups."""


@configclass
class CarryBoxPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_class = "OnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    save_best_after = 100
    # seed=0
    experiment_name = "OmniContact_RSL_PPO"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RslRlTransformerActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticTransformer"
    init_noise_std: float = 1.0
    # Required by the base config; only transformer_* controls network size.
    actor_hidden_dims: list[int] = []
    critic_hidden_dims: list[int] = []
    activation: str = "silu"
    transformer_embed_dim: int = 256
    transformer_num_heads: int = 8
    transformer_num_layers: int = 3
    transformer_ff_dim: int = 512


@configclass
class CarryBoxTransformerRunnerCfg(CarryBoxPPORunnerCfg):
    experiment_name = "OmniContact_Transformer"
    policy = RslRlTransformerActorCriticCfg()
