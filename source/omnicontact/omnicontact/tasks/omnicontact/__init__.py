"""OmniContact task definitions and Gym environment registrations."""

import gymnasium as gym


def _register(task_id: str, env_cfg: str, agent_cfg: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{env_cfg}",
            "rsl_rl_cfg_entry_point": agent_cfg,
        },
    )


_TASKS = [
    (
        "OmniContact",
        "env_cfg:CFBaseEnvCfg",
        f"{__name__}.agents.rsl_rl_ppo_cfg:CarryBoxPPORunnerCfg",
    ),
    (
        "OmniContact-play",
        "env_cfg:CFBaseEnvPlayCfg",
        f"{__name__}.agents.rsl_rl_ppo_cfg:CarryBoxPPORunnerCfg",
    ),
    (
        "OmniContact-Transformer",
        "env_cfg:CFBaseEnvCfg",
        f"{__name__}.agents.rsl_rl_ppo_cfg:CarryBoxTransformerRunnerCfg",
    ),
    (
        "OmniContact-Transformer-play",
        "env_cfg:CFBaseEnvPlayCfg",
        f"{__name__}.agents.rsl_rl_ppo_cfg:CarryBoxTransformerRunnerCfg",
    ),
    (
        "OmniContact-AMP", 
        "amp_env_cfg:CFAMPEnvCfg", 
        f"{__name__}.agents.amp_rsl_rl_cfg:AMPCarryBoxRunnerCfg"
    ),
    (
        "OmniContact-AMP-play",
        "amp_env_cfg:CFAMPEnvPlayCfg",
        f"{__name__}.agents.amp_rsl_rl_cfg:AMPCarryBoxRunnerCfg",
    ),
    (
        "OmniContact-AMP-Transformer",
        "amp_env_cfg:CFAMPEnvCfg",
        f"{__name__}.agents.amp_rsl_rl_cfg:AMPCarryBoxTransformerRunnerCfg",
    ),
    (
        "OmniContact-AMP-Transformer-play",
        "amp_env_cfg:CFAMPEnvPlayCfg",
        f"{__name__}.agents.amp_rsl_rl_cfg:AMPCarryBoxTransformerRunnerCfg",
    ),
]

for _task_id, _env_cfg, _agent_cfg in _TASKS:
    _register(_task_id, _env_cfg, _agent_cfg)
