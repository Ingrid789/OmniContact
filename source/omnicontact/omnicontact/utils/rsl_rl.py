"""OmniContact observation adapter and native RSL-RL runner selection."""

import torch

from rsl_rl.env import VecEnv


class OmniContactVecEnvWrapper(VecEnv):
    """Adapt split Isaac Lab observation groups to the RSL-RL 2.3 tuple API.

    Reuse observations returned by reset/step: recomputing them would resample
    observation noise and can advance history buffers on some Isaac Lab versions.
    """

    actor_obs_keys = ("policy_command", "policy_proprio")
    critic_obs_keys = ("critic_command", "critic_proprio")

    def __init__(self, env, clip_actions=None):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.unwrapped.num_envs
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        obs, _ = self.reset()
        self.num_obs = obs.shape[-1]
        self.num_privileged_obs = self._observations["critic"].shape[-1]
        self.actor_obs_group_dims = tuple(self._observations[key].shape[-1] for key in self.actor_obs_keys)
        self.critic_obs_group_dims = tuple(self._observations[key].shape[-1] for key in self.critic_obs_keys)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def _adapt(self, observations):
        observations = dict(observations)
        for target, keys in (("policy", self.actor_obs_keys), ("critic", self.critic_obs_keys)):
            observations[target] = torch.cat([observations[key] for key in keys], dim=-1)
        self._observations = observations
        return observations["policy"]

    def get_observations(self):
        return self._observations["policy"], {"observations": self._observations}

    def reset(self):
        observations, extras = self.env.reset()
        obs = self._adapt(observations)
        return obs, {**extras, "observations": self._observations}

    def step(self, actions):
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        obs = self._adapt(observations)
        extras = {**extras, "observations": self._observations}
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return obs, rewards, (terminated | truncated).long(), extras

    def close(self):
        return self.env.close()

    def seed(self, seed=-1):
        return self.unwrapped.seed(seed)


def runner_and_wrapper(agent_cfg):
    """Select an RSL-RL runner and the split-observation adapter."""
    name = getattr(agent_cfg, "runner_class", "OnPolicyRunner")
    if name == "MotionOnPolicyRunner":
        from omnicontact.utils.my_on_policy_runner import MotionOnPolicyRunner

        return MotionOnPolicyRunner, OmniContactVecEnvWrapper
    if name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner: {name}")
    from rsl_rl.runners import OnPolicyRunner

    return OnPolicyRunner, OmniContactVecEnvWrapper


def inference_runner_cfg(agent_cfg):
    """Inference needs only PPO and the actor/critic, never the expert dataset."""
    cfg = agent_cfg.to_dict()
    cfg["algorithm"] = {k: v for k, v in cfg["algorithm"].items() if not k.startswith("amp_")}
    cfg["algorithm"]["class_name"] = "PPO"
    cfg.pop("amp_data", None)
    return cfg
