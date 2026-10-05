"""RSL-RL PPO with the OmniContact adversarial motion prior.

Expert datasets expose observation_dim, observation_dims, history_length and
feed_forward_generator(num_mini_batch, mini_batch_size). Histories are flattened
per observation term (oldest to newest), as in Isaac Lab's observation manager.
"""

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.amp import AMPDiscriminator, AMPNormalizer
from rsl_rl.storage.amp_replay_buffer import AMPReplayBuffer


class AMPPPO(PPO):
    def __init__(
        self, policy, amp_data=None, amp_obs_key="amp", amp_replay_buffer_size=100000,
        amp_discr_hidden_dims=(256, 256), amp_reward_coef=0.5, amp_tracking_weight=0.9,
        amp_grad_pen_lambda=1.0, amp_loss_coef=1.0, amp_grad_pen_coef=1.0,
        amp_normalize_obs=True, amp_discriminator_lr=None, **kwargs,
    ):
        super().__init__(policy, **kwargs)
        if amp_data is None:
            raise ValueError("AMPPPO requires an expert dataset in algorithm.amp_data.")
        if not 0 <= amp_tracking_weight <= 1 or amp_reward_coef < 0:
            raise ValueError("AMP tracking weight must be in [0, 1] and reward coefficient nonnegative.")
        self.amp_data = amp_data
        self.amp_obs_key = amp_obs_key
        self.amp_obs_dim = int(amp_data.observation_dim)
        history = int(amp_data.history_length)
        dimensions = list(amp_data.observation_dims)
        self.amp_schema = {
            "history_length": history, "observation_dims": dimensions,
            "observation_terms": list(getattr(amp_data, "observation_terms", [])),
        }
        if history < 1 or sum(dimensions) != self.amp_obs_dim or any(d <= 0 or d % history for d in dimensions):
            raise ValueError("AMP observation dimensions must sum to observation_dim and be divisible by history_length.")
        indices, offset = [], 0
        for dim in dimensions:
            indices.extend(range(offset + dim - dim // history, offset + dim))
            offset += dim
        self.last_frame_indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        self.discriminator = AMPDiscriminator(self.amp_obs_dim + len(indices), amp_discr_hidden_dims).to(self.device)
        self.amp_normalizer = AMPNormalizer(self.amp_obs_dim).to(self.device) if amp_normalize_obs else None
        self.amp_replay = AMPReplayBuffer(self.amp_obs_dim, amp_replay_buffer_size, self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=amp_discriminator_lr or self.learning_rate
        )
        self.amp_reward_coef = float(amp_reward_coef)
        self.amp_tracking_weight = float(amp_tracking_weight)
        self.amp_grad_pen_lambda = float(amp_grad_pen_lambda)
        self.amp_loss_coef = float(amp_loss_coef)
        self.amp_grad_pen_coef = float(amp_grad_pen_coef)
        self._amp_obs = None
        self._reward_stats = []

    def set_amp_observations(self, observations):
        if not isinstance(observations, torch.Tensor) or observations.ndim != 2 or observations.shape[1] != self.amp_obs_dim:
            raise ValueError(f"Expected AMP observations with shape (num_envs, {self.amp_obs_dim}).")
        self._amp_obs = observations.to(self.device).detach().clone()

    def discriminator_input(self, states, next_states):
        if self.amp_normalizer is not None:
            states = self.amp_normalizer(states)
            next_states = self.amp_normalizer(next_states)
        # H frames at t plus only the newest frame at t+1, not two overlapping windows.
        return torch.cat((states, next_states.index_select(-1, self.last_frame_indices)), dim=-1)

    def process_env_step(self, rewards, dones, infos):
        next_states = infos["observations"][self.amp_obs_key].to(self.device)
        states = self._amp_obs
        if states is None:
            raise RuntimeError("Initialize AMP observations before the first rollout action.")
        # Isaac Lab returns reset observations for done environments. Never learn
        # cross-episode transitions or give them a spurious discriminator reward.
        valid = ~dones.reshape(-1).bool()
        with torch.no_grad():
            style = self.discriminator.reward(self.discriminator_input(states, next_states), self.amp_reward_coef)
            style = torch.where(valid, style, 0.0)
            mixed = self.amp_tracking_weight * rewards + (1 - self.amp_tracking_weight) * style
            self.amp_replay.insert(states[valid], next_states[valid])
            self._reward_stats.append(torch.stack((style.mean(), rewards.mean(), mixed.mean())))
        self.set_amp_observations(next_states)
        super().process_env_step(mixed, dones, infos)

    def update(self):
        losses = super().update()
        available = torch.tensor(int(self.amp_replay.size > 0), device=self.device)
        if self.is_multi_gpu:
            # All ranks must enter the same collectives, including on empty rollouts.
            torch.distributed.all_reduce(available, op=torch.distributed.ReduceOp.MIN)
        if available.item():
            updates = self.num_learning_epochs * self.num_mini_batches
            batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
            stats = torch.zeros(4, device=self.device)
            expert_batches = self.amp_data.feed_forward_generator(updates, batch_size)
            for _ in range(updates):
                states, next_states = self.amp_replay.sample(batch_size)
                expert, expert_next = (x.to(self.device) for x in next(expert_batches))
                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(torch.cat((states, expert)), distributed=self.is_multi_gpu)
                policy_logits = self.discriminator(self.discriminator_input(states, next_states))
                expert_input = self.discriminator_input(expert, expert_next)
                expert_logits = self.discriminator(expert_input)
                loss = 0.5 * (F.mse_loss(policy_logits, -torch.ones_like(policy_logits))
                              + F.mse_loss(expert_logits, torch.ones_like(expert_logits)))
                penalty = self.amp_grad_pen_lambda * self.discriminator.gradient_penalty(expert_input)
                self.discriminator_optimizer.zero_grad()
                (self.amp_loss_coef * loss + self.amp_grad_pen_coef * penalty).backward()
                if self.is_multi_gpu:
                    for param in self.discriminator.parameters():
                        if param.grad is not None:
                            torch.distributed.all_reduce(param.grad)
                            param.grad.div_(self.gpu_world_size)
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.discriminator_optimizer.step()
                stats += torch.stack((loss.detach(), penalty.detach(), policy_logits.mean().detach(), expert_logits.mean().detach()))
            losses.update(zip(("amp_discriminator", "amp_grad_penalty", "amp_policy_prediction", "amp_expert_prediction"),
                              (stats / updates).tolist()))
        if self._reward_stats:
            losses.update(zip(("amp_style_reward", "amp_task_reward", "amp_mixed_reward"),
                              torch.stack(self._reward_stats).mean(0).tolist()))
            self._reward_stats.clear()
        return losses

    def amp_state_dict(self):
        return {
            "schema": self.amp_schema,
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.discriminator_optimizer.state_dict(),
            "normalizer": self.amp_normalizer.state_dict() if self.amp_normalizer is not None else None,
        }

    def load_amp_state_dict(self, state, load_optimizer=True):
        if state["schema"] != self.amp_schema:
            raise ValueError("Checkpoint AMP observation schema does not match the expert dataset.")
        self.discriminator.load_state_dict(state["discriminator"])
        if (self.amp_normalizer is None) != (state["normalizer"] is None):
            raise ValueError("Checkpoint AMP normalization setting does not match the training configuration.")
        if self.amp_normalizer is not None:
            self.amp_normalizer.load_state_dict(state["normalizer"])
        if load_optimizer:
            self.discriminator_optimizer.load_state_dict(state["optimizer"])

    def broadcast_parameters(self):
        super().broadcast_parameters()
        for tensor in self.discriminator.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)
        if self.amp_normalizer is not None:
            for tensor in self.amp_normalizer.buffers():
                torch.distributed.broadcast(tensor, src=0)
