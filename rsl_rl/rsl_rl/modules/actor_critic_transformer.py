"""Feed-forward Transformer actor/critic with native RSL-RL PPO interfaces."""

import torch
from torch import nn
from torch.nn import functional as F

from .actor_critic import ActorCritic


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1.0e-6) * self.weight


class ObservationAttentionBlock(nn.Module):
    """Pre-normalized attention and SwiGLU, using ONNX-compatible tensor ops."""

    def __init__(self, width, heads, feedforward_width):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.attention_norm = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.feedforward_norm = RMSNorm(width)
        self.gate_and_value = nn.Linear(width, 2 * feedforward_width, bias=False)
        self.feedforward_output = nn.Linear(feedforward_width, width, bias=False)

    def forward(self, tokens):
        batch, length, width = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens)).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = torch.softmax((query @ key.transpose(-2, -1)) * self.head_dim ** -0.5, dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.attention_output(attended)
        gate, value = self.gate_and_value(self.feedforward_norm(tokens)).chunk(2, dim=-1)
        return tokens + self.feedforward_output(F.silu(gate) * value)


class ObservationGroupTransformer(nn.Module):
    """Project each flattened observation group to a token and read out a learned token.

    History stays inside its original group vector. Normalization is applied by
    OnPolicyRunner before this module, as for the native MLP policy.
    """

    def __init__(self, observation_dims, output_dim, embed_dim=256, num_heads=8, num_layers=3, ff_dim=512):
        super().__init__()
        if not observation_dims or any(dim <= 0 for dim in observation_dims):
            raise ValueError("Transformer observation group dimensions must be positive.")
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("Transformer embedding dimension must be positive and divisible by the head count.")
        if num_layers <= 0 or ff_dim <= 0:
            raise ValueError("Transformer layer count and feed-forward width must be positive.")
        self.observation_dims = tuple(observation_dims)
        self.input_dim = sum(observation_dims)
        self.projections = nn.ModuleList(nn.Linear(dim, embed_dim) for dim in observation_dims)
        self.readout_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, len(observation_dims) + 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList(
            ObservationAttentionBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        )
        self.output_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, observations):
        chunks = observations.split(self.observation_dims, dim=-1)
        group_tokens = [projection(chunk).unsqueeze(1) for projection, chunk in zip(self.projections, chunks)]
        tokens = torch.cat([self.readout_token.expand(observations.shape[0], -1, -1), *group_tokens], dim=1)
        tokens = tokens + self.position_embedding
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.output_norm(tokens[:, 0]))


class ActorCriticTransformer(ActorCritic):
    """Independent Transformer actor and critic, sharing ActorCritic's PPO API."""

    def __init__(
        self, num_actor_obs, num_critic_obs, num_actions,
        actor_obs_group_dims=None, critic_obs_group_dims=None,
        transformer_embed_dim=256, transformer_num_heads=8,
        transformer_num_layers=3, transformer_ff_dim=512,
        init_noise_std=1.0, noise_std_type="scalar",
        actor_hidden_dims=None, critic_hidden_dims=None, activation=None,
    ):
        # Initialize nn.Module directly: this architecture does not build MLPs.
        nn.Module.__init__(self)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        actor_dims = tuple(actor_obs_group_dims) if actor_obs_group_dims is not None else (num_actor_obs,)
        critic_dims = tuple(critic_obs_group_dims) if critic_obs_group_dims is not None else (num_critic_obs,)
        if sum(actor_dims) != num_actor_obs or sum(critic_dims) != num_critic_obs:
            raise ValueError("Transformer observation group dimensions must match the actor/critic input widths.")
        # The MLP fields are accepted for compatibility with Isaac Lab's policy
        # config base class. Transformer size is controlled by transformer_*.
        settings = (transformer_embed_dim, transformer_num_heads, transformer_num_layers, transformer_ff_dim)
        self.actor = ObservationGroupTransformer(actor_dims, num_actions, *settings)
        self.critic = ObservationGroupTransformer(critic_dims, 1, *settings)
        self._init_action_distribution(num_actions, init_noise_std, noise_std_type)
        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Transformer: {self.critic}")
