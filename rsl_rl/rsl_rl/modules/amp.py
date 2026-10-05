"""Adversarial motion prior networks and device-local running statistics."""

import torch
from torch import nn


class AMPNormalizer(nn.Module):
    """Serializable moments; optional distributed updates pool all workers' samples."""

    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1.0e-4))

    @torch.no_grad()
    def update(self, samples: torch.Tensor, distributed: bool = False):
        samples = samples.float()
        moments = torch.cat((samples.sum(0), samples.square().sum(0), samples.new_tensor([len(samples)])))
        if distributed:
            torch.distributed.all_reduce(moments)
        size = self.mean.numel()
        count = moments[-1]
        if count == 0:
            return
        mean = moments[:size] / count
        var = (moments[size:2 * size] / count - mean.square()).clamp_min(0)
        delta = mean - self.mean
        total = self.count + count
        self.var.copy_((self.var * self.count + var * count + delta.square() * self.count * count / total) / total)
        self.mean.add_(delta * count / total)
        self.count.copy_(total)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return ((samples - self.mean) / (self.var + 1.0e-4).sqrt()).clamp(-10, 10)


class AMPDiscriminator(nn.Module):
    """Least-squares discriminator: expert target +1, policy target -1."""

    def __init__(self, input_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        for width in hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observations):
        return self.network(observations).squeeze(-1)

    def gradient_penalty(self, expert_input):
        expert_input = expert_input.detach().requires_grad_(True)
        prediction = self(expert_input)
        gradient = torch.autograd.grad(prediction.sum(), expert_input, create_graph=True)[0]
        return gradient.square().sum(-1).mean()

    @torch.no_grad()
    def reward(self, observations, coefficient):
        return coefficient * (1 - 0.25 * (self(observations) - 1).square()).clamp_min(0)
