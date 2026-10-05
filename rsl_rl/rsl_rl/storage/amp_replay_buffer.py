"""Bounded, GPU-resident replay of policy motion transitions."""

import torch


class AMPReplayBuffer:
    def __init__(self, observation_dim: int, capacity: int, device="cpu"):
        if capacity <= 0:
            raise ValueError("AMP replay capacity must be positive.")
        self.states = torch.empty(capacity, observation_dim, device=device)
        self.next_states = torch.empty_like(self.states)
        self.capacity = capacity
        self.position = 0
        self.size = 0

    @torch.no_grad()
    def insert(self, states, next_states):
        # Keep the newest samples even when a vectorized step exceeds capacity.
        states, next_states = states[-self.capacity:], next_states[-self.capacity:]
        count = len(states)
        indices = (torch.arange(count, device=self.states.device) + self.position) % self.capacity
        self.states[indices] = states.detach()
        self.next_states[indices] = next_states.detach()
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, count):
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty AMP replay buffer.")
        indices = torch.randint(self.size, (count,), device=self.states.device)
        return self.states[indices], self.next_states[indices]
