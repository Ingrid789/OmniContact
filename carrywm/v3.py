"""V3 actor and critic updates on the same imagined transitions."""
from typing import NamedTuple

import torch

from .train import HORIZON, Learner, frozen, lambda_returns, tensor


CRITIC_EPOCHS = 5
CRITIC_MINIBATCHES = 4
CRITIC_CHUNK = 2048


class CriticBatch(NamedTuple):
    states: torch.Tensor
    goals: torch.Tensor
    contexts: torch.Tensor
    targets: torch.Tensor
    valid: torch.Tensor


class LearnerV3(Learner):
    def _imagine(self, inputs, effective_batch):
        """Accumulate this BPTT gradient and retain detached critic examples."""
        ctx = inputs[2]
        count = len(ctx)
        ret, mask, states, goals, rewards, continuations, masks = self._rollout(
            inputs, checkpoint_encoder=getattr(self.args, "checkpoint_encoder", False))
        (-ret.sum() / effective_batch).backward()

        # Lambda targets use one common, pre-update critic across all microbatches.
        # All stored tensors are detached from the 32-step actor graph.
        with torch.no_grad():
            states = torch.stack(states).flatten(0, 1)
            goals = torch.stack(goals).flatten(0, 1)
            contexts = ctx.expand(HORIZON+1, -1, -1).reshape(-1, ctx.shape[-1])
            old = torch.cat([
                self._value(states[i:i+CRITIC_CHUNK], goals[i:i+CRITIC_CHUNK],
                            contexts[i:i+CRITIC_CHUNK])
                for i in range(0, len(states), CRITIC_CHUNK)
            ]).reshape(HORIZON+1, count)
            targets = lambda_returns(torch.stack(rewards), old[1:],
                                     torch.stack(continuations))
            examples = CriticBatch(states[:-count], goals[:-count], contexts[:-count], targets.flatten(),
                                   torch.stack(masks).flatten().bool())
            return examples, ret.detach().sum(), mask.sum()

    def _accumulate_actor(self, inputs, batch_size, microbatch):
        parts = []
        total_return = torch.zeros((), device=self.device)
        surviving = torch.zeros_like(total_return)
        with frozen(self.wm, self.critic):
            for start in range(0, batch_size, microbatch):
                chunk = tuple(v[start:start+microbatch] for v in inputs)
                part, returns, alive = self._imagine(chunk, batch_size)
                parts.append(part)
                total_return += returns
                surviving += alive
        batch = parts[0] if len(parts) == 1 else CriticBatch(*(torch.cat(fields) for fields in zip(*parts)))
        return batch, total_return / batch_size, surviving / batch_size

    def _critic_backward(self, data, indices, chunk_size):
        """Accumulate gradients for one logical minibatch of valid time steps."""
        total = torch.zeros((), device=self.device)
        for rows in indices.split(chunk_size):
            pred = self._value(data.states[rows], data.goals[rows], data.contexts[rows])
            loss = (pred - data.targets[rows]).square().sum() / len(indices)
            loss.backward()
            total += loss.detach()
        return total

    def _fit_critic(self, batch):
        valid = torch.nonzero(batch.valid, as_tuple=True)[0]
        if len(valid) < CRITIC_MINIBATCHES:
            raise ValueError("Need at least four valid imagined transitions for four critic minibatches")
        opt = self.optimizers[2]
        chunk = getattr(self.args, "critic_chunk", 32768)
        total = torch.zeros((), device=self.device)
        for _ in range(CRITIC_EPOCHS):
            shuffled = valid[torch.randperm(len(valid), device=self.device)]
            for indices in shuffled.tensor_split(CRITIC_MINIBATCHES):
                while True:
                    opt.zero_grad(set_to_none=True)
                    try:
                        loss = self._critic_backward(batch, indices, chunk)
                        break
                    except torch.cuda.OutOfMemoryError:
                        if chunk == 1 or self.device.type != "cuda":
                            raise
                    # Retry only the current minibatch; earlier optimizer steps stand.
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    chunk = max(1, min(chunk, len(indices)) // 2)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                total += loss * (len(indices) / len(valid))
        opt.zero_grad(set_to_none=True)
        return total.item() / CRITIC_EPOCHS

    def policy_update(self, replay, batch_size):
        inputs = tuple(tensor(x, self.device) for x in replay.actor(batch_size, HORIZON))
        state, _, _, history, raw, _ = inputs
        # Every trajectory in this logical batch uses the same normalization statistics.
        with torch.no_grad():
            self.actor.update_stats(self.reference(state, raw[:, self.offsets]), history)
        microbatch = min(self.args.actor_microbatch, batch_size)
        if microbatch < 1:
            raise ValueError("actor_microbatch must be positive")
        opt = self.optimizers[1]
        while True:
            opt.zero_grad(set_to_none=True)
            try:
                data, mean_return, survival = self._accumulate_actor(inputs, batch_size, microbatch)
                break
            except torch.cuda.OutOfMemoryError:
                if microbatch == 1 or self.device.type != "cuda":
                    raise RuntimeError("Insufficient memory for a V3 actor rollout")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            microbatch = max(1, microbatch // 2)
            self.args.actor_microbatch = microbatch

        # An OOM after an optimizer step must never retry the whole update.
        try:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            opt.zero_grad(set_to_none=True)
            critic_loss = self._fit_critic(data)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("V3 optimizer update exhausted memory; resume the last checkpoint") from exc
        return {"actor/return": mean_return.item(), "actor/alive32": survival.item(),
                "actor/updates": 1, "actor/rollouts": batch_size,
                "actor/microbatches": (batch_size + microbatch - 1) // microbatch,
                "actor/microbatch_size": microbatch,
                "actor/checkpoint_encoder": getattr(self.args, "checkpoint_encoder", False),
                "critic/loss": critic_loss,
                "critic/updates": CRITIC_EPOCHS * CRITIC_MINIBATCHES,
                "critic/epochs": CRITIC_EPOCHS}
