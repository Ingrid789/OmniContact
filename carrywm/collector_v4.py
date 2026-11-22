"""One batched policy call and one batched physical control step per vector tick."""
from collections import Counter
from functools import partial

import numpy as np
import torch

from .data import OFFSETS
from .replay_v4 import gather_reference


class VectorCollector:
    def __init__(self, env, learner, replay, rng):
        self.env, self.net, self.replay, self.rng = env, learner, replay, rng
        self.count = len(env.envs)
        if self.count != replay.num_envs:
            raise ValueError("Replay and simulator world counts differ")
        self.steps = 0
        self.episodes = Counter()
        self.fresh = np.ones(self.count, bool)
        self.previous = np.zeros((self.count, 29), np.float32)
        self.state = np.empty((self.count, 84), np.float32)
        self.context = np.empty((self.count, 16), np.float32)
        self.hist = None
        replay.begin_new_episodes()

    @torch.no_grad()
    def one_tick(self, teacher_prob=0., noise=0., update_normalizers=False):
        envs, net, fresh = self.env.envs, self.net, self.fresh
        ids = np.flatnonzero(fresh)
        if len(ids):
            self.env.reset(ids)
            for i in ids:
                env = envs[i]
                self.state[i], self.context[i] = env.state, env.context
                self.episodes[f"{env.source}/started"] += 1
            self.previous[ids] = 0

        device = net.device
        as_tensor = partial(torch.as_tensor, device=device, dtype=torch.float32)
        state = as_tensor(self.state)
        ctx = as_tensor(self.context)
        previous = as_tensor(self.previous)
        obs = net.geo.observation(state, previous, ctx)
        if self.hist is None:
            self.hist = obs[:, None].expand(-1, 10, -1).clone()
        else:
            self.hist = torch.cat((self.hist[:, 1:], obs[:, None]), dim=1)
            if len(ids):
                reset = torch.as_tensor(ids, device=device)
                self.hist[reset] = obs[reset, None].expand(-1, 10, -1)

        clips = [env.clip for env in envs]
        frames = np.fromiter((env.index for env in envs), dtype=np.int32, count=self.count)
        goal = net.reference(state, as_tensor(gather_reference(clips, frames, OFFSETS)))
        with net.autocast():
            actions = net.actor(goal, self.hist).float().cpu().numpy()
        teacher_ids = np.flatnonzero(self.rng.random(self.count) < teacher_prob) if teacher_prob else ()
        for i in teacher_ids:
            actions[i] = envs[i].teacher_action()
        if noise:
            actions += self.rng.normal(0, noise, actions.shape).astype(np.float32)
        np.clip(actions, -10, 10, out=actions)
        after, contacts, done, infos = self.env.step(actions)
        # Simulator owns terminal states until the next tick's masked reset.
        done = np.asarray(done, bool)
        executed = actions.copy()
        drops = 0
        for i, info in enumerate(infos):
            if "executed_action" in info:
                executed[i] = info["executed_action"]
            if done[i]:
                source = envs[i].source
                self.episodes[f"{source}/done"] += 1
                for event in ("success", "drop", "timeout", "recovered", "fallen"):
                    self.episodes[f"{source}/{event}"] += int(info[event])
            drops += int(info["drop"])
        action = as_tensor(executed)
        if update_normalizers:
            net.actor.update_stats(goal, self.hist)
            net.wm.update_stats(state, action, ctx)
        self.replay.append(self.state, after, self.previous, executed, contacts, self.context,
                           obs.cpu().numpy(), frames, clips, fresh)
        target = gather_reference(clips, frames, (1,))[:, 0]
        terms = net.geo.reward_terms(state, as_tensor(after), previous, action,
                                     as_tensor(target), as_tensor(contacts))
        reward = sum(term.mean() for term in terms.values()).item() - net.args.drop_penalty * drops / self.count
        self.fresh = done.copy()
        self.state[:] = after
        self.previous = executed
        self.steps += self.count
        return {"sim/teacher_fraction": len(teacher_ids) / self.count, "sim/reward": reward}
