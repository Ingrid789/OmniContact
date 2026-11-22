"""Exercise the vector collector contract with two worlds and asynchronous resets."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from carrywm.collector_v4 import VectorCollector
from carrywm.replay_v4 import VectorReplay


class Worlds:
    def __init__(self):
        self.envs = [SimpleNamespace(clip=np.zeros((60, 111), np.float32), source="npz")
                     for _ in range(2)]
        self.resets = []
        self.actions = []
        self.epoch = np.zeros(2, int)

    def reset(self, ids):
        self.resets.append(ids.tolist())
        for i in ids:
            env = self.envs[i]
            self.epoch[i] += 1
            env.state = np.full(84, 100 * self.epoch[i] + 10 * i, np.float32)
            env.context = np.zeros(16, np.float32)
            env.index = 0

    def step(self, actions):
        self.actions.append(actions.copy())
        done = np.array([len(self.actions) == 2, False])
        infos = []
        for i, env in enumerate(self.envs):
            env.state = env.state + 1
            env.index += 1
            infos.append(dict(executed_action=actions[i] + 1, success=False, drop=done[i],
                              timeout=False, recovered=False, fallen=False))
        return np.stack([e.state for e in self.envs]), np.zeros((2, 6), np.float32), done, infos


class Policy:
    def __init__(self):
        self.calls = []
        self.stat_updates = 0

    def __call__(self, goal, history):
        self.calls.append(history.clone().numpy())
        return torch.zeros((len(goal), 29))

    def update_stats(self, *inputs):
        self.stat_updates += 1


class VectorCollectorTests(unittest.TestCase):
    def test_masked_reset_history_executed_action_and_exploration(self):
        actor, wm = Policy(), Policy()
        def observations(state, previous, ctx):
            obs = state[:, :1].expand(-1, 141).clone()
            obs[:, 1] = previous[:, 0]
            return obs

        geo = SimpleNamespace(observation=observations, reward_terms=lambda *args: {"constant": args[0][:, 0]*0})
        net = SimpleNamespace(geo=geo, actor=actor, wm=wm, autocast=nullcontext,
                              reference=lambda s, g: g, device=torch.device("cpu"),
                              args=SimpleNamespace(drop_penalty=1.))
        env = Worlds()
        replay = VectorReplay(20, 5, 2)
        collector = VectorCollector(env, net, replay, np.random.default_rng(10))
        collector.one_tick(noise=0.0, update_normalizers=True)
        result = collector.one_tick(noise=0.1)
        collector.one_tick(noise=0.1)
        self.assertEqual(env.resets, [[0, 1], [0]])
        self.assertEqual(actor.stat_updates, 1)
        self.assertEqual(wm.stat_updates, 1)
        self.assertEqual(result["sim/reward"], -.5)
        self.assertEqual(collector.steps, 6)
        self.assertEqual(collector.episodes["npz/drop"], 1)
        self.assertEqual(collector.episodes["npz/started"], 3)
        self.assertEqual(len(actor.calls), 3)
        np.testing.assert_array_equal(actor.calls[-1][0, :, 0], np.repeat(200, 10))
        np.testing.assert_array_equal(actor.calls[-1][1, :, 0], [110]*8 + [111, 112])
        self.assertEqual(actor.calls[-1][0, -1, 1], 0.)
        self.assertEqual(actor.calls[-1][1, -1, 1], env.actions[1][1, 0] + 1)
        self.assertEqual(replay.valid_windows(3), 1)
        self.assertEqual(replay.after[1, 0, 0], 102)
        self.assertEqual(replay.states[2, 0, 0], 200)
        np.testing.assert_array_equal(env.actions[0], np.zeros((2, 29)))
        self.assertFalse(np.allclose(env.actions[1], 0))
        np.testing.assert_array_equal(replay.actions[1], env.actions[1] + 1)


if __name__ == "__main__":
    unittest.main()
