"""Count actual optimizer calls and check gradients, targets and sample coverage."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from carrywm.model import Actor, Critic, WorldModel
from carrywm.replay import Replay
from carrywm.train import Collector, HORIZON, OFFSETS, lambda_returns
from carrywm.v3 import CriticBatch, LearnerV3


class TinyGeometry:
    """Use the real 32-step learner with inexpensive differentiable observations."""

    def forward_kinematics(self, state):
        return state[:, :3]

    def fk_from_target(self, raw):
        return raw[:, :3]

    def reference(self, state, raw, fk=None):
        return F.pad(raw, (0, 2))

    def observation(self, state, action, ctx, fk=None):
        return F.pad(torch.cat((state, action), -1), (0, 28))

    def reward(self, before, after, prev, action, target, contacts, **kwargs):
        return 0.001 * (before[:, 0] * action[:, 0] + after[:, 42] + contacts[:, 0])


def tiny_learner(microbatch):
    with torch.random.fork_rng():
        torch.manual_seed(79)
        net = LearnerV3.__new__(LearnerV3)
        net.args = SimpleNamespace(precision="fp32", goal_mode="full", actor_microbatch=microbatch,
                                   drop_penalty=1)
        net.device = torch.device("cpu")
        net.offsets = torch.tensor(OFFSETS)
        net.geo = TinyGeometry()
        net.actor = Actor(first_width=12, decoder_width=12, hidden=(8,))
        net.wm = WorldModel(16, width=12)
        net.critic = Critic(16, hidden=(12,))
        net.optimizers = [torch.optim.Adam(m.parameters(), lr=.001)
                          for m in (net.wm, net.actor, net.critic)]
    return net


class DeterministicReplay:
    def actor(self, batch, horizon):
        rng = torch.Generator().manual_seed(42)
        state = torch.zeros(batch, 84)
        state[:, 0] = torch.arange(batch)
        state[:, [3, 74]] = 1
        hist = torch.randn(batch, 10, 141, generator=rng)
        raw = .01 * torch.randn(batch, horizon+51, 111, generator=rng)
        return (state, torch.zeros(batch,29), torch.zeros(batch,16), hist, raw,
                torch.full((batch,), horizon + 2))


class UpdateScheduleTests(unittest.TestCase):
    def test_production_defaults(self):
        from carrywm.train_v3 import args_from_cli
        args = args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."])
        self.assertEqual((args.wm_batch_size, args.batch_size, args.actor_microbatch),
                         (1024, 4096, 2048))
        self.assertIsNone(args.resume)
        self.assertEqual((args.warmup_steps, args.warmup_teacher, args.wm_pretrain), (4096, False, 0))

    def test_warmup_cannot_use_teacher_or_wm_pretraining(self):
        from carrywm.train_v3 import args_from_cli
        common = ["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."]
        for flags in (["--warmup-teacher"], ["--wm-pretrain", "1"]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                args_from_cli(common + flags)

    def test_runner_rejects_v2_checkpoint(self):
        from carrywm import train
        from carrywm.train_v3 import args_from_cli
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp, "v2.pt")
            torch.save({"version": 2}, checkpoint)
            args = args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--device", "cpu",
                                  "--output-dir", str(Path(tmp, "v3")), "--resume", str(checkpoint)])
            with patch.object(train, "CarryEnv"), self.assertRaisesRegex(ValueError, "V2.*V3"):
                train.train(args, version=3, learner_type=lambda _: tiny_learner(4))

    def test_warmup_uses_current_actor_and_executes_no_optimizer_updates(self):
        class Scene:
            def reset(self):
                self.state, self.context = np.zeros(84,np.float32), np.zeros(16,np.float32)
                self.clip = np.zeros((100,111), np.float32)
                self.index, self.name, self.source = 0, "synthetic", "npz"
                self.proposals = []

            def teacher_action(self):
                raise AssertionError("a warmup action came from the teacher")

            def step(self, proposal):
                self.proposals.append(proposal.copy())
                self.index += 1
                self.state = np.full(84, self.index, dtype=np.float32)
                return self.state, np.zeros(6), False, {"executed_action": proposal * .5}

        net = tiny_learner(4)
        net.geo.reward_terms = lambda s, after, *rest: {"progress": after[:,0]-s[:,0]}
        actor_outputs = []
        net.actor.register_forward_hook(lambda _, inp, out: actor_outputs.append(out.detach().clone()))
        env, replay = Scene(), Replay(64, seed=7)
        collector = Collector(env, net, replay, np.random.default_rng(8))
        stats = collector.collect(9, teacher_prob=0., noise=.1, update_normalizers=True)
        self.assertEqual(len(actor_outputs), 9)
        self.assertEqual(stats["sim/teacher_fraction"], 0.)
        self.assertFalse(np.allclose(torch.cat(actor_outputs).numpy(), env.proposals))
        self.assertEqual(replay.size, 9)
        self.assertEqual(replay.wm(1, 8)[0].shape, (1, 9, 84))
        self.assertEqual(float(net.actor.goals.count), 9*11)
        self.assertEqual(float(net.actor.observations.count), 9*10)
        self.assertEqual(float(net.wm.inputs.count), 9)
        applied = np.stack(replay.segments[0].actions)
        np.testing.assert_allclose(applied, np.array(env.proposals) * .5)
        np.testing.assert_allclose(net.wm.inputs.mean[84:113].numpy(), applied.mean(0), atol=1e-6)
        self.assertTrue(all(not opt.state for opt in net.optimizers))

    def test_world_model_draws_and_steps_ten_times(self):
        from carrywm.train_v3 import wm_iteration
        net = tiny_learner(3)
        calls = []

        class Windows:
            def wm(self, batch, steps):
                calls.append((batch, steps))
                state = torch.zeros(batch, steps+1, 84)
                state[..., [3, 74]] = 1
                state[..., 7] = .01 * len(calls)
                return state, torch.zeros(batch,steps,29), torch.zeros(batch,steps,6), torch.zeros(batch,16)

        weight = next(net.wm.parameters())
        wm_iteration(net, Windows())
        self.assertEqual(calls, [(1024, 8)] * 10)
        self.assertEqual(net.optimizers[0].state[weight]["step"].item(), 10)

    def test_unequal_actor_chunks_equal_one_full_gradient(self):
        def run(micro):
            net = tiny_learner(micro)
            gradients, counts = [], Counter()
            def save_actor_gradient(opt, args, kwargs):
                gradients.append(torch.cat([p.grad.detach().flatten().clone() for p in net.actor.parameters()]))
            net.optimizers[1].register_step_pre_hook(save_actor_gradient)
            def step_critic(*_):
                counts["critic_steps"] += 1
            net.optimizers[2].register_step_post_hook(step_critic)
            def observe_rollout(module, args, result):
                counts["imagined_transitions"] += len(result[0])
            net.wm.register_forward_hook(observe_rollout)
            def make_targets(*args, **kwargs):
                self.assertEqual(counts["critic_steps"], 0)
                counts["target_trajectories"] += args[0].shape[1]
                return lambda_returns(*args, **kwargs)
            with patch("carrywm.v3.lambda_returns", side_effect=make_targets):
                net.policy_update(DeterministicReplay(), 5)
            self.assertEqual(len(gradients), 1)
            self.assertGreater(float(gradients[0].norm()), 0)
            self.assertEqual(counts, {"imagined_transitions": 5 * HORIZON,
                                      "target_trajectories": 5, "critic_steps": 20})
            self.assertTrue(all(p.grad is None for p in net.wm.parameters()))
            return gradients[0]

        torch.testing.assert_close(run(3), run(5), atol=3e-6, rtol=2e-4)

    def test_critic_visits_only_valid_rows_once_per_epoch(self):
        net = tiny_learner(3)
        oracle = deepcopy(net.critic)
        oracle_opt = torch.optim.Adam(oracle.parameters(), lr=.001)
        rng = torch.Generator().manual_seed(19)
        n = 21
        batch = CriticBatch(torch.randn(n,84,generator=rng), torch.randn(n,11,113,generator=rng),
                            torch.randn(n,16,generator=rng), torch.randn(n,generator=rng),
                            torch.arange(n) % 6 != 0)
        net.args.critic_chunk = 2
        initial_targets = batch.targets.clone()
        batches = []
        original = net._critic_backward
        def check_minibatch(data, indices, chunk_size):
            torch.testing.assert_close(data.targets, initial_targets)
            self.assertEqual(chunk_size, 2)
            batches.append(indices.clone())
            return original(data, indices, chunk_size)
        with patch.object(net, "_critic_backward", side_effect=check_minibatch):
            net._fit_critic(batch)
        self.assertEqual(len(batches), 20)
        epochs = [torch.cat(batches[i:i+4]) for i in range(0,20,4)]
        for order in epochs:
            torch.testing.assert_close(torch.sort(order).values, torch.where(batch.valid)[0])
        self.assertGreater(len({tuple(order.tolist()) for order in epochs}), 1)

        # Oracle takes one mean squared error and one Adam step per logical minibatch.
        for rows in batches:
            oracle_opt.zero_grad(set_to_none=True)
            values = oracle(batch.states[rows], batch.goals[rows], batch.contexts[rows])
            F.mse_loss(values, initial_targets[rows]).backward()
            torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.)
            oracle_opt.step()
        for actual, expected in zip(net.critic.parameters(), oracle.parameters()):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
