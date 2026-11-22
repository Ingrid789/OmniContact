"""Check the soft pelvis termination on the learner's actual BPTT and critic paths."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from carrywm.train import GAMMA, HORIZON, LAMBDA, OFFSETS
from carrywm.v3 import LearnerV3
from carrywm.v5 import LearnerV5, VALUE_MAX, soft_termination_probability


class HeightActor(torch.nn.Module):
    """Only one parameter affects the return, via the predicted pelvis height."""

    def __init__(self, height):
        super().__init__()
        self.height = torch.nn.Parameter(torch.tensor(height, dtype=torch.float32))

    def forward(self, goal, history, checkpoint_encoder=False):
        return self.height.expand(len(goal), 29)


def make_learner(height, cls=LearnerV5):
    net = cls.__new__(cls)
    net.args = SimpleNamespace(precision="fp32", goal_mode="full", drop_penalty=1.)
    net.offsets = torch.tensor(OFFSETS)
    net.geo = SimpleNamespace(
        fk_from_target=lambda targets: targets[:, :3],
        forward_kinematics=lambda state: state[:, :3],
        reference=lambda state, raw, fk=None: raw,
        observation=lambda state, action, context, fk=None: state.new_zeros(len(state), 141),
        reward=lambda before, *args, **kwargs: before.new_ones(len(before)),
    )
    net.actor = HeightActor(height)
    net.wm = lambda state, action, context: (
        torch.cat((state[:, :2], action[:, :1], state[:, 3:]), dim=1),
        state.new_zeros(len(state), 6),
    )
    net._value = lambda state, goal, context: state.new_full((len(state),), 2.)
    return net


def inputs(steps=HORIZON+1):
    state = torch.zeros(1, 84)
    state[:, [3, 74]] = 1.
    raw = torch.zeros(1, HORIZON+51, 111)
    raw[..., [100, 107]] = 1.
    return (state, torch.zeros(1, 29), torch.zeros(1, 16), torch.zeros(1, 10, 141),
            raw, torch.tensor([float(steps)]))


class SoftTerminationTests(unittest.TestCase):
    def test_v5_entry_uses_the_vector_runner_and_v5_learner(self):
        from carrywm import train_v5
        flags = ["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."]
        args = train_v5.args_from_cli(flags)
        self.assertEqual((args.num_envs, args.collect_steps, args.wm_batch_size, args.batch_size),
                         (1024, 5, 1024, 2048))
        self.assertEqual((args.termination_height_m, args.termination_gravity_z, args.termination_softness),
                         (.25, .8, .05))
        self.assertAlmostEqual(args.critic_value_max, 10.8)
        with self.assertRaises(ValueError):
            train_v5.args_from_cli(flags + ["--drop-penalty", "-1"])
        with patch("carrywm.train_v4.train") as runner:
            train_v5.train(args)
        runner.assert_called_once_with(args, version=5, learner_type=LearnerV5)

    def test_checkpoint_version_isolation(self):
        from carrywm import train_v4, train_v5
        flags = ["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."]
        args = train_v5.args_from_cli(flags)
        checkpoint = {"version": 5, "iteration": 20, "config": vars(args).copy()}
        train_v4.check_resume(checkpoint, args, version=5)
        with self.assertRaisesRegex(ValueError, "V5.*V4"):
            train_v4.check_resume(checkpoint, train_v4.args_from_cli(flags), version=4)
        with self.assertRaisesRegex(ValueError, "V4.*V5"):
            train_v4.check_resume(dict(checkpoint, version=4), args, version=5)
        checkpoint["config"]["drop_penalty"] += 1
        with self.assertRaisesRegex(ValueError, "drop_penalty"):
            train_v4.check_resume(checkpoint, args, version=5)

    def test_relative_pelvis_height_and_projected_gravity(self):
        state = torch.zeros(5, 84, dtype=torch.float64)
        raw = torch.zeros(5, 111, dtype=torch.float64)
        state[:, 3] = raw[:, 100] = 1.
        state[:, 2] = raw[:, 99] = .8
        # Both signs of 25 cm produce the same probability; global yaw is irrelevant.
        state[0, 2] += .25
        state[1, 2] -= .25
        state[2, 3:7] = torch.tensor([0., 0., 0., 1.])
        # Relative gravity-z difference of 0.8 with a tilted reference pelvis.
        state[3, 3:7] = torch.tensor([math.sqrt(.5), 0., math.sqrt(.5), 0.], dtype=state.dtype)
        raw[3, 100:104] = torch.tensor([math.sqrt(.9), math.sqrt(.1), 0., 0.], dtype=raw.dtype)
        # A common tilted pose, including its equivalent negated quaternion, has zero error.
        state[4, 3:7] = -state[3, 3:7]
        raw[4, 100:104] = state[3, 3:7]
        tiny = 1 / (1 + math.exp(20))
        p = soft_termination_probability(state, raw)
        torch.testing.assert_close(p, p.new_tensor([.5, .5, tiny, .5, tiny]), atol=1e-6, rtol=1e-6)

    def test_32_step_rewards_bootstrap_and_actor_gradient(self):
        # Choose p=0.05 so the final value has appreciable weight even after 32 steps.
        p = .05
        height = .25 * (1 + .05 * math.log(p / (1-p)))
        net = make_learner(height)
        ret, survival, *_ = net._rollout(inputs())
        q = GAMMA * (1-p)
        expected = sum(q**t for t in range(HORIZON)) + 2*q**HORIZON
        self.assertAlmostEqual(ret.item(), expected, places=4)
        self.assertAlmostEqual(survival.item(), (1-p)**HORIZON, places=5)
        # Only d(1-p)/d(height) provides a gradient in this controlled rollout.
        ret.sum().backward()
        dq = -GAMMA * p*(1-p)/(.25*.05)
        expected_grad = dq * (sum(t*q**(t-1) for t in range(1,HORIZON)) + 2*HORIZON*q**(HORIZON-1))
        self.assertAlmostEqual(net.actor.height.grad.item(), expected_grad, delta=abs(expected_grad)*2e-5)

        # V4's binary 0.5 m pelvis gate gives the same trajectory full weight.
        v4_return, v4_survival, *_ = make_learner(height, LearnerV3)._rollout(inputs())
        self.assertEqual(v4_survival.item(), 1.)
        self.assertAlmostEqual(v4_return.item(), sum(GAMMA**t for t in range(HORIZON)) + 2*GAMMA**HORIZON,
                               places=4)

    def test_lambda_target_uses_each_next_reference_and_clip_end(self):
        net = make_learner(.25)
        data = inputs(steps=3)
        # First step's hazard is .5. Second step has zero height error, hence
        # effectively zero hazard. Third step hits the end of the reference.
        data[4][:, 2, 99] = .25
        examples, ret, survival = net._imagine(data, 1)
        expected_ret = 1 + GAMMA*.5 + GAMMA**2*.5
        self.assertAlmostEqual(ret.item(), expected_ret, places=5)
        self.assertEqual(survival.item(), 0.)
        self.assertEqual(torch.where(examples.valid)[0].tolist(), [0, 1, 2])
        step2 = 1 + GAMMA*((1-LAMBDA)*2 + LAMBDA*1)
        step1 = 1 + GAMMA*.5*((1-LAMBDA)*2 + LAMBDA*step2)
        torch.testing.assert_close(examples.targets[:3], examples.targets.new_tensor([step1, step2, 1.]))

    def test_bootstrap_caps_positive_estimates_only(self):
        self.assertAlmostEqual(VALUE_MAX, 10.8)
        net = make_learner(0.)
        discounted_rewards = sum(GAMMA**t for t in range(HORIZON))
        for value in (100., -100.):
            with self.subTest(value=value):
                net._value = lambda state, goal, context: state.new_full((len(state),), value)
                ret, _, *_ = net._rollout(inputs())
                expected = discounted_rewards + GAMMA**HORIZON * min(value, 10.8)
                self.assertAlmostEqual(ret.item(), expected, places=4)

    def test_critic_caps_finished_lambda_targets_after_the_recursion(self):
        net = make_learner(.25)
        data = inputs(steps=3)
        data[4][0, :, 0] = torch.arange(HORIZON+51)
        # Step 2's raw target exceeds the bound. The previous target is negative
        # and must still bootstrap from that raw number and the raw critic V1.
        values = torch.zeros(HORIZON+1)
        values[1:3] = torch.tensor([-600., 500.])
        net._value = lambda state, goal, context: values[goal[:, 0, 0].long()]
        examples, _, _ = net._imagine(data, 1)
        raw2 = 1 + GAMMA*.5*((1-LAMBDA)*500 + LAMBDA*1)
        raw1 = 1 + GAMMA*.5*((1-LAMBDA)*-600 + LAMBDA*raw2)
        self.assertGreater(raw2, VALUE_MAX)
        self.assertLess(raw1, 0.)
        torch.testing.assert_close(examples.targets[:3], examples.targets.new_tensor([raw1, VALUE_MAX, 1.]))

    def test_only_hard_boundaries_remove_critic_rows(self):
        net = make_learner(.9)
        # Saturated soft p=1 truncates credit to later rewards. Critic training
        # still includes all three real reference frames up to the hard boundary.
        data = inputs(steps=3)
        examples, ret, survival = net._imagine(data, 1)
        self.assertEqual(ret.item(), 1.)
        self.assertEqual(survival.item(), 0.)
        self.assertEqual(torch.where(examples.valid)[0].tolist(), [0, 1, 2])
        torch.testing.assert_close(examples.targets[:3], torch.ones(3))

        # The carry-specific box divergence remains a real hard stop in imagination.
        data[4][:, 1, 106] = .6
        examples, _, _ = net._imagine(data, 1)
        self.assertEqual(torch.where(examples.valid)[0].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
