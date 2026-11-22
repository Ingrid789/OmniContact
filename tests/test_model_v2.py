"""Run with SIM2SIM_ROOT pointing to the official OmniContact_sim2sim checkout."""
import os
from pathlib import Path
import unittest

import torch

from carrywm.validation import validate_dynamics, aggregate_dynamics

from carrywm.model import Actor, CarryGeometry, Critic, WorldModel, world_loss


def scene_state(batch):
    s = torch.zeros(batch, 84)
    s[:, 2] = 0.77
    s[:, 3] = 1
    s[:, 71:74] = torch.tensor([0.5, 0.0, 0.6])
    s[:, 74] = 1
    return s


class DynamicsTests(unittest.TestCase):
    def test_units_and_independent_free_box(self):
        wm = WorldModel(16, width=16)
        for p in wm.parameters():
            p.data.zero_()
        wm.robot.bias.data[6] = 1
        wm.box.bias.data[2] = -0.2
        s = scene_state(3)
        after, contact = wm(s, torch.zeros(3,29), torch.zeros(3,16))
        torch.testing.assert_close(after[:,42], torch.ones(3))
        torch.testing.assert_close(after[:,7], torch.full((3,), 0.0125))
        torch.testing.assert_close(after[:,80], torch.full((3,), -0.2))
        self.assertTrue(torch.all(after[:,73] < s[:,73]))
        self.assertEqual(tuple(contact.shape), (3,6))

    def test_quaternion_sign_invariance_and_finite_gradients(self):
        pred = scene_state(2).requires_grad_()
        truth = pred.detach().clone()
        truth[:, 3:7] *= -1
        truth[:, 74:78] *= -1
        logits = torch.zeros(2,6,requires_grad=True)
        loss = world_loss(pred, truth, logits, torch.zeros_like(logits))
        self.assertAlmostEqual(loss.item(), 0.693147, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_frozen_dynamics_still_transmits_policy_gradients(self):
        ctx = torch.zeros(2,16)
        s = scene_state(2)
        actor = Actor(first_width=32, decoder_width=32, hidden=(16,))
        wm = WorldModel(16,width=32).requires_grad_(False)
        critic = Critic(16, hidden=(16,)).requires_grad_(False)
        history = torch.randn(2,10,141)
        goal = torch.randn(2,11,113)
        actor.update_stats(goal, history)
        for _ in range(3):
            action = actor(goal, history)
            s, _ = wm(s, action, ctx)
            next_obs = torch.cat((s, action, s[...,:28]), -1)
            history = torch.cat((history[:,1:], next_obs.unsqueeze(1)), 1)
        critic(s, goal, ctx).mean().backward()
        self.assertGreater(float(actor.encoder[0].weight.grad.norm()), 0)
        self.assertTrue(torch.isfinite(actor.encoder[0].weight.grad).all())
        self.assertTrue(all(p.grad is None for p in wm.parameters()))

    def test_heldout_windows_never_cross_reset(self):
        # A perfect logged-action predictor; a reset after transition 15
        # removes all 32-step windows in this 40-transition sequence.
        class KnownTransition:
            def __call__(self, s, a, ctx):
                pred = s.clone()
                pred[:,71] += a[:,0]
                return pred, torch.zeros(len(s), 6)
        count = 40
        s = scene_state(count+1)
        s[:,71] = torch.arange(count+1)*.1
        actions = torch.zeros(count,29)
        actions[:,0] = .1
        labels = torch.zeros(count,6)
        labels[5:10,2:4] = 1
        targets = torch.zeros(count+1,111)
        targets[6:15,37:39] = 1
        terminals = torch.zeros(count,dtype=torch.bool)
        terminals[[15,39]] = True
        events = [dict(lifted=(i>=5), supported=(i==14)) for i in range(count)]
        metrics = validate_dynamics(KnownTransition(),s,actions,torch.zeros(16),labels,targets,terminals,
                                    events=events,stride=1)
        self.assertNotIn("h32/all/windows", metrics)
        self.assertEqual(metrics["h08/all/windows"],(16-7)+(24-7))
        self.assertEqual(metrics["h01/slip/windows"],4)
        self.assertEqual(metrics["h01/landing/windows"],1)
        self.assertLess(metrics["h08/all/box_position_m"],1e-5)
        self.assertAlmostEqual(metrics["h08/all/stationary_box_position_m"],.8,places=5)
        merged = aggregate_dynamics([metrics,metrics])
        self.assertEqual(merged["h01/all/windows"],2*count)
        self.assertEqual(merged["h01/all/box_position_m"],metrics["h01/all/box_position_m"])

    def test_fk_and_features_match_independent_mujoco(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo required for FK cross-validation")
        candidates = (os.environ.get("SIM2SIM_ROOT"), os.environ.get("OMNICONTACT_SIM2SIM"),
                      "../OmniContact_sim2sim","../papers/OmniContact/code-sim2sim")
        root = next((Path(p) for p in candidates if p and (Path(p)/"g1_description/g1_29dof.xml").is_file()), None)
        if root is None:
            self.skipTest("Set SIM2SIM_ROOT to run the physics cross-validation")
        geom = CarryGeometry(root)
        model = mujoco.MjModel.from_xml_path(str(root/"g1_description/g1_29dof.xml"))
        data = mujoco.MjData(model)
        s = scene_state(1)
        s[:,3:7] = torch.tensor([0.5,0.5,0.5,0.5])
        s[:,7:36] = geom.default + torch.linspace(-0.12,0.12,29)
        data.qpos[:] = s[0,:36].numpy()
        mujoco.mj_forward(model,data)
        fk = geom.forward_kinematics(s)
        for i, name in enumerate(geom.names[1:],1):
            torch.testing.assert_close(fk[0,i,:3,3],torch.from_numpy(data.body(name).xpos.copy()).float(),atol=2e-5,rtol=1e-5)
            torch.testing.assert_close(fk[0,i,:3,:3],torch.from_numpy(data.body(name).xmat.copy()).reshape(3,3).float(),atol=2e-5,rtol=1e-5)
        target = torch.zeros(1,111)
        for i, name in enumerate(geom.tracked):
            target[0,7*i:7*i+3] = torch.from_numpy(data.body(name).xpos.copy())
            target[0,7*i+3:7*i+7] = torch.from_numpy(data.body(name).xquat.copy())
        target[:,39:68] = s[:,7:36]
        target[:,97:104] = s[:,:7]
        target[:,104:111] = s[:,71:78]
        ctx = torch.ones(1,16)*0.1
        history_frame = geom.observation(s, torch.zeros(1,29), ctx, fk)
        self.assertEqual(tuple(history_frame.shape), (1,141))
        window = geom.reference(s, target[:,None].expand(-1,11,-1), fk)
        self.assertEqual(tuple(window.shape), (1,11,113))
        torch.testing.assert_close(window[:,0,49:78], s[:,7:36])
        torch.testing.assert_close(window[:,0,107:113], torch.tensor([[1.,0.,0.,1.,0.,0.]]))
        terms = geom.reward_terms(s,s,torch.zeros(1,29),torch.zeros(1,29),target,torch.zeros(1,6),prev_fk=fk,fk=fk)
        for name, val in (("wrist_pos",.008),("body_rot",.004),("box_pos",.02),("foot_slip",0.)):
            self.assertAlmostEqual(float(terms[name]), val, places=5)
        s.requires_grad_()
        scalar = geom.observation(s, torch.zeros(1,29), ctx).sum() + geom.reference(s,target.unsqueeze(1)).sum()
        scalar.backward()
        self.assertTrue(torch.isfinite(s.grad).all())


if __name__ == "__main__":
    unittest.main()
