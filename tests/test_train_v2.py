"""Regression tests for episode-local windows, reference timing and discounted targets."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from carrywm.replay import Replay
from carrywm.train import lambda_returns, frozen


class TrainingTest(unittest.TestCase):
    def test_rollouts_keep_clip_context_and_start_history(self):
        bank = Replay(100, seed=11)
        for marker in (0, 100):
            clip = np.tile(np.arange(20,dtype=np.float32)[:, None], (1,111)) + marker
            env = SimpleNamespace(clip=clip, context=[marker], source="npz", name=str(marker), state=[marker])
            seg = bank.begin(env)
            for k in range(10):
                obs = np.full(141,k+marker,np.float32)
                seg.append(k,obs,np.zeros(29),[k+marker],[k+marker],[k+marker+1],{})
                bank.added()
        states, actions, labels, contexts = bank.wm(128,8)
        np.testing.assert_array_equal(states[:,1:,0], actions[:,:,0]+1)
        np.testing.assert_array_equal(actions,labels)
        self.assertTrue(np.all((states[:,:,0]-contexts>=0)&(states[:,:,0]-contexts<=10)))
        s,prev,c,h,raw,left = bank.actor(64,32)
        np.testing.assert_array_equal(raw[:,0,0], s[:,0])
        np.testing.assert_array_equal(h[:,-1,0],s[:,0])
        np.testing.assert_array_equal(h[:,0,0],c[:,0])
        np.testing.assert_array_equal(raw[:,-1,0],c[:,0]+19)
        self.assertEqual(raw.shape,(64,83,111))
        np.testing.assert_array_equal(left,raw[:,-1,0]-s[:,0])

    def test_lambda_returns_do_not_bootstrap_through_terminal(self):
        r = torch.tensor([[1.], [2.], [13.]])
        v = torch.tensor([[5.], [6.], [19.]])
        c = torch.tensor([[1.], [0.], [1.]])
        actual=lambda_returns(r,v,c,gamma=.5,lam=1.)
        torch.testing.assert_close(actual,torch.tensor([[2.],[2.],[22.5]]))

    def test_freeze_keeps_input_gradient(self):
        model=torch.nn.Linear(2,1)
        x=torch.ones(2,2,requires_grad=True)
        with frozen(model):
            model(x).sum().backward()
            self.assertTrue(x.grad.abs().sum()>0)
        self.assertTrue(model.weight.requires_grad)
        self.assertIsNone(model.weight.grad)


if __name__=="__main__":
    unittest.main()
