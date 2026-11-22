"""Small physical regression with the official carry MJCF and motion clips."""
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(os.getenv("CARRY_SIM_ROOT") and os.getenv("CARRY_WARP_DEVICE"),
                     "set CARRY_SIM_ROOT and CARRY_WARP_DEVICE to run the real physics check")
class WarpPhysicsTests(unittest.TestCase):
    def test_reset_isolation_geometry_and_cpu_one_step(self):
        from carrywm.sim import CarryEnv, _scene
        from carrywm.warp_sim import WarpCarryWorlds

        root = Path(os.environ["CARRY_SIM_ROOT"])
        motion = Path(os.environ.get("CARRY_MOTION_DIR", root / "data/carrybox"))
        worlds = WarpCarryWorlds(root, motion, 2, seed=17, device=os.environ["CARRY_WARP_DEVICE"],
                                 cfgen_ratio=0.)
        cpu = [CarryEnv(root, motion, seed=17+i, cfgen_ratio=0.) for i in range(2)]
        worlds.reset(range(2), source="npz")
        for i, e in enumerate(cpu):
            e.reset("npz")
            np.testing.assert_array_equal(e.state, worlds.envs[i].state)
            np.testing.assert_allclose(e.context, worlds.envs[i].context, atol=1e-6)
        for half in (np.array([.1,.17,.21]), np.array([.2,.13,.13])):
            compiled = _scene(root.resolve(),tuple(half),headless=True)
            for key, values in worlds._geometry(half).items():
                np.testing.assert_allclose(values, getattr(compiled,key).reshape(values.shape),
                                           rtol=1e-5, atol=1e-6, err_msg=key)
        actions = np.repeat(np.linspace(-.1,.1,29,dtype=np.float32)[None], 2, axis=0)
        states, labels, _, infos = worlds.step(actions)
        self.assertEqual(states.shape, (2,84))
        self.assertTrue(np.isin(labels,[0,1]).all())
        for i, e in enumerate(cpu):
            after, _, _, info = e.step(actions[i])
            # CPU and GPU solvers can have distinct numerical trajectories.
            # Translational positions after one 20 ms tick should still be close.
            np.testing.assert_allclose(states[i,:3], after[:3], atol=.025)
            np.testing.assert_allclose(states[i,71:74], after[71:74], atol=.025)
            self.assertEqual(infos[i]["fallen"], info["fallen"])
        before_qpos = worlds.d.qpos.numpy()[1].copy()
        before_qvel = worlds.d.qvel.numpy()[1].copy()
        worlds.reset([0],source="npz")
        np.testing.assert_array_equal(worlds.d.qpos.numpy()[1],before_qpos)
        np.testing.assert_array_equal(worlds.d.qvel.numpy()[1],before_qvel)
        self.assertEqual(worlds.envs[1].index,1)
        self.assertEqual(worlds.envs[0].index,0)

    def test_contact_forces_and_support_stay_in_their_world(self):
        import warp as wp
        from carrywm.warp_sim import _gather_contact

        # IDs 0..4 identify box, floor, origin, destination and left wrist.
        # Last contact has zero normal force and must not mark the box grounded.
        with wp.ScopedDevice(os.environ["CARRY_WARP_DEVICE"]):
            world = wp.array([0, 0, 1, 1], dtype=int)
            pairs = wp.array([[0,4],[0,3],[4,0],[1,0]], dtype=wp.vec2i)
            limbs = wp.array([-1,-1,-1,-1,2],dtype=int)
            count = wp.array([4], dtype=int)
            frame = wp.array(np.repeat(np.eye(3)[None],4,0), dtype=wp.mat33)
            wrench = wp.array([[7.,2.,0,0,0,0],[1.,0,0,0,0,0],
                               [8.,3.,0,0,0,0],[0.,0,0,0,0,0]],dtype=wp.spatial_vector)
            forces = wp.zeros((2,6,3),dtype=float)
            support = wp.zeros((2,3),dtype=int)
            wp.launch(_gather_contact,4,inputs=[count,world,pairs,frame,wrench,limbs,
                                                0,1,3,2,forces,support])
            np.testing.assert_array_equal(forces.numpy()[:,2],[[7,2,0],[-8,-3,0]])
            np.testing.assert_array_equal(support.numpy(),[[0,1,0],[0,0,0]])
