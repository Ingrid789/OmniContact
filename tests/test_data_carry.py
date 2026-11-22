"""Data and simulator regression checks, independent of Isaac Lab."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from carrywm.data import DT, LAB2MJ, generate_clip, load_demo, raw_window, split_paths


class ReferenceTests(unittest.TestCase):
    def test_generation_differences_use_dense_time(self):
        identity = np.array([1.,0.,0.,0.])
        class Stub:
            def generate(self, *_args, **_kwargs):
                n = 75
                q = np.square(np.arange(n, dtype=float))[:, None] * np.ones((1,29))
                p = np.zeros((n,3))
                r = np.tile(identity,(n,1))
                t = dict(dof_pos=q,ref_contact=np.ones((n,4)),ref_phase=np.arange(n),
                         ref_base_pos=p,ref_base_quat=r,ref_object_pos=p,ref_object_quat=r)
                for name in ("left_wrist","right_wrist","torso_future","left_ankle_future","right_ankle_future"):
                    t[f"ref_{name}_pos"], t[f"ref_{name}_quat"] = p,r
                return t, 0.
        pose = np.r_[np.zeros(3),identity]
        clip, _ = generate_clip(Stub(),pose,pose,np.ones(3),np.zeros(3))
        window = raw_window(clip,10)
        self.assertEqual(window.shape,(11,111))
        np.testing.assert_allclose(window[-1,68:97],2.*60 / DT)
        np.testing.assert_allclose(raw_window(clip,74),np.repeat(clip[-1:],11,axis=0))

    def test_split_preserves_source_across_size_and_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            for source in range(30):
                for dims,cut in (("20-20-20","000"),("30-20-30","001")):
                    Path(tmp,f"carry_box_{dims}_{source}_12345_{cut}.npz").touch()
            groups = [{p.name.split("_")[3] for p in split_paths(tmp, s)} for s in ("train","val","test")]
            self.assertEqual(sum(map(len,groups)), len(set.union(*groups)))
            self.assertEqual(len(set.union(*groups)),30)

    @unittest.skipUnless(os.getenv("CARRY_SIM_ROOT"),"set CARRY_SIM_ROOT to the official MuJoCo checkout")
    def test_npz_targets_agree_with_scene_fk(self):
        import mujoco as mj
        root = Path(os.environ["CARRY_SIM_ROOT"])
        sample = sorted((root/"data/carrybox").glob("*.npz"))[0]
        scene = load_demo(sample)
        m = mj.MjModel.from_xml_path(str(root/"g1_description/g1_29dof.xml"))
        d = mj.MjData(m)
        f = scene.frames[len(scene.frames)//2]
        d.qpos[:7], d.qpos[7:36] = f[97:104], f[39:68]
        mj.mj_forward(m,d)
        for j,n in enumerate(("left_palm_link","right_palm_link","torso_link","left_ankle_pitch_link","right_ankle_pitch_link")):
            b=m.body(n).id
            np.testing.assert_allclose(d.xpos[b], f[j*7:j*7+3],atol=.002)
            self.assertGreater(abs(d.xquat[b] @ f[j*7+3:j*7+7]), .999)
        with np.load(sample) as npz:
            np.testing.assert_allclose(f[39:68],npz["joint_pos"][len(scene.frames)//2,LAB2MJ])

    @unittest.skipUnless(os.getenv("CARRY_SIM_ROOT"),"set CARRY_SIM_ROOT to the official MuJoCo checkout")
    def test_physical_box_has_reported_geometry_and_finite_step(self):
        from carrywm.sim import CarryEnv
        root = Path(os.environ["CARRY_SIM_ROOT"])
        env = CarryEnv(root,root/"data/carrybox",seed=42,cfgen_ratio=0)
        env.reset()
        np.testing.assert_allclose(env.m.geom("box_geom").size,env.context[:3])
        np.testing.assert_allclose(env.m.body("box").inertia,env.context[4:7])
        self.assertEqual((env.state.shape,env.context.shape),((84,),(16,)))
        state,labels,done,info = env.step(np.zeros(29))
        self.assertTrue(np.isfinite(state).all())
        self.assertEqual(labels.shape,(6,))
        self.assertTrue(np.isin(labels,[0,1]).all())
        self.assertFalse(info["lifted"])
        self.assertEqual(env.source,"npz")


if __name__ == "__main__":
    unittest.main()
