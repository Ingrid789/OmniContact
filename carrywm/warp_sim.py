"""Batched, 50 Hz MuJoCo Warp carry physics; one MjWarp Data for all worlds."""
from contextlib import nullcontext
import copy
from functools import lru_cache
from types import SimpleNamespace

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .data import generate_clip, load_demo
from .sim import CarryEnv, _generator, _scene


@wp.kernel
def _gather_contact(nacon: wp.array[int], worlds: wp.array[int], geoms: wp.array[wp.vec2i],
                    frames: wp.array[wp.mat33], wrenches: wp.array[wp.spatial_vector],
                    limbs: wp.array[int], box: int, floor: int, target: int, origin: int,
                    forces: wp.array3d[float], support: wp.array2d[int]):
    i = wp.tid()
    if i >= nacon[0]:
        return
    wrench = wrenches[i]
    force = wp.vec3(wrench[0], wrench[1], wrench[2])
    frame = frames[i]
    normal = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
    if wp.dot(force, normal) <= 0.0:
        return
    world = worlds[i]
    g = geoms[i]
    for k in range(2):
        geom = g[k]
        other = g[1-k]
        limb = limbs[geom]
        sign = float(2*k-1)
        if limb == 0 or limb == 1:
            for axis in range(3):
                wp.atomic_add(forces, world, 4+limb, axis, sign*force[axis])
        if other == box:
            if limb >= 0:
                for axis in range(3):
                    wp.atomic_add(forces, world, limb, axis, sign*force[axis])
            if geom == floor:
                wp.atomic_max(support, world, 0, 1)
            elif geom == target:
                wp.atomic_max(support, world, 1, 1)
            elif geom == origin:
                wp.atomic_max(support, world, 2, 1)


@wp.kernel
def _check_capacity(nacon: wp.array[int], nefc: wp.array[int], pairs: wp.array[int],
                    capacity: int, constraints: int, overflow: wp.array[int]):
    w = wp.tid()
    if w == 0 and nacon[0] > capacity:
        wp.atomic_max(overflow, 0, nacon[0])
    if w == 0 and pairs[0] > capacity:
        wp.atomic_max(overflow, 2, pairs[0])
    if nefc[w] > constraints:
        wp.atomic_max(overflow, 1, nefc[w])


@lru_cache(maxsize=256)
def _cached_plan(root, pose, box, half, goal):
    # Official carry IK uses deterministic inputs. Cache only the zero-noise plans.
    return generate_clip(_generator(root), *(np.asarray(x) for x in (pose, box, half, goal)))


class _Episode(CarryEnv):
    """Per-world reference and episode logic; physical state lives in MjWarp Data."""

    def _load_scene(self, path):
        # One packed SceneClip per selected source file across all worlds.
        if path not in self.scenes:
            self.scenes[path] = load_demo(path)
        return self.scenes[path]

    def _bind_scene(self, half):
        self.half = np.asarray(half, dtype=float)
        self.d.qpos[:] = self.m.qpos0
        self.d.qvel.fill(0)

    def _forward_reset(self):
        # The owner writes all reset worlds together after references are initialized.
        pass

    def _generate_reference(self, scene):
        if self.cfgen_goal_noise_std_m or self.split != "train":
            return super()._generate_reference(scene)
        return _cached_plan(self.root, tuple(scene.frames[0,97:104]), tuple(scene.frames[0,104:111]),
                            tuple(self.half), tuple(self.goal))

    def _update_teacher_history(self):
        if self.teacher_history and not self.in_reset:
            super()._update_teacher_history()

    def teacher_action(self):
        if not self.teacher_history:
            raise RuntimeError("Enable teacher_history to collect teacher actions")
        return super().teacher_action()


class WarpCarryWorlds:
    """All worlds share topology and independently sample clips, box sizes and reset times."""

    def __init__(self, sim_root, motion_dir, num_envs, seed=42, split="train", cfgen_ratio=.1,
                 cfgen_goal_noise_std_m=0., device="cuda:0", nconmax=64, njmax=256,
                 teacher_history=False):
        if num_envs < 1 or nconmax < 1 or njmax < 1:
            raise ValueError("World count and contact/constraint capacities must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.teacher_history = teacher_history
        prototype = _Episode(sim_root, motion_dir, seed, split, cfgen_ratio, cfgen_goal_noise_std_m)
        model = _scene(prototype.root, (.15, .15, .15), headless=True)
        self._host = model
        self._box = model.geom("box_geom").id
        self._box_body = model.body("box").id
        self._scratch = copy.copy(model)
        self._scratch_data = mj.MjData(self._scratch)
        self._geometry_cache = {}
        self.envs = []
        prototype.scenes = {}
        for i in range(num_envs):
            e = copy.copy(prototype)
            e.rng = np.random.default_rng(seed+i)
            e.teacher_history, e.in_reset = teacher_history, False
            # Only a few host arrays for the existing reset, teacher and recovery logic.
            shadow = SimpleNamespace(qpos=model.qpos0.copy(), qvel=np.zeros(model.nv),
                                     mocap_pos=np.zeros((model.nmocap, 3)))
            if teacher_history:
                shadow.xpos = np.zeros((model.nbody, 3))
                shadow.xmat = np.zeros((model.nbody, 9))
            e._set_scene(model, shadow, (.15, .15, .15))
            self.envs.append(e)

        with wp.ScopedDevice(str(self.device)), self._stream():
            self.m = mjw.put_model(model)
            self.d = mjw.make_data(model, nworld=num_envs, nconmax=nconmax, njmax=njmax)
            # Expand the model fields whose compiled values vary with box dimensions.
            self._model_views = {}
            for key in ("geom_size","geom_aabb","geom_rbound","body_inertia","body_invweight0","dof_invweight0"):
                arr = getattr(self.m, key)
                expanded = wp.array(np.repeat(arr.numpy(), num_envs, axis=0), dtype=arr.dtype)
                setattr(self.m, key, expanded)
                self._model_views[key] = wp.to_torch(expanded)
            self._ids = wp.array(np.arange(self.d.naconmax, dtype=np.int32), dtype=int)
            self._wrenches = wp.zeros(self.d.naconmax, dtype=wp.spatial_vector)
            self._forces = wp.zeros((num_envs, 6, 3), dtype=float)
            self._support = wp.zeros((num_envs, 3), dtype=int)
            self._overflow = wp.zeros(3, dtype=int)
            geom_limbs = [e._limb.get(int(body), -1) for body in model.geom_bodyid]
            self._limbs = wp.array(geom_limbs, dtype=int)
            self._qpos = wp.to_torch(self.d.qpos)
            self._qvel = wp.to_torch(self.d.qvel)
            self._ctrl = wp.to_torch(self.d.ctrl)
            self._mocap = wp.to_torch(self.d.mocap_pos)
            self._pd = [torch.as_tensor(x, dtype=torch.float32, device=self.device)
                        for x in (e.default, e.scale, e.kp, e.kd, e.limit)]

    def _stream(self):
        if self.device.type == "cuda":
            return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))
        return nullcontext()

    def _geometry(self, half):
        key = tuple(half)
        if key not in self._geometry_cache:
            m = self._scratch
            g, body = self._box, self._box_body
            m.geom_size[g] = half
            m.geom_aabb[g, 3:] = half
            m.geom_rbound[g] = np.linalg.norm(half)
            x,y,z = 2*np.asarray(half)
            m.body_inertia[body] = m.body_mass[body] / 12 * np.array([y*y+z*z,x*x+z*z,x*x+y*y])
            mj.mj_setConst(m, self._scratch_data)
            values = {}
            for name, view in self._model_views.items():
                values[name] = getattr(m, name).reshape(tuple(view.shape[1:])).astype(np.float32)
            self._geometry_cache[key] = values
        return self._geometry_cache[key]

    def _write_rows(self, view, ids, values):
        view.index_copy_(0, ids, torch.as_tensor(np.asarray(values), device=self.device, dtype=view.dtype))

    def reset(self, ids, source=None):
        ids = list(map(int, ids))
        if not ids:
            return
        if len(set(ids)) != len(ids) or min(ids)<0 or max(ids)>=self.num_envs:
            raise ValueError("Invalid or duplicate world IDs")
        new = []
        variants = []
        for i in ids:
            e = self.envs[i]
            e.in_reset = True
            e.reset(source)
            e.in_reset = False
            v = self._geometry(e.half)
            e.context[4:7] = v["body_inertia"][self._box_body]
            if e.source != "recovery":
                new.append(i)
                variants.append(v)
        with wp.ScopedDevice(str(self.device)), self._stream():
            if new:
                mask = np.zeros(self.num_envs, dtype=bool)
                mask[new] = True
                mjw.reset_data(self.m, self.d, wp.array(mask, dtype=bool))
                idx = torch.tensor(new, device=self.device)
                for key, view in self._model_views.items():
                    self._write_rows(view, idx, [v[key] for v in variants])
                for view, key in ((self._qpos,"qpos"),(self._qvel,"qvel"),(self._mocap,"mocap_pos")):
                    self._write_rows(view, idx, [getattr(self.envs[i].d,key) for i in new])
            if self.teacher_history:
                mjw.kinematics(self.m, self.d)
                self._read_teacher_pose(ids)
                for i in ids:
                    self.envs[i]._update_teacher_history()

    def _read_teacher_pose(self, ids):
        pos = self.d.xpos.numpy()
        mat = self.d.xmat.numpy().reshape(self.num_envs, self._host.nbody, 9)
        for i in ids:
            self.envs[i].d.xpos[:] = pos[i]
            self.envs[i].d.xmat[:] = mat[i]

    def _check_buffers(self):
        wp.launch(_check_capacity, self.num_envs,
                  inputs=[self.d.nacon, self.d.nefc, self.d.ncollision,
                          self.d.naconmax, self.d.njmax, self._overflow])

    def step(self, actions):
        if any(e._done for e in self.envs):
            raise RuntimeError("Reset every terminal world before stepping the vector environment")
        with wp.ScopedDevice(str(self.device)), self._stream(), torch.no_grad():
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
            if actions.shape != (self.num_envs, 29) or not torch.isfinite(actions).all():
                raise ValueError("Expected a finite [num_envs,29] action matrix")
            default, scale, kp, kd, limit = self._pd
            target = default + scale*actions
            for _ in range(4):
                torque = kp*(target-self._qpos[:,7:36]) - kd*self._qvel[:,6:35]
                self._ctrl.copy_(torch.clamp(torque, -limit, limit))
                mjw.step(self.m, self.d)
                self._check_buffers()
            mjw.forward(self.m, self.d)
            self._check_buffers()
            self._forces.zero_()
            self._support.zero_()
            mjw.contact_force(self.m, self.d, self._ids, True, self._wrenches)
            e = self.envs[0]
            wp.launch(_gather_contact, self.d.naconmax,
                      inputs=[self.d.nacon,self.d.contact.worldid,self.d.contact.geom,
                              self.d.contact.frame,self._wrenches,self._limbs,
                              self._box,e.floor_geom,e.tables_geom[1],e.tables_geom[0],
                              self._forces,self._support])
            overflow = self._overflow.numpy()
            if np.any(overflow):
                raise RuntimeError(f"MuJoCo Warp capacity exceeded (contacts, constraints, broadphase pairs): {overflow}")
            contacts = (np.linalg.norm(self._forces.numpy(),axis=2) > [0,0,5,5,1,1]).astype(np.float32)
            supported = self._support.numpy().astype(bool)
            qp, qv, act = self.d.qpos.numpy(), self.d.qvel.numpy(), actions.cpu().numpy()
            if self.device.type == "cpu":
                # CPU Warp exposes live views; keep host episode reset staging independent.
                qp, qv = qp.copy(), qv.copy()
            if not np.isfinite(qp).all() or not np.isfinite(qv).all():
                raise FloatingPointError("Non-finite MuJoCo Warp physics state")
            if self.teacher_history:
                self._read_teacher_pose(range(self.num_envs))
        states, dones, infos = [], [], []
        for i,e in enumerate(self.envs):
            e.d.qpos, e.d.qvel = qp[i], qv[i]
            state, _, done, info = e._finish_step(act[i], (contacts[i], *supported[i]))
            states.append(state)
            dones.append(done)
            infos.append(info)
        return np.stack(states), contacts, np.asarray(dones), infos
