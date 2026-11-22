"""Batched, 50 Hz MuJoCo Warp carry physics; one MjWarp Data for all worlds."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import copy
from functools import lru_cache
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
import warnings
import weakref

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .data import generate_clip, load_demo
from .sim import CarryEnv, _generator, _matrix, _scene


_UNSET = object()


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


@wp.kernel
def _pd_ctrl(target: wp.array2d[float], qpos: wp.array2d[float], qvel: wp.array2d[float],
             kp: wp.array[float], kd: wp.array[float], limit: wp.array[float],
             ctrl: wp.array2d[float]):
    w, j = wp.tid()
    tau = kp[j] * (target[w, j] - qpos[w, 7 + j]) - kd[j] * qvel[w, 6 + j]
    lim = limit[j]
    ctrl[w, j] = wp.clamp(tau, -lim, lim)


def finish_step_gpu(state, contacts, support, initial_box, goal, goal_on_floor,
                    env_index, clip_len, lifted, carried_steps, lost_steps, stable_steps):
    """Vectorized torch equivalent of CarryEnv._finish_step across all parallel worlds."""
    zero_i = env_index.new_zeros(())
    next_index = torch.minimum(env_index + 1, clip_len - 1)
    ground = support[:, 0]
    dest = support[:, 1]
    origin = support[:, 2]
    hand = (contacts[:, 2] > 0.5) | (contacts[:, 3] > 0.5)
    displacement = state[:, 71:74] - initial_box
    cleared_initial_pose = (displacement[:, 2] > 0.025) | (displacement[:, :2].norm(dim=-1) > 0.1)
    held_free = hand & ~(ground | dest | origin) & cleared_initial_pose
    next_carried = torch.where(held_free, carried_steps + 1, zero_i)
    next_lifted = lifted | (next_carried >= 2)
    next_lost = torch.where(next_lifted & ~hand, lost_steps + 1, zero_i)
    err = (state[:, 71:74] - goal).norm(dim=-1)
    speed = state[:, 78:81].norm(dim=-1)
    ang_speed = state[:, 81:84].norm(dim=-1)
    stable = (speed < 0.1) & (ang_speed < 0.2)
    on_goal_support = torch.where(goal_on_floor, ground, dest)
    supported_goal = next_lifted & on_goal_support & (err < 0.2) & stable & ~hand
    next_stable = torch.where(supported_goal, stable_steps + 1, zero_i)
    success = next_stable >= 5
    off_table = ground & (displacement[:, 2] < -0.08) & ~hand
    lost_grasp = (next_lost >= 3) & (ground | origin | dest)
    drop = (off_table | lost_grasp) & ((err >= 0.2) | ~on_goal_support) & stable
    q = state[:, 3:7]
    q_norm = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    r22 = 1.0 - 2.0 * (q_norm[:, 1].square() + q_norm[:, 2].square())
    fallen = (state[:, 2] < 0.3) | (r22 < 0.35)
    timeout = (next_index == (clip_len - 1)) & ~(fallen | drop | success)
    done = drop | fallen | timeout | success
    return (next_index, next_lifted, next_carried, next_lost, next_stable,
            done, success, drop, timeout, fallen, err)


def _init_cpu_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _cfgen_plan_worker(root_str, pose, box, half, goal):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return generate_clip(_generator(Path(root_str)),
                         np.asarray(pose), np.asarray(box),
                         np.asarray(half, dtype=float), np.asarray(goal, dtype=np.float32))


def _recovery_worker(root_str, root_pose, box_pose, half, goal):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gen = _generator(Path(root_str))
    try:
        frames, phase = generate_clip(gen, root_pose, box_pose, half, goal)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None
    grasp = np.flatnonzero((frames[:, 37:39] > .5).all(axis=1))
    if not len(grasp):
        return None
    first = frames[grasp[0]]
    face, heading, _ = gen._get_target_face_info(box_pose[:3], box_pose[3:7], root_pose[:3], half)
    targets = gen._grasp_targets(box_pose[:3], face, heading)
    for i, name in enumerate(("lw_contact", "rw_contact")):
        palm = first[i * 7:i * 7 + 7]
        offset = np.array([.1285, .012 if i == 0 else -.012, 0.])
        pad = palm[:3] + _matrix(palm[3:]) @ offset
        if np.linalg.norm(pad - targets[name]) > .05:
            return None
    return frames, phase


def _plan_key(pose, box, half, goal):
    return (
        np.asarray(pose, dtype=np.float32).tobytes(),
        np.asarray(box, dtype=np.float32).tobytes(),
        np.asarray(half, dtype=np.float64).tobytes(),
        np.asarray(goal, dtype=np.float32).tobytes(),
    )


def _make_plan(root, pose, box, half, goal):
    # Official carry IK uses deterministic inputs. Cache only the zero-noise plans.
    return generate_clip(_generator(root), *(np.asarray(x) for x in (pose, box, half, goal)))


def _cache_signature(root, table2_size_xy, paths):
    h = hashlib.sha256()
    h.update(str(root).encode("utf-8"))
    h.update(np.asarray(table2_size_xy, dtype=np.float64).tobytes())
    for rel in ("policy/omnicontact/CFgen_meta2_carrybox.py", "policy/omnicontact/CFgen_builder.py"):
        p = root / rel
        if p.exists():
            st = p.stat()
            h.update(f"{rel}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
    for p in paths:
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
    return h.hexdigest()[:16]


def _warm_cfgen_cache(prototype, table2_geom, model, cache_dir, plan_workers):
    table2_size = model.geom_size[table2_geom, :2].copy()
    specs = []
    seen = set()
    for path in prototype.paths:
        scene = prototype.scenes[path]
        pose = scene.frames[0, 97:104].copy()
        box = scene.frames[0, 104:111].copy()
        half = np.asarray(scene.half, dtype=np.float64).copy()
        goal = scene.frames[-1, 104:107].copy()
        reach = np.maximum(table2_size - np.linalg.norm(half[:2]), 0.0)
        center = scene.tables[1, :2]
        goal[:2] = np.clip(goal[:2], center - reach, center + reach)
        key = _plan_key(pose, box, half, goal)
        if key not in seen:
            seen.add(key)
            specs.append((key, pose, box, half, goal))

    if not specs:
        return

    cache_root = Path(cache_dir) if cache_dir is not None else Path(
        os.environ.get("CARRYWM_CACHE_DIR", Path.home() / ".cache" / "carrywm")
    )
    sig = _cache_signature(prototype.root, table2_size, prototype.paths)
    cache_path = cache_root / f"cfgen_zero_noise_{sig}.npz"

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if int(data["count"][0]) == len(specs):
                    for idx, (key, _, _, _, _) in enumerate(specs):
                        prototype._plan_cache[key] = (
                            np.array(data[f"f_{idx}"], dtype=np.float32, copy=True),
                            np.array(data[f"p_{idx}"], copy=True),
                        )
                    return
        except (OSError, ValueError, KeyError) as exc:
            warnings.warn(f"Ignoring unreadable CF-Gen cache {cache_path}: {exc}", RuntimeWarning)

    workers = plan_workers if plan_workers is not None else min(16, max(1, (os.cpu_count() or 2) // 2))
    results = []
    if workers > 1 and len(specs) > 2:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(specs)), mp_context=ctx,
                                 initializer=_init_cpu_worker) as pool:
            futs = [pool.submit(_cfgen_plan_worker, str(prototype.root), pose, box, half, goal)
                    for _, pose, box, half, goal in specs]
            results = [f.result() for f in futs]
    else:
        results = [_make_plan(prototype.root, pose, box, half, goal)
                   for _, pose, box, half, goal in specs]

    payload = {"count": np.array([len(specs)], dtype=np.int64)}
    for idx, ((key, _, _, _, _), (frames, phase)) in enumerate(zip(specs, results)):
        frames_f32 = np.ascontiguousarray(frames, dtype=np.float32)
        phase_arr = np.ascontiguousarray(phase)
        prototype._plan_cache[key] = (frames_f32, phase_arr)
        payload[f"f_{idx}"] = frames_f32
        payload[f"p_{idx}"] = phase_arr

    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npz")
        with tmp.open("wb") as f:
            np.savez(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cache_path)
    except OSError as exc:
        warnings.warn(f"Could not write CF-Gen cache {cache_path}: {exc}", RuntimeWarning)


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
        key = _plan_key(scene.frames[0, 97:104], scene.frames[0, 104:111], self.half, self.goal)
        cached = self._plan_cache.get(key)
        if cached is not None:
            return cached
        return self.reference_plan(self.root, tuple(scene.frames[0, 97:104]),
                                   tuple(scene.frames[0, 104:111]),
                                   tuple(self.half), tuple(self.goal))

    def _try_recovery(self):
        pre = getattr(self, "_precomputed_recovery", _UNSET)
        if pre is not _UNSET:
            self._precomputed_recovery = _UNSET
            if pre is None:
                return False
            self.clip, self.phase = pre
            return True
        return super()._try_recovery()

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
                 teacher_history=False, cfgen_cache_dir=None, plan_workers=None,
                 recovery_workers=None, use_cuda_graph=True):
        if num_envs < 1 or nconmax < 1 or njmax < 1:
            raise ValueError("World count and contact/constraint capacities must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.teacher_history = teacher_history
        self._use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self._step_graph = None
        self._graph_warmed = False
        self._graph_failed = False
        self._rec_pool = None
        self._recovery_workers = (
            recovery_workers if recovery_workers is not None
            else min(8, max(1, (os.cpu_count() or 2) // 4))
        )

        prototype = _Episode(sim_root, motion_dir, seed, split, cfgen_ratio, cfgen_goal_noise_std_m)
        model = _scene(prototype.root, (.15, .15, .15), headless=True)
        self._host = model
        self._box = model.geom("box_geom").id
        self._box_body = model.body("box").id
        self._scratch = copy.copy(model)
        self._scratch_data = mj.MjData(self._scratch)
        self._geometry_cache = {}
        self._geometry_gpu_cache = {}
        self.envs = []
        prototype.scenes = {path: load_demo(path) for path in prototype.paths}
        prototype._plan_cache = {}
        # Each fixed training file supplies at most one initial zero-noise plan.
        # Recovery references still use the actual live robot and box poses.
        prototype.reference_plan = lru_cache(maxsize=len(prototype.paths))(_make_plan)

        for i in range(num_envs):
            e = copy.copy(prototype)
            e.rng = np.random.default_rng(seed + i)
            e.teacher_history, e.in_reset = teacher_history, False
            e._precomputed_recovery = _UNSET
            # Only a few host arrays for the existing reset, teacher and recovery logic.
            shadow = SimpleNamespace(qpos=model.qpos0.copy(), qvel=np.zeros(model.nv),
                                     mocap_pos=np.zeros((model.nmocap, 3)))
            if teacher_history:
                shadow.xpos = np.zeros((model.nbody, 3))
                shadow.xmat = np.zeros((model.nbody, 9))
            e._set_scene(model, shadow, (.15, .15, .15))
            self.envs.append(e)

        e0 = self.envs[0]
        self._bp = e0.bp
        self._bv = e0.bv
        self._table2_id = e0.table_ids[1]
        self._table2_geom = e0.tables_geom[1]
        self._table2_half_z = float(model.geom_size[self._table2_geom, 2])

        if split == "train" and cfgen_ratio > 0 and cfgen_goal_noise_std_m == 0:
            _warm_cfgen_cache(prototype, self._table2_geom, model, cfgen_cache_dir, plan_workers)

        if self.device.type == "cuda":
            self._cuda_stream = torch.cuda.Stream(device=self.device)
            self._wp_stream = wp.stream_from_torch(self._cuda_stream)
        else:
            self._cuda_stream = None
            self._wp_stream = None

        with wp.ScopedDevice(str(self.device)), self._stream():
            self.m = mjw.put_model(model)
            self.d = mjw.make_data(model, nworld=num_envs, nconmax=nconmax, njmax=njmax)
            # Expand the model fields whose compiled values vary with box dimensions.
            self._model_views = {}
            for key in ("geom_size", "geom_aabb", "geom_rbound", "body_inertia",
                        "body_invweight0", "dof_invweight0"):
                arr = getattr(self.m, key)
                expanded = wp.array(np.repeat(arr.numpy(), num_envs, axis=0), dtype=arr.dtype)
                setattr(self.m, key, expanded)
                self._model_views[key] = wp.to_torch(expanded)
            self._ids = wp.array(np.arange(self.d.naconmax, dtype=np.int32), dtype=int)
            self._wrenches = wp.zeros(self.d.naconmax, dtype=wp.spatial_vector)
            self._forces = wp.zeros((num_envs, 6, 3), dtype=float)
            self._support = wp.zeros((num_envs, 3), dtype=int)
            self._overflow = wp.zeros(3, dtype=int)
            self._forces_torch = wp.to_torch(self._forces)
            self._support_torch = wp.to_torch(self._support)
            self._overflow_torch = wp.to_torch(self._overflow)
            geom_limbs = [e._limb.get(int(body), -1) for body in model.geom_bodyid]
            self._limbs = wp.array(geom_limbs, dtype=int)
            self._qpos = wp.to_torch(self.d.qpos)
            self._qvel = wp.to_torch(self.d.qvel)
            self._ctrl = wp.to_torch(self.d.ctrl)
            self._mocap = wp.to_torch(self.d.mocap_pos)
            self._pd = [torch.as_tensor(x, dtype=torch.float32, device=self.device)
                        for x in (e.default, e.scale, e.kp, e.kd, e.limit)]
            self._target_wp = wp.zeros((num_envs, 29), dtype=float)
            self._target = wp.to_torch(self._target_wp)
            self._kp_wp = wp.from_torch(self._pd[2], dtype=wp.float32)
            self._kd_wp = wp.from_torch(self._pd[3], dtype=wp.float32)
            self._limit_wp = wp.from_torch(self._pd[4], dtype=wp.float32)
            self._reset_mask_wp = wp.zeros(num_envs, dtype=bool)
            self._reset_mask_torch = wp.to_torch(self._reset_mask_wp)

        # Preallocated GPU per-world episode tracking buffers.
        self._contact_thresh = torch.tensor([0., 0., 5., 5., 1., 1.], dtype=torch.float32, device=self.device)
        self._initial_box = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self._goal_gpu = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self._goal_on_floor = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._clip_len = torch.ones(num_envs, dtype=torch.int32, device=self.device)
        self._env_index = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._lifted = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._carried_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._lost_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._stable_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._state_gpu = torch.zeros((num_envs, 84), dtype=torch.float32, device=self.device)
        self._context_gpu = torch.zeros((num_envs, 16), dtype=torch.float32, device=self.device)
        self._packed_flags = torch.empty((num_envs + 1, 5), dtype=torch.uint8, device=self.device)
        self._finalizer = weakref.finalize(self, _shutdown_pool_ref, self)

    def _stream(self):
        if self._wp_stream is not None:
            return wp.ScopedStream(self._wp_stream)
        return nullcontext()

    def close(self):
        pool = getattr(self, "_rec_pool", None)
        if pool is not None:
            self._rec_pool = None
            pool.shutdown(wait=True)

    def _ensure_rec_pool(self):
        if self._rec_pool is None:
            ctx = mp.get_context("spawn")
            self._rec_pool = ProcessPoolExecutor(
                max_workers=self._recovery_workers, mp_context=ctx, initializer=_init_cpu_worker
            )
        return self._rec_pool

    def _geometry(self, half):
        key = tuple(half)
        if key not in self._geometry_cache:
            m = self._scratch
            g, body = self._box, self._box_body
            m.geom_size[g] = half
            m.geom_aabb[g, 3:] = half
            m.geom_rbound[g] = np.linalg.norm(half)
            x, y, z = 2 * np.asarray(half)
            m.body_inertia[body] = m.body_mass[body] / 12 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
            mj.mj_setConst(m, self._scratch_data)
            values = {}
            for name, view in self._model_views.items():
                values[name] = getattr(m, name).reshape(tuple(view.shape[1:])).astype(np.float32)
            self._geometry_cache[key] = values
        return self._geometry_cache[key]

    def _geometry_gpu(self, half):
        key = tuple(half)
        if key not in self._geometry_gpu_cache:
            cpu_vals = self._geometry(half)
            self._geometry_gpu_cache[key] = {
                name: torch.as_tensor(arr, device=self.device, dtype=self._model_views[name].dtype)
                for name, arr in cpu_vals.items()
            }
        return self._geometry_gpu_cache[key]

    def _sync_worlds_to_host(self, ids):
        if not ids:
            return
        idx = list(ids)
        qp = self._qpos[idx].detach().cpu().numpy()
        qv = self._qvel[idx].detach().cpu().numpy()
        frames = self._env_index[idx].detach().cpu().tolist()
        for k, i in enumerate(idx):
            e = self.envs[i]
            e.d.qpos[:] = qp[k]
            e.d.qvel[:] = qv[k]
            e.index = int(frames[k])

    def reset(self, ids, source=None):
        ids = list(map(int, ids))
        if not ids:
            return
        if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= self.num_envs:
            raise ValueError("Invalid or duplicate world IDs")

        if source is None:
            rec_ids = [i for i in ids if self.envs[i].pending_recovery]
            if len(rec_ids) >= 2 and self._recovery_workers > 1:
                self._sync_worlds_to_host(rec_ids)
                pool = self._ensure_rec_pool()
                futs = []
                for i in rec_ids:
                    e = self.envs[i]
                    root = e.d.qpos[:7].copy()
                    box = e.d.qpos[e.bp:e.bp + 7].copy()
                    futs.append(pool.submit(_recovery_worker, str(e.root), root, box, e.half.copy(), e.goal.copy()))
                for i, fut in zip(rec_ids, futs):
                    self.envs[i]._precomputed_recovery = fut.result()

        new = []
        variants = []
        init_boxes, goals, on_floors, clip_lens, indices = [], [], [], [], []
        states_host, contexts_host = [], []
        for i in ids:
            e = self.envs[i]
            e.in_reset = True
            e.reset(source)
            e.in_reset = False
            v_cpu = self._geometry(e.half)
            v_gpu = self._geometry_gpu(e.half)
            e.context[4:7] = v_cpu["body_inertia"][self._box_body]
            if e.source != "recovery":
                new.append(i)
                variants.append(v_gpu)
            goal_bottom = float(e.goal[2] - np.abs(_matrix(e.clip[-1, 107:111])[2]) @ e.half)
            tabletop = float(e.d.mocap_pos[self._table2_id, 2] + self._table2_half_z)
            on_floor = abs(goal_bottom) < abs(goal_bottom - tabletop)
            st = e.state
            init_boxes.append(st[71:74])
            goals.append(e.goal)
            on_floors.append(on_floor)
            clip_lens.append(len(e.clip))
            indices.append(e.index)
            states_host.append(st)
            contexts_host.append(e.context)

        with wp.ScopedDevice(str(self.device)), self._stream():
            if new:
                self._reset_mask_torch.zero_()
                self._reset_mask_torch[new] = True
                if self._cuda_stream is not None:
                    self._cuda_stream.wait_stream(torch.cuda.current_stream(self.device))
                mjw.reset_data(self.m, self.d, self._reset_mask_wp)
                if self._cuda_stream is not None:
                    torch.cuda.current_stream(self.device).wait_stream(self._cuda_stream)
                idx = torch.tensor(new, dtype=torch.int64, device=self.device)
                for key, view in self._model_views.items():
                    view.index_copy_(0, idx, torch.stack([v[key] for v in variants], dim=0))
                for view, key in ((self._qpos, "qpos"), (self._qvel, "qvel"), (self._mocap, "mocap_pos")):
                    arr = np.stack([getattr(self.envs[i].d, key) for i in new])
                    view.index_copy_(0, idx, torch.as_tensor(arr, device=self.device, dtype=view.dtype))

            idx_all = torch.tensor(ids, dtype=torch.int64, device=self.device)
            self._initial_box.index_copy_(0, idx_all, torch.as_tensor(np.stack(init_boxes), dtype=torch.float32, device=self.device))
            self._goal_gpu.index_copy_(0, idx_all, torch.as_tensor(np.stack(goals), dtype=torch.float32, device=self.device))
            self._goal_on_floor.index_copy_(0, idx_all, torch.as_tensor(on_floors, dtype=torch.bool, device=self.device))
            self._clip_len.index_copy_(0, idx_all, torch.as_tensor(clip_lens, dtype=torch.int32, device=self.device))
            self._env_index.index_copy_(0, idx_all, torch.as_tensor(indices, dtype=torch.int32, device=self.device))
            self._lifted[idx_all] = False
            self._carried_steps[idx_all] = 0
            self._lost_steps[idx_all] = 0
            self._stable_steps[idx_all] = 0
            self._state_gpu.index_copy_(0, idx_all, torch.as_tensor(np.stack(states_host), dtype=torch.float32, device=self.device))
            self._context_gpu.index_copy_(0, idx_all, torch.as_tensor(np.stack(contexts_host), dtype=torch.float32, device=self.device))

            if self.teacher_history:
                if self._cuda_stream is not None:
                    self._cuda_stream.wait_stream(torch.cuda.current_stream(self.device))
                mjw.kinematics(self.m, self.d)
                if self._cuda_stream is not None:
                    torch.cuda.current_stream(self.device).wait_stream(self._cuda_stream)
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

    def _launch_step_kernels(self):
        e = self.envs[0]
        self._overflow.zero_()
        for _ in range(4):
            wp.launch(_pd_ctrl, dim=(self.num_envs, 29),
                      inputs=[self._target_wp, self.d.qpos, self.d.qvel,
                              self._kp_wp, self._kd_wp, self._limit_wp, self.d.ctrl])
            mjw.step(self.m, self.d)
            self._check_buffers()
        mjw.forward(self.m, self.d)
        self._check_buffers()
        self._forces.zero_()
        self._support.zero_()
        mjw.contact_force(self.m, self.d, self._ids, True, self._wrenches)
        wp.launch(_gather_contact, self.d.naconmax,
                  inputs=[self.d.nacon, self.d.contact.worldid, self.d.contact.geom,
                          self.d.contact.frame, self._wrenches, self._limbs,
                          self._box, e.floor_geom, e.tables_geom[1], e.tables_geom[0],
                          self._forces, self._support])

    def _step_physics(self, actions):
        if any(e._done for e in self.envs):
            raise RuntimeError("Reset every terminal world before stepping the vector environment")
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != (self.num_envs, 29) or not torch.isfinite(actions).all():
            raise ValueError("Expected a finite [num_envs,29] action matrix")
        default, scale = self._pd[0], self._pd[1]
        self._target.copy_(default + scale * actions)
        torch_stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        with wp.ScopedDevice(str(self.device)), self._stream():
            if self._cuda_stream is not None:
                self._cuda_stream.wait_stream(torch_stream)
            if self._step_graph is not None:
                wp.capture_launch(self._step_graph, stream=self._wp_stream)
            elif self._use_cuda_graph and self._graph_warmed and not self._graph_failed:
                try:
                    with wp.ScopedCapture(device=str(self.device), stream=self._wp_stream,
                                          force_module_load=True) as capture:
                        self._launch_step_kernels()
                    self._step_graph = capture.graph
                    wp.capture_launch(self._step_graph, stream=self._wp_stream)
                except Exception as exc:
                    self._graph_failed = True
                    self._step_graph = None
                    warnings.warn(f"Warp CUDA Graph capture failed, falling back to eager kernels: {exc}",
                                  RuntimeWarning)
                    self._launch_step_kernels()
            else:
                self._launch_step_kernels()
                self._graph_warmed = True
            if self._cuda_stream is not None:
                torch_stream.wait_stream(self._cuda_stream)
        return actions

    def _eval_gpu_step(self):
        contacts = (self._forces_torch.norm(dim=2) > self._contact_thresh).to(torch.float32)
        support = self._support_torch != 0
        states = torch.cat((
            self._qpos[:, :36], self._qvel[:, :35],
            self._qpos[:, self._bp:self._bp + 7], self._qvel[:, self._bv:self._bv + 6],
        ), dim=-1)
        (next_index, next_lifted, next_carried, next_lost, next_stable,
         done, success, drop, timeout, fallen, err) = finish_step_gpu(
            states, contacts, support, self._initial_box, self._goal_gpu, self._goal_on_floor,
            self._env_index, self._clip_len, self._lifted, self._carried_steps,
            self._lost_steps, self._stable_steps,
        )
        self._env_index.copy_(next_index)
        self._lifted.copy_(next_lifted)
        self._carried_steps.copy_(next_carried)
        self._lost_steps.copy_(next_lost)
        self._stable_steps.copy_(next_stable)
        return states, contacts, support, done, success, drop, timeout, fallen, err

    def step_gpu(self, actions):
        """Advance all worlds on GPU and transfer only compact termination flags to host."""
        with torch.no_grad():
            self._step_physics(actions)
            states, contacts, _, done, success, drop, timeout, fallen, _ = self._eval_gpu_step()
            overflow_any = (self._overflow_torch != 0).any()
            non_finite = (~torch.isfinite(self._qpos).all()) | (~torch.isfinite(self._qvel).all())
            n = self.num_envs
            self._packed_flags[:n, 0] = done
            self._packed_flags[:n, 1] = success
            self._packed_flags[:n, 2] = drop
            self._packed_flags[:n, 3] = timeout
            self._packed_flags[:n, 4] = fallen
            self._packed_flags[n, 0] = overflow_any
            self._packed_flags[n, 1] = non_finite
            self._packed_flags[n, 2:] = 0
            packed_np = self._packed_flags.cpu().numpy()

        if packed_np[n, 0]:
            raise RuntimeError(
                f"MuJoCo Warp capacity exceeded (contacts, constraints, broadphase pairs): "
                f"{self._overflow_torch.cpu().numpy()}"
            )
        if packed_np[n, 1]:
            raise FloatingPointError("Non-finite MuJoCo Warp physics state")

        flags_np = packed_np[:n]
        done_np = flags_np[:, 0].astype(bool)
        done_ids = np.flatnonzero(done_np)
        event_counts = Counter()
        drops = int(flags_np[:, 2].sum())
        rec_ids = []
        for i in done_ids:
            e = self.envs[i]
            e._done = True
            is_success = bool(flags_np[i, 1])
            is_drop = bool(flags_np[i, 2])
            is_timeout = bool(flags_np[i, 3])
            is_fallen = bool(flags_np[i, 4])
            e.pending_recovery = is_drop and not is_fallen and e.source != "recovery"
            if e.pending_recovery:
                rec_ids.append(i)
            src = e.source
            event_counts[f"{src}/done"] += 1
            event_counts[f"{src}/success"] += int(is_success)
            event_counts[f"{src}/drop"] += int(is_drop)
            event_counts[f"{src}/timeout"] += int(is_timeout)
            event_counts[f"{src}/fallen"] += int(is_fallen)
            event_counts[f"{src}/recovered"] += int(is_success and src == "recovery")
        if rec_ids:
            self._sync_worlds_to_host(rec_ids)
        if self.teacher_history:
            self._sync_worlds_to_host(range(self.num_envs))
            self._read_teacher_pose(range(self.num_envs))
            for e in self.envs:
                e._update_teacher_history()
        return states, contacts, done_np, event_counts, drops

    def step(self, actions):
        with torch.no_grad():
            actions_t = self._step_physics(actions)
            states_t, contacts_t, _, done, success, drop, timeout, fallen, err = self._eval_gpu_step()
            overflow = self._overflow_torch.cpu().numpy()
            if np.any(overflow):
                raise RuntimeError(
                    f"MuJoCo Warp capacity exceeded (contacts, constraints, broadphase pairs): {overflow}"
                )
            qp = self._qpos.cpu().numpy()
            qv = self._qvel.cpu().numpy()
            if not np.isfinite(qp).all() or not np.isfinite(qv).all():
                raise FloatingPointError("Non-finite MuJoCo Warp physics state")
            contacts = contacts_t.cpu().numpy()
            states = states_t.cpu().numpy()
            supported = (self._support_torch != 0).cpu().numpy()
            act = actions_t.cpu().numpy()
            if self.teacher_history:
                self._read_teacher_pose(range(self.num_envs))

        done_np = done.cpu().numpy()
        success_np = success.cpu().numpy()
        drop_np = drop.cpu().numpy()
        timeout_np = timeout.cpu().numpy()
        fallen_np = fallen.cpu().numpy()
        err_np = err.cpu().numpy()
        idx_np = self._env_index.cpu().numpy()
        lifted_np = self._lifted.cpu().numpy()
        carried_np = self._carried_steps.cpu().numpy()
        lost_np = self._lost_steps.cpu().numpy()
        stable_np = self._stable_steps.cpu().numpy()

        infos = []
        for i, e in enumerate(self.envs):
            e.d.qpos, e.d.qvel = qp[i], qv[i]
            e.last_action = act[i].copy()
            e.index = int(idx_np[i])
            if self.teacher_history:
                e._update_teacher_history()
            e._lifted = bool(lifted_np[i])
            e._carried_steps = int(carried_np[i])
            e._lost_steps = int(lost_np[i])
            e._stable_steps = int(stable_np[i])
            e._done = bool(done_np[i])
            e.pending_recovery = bool(drop_np[i] and not fallen_np[i] and e.source != "recovery")
            ground, dest, origin = bool(supported[i, 0]), bool(supported[i, 1]), bool(supported[i, 2])
            infos.append(dict(
                success=bool(success_np[i]), drop=bool(drop_np[i]),
                timeout=bool(timeout_np[i]), fallen=bool(fallen_np[i]),
                recovered=bool(success_np[i] and e.source == "recovery"),
                goal_error=float(err_np[i]),
                box_error=float(np.linalg.norm(states[i, 71:74] - e.clip[e.index, 104:107])),
                source=e.source, recovery_rejected=e.recovery_rejected,
                grounded=ground, supported=bool(ground or dest or origin),
                lifted=bool(lifted_np[i]),
            ))
        return states, contacts, done_np, infos


def _shutdown_pool_ref(world):
    pool = getattr(world, "_rec_pool", None)
    if pool is not None:
        world._rec_pool = None
        pool.shutdown(wait=False)
