"""50 Hz CPU MuJoCo collection with a free box and the official CF-Gen planner."""
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import yaml

from .data import OFFSETS, LAB2MJ, MJ2LAB, LINKS, generate_clip, load_demo, raw_window, split_paths


def _matrix(quat):
    # MuJoCo and the motion archives use scalar-first quaternions.
    quat = np.asarray(quat)
    w,x,y,z = np.moveaxis(quat/np.linalg.norm(quat, axis=-1, keepdims=True), -1, 0)
    values = (1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
              2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
              2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))
    return np.stack(values,axis=-1).reshape(*quat.shape[:-1],3,3)


@lru_cache(maxsize=6)
def _scene(root, half):
    """Compile actual collision geometry, broadphase, mass and inertia per size."""
    scene = root / "g1_description" / "omnicontact_carry_box.xml"
    xml = ET.parse(scene)
    size = " ".join(str(v) for v in half)
    for name in ("box_geom", "ghost_box_geom"):
        xml.find(f".//geom[@name='{name}']").set("size", size)
    i = xml.find(".//body[@name='box']/inertial")
    x, y, z = np.asarray(half) * 2
    inertia = float(i.get("mass")) / 12 * np.asarray([y*y+z*z, x*x+z*z, x*x+y*y])
    i.set("diaginertia", " ".join(str(v) for v in inertia))
    with tempfile.TemporaryDirectory(prefix="carry-scene-") as tmp:
        work = Path(tmp)
        for asset in scene.parent.iterdir():
            if asset != scene:
                (work / asset.name).symlink_to(asset)
        xml.write(work / scene.name, encoding="utf-8")
        model = mj.MjModel.from_xml_path(str(work / scene.name))
    model.opt.timestep = .005
    if model.nu != 29 or not np.array_equal(model.jnt_qposadr[model.actuator_trnid[:, 0]], np.arange(7,36)):
        raise ValueError("Scene actuator order differs from the released G1 29-DoF carry scene")
    return model


@lru_cache(maxsize=2)
def _generator(root):
    sys.path.insert(0, str(root))
    from policy.omnicontact.CFgen_meta2_carrybox import CfGenCarryBox
    return CfGenCarryBox()


@lru_cache(maxsize=2)
def _teacher(root):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    path = root / "policy/omnicontact/model/omnicontact_mlp.onnx"
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class CarryEnv:
    """One physical simulator. step() never resets or replaces the target clip."""

    def __init__(self, sim_root, motion_dir, seed=42, split="train", cfgen_ratio=.1):
        if not 0 <= cfgen_ratio <= 1:
            raise ValueError("cfgen_ratio must be a probability")
        self.root = Path(sim_root).resolve()
        self.paths = split_paths(motion_dir, split)
        self.split, self.cfgen_ratio = split, cfgen_ratio
        self.rng = np.random.default_rng(seed)
        self.cfg = yaml.safe_load((self.root / "policy/omnicontact/config/OmniContact.yaml").read_text())
        if not np.array_equal(self.cfg["lab2mj"], LAB2MJ):
            raise ValueError("Upstream action permutation no longer matches the carry data layout")
        self.default, self.scale, self.kp, self.kd, self.limit = [
            np.array(self.cfg[k])[LAB2MJ] for k in ("default_angles_lab", "action_scale_lab",
                                                     "kp_lab", "kd_lab", "tau_limit_lab")]
        self.pending_recovery = False
        self.episode = 0
        self.recovery_rejected = 0
        self._done = True

    def _bind_scene(self, half):
        self.m = _scene(self.root, tuple(half))
        self.d = mj.MjData(self.m)
        self.half = np.asarray(half, dtype=float)
        self.bp = int(self.m.joint("box").qposadr[0])
        self.bv = int(self.m.joint("box").dofadr[0])
        self.box_geom = self.m.geom("box_geom").id
        self.floor_geom = self.m.geom("floor").id
        self.tables_geom = [self.m.geom(f"plane_{j}").id for j in (1, 2)]
        self.table_ids = [int(self.m.body(f"plane_{j}_holder").mocapid[0]) for j in (1,2)]
        self.body_ids = [self.m.body(n).id for n in (*LINKS, "mid360_link")]
        contact_bodies = ("left_ankle_roll_link", "right_ankle_roll_link",
                          "left_wrist_yaw_link", "right_wrist_yaw_link")
        self._limb = {self.m.body(n).id: j for j,n in enumerate(contact_bodies)}

    def _try_recovery(self):
        """Verify the solved IK actually places both fingertips on the requested faces."""
        root = self.d.qpos[:7].copy()
        box = self.d.qpos[self.bp:self.bp+7].copy()
        gen = _generator(self.root)
        try:
            frames, phase = generate_clip(gen, root, box, self.half, self.goal)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return False
        grasp = np.flatnonzero((frames[:, 37:39] > .5).all(axis=1))
        if not len(grasp):
            return False
        first = frames[grasp[0]]
        face, heading, _ = gen._get_target_face_info(box[:3], box[3:7], root[:3], self.half)
        targets = gen._grasp_targets(box[:3], face, heading)
        for i, name in enumerate(("lw_contact", "rw_contact")):
            palm = first[i*7:i*7+7]
            # IK uses the pad center; ref frames store the palm-link origin.
            offset = np.array([.1285, .012 if i == 0 else -.012, 0.])
            pad = palm[:3] + _matrix(palm[3:]) @ offset
            if np.linalg.norm(pad-targets[name]) > .05:
                return False
        self.clip, self.phase = frames, phase
        return True

    def reset(self, source=None):
        attempt = self.pending_recovery and source is None
        self.pending_recovery = False
        if attempt and self._try_recovery():
            self._teacher_ankles = None
            self.source = "recovery"
            self.name = f"{self.name}/recovery"
            self.index = 0
        else:
            self.recovery_rejected += int(attempt)
            if source is None:
                source = "cfgen" if self.rng.random() < self.cfgen_ratio else "npz"
            if source not in {"npz", "cfgen"}:
                raise ValueError(f"Unknown reference source {source}")
            path = self.paths[self.rng.integers(len(self.paths))]
            scene = load_demo(path)
            self._bind_scene(scene.half)
            self.source, self.name = source, path.stem
            self.episode += 1
            self.d.mocap_pos[self.table_ids] = scene.tables
            self.goal = scene.frames[-1, 104:107].copy()
            self.clip, self.phase = scene.frames, None
            self._teacher_ankles = scene.teacher_ankles if source == "npz" else None
            self.index = 0
            twists = scene.twists.copy()
            if source == "cfgen":
                # Use scene geometry and the requested goal. The demonstration's
                # future poses and contacts do not condition the generator.
                if self.split == "train":
                    noise = self.rng.normal(0, .005, 2)
                    # Retain full footprint support on the target table.
                    reach = np.maximum(self.m.geom_size[self.tables_geom[1],:2] - np.linalg.norm(self.half[:2]), 0)
                    center = scene.tables[1,:2]
                    self.goal[:2] = np.clip(self.goal[:2] + noise, center-reach, center+reach)
                self.clip, self.phase = generate_clip(_generator(self.root), scene.frames[0,97:104],
                                                       scene.frames[0,104:111], self.half, self.goal)
                twists[:] = 0
            f = self.clip[self.index]
            self.d.qpos[:7] = f[97:104]
            self.d.qpos[7:36] = f[39:68]
            self.d.qvel[:3] = twists[0]
            self.d.qvel[3:6] = _matrix(f[100:104]).T @ twists[1]
            self.d.qvel[6:35] = f[68:97]
            self.d.qpos[self.bp:self.bp+7] = f[104:111]
            self.d.qvel[self.bv:self.bv+3] = twists[2]
            self.d.qvel[self.bv+3:self.bv+6] = _matrix(f[107:111]).T @ twists[3]
            mj.mj_forward(self.m, self.d)
        self.context = np.concatenate((self.half, self.m.body("box").mass,
                                       self.m.body("box").inertia, self.m.geom_friction[self.box_geom],
                                       self.d.mocap_pos[self.table_ids].ravel())).astype(np.float32)
        self._done = self._lifted = False
        self._lost_steps = self._stable_steps = self._carried_steps = 0
        self._initial_box = self.state[71:74].copy()
        self._teacher_hist = np.zeros((5,141), np.float32)
        self.last_action = np.zeros(29, np.float32)
        self._update_teacher_history()

    @property
    def state(self):
        return np.concatenate((self.d.qpos[:36], self.d.qvel[:35], self.d.qpos[self.bp:self.bp+7],
                               self.d.qvel[self.bv:self.bv+6])).astype(np.float32)

    def _contact(self):
        force = np.zeros((6,3))
        grounded = on_target = other_support = False
        wrench = np.empty(6)
        for ci, con in enumerate(self.d.contact):
            g1, g2 = int(con.geom1), int(con.geom2)
            mj.mj_contactForce(self.m, self.d, ci, wrench)
            normal = max(wrench[0], 0.)
            if normal <= 0.:
                continue
            world_force = np.asarray(con.frame).reshape(3,3).T @ wrench[:3]
            for sign, geom in ((-1,g1), (1,g2)):
                limb = self._limb.get(self.m.geom_bodyid[geom])
                if limb in (0,1):
                    force[4+limb] += sign * world_force
            if self.box_geom in (g1, g2):
                g = g2 if g1 == self.box_geom else g1
                grounded |= g == self.floor_geom
                on_target |= g == self.tables_geom[1]
                other_support |= g == self.tables_geom[0]
                idx = self._limb.get(self.m.geom_bodyid[g])
                if idx is not None:
                    force[idx] += world_force if g == g2 else -world_force
        labels = np.linalg.norm(force, axis=1) > [0,0,5,5,1,1]
        return labels.astype(np.float32), grounded, on_target, other_support

    def _update_teacher_history(self):
        s = self.state
        ids = self.body_ids
        pos = self.d.xpos[ids]
        torso = self.d.xmat[ids[2]].reshape(3,3)
        yaw = np.arctan2(torso[1,0],torso[0,0])
        c, sn = np.cos(yaw), np.sin(yaw)
        self.heading = np.array([[c,-sn,0],[sn,c,0],[0,0,1]])
        ee = (pos[[0,1,3,4,5]] - pos[2]) @ torso
        boxpos = (s[71:74] - pos[2]) @ self.heading
        boxrot = _matrix(s[74:78])
        corners = np.asarray(list(product([1,-1],repeat=3))) * self.half
        corners = ((corners @ boxrot.T + s[71:74])-pos[2]) @ self.heading
        boxpos *= min(1.,4./max(np.linalg.norm(boxpos),1e-7))
        corners *= np.minimum(1.,4./np.maximum(np.linalg.norm(corners,axis=-1,keepdims=True),1e-7))
        frame = np.concatenate((ee.ravel(),s[39:42],-_matrix(s[3:7])[2],(s[7:36]-self.default)[MJ2LAB],
                                s[42:71][MJ2LAB],self.last_action[MJ2LAB],boxpos,
                                (self.heading.T @ boxrot)[:,:2].ravel(),corners.ravel()))
        self._teacher_hist[:-1] = self._teacher_hist[1:].copy()
        self._teacher_hist[-1] = frame

    def teacher_action(self):
        """Original ONNX MLP, driven by the executed observation history."""
        sess = _teacher(self.root)
        raw = raw_window(self.clip, self.index)
        targets = raw[:,:35].reshape(11,5,7).copy()
        if self._teacher_ankles is not None:
            times = np.minimum(self.index+OFFSETS, len(self.clip)-1)
            targets[:, 3:5] = self._teacher_ankles[times]
        p = (targets[:,:,:3] - self.d.xpos[self.body_ids[2]]) @ self.heading
        R = _matrix(targets[:,:,3:].reshape(-1,4)).reshape(11,5,3,3)
        r6 = (self.heading.T @ R)[:,:,:,:2].reshape(11,5,6)
        track = np.concatenate((p,r6),axis=-1).ravel()
        bounds = (0,15,18,21,50,79,108,111,117,141)
        hist = [self._teacher_hist[:,a:b].ravel() for a,b in zip(bounds,bounds[1:])]
        x = np.concatenate((track,raw[:,35:39].ravel(),*hist)).astype(np.float32)[None]
        inputs = sess.get_inputs()
        feed = {inputs[0].name:x}
        if len(inputs) > 1:
            feed[inputs[1].name] = np.zeros((1,1),np.float32)
        return sess.run(None,feed)[0].reshape(29)[LAB2MJ]

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() required before the next action")
        action = np.asarray(action,dtype=np.float32).reshape(29)
        if not np.isfinite(action).all():
            raise ValueError("Non-finite executed action")
        target = self.default + self.scale*action
        for _ in range(4):
            tau = self.kp*(target-self.d.qpos[7:36]) - self.kd*self.d.qvel[6:35]
            self.d.ctrl[:] = np.clip(tau,-self.limit,self.limit)
            mj.mj_step(self.m,self.d)
        mj.mj_forward(self.m,self.d)
        self.last_action = action.copy()
        self.index = min(self.index+1,len(self.clip)-1)
        self._update_teacher_history()
        state = self.state
        labels, ground, dest, origin = self._contact()
        hand = labels[2:4].any()
        # A box on its original support before grasp is not a dropped box.
        displacement = state[71:74] - self._initial_box
        cleared_initial_pose = displacement[2] > .025 or np.linalg.norm(displacement[:2]) > .1
        held_free = hand and not (ground or dest or origin) and cleared_initial_pose
        self._carried_steps = self._carried_steps + 1 if held_free else 0
        self._lifted |= self._carried_steps >= 2
        self._lost_steps = self._lost_steps+1 if self._lifted and not hand else 0
        err = float(np.linalg.norm(state[71:74]-self.goal))
        speed = np.linalg.norm(state[78:81])
        stable = speed < .1 and np.linalg.norm(state[81:84]) < .2
        goal_bottom = self.goal[2] - np.abs(_matrix(self.clip[-1,107:111])[2]) @ self.half
        tabletop = self.d.mocap_pos[self.table_ids[1],2] + self.m.geom_size[self.tables_geom[1],2]
        on_goal_support = ground if abs(goal_bottom) < abs(goal_bottom-tabletop) else dest
        supported_goal = self._lifted and on_goal_support and err < .2 and stable and not hand
        self._stable_steps = self._stable_steps+1 if supported_goal else 0
        success = self._stable_steps >= 5
        off_table = ground and displacement[2] < -.08 and not hand
        lost_grasp = self._lost_steps >= 3 and (ground or origin or dest)
        drop = bool((off_table or lost_grasp) and (err >= .2 or not on_goal_support) and stable)
        fallen = bool(state[2] < .3 or _matrix(state[3:7])[2,2] < .35)
        timeout = self.index == len(self.clip)-1 and not (fallen or drop or success)
        self._done = bool(drop or fallen or timeout or success)
        self.pending_recovery = drop and not fallen and self.source != "recovery"
        info = dict(success=bool(success), drop=drop, timeout=timeout, fallen=fallen,
                    recovered=bool(success and self.source=="recovery"), goal_error=err,
                    box_error=float(np.linalg.norm(state[71:74]-self.clip[self.index,104:107])),
                    source=self.source, recovery_rejected=self.recovery_rejected,
                    grounded=bool(ground), supported=bool(ground or dest or origin), lifted=self._lifted)
        return state, labels, self._done, info
