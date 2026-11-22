"""Differentiable G1/box model. All joints use MJCF order; quaternions are wxyz.

State (84): root xyz, quat, q(29), world v, local omega, dq(29),
box xyz, quat, world v, local omega. Control period is 20 ms.
"""
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import yaml


DT = 0.02
STATE_DIM = 84
TARGET_DIM = 111
OBS_DIM = 141
GOAL_DIM = 113
ACTION_DIM = 29


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), dim=-1)


def quat_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).unflatten(-1, (3, 3))


def integrate_rotation(q, local_w):
    half_angle = 0.5 * DT * local_w.norm(dim=-1, keepdim=True)
    delta = torch.cat((torch.cos(half_angle),
                       (0.5 * DT) * torch.sinc(half_angle / torch.pi) * local_w), -1)
    return F.normalize(quat_mul(q, delta), dim=-1)


def _rotation_error(actual, target):
    rel = actual.transpose(-1, -2) @ target
    skew = torch.stack((rel[..., 2, 1] - rel[..., 1, 2],
                        rel[..., 0, 2] - rel[..., 2, 0],
                        rel[..., 1, 0] - rel[..., 0, 1]), -1)
    return torch.atan2(skew.norm(dim=-1), rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)


def _rot6d(mat):
    return mat[..., :, :2].flatten(-2)


def _yaw_rotation(mat):
    heading = F.normalize(mat[..., :2, 0], dim=-1)
    c, s = heading.unbind(-1)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).unflatten(-1, (3, 3))


def _apply(mat, vec):
    return (mat @ vec.unsqueeze(-1)).squeeze(-1)


def _homogeneous(rotation, position):
    top = torch.cat((rotation, position.unsqueeze(-1)), -1)
    bottom = torch.cat((torch.zeros_like(position), torch.ones_like(position[..., :1])), -1)
    return torch.cat((top, bottom.unsqueeze(-2)), -2)


class RunningNorm(nn.Module):
    """Parallel Welford moments across real samples only; constant channels stay bounded."""
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0., dtype=torch.float64))

    @torch.no_grad()
    def update(self, x):
        x = x.detach().reshape(-1, self.mean.numel()).float()
        var, mean = torch.var_mean(x, dim=0, unbiased=False)
        total = self.count + x.shape[0]
        w = (x.shape[0] / total).to(self.mean.dtype)
        delta = mean - self.mean
        self.var.copy_(self.var * (1-w) + var*w + delta.square()*w*(1-w))
        self.mean.add_(delta*w)
        self.count.copy_(total)

    def forward(self, x):
        return ((x-self.mean) * self.var.clamp_min(0.0025).rsqrt()).clamp(-10, 10)


def _mlp(input_dim, hidden, output_dim):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class Actor(nn.Module):
    def __init__(self, first_width=2048, hidden=(1024, 512, 256), decoder_width=2048):
        super().__init__()
        self.goals = RunningNorm(GOAL_DIM)
        self.observations = RunningNorm(OBS_DIM)
        self.encoder = _mlp(11*GOAL_DIM, (first_width, *hidden), 32)
        self.decoder = _mlp(32+10*OBS_DIM, (decoder_width, *hidden), ACTION_DIM)
        nn.init.normal_(self.decoder[-1].weight, std=0.005)
        nn.init.zeros_(self.decoder[-1].bias)

    def update_stats(self, goal, history):
        self.goals.update(goal)
        self.observations.update(history)

    def forward(self, goal, history, *, checkpoint_encoder=False):
        goal = self.goals(goal).flatten(-2)
        if checkpoint_encoder and torch.is_grad_enabled():
            # The encoder MLP is deterministic; only its activations are recomputed.
            encoded = checkpoint(self.encoder, goal, use_reentrant=False, preserve_rng_state=False)
        else:
            encoded = self.encoder(goal)
        z = encoded.tanh()
        return self.decoder(torch.cat((z, self.observations(history).flatten(-2)), -1))


class Critic(nn.Module):
    def __init__(self, context_dim, hidden=(512, 256)):
        super().__init__()
        self.value = _mlp(STATE_DIM + 11*GOAL_DIM + context_dim, hidden, 1)

    def forward(self, state, goal, ctx):
        return self.value(torch.cat((state, goal.flatten(-2), ctx), -1)).squeeze(-1)


class WorldModel(nn.Module):
    """Shared dynamics trunk and independent 35-DoF robot / 6-DoF box velocity heads."""
    def __init__(self, context_dim, width=2048):
        super().__init__()
        input_dim = STATE_DIM + ACTION_DIM + context_dim
        self.inputs = RunningNorm(input_dim)
        self.trunk = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU(),
                                   nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
                                   nn.Linear(width, width), nn.LayerNorm(width), nn.GELU())
        self.robot = nn.Linear(width, 35)
        self.box = nn.Linear(width, 6)
        self.contact = nn.Linear(width, 6)  # four limb-box flags and two foot support flags
        for head in (self.robot, self.box):
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)

    def update_stats(self, state, action, ctx):
        self.inputs.update(torch.cat((state, action, ctx), -1))

    def forward(self, state, action, ctx):
        initial = torch.cat((state, action, ctx), -1)
        h = self.trunk(self.inputs(initial))
        dr, db = self.robot(h).float(), self.box(h).float()
        # Four physics substeps per tick. With a constant acceleration and
        # semi-implicit integration the mean velocity is v0 + 5/8 * delta.
        c = 0.625
        def pos(s, vs, dv):
            return state[..., s] + DT * (state[..., vs] + c*dv)
        def rot(s, vs, dw):
            return integrate_rotation(state[..., s], state[..., vs] + c*dw)
        next_state = torch.cat((
            pos(slice(0, 3), slice(36, 39), dr[..., :3]),
            rot(slice(3, 7), slice(39, 42), dr[..., 3:6]),
            pos(slice(7, 36), slice(42, 71), dr[..., 6:]),
            state[..., 36:71] + dr,
            pos(slice(71, 74), slice(78, 81), db[..., :3]),
            rot(slice(74, 78), slice(81, 84), db[..., 3:]),
            state[..., 78:84] + db), -1)
        return next_state, self.contact(h).float()


def world_loss(pred_state, truth_state, logits, labels):
    # Scale squared errors by a nominal error in metres, rad, m/s or rad/s.
    terms = []
    for lo, hi, scale in ((0,3,.08), (7,36,.15), (36,39,.75),
                          (39,42,1.5), (42,71,3.), (71,74,.04),
                          (78,81,.5), (81,84,1.)):
        terms.append(((pred_state[..., lo:hi]-truth_state[..., lo:hi])/scale).square().mean())
    for lo, scale in ((3,.20), (74,.15)):
        angle = _rotation_error(quat_matrix(pred_state[..., lo:lo+4]), quat_matrix(truth_state[..., lo:lo+4]))
        terms.append((angle/scale).square().mean())
    return torch.stack(terms).mean() + F.binary_cross_entropy_with_logits(logits, labels.float())


class CarryGeometry(nn.Module):
    """Batched G1 forward kinematics driven by official MJCF link transforms.

    Each parent-jump composes 1, 2, 4, ... ancestors in parallel. Index zero
    is an identity sentinel so the pelvis world transform is applied exactly once.
    """
    tracked = ("left_palm_link", "right_palm_link", "torso_link", "left_ankle_pitch_link", "right_ankle_pitch_link")
    bodies = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
              "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
              "left_shoulder_roll_link", "left_elbow_link", "left_palm_link",
              "right_shoulder_roll_link", "right_elbow_link", "right_palm_link")

    def __init__(self, sim2sim_root):
        super().__init__()
        root = Path(sim2sim_root)
        with (root/"policy/omnicontact/config/OmniContact.yaml").open() as f:
            config = yaml.safe_load(f)
        order = config["lab2mj"]
        for name, field in (("default", "default_angles"), ("kp", "kp"), ("kd", "kd"),
                            ("scale", "action_scale"), ("limit", "tau_limit"),
                            ("low", "joint_pos_lowerlimit"), ("high", "joint_pos_upperlimit")):
            self.register_buffer(name, torch.tensor(config[field+"_lab"], dtype=torch.float32)[order])
        body = ET.parse(root/"g1_description/g1_29dof.xml").find('.//worldbody/body[@name="pelvis"]')
        if body is None:
            raise ValueError("G1 pelvis missing from XML")
        names = ["identity"]
        parents, angles, axes, offsets, quats = [0], [29], [[0.,0.,0.]], [[0.,0.,0.]], [[1.,0.,0.,0.]]
        def numbers(node, key, default):
            return [float(s) for s in node.get(key, default).split()]
        joint_names = []
        def add(node, parent):
            idx = len(names)
            names.append(node.attrib["name"])
            parents.append(parent)
            joint = node.find("joint")
            is_hinge = joint is not None and joint.get("type", "hinge") == "hinge"
            angles.append(len(joint_names) if is_hinge else 29)
            axes.append(numbers(joint,"axis","0 0 1") if is_hinge else [0.,0.,0.])
            if is_hinge:
                if any(numbers(joint,"pos","0 0 0")):
                    raise ValueError("G1 hinge away from body origin")
                joint_names.append(joint.attrib["name"])
            offsets.append(numbers(node,"pos","0 0 0") if parent else [0.,0.,0.])
            quats.append(numbers(node, "quat", "1 0 0 0"))
            for child in node.findall("body"):
                add(child, idx)
        add(body, 0)
        if len(joint_names) != ACTION_DIM:
            raise ValueError("G1 requires 29 MJCF hinge joints")
        self.names = tuple(names)
        self.joint_names = tuple(joint_names)
        for name, values, dtype in (("axis",axes,torch.float32), ("offset",offsets,torch.float32),
                                    ("rest_quat",quats,torch.float32), ("hinge",angles,torch.long)):
            self.register_buffer(name, torch.tensor(values, dtype=dtype))
        self.jumps = 0
        parents = torch.tensor(parents)
        while parents.any():
            self.register_buffer(f"ancestors_{self.jumps}", parents)
            self.jumps += 1
            parents = parents[parents]
        # Python index tuples work on CPU/CUDA; use buffers for repeated gather.
        for name, links in (("flow_ids", self.tracked), ("body_ids", self.bodies),
                            ("ee_ids", (*self.tracked[:2], *self.tracked[3:], "mid360_link")),
                            ("feet_ids", ("left_ankle_roll_link", "right_ankle_roll_link"))):
            self.register_buffer(name, torch.tensor([names.index(link) for link in links]))
        self.register_buffer("arm_ids",torch.tensor([i for i,n in enumerate(joint_names)
                                                   if "elbow" in n or "wrist" in n]))
        self.torso = names.index("torso_link")
        self.register_buffer("corners", torch.tensor(list(product((1., -1.), repeat=3))))

    def fk(self, root_pos, root_quat, joints):
        padded = F.pad(joints, (0,1))
        half = .5*padded.index_select(-1, self.hinge)
        moving = torch.cat((half.cos().unsqueeze(-1), half.sin().unsqueeze(-1)*self.axis), -1)
        local = quat_mul(self.rest_quat.expand_as(moving), moving)
        pos = self.offset.expand_as(local[..., :3])
        body_transforms = _homogeneous(quat_matrix(local[..., 2:, :]), pos[..., 2:, :])
        root = _homogeneous(quat_matrix(root_quat), root_pos).unsqueeze(-3)
        identity = torch.eye(4, device=root.device, dtype=root.dtype).expand_as(root)
        nodes = torch.cat((identity, root, body_transforms), -3)
        for level in range(self.jumps):
            nodes = nodes.index_select(-3, getattr(self, f"ancestors_{level}")) @ nodes
        return nodes

    def forward_kinematics(self, state):
        return self.fk(state[..., :3], state[..., 3:7], state[..., 7:36])

    def fk_from_target(self, target):
        return self.fk(target[..., 97:100], target[..., 100:104], target[..., 39:68])

    def observation(self, state, prev_action, ctx, fk=None):
        fk = self.forward_kinematics(state) if fk is None else fk
        torso = fk[..., self.torso, :, :]
        torso_pos, torso_r = torso[..., :3, 3], torso[..., :3, :3]
        yaw_inv = _yaw_rotation(torso_r).transpose(-1,-2)
        ee = fk.index_select(-3, self.ee_ids)[..., :3, 3]
        ee_local = _apply(torso_r.transpose(-1,-2).unsqueeze(-3), ee - torso_pos.unsqueeze(-2))
        box_r = quat_matrix(state[..., 74:78])
        box_rel = state[..., 71:74] - torso_pos
        rel = _apply(yaw_inv, box_rel)
        def clip4(vec):
            return vec * (4.0 / vec.norm(dim=-1,keepdim=True).clamp_min(4.0))
        off = ctx[..., None, :3] * self.corners
        corners = _apply(box_r.unsqueeze(-3), off) + box_rel.unsqueeze(-2)
        local_corners = clip4(_apply(yaw_inv.unsqueeze(-3), corners))
        gravity = -quat_matrix(state[..., 3:7])[..., 2, :]
        return torch.cat((ee_local.flatten(-2), state[..., 39:42], gravity, state[..., 7:36]-self.default,
                          state[..., 42:71], prev_action, clip4(rel), _rot6d(yaw_inv@box_r),
                          local_corners.flatten(-2)), -1)

    def reference(self, state, target_window, fk=None):
        fk = self.forward_kinematics(state) if fk is None else fk
        torso = fk[..., self.torso, :, :]
        inv = _yaw_rotation(torso[..., :3,:3]).transpose(-1,-2)[..., None,None,:,:]
        poses = target_window[..., :35].unflatten(-1,(5,7))
        p = _apply(inv, poses[..., :3] - torso[..., None,None,:3,3])
        r = inv @ quat_matrix(poses[..., 3:])
        root_rel = quat_matrix(state[..., 3:7]).transpose(-1,-2).unsqueeze(-3) @ quat_matrix(target_window[..., 100:104])
        return torch.cat((torch.cat((p, _rot6d(r)), -1).flatten(-2), target_window[..., 35:97],
                          _rot6d(root_rel)), dim=-1)

    def reward_terms(self, prev_state, state, prev_action, action, target, contacts, prev_fk=None, fk=None, target_fk=None):
        """Signed, timestep-scaled OmniContact terms; contacts are probabilities/real labels."""
        fk = self.forward_kinematics(state) if fk is None else fk
        prev_fk = self.forward_kinematics(prev_state) if prev_fk is None else prev_fk
        target_fk = self.fk_from_target(target) if target_fk is None else target_fk
        actual = fk.index_select(-3, self.flow_ids)
        target_flow = target[..., :35].unflatten(-1,(5,7))
        pos, rot = actual[..., :3,3], actual[..., :3,:3]
        ref_pos, ref_rot = target_flow[..., :3], quat_matrix(target_flow[..., 3:])
        anchor, ref_anchor = pos[..., 2,:], ref_pos[..., 2,:]
        arot, ref_arot = rot[..., 2,:,:], ref_rot[..., 2,:,:]
        yaw = _yaw_rotation(arot @ ref_arot.transpose(-2,-1))
        ref_bodies = target_fk.index_select(-3, self.body_ids)
        bodies = fk.index_select(-3, self.body_ids)
        translated = torch.cat((anchor[..., :2], ref_anchor[..., 2:]), -1)
        ref_bp = translated.unsqueeze(-2) + _apply(yaw.unsqueeze(-3), ref_bodies[..., :3,3]-ref_anchor.unsqueeze(-2))
        ref_br = yaw.unsqueeze(-3) @ ref_bodies[..., :3,:3]
        box_p, box_r = state[..., 71:74], quat_matrix(state[..., 74:78])
        target_box_p, target_box_r = target[..., 104:107], quat_matrix(target[..., 107:111])
        hand_to_box = pos[..., :2,:] - box_p.unsqueeze(-2)
        ref_hand_to_box = ref_pos[..., :2,:] - target_box_p.unsqueeze(-2)
        weight = torch.exp(-2.*ref_hand_to_box.norm(dim=-1))
        interaction_yaw = _yaw_rotation(ref_arot.transpose(-2,-1) @ arot)
        grip_p_error = (_apply(interaction_yaw.unsqueeze(-3), hand_to_box)-ref_hand_to_box).square().sum(-1)
        grip_angle = _rotation_error(box_r.unsqueeze(-3), rot[..., :2,:,:])
        ref_grip_angle = _rotation_error(target_box_r.unsqueeze(-3), ref_rot[..., :2,:,:])
        # Approximate applied PD torque at the beginning of the 20 ms control interval.
        torque = ((self.default + self.scale*action - prev_state[..., 7:36])*self.kp
                  - prev_state[..., 42:71]*self.kd).clamp(-self.limit, self.limit)
        q = state[..., 7:36]
        # G1 soft limits in IsaacLab use 90% of the full interval.
        margin = .05*(self.high-self.low)
        limits = (self.low + margin - q).clamp_min(0) + (q - self.high + margin).clamp_min(0)
        feet = fk.index_select(-3,self.feet_ids)[..., :2,3]
        last_feet = prev_fk.index_select(-3,self.feet_ids)[..., :2,3]
        feet_speed = (feet-last_feet).norm(dim=-1)/DT
        terms = {
            "anchor_pos": .2*torch.exp(-(anchor-ref_anchor).square().sum(-1)/.3**2),
            "anchor_rot": .2*torch.exp(-_rotation_error(arot,ref_arot).square()/.4**2),
            "body_pos": .2*torch.exp(-(ref_bp-bodies[..., :3,3]).square().sum(-1).mean(-1)/.3**2),
            "body_rot": .2*torch.exp(-_rotation_error(ref_br,bodies[..., :3,:3]).square().mean(-1)/.4**2),
            "wrist_pos": .4*torch.exp(-(ref_pos[..., :2,:]-pos[..., :2,:]).square().sum(-1).mean(-1)/.3**2),
            "wrist_rot": .4*torch.exp(-_rotation_error(ref_rot[..., :2,:,:],rot[..., :2,:,:]).square().mean(-1)/.4**2),
            "box_pos": torch.exp(-(box_p-target_box_p).square().sum(-1)/.3**2),
            "box_rot": torch.exp(-_rotation_error(box_r,target_box_r).square()/.3**2),
            "interaction_pos": .4*torch.exp(-grip_p_error*weight.square()/.3**2).mean(-1),
            "interaction_rot": .4*torch.exp(-(grip_angle-ref_grip_angle).square()*weight.square()/.3**2).mean(-1),
            "contact": torch.exp(-(contacts[..., :4]-target[..., 35:39]).square().mean(-1)),
            "action_rate": -.1*(action-prev_action).square().sum(-1),
            "arm_torque": -.005*torque.index_select(-1,self.arm_ids).square().sum(-1),
            "joint_limit": -10.*limits.sum(-1),
            "foot_slip": -.1*(contacts[..., 4:]*feet_speed).sum(-1),
        }
        return {name: DT*value for name, value in terms.items()}

    def reward(self, *args, **kwargs):
        return sum(self.reward_terms(*args, **kwargs).values())
