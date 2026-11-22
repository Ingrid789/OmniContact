"""Carry reference clips in MuJoCo joint order and world coordinates (wxyz)."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import numpy as np


OFFSETS = np.asarray([0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50])
DT = 0.02
RAW_DIM = 111
LINKS = ("left_palm_link", "right_palm_link", "torso_link",
         "left_ankle_pitch_link", "right_ankle_pitch_link")
# The same permutation as OmniContact.yaml's lab2mj; a[:, LAB2MJ] is MJ order.
LAB2MJ = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                   11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
MJ2LAB = np.argsort(LAB2MJ)
SIZE = re.compile(r"^carry_box_(\d+)-(\d+)-(\d+)_(.+)$")


def split_paths(directory, split):
    """All geometries and numbered cuts of a source motion share a partition."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown partition {split!r}")
    groups = {}
    for file in sorted(Path(directory).glob("*.npz")):
        match = SIZE.match(file.stem)
        source = match.group(4).rsplit("_", 1)[0] if match else file.stem
        groups.setdefault(source, []).append(file)
    if len(groups) < 3:
        raise ValueError(f"Need at least 3 independent motions in {directory} for disjoint splits")
    keys = np.asarray(sorted(groups))
    np.random.default_rng(5302).shuffle(keys)
    n = max(1, int(len(keys) * .1))
    partition = {"val": keys[:n], "test": keys[n:2*n], "train": keys[2*n:]}[split]
    return tuple(p for key in partition for p in groups[key])


def _pose(pos, quat):
    quat = np.asarray(quat)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.e-5):
        raise ValueError("Reference contains an invalid zero quaternion")
    return np.concatenate((pos, quat / norm), axis=-1)


def pack_reference(links, contact, q, dq, pelvis, box):
    frames = np.concatenate((links.reshape(-1, 35), contact.reshape(-1, 4), q, dq,
                             pelvis, box), axis=-1).astype(np.float32)
    if frames.shape[1] != RAW_DIM or len(frames) < 2 or not np.isfinite(frames).all():
        raise ValueError(f"Invalid carry reference {frames.shape}")
    return frames


def raw_window(clip, index):
    return clip[np.minimum(np.asarray(index)[..., None] + OFFSETS, len(clip)-1)]


@dataclass
class SceneClip:
    frames: np.ndarray
    half: np.ndarray
    tables: np.ndarray
    # First root and box twists in world frame for the physics reset.
    twists: np.ndarray
    teacher_ankles: np.ndarray


@lru_cache(maxsize=32)
def load_demo(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        if not np.isclose(np.ravel(d["fps"])[0], 1/DT):
            raise ValueError(f"{path} is not a 50 Hz carry clip")
        # The NPZ tracking ankles (25,26) are roll links, 17.558 mm below
        # the pitch links (21,22). Use matching pitch targets for both V2 sources.
        # The legacy ONNX teacher retains its training-time roll targets.
        bodies = _pose(d["body_pos_w"], d["body_quat_w"])
        links = bodies[:, [37, 38, 11, 21, 22]]
        teacher_ankles = bodies[:, [25, 26]].copy()
        box = _pose(d["object_pos_w"], d["object_quat_w"])
        frames = pack_reference(links, d["contact_info"], d["joint_pos"][:, LAB2MJ],
                                d["joint_vel"][:, LAB2MJ], bodies[:, 0], box)
        if "carry_box_selected_size_cm" in d:
            half = np.ravel(d["carry_box_selected_size_cm"]).astype(float)/200
        elif (match := SIZE.match(path.stem)) is not None:
            half = np.array(match.groups()[:3], dtype=float)/200
        elif path.name.startswith("carrybox_motion_"):
            # Official sim2sim demo uses the included 30 cm cube scene.
            half = np.full(3, .15)
        else:
            raise ValueError(f"No physical box dimensions found for {path}")
        tables = np.stack([d[key].reshape(-1, 3)[0] for key in ("table1_pos_w", "table2_pos_w")])
        twists = np.stack((d["body_lin_vel_w"][0, 0], d["body_ang_vel_w"][0, 0],
                           d["object_lin_vel_w"][0], d["object_ang_vel_w"][0]))
    if half.shape != (3,) or np.any(half <= 0) or not np.isfinite(half).all():
        raise ValueError(f"Invalid physical half-sizes {half} for {path}")
    return SceneClip(frames, half, tables, twists, teacher_ankles)


def generate_clip(generator, root, box, half, goal):
    traj, _ = generator.generate(root[:3], root[3:7], box[:3], box[3:7],
                                 box_half_dims=half, target_obj_pos=goal)
    pairs = (("ref_left_wrist_pos", "ref_left_wrist_quat"),
             ("ref_right_wrist_pos", "ref_right_wrist_quat"),
             ("ref_torso_future_pos", "ref_torso_future_quat"),
             ("ref_left_ankle_future_pos", "ref_left_ankle_future_quat"),
             ("ref_right_ankle_future_pos", "ref_right_ankle_future_quat"))
    links = np.stack([_pose(traj[p], traj[r]) for p, r in pairs], axis=1)
    q = np.asarray(traj["dof_pos"], dtype=np.float32)
    dq = np.gradient(q, DT, axis=0)
    frames = pack_reference(links, traj["ref_contact"], q, dq,
                            _pose(traj["ref_base_pos"], traj["ref_base_quat"]),
                            _pose(traj["ref_object_pos"], traj["ref_object_quat"]))
    return frames, traj["ref_phase"]
