"""FK-based pure-locomotion primitives for OmniContact CF generators."""

import numpy as np
import yaml

from common.mujoco_kinematics import MujocoKinematics
from common.path_config import PROJECT_ROOT
from common.utils import (
    align_quat_hemisphere,
    normalize_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_slerp,
    yaw_quat,
    yaw_to_quat,
)


FK_LINKS = (
    ("torso", "torso_link"),
    ("la", "left_ankle_pitch_link"),
    ("ra", "right_ankle_pitch_link"),
    ("lw", "left_palm_link"),
    ("rw", "right_palm_link"),
)
DEFAULT_PELVIS_Z = 0.77
NO_CONTACT = np.zeros(4, dtype=np.float32)

KINEMATICS = MujocoKinematics(xml_path=f"{PROJECT_ROOT}/g1_description/g1_29dof.xml")
with open(f"{PROJECT_ROOT}/policy/defaultpose/config/DefaultPose.yaml", "r") as f:
    DEFAULT_JOINT_POS_MJ = np.asarray(yaml.load(f, Loader=yaml.FullLoader)["default_angles"], dtype=np.float32).reshape(-1)


def _quat_to_rpy_deg(q: np.ndarray) -> tuple[float, float, float]:
    """Convert quat [w, x, y, z] to roll/pitch/yaw in degrees."""
    q = normalize_quat(q)
    w, x, y, z = [float(v) for v in q]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(np.rad2deg(roll)), float(np.rad2deg(pitch)), float(np.rad2deg(yaw))


def _interp_linear(p0: np.ndarray, p1: np.ndarray, step: float) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float32).reshape(3)
    p1 = np.asarray(p1, dtype=np.float32).reshape(3)
    dist = float(np.linalg.norm(p1 - p0))
    if dist < 1e-6:
        return p0[None, :]
    n = max(int(dist / float(step)), 1) + 1
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return (p0[None, :] + (p1 - p0)[None, :] * t).astype(np.float32)


def _interp_slerp(q0: np.ndarray, q1: np.ndarray, step_rad: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.clip(abs(float(np.dot(q0, q1))), -1.0, 1.0))
    ang = float(np.arccos(dot)) * 2.0
    if ang < 1e-6:
        return q0[None, :]
    n = max(int(ang / float(step_rad)), 1) + 1
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.array([normalize_quat(quat_slerp(q0, q1, float(t))) for t in ts], dtype=np.float32)


def _tile(x: np.ndarray, n: int) -> np.ndarray:
    """Repeat one vector or array for n trajectory frames."""
    x = np.asarray(x, dtype=np.float32)
    return np.tile(x.reshape((1,) + x.shape), (n,) + (1,) * x.ndim).astype(np.float32)


def _fk_reference_sequence(
    base_pos_seq: np.ndarray,
    base_quat_seq: np.ndarray,
    dof_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run FK over a base trajectory and collect the reference link poses."""
    dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(-1)
    base_pos_seq = np.asarray(base_pos_seq, dtype=np.float32).reshape(-1, 3)
    base_quat_seq = np.asarray(base_quat_seq, dtype=np.float32).reshape(-1, 4)
    n = int(len(base_pos_seq))
    refs = {
        "base_p": base_pos_seq,
        "base_q": align_quat_hemisphere(base_quat_seq),
        "joint_pos_mj": _tile(dof_pos, n),
    }
    for key, _ in FK_LINKS:
        refs[f"{key}_p"] = np.zeros((n, 3), dtype=np.float32)
        refs[f"{key}_q"] = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        fk_info = KINEMATICS.forward(dof_pos, base_pos_seq[i], base_quat_seq[i])
        for key, body_name in FK_LINKS:
            refs[f"{key}_p"][i] = fk_info[body_name]["pos"]
            refs[f"{key}_q"][i] = fk_info[body_name]["quat"]

    for key, _ in FK_LINKS:
        refs[f"{key}_q"] = align_quat_hemisphere(refs[f"{key}_q"])
    return refs


def _fk_reference_sequence_from_joints(
    base_pos_seq: np.ndarray,
    base_quat_seq: np.ndarray,
    joint_seq: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run FK over base and joint trajectories with one joint vector per frame."""
    base_pos_seq = np.asarray(base_pos_seq, dtype=np.float32).reshape(-1, 3)
    base_quat_seq = align_quat_hemisphere(np.asarray(base_quat_seq, dtype=np.float32).reshape(-1, 4))
    joint_seq = np.asarray(joint_seq, dtype=np.float32).reshape(-1, len(DEFAULT_JOINT_POS_MJ))
    n = int(len(joint_seq))
    if len(base_pos_seq) != n or len(base_quat_seq) != n:
        raise ValueError("base_pos_seq, base_quat_seq, and joint_seq must have the same length.")

    refs = {
        "base_p": base_pos_seq,
        "base_q": base_quat_seq,
        "joint_pos_mj": joint_seq,
    }
    for key, _ in FK_LINKS:
        refs[f"{key}_p"] = np.zeros((n, 3), dtype=np.float32)
        refs[f"{key}_q"] = np.zeros((n, 4), dtype=np.float32)

    for i, q in enumerate(joint_seq):
        fk_info = KINEMATICS.forward(q, base_pos_seq[i], base_quat_seq[i])
        for key, body_name in FK_LINKS:
            refs[f"{key}_p"][i] = fk_info[body_name]["pos"]
            refs[f"{key}_q"][i] = fk_info[body_name]["quat"]

    for key, _ in FK_LINKS:
        refs[f"{key}_q"] = align_quat_hemisphere(refs[f"{key}_q"])
    return refs


def _lerp_sequence(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    """Fixed-length linear interpolation."""
    u = np.linspace(0.0, 1.0, max(int(n), 2), dtype=np.float32)
    start = np.asarray(start, dtype=np.float32).reshape(1, -1)
    end = np.asarray(end, dtype=np.float32).reshape(1, -1)
    return ((1.0 - u[:, None]) * start + u[:, None] * end).astype(np.float32)


def _base_sequence_from_torso(
    torso_pos_seq: np.ndarray,
    base_quat_seq: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    """Convert legacy torso-position inputs into pelvis/base positions."""
    torso_pos_seq = np.asarray(torso_pos_seq, dtype=np.float32).reshape(-1, 3)
    base_quat_seq = np.asarray(base_quat_seq, dtype=np.float32).reshape(-1, 4)
    fk_info0 = KINEMATICS.forward(
        np.asarray(dof_pos, dtype=np.float32).reshape(-1),
        np.zeros(3, dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    base_to_torso = fk_info0["torso_link"]["pos"].astype(np.float32)
    return np.array(
        [torso_p - quat_apply(base_q, base_to_torso) for torso_p, base_q in zip(torso_pos_seq, base_quat_seq)],
        dtype=np.float32,
    )


def _append_fk_block(
    builder,
    phase: int,
    *,
    fk_refs: dict[str, np.ndarray],
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    contact: np.ndarray,
) -> None:
    """Append FK-derived body references to the shared trajectory builder."""
    n = int(len(fk_refs["torso_p"]))
    object_pos = np.asarray(object_pos, dtype=np.float32)
    object_quat = np.asarray(object_quat, dtype=np.float32)
    if object_pos.ndim == 1:
        object_pos = _tile(object_pos, n)
    if object_quat.ndim == 1:
        object_quat = _tile(object_quat, n)
    builder.append(
        int(phase),
        lw_p=fk_refs["lw_p"],
        lw_q=fk_refs["lw_q"],
        rw_p=fk_refs["rw_p"],
        rw_q=fk_refs["rw_q"],
        obj_p=object_pos,
        obj_q=object_quat,
        torso_p=fk_refs["torso_p"],
        torso_yaw_q=fk_refs["torso_q"],
        torso_pitch_deg=np.zeros(n, dtype=np.float32),
        la_p=fk_refs["la_p"],
        la_q=fk_refs["la_q"],
        ra_p=fk_refs["ra_p"],
        ra_q=fk_refs["ra_q"],
        dof_pos=fk_refs["joint_pos_mj"],
        base_p=fk_refs["base_p"],
        base_q=fk_refs["base_q"],
        contact=contact,
    )


def _object_sequence_in_base_frame(
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    base_pos_seq: np.ndarray,
    base_quat_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep object pose fixed in the first-frame pelvis/base frame."""
    object_pos = np.asarray(object_pos, dtype=np.float32).reshape(3)
    object_quat = np.asarray(object_quat, dtype=np.float32).reshape(4)
    base_pos_seq = np.asarray(base_pos_seq, dtype=np.float32).reshape(-1, 3)
    base_quat_seq = align_quat_hemisphere(np.asarray(base_quat_seq, dtype=np.float32).reshape(-1, 4))

    inv_base0 = quat_conjugate(base_quat_seq[0])
    local_pos = quat_apply(inv_base0, object_pos - base_pos_seq[0]).astype(np.float32)
    local_quat = quat_mul(inv_base0, object_quat).astype(np.float32)

    object_pos_seq = np.array(
        [base_pos + quat_apply(base_quat, local_pos) for base_pos, base_quat in zip(base_pos_seq, base_quat_seq)],
        dtype=np.float32,
    )
    object_quat_seq = align_quat_hemisphere(
        np.array([quat_mul(base_quat, local_quat) for base_quat in base_quat_seq], dtype=np.float32)
    )
    return object_pos_seq, object_quat_seq


def _append_turn(
    builder,
    phase: int,
    *,
    pelvis_pos: np.ndarray | None,
    torso_pos: np.ndarray | None,
    yaw_start: np.ndarray,
    yaw_target: np.ndarray,
    step_angular: float,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    dof_pos: np.ndarray,
    contact: np.ndarray,
    object_in_base_frame: bool = False,
    keep_pelvis_z: bool = False,
) -> dict[str, np.ndarray]:
    """Append an FK turn with explicit joint pose and contact flags."""
    base_quat_seq = align_quat_hemisphere(_interp_slerp(yaw_start, yaw_target, step_angular))
    if pelvis_pos is not None:
        base_pos_seq = _tile(np.asarray(pelvis_pos, dtype=np.float32).reshape(3), len(base_quat_seq))
        if not keep_pelvis_z:
            base_pos_seq[:, 2] = DEFAULT_PELVIS_Z
    elif torso_pos is not None:
        torso_pos_seq = _tile(np.asarray(torso_pos, dtype=np.float32).reshape(3), len(base_quat_seq))
        base_pos_seq = _base_sequence_from_torso(torso_pos_seq, base_quat_seq, dof_pos)
    else:
        raise ValueError("_append_loco_turn needs either pelvis_pos or torso_pos.")
    fk_refs = _fk_reference_sequence(base_pos_seq, base_quat_seq, dof_pos)
    if object_in_base_frame:
        object_pos, object_quat = _object_sequence_in_base_frame(object_pos, object_quat, base_pos_seq, base_quat_seq)
    _append_fk_block(
        builder,
        phase,
        fk_refs=fk_refs,
        object_pos=object_pos,
        object_quat=object_quat,
        contact=contact,
    )
    fk_refs["yaw"] = base_quat_seq
    return fk_refs


def _append_walk(
    builder,
    phase: int,
    *,
    pelvis_start: np.ndarray | None,
    pelvis_target: np.ndarray | None,
    torso_start: np.ndarray | None,
    torso_target: np.ndarray | None,
    yaw: np.ndarray,
    step_linear: float,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    dof_pos: np.ndarray,
    contact: np.ndarray,
    object_in_base_frame: bool = False,
    keep_pelvis_z: bool = False,
) -> dict[str, np.ndarray]:
    """Append an FK walk with explicit joint pose and contact flags."""
    if pelvis_start is not None and pelvis_target is not None:
        base_pos_seq = _interp_linear(pelvis_start, pelvis_target, step_linear).astype(np.float32)
        if not keep_pelvis_z:
            base_pos_seq[:, 2] = DEFAULT_PELVIS_Z
    elif torso_start is not None and torso_target is not None:
        torso_pos_seq = _interp_linear(torso_start, torso_target, step_linear).astype(np.float32)
        torso_pos_seq[:, 2] = DEFAULT_PELVIS_Z
        base_quat_seq_for_torso = _tile(yaw, len(torso_pos_seq))
        base_pos_seq = _base_sequence_from_torso(torso_pos_seq, base_quat_seq_for_torso, dof_pos)
    else:
        raise ValueError("_append_loco_walk needs either pelvis_start/pelvis_target or torso_start/torso_target.")
    base_quat_seq = _tile(yaw, len(base_pos_seq))
    fk_refs = _fk_reference_sequence(base_pos_seq, base_quat_seq, dof_pos)
    if object_in_base_frame:
        object_pos, object_quat = _object_sequence_in_base_frame(object_pos, object_quat, base_pos_seq, base_quat_seq)
    _append_fk_block(
        builder,
        phase,
        fk_refs=fk_refs,
        object_pos=object_pos,
        object_quat=object_quat,
        contact=contact,
    )
    fk_refs["yaw"] = base_quat_seq
    return fk_refs


def _append_loco_turn(
    builder,
    phase: int,
    *,
    pelvis_pos: np.ndarray | None = None,
    torso_pos: np.ndarray | None = None,
    yaw_start: np.ndarray,
    yaw_target: np.ndarray,
    step_angular: float,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
) -> dict[str, np.ndarray]:
    """Append an FK-based in-place turn using the DefaultPose G1 joint pose."""
    return _append_turn(
        builder,
        phase,
        pelvis_pos=pelvis_pos,
        torso_pos=torso_pos,
        yaw_start=yaw_start,
        yaw_target=yaw_target,
        step_angular=step_angular,
        object_pos=object_pos,
        object_quat=object_quat,
        dof_pos=DEFAULT_JOINT_POS_MJ,
        contact=NO_CONTACT,
    )


def _append_loco_walk(
    builder,
    phase: int,
    *,
    pelvis_start: np.ndarray | None = None,
    pelvis_target: np.ndarray | None = None,
    torso_start: np.ndarray | None = None,
    torso_target: np.ndarray | None = None,
    yaw: np.ndarray,
    step_linear: float,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
) -> dict[str, np.ndarray]:
    """Append an FK-based walk using the DefaultPose G1 joint pose and fixed yaw."""
    return _append_walk(
        builder,
        phase,
        pelvis_start=pelvis_start,
        pelvis_target=pelvis_target,
        torso_start=torso_start,
        torso_target=torso_target,
        yaw=yaw,
        step_linear=step_linear,
        object_pos=object_pos,
        object_quat=object_quat,
        dof_pos=DEFAULT_JOINT_POS_MJ,
        contact=NO_CONTACT,
    )


def _append_loco_approach(
    builder,
    *,
    phase_turn_to_walk: int = 11,
    phase_walk: int = 12,
    phase_turn_to_target: int = 13,
    pelvis_start: np.ndarray,
    pelvis_target: np.ndarray,
    yaw_start: np.ndarray,
    yaw_target: np.ndarray,
    step_linear: float,
    step_angular: float,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Append the common approach sequence: turn toward target, walk, then final turn."""
    walk_delta = (
        np.asarray(pelvis_target, dtype=np.float32).reshape(3)
        - np.asarray(pelvis_start, dtype=np.float32).reshape(3)
    ).astype(np.float32)
    if float(np.linalg.norm(walk_delta[:2])) < 1e-6:
        walk_yaw = np.asarray(yaw_start, dtype=np.float32).reshape(4).copy()
    else:
        walk_yaw = yaw_to_quat(np.arctan2(float(walk_delta[1]), float(walk_delta[0]))).astype(np.float32)
    turn11 = _append_loco_turn(
        builder,
        phase_turn_to_walk,
        pelvis_pos=pelvis_start,
        yaw_start=yaw_start,
        yaw_target=walk_yaw,
        step_angular=step_angular,
        object_pos=object_pos,
        object_quat=object_quat,
    )
    walk12 = _append_loco_walk(
        builder,
        phase_walk,
        pelvis_start=turn11["base_p"][-1],
        pelvis_target=pelvis_target,
        yaw=turn11["base_q"][-1],
        step_linear=step_linear,
        object_pos=object_pos,
        object_quat=object_quat,
    )
    turn13 = _append_loco_turn(
        builder,
        phase_turn_to_target,
        pelvis_pos=walk12["base_p"][-1],
        yaw_start=walk12["base_q"][-1],
        yaw_target=yaw_target,
        step_angular=step_angular,
        object_pos=object_pos,
        object_quat=object_quat,
    )

    return {"turn_to_walk": turn11, "walk": walk12, "turn_to_target": turn13}


def _append_contactloco_turn(
    builder,
    phase: int,
    *,
    pelvis_pos: np.ndarray | None = None,
    torso_pos: np.ndarray | None = None,
    yaw_start: np.ndarray,
    yaw_target: np.ndarray,
    step_angular: float,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    dof_pos: np.ndarray,
    contact: np.ndarray,
    preserve_pelvis_tilt: bool = True,
) -> dict[str, np.ndarray]:
    """Append an FK-based in-place turn with caller-provided joints and contact."""
    if preserve_pelvis_tilt:
        yaw_start_only = yaw_quat(yaw_start).astype(np.float32)
        pelvis_tilt = quat_mul(quat_conjugate(yaw_start_only), yaw_start).astype(np.float32)
        yaw_seq = _interp_slerp(yaw_start_only, yaw_target, step_angular)
        base_quat_seq = align_quat_hemisphere(
            np.array([quat_mul(yaw, pelvis_tilt) for yaw in yaw_seq], dtype=np.float32)
        )
        if pelvis_pos is not None:
            base_pos_seq = _tile(np.asarray(pelvis_pos, dtype=np.float32).reshape(3), len(base_quat_seq))
        elif torso_pos is not None:
            torso_pos_seq = _tile(np.asarray(torso_pos, dtype=np.float32).reshape(3), len(base_quat_seq))
            base_pos_seq = _base_sequence_from_torso(torso_pos_seq, base_quat_seq, dof_pos)
        else:
            raise ValueError("_append_contactloco_turn needs either pelvis_pos or torso_pos.")
        fk_refs = _fk_reference_sequence(base_pos_seq, base_quat_seq, dof_pos)
        object_pos, object_quat = _object_sequence_in_base_frame(object_pos, object_quat, base_pos_seq, base_quat_seq)
        _append_fk_block(
            builder,
            phase,
            fk_refs=fk_refs,
            object_pos=object_pos,
            object_quat=object_quat,
            contact=contact,
        )
        fk_refs["yaw"] = base_quat_seq
        return fk_refs
    return _append_turn(
        builder,
        phase,
        pelvis_pos=pelvis_pos,
        torso_pos=torso_pos,
        yaw_start=yaw_start,
        yaw_target=yaw_target,
        step_angular=step_angular,
        object_pos=object_pos,
        object_quat=object_quat,
        dof_pos=dof_pos,
        contact=contact,
        object_in_base_frame=True,
        keep_pelvis_z=True,
    )


def _append_contactloco_walk(
    builder,
    phase: int,
    *,
    pelvis_start: np.ndarray | None = None,
    pelvis_target: np.ndarray | None = None,
    torso_start: np.ndarray | None = None,
    torso_target: np.ndarray | None = None,
    yaw: np.ndarray,
    step_linear: float,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    dof_pos: np.ndarray,
    contact: np.ndarray,
    keep_pelvis_z: bool = False,
) -> dict[str, np.ndarray]:
    """Append an FK-based walk with caller-provided joints and contact."""
    return _append_walk(
        builder,
        phase,
        pelvis_start=pelvis_start,
        pelvis_target=pelvis_target,
        torso_start=torso_start,
        torso_target=torso_target,
        yaw=yaw,
        step_linear=step_linear,
        object_pos=object_pos,
        object_quat=object_quat,
        dof_pos=dof_pos,
        contact=contact,
        object_in_base_frame=True,
        keep_pelvis_z=keep_pelvis_z,
    )


def _append_contactloco_recover(
    builder,
    phase: int,
    *,
    recover_frames: int = 61,
    recover_contact: np.ndarray = NO_CONTACT,
    recover_dof_pos: np.ndarray = DEFAULT_JOINT_POS_MJ,
    recover_pelvis_z: float = DEFAULT_PELVIS_Z,
    recover_pelvis_pitch: bool = True,
    recover_pad: int = 0,
) -> dict[str, np.ndarray]:
    """Append a contact-locomotion recovery phase to the default pose."""
    n = int(max(2, recover_frames))
    qseq = _lerp_sequence(builder.last("dof_pos"), recover_dof_pos, n)

    pelvis_target = np.asarray(builder.last("base_p"), dtype=np.float32).copy()
    pelvis_target[2] = float(recover_pelvis_z)
    pelvis_seq = _lerp_sequence(builder.last("base_p"), pelvis_target, n).reshape(n, 3)

    base_q_start = np.asarray(builder.last("base_q"), dtype=np.float32).reshape(4)
    base_q_target = yaw_quat(base_q_start).astype(np.float32) if recover_pelvis_pitch else base_q_start.copy()
    u = np.linspace(0.0, 1.0, n, dtype=np.float32)
    base_q_seq = align_quat_hemisphere(
        np.array([quat_slerp(base_q_start, base_q_target, float(t)) for t in u], dtype=np.float32)
    )

    refs = _fk_reference_sequence_from_joints(pelvis_seq, base_q_seq, qseq)
    _append_fk_block(
        builder,
        phase,
        fk_refs=refs,
        object_pos=builder.last("obj_p"),
        object_quat=builder.last("obj_q"),
        contact=recover_contact,
    )
    builder.pad(phase, contact=recover_contact, count=recover_pad)
    return refs


def _contactloco_walk_object_target(
    obj_start: np.ndarray,
    *,
    object_goal_pos: np.ndarray | None,
    object_target_pos: np.ndarray | None,
    object_goal_standoff: float,
    keep_pelvis_z: bool,
) -> np.ndarray:
    """Resolve the object walk target from either an exact target or a goal plus standoff."""
    if object_target_pos is not None:
        obj_target = np.asarray(object_target_pos, dtype=np.float32).reshape(3).copy()
    else:
        if object_goal_pos is None:
            raise ValueError("Either object_target_pos or object_goal_pos must be provided.")
        obj_target = np.asarray(object_goal_pos, dtype=np.float32).reshape(3).copy()
        move_dir = (obj_target - obj_start).astype(np.float32)
        move_dir[2] = 0.0
        move_norm = float(np.linalg.norm(move_dir[:2]))
        move_xy = np.divide(
            move_dir[:2],
            move_norm,
            out=np.zeros(2, dtype=np.float32),
            where=move_norm >= 1e-6,
        )
        obj_target[:2] -= move_xy * float(object_goal_standoff)

    if keep_pelvis_z:
        obj_target[2] = float(obj_start[2])
    return obj_target


def _append_contactloco_turn_walk_recover(
    builder,
    *,
    phase_turn: int | None,
    phase_walk: int,
    phase_recover: int | None = None,
    yaw_target: np.ndarray | None = None,
    object_goal_pos: np.ndarray | None = None,
    object_target_pos: np.ndarray | None = None,
    object_goal_standoff: float = 0.0,
    step_angular: float,
    step_linear: float,
    contact: np.ndarray,
    keep_pelvis_z: bool = True,
    turn_pad: int = 0,
    walk_pad: int = 0,
    recover_pad: int = 0,
    recover_frames: int = 61,
    recover_contact: np.ndarray = NO_CONTACT,
    recover_dof_pos: np.ndarray = DEFAULT_JOINT_POS_MJ,
    recover_pelvis_z: float = DEFAULT_PELVIS_Z,
    recover_pelvis_pitch: bool = True,
    preserve_turn_pelvis_tilt: bool = True,
) -> dict[str, dict[str, np.ndarray] | None]:
    """Append a common contact-locomotion sequence: optional turn, walk, optional recover."""

    out: dict[str, dict[str, np.ndarray] | None] = {"turn": None, "walk": None, "recover": None}

    if phase_turn is not None:
        if yaw_target is None:
            raise ValueError("yaw_target is required when phase_turn is not None.")
        out["turn"] = _append_contactloco_turn(
            builder,
            phase_turn,
            pelvis_pos=builder.last("base_p"),
            yaw_start=builder.last("base_q"),
            yaw_target=yaw_target,
            step_angular=step_angular,
            object_pos=builder.last("obj_p"),
            object_quat=builder.last("obj_q"),
            dof_pos=builder.last("dof_pos"),
            contact=contact,
            preserve_pelvis_tilt=preserve_turn_pelvis_tilt,
        )
        builder.pad(phase_turn, contact=contact, count=turn_pad)

    obj_start = np.asarray(builder.last("obj_p"), dtype=np.float32).reshape(3)
    obj_target = _contactloco_walk_object_target(
        obj_start,
        object_goal_pos=object_goal_pos,
        object_target_pos=object_target_pos,
        object_goal_standoff=object_goal_standoff,
        keep_pelvis_z=keep_pelvis_z,
    )
    delta = (obj_target - obj_start).astype(np.float32)
    if keep_pelvis_z:
        delta[2] = 0.0
    walk = _append_contactloco_walk(
        builder,
        phase_walk,
        pelvis_start=builder.last("base_p"),
        pelvis_target=builder.last("base_p") + delta,
        yaw=builder.last("base_q"),
        step_linear=step_linear,
        object_pos=obj_start,
        object_quat=builder.last("obj_q"),
        dof_pos=builder.last("dof_pos"),
        contact=contact,
        keep_pelvis_z=keep_pelvis_z,
    )
    out["walk"] = walk
    builder.pad(phase_walk, contact=contact, count=walk_pad)

    if phase_recover is not None:
        out["recover"] = _append_contactloco_recover(
            builder,
            phase_recover,
            recover_frames=recover_frames,
            recover_contact=recover_contact,
            recover_dof_pos=recover_dof_pos,
            recover_pelvis_z=recover_pelvis_z,
            recover_pelvis_pitch=recover_pelvis_pitch,
            recover_pad=recover_pad,
        )

    return out


class CfGenLoco:
    """CF generator for pure locomotion from the current pelvis pose to a target pelvis pose."""

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        self.pad = int(pad)
        self.step_size_linear = float(step_size_linear)
        self.step_size_angular = float(step_size_angular)

    @staticmethod
    def _target_yaw(pelvis_pos: np.ndarray, target_pos: np.ndarray, pelvis_quat: np.ndarray) -> tuple[np.ndarray, float]:
        delta = (np.asarray(target_pos, dtype=np.float32).reshape(3) - np.asarray(pelvis_pos, dtype=np.float32).reshape(3))
        delta[2] = 0.0
        if float(np.linalg.norm(delta[:2])) < 1e-6:
            yaw_q = yaw_quat(pelvis_quat).astype(np.float32)
            _, _, yaw_deg = _quat_to_rpy_deg(yaw_q)
            return yaw_q, float(np.deg2rad(yaw_deg))

        yaw = float(np.arctan2(float(delta[1]), float(delta[0])))
        return yaw_to_quat(yaw).astype(np.float32), yaw

    def generate(
        self,
        *,
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        target_obj_pos: np.ndarray,
        obj_pos: np.ndarray | None = None,
        obj_quat: np.ndarray | None = None,
        box_half_dims: np.ndarray | None = None,
    ) -> tuple[dict, float]:
        from policy.omnicontact.CFgen_builder import _TrajBuilder

        pelvis_start = np.asarray(pelvis_pos, dtype=np.float32).reshape(3).copy()
        pelvis_target = np.asarray(target_obj_pos, dtype=np.float32).reshape(3).copy()
        pelvis_start[2] = float(DEFAULT_PELVIS_Z)
        pelvis_target[2] = float(DEFAULT_PELVIS_Z)

        object_pos = pelvis_start.copy() if obj_pos is None else np.asarray(obj_pos, dtype=np.float32).reshape(3).copy()
        object_quat = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if obj_quat is None
            else np.asarray(obj_quat, dtype=np.float32).reshape(4).copy()
        )

        yaw_start = yaw_quat(pelvis_quat).astype(np.float32)
        yaw_target_q, target_yaw = self._target_yaw(pelvis_start, pelvis_target, pelvis_quat)

        builder = _TrajBuilder()
        _append_loco_approach(
            builder,
            phase_turn_to_walk=11,
            phase_walk=12,
            phase_turn_to_target=13,
            pelvis_start=pelvis_start,
            pelvis_target=pelvis_target,
            yaw_start=yaw_start,
            yaw_target=yaw_target_q,
            step_linear=self.step_size_linear,
            step_angular=self.step_size_angular,
            object_pos=object_pos,
            object_quat=object_quat,
        )
        builder.pad(13, contact=NO_CONTACT, count=self.pad)
        return builder.finalize(), target_yaw
