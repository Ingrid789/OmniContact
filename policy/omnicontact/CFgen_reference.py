from typing import Any

import numpy as np

from policy.omnicontact.CFgen_meta1_loco import CfGenLoco
from policy.omnicontact.CFgen_meta2_carrybox import CfGenCarryBox
from policy.omnicontact.CFgen_meta3_pushbox_innerside import CfGenPushBoxInnerSide
from policy.omnicontact.CFgen_meta3_pushbox_twosides import CfGenPushBoxTwoSides
from policy.omnicontact.CFgen_meta4_slidebox import CfGenSlideBox
from policy.omnicontact.CFgen_meta5_relocateball import CfGenRelocateBall
from policy.omnicontact.CFgen_meta6_kickball import CfGenKickBall
from policy.omnicontact.CFgen_metaskill_chaining import (
    CfGenCarryCarryCarryBox,
    CfGenCarryPushBox,
    CfGenPushCarryBox,
    CfGenPushRelocateBall,
)
from policy.omnicontact.CFgen_stage_plans import (
    carry_carry_carry_plan,
    carrybox_pushbox_plan,
    push_relocate_plan,
)


SINGLE_OBJECT_CFGEN_TASKS = {
    "loco": (None, CfGenLoco),
    "pushbox-two": ("box", CfGenPushBoxTwoSides),
    "pushbox-in": ("box", CfGenPushBoxInnerSide),
    "slidebox": ("box", CfGenSlideBox),
    "slidebox-left": ("box", CfGenSlideBox),
    "slidebox-right": ("box", CfGenSlideBox),
    "kickball": ("ball", CfGenKickBall),
    "relocateball": ("ball", CfGenRelocateBall),
}

STAGED_CFGEN_TASKS = {
    "push-carry": carrybox_pushbox_plan,
    "carry-push": carrybox_pushbox_plan,
    "push-relocate": push_relocate_plan,
    "stackbox": carry_carry_carry_plan,
    "carry-carry": carry_carry_carry_plan,
    "carry-carry-carry": carry_carry_carry_plan,
}

SLIDEBOX_TASKS = {"slidebox", "slidebox-left", "slidebox-right"}
STAGE_SKIP_TOLERANCE = 0.2


def init_cfgen_state(policy: Any, pad: int = 30) -> None:
    policy.push_carry_stage = "idle"
    policy.push_carry_cfgen = CfGenPushCarryBox(pad=pad)
    policy.carry_push_cfgen = CfGenCarryPushBox(pad=pad)
    policy.push_relocate_stage = "idle"
    policy.push_relocate_cfgen = CfGenPushRelocateBall(pad=pad)
    policy.carry3_cfgen = CfGenCarryCarryCarryBox(pad=pad)
    policy.stackbox_stage_idx = 0
    policy.stackbox_stage_count = 3


def set_active_object_profile(policy: Any, object_name: str, dims: np.ndarray) -> None:
    policy.active_object_name = str(object_name)
    policy.box_dims = np.asarray(dims, dtype=np.float32).reshape(3).copy()
    policy.bbox_scale = policy.box_dims * 2.0
    policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)


def commit_cfgen_reference(policy: Any, traj_data: dict, target_yaw: float, goal_pos: np.ndarray) -> None:
    policy.target_yaw = float(target_yaw)
    for attr_name, data in traj_data.items():
        setattr(policy, attr_name, data)
    policy.goal_pos = goal_pos.copy()


def _near_goal(pos: np.ndarray, goal: np.ndarray, tolerance: float = STAGE_SKIP_TOLERANCE) -> bool:
    return float(np.linalg.norm(np.asarray(pos, dtype=np.float32).reshape(3) - np.asarray(goal, dtype=np.float32).reshape(3))) <= float(tolerance)


def _first_stage_plan(policy: Any) -> dict | None:
    if policy.task == "push-carry":
        return policy.push_carry_cfgen.stage_plan(
            policy.push_carry_cfgen.PUSH_STAGE,
            push_box_pos=policy.state_cmd.push_box_pos,
            push_box_quat=policy.state_cmd.push_box_quat,
            carry_box_pos=policy.state_cmd.carry_box_pos,
            carry_box_quat=policy.state_cmd.carry_box_quat,
            push_box_dims=policy.push_box_dims,
            carry_box_dims=policy.carry_box_dims,
            push_goal=policy.goal_pos,
        )
    if policy.task == "carry-push":
        return policy.carry_push_cfgen.stage_plan(
            policy.carry_push_cfgen.CARRY_STAGE,
            push_box_pos=policy.state_cmd.push_box_pos,
            push_box_quat=policy.state_cmd.push_box_quat,
            carry_box_pos=policy.state_cmd.carry_box_pos,
            carry_box_quat=policy.state_cmd.carry_box_quat,
            push_box_dims=policy.push_box_dims,
            carry_box_dims=policy.carry_box_dims,
            push_goal=policy.goal_pos,
        )
    if policy.task == "push-relocate":
        return policy.push_relocate_cfgen.stage_plan(
            policy.push_relocate_cfgen.PUSH_STAGE,
            push_box_pos=policy.state_cmd.push_box_pos,
            push_box_quat=policy.state_cmd.push_box_quat,
            ball_pos=policy.state_cmd.ball_pos,
            ball_quat=policy.state_cmd.ball_quat,
            push_box_dims=policy.push_box_dims,
            ball_dims=policy.ball_dims,
            push_goal=policy.goal_pos,
        )
    return None


def _first_stage_object_pos(policy: Any) -> np.ndarray | None:
    if policy.task in {"push-carry", "push-relocate"}:
        return np.asarray(policy.state_cmd.push_box_pos, dtype=np.float32).reshape(3)
    if policy.task == "carry-push":
        return np.asarray(policy.state_cmd.carry_box_pos, dtype=np.float32).reshape(3)
    return None


def _should_skip_first_stage(policy: Any) -> bool:
    plan = _first_stage_plan(policy)
    obj_pos = _first_stage_object_pos(policy)
    if plan is None or obj_pos is None:
        return False
    return _near_goal(obj_pos, plan["goal"])


def plan_cfgen_reference(policy: Any, fk_info: dict) -> None:
    plan_goal = policy.goal_pos.copy()

    staged_plan_fn = STAGED_CFGEN_TASKS.get(policy.task)
    if staged_plan_fn is not None:
        plan, traj_data, target_yaw = staged_plan_fn(policy, fk_info)
        plan_goal = plan["goal"]
        commit_cfgen_reference(policy, traj_data, target_yaw, plan_goal)
        return

    object_profile, cfgen_cls = SINGLE_OBJECT_CFGEN_TASKS.get(policy.task, (None, CfGenCarryBox))
    if object_profile is not None:
        set_active_object_profile(policy, object_profile, policy.box_dims)

    policy.traj_generator = cfgen_cls(pad=30)
    generate_kwargs = dict(
        pelvis_pos=fk_info["pelvis"]["pos"],
        pelvis_quat=fk_info["pelvis"]["quat"],
        obj_pos=policy.state_cmd.obj_pos.copy(),
        obj_quat=policy.state_cmd.obj_quat.copy(),
        box_half_dims=policy.box_dims,
        target_obj_pos=plan_goal,
    )
    if policy.task in SLIDEBOX_TASKS:
        generate_kwargs["task"] = policy.task

    traj_data, target_yaw = policy.traj_generator.generate(**generate_kwargs)
    commit_cfgen_reference(policy, traj_data, target_yaw, plan_goal)


def initialize_cfgen_reference(policy: Any, fk_info: dict) -> None:
    if policy.task == "push-carry":
        if _should_skip_first_stage(policy):
            policy.push_carry_stage = policy.push_carry_cfgen.CARRY_STAGE
            set_active_object_profile(policy, "carry_box", policy.carry_box_dims)
            print("[CFgen] skip push-carry first stage; push_box already near goal.")
        else:
            policy.push_carry_stage = policy.push_carry_cfgen.PUSH_STAGE
            set_active_object_profile(policy, "push_box", policy.push_box_dims)
    elif policy.task == "carry-push":
        if _should_skip_first_stage(policy):
            policy.push_carry_stage = policy.carry_push_cfgen.PUSH_STAGE
            set_active_object_profile(policy, "push_box", policy.push_box_dims)
            print("[CFgen] skip carry-push first stage; carry_box already near goal.")
        else:
            policy.push_carry_stage = policy.carry_push_cfgen.CARRY_STAGE
            set_active_object_profile(policy, "carry_box", policy.carry_box_dims)
    elif policy.task == "push-relocate":
        if _should_skip_first_stage(policy):
            policy.push_relocate_stage = policy.push_relocate_cfgen.RELOCATE_STAGE
            set_active_object_profile(policy, "ball", policy.ball_dims)
            print("[CFgen] skip push-relocate first stage; push_box already near goal.")
        else:
            policy.push_relocate_stage = policy.push_relocate_cfgen.PUSH_STAGE
            set_active_object_profile(policy, "push_box", policy.push_box_dims)
    elif policy.task in {"stackbox", "carry-carry", "carry-carry-carry"}:
        policy.stackbox_stage_idx = 0
        policy.stackbox_stage_count = 2 if policy.task == "carry-carry" else 3

    plan_cfgen_reference(policy, fk_info)
