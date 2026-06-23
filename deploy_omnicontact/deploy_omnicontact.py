import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

from dataclasses import dataclass, field
import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import PolicyOutput, StateAndCmd
from FSM.FSM import FSM
from common.utils import FSMCommand, get_gravity_orientation, quat_mul, quat_conjugate, yaw_quat
from common.joystick import JoyStick, JoystickButton

def pd_control(target_q, q, kp, target_dq, dq, kd, torque_limit_mj=None, torque_clip=True):
    """ Calculates torques from position commands """
    tau = (target_q - q) * kp + (target_dq - dq) * kd
    
    if torque_clip and torque_limit_mj is not None:
        tau = np.clip(tau, -torque_limit_mj, torque_limit_mj)
    return tau


def config_array(config, key, *, dtype=np.float32, shape=None):
    arr = np.asarray(config[key], dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def require_length(name, arr, length):
    if len(arr) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(arr)}.")


TASK_ALIASES = {
    "pushbox": "pushbox-in",
}


TASK_XML_PATHS = {
    "carrybox": "g1_description/omnicontact_carry_box.xml",
    "pushbox": "g1_description/omnicontact_push_box.xml",
    "pushbox-two": "g1_description/omnicontact_push_box.xml",
    "pushbox-in": "g1_description/omnicontact_push_box.xml",
    "slidebox": "g1_description/omnicontact_slide_box.xml",
    "slidebox-left": "g1_description/omnicontact_slide_box.xml",
    "slidebox-right": "g1_description/omnicontact_slide_box.xml",
    "relocateball": "g1_description/omnicontact_relocate_ball.xml",
    "kickball": "g1_description/omnicontact_kick_ball.xml",
    "kickbox": "g1_description/omnicontact_kick_ball.xml",
    "push-carry": "g1_description/omnicontact_pushcarry_box.xml",
    "carry-push": "g1_description/omnicontact_pushcarry_box.xml",
    "push-relocate": "g1_description/omnicontact_pushrelocate_ball.xml",
    "carry-carry": "g1_description/omnicontact_stack_2box.xml",
    "carry-carry-carry": "g1_description/omnicontact_stack_3box.xml",
    "carryheart": "g1_description/omnicontact_heart_10box.xml",
}

INIT_Z_AUTO_TASKS = {
    "pushbox-two",
    "pushbox-in",
    "slidebox",
    "slidebox-left",
    "slidebox-right",
    "relocateball",
    "kickball",
    "kickbox",
    "push-carry",
    "carry-push",
    "push-relocate",
}

GOAL_Z_AUTO_TASKS = {
    "pushbox-two",
    "pushbox-in",
    "slidebox",
    "slidebox-left",
    "slidebox-right",
    "kickball",
    "kickbox",
    "push-carry",
    "carry-push",
    "push-relocate",
}


def resolve_xml_path(task: str, xml_path_override: str, config: dict) -> str:
    xml_path = str(xml_path_override).strip()
    if not xml_path:
        xml_path = TASK_XML_PATHS.get(task, str(config.get("xml_path", "g1_description/omnicontact_carry_box.xml")))
    path = Path(xml_path).expanduser()
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    return str(path.resolve())


def geom_half_dims(model: mujoco.MjModel, *geom_names: str, default: tuple[float, float, float]) -> np.ndarray:
    for geom_name in geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            size = np.asarray(model.geom_size[geom_id, :3], dtype=np.float32).copy()
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                radius = float(size[0])
                return np.array([radius, radius, radius], dtype=np.float32)
            return size
    return np.asarray(default, dtype=np.float32).reshape(3).copy()


def infer_policy_dims_from_model(model: mujoco.MjModel, override_dims: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if override_dims is not None:
        dims = np.asarray(override_dims, dtype=np.float32).reshape(3)
        stack_dims = np.asarray(
            [[0.20, 0.20, 0.15], [0.15, 0.15, 0.15], [0.10, 0.10, 0.10]],
            dtype=np.float32,
        )
        return {
            "box_dims": dims.copy(),
            "ball_dims": dims.copy(),
            "push_box_dims": dims.copy(),
            "carry_box_dims": dims.copy(),
            "stack_box_dims": stack_dims,
        }

    stack_dims = []
    for geom_name, default in zip(
        ("stack_box_large_geom", "stack_box_mid_geom", "stack_box_small_geom"),
        ((0.20, 0.20, 0.15), (0.15, 0.15, 0.15), (0.10, 0.10, 0.10)),
    ):
        stack_dims.append(geom_half_dims(model, geom_name, default=default))

    return {
        "box_dims": geom_half_dims(
            model,
            "ghost_box_geom",
            "box_geom",
            "ball_geom",
            default=(0.15, 0.15, 0.15),
        ),
        "push_box_dims": geom_half_dims(
            model,
            "ghost_push_box_geom",
            "ghost_box_geom",
            "push_box_geom_top",
            "box_geom_top",
            default=(0.23, 0.25, 0.26),
        ),
        "ball_dims": geom_half_dims(
            model,
            "ghost_ball_geom",
            "ball_geom",
            default=(0.10, 0.10, 0.10),
        ),
        "carry_box_dims": geom_half_dims(
            model,
            "ghost_carry_box_geom",
            "carry_box_geom",
            default=(0.15, 0.15, 0.15),
        ),
        "stack_box_dims": np.asarray(stack_dims, dtype=np.float32).reshape(3, 3),
    }


def select_active_box_dims(task: str, dims_by_profile: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    if task in {"push-carry", "push-relocate"}:
        return "push_box", np.asarray(dims_by_profile["push_box_dims"], dtype=np.float32).copy()
    if task == "carry-push":
        return "carry_box", np.asarray(dims_by_profile["carry_box_dims"], dtype=np.float32).copy()
    if task in {"relocateball", "kickball", "kickbox"}:
        return "ball", np.asarray(dims_by_profile["ball_dims"], dtype=np.float32).copy()
    return "box", np.asarray(dims_by_profile["box_dims"], dtype=np.float32).copy()


def normalize_object_pos_for_task(pos: np.ndarray, task: str, half_z: float, *, is_goal: bool) -> np.ndarray:
    out = np.asarray(pos, dtype=np.float32).reshape(3).copy()
    auto_tasks = GOAL_Z_AUTO_TASKS if is_goal else INIT_Z_AUTO_TASKS
    if task in auto_tasks:
        out[2] = float(half_z)
    return out


def parse_object_pos_arg(value, default, half_z: float, task: str, *, is_goal: bool) -> np.ndarray:
    pos = np.asarray(value if value is not None else default, dtype=np.float32).reshape(-1)
    if len(pos) == 2:
        pos = np.array([pos[0], pos[1], half_z], dtype=np.float32)
    elif len(pos) == 3:
        pos = pos.astype(np.float32).copy()
    else:
        raise ValueError("--init-pos/--goal-pos must provide either X Y or X Y Z.")
    return normalize_object_pos_for_task(pos, task, half_z, is_goal=is_goal)


@dataclass
class ReplanState:
    identity_quat: np.ndarray
    has_held_object: bool = False
    detach_counter: int = 0
    cooldown_counter: int = 0
    replan_counter: int = 0
    waiting_for_static: bool = False
    static_counter: int = 0
    obj_quat_offset: np.ndarray = field(init=False)

    def __post_init__(self):
        self.obj_quat_offset = self.identity_quat.copy()

    def reset_detection(self, *, reset_held: bool):
        if reset_held:
            self.has_held_object = False
        self.detach_counter = 0
        self.waiting_for_static = False
        self.static_counter = 0

    def mark_held(self):
        self.has_held_object = True
        self.reset_detection(reset_held=False)

    def reset_session(self):
        self.reset_detection(reset_held=True)
        self.cooldown_counter = 0
        self.obj_quat_offset = self.identity_quat.copy()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config = config or {}
    task_choices = config.get("task_choices", ["carrybox", "pushbox-two", "pushbox-in", "relocateball", "kickball", "kickbox"])
    task_choices = list(task_choices)
    for alias in TASK_ALIASES:
        if alias not in task_choices:
            task_choices.append(alias)
    lab2mj = config_array(config, "lab2mj", dtype=np.int32)
    torque_limit_lab = config_array(config, "torque_limit_lab")
    default_joint_pos_mj = config_array(config, "default_joint_pos_mj")
    identity_quat = config_array(config, "identity_quat", shape=4)
    reference_yellow = config_array(config, "reference_no_contact_rgba", shape=4)
    reference_red = config_array(config, "reference_contact_rgba", shape=4)

    parser = argparse.ArgumentParser(description="Interactive Mujoco deploy script for carrybox and pushbox skills.")
    parser.add_argument(
        "--init-pos",
        type=float,
        nargs="+",
        default=config.get("init_pos", (1.0, 0.0, 0.55)),
        metavar="POS",
        help="Initial object position used by CFgen reset.",
    )
    parser.add_argument(
        "--box-half-dims",
        type=float,
        nargs=3,
        default=None,
        metavar=("HX", "HY", "HZ"),
        help="Optional override for object half dimensions. If omitted, dimensions are inferred from the selected XML.",
    )
    parser.add_argument(
        "--goal-pos",
        type=float,
        nargs="+",
        default=config.get("goal_pos", (3.0, 0.0, 0.26)),
        metavar="POS",
        help="Goal object position used by CFgen reset and table visualization.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=config.get("task", "carrybox"),
        choices=task_choices,
        help="Task preset used by WTAC CFgen planner.",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default="",
        help="Override Mujoco XML path. If omitted, the XML is selected from --task.",
    )
    parser.add_argument(
        "--replan",
        dest="replan",
        action="store_true",
        default=not bool(config.get("disable_replan", False)),
        help="Enable drop-triggered closed-loop replan for WTAC carrybox CFgen. Enabled by default.",
    )
    parser.add_argument(
        "--disable-replan",
        dest="replan",
        action="store_false",
        help="Disable drop-triggered closed-loop replan for WTAC carrybox CFgen.",
    )
    args = parser.parse_args()
    args.task = TASK_ALIASES.get(args.task, args.task)
    xml_path = resolve_xml_path(args.task, args.xml_path, config)
    print(f"[deploy] task={args.task} xml_path={xml_path}")

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    dims_by_profile = infer_policy_dims_from_model(
        m,
        None if args.box_half_dims is None else np.asarray(args.box_half_dims, dtype=np.float32).reshape(3),
    )
    active_object_name, box_half_dims = select_active_box_dims(args.task, dims_by_profile)
    init_pos = parse_object_pos_arg(
        args.init_pos,
        config.get("init_pos", (1.0, 0.0, 0.55)),
        float(box_half_dims[2]),
        args.task,
        is_goal=False,
    )
    goal_pos = parse_object_pos_arg(
        args.goal_pos,
        config.get("goal_pos", (3.0, 0.0, 0.26)),
        float(box_half_dims[2]),
        args.task,
        is_goal=True,
    )

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    def mj_id(obj_type, name: str) -> int:
        return mujoco.mj_name2id(m, obj_type, name)

    def body_id(name: str) -> int:
        return mj_id(mujoco.mjtObj.mjOBJ_BODY, name)

    def geom_id(name: str) -> int:
        return mj_id(mujoco.mjtObj.mjOBJ_GEOM, name)

    def joint_id(name: str) -> int:
        return mj_id(mujoco.mjtObj.mjOBJ_JOINT, name)

    def mocap_id(name: str) -> int:
        bid = body_id(name)
        return int(m.body_mocapid[bid]) if bid >= 0 else -1

    # `ref_*` mocap bodies drive wrist, torso, ankle, and table visualizations.
    ref_mocap_ids = {
        "l_wrist": mocap_id("ref_l_wrist_frame"),
        "r_wrist": mocap_id("ref_r_wrist_frame"),
        "torso": mocap_id("ref_torso_frame"),
        "l_ankle": mocap_id("ref_l_ankle_frame"),
        "r_ankle": mocap_id("ref_r_ankle_frame"),
        "plane_1": mocap_id("plane_1_holder"),
        "plane_2": mocap_id("plane_2_holder"),
    }
    ref_contact_geom_ids = [
        geom_id("ref_l_ankle_mesh"),
        geom_id("ref_r_ankle_mesh"),
        geom_id("ref_l_rubber_hand"),
        geom_id("ref_r_rubber_hand"),
    ]
    print('Successfully load wrist/torso/anklereference visualization!', ref_mocap_ids["l_wrist"], ref_mocap_ids["r_wrist"])
    m.opt.timestep = simulation_dt
    num_joints = m.nu
    require_length("lab2mj", lab2mj, num_joints)
    require_length("torque_limit_lab", torque_limit_lab, num_joints)
    require_length("default_joint_pos_mj", default_joint_pos_mj, num_joints)
    if np.any(lab2mj < 0) or np.any(lab2mj >= num_joints):
        raise ValueError(f"lab2mj indices must be in [0, {num_joints - 1}].")
    torque_limit_mj = torque_limit_lab[lab2mj]
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    wrist_state = np.zeros(14, dtype=np.float32)
    contact_state = np.zeros(4, dtype=np.float32)
    torso_goal = np.zeros(7, dtype=np.float32)
    l_ankle_goal = np.zeros(7, dtype=np.float32)
    r_ankle_goal = np.zeros(7, dtype=np.float32)
    sim_counter = 0
    
    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)
    contactflow_policy = FSM_controller.omnicontact
    contactflow_policy.task = args.task
    contactflow_policy.active_object_name = active_object_name
    contactflow_policy.ball_dims = np.asarray(dims_by_profile["ball_dims"], dtype=np.float32).copy()
    contactflow_policy.push_box_dims = np.asarray(dims_by_profile["push_box_dims"], dtype=np.float32).copy()
    contactflow_policy.carry_box_dims = np.asarray(dims_by_profile["carry_box_dims"], dtype=np.float32).copy()
    contactflow_policy.stack_box_dims = np.asarray(dims_by_profile["stack_box_dims"], dtype=np.float32).copy()
    contactflow_policy.box_dims = box_half_dims.copy()
    contactflow_policy.goal_pos_override = goal_pos.copy()
    contactflow_policy.bbox_scale = contactflow_policy.box_dims * 2.0
    contactflow_policy.bbox_offsets_scaled = contactflow_policy.bbox_offsets * contactflow_policy.bbox_scale.reshape(1, 3)
    contactflow_policy.replan_active = False
    print(f"[deploy] active_object={active_object_name} half_dims={box_half_dims.tolist()}")

    ghost_robot_joint_id = joint_id("ghost_floating_base_joint")
    ghost_robot_qpos_adr = int(m.jnt_qposadr[ghost_robot_joint_id]) if ghost_robot_joint_id >= 0 else -1
    ghost_robot_qvel_adr = int(m.jnt_dofadr[ghost_robot_joint_id]) if ghost_robot_joint_id >= 0 else -1

    def ghost_joint_qpos_adr(name: str) -> int:
        jid = joint_id(f"ghost_{name}")
        return int(m.jnt_qposadr[jid]) if jid >= 0 else -1

    ghost_robot_joint_qpos_adrs = np.array(
        [ghost_joint_qpos_adr(name) for name in contactflow_policy.kinematics.joint_names],
        dtype=np.int32,
    )
    
    joystick = JoyStick()
    Running = True

    box_body_id = body_id("box")
    box_geom_id = geom_id("box_geom")
    box_joint_id = joint_id("box")
    box_qvel_adr = int(m.jnt_dofadr[box_joint_id]) if box_joint_id >= 0 else -1
    left_palm_body_id = body_id("left_palm_link")
    right_palm_body_id = body_id("right_palm_link")
    left_wrist_yaw_body_id = body_id("left_wrist_yaw_link")
    right_wrist_yaw_body_id = body_id("right_wrist_yaw_link")

    def body_geom_ids(body_id: int) -> list[int]:
        if body_id < 0:
            return []
        geom_num = int(m.body_geomnum[body_id])
        geom_adr = int(m.body_geomadr[body_id])
        return [geom_adr + i for i in range(geom_num)]

    hand_collision_geom_set = set(body_geom_ids(left_wrist_yaw_body_id) + body_geom_ids(right_wrist_yaw_body_id))

    replan_config = config.get("closed_loop_replan", {})
    REPLAN_DETACH_DIST_THRESHOLD = float(replan_config.get("detach_dist_threshold", 0.32))
    REPLAN_DETACH_TRIGGER_TICKS = int(replan_config.get("detach_trigger_ticks", 8))
    REPLAN_COOLDOWN_TICKS = int(replan_config.get("cooldown_ticks", 80))
    REPLAN_GOAL_TOLERANCE = float(replan_config.get("goal_tolerance", 0.2))
    REPLAN_OBJ_SPEED_THRESHOLD = float(replan_config.get("obj_speed_threshold", 0.05))
    REPLAN_OBJ_ANG_SPEED_THRESHOLD = float(replan_config.get("obj_ang_speed_threshold", 0.15))
    REPLAN_STATIC_TICKS = int(replan_config.get("static_ticks", 5))

    replan_state = ReplanState(identity_quat=identity_quat)

    def reset_replan_session():
        replan_state.reset_session()
        contactflow_policy.replan_active = False

    def wtac_carrybox_replan_enabled() -> bool:
        return (
            FSM_controller.cur_policy is contactflow_policy
            and contactflow_policy.reference_source == "CFgen"
            and contactflow_policy.task != "kickball"
        )

    def object_contact_with_hand_geoms() -> bool:
        if box_geom_id < 0 or not hand_collision_geom_set:
            return False
        for i in range(int(d.ncon)):
            contact = d.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == box_geom_id and geom2 in hand_collision_geom_set:
                return True
            if geom2 == box_geom_id and geom1 in hand_collision_geom_set:
                return True
        return False

    def object_palm_min_dist() -> float:
        if box_body_id < 0:
            return np.inf
        obj_pos = d.xpos[box_body_id]
        dists = []
        if left_palm_body_id >= 0:
            dists.append(float(np.linalg.norm(obj_pos - d.xpos[left_palm_body_id])))
        if right_palm_body_id >= 0:
            dists.append(float(np.linalg.norm(obj_pos - d.xpos[right_palm_body_id])))
        if not dists:
            return np.inf
        return min(dists)

    def is_object_held() -> tuple[bool, float, bool]:
        min_dist = object_palm_min_dist()
        in_contact = object_contact_with_hand_geoms()
        return in_contact or (min_dist <= REPLAN_DETACH_DIST_THRESHOLD), min_dist, in_contact

    def current_goal_pos() -> np.ndarray:
        if hasattr(contactflow_policy, "goal_pos"):
            return np.asarray(contactflow_policy.goal_pos, dtype=np.float32).reshape(3).copy()
        return goal_pos.copy()

    def is_object_near_goal() -> bool:
        if box_body_id < 0:
            return False
        return float(np.linalg.norm(d.xpos[box_body_id] - current_goal_pos())) <= REPLAN_GOAL_TOLERANCE

    def current_obj_linear_speed() -> float:
        if box_qvel_adr < 0 or box_qvel_adr + 3 > d.qvel.shape[0]:
            return 0.0
        return float(np.linalg.norm(d.qvel[box_qvel_adr : box_qvel_adr + 3]))

    def current_obj_angular_speed() -> float:
        start = box_qvel_adr + 3
        if box_qvel_adr < 0 or start + 3 > d.qvel.shape[0]:
            return 0.0
        return float(np.linalg.norm(d.qvel[start : start + 3]))

    def compute_replan_object_quat_offset() -> np.ndarray:
        if box_body_id < 0:
            return identity_quat.copy()
        raw_obj_quat = d.xquat[box_body_id].copy().astype(np.float32)
        target_obj_quat = yaw_quat(raw_obj_quat).astype(np.float32)
        quat_offset = quat_mul(target_obj_quat, quat_conjugate(raw_obj_quat)).astype(np.float32)
        quat_offset /= max(float(np.linalg.norm(quat_offset)), 1e-8)
        return quat_offset

    def should_apply_replan_object_quat_offset() -> bool:
        return wtac_carrybox_replan_enabled() and bool(getattr(contactflow_policy, "replan_active", False))

    def sync_object_state():
        if box_body_id < 0:
            return
        state_cmd.obj_pos = d.xpos[box_body_id].copy()
        state_cmd.obj_quat = d.xquat[box_body_id].copy()
        if should_apply_replan_object_quat_offset():
            # Re-anchor a dropped box to a yaw-only baseline while keeping later relative in-hand tilt cues.
            state_cmd.obj_quat = quat_mul(
                replan_state.obj_quat_offset,
                state_cmd.obj_quat.astype(np.float32),
            ).astype(np.float32)
            state_cmd.obj_quat /= max(float(np.linalg.norm(state_cmd.obj_quat)), 1e-8)

    def replan_from_current_state(min_dist: float, in_contact: bool, obj_lin_speed: float, obj_ang_speed: float):
        if not wtac_carrybox_replan_enabled():
            return
        replan_state.obj_quat_offset = compute_replan_object_quat_offset()
        contactflow_policy.replan_active = True
        sync_object_state()
        contactflow_policy.enter()
        sync_object_state()
        replan_state.replan_counter += 1
        replan_state.reset_detection(reset_held=True)
        replan_state.cooldown_counter = REPLAN_COOLDOWN_TICKS
        policy_output.switch_to_loco = False
        policy_output.success = ""
        print(
            f"[closed_loop] replan#{replan_state.replan_counter} | reason=drop_box_wait_static, "
            f"min_palm_dist={min_dist:.3f}, hand_contact={int(in_contact)}, "
            f"obj_lin_speed={obj_lin_speed:.3f} m/s, obj_ang_speed={obj_ang_speed:.3f} rad/s"
        )

    def maybe_closed_loop_replan():
        if not wtac_carrybox_replan_enabled():
            contactflow_policy.replan_active = False
            return

        held, min_dist, in_contact = is_object_held()
        near_goal = is_object_near_goal()

        if held:
            replan_state.mark_held()
            return

        if replan_state.cooldown_counter > 0:
            replan_state.cooldown_counter -= 1
            return

        if (not replan_state.has_held_object) or near_goal:
            replan_state.reset_detection(reset_held=False)
            return

        obj_lin_speed = current_obj_linear_speed()
        obj_ang_speed = current_obj_angular_speed()

        if replan_state.waiting_for_static:
            if obj_lin_speed <= REPLAN_OBJ_SPEED_THRESHOLD and obj_ang_speed <= REPLAN_OBJ_ANG_SPEED_THRESHOLD:
                replan_state.static_counter += 1
            else:
                replan_state.static_counter = 0

            if replan_state.static_counter >= REPLAN_STATIC_TICKS:
                replan_from_current_state(min_dist, in_contact, obj_lin_speed, obj_ang_speed)
            return

        replan_state.detach_counter += 1
        if replan_state.detach_counter >= REPLAN_DETACH_TRIGGER_TICKS:
            replan_state.waiting_for_static = True
            replan_state.static_counter = 0
            print(
                f"[closed_loop] drop detected, waiting for object to settle | "
                f"lin={obj_lin_speed:.3f}/{REPLAN_OBJ_SPEED_THRESHOLD:.3f} m/s, "
                f"ang={obj_ang_speed:.3f}/{REPLAN_OBJ_ANG_SPEED_THRESHOLD:.3f} rad/s"
            )

    def reset_robot_pose():
        d.qpos[7 : 7 + num_joints] = default_joint_pos_mj
        d.qvel[6 : 6 + num_joints] = 0.0
        d.qvel[0:6] = 0.0

    def set_object_pose(pos, quat=None):
        if quat is None:
            quat = identity_quat
        box_qpos_adr = int(m.jnt_qposadr[box_joint_id])
        d.qpos[box_qpos_adr : box_qpos_adr + 3] = pos
        d.qpos[box_qpos_adr + 3 : box_qpos_adr + 7] = quat
        if box_qvel_adr >= 0:
            d.qvel[box_qvel_adr : box_qvel_adr + 6] = 0.0

    def set_table_positions(table_1_pos, table_2_pos):
        d.mocap_pos[ref_mocap_ids["plane_1"]] = table_1_pos
        d.mocap_pos[ref_mocap_ids["plane_2"]] = table_2_pos

    def reset_env_ref(op, oq, tp_1, tp_2):
        reset_robot_pose()
        set_object_pose(op, oq)
        set_table_positions(tp_1, tp_2)
        mujoco.mj_step(m, d)
    
    def reset_env():
        reset_replan_session()
        set_object_pose(init_pos)
        table_z_offset = float(contactflow_policy.box_dims[2]) + 0.005
        table_offset = np.array([0.0, 0.0, table_z_offset], dtype=np.float32)
        set_table_positions(init_pos - table_offset, goal_pos - table_offset)
        mujoco.mj_step(m, d)

    def request_skill(command, reset_fn=None) -> bool:
        state_cmd.skill_cmd = command
        if reset_fn is not None:
            reset_fn()
        return True

    def handle_joystick() -> tuple[bool, bool]:
        if joystick.is_button_pressed(JoystickButton.SELECT):
            return False, False

        reset_counter = False
        joystick.update()
        if joystick.is_button_released(JoystickButton.L3):
            state_cmd.skill_cmd = FSMCommand.PASSIVE
        if joystick.is_button_released(JoystickButton.START):
            state_cmd.skill_cmd = FSMCommand.POS_RESET
        if joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.R1):
            state_cmd.skill_cmd = FSMCommand.LOCO
        if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
            state_cmd.skill_cmd = FSMCommand.SKILL_3
        if joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.L1):
            reset_replan_session()
            reset_fn_by_source = {
                "CFgen": reset_env,
            }
            reset_counter = request_skill(
                FSMCommand.SKILL_OmniContact,
                reset_fn_by_source.get(contactflow_policy.reference_source),
            )

        return True, reset_counter

    def sync_robot_state():
        quat = d.qpos[3:7]
        state_cmd.q = d.qpos[7 : 7 + num_joints].copy()
        state_cmd.dq = d.qvel[6 : 6 + num_joints].copy()
        state_cmd.gravity_ori = get_gravity_orientation(quat).copy()
        state_cmd.base_pos = d.qpos[:3].copy()
        state_cmd.base_quat = quat.copy()
        state_cmd.ang_vel = d.qvel[3:6].copy()
        state_cmd.lin_vel = d.qvel[0:3].copy()
        sync_object_state()

    def set_mocap_pose(mocap_name: str, pose):
        d.mocap_pos[ref_mocap_ids[mocap_name]] = pose[0:3]
        d.mocap_quat[ref_mocap_ids[mocap_name]] = pose[3:7]

    def set_freejoint_pose(qpos_adr: int, qvel_adr: int, pose7):
        if qpos_adr < 0:
            return
        pose7 = np.asarray(pose7, dtype=np.float32).reshape(7)
        d.qpos[qpos_adr : qpos_adr + 7] = pose7
        if qvel_adr >= 0:
            d.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def update_ghost_robot_visualization():
        if ghost_robot_qpos_adr < 0:
            return

        ref_joint_pos = getattr(contactflow_policy, "ref_joint_pos", None)
        if (
            ref_joint_pos is not None
            and hasattr(contactflow_policy, "ref_base_pos")
            and hasattr(contactflow_policy, "ref_base_quat")
        ):
            dof_pos = np.asarray(ref_joint_pos, dtype=np.float32)
            if dof_pos.ndim != 2 or len(dof_pos) == 0:
                return
            lab2mj_policy = getattr(contactflow_policy, "lab2mj", None)
            if lab2mj_policy is not None and dof_pos.shape[1] == len(lab2mj_policy):
                dof_pos = dof_pos[:, lab2mj_policy]
            curr_idx = min(contactflow_policy.counter_step, len(dof_pos) - 1)
            ghost_base_pose = np.concatenate(
                [
                    contactflow_policy.ref_base_pos[curr_idx],
                    contactflow_policy.ref_base_quat[curr_idx],
                ],
                axis=0,
            )
            ghost_q = dof_pos[curr_idx]
        else:
            ghost_base_pose = np.concatenate([state_cmd.base_pos, state_cmd.base_quat], axis=0)
            ghost_q = state_cmd.q

        set_freejoint_pose(ghost_robot_qpos_adr, ghost_robot_qvel_adr, ghost_base_pose)
        ghost_q = np.asarray(ghost_q, dtype=np.float32).reshape(-1)
        count = min(len(ghost_robot_joint_qpos_adrs), ghost_q.shape[0])
        if count <= 0:
            return
        valid = ghost_robot_joint_qpos_adrs[:count] >= 0
        d.qpos[ghost_robot_joint_qpos_adrs[:count][valid]] = ghost_q[:count][valid]

    def update_reference_visualization():
        set_mocap_pose("l_wrist", wrist_state[0:7])
        set_mocap_pose("r_wrist", wrist_state[7:14])
        set_mocap_pose("torso", torso_goal)
        set_mocap_pose("l_ankle", l_ankle_goal)
        set_mocap_pose("r_ankle", r_ankle_goal)
        update_ghost_robot_visualization()

        for geom, contact in zip(ref_contact_geom_ids, contact_state):
            m.geom_rgba[geom] = reference_red if contact >= 0.5 else reference_yellow
        

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and Running:
            try:
                Running, should_reset_counter = handle_joystick()
                if should_reset_counter:
                    sim_counter = 0

                step_start = time.time()
                if sim_counter % control_decimation == 0:
                    sync_robot_state()
                    FSM_controller.run()
                    maybe_closed_loop_replan()

                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()
                    wrist_state = policy_output.wrist_goal.copy()
                    contact_state = policy_output.contact.copy()
                    torso_goal = policy_output.torso_goal.copy()
                    l_ankle_goal = policy_output.l_ankle_goal.copy()
                    r_ankle_goal = policy_output.r_ankle_goal.copy()
                    
                robot_qpos = d.qpos[7 : 7 + num_joints]
                robot_qvel = d.qvel[6 : 6 + num_joints]
                tau = pd_control(
                    policy_output_action,
                    robot_qpos,
                    kps,
                    np.zeros_like(kps),
                    robot_qvel,
                    kds,
                    torque_limit_mj=torque_limit_mj,
                )
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                
                update_reference_visualization()

                sim_counter += 1
            except ValueError as e:
                print(f"ValueError detected: {str(e)}")
                print(f"Debug Info: num_joints={num_joints}, qpos.shape={d.qpos.shape}, qvel.shape={d.qvel.shape}")
                break
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
