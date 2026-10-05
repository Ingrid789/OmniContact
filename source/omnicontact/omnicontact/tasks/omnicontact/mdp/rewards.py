from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_object_rotation_reward_mask(command: MotionCommand) -> torch.Tensor:
    """Return a mask for envs whose object rotation reward should be kept."""
    asset_labels = getattr(command, "env_asset_labels", None)
    if asset_labels is None:
        return torch.ones(command.object_quat_w.shape[0], dtype=torch.bool, device=command.object_quat_w.device)

    keep_reward = [not str(label).lower().startswith(("ball", "soccer")) for label in asset_labels]
    return torch.tensor(keep_reward, dtype=torch.bool, device=command.object_quat_w.device)


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)

def motion_global_root_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    robot_lin_vel = quat_apply(quat_inv(command.robot_anchor_quat_w), command.robot_anchor_lin_vel_w)
    ref_lin_vel = quat_apply(quat_inv(command.anchor_quat_w), command.anchor_lin_vel_w)

    error = torch.sum(torch.square(ref_lin_vel - robot_lin_vel), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_root_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    robot_ang_vel = quat_apply(quat_inv(command.robot_anchor_quat_w), command.robot_anchor_ang_vel_w)
    ref_ang_vel = quat_apply(quat_inv(command.anchor_quat_w), command.anchor_ang_vel_w)

    error = torch.sum(torch.square(ref_ang_vel - robot_ang_vel), dim=-1)
    return torch.exp(-error / std**2)

def motion_local_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_local_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]) ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_global_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]) ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_com_offset(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward for keeping COM close to the lower foot in xy plane, with foot-height condition."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    left_foot_index = _get_body_indexes(command, ["left_ankle_roll_link"])
    right_foot_index = _get_body_indexes(command, ["right_ankle_roll_link"])

    left_foot_pos = command.robot_body_pos_w[:, left_foot_index]
    right_foot_pos = command.robot_body_pos_w[:, right_foot_index]

    foot_height_diff = torch.abs(left_foot_pos[:, :, 2] - right_foot_pos[:, :, 2])
    lower_foot_pos = torch.where(left_foot_pos[:, :, 2] < right_foot_pos[:, :, 2], left_foot_pos, right_foot_pos)

    com_xy = command.robot.data.root_com_state_w[:, :2]
    lower_foot_xy = lower_foot_pos[:, :, :2]
    error = torch.sum(torch.square(com_xy[:, None, :] - lower_foot_xy), dim=-1)
    error = torch.mean(error, dim=-1)  
    exp_term = torch.exp(-error / (std**2)) 


    height_mask = (foot_height_diff**2 > 0.05).float()

    reward = exp_term * (height_mask.mean(dim=-1))

    return reward
    
def foot_slip_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    is_contact = torch.linalg.norm(contact_forces, dim=-1) > threshold

    foot_velocities = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    foot_planar_velocity = torch.linalg.norm(foot_velocities, dim=-1)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def motion_global_object_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.object_pos_w - command.data_object_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_object_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.object_quat_w, command.data_object_quat_w) ** 2
    reward = torch.exp(-error / std**2)
    reward_mask = _get_object_rotation_reward_mask(command)
    return torch.where(reward_mask, reward, torch.zeros_like(reward))


def motion_local_interact_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    rel_pos = command.robot_ee_pos_w[:, :2] - command.object_pos_w.unsqueeze(1)
    data_rel_pos = command.data_ee_pos_w[:, :2] - command.data_object_pos_w.unsqueeze(1)

    delta_ori_w = yaw_quat(quat_mul(quat_inv(command.anchor_quat_w), command.robot_anchor_quat_w))
    rel_pos_local = quat_apply(delta_ori_w.unsqueeze(1).expand(-1, 2, -1), rel_pos)
    error = torch.sum(torch.square(rel_pos_local - data_rel_pos), dim=-1)
    
    data_rel_pos_l2 = torch.linalg.norm(data_rel_pos, dim=-1) # wrist_pos
    scale = torch.exp(-2 * data_rel_pos_l2)
    scaled_std = std / (scale + 1e-6)

    return torch.mean(torch.exp(-error / scaled_std**2), dim=-1)


def motion_local_interact_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
 
    rel_ori = quat_error_magnitude(command.object_quat_w.unsqueeze(1).expand(-1, 2, -1), command.robot_ee_quat_w[:, :2]) # wrist_quat
    data_rel_ori = quat_error_magnitude(command.data_object_quat_w.unsqueeze(1).expand(-1, 2, -1), command.data_ee_quat_w[:, :2])
    error = torch.square(rel_ori - data_rel_ori)

    data_rel_pos = command.data_ee_pos_w[:, :2] - command.data_object_pos_w.unsqueeze(1)
    data_rel_pos_l2 = torch.linalg.norm(data_rel_pos, dim=-1) # wrist_pos
    scale = torch.exp(-2 * data_rel_pos_l2)
    scaled_std = std / (scale + 1e-6)

    return torch.mean(torch.exp(-error / scaled_std**2), dim=-1)

def contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    hand_threshold: float = 1.0,
    foot_threshold: float = 0.0,
    hand_sensor_names: tuple[str, str] = (
        "left_wrist_contact_forces_object",
        "right_wrist_contact_forces_object",
    ),
    std: float = 1.0,
    contact_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    measured_contact = command.robot_object_contact_info(
        hand_threshold=hand_threshold,
        foot_threshold=foot_threshold,
        hand_sensor_names=hand_sensor_names,
    )

    ref_contact = command.contact_info
    ref_contact = ref_contact.reshape(ref_contact.shape[0], -1) # (N, 4)

    if contact_indices is not None:
        measured_contact = measured_contact[:, contact_indices]
        ref_contact = ref_contact[:, contact_indices]

    mse = torch.mean(torch.square(measured_contact - ref_contact), dim=-1)
    return torch.exp(-mse / (std**2))
