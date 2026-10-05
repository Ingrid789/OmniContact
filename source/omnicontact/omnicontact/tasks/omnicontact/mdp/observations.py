from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, yaw_quat, quat_conjugate

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_object_orientation_observation_mask(command: MotionCommand) -> torch.Tensor:
    """Return a mask for envs whose object orientation observations should be kept."""
    asset_labels = getattr(command, "env_asset_labels", None)
    if asset_labels is None:
        return torch.ones(command.object_quat_w.shape[0], dtype=torch.bool, device=command.object_quat_w.device)

    keep_observation = [not str(label).lower().startswith(("ball", "soccer")) for label in asset_labels]
    return torch.tensor(keep_observation, dtype=torch.bool, device=command.object_quat_w.device)


def _identity_quat_like(quat: torch.Tensor) -> torch.Tensor:
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return identity


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_ee_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.ee_body_names)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        command.robot_ee_pos_w,
        command.robot_ee_quat_w,
    )
    return pos.view(env.num_envs, -1)


def base_height(env: ManagerBasedEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, 2:3]


def object_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_heading = yaw_quat(command.robot_anchor_quat_w)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        robot_heading,
        command.object_pos_w,
        command.object_quat_w,
    )
    norm = pos.norm(dim=-1, keepdim=True)
    pos = torch.where(norm > 4, (pos / norm) * 4, pos) # Clip to max 4 meters
    return pos.view(env.num_envs, -1)

def object_pos_amp_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    threshold = 1.0
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_heading = yaw_quat(command.robot_anchor_quat_w)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        robot_heading,
        command.object_pos_w,
        command.object_quat_w,
    )

    xy = pos[:, :2]
    xy_norm = torch.norm(xy, dim=-1, keepdim=True)
    far_mask = xy_norm > threshold

    scaled_xy = xy / xy_norm * threshold
    pos[:, :2] = torch.where(far_mask, scaled_xy, xy)
    # For far objects, remove vertical offset to keep only planar direction.
    pos[far_mask.squeeze(-1), 2] = 0
    return pos.view(env.num_envs, -1)


def robot_object_contact_info_obs(
    env: ManagerBasedEnv,
    command_name: str,
    hand_threshold: float = 1.0,
    foot_threshold: float = 0.0,
    hand_sensor_names: tuple[str, str] = (
        "left_wrist_contact_forces_object",
        "right_wrist_contact_forces_object",
    ),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_object_contact_info(
        hand_threshold=hand_threshold,
        foot_threshold=foot_threshold,
        hand_sensor_names=hand_sensor_names,
    )


def object_ori_6d_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_heading = yaw_quat(command.robot_anchor_quat_w)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        robot_heading,
        command.object_pos_w,
        command.object_quat_w,
    )
    keep_mask = _get_object_orientation_observation_mask(command)
    ori = torch.where(keep_mask.unsqueeze(-1), ori, _identity_quat_like(ori))
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def object_bbox_corners_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    bbox_scaled_offset = command.bbox_offset * command.object_dims.unsqueeze(1) 
    obj_quat_expanded = command.object_quat_w.unsqueeze(1).expand(-1, 8, -1)
    obj_pos_expanded = command.object_pos_w.unsqueeze(1)
    bbox_pos_w = quat_apply(obj_quat_expanded, bbox_scaled_offset) + obj_pos_expanded
    
    robot_heading = yaw_quat(command.robot_anchor_quat_w)
    robot_heading_inv = quat_conjugate(robot_heading).unsqueeze(1).expand(-1, 8, -1)
    bbox_pos_rel = bbox_pos_w - command.robot_anchor_pos_w.unsqueeze(1)
    bbox_pos_obs = quat_apply(robot_heading_inv, bbox_pos_rel)
    
    norm = bbox_pos_obs.norm(dim=-1, keepdim=True)
    bbox_pos_obs = torch.where(norm > 4, (bbox_pos_obs / norm) * 4, bbox_pos_obs) # Clip to max 4 meters

    return bbox_pos_obs.reshape(env.num_envs, 3 * 8)

def _compute_tracking_features(env: ManagerBasedEnv, command_name: str, body_names: list[str] = None) -> torch.Tensor:
    """
    Compute sparse tracking features (position and 6D rotation) for selected bodies and the object at future frames.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    nframes = len(command.cfg.future_frames)
    
    # --- 1. Prepare the reference frame (anchor and heading). ---
    robot_heading = yaw_quat(command.robot_anchor_quat_w) 
    robot_heading_flat = robot_heading.unsqueeze(1).expand(-1, nframes, -1).reshape(-1, 4)
    anchor_pos_flat = command.robot_anchor_pos_w.unsqueeze(1).expand(-1, nframes, -1).reshape(-1, 3)
    features = []
    # --- 2. Process robot bodies. ---
    if body_names:
        for name in body_names:
            body_idx = command.cfg.body_names.index(name)
            target_pos_flat = command.ref_robot_future_body_pos_w[:, :, body_idx].reshape(-1, 3)
            target_quat_flat = command.ref_robot_future_body_quat_w[:, :, body_idx].reshape(-1, 4)
            # World -> Local
            local_pos, local_quat = subtract_frame_transforms(
                anchor_pos_flat, robot_heading_flat, target_pos_flat, target_quat_flat
            )
            # Quat -> Matrix -> 6D
            rot_mat = matrix_from_quat(local_quat)
            rot_6d = rot_mat[..., :2].reshape(-1, 6)
            features.append(local_pos)
            features.append(rot_6d)

    sparse_tracking_target = torch.cat(features, dim=-1)
    return sparse_tracking_target.view(env.num_envs, -1)


def _future_contact_features(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.future_contact_info.reshape(env.num_envs, -1)


def tracking_commands_contact_flow(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    # Pose/orientation tracking targets (sparse future frames)
    tracking = _compute_tracking_features(
        env,
        command_name,
        body_names=[
            "left_rubber_hand",
            "right_rubber_hand",
            "torso_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ],
    )

    # Contact targets aligned with the same future frames.
    contact = _future_contact_features(env, command_name)
    return torch.cat([tracking, contact], dim=-1)
