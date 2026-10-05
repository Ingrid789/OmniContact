from __future__ import annotations

import math
import numpy as np
import os
import omni.usd
from pxr import Usd, UsdGeom
import re
import torch
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from omnicontact.utils.motion_labels import asset_label_from_motion_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int] | torch.Tensor, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.motion_file = os.path.abspath(motion_file)
        self._motion_id = motion_file.split("/")[-1].split(".")[0]
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._ee_pos_w = torch.tensor(data["ee_pos_w"], dtype=torch.float32, device=device)
        self._ee_quat_w = torch.tensor(data["ee_quat_w"], dtype=torch.float32, device=device)
        self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
        self._object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
        self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
        self._object_ang_vel_w = torch.tensor(data["object_ang_vel_w"], dtype=torch.float32, device=device)
        # Missing tables use a constant position five metres below the ground.
        for key in ("table1_pos_w", "table2_pos_w"):
            if key in data.files:
                table_pos = torch.tensor(data[key], dtype=torch.float32, device=device)
            else:
                table_pos = torch.zeros((self.joint_pos.shape[0], 3), dtype=torch.float32, device=device)
                table_pos[:, 2] = -5.0
            setattr(self, f"_{key}", table_pos)
        # Optional: contact info for (left_foot, right_foot, left_hand, right_hand)
        # Shape: (nframes, 4), values are typically {0, 1}.
        if "contact_info" in data.files:
            contact_info = torch.tensor(data["contact_info"], dtype=torch.float32, device=device)
            self._contact_info = contact_info.reshape(self.joint_pos.shape[0], -1).contiguous()
        else:
            self._contact_info = torch.zeros((self._body_pos_w.shape[0], 4), dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def motion_id(self) -> str:
        return self._motion_id

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def ee_pos_w(self) -> torch.Tensor:
        return self._ee_pos_w
    
    @property
    def ee_quat_w(self) -> torch.Tensor:
        return self._ee_quat_w
    
    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w
    
    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self._object_ang_vel_w
    
    @property
    def table1_pos_w(self) -> torch.Tensor:
        return self._table1_pos_w

    @property
    def table2_pos_w(self) -> torch.Tensor:
        return self._table2_pos_w

    @property
    def contact_info(self) -> torch.Tensor:
        return self._contact_info


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.box: RigidObject = env.scene[cfg.object_name]
        self.table1: RigidObject = env.scene[cfg.table1_name]
        self.table2: RigidObject | None = (
            env.scene[cfg.table2_name]
            if cfg.table2_name is not None
            else None
        )
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        
        # Body indexes on CPU for MotionLoader slicing
        self.body_indexes_cpu = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device="cpu"
        )
        self.body_indexes = self.body_indexes_cpu.to(self.device)
        
        self.ee_body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.ee_body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device
        )

        assert isinstance(self.cfg.motion_file_list, (list, tuple)), "motion_file must be a list of strings."
        assert len(self.cfg.motion_file_list) > 0, "No motion files provided."

        # Load motions (on CPU)
        # [OPTIMIZATION] Pass "cpu" as device to MotionLoader
        with ThreadPoolExecutor() as executor:
            self.motions: list[MotionLoader] = list(executor.map(
                lambda path: MotionLoader(path, self.body_indexes_cpu, device="cpu"),
                self.cfg.motion_file_list
            ))
        self.motion = self.motions[0]
        self.num_motions = len(self.motions)
        self._motion_uses_table1 = torch.tensor(
            [self._uses_table1(motion) for motion in self.motions],
            device=self.device,
            dtype=torch.bool,
        )
        self._table1_park_offset = torch.tensor(
            self.cfg.table1_park_offset,
            device=self.device,
            dtype=torch.float32,
        )

        self._body_attr_cache: dict[str, torch.Tensor] = {}
        self._joint_attr_cache: dict[str, torch.Tensor] = {}
        self._object_attr_cache: dict[str, torch.Tensor] = {}
        
        motion_lengths = torch.tensor(
            [motion.time_step_total for motion in self.motions],
            dtype=torch.long,
            device=self.device,
        )
        motion_offsets = torch.zeros_like(motion_lengths)
        if motion_lengths.numel() > 1:
            motion_offsets[1:] = torch.cumsum(motion_lengths[:-1], dim=0)
        self._motion_lengths = motion_lengths
        self._motion_offsets = motion_offsets

        # [OPTIMIZATION] Concatenate on CPU first, then move to GPU once.
        # This is much faster than moving thousands of small tensors to GPU and then catting.
        print("[INFO] Building Motion Cache on GPU...")
        for attr_name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "ee_pos_w", "ee_quat_w"):
            if attr_name not in self._body_attr_cache:
                cpu_cat = torch.cat([getattr(motion, attr_name) for motion in self.motions], dim=0)
                self._body_attr_cache[attr_name] = cpu_cat.to(self.device).contiguous()

        joint_attr_names = [
            "joint_pos", "joint_vel",
            "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w",
            "table1_pos_w", "table2_pos_w",
            "contact_info",
        ]
        for attr_name in joint_attr_names:
            if attr_name not in self._joint_attr_cache:
                cpu_cat = torch.cat([getattr(motion, attr_name) for motion in self.motions], dim=0)
                self._joint_attr_cache[attr_name] = cpu_cat.to(self.device).contiguous()
        
        # Assign motion indices
        self.env_motion_idx = (torch.arange(self.num_envs, device=self.device) % self.num_motions).long()

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.env_asset_labels = self.get_env_asset_names()
        self.object_dims = torch.tensor([self.parse_dims(l) for l in self.env_asset_labels], device=self.device) # "18-24-12" -> [0.18, 0.24, 0.12]
        self.bbox_offset = torch.tensor([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ], device=self.device, dtype=torch.float32) * 0.5  

        if self.cfg.sampling_method in [
            "uniform_bin",
            "adaptive_bin",
        ]:
            self._init_sampler(env, self.env_asset_labels)
        elif self.cfg.sampling_method in [
            "uniform_frame",
            "adaptive_frame",
            "fixed_frame",
        ]:
            self._init_sampler(env, self.env_asset_labels, frame_level=True)
        else:
            raise NotImplementedError(f"Unknown motion sampling method: {self.cfg.sampling_method}")
        print(f"[INFO] Using {self.cfg.sampling_method} motion sampling with {self.num_motions} motions.")

    @staticmethod
    def _uses_table1(motion: MotionLoader) -> bool:
        table1_pos = motion.table1_pos_w
        has_position = torch.any(torch.abs(table1_pos) > 1.0e-6)
        is_parked = torch.all(table1_pos[:, 2] < -5.0)
        return bool((has_position & ~is_parked).item())

    def parse_dims(self, label):
        label_str = str(label).lower()
        if label_str.startswith(("ball", "soccer")):
            nums = re.findall(r"[\d\.]+", label_str)
            diameter = float(nums[0]) / 100.0 if nums else 0.1
            return [diameter, diameter, diameter]

        nums = re.findall(r"[\d\.]+", str(label))
        if len(nums) >= 3:
            return [float(x) / 100.0 for x in nums[-3:]]
        if len(nums) == 1:
            side = float(nums[0]) / 100.0
            return [side, side, side]
        return [0.1, 0.1, 0.1]

    @staticmethod
    def normalize_semantic_label(label):
        label = str(label)
        match = re.fullmatch(r"(?P<prefix>.*?)(?P<dims>\d+_\d+_\d+)", label)
        if match is None:
            return label
        return f"{match.group('prefix')}{match.group('dims').replace('_', '-')}"

    def get_env_asset_names(self):
        stage = omni.usd.get_context().get_stage()
        env_asset_map = []

        for i in range(self.num_envs):
            # 1. Get the prim.
            current_prim_path = f"/World/envs/env_{i}/Object"
            prim = stage.GetPrimAtPath(current_prim_path)
            if not prim.IsValid():
                env_asset_map.append("unknown")
                continue

            # 2. Search all properties for a semantic label.
            found_label = None
            for prop in prim.GetProperties():
                prop_name = prop.GetName()
                if "semantic" in prop_name and "semanticData" in prop_name:
                    val = prop.Get()
                    if val:
                        found_label = self.normalize_semantic_label(val)
                        break
            
            env_asset_map.append(found_label) if found_label else env_asset_map.append("no_tag")
        return env_asset_map

    def _init_sampler(self, env, asset_labels, frame_level: bool = False):
        """
        Initialize the sampler that matches motions to assets.
        Map asset labels to motion indices.
        """
        self.asset_to_motion_indices = {}

        # 1. Map each label to a tensor of motion indices.
        unique_labels = set(self.env_asset_labels)
        motion_labels = [asset_label_from_motion_file(m.motion_file) for m in self.motions]
        print(f"[INFO] Bounding Assets Sampler: Found {len(unique_labels)} unique asset labels.")

        for label in unique_labels:
            if label in ["unknown", "no_tag", None]:
                continue
            indices = [i for i, motion_label in enumerate(motion_labels) if label == motion_label]
            if len(indices) > 0:
                self.asset_to_motion_indices[label] = torch.tensor(indices, device=self.device, dtype=torch.long)
            else:
                print(f"[WARNING] No motions found matching asset label: '{label}'. These envs will use random sampling.")
        
        # Store each environment's asset ID in a tensor.
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        asset_ids_list = [self.label_to_id[label] for label in self.env_asset_labels]
        self.env_asset_ids_tensor = torch.tensor(asset_ids_list, device=self.device, dtype=torch.long)
        # Matrix with shape [Num_Asset_Classes, Max_Motions_Per_Class].
        max_motions = max([len(indices) for indices in self.asset_to_motion_indices.values()], default=1)
        num_classes = len(unique_labels)
        self.valid_motions_table = torch.zeros((num_classes, max_motions), device=self.device, dtype=torch.long)
        self.valid_motions_counts = torch.zeros(num_classes, device=self.device, dtype=torch.long)

        for label, indices in self.asset_to_motion_indices.items():
            idx = self.label_to_id[label]
            count = len(indices)
            self.valid_motions_counts[idx] = count
            self.valid_motions_table[idx, :count] = torch.tensor(indices, device=self.device)

        # 2. Initialize sampling statistics.
        dt = env.cfg.sim.dt
        decimation = env.cfg.decimation
        step_duration = 1 / (decimation * dt)

        self.motion_total_steps = torch.tensor([m.time_step_total for m in self.motions], device=self.device)
        if frame_level:
            self.bin_count_per_motion = self.motion_total_steps.clone()
        else:
            self.bin_count_per_motion = (self.motion_total_steps // step_duration).long() + 1
        self.bin_counts = self.bin_count_per_motion
        self.max_bin_count = int(self.bin_count_per_motion.max().item())
        self.bin_mask = (
            torch.arange(self.max_bin_count, device=self.device).unsqueeze(0)
            < self.bin_count_per_motion.unsqueeze(1)
        )
        self.bin_failed_count_per_motion = torch.zeros(self.num_motions, self.max_bin_count, device=self.device)
        self._current_bin_failed_per_motion = torch.zeros_like(self.bin_failed_count_per_motion)
        self.motion_bin_offsets = torch.cat([
            torch.tensor([0], device=self.device),
            self.bin_count_per_motion.cumsum(0)[:-1],
        ])

        # 3. Initialize motion failure tracking.
        self.total_bins = int(self.bin_count_per_motion.sum().item())
        self.motion_failed_count = torch.zeros(self.num_motions, dtype=torch.float, device=self.device)
        self._current_motion_failed = torch.zeros(self.num_motions, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda ** i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()
        
        # 4. Initialize metrics.
        self.metrics.update({
            "error_anchor_pos": torch.zeros(self.num_envs, device=self.device),
            "error_anchor_rot": torch.zeros(self.num_envs, device=self.device),
            "error_anchor_lin_vel": torch.zeros(self.num_envs, device=self.device),
            "error_anchor_ang_vel": torch.zeros(self.num_envs, device=self.device),
            "error_body_pos": torch.zeros(self.num_envs, device=self.device),
            "error_body_rot": torch.zeros(self.num_envs, device=self.device),
            "error_joint_pos": torch.zeros(self.num_envs, device=self.device),
            "error_joint_vel": torch.zeros(self.num_envs, device=self.device),
            "sampling_entropy": torch.zeros(self.num_envs, device=self.device),
            "sampling_top1_prob": torch.zeros(self.num_envs, device=self.device),
            "sampling_top1_bin": torch.zeros(self.num_envs, device=self.device),
            "sampling_motion_entropy": torch.zeros(self.num_envs, device=self.device),
            "sampling_motion_top1_prob": torch.zeros(self.num_envs, device=self.device),
        })

    def _update_current_failures_tensorized(self, env_ids: torch.Tensor):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if not torch.any(episode_failed):
            return

        failed_env_ids = env_ids[episode_failed]
        failed_motion_indices = self.env_motion_idx[failed_env_ids]

        motion_fail_counts = torch.bincount(failed_motion_indices, minlength=self.num_motions)
        self._current_motion_failed[:] = motion_fail_counts.float()

        failed_motion_total_steps = self.motion_total_steps[failed_motion_indices]
        failed_bin_counts = self.bin_count_per_motion[failed_motion_indices]

        current_bin_index = torch.clamp(
            (self.time_steps[failed_env_ids] * failed_bin_counts) // torch.clamp(failed_motion_total_steps, min=1),
            min=torch.zeros_like(failed_bin_counts),
            max=failed_bin_counts - 1,
        )

        super_bin_index = self.motion_bin_offsets[failed_motion_indices] + current_bin_index
        super_bin_fail_counts = torch.bincount(super_bin_index, minlength=self.total_bins).float()

        idxs = torch.arange(self.max_bin_count, device=self.device).unsqueeze(0)
        flat_idxs = self.motion_bin_offsets.unsqueeze(1) + idxs
        valid_flat_idxs = flat_idxs[self.bin_mask]
        self._current_bin_failed_per_motion[self.bin_mask] = super_bin_fail_counts[valid_flat_idxs]

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def motion_id(self) -> torch.Tensor:
        return self.env_motion_idx

    @property
    def joint_pos(self) -> torch.Tensor:
        # ori_joint_pos = self.motion.joint_pos[self.time_steps]
        joint_pos = self._gather_joint_attr("joint_pos")
        # error = torch.sum(torch.abs(joint_pos - ori_joint_pos))
        # print("Joint pos error:", error)
        return joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        # ori_joint_vel = self.motion.joint_vel[self.time_steps]
        joint_vel = self._gather_joint_attr("joint_vel")
        # error = torch.sum(torch.abs(joint_vel - ori_joint_vel))
        # print("Joint vel error:", error)
        return joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        # ori_body_pos_w = self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
        body_pos_w = self._gather_body_attr("body_pos_w") + self._env.scene.env_origins[:, None, :]
        # error = torch.sum(torch.abs(body_pos_w - ori_body_pos_w))
        # print("Body pos w error:", error)
        return body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        # ori_body_quat_w = self.motion.body_quat_w[self.time_steps]
        body_quat_w = self._gather_body_attr("body_quat_w")
        # error = torch.sum(torch.abs(body_quat_w - ori_body_quat_w))
        # print("Body quat w error:", error)
        return body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        # ori_body_lin_vel_w = self.motion.body_lin_vel_w[self.time_steps]
        body_lin_vel_w = self._gather_body_attr("body_lin_vel_w")
        # error = torch.sum(torch.abs(body_lin_vel_w - ori_body_lin_vel_w))
        # print("Body lin_vel w error:", error)
        return body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        # ori_body_ang_vel_w = self.motion.body_ang_vel_w[self.time_steps]
        body_ang_vel_w = self._gather_body_attr("body_ang_vel_w")
        # error = torch.sum(torch.abs(body_ang_vel_w - ori_body_ang_vel_w))
        # print("Body ang w error:", error)
        return body_ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        # ori_anchor_pos_w = self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins
        anchor_pos_w = self._gather_body_attr("body_pos_w")[:, self.motion_anchor_body_index] + self._env.scene.env_origins
        # error = torch.sum(torch.abs(anchor_pos_w - ori_anchor_pos_w))
        # print("Anchor pos w error:", error)
        return anchor_pos_w
    
    @property
    def anchor_quat_w(self) -> torch.Tensor:
        # ori_anchor_quat_w = self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]
        anchor_quat_w = self.body_quat_w[:, self.motion_anchor_body_index]
        # error = torch.sum(torch.abs(anchor_quat_w - ori_anchor_quat_w))
        # print("Anchor quat w error:", error)
        return anchor_quat_w

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        # ori_anchor_lin_vel_w = self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
        anchor_lin_vel_w = self.body_lin_vel_w[:, self.motion_anchor_body_index]
        # error = torch.sum(torch.abs(anchor_lin_vel_w - ori_anchor_lin_vel_w))
        # print("Anchor linvel w error:", error)
        return anchor_lin_vel_w

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        # ori_anchor_ang_vel_w = self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
        anchor_ang_vel_w = self.body_ang_vel_w[:, self.motion_anchor_body_index]
        # error = torch.sum(torch.abs(anchor_ang_vel_w - ori_anchor_ang_vel_w))
        # print("Anchor angvel w error:", error)
        return anchor_ang_vel_w
    
    @property
    def data_ee_pos_w(self) -> torch.Tensor:
        ee_pos_w = self._gather_body_attr("ee_pos_w") + self._env.scene.env_origins[:, None, :]
        return ee_pos_w
    
    @property
    def data_ee_quat_w(self) -> torch.Tensor:
        ee_quat_w = self._gather_body_attr("ee_quat_w")
        return ee_quat_w

    @property
    def data_object_pos_w(self) -> torch.Tensor:
        object_pos_w = self._gather_joint_attr("object_pos_w") + self._env.scene.env_origins
        return object_pos_w
    
    @property
    def data_object_quat_w(self) -> torch.Tensor:
        object_quat_w = self._gather_joint_attr("object_quat_w")
        return object_quat_w
    
    @property
    def data_object_lin_vel_w(self) -> torch.Tensor:
        object_lin_vel_w = self._gather_joint_attr("object_lin_vel_w")
        return object_lin_vel_w
    
    @property
    def data_object_ang_vel_w(self) -> torch.Tensor:
        object_ang_vel_w = self._gather_joint_attr("object_ang_vel_w")
        return object_ang_vel_w
    
    @property
    def table1_pos_w(self) -> torch.Tensor:
        table1_pos_w = self._gather_joint_attr("table1_pos_w") + self._env.scene.env_origins
        return table1_pos_w

    @property
    def table2_pos_w(self) -> torch.Tensor:
        table2_pos_w = (
            self._gather_joint_attr("table2_pos_w") + self._env.scene.env_origins
        )
        return table2_pos_w

    @property
    def contact_info(self) -> torch.Tensor:
        """
        Contact info for current reference frame.
        Order is: (left_foot, right_foot, left_hand, right_hand), shape (num_envs, 4).
        """
        return self._gather_joint_attr("contact_info")

    def robot_object_contact_info(
        self,
        hand_threshold: float = 1.0,
        foot_threshold: float = 0.0,
        hand_sensor_names: tuple[str, str] = (
            "left_wrist_contact_forces_object",
            "right_wrist_contact_forces_object",
        ),
    ) -> torch.Tensor:
        """Return contacts with the tracked object as [left_foot, right_foot, left_hand, right_hand]."""
        sensor_names = (
            "left_ankle_contact_forces_object",
            "right_ankle_contact_forces_object",
            *hand_sensor_names,
        )
        thresholds = self.robot.data.root_pos_w.new_tensor(
            [foot_threshold, foot_threshold, hand_threshold, hand_threshold]
        )
        contacts = []
        for sensor_name, threshold in zip(sensor_names, thresholds):
            force_matrix = self._env.scene.sensors[sensor_name].data.force_matrix_w
            contact = (torch.linalg.norm(force_matrix, dim=-1) > threshold).flatten(start_dim=1).any(dim=1)
            contacts.append(contact)
        return torch.stack(contacts, dim=1).to(torch.float32)

    def robot_net_contact_info(self, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
        """Return unfiltered contacts as [left_foot, right_foot, left_hand, right_hand]."""
        contact_sensor: ContactSensor = self._env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

        # Exclude z-force for feet to avoid treating pure ground-normal force as object contact.
        foot_contact = torch.linalg.norm(forces[:, :2, :2], dim=-1) > threshold
        hand_contact = torch.linalg.norm(forces[:, 2:4, :], dim=-1) > threshold
        return torch.cat([foot_contact, hand_contact], dim=1).to(torch.float32)
    
    @property
    def future_anchor_pos(self) -> torch.Tensor:
        future_steps = torch.tensor(self.cfg.future_frames, device=self.device)
        joint_pos = self._gather_body_attr_future("body_pos_w", future_steps) + self._env.scene.env_origins[
            :, None, None, :]
        return joint_pos

    @property
    def future_anchor_ori(self) -> torch.Tensor:
        future_steps = torch.tensor(self.cfg.future_frames, device=self.device)
        joint_ori = self._gather_body_attr_future("body_quat_w", future_steps)
        return joint_ori

    @property
    def future_body_lin_vel(self) -> torch.Tensor:
        future_steps = torch.tensor(self.cfg.future_frames, device=self.device)
        joint_vel = self._gather_body_attr_future("body_lin_vel_w", future_steps)
        return joint_vel

    @property
    def future_body_ang_vel(self) -> torch.Tensor:
        future_steps = torch.tensor(self.cfg.future_frames, device=self.device)
        joint_vel = self._gather_body_attr_future("body_ang_vel_w", future_steps)
        return joint_vel
    
    @property
    def future_object_pos(self) -> torch.Tensor:
        future_steps = torch.tensor(self.cfg.future_frames, device=self.device)
        joint_pos = self._gather_joint_attr_future("object_pos_w", future_steps)
        return joint_pos
    
    @property
    def future_object_ori(self) -> torch.Tensor:
        future_steps = torch.tensor(self.cfg.future_frames, device=self.device)
        joint_ori = self._gather_joint_attr_future("object_quat_w", future_steps)
        return joint_ori

    @property
    def future_contact_info(self) -> torch.Tensor:
        """Contact info for future reference frames.

        Shape: (num_envs, num_future_frames, 4)
        Order: (left_foot, right_foot, left_hand, right_hand)
        """
        future_steps = torch.tensor(self.cfg.future_frames, device=self.device)
        return self._gather_joint_attr_future("contact_info", future_steps)

    def _gather_joint_attr(self, attr_name: str) -> torch.Tensor:
        attr_concat = self._joint_attr_cache.get(attr_name)
        time_steps = self.time_steps.clone()
        torch.clamp_(time_steps, min=0)
        max_steps = self._motion_lengths[self.env_motion_idx] - 1
        torch.clamp_(max_steps, min=0)
        torch.minimum(time_steps, max_steps, out=time_steps)

        global_indices = self._motion_offsets[self.env_motion_idx] + time_steps
        gathered = torch.index_select(attr_concat, 0, global_indices)
        return gathered

    def _gather_body_attr(self, attr_name: str) -> torch.Tensor:
        attr_concat = self._body_attr_cache.get(attr_name)
        time_steps = self.time_steps.clone()
        torch.clamp_(time_steps, min=0)
        max_steps = self._motion_lengths[self.env_motion_idx] - 1
        torch.clamp_(max_steps, min=0)
        torch.minimum(time_steps, max_steps, out=time_steps)

        global_indices = self._motion_offsets[self.env_motion_idx] + time_steps
        gathered = torch.index_select(attr_concat, 0, global_indices)
        return gathered
    
    def _gather_joint_attr_future(self, attr_name: str, future_steps: torch.Tensor) -> torch.Tensor:
        """Get future joint attribute for each env using GPU cache."""
        attr_concat = self._joint_attr_cache.get(attr_name)
        current_steps = self.time_steps.clone().unsqueeze(-1) 
        future_offsets = future_steps.to(self.device).unsqueeze(0)
        target_steps = current_steps + future_offsets
        max_steps = (self._motion_lengths[self.env_motion_idx] - 1).unsqueeze(-1)
        torch.clamp_(target_steps, min=0)
        torch.minimum(target_steps, max_steps, out=target_steps)

        motion_start_indices = self._motion_offsets[self.env_motion_idx].unsqueeze(-1)
        global_indices = motion_start_indices + target_steps
        
        flat_indices = global_indices.view(-1).long()
        gathered_flat = torch.index_select(attr_concat, 0, flat_indices)
        
        feat_dim = attr_concat.shape[1]
        return gathered_flat.view(self.num_envs, self.cfg.num_future_frames, feat_dim)


    def _gather_body_attr_future(self, attr_name: str, future_steps: torch.Tensor) -> torch.Tensor:
        """Get future body attribute for each env using GPU cache."""
        # --- Vectorized Path (Fast, Pure GPU) ---
        attr_concat = self._body_attr_cache.get(attr_name)
        
        # [Num_Envs, 1]
        current_steps = self.time_steps.clone().unsqueeze(-1)
        # [1, Num_Future_Frames]
        future_offsets = future_steps.to(self.device).unsqueeze(0)
        
        # [Num_Envs, Num_Future_Frames]
        target_steps = current_steps + future_offsets
        
        # Clamp
        max_steps = (self._motion_lengths[self.env_motion_idx] - 1).unsqueeze(-1)
        torch.clamp_(target_steps, min=0)
        torch.minimum(target_steps, max_steps, out=target_steps)

        # Global indices
        motion_start_indices = self._motion_offsets[self.env_motion_idx].unsqueeze(-1)
        global_indices = motion_start_indices + target_steps
        
        flat_indices = global_indices.view(-1).long()
        gathered_flat = torch.index_select(attr_concat, 0, flat_indices)

        num_bodies = attr_concat.shape[1]
        dim = attr_concat.shape[2]
        
        return gathered_flat.view(self.num_envs, self.cfg.num_future_frames, num_bodies, dim)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        quat_w = self.robot.data.body_quat_w[:, self.robot_anchor_body_index]
        rpy_bias = getattr(self._env, "torso_rpy_obs_bias_rad", None)
        if rpy_bias is None:
            return quat_w
        rpy_bias = rpy_bias.to(device=quat_w.device, dtype=quat_w.dtype)
        bias_delta = quat_from_euler_xyz(rpy_bias[:, 0], rpy_bias[:, 1], rpy_bias[:, 2])
        return quat_mul(bias_delta, quat_w)

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]
    
    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_ee_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.ee_body_indexes]
    
    @property
    def robot_ee_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.ee_body_indexes]
    
    @property
    def robot_ee_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.ee_body_indexes]
    
    @property
    def robot_ee_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.ee_body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self.box.data.root_pos_w
    
    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.box.data.root_quat_w
    
    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self.box.data.root_lin_vel_w
    
    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self.box.data.root_ang_vel_w
    
    @property
    def ref_robot_future_body_pos_w(self) -> torch.Tensor:
        return self.future_anchor_pos

    @property
    def ref_robot_future_body_quat_w(self) -> torch.Tensor:
        return self.future_anchor_ori

    @property
    def ref_robot_future_body_lin_vel_w(self) -> torch.Tensor:
        return self.future_body_lin_vel

    @property
    def ref_robot_future_body_ang_vel_w(self) -> torch.Tensor:
        return self.future_body_ang_vel
    
    @property
    def ref_object_future_pos_w(self) -> torch.Tensor:
        return self.future_object_pos
    
    @property
    def ref_object_future_quat_w(self) -> torch.Tensor:
        return self.future_object_ori

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        self.metrics["error_object_pos"] = torch.norm(self.data_object_pos_w - self.object_pos_w, dim=-1)
        self.metrics["error_object_rot"] = quat_error_magnitude(self.data_object_quat_w, self.object_quat_w)

    def _adaptive_bin_sampling(self, env_ids: torch.Tensor, frame_level: bool = False):
        """Adaptive sampling within the motion subset that matches each environment's asset label."""
        if len(env_ids) == 0:
            return

        self._update_current_failures_tensorized(env_ids)

        current_asset_ids = self.env_asset_ids_tensor[env_ids]
        unique_asset_ids = torch.unique(current_asset_ids)
        all_motion_indices = torch.arange(self.num_motions, device=self.device)

        for asset_id in unique_asset_ids.tolist():
            asset_env_mask = current_asset_ids == asset_id
            asset_env_ids = env_ids[asset_env_mask]
            valid_motion_count = int(self.valid_motions_counts[asset_id].item())

            if valid_motion_count > 0:
                valid_motions = self.valid_motions_table[asset_id, :valid_motion_count]
            else:
                valid_motions = all_motion_indices
                valid_motion_count = self.num_motions

            asset_bin_counts = self.bin_count_per_motion[valid_motions]
            asset_total_bins = int(asset_bin_counts.sum().item())
            asset_bin_mask = self.bin_mask[valid_motions]
            asset_bin_failed_count = self.bin_failed_count_per_motion[valid_motions][asset_bin_mask]

            bin_sampling_probs = asset_bin_failed_count + self.cfg.adaptive_uniform_ratio / float(asset_total_bins)
            bin_sampling_probs = torch.nn.functional.pad(
                bin_sampling_probs.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            bin_sampling_probs = torch.nn.functional.conv1d(
                bin_sampling_probs, self.kernel.view(1, 1, -1)
            ).view(-1)
            bin_sampling_probs = bin_sampling_probs / bin_sampling_probs.sum()

            sampled_asset_bins = torch.multinomial(bin_sampling_probs, len(asset_env_ids), replacement=True)
            asset_motion_offsets = torch.cat([
                torch.tensor([0], device=self.device),
                asset_bin_counts.cumsum(0)[:-1],
            ])
            sampled_motion_local_idx = torch.bucketize(sampled_asset_bins, asset_motion_offsets[1:], right=True)
            sampled_motion_indices = valid_motions[sampled_motion_local_idx]
            sampled_local_bins = sampled_asset_bins - asset_motion_offsets[sampled_motion_local_idx]

            self.env_motion_idx[asset_env_ids] = sampled_motion_indices

            selected_motion_total_steps = self.motion_total_steps[sampled_motion_indices]
            selected_bin_counts = self.bin_count_per_motion[sampled_motion_indices]
            if frame_level:
                new_time_steps = sampled_local_bins
            else:
                new_time_steps = (
                    sampled_local_bins * (selected_motion_total_steps - 1) // selected_bin_counts
                ).long()
            self.time_steps[asset_env_ids] = new_time_steps

            motion_failed_count = self.bin_failed_count_per_motion[valid_motions].sum(dim=1)
            motion_sampling_probs = motion_failed_count + self.cfg.adaptive_uniform_ratio / float(valid_motion_count)
            motion_sampling_probs = motion_sampling_probs / motion_sampling_probs.sum()

            motion_H = -(motion_sampling_probs * (motion_sampling_probs + 1e-12).log()).sum()
            motion_H_norm = (
                motion_H / math.log(valid_motion_count)
                if valid_motion_count > 1
                else torch.tensor(0.0, device=self.device)
            )
            motion_pmax, _ = motion_sampling_probs.max(dim=0)
            self.metrics["sampling_motion_entropy"][asset_env_ids] = motion_H_norm
            self.metrics["sampling_motion_top1_prob"][asset_env_ids] = motion_pmax

            bin_H = -(bin_sampling_probs * (bin_sampling_probs + 1e-12).log()).sum()
            bin_H_norm = (
                bin_H / math.log(asset_total_bins)
                if asset_total_bins > 1
                else torch.tensor(0.0, device=self.device)
            )
            bin_pmax, bin_imax = bin_sampling_probs.max(dim=0)

            top_motion_local = torch.bucketize(bin_imax, asset_motion_offsets[1:], right=True)
            top_local_bin = bin_imax - asset_motion_offsets[top_motion_local]
            top_position_denominator = asset_bin_counts[top_motion_local]
            if frame_level:
                top_position_denominator = torch.clamp(top_position_denominator - 1, min=1)

            self.metrics["sampling_entropy"][asset_env_ids] = bin_H_norm
            self.metrics["sampling_top1_prob"][asset_env_ids] = bin_pmax
            self.metrics["sampling_top1_bin"][asset_env_ids] = (
                top_local_bin.float() / top_position_denominator
            )

    def _adaptive_frame_sampling(self, env_ids: torch.Tensor):
        """Adaptively sample exact frames from motions matching each environment's asset."""
        self._adaptive_bin_sampling(env_ids, frame_level=True)

    def _uniform_bin_sampling(self, env_ids: torch.Tensor, frame_level: bool = False):
        """Uniformly sample matching motions, then uniformly sample bins or exact frames."""
        if len(env_ids) == 0:
            return

        # --- 1. Get asset information for each environment. ---
        current_asset_ids = self.env_asset_ids_tensor[env_ids]
        counts = self.valid_motions_counts[current_asset_ids]
        has_matching_motions = counts > 0
        effective_counts = torch.where(
            has_matching_motions,
            counts,
            torch.full_like(counts, self.num_motions),
        )

        # --- 2. Sample motions. ---
        selected_motions = torch.empty(len(env_ids), device=self.device, dtype=torch.long)

        if torch.any(has_matching_motions):
            matched_asset_ids = current_asset_ids[has_matching_motions]
            matched_counts = counts[has_matching_motions]
            rand_local_indices = (torch.rand(matched_asset_ids.numel(), device=self.device) * matched_counts).long()
            selected_motions[has_matching_motions] = self.valid_motions_table[matched_asset_ids, rand_local_indices]

        if torch.any(~has_matching_motions):
            fallback_count = int((~has_matching_motions).sum().item())
            selected_motions[~has_matching_motions] = torch.randint(
                0, self.num_motions, (fallback_count,), device=self.device
            )

        self.env_motion_idx[env_ids] = selected_motions

        # --- 3. Sample time steps. ---
        selected_total_steps = self.motion_total_steps[selected_motions]
        selected_sample_counts = selected_total_steps if frame_level else self.bin_counts[selected_motions]
        sampled_indices = torch.floor(
            torch.rand(len(env_ids), device=self.device) * selected_sample_counts.float()
        ).long()
        sampled_indices = torch.clamp(sampled_indices, max=selected_sample_counts - 1)

        if frame_level:
            self.time_steps[env_ids] = sampled_indices
        else:
            new_time_steps = (
                sampled_indices.float()
                / selected_sample_counts.float()
                * (selected_total_steps.float() - 1.0)
            )
            self.time_steps[env_ids] = new_time_steps.long()

        # --- 4. Update metrics using vectorized operations. ---
        # Compute motion entropy from valid motion counts; use global uniform sampling when no motions match.
        prob_motion = 1.0 / effective_counts.float()
        motion_H = -torch.log(prob_motion + 1e-12)
        log_counts = torch.log(effective_counts.float())

        safe_log_counts = torch.where(effective_counts > 1, log_counts, torch.ones_like(log_counts))
        raw_norm = motion_H / safe_log_counts
        motion_H_norm = torch.where(effective_counts > 1, raw_norm, torch.zeros_like(raw_norm))
        
        self.metrics["sampling_motion_entropy"][env_ids] = motion_H_norm
        self.metrics["sampling_motion_top1_prob"][env_ids] = prob_motion
        sample_prob = 1.0 / selected_sample_counts.float()
        self.metrics["sampling_entropy"][env_ids] = (selected_sample_counts > 1).float()
        self.metrics["sampling_top1_prob"][env_ids] = sample_prob
        position_denominator = selected_sample_counts
        if frame_level:
            position_denominator = torch.clamp(position_denominator - 1, min=1)
        self.metrics["sampling_top1_bin"][env_ids] = sampled_indices.float() / position_denominator.float()

    def _uniform_frame_sampling(self, env_ids: torch.Tensor):
        """Uniformly sample matching motions, then uniformly sample exact frames."""
        self._uniform_bin_sampling(env_ids, frame_level=True)

    def _fixed_frame_sampling(self, env_ids: torch.Tensor):
        """Select the first matching motion and clamp the configured frame to its final frame."""
        if len(env_ids) == 0:
            return

        asset_ids = self.env_asset_ids_tensor[env_ids]
        counts = self.valid_motions_counts[asset_ids]
        selected_motions = torch.zeros(len(env_ids), device=self.device, dtype=torch.long)
        has_matching_motions = counts > 0
        if torch.any(has_matching_motions):
            selected_motions[has_matching_motions] = self.valid_motions_table[
                asset_ids[has_matching_motions], 0
            ]

        self.env_motion_idx[env_ids] = selected_motions
        selected_total_steps = self.motion_total_steps[selected_motions]
        fixed_steps = torch.full_like(selected_total_steps, int(self.cfg.fixed_frame))
        self.time_steps[env_ids] = torch.minimum(fixed_steps, selected_total_steps - 1)
        denominator = torch.clamp(selected_total_steps - 1, min=1)

        self.metrics["sampling_entropy"][env_ids] = 0.0
        self.metrics["sampling_top1_prob"][env_ids] = 1.0
        self.metrics["sampling_top1_bin"][env_ids] = self.time_steps[env_ids].float() / denominator.float()
        self.metrics["sampling_motion_entropy"][env_ids] = 0.0
        self.metrics["sampling_motion_top1_prob"][env_ids] = 1.0

    def _resample_command(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        # TODO(hanlei): update adaptive sampling
        if self.cfg.sampling_method == "uniform_bin":
            self._uniform_bin_sampling(env_ids)
        elif self.cfg.sampling_method == "uniform_frame":
            self._uniform_frame_sampling(env_ids)
        elif self.cfg.sampling_method == "adaptive_bin":
            self._adaptive_bin_sampling(env_ids)
        elif self.cfg.sampling_method == "adaptive_frame":
            self._adaptive_frame_sampling(env_ids)
        elif self.cfg.sampling_method == "fixed_frame":
            self._fixed_frame_sampling(env_ids)
        else:
            raise ValueError(f"Unknown sampling method {self.cfg.sampling_method}")

        ### Add noise to the robot state ###
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        
        ### Add noise to the object state ###
        object_pos = self.data_object_pos_w.clone()
        object_ori = self.data_object_quat_w.clone()
        object_lin_vel = self.data_object_lin_vel_w.clone()
        object_ang_vel = self.data_object_ang_vel_w.clone()

        range_list = [self.cfg.object_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        object_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        object_ori[env_ids] = quat_mul(orientations_delta, object_ori[env_ids])
        range_list = [self.cfg.object_velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        object_lin_vel[env_ids] += rand_samples[:, :3]
        object_ang_vel[env_ids] += rand_samples[:, 3:]

        self.box.write_root_state_to_sim(
            torch.cat([object_pos[env_ids], object_ori[env_ids], object_lin_vel[env_ids], object_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

        ### Write table1 and table2 states ###
        table1_pos = self.table1_pos_w.clone()
        uses_table1 = self._motion_uses_table1[self.env_motion_idx[env_ids]]
        ground_env_ids = env_ids[~uses_table1]
        if ground_env_ids.numel() > 0:
            table1_pos[ground_env_ids] = (
                self._env.scene.env_origins[ground_env_ids] + self._table1_park_offset
            )
        default_ori = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        default_vel = torch.zeros((len(env_ids), 6), device=self.device)
        self.table1.write_root_state_to_sim(
            torch.cat([table1_pos[env_ids], default_ori, default_vel], dim=-1),
            env_ids=env_ids,
        )
        table2_pos = self.table2_pos_w
        self.table2.write_root_state_to_sim(
            torch.cat([table2_pos[env_ids], default_ori, default_vel], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        
        max_steps = self._motion_lengths[self.env_motion_idx]
        env_ids = torch.where(self.time_steps >= max_steps)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)
        
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        if "adaptive" in self.cfg.sampling_method:
            if isinstance(self.bin_failed_count_per_motion, torch.Tensor):
                self.bin_failed_count_per_motion = (
                        self.cfg.adaptive_alpha * self._current_bin_failed_per_motion
                        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count_per_motion
                )
                self._current_bin_failed_per_motion.zero_()
            else:
                for i in range(self.num_motions):
                    self.bin_failed_count_per_motion[i] = (
                            self.cfg.adaptive_alpha * self._current_bin_failed_per_motion[i]
                            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count_per_motion[i]
                    )
                    self._current_bin_failed_per_motion[i].zero_()

        # Update motion-level failure
        self.motion_failed_count = (
                self.cfg.adaptive_alpha * self._current_motion_failed
                + (1 - self.cfg.adaptive_alpha) * self.motion_failed_count
        )
        self._current_motion_failed.zero_()
    

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name))
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name))
                    )

            # 2. Initialize the wireframe drawing interface.
            if not hasattr(self, "_draw_interface"):
                self._init_bbox_lines()

            # Make the markers visible.
            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
            
            if hasattr(self, "_draw_interface"):
                self._draw_interface.clear_lines()


    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])

        # BBox Visualization ---
        if hasattr(self, "object_dims"):
            self._draw_interface.clear_lines()
            
            bbox_scaled_offset = self.bbox_offset * self.object_dims.unsqueeze(1) # [num_envs, 8, 3]
            bbox_pos_w = quat_apply(self.object_quat_w.unsqueeze(1).repeat(1, 8, 1), bbox_scaled_offset) + self.object_pos_w.unsqueeze(1)  # [num_envs, 8, 3]
            
            line_starts = bbox_pos_w[:, self.line_indices[:, 0]].reshape(-1, 3) # [num_envs * 12, 3]
            line_ends   = bbox_pos_w[:, self.line_indices[:, 1]].reshape(-1, 3) # [num_envs * 12, 3]

            colors = (0.0, 1.0, 0.0, 1.0)
            self._draw_interface.draw_lines(
                line_starts.tolist(), 
                line_ends.tolist(), 
                [colors] * line_starts.shape[0], 
                [2.0] * (line_starts.shape[0])
            )
        

    def _init_bbox_lines(self):
        from isaacsim.util.debug_draw import _debug_draw
        self.line_indices = torch.tensor([
            [0, 1], [2, 3], [4, 5], [6, 7],
            [0, 2], [1, 3], [4, 6], [5, 7], 
            [0, 4], [1, 5], [2, 6], [3, 7]  
        ], device=self.device, dtype=torch.long)
        self._draw_interface = _debug_draw.acquire_debug_draw_interface()


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING
    object_name: str = "box"
    table1_name: str = "table1"
    table2_name: str = "table2"
    table1_park_offset: tuple[float, float, float] = (0.0, 0.0, -10.0)
    motion_file_list: Any = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    object_pose_range: dict[str, tuple[float, float]] = {}
    object_velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # Supported modes: uniform_bin, uniform_frame, adaptive_bin, adaptive_frame, fixed_frame.
    sampling_method: str = "uniform_bin"
    fixed_frame: int = 0

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    future_frames = (0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50)
    num_future_frames = len(future_frames)

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    bbox_line_thickness: float = 2.0
    bbox_line_color: tuple = (0.0, 1.0, 0.0, 1.0) # Green.
