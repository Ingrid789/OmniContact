from __future__ import annotations

import os
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, yaw_quat
from omnicontact.utils.math_utils import quat_apply_inverse


class AMPCarryBoxMotionDataset:
    """Builds AMP expert transitions (s_t, s_{t+1}) from carrybox motion files."""

    def __init__(self, cfg: AMPCarryBoxMotionDatasetCfg, env, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.history_length = int(cfg.history_length)

        self.robot = env.scene[cfg.asset_name]
        # Keep for compatibility with existing AMP cfg schemas.
        self.body_names = list(cfg.body_names) if cfg.body_names is not None else []

        self.observation_terms = list(cfg.amp_obs_terms)
        self._term_tensor_cache: dict[str, torch.Tensor] = {}
        self._default_joint_pos = self.robot.data.default_joint_pos[0].to(self.device).detach()
        self._gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)

        self._load_motions()
        body_dim = int(self.body_pos_w_all.shape[1])
        self.anchor_body_index = self._resolve_body_index(cfg.anchor_name, body_dim)
        self.root_body_index = self._resolve_body_index(cfg.root_body_name, body_dim)
        self._init_observation_dims()

    def _find_single_body_index_in_robot(self, body_name: str) -> int:
        body_ids, _ = self.robot.find_bodies(body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(f"Expected exactly one body for '{body_name}', got {len(body_ids)}.")
        return int(body_ids[0])

    def _resolve_body_index(self, body_name: str, body_dim: int) -> int:
        # Preferred: use cfg.body_names order because exported carrybox npz usually stores only these bodies.
        if self.body_names:
            if body_name not in self.body_names:
                raise ValueError(f"Body '{body_name}' is not present in amp_data.body_names.")
            cfg_index = self.body_names.index(body_name)
            if cfg_index < body_dim:
                return cfg_index

        # Fallback: use global robot body index (for full-body motion dumps).
        robot_index = self._find_single_body_index_in_robot(body_name)
        if robot_index < body_dim:
            return robot_index

        raise ValueError(
            f"Cannot resolve body index for '{body_name}'. "
            f"Resolved robot index={robot_index}, but motion body dim={body_dim}."
        )

    def _load_motions(self):
        if not self.cfg.motion_files:
            raise ValueError("amp_data.motion_files is empty.")

        joint_pos_list: list[torch.Tensor] = []
        body_pos_w_list: list[torch.Tensor] = []
        body_quat_w_list: list[torch.Tensor] = []
        body_lin_vel_w_list: list[torch.Tensor] = []
        body_ang_vel_w_list: list[torch.Tensor] = []
        ee_pos_w_list: list[torch.Tensor] = []
        ee_quat_w_list: list[torch.Tensor] = []
        object_pos_w_list: list[torch.Tensor] = []
        object_quat_w_list: list[torch.Tensor] = []
        contact_info_list: list[torch.Tensor] = []
        traj_lengths: list[int] = []

        required_keys = (
            "joint_pos",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "ee_pos_w",
            "ee_quat_w",
            "object_pos_w",
            "object_quat_w",
        )

        for motion_file in self.cfg.motion_files:
            if not os.path.isfile(motion_file):
                raise FileNotFoundError(f"Invalid motion file: {motion_file}")
            data = np.load(motion_file)
            missing_keys = [k for k in required_keys if k not in data.files]
            if missing_keys:
                raise KeyError(f"Missing keys {missing_keys} in motion file: {motion_file}")

            joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32)
            body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32)
            body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32)
            body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32)
            body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32)
            ee_pos_w = torch.tensor(data["ee_pos_w"], dtype=torch.float32)
            ee_quat_w = torch.tensor(data["ee_quat_w"], dtype=torch.float32)
            object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32)
            object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32)
            if "contact_info" in data.files:
                contact_info = torch.tensor(data["contact_info"], dtype=torch.float32).reshape(joint_pos.shape[0], -1)
            else:
                contact_info = torch.zeros((joint_pos.shape[0], 4), dtype=torch.float32)
            traj_len = int(joint_pos.shape[0])

            traj_lengths.append(traj_len)
            joint_pos_list.append(joint_pos)
            body_pos_w_list.append(body_pos_w)
            body_quat_w_list.append(body_quat_w)
            body_lin_vel_w_list.append(body_lin_vel_w)
            body_ang_vel_w_list.append(body_ang_vel_w)
            ee_pos_w_list.append(ee_pos_w)
            ee_quat_w_list.append(ee_quat_w)
            object_pos_w_list.append(object_pos_w)
            object_quat_w_list.append(object_quat_w)
            contact_info_list.append(contact_info)

        self.joint_pos_all = torch.cat(joint_pos_list, dim=0).to(self.device)
        self.body_pos_w_all = torch.cat(body_pos_w_list, dim=0).to(self.device)
        self.body_quat_w_all = torch.cat(body_quat_w_list, dim=0).to(self.device)
        self.body_lin_vel_w_all = torch.cat(body_lin_vel_w_list, dim=0).to(self.device)
        self.body_ang_vel_w_all = torch.cat(body_ang_vel_w_list, dim=0).to(self.device)
        self.ee_pos_w_all = torch.cat(ee_pos_w_list, dim=0).to(self.device)
        self.ee_quat_w_all = torch.cat(ee_quat_w_list, dim=0).to(self.device)
        self.object_pos_w_all = torch.cat(object_pos_w_list, dim=0).to(self.device)
        self.object_quat_w_all = torch.cat(object_quat_w_list, dim=0).to(self.device)
        self.contact_info_all = torch.cat(contact_info_list, dim=0).to(self.device)

        self.total_dataset_size = int(self.joint_pos_all.shape[0])
        self._traj_lengths = traj_lengths
        self.index_t, self.index_tp1 = self._build_transition_indices(traj_lengths)

    def _build_transition_indices(self, traj_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        idx_t = []
        idx_tp1 = []
        offset = 0
        for length in traj_lengths:
            if length >= 2:
                t = torch.arange(offset, offset + length - 1, device=self.device)
                idx_t.append(t)
                idx_tp1.append(t + 1)
            offset += length
        if not idx_t:
            raise ValueError("No valid transitions found in motion dataset.")
        return torch.cat(idx_t), torch.cat(idx_tp1)

    def _anchor_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.body_pos_w_all[:, self.anchor_body_index],
            self.body_quat_w_all[:, self.anchor_body_index],
        )

    def _root_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.body_pos_w_all[:, self.root_body_index],
            self.body_quat_w_all[:, self.root_body_index],
        )

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.joint_pos_all - self._default_joint_pos.unsqueeze(0)

    @property
    def ee_pos(self) -> torch.Tensor:
        anchor_pos_w, anchor_quat_w = self._anchor_pose_w()
        num_ee = self.ee_pos_w_all.shape[1]
        pos_b, _ = subtract_frame_transforms(
            anchor_pos_w[:, None, :].expand(-1, num_ee, -1),
            anchor_quat_w[:, None, :].expand(-1, num_ee, -1),
            self.ee_pos_w_all,
            self.ee_quat_w_all,
        )
        return pos_b.reshape(self.total_dataset_size, -1)

    @property
    def base_height(self) -> torch.Tensor:
        root_pos_w, _ = self._root_pose_w()
        return root_pos_w[:, 2:3]

    @property
    def base_lin_vel(self) -> torch.Tensor:
        _, root_quat_w = self._root_pose_w()
        root_lin_vel_w = self.body_lin_vel_w_all[:, self.root_body_index]
        return quat_apply_inverse(root_quat_w, root_lin_vel_w)

    @property
    def base_ang_vel(self) -> torch.Tensor:
        _, root_quat_w = self._root_pose_w()
        root_ang_vel_w = self.body_ang_vel_w_all[:, self.root_body_index]
        return quat_apply_inverse(root_quat_w, root_ang_vel_w)

    @property
    def projected_gravity(self) -> torch.Tensor:
        _, root_quat_w = self._root_pose_w()
        gravity = self._gravity_vec_w.unsqueeze(0).expand(self.total_dataset_size, -1)
        return quat_apply_inverse(root_quat_w, gravity)

    @property
    def box_pos_local(self) -> torch.Tensor:
        threshold = float(self.cfg.object_pos_amp_threshold)
        anchor_pos_w, anchor_quat_w = self._anchor_pose_w()
        anchor_heading = yaw_quat(anchor_quat_w)
        pos, _ = subtract_frame_transforms(
            anchor_pos_w,
            anchor_heading,
            self.object_pos_w_all,
            self.object_quat_w_all,
        )

        xy = pos[:, :2]
        xy_norm = torch.norm(xy, dim=-1, keepdim=True)
        far_mask = xy_norm > threshold
        safe_norm = xy_norm.clamp_min(1.0e-6)
        scaled_xy = xy / safe_norm * threshold
        pos[:, :2] = torch.where(far_mask, scaled_xy, xy)
        pos[far_mask.squeeze(-1), 2] = 0.0
        return pos.reshape(self.total_dataset_size, -1)

    @property
    def contact_info(self) -> torch.Tensor:
        return self.contact_info_all.reshape(self.total_dataset_size, -1)

    def _build_history_tensor(self, base_obs: torch.Tensor) -> torch.Tensor:
        """Match IsaacLab circular buffer semantics: oldest -> newest, first frame repeated at sequence start."""
        if self.history_length <= 1:
            return base_obs

        hist_chunks = []
        start = 0
        hist_ids = torch.arange(self.history_length, device=self.device).unsqueeze(0)
        for length in self._traj_lengths:
            end = start + length
            seq = base_obs[start:end]  # (L, D)
            # At reset, IsaacLab fills full history with the first observed frame.
            pad = seq[:1].repeat(self.history_length - 1, 1)  # (H-1, D)
            padded = torch.cat([pad, seq], dim=0)  # (L+H-1, D)
            base_ids = torch.arange(length, device=self.device).unsqueeze(1)  # (L, 1)
            window = padded[base_ids + hist_ids]  # (L, H, D), oldest -> newest
            hist_chunks.append(window.reshape(length, -1))
            start = end
        return torch.cat(hist_chunks, dim=0)

    def _get_term_tensor(self, term_name: str) -> torch.Tensor:
        if term_name not in self._term_tensor_cache:
            try:
                base_term = getattr(self, term_name)
            except AttributeError as exc:
                raise AttributeError(f"Unknown AMP observation term: {term_name}") from exc
            if not isinstance(base_term, torch.Tensor):
                raise TypeError(f"AMP observation term '{term_name}' is not a tensor.")
            self._term_tensor_cache[term_name] = self._build_history_tensor(base_term)
        return self._term_tensor_cache[term_name]

    def _observation_dim(self, term_name: str) -> int:
        term = self._get_term_tensor(term_name)
        return int(term.shape[-1])

    def _init_observation_dims(self):
        self.observation_dims = [self._observation_dim(term) for term in self.observation_terms]
        self.observation_dim = int(sum(self.observation_dims))

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_ids = torch.randint(0, len(self.index_t), (batch_size,), device=self.device)
        return self.index_t[sample_ids], self.index_tp1[sample_ids]

    def build_transition(self, t: torch.Tensor, tp1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_terms = []
        next_state_terms = []
        for term in self.observation_terms:
            obs = self._get_term_tensor(term)
            state_terms.append(obs[t])
            next_state_terms.append(obs[tp1])
        return torch.cat(state_terms, dim=-1), torch.cat(next_state_terms, dim=-1)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            t, tp1 = self.sample_batch(mini_batch_size)
            yield self.build_transition(t, tp1)


@configclass
class AMPCarryBoxMotionDatasetCfg:
    class_type: type[AMPCarryBoxMotionDataset] = AMPCarryBoxMotionDataset
    asset_name: str = "robot"
    motion_files: list[str] = MISSING
    body_names: list[str] | None = None
    anchor_name: str = "torso_link"
    root_body_name: str = "pelvis"
    amp_obs_terms: list[str] | None = None
    history_length: int = 1
    object_pos_amp_threshold: float = 1.0
