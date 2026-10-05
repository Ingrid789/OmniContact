# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING
import torch

import onnx
from rsl_rl.modules import ActorCritic

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def export_motion_policy_as_onnx(
        env: ManagerBasedRLEnv,
        actor_critic: object,
        path: str,
        normalizer: object | None = None,
        filename="policy.onnx",
        verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(torch.nn.Module):
    def __init__(self,
                 env: ManagerBasedRLEnv,
                 actor_critic: object,
                 normalizer=None,
                 verbose=False):
        super().__init__()
        if actor_critic.is_recurrent:
            raise ValueError("Motion ONNX export currently supports feed-forward policies only.")
        if not isinstance(actor_critic, ActorCritic):
            raise TypeError("Motion ONNX export requires an RSL-RL ActorCritic policy.")
        self.actor = copy.deepcopy(actor_critic.actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()
        self.verbose = verbose
        self.num_actor_obs = actor_critic.num_actor_obs

        cmd = env.command_manager.get_term("motion")
        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.extra_motion_outputs = []
        extra_motion_names = ["object_pos_w", "object_quat_w", "contact_info", "table1_pos_w"]
        if cmd.cfg.table2_name is not None:
            extra_motion_names.append("table2_pos_w")
        for name in extra_motion_names:
            if hasattr(cmd.motion, name):
                setattr(self, name, getattr(cmd.motion, name).to("cpu"))
                self.extra_motion_outputs.append(name)
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), min=0, max=self.time_step_total - 1)
        actor_outputs = self.actor(self.normalizer(x))
        outputs = (
            actor_outputs,
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )
        return outputs + tuple(getattr(self, name)[time_step_clamped] for name in self.extra_motion_outputs)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.num_actor_obs)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=17,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ] + self.extra_motion_outputs,
            dynamic_axes={},
            dynamo=False,
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    robot_data = env.scene["robot"].data
    joint_action = env.action_manager.get_term("joint_pos")
    joint_ids = getattr(joint_action, "_joint_ids", slice(None))
    joint_names = getattr(joint_action, "_joint_names", robot_data.joint_names)
    default_joint_pos = getattr(robot_data, "default_joint_pos_nominal", None)
    if default_joint_pos is None:
        default_joint_pos = robot_data.default_joint_pos[0]
    metadata = {
        "run_path": run_path,
        "joint_names": joint_names,
        "joint_stiffness": robot_data.joint_stiffness[0, joint_ids].cpu().tolist(),
        "joint_damping": robot_data.joint_damping[0, joint_ids].cpu().tolist(),
        "default_joint_pos": default_joint_pos[joint_ids].cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": [env.observation_manager.active_terms[name] for name in env.observation_manager.active_terms if 'policy' in name],
        "action_scale": joint_action._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
