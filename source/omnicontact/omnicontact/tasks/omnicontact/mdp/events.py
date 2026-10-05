from __future__ import annotations

import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import omni.usd
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_box_material_by_label(ManagerTermBase):
    """Randomize label-specific materials, updating all selected environments together."""

    _LABEL_PATTERNS = {
        "ball": re.compile(r"ball-?\d+$", re.IGNORECASE),
        "soccer": re.compile(r"soccer-?\d+$", re.IGNORECASE),
        "push": re.compile(r"push-\d+(?:-\d+){2}$", re.IGNORECASE),
    }

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params.copy()
        self.asset_cfg: SceneEntityCfg = params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        self.num_buckets = int(params.get("num_buckets", 1))

        # Soccer uses the box ranges when no separate ranges are configured.
        for kind in ("static", "dynamic"):
            params.setdefault(f"soccer_{kind}_friction_range", params[f"box_{kind}_friction_range"])

        self.material_buckets = {
            group: self._sample_material_buckets(
                static_friction_range=params[f"{group}_static_friction_range"],
                dynamic_friction_range=params[f"{group}_dynamic_friction_range"],
                restitution_range=params.get("restitution_range", (0.0, 0.0)),
            )
            for group in ("push", "ball", "soccer", "box")
        }

    def _sample_material_buckets(
        self,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
    ) -> torch.Tensor:
        ranges = torch.tensor(
            [static_friction_range, dynamic_friction_range, restitution_range], device="cpu"
        )
        return math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (self.num_buckets, 3), device="cpu"
        )

    def _get_material_group(self, stage, env_id: int) -> str:
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Object")
        if prim.IsValid():
            for prop in prim.GetProperties():
                name = prop.GetName()
                if "semantic" in name and "semanticData" in name:
                    value = prop.Get()
                    if value:
                        label = str(value).replace("_", "-")
                        return next(
                            (group for group, pattern in self._LABEL_PATTERNS.items() if pattern.fullmatch(label)),
                            "box",
                        )
        return "box"

    # Explicit parameters are required by Isaac Lab's event configuration validation.
    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        push_static_friction_range: tuple[float, float],
        push_dynamic_friction_range: tuple[float, float],
        ball_static_friction_range: tuple[float, float],
        ball_dynamic_friction_range: tuple[float, float],
        box_static_friction_range: tuple[float, float],
        box_dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        soccer_static_friction_range: tuple[float, float] | None = None,
        soccer_dynamic_friction_range: tuple[float, float] | None = None,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.long)
        if env_ids.numel() == 0:
            return

        stage = omni.usd.get_context().get_stage()
        groups = {group: [] for group in (*self._LABEL_PATTERNS, "box")}
        for env_id in env_ids.tolist():
            groups[self._get_material_group(stage, env_id)].append(env_id)

        view = self.asset.root_physx_view
        materials = view.get_material_properties()
        for group, ids in groups.items():
            if ids:
                bucket_ids = torch.randint(self.num_buckets, (len(ids), view.max_shapes), device="cpu")
                materials[ids] = self.material_buckets[group][bucket_ids]
        view.set_material_properties(materials, env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_torso_rpy_obs_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    roll_range_deg: tuple[float, float] = (-5.0, 5.0),
    pitch_range_deg: tuple[float, float] = (-5.0, 5.0),
    yaw_range_deg: tuple[float, float] = (-12.0, 12.0),
    bias_attr: str = "torso_rpy_obs_bias_rad",
):
    """Sample a per-env torso roll/pitch/yaw observation bias at reset.

    The sampled bias only affects observation functions that explicitly read ``bias_attr`` from ``env``.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    if not hasattr(env, bias_attr):
        setattr(env, bias_attr, torch.zeros(env.scene.num_envs, 3, device=env.device))
    bias = getattr(env, bias_attr)
    low = torch.tensor(
        [float(roll_range_deg[0]), float(pitch_range_deg[0]), float(yaw_range_deg[0])],
        device=env.device,
    )
    high = torch.tensor(
        [float(roll_range_deg[1]), float(pitch_range_deg[1]), float(yaw_range_deg[1])],
        device=env.device,
    )
    sampled_deg = math_utils.sample_uniform(low, high, (len(env_ids), 3), device=env.device)
    bias[env_ids] = torch.deg2rad(sampled_deg)
