"""Object asset configurations selected from reference motion files."""

from __future__ import annotations

import re

import isaaclab.sim as sim_utils

from omnicontact.assets import asset_path
from omnicontact.utils.motion_labels import asset_label_from_motion_file


class OBJ_SPAWNER_CFG:
    _PUSH_PREFIX = "push_"
    _BOX_SIZE = r"\d+(?:-\d+){2}"
    
    @classmethod
    def _asset_cfg_from_label(cls, label: str) -> sim_utils.SpawnerCfg:
        asset_label = label[len(cls._PUSH_PREFIX) :] if label.lower().startswith(cls._PUSH_PREFIX) else label
        if cls._is_ball_label(asset_label):
            asset_dir = asset_label.lower()
            usd_name = asset_dir
            mass = 0.5
        elif cls._is_box_label(asset_label):
            asset_dir = f"box_{asset_label}"
            usd_name = f"box_{asset_label}"
            mass = 2.0
        usd_path = asset_path("objects", asset_dir, f"{usd_name}.usd")

        return sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            semantic_tags=[("class", cls._semantic_tag_from_label(label))],
        )


    @staticmethod
    def _is_ball_label(label: str) -> bool:
        return re.fullmatch(r"(?:ball|soccer)_?\d+", label.lower()) is not None

    @staticmethod
    def _is_box_label(label: str) -> bool:
        return re.fullmatch(OBJ_SPAWNER_CFG._BOX_SIZE, str(label)) is not None

    @staticmethod
    def _semantic_tag_from_label(label: str) -> str:
        return str(label).replace("-", "_")

    @classmethod
    def _extract_label_counts_from_motion_files(cls, motion_files: list[str] | tuple[str, ...]) -> list[tuple[str, int]]:
        label_counts: dict[str, int] = {}
        for motion_file in motion_files:
            label = asset_label_from_motion_file(motion_file)
            if label is None:
                continue
            label_counts[label] = label_counts.get(label, 0) + 1
        return list(label_counts.items())

    @classmethod
    def get_all_assets(cls, motion_files: list[str] | tuple[str, ...] | None = None) -> list[sim_utils.SpawnerCfg]:
        if not motion_files:
            return [cls._asset_cfg_from_label("30-30-30")]

        label_counts = cls._extract_label_counts_from_motion_files(motion_files)
        
        assets_cfg: list[sim_utils.SpawnerCfg] = []
        for label, count in label_counts:
            assets_cfg.extend(cls._asset_cfg_from_label(label) for _ in range(count))
        return assets_cfg


def configure_box_assets_for_motion_files(env_cfg) -> None:
    motion_files = list(getattr(env_cfg.commands.motion, "motion_file_list", []) or [])
    env_cfg.scene.box.spawn.assets_cfg = OBJ_SPAWNER_CFG.get_all_assets(motion_files=motion_files)
