from __future__ import annotations

from typing import Any

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import omnicontact.tasks.omnicontact.mdp as mdp
from omnicontact.tasks.omnicontact.env_cfg import CFBaseEnvCfg, CFBaseEnvPlayCfg, ObservationsCfg


@configclass
class AMPObservationsCfg(ObservationsCfg):
    """Carrybox observations with additional AMP group."""

    @configclass
    class AmpCfg(ObsGroup):
        """Observations for AMP discriminator."""

        base_height = ObsTerm(func=mdp.base_height)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        box_pos_local = ObsTerm(func=mdp.object_pos_amp_b, params={"command_name": "motion"})
        contact_info = ObsTerm(
            func=mdp.robot_object_contact_info_obs,
            params={
                "command_name": "motion",
                "hand_threshold": 5.0,
                "foot_threshold": 0.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10

    amp: AmpCfg = AmpCfg()


@configclass
class CFAMPEnvCfg(CFBaseEnvCfg):
    observations: AMPObservationsCfg | Any = AMPObservationsCfg()

@configclass
class CFAMPEnvPlayCfg(CFBaseEnvPlayCfg):
    observations: AMPObservationsCfg | Any = AMPObservationsCfg()
