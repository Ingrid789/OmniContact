from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omnicontact.tasks.omnicontact.mdp as mdp
from omnicontact.assets.object_spawner import OBJ_SPAWNER_CFG

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = MISSING
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=OBJ_SPAWNER_CFG.get_all_assets(),
            random_choice=True,
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    table1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table1",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.25, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,     
                angular_damping=1000.0,
                linear_damping=1000.0,  
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, 
                rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0) # red
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, -0.2, 0.94), 
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    table2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table2",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.25, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                angular_damping=1000.0,
                linear_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0)  # green
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, -0.2, 0.94),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
        
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=False
    )
    left_ankle_contact_forces_object = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=0.0,
        debug_vis=False,
    )
    right_ankle_contact_forces_object = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=0.0,
        debug_vis=False,
    )
    left_wrist_contact_forces_object = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=10.0,
        debug_vis=False,
    )
    right_wrist_contact_forces_object = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=10.0,
        debug_vis=False,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        table1_name="table1",
        table2_name="table2",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        object_pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        object_velocity_range={},
        joint_position_range=(-0.1, 0.1),
        sampling_method="uniform_bin",  # frame options: uniform_frame, adaptive_frame
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCommandCfg(ObsGroup):
        """Observations for policy commands (no history)."""
        # Command observations only.
        command = ObsTerm(func=mdp.tracking_commands_contact_flow, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = True # Command noise is usually unnecessary; adjust as needed.
            self.concatenate_terms = True


    @configclass
    class PolicyProprioCfg(ObsGroup):
        """Observations for proprioception (with history)."""
        ee_pos = ObsTerm(
            func=mdp.motion_ee_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        
        # object
        box_pos_local = ObsTerm(func=mdp.object_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05))
        box_ori_local = ObsTerm(func=mdp.object_ori_6d_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PrivilegedCommandCfg(ObsGroup):
        command = ObsTerm(func=mdp.tracking_commands_contact_flow, params={"command_name": "motion"})
        
        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class PrivilegedProprioCfg(ObsGroup):
        # robot
        ee_pos = ObsTerm(func=mdp.motion_ee_pos_b, params={"command_name": "motion"})
        anchor_pos_err = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        anchor_ori_err = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # object
        box_pos_local = ObsTerm(func=mdp.object_pos_b, params={"command_name": "motion"})
        box_ori_local = ObsTerm(func=mdp.object_ori_6d_b, params={"command_name": "motion"})
        
        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 5

    # observation groups
    policy_command: PolicyCommandCfg = PolicyCommandCfg()
    policy_proprio: PolicyProprioCfg = PolicyProprioCfg()
    critic_command: PrivilegedCommandCfg = PrivilegedCommandCfg()
    critic_proprio: PrivilegedProprioCfg = PrivilegedProprioCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=0,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.75, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    torso_rpy_obs_bias = EventTerm(
        func=mdp.randomize_torso_rpy_obs_bias,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "roll_range_deg": (-5.0, 5.0),
            "pitch_range_deg": (-5.0, 5.0),
            "yaw_range_deg": (-8.0, 8.0),
            "bias_attr": "torso_rpy_obs_bias_rad",
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    # -- object
    # set box friction explicitly at reset
    box_physics_material = EventTerm(
        func=mdp.randomize_box_material_by_label,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("box", body_names=".*"),
            "push_static_friction_range": (0.2, 0.4),
            "push_dynamic_friction_range": (0.1, 0.3),
            "ball_static_friction_range": (0.06, 0.12),
            "ball_dynamic_friction_range": (0.04, 0.08),
            "soccer_static_friction_range": (0.5, 0.8),
            "soccer_dynamic_friction_range": (0.4, 0.7),
            "box_static_friction_range": (0.5, 0.8),
            "box_dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    box_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("box", body_names=".*"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )


@configclass
class RewardsCfg:
    # Robot rewards
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.2,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.2,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_local_body_position_error_exp,
        weight=0.2,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_local_body_orientation_error_exp,
        weight=0.2,
        params={"command_name": "motion", "std": 0.4},
    )
    # Wrist Tracking Reward
    motion_global_body_pos = RewTerm(
        func=mdp.motion_global_body_position_error_exp,
        weight=0.4,
        params={"command_name": "motion", "std": 0.3, "body_names": ["left_rubber_hand", "right_rubber_hand"]},
    )
    motion_global_body_ori = RewTerm(
        func=mdp.motion_global_body_orientation_error_exp,
        weight=0.4,
        params={"command_name": "motion", "std": 0.4, "body_names": ["left_rubber_hand", "right_rubber_hand"]},
    )
    # Object rewards
    object_global_pos = RewTerm(
        func=mdp.motion_global_object_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    object_global_ori = RewTerm(
        func=mdp.motion_global_object_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    # Interaction rewards
    interact_pos = RewTerm(
        func=mdp.motion_local_interact_position_error_exp,
        weight=0.4,
        params={"command_name": "motion", "std": 0.3},
    )
    interact_ori = RewTerm(
        func=mdp.motion_local_interact_orientation_error_exp,
        weight=0.4,
        params={"command_name": "motion", "std": 0.3},
    )
    # Contact Reward
    contact_reward = RewTerm(
        func=mdp.contact_reward,
        weight=1.0,
        params={
            "command_name": "motion",
            "hand_threshold": 5.0,
            "foot_threshold": 0.0,
            "std": 1.0,
        },
    )

    # Penality
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    elbow_wrist_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            ),
        },
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    foot_slip_penalty = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.5},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.5,
            "body_names": [
                "left_rubber_hand",
                "right_rubber_hand",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "mid360_link",
            ],
        },
    )
    box_pos = DoneTerm(
        func=mdp.bad_object_pos_z,
        params={"command_name": "motion", "threshold": 0.5},
    )


##
# Environment configuration
##
@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg | Any = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        # viewer settings
        self.viewer.eye = (4.0, 4.0, 4.0)       # Camera position (x, y, z); keep some distance.
        self.viewer.lookat = (0.0, 0.0, 0.0)    # Camera target (x, y, z).
        self.viewer.origin_type = "env"         # "env" makes coordinates relative to the environment origin.
        self.viewer.env_index = 0               # View environment 0.


from omnicontact.assets.robots.omnicontact_rubberhand import G1_ACTION_SCALE, G1_CYLINDER_CFG, G1_GHOST_CFG

@configclass
class BaseEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_rubber_hand",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_rubber_hand",
        ]
        self.commands.motion.ee_body_names = [
            "left_rubber_hand", 
            "right_rubber_hand",
            "left_ankle_pitch_link",
            "right_ankle_pitch_link",
            "mid360_link"
        ]


@configclass
class BaseEnvPlayCfg(BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.ghost_robot = G1_GHOST_CFG.replace(prim_path="{ENV_REGEX_NS}/GhostRobot")

        # Disable policy observation noise during playback.
        self.observations.policy_command.enable_corruption = False
        self.observations.policy_proprio.enable_corruption = False

        # Use the assets' configured physical properties without randomization.
        self.events.physics_material = None
        self.events.robot_joint_stiffness_and_damping = None
        self.events.torso_rpy_obs_bias = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.box_physics_material = None
        self.events.box_scale_mass = None

        # Reset directly to the reference state without pose or velocity noise.
        self.commands.motion.pose_range = {}
        self.commands.motion.object_pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.object_velocity_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)


@configclass
class CFBaseEnvCfg(BaseEnvCfg):
    observations: ObservationsCfg | Any = ObservationsCfg()
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy_command.command = ObsTerm(
            func=mdp.tracking_commands_contact_flow, 
            params={"command_name": "motion"}
        )
        self.observations.policy_proprio.motion_anchor_pos_b = None
        self.observations.policy_proprio.base_lin_vel = None
        self.observations.policy_proprio.bbox_local = ObsTerm(
            func=mdp.object_bbox_corners_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        self.observations.critic_command.command = ObsTerm(
            func=mdp.tracking_commands_contact_flow,
            params={"command_name": "motion"}
        )
        self.observations.critic_proprio.bbox_local = ObsTerm(
            func=mdp.object_bbox_corners_b,
            params={"command_name": "motion"},
        )


@configclass
class CFBaseEnvPlayCfg(BaseEnvPlayCfg):
    observations: ObservationsCfg | Any = ObservationsCfg()
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy_command.command = ObsTerm(
            func=mdp.tracking_commands_contact_flow, 
            params={"command_name": "motion"}
        )
        self.observations.policy_proprio.motion_anchor_pos_b = None
        self.observations.policy_proprio.base_lin_vel = None
        self.observations.policy_proprio.bbox_local = ObsTerm(
            func=mdp.object_bbox_corners_b,
            params={"command_name": "motion"},
        )

        self.observations.critic_command.command = ObsTerm(
            func=mdp.tracking_commands_contact_flow,
            params={"command_name": "motion"}
        )
        self.observations.critic_proprio.bbox_local = ObsTerm(
            func=mdp.object_bbox_corners_b,
            params={"command_name": "motion"},
        )
