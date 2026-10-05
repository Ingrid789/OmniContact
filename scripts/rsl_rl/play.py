"""Play an RSL-RL checkpoint and visualize its reference motion."""

import argparse
import glob
import re
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_name_prefix", type=str, default="rl-video", help="Prefix for the recorded video file.")
parser.add_argument("--start_frame", type=int, default=None, help="Start the reference motion at this zero-based frame.")
parser.add_argument("--object_usd_override", type=str, default=None, help="Use this USD for every spawned object in play.")
parser.add_argument("--stop_on_reset", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file_dir", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--registry_name", type=str, nargs='+', default=None,
                    help="One or More motions registry names from wandb.")
parser.add_argument("--disable_wandb", action="store_true", default=False, help="Load the checkpoint and motions locally.")
parser.add_argument("--log_root_dir", type=str, default=None, help="Root directory for logging if no wandb.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import pathlib
import torch

# Runners
from omnicontact.utils.rsl_rl import inference_runner_cfg, runner_and_wrapper

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import omnicontact.tasks  # noqa: F401
from omnicontact.assets.object_spawner import configure_box_assets_for_motion_files
from omnicontact.assets import asset_path
from omnicontact.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg


def _create_contact_visualizer(body_name: str, prim_path: str) -> VisualizationMarkers:
    """Create a reference mesh with yellow/no-contact and red/contact variants."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    usd_path = MeshConverter(
        MeshConverterCfg(
            asset_path=asset_path("unitree_description", "meshes", "g1", f"{body_name}.STL"),
            usd_dir=str(repo_root / "outputs" / "mesh_cache" / "g1_contact_markers" / body_name),
            usd_file_name=f"{body_name}.usd",
            collision_approximation="none",
        )
    ).usd_path
    visualizer = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                name: sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    scale=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=0.3),
                )
                for name, color in (("no_contact", (1.0, 1.0, 0.0)), ("contact", (1.0, 0.0, 0.0)))
            },
        )
    )
    visualizer.set_visibility(True)
    return visualizer


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.disable_wandb or args_cli.motion_file_dir is not None:
        if args_cli.motion_file_dir is not None:
            motion_dir = os.path.abspath(args_cli.motion_file_dir)
            npz_paths = [motion_dir] if os.path.isfile(motion_dir) and motion_dir.endswith(".npz") else glob.glob(os.path.join(motion_dir, "**", "*.npz"), recursive=True)
            env_cfg.commands.motion.motion_file_list = sorted({os.path.abspath(p) for p in npz_paths})
            if not env_cfg.commands.motion.motion_file_list:
                raise ValueError(f"No .npz files found under motion_file_dir (recursive): {motion_dir}")
        else:
            raise ValueError("--motion_file_dir must be specified when wandb is disabled.")
        # Isaac Lab's resolver joins the root with DirEntry paths, which already
        # include the root. An absolute path avoids duplicating a relative root.
        log_root_path = os.path.abspath(args_cli.log_root_dir or os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint and os.path.isfile(args_cli.checkpoint):
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    else:
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)

        if args_cli.wandb_path:
            import wandb

            run_path = args_cli.wandb_path

            api = wandb.Api()
            if "model" in args_cli.wandb_path:
                run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
            wandb_run = api.run(run_path)
            if "model" in args_cli.wandb_path:
                file = args_cli.wandb_path.split("/")[-1]
            else:
                files = [file.name for file in wandb_run.files() if "model" in file.name]
                file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

            wandb_file = wandb_run.file(str(file))
            wandb_file.download("./logs/rsl_rl/temp", replace=True)

            print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
            resume_path = f"./logs/rsl_rl/temp/{file}"

        else:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        if args_cli.registry_name is not None:
            import wandb

            api = wandb.Api()
            motion_file_paths = []
            for registry_name in args_cli.registry_name:
                if ":" not in registry_name:
                    registry_name += ":latest"
                artifact = api.artifact(registry_name)
                motion_file_path = str(pathlib.Path(artifact.download()) / "motion.npz")
                motion_file_paths.append(motion_file_path)
                print(f"[INFO] Downloaded motion file from {registry_name}: {motion_file_path}")

            env_cfg.commands.motion.motion_file_list = motion_file_paths
        elif args_cli.wandb_path:
            artifacts = [a for a in wandb_run.used_artifacts() if a.type == "motions"]
            if not artifacts:
                print("[WARN] No motion artifacts found in the run.")
            else:
                motion_file_paths = []
                for art in artifacts:
                    motion_file_path = str(pathlib.Path(art.download()) / "motion.npz")
                    motion_file_paths.append(motion_file_path)
                    print(f"[INFO] Downloaded motion file from artifact: {motion_file_path}")
                env_cfg.commands.motion.motion_file_list = motion_file_paths

    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "box") and hasattr(env_cfg, "commands"):
        configure_box_assets_for_motion_files(env_cfg)
        if args_cli.object_usd_override is not None:
            object_usd_override = os.path.abspath(args_cli.object_usd_override)
            if not os.path.isfile(object_usd_override):
                raise FileNotFoundError(f"Object USD override does not exist: {object_usd_override}")
            for object_asset_cfg in env_cfg.scene.box.spawn.assets_cfg:
                object_asset_cfg.usd_path = object_usd_override
            print(f"[INFO] Overriding spawned object USD with: {object_usd_override}")
    if args_cli.start_frame is not None:
        if args_cli.start_frame < 0:
            raise ValueError("--start_frame must be non-negative.")
        env_cfg.commands.motion.sampling_method = "fixed_frame"
        env_cfg.commands.motion.fixed_frame = args_cli.start_frame

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": args_cli.video_name_prefix,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    runner_class, wrapper_class = runner_and_wrapper(agent_cfg)
    env = wrapper_class(env, clip_actions=agent_cfg.clip_actions)
    ppo_runner = runner_class(env, inference_runner_cfg(agent_cfg), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False)
    policy_inference_func = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(log_dir, "exported")
    export_motion_policy_as_onnx(
        env.unwrapped, ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir, filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path or "none", export_model_dir)
    obs, _ = env.get_observations()

    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    ghost_robot = env.unwrapped.scene.articulations["ghost_robot"]
    ee_body_names = list(motion_cmd.cfg.ee_body_names)
    show_rubber_hand_contacts = all(
        body_name in ee_body_names for body_name in ("left_rubber_hand", "right_rubber_hand")
    )
    if show_rubber_hand_contacts:
        left_hand_index = ee_body_names.index("left_rubber_hand")
        right_hand_index = ee_body_names.index("right_rubber_hand")
        ref_left_hand_visualizer = _create_contact_visualizer("left_rubber_hand", "/Visuals/RefLeftHandContact")
        ref_right_hand_visualizer = _create_contact_visualizer("right_rubber_hand", "/Visuals/RefRightHandContact")

    body_names = list(motion_cmd.cfg.body_names)
    show_ankle_contacts = all(
        body_name in body_names for body_name in ("left_ankle_roll_link", "right_ankle_roll_link")
    )
    if show_ankle_contacts:
        left_ankle_roll_body_index = body_names.index("left_ankle_roll_link")
        right_ankle_roll_body_index = body_names.index("right_ankle_roll_link")
        ref_left_ankle_visualizer = _create_contact_visualizer("left_ankle_roll_link", "/Visuals/RefLeftAnkleContact")
        ref_right_ankle_visualizer = _create_contact_visualizer("right_ankle_roll_link", "/Visuals/RefRightAnkleContact")

    # Reference objects use visual-only primitives for boxes and balls.
    env_asset_labels = motion_cmd.env_asset_labels
    ghost_object_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), opacity=0.55)
    ghost_object_markers = {}
    ghost_label_to_marker_index = {}
    ghost_marker_indices = []
    for label in env_asset_labels:
        label = str(label)
        if label not in ghost_label_to_marker_index:
            sanitized_label = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_") or "asset"
            # USD identifiers must start with a letter or underscore. Object labels
            # commonly start with dimensions, for example "30-30-30".
            marker_name = f"object_{len(ghost_object_markers)}_{sanitized_label}"
            ghost_label_to_marker_index[label] = len(ghost_object_markers)
            geometry_label = label.lower().removeprefix("push_")
            box_size_match = re.fullmatch(r"(\d+)-(\d+)-(\d+)", geometry_label)
            ball_size_match = re.fullmatch(r"(?:ball|soccer)_?(\d+)", geometry_label)
            if box_size_match is not None:
                # Never use the full object USD as a marker prototype: imported
                # physics schemas can survive instancing even when collision is
                # overridden. A primitive with visual material only has no rigid
                # body and no collider by construction.
                box_size_m = tuple(float(value) / 100.0 for value in box_size_match.groups())
                ghost_object_markers[marker_name] = sim_utils.CuboidCfg(
                    size=box_size_m,
                    visual_material=ghost_object_material,
                )
            elif ball_size_match is not None:
                # Ball labels encode the diameter in centimeters. Use a visual-only
                # primitive: a referenced USD can retain instanced collision prims
                # even when collision properties are overridden on its root.
                ball_radius_m = float(ball_size_match.group(1)) / 200.0
                ghost_object_markers[marker_name] = sim_utils.SphereCfg(
                    radius=ball_radius_m,
                    visual_material=ghost_object_material,
                )
            else:
                asset_label = geometry_label if geometry_label.startswith(("ball", "soccer")) else label
                usd_path = (
                    os.path.abspath(args_cli.object_usd_override)
                    if args_cli.object_usd_override is not None
                    else asset_path("objects", asset_label, f"{asset_label}.usd")
                )
                if not os.path.isfile(usd_path):
                    raise FileNotFoundError(f"Missing ghost object USD for label '{label}': {usd_path}")
                ghost_object_markers[marker_name] = sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=ghost_object_material,
                )
        ghost_marker_indices.append(ghost_label_to_marker_index[label])

    ghost_object_marker_indices = torch.tensor(ghost_marker_indices, device=env.unwrapped.device, dtype=torch.long)
    ghost_object_visualizer = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GhostObject",
            markers=ghost_object_markers,
        )
    )
    ghost_object_visualizer.set_visibility(True)

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy_inference_func(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # Update ghost robot pose
            ref_joint_pos = motion_cmd.joint_pos.clone()
            ref_joint_vel = motion_cmd.joint_vel.clone()
            root_pos = motion_cmd.body_pos_w[:, 0].clone()
            root_ori = motion_cmd.body_quat_w[:, 0].clone()
            root_lin_vel = motion_cmd.body_lin_vel_w[:, 0].clone()
            root_ang_vel = motion_cmd.body_ang_vel_w[:, 0].clone()
            ghost_robot.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel)
            ghost_robot.write_root_state_to_sim(
                torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            )
            
            object_pos = motion_cmd.data_object_pos_w.clone()
            object_ori = motion_cmd.data_object_quat_w.clone()
            ghost_object_visualizer.visualize(object_pos, object_ori, marker_indices=ghost_object_marker_indices)

            contact_info = getattr(motion_cmd, "contact_info", None)
            if contact_info is not None:
                # contact_info order: (left_foot, right_foot, left_hand, right_hand).
                contact_info = contact_info.reshape(contact_info.shape[0], -1)

                if show_rubber_hand_contacts and contact_info.shape[1] >= 4:
                    left_hand_contact = (contact_info[:, 2] > 0.5).to(torch.int64)
                    right_hand_contact = (contact_info[:, 3] > 0.5).to(torch.int64)
                    ref_left_hand_visualizer.visualize(
                        translations=motion_cmd.data_ee_pos_w[:, left_hand_index],
                        orientations=motion_cmd.data_ee_quat_w[:, left_hand_index],
                        marker_indices=left_hand_contact,
                    )
                    ref_right_hand_visualizer.visualize(
                        translations=motion_cmd.data_ee_pos_w[:, right_hand_index],
                        orientations=motion_cmd.data_ee_quat_w[:, right_hand_index],
                        marker_indices=right_hand_contact,
                    )

                if show_ankle_contacts and contact_info.shape[1] >= 2:
                    left_foot_contact = (contact_info[:, 0] > 0.5).to(torch.int64)
                    right_foot_contact = (contact_info[:, 1] > 0.5).to(torch.int64)
                    ref_left_ankle_visualizer.visualize(
                        translations=motion_cmd.body_pos_w[:, left_ankle_roll_body_index],
                        orientations=motion_cmd.body_quat_w[:, left_ankle_roll_body_index],
                        marker_indices=left_foot_contact,
                    )
                    ref_right_ankle_visualizer.visualize(
                        translations=motion_cmd.body_pos_w[:, right_ankle_roll_body_index],
                        orientations=motion_cmd.body_quat_w[:, right_ankle_roll_body_index],
                        marker_indices=right_foot_contact,
                    )

        timestep += 1
        # Ignore the initialization reset reported by the first simulation step.
        if args_cli.stop_on_reset and timestep > 1 and bool(torch.any(dones).item()):
            print(f"[INFO] Stopping at step {timestep}: environment reset.")
            break
        if args_cli.video:
            if timestep % 10 == 0:
                print("[INFO] Video recording at step:", timestep)
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
