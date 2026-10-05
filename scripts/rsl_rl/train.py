# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, nargs='+', default=None,
                    help="The name(s) of the wandb registry. Can specify multiple registry names for multiple motions.")
parser.add_argument("--disable_wandb", action="store_true", default=False, help="Use wandb?")
parser.add_argument("--motion_file_dir", type=str, required=False, help="The name of the wand registry.")
parser.add_argument(
    "--amp_data_dir",
    type=str,
    default=None,
    required=False,
    help="Directory containing .npz files for AMP expert observations (scanned recursively).",
)
parser.add_argument("--resume_root_dir", type=str, default=None, help="Resume directory.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")

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

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
import glob

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from omnicontact.assets.object_spawner import configure_box_assets_for_motion_files

# Runners
import omnicontact.tasks  # noqa: F401
from omnicontact.utils.rsl_rl import runner_and_wrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _find_npz_files(data_dir: str) -> list[str]:
    data_dir_abs = os.path.abspath(data_dir)
    if os.path.isfile(data_dir_abs):
        return [data_dir_abs] if data_dir_abs.endswith(".npz") else []
    if not os.path.isdir(data_dir_abs):
        return []
    npz_paths = glob.glob(os.path.join(data_dir_abs, "**", "*.npz"), recursive=True)
    return sorted({os.path.abspath(path) for path in npz_paths})


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    if args_cli.disable_wandb:
        agent_cfg.logger = "tensorboard"

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    if args_cli.motion_file_dir:
        env_cfg.commands.motion.motion_file_list = _find_npz_files(args_cli.motion_file_dir)
    elif args_cli.registry_name and not args_cli.disable_wandb:
        import pathlib
        import wandb

        api = wandb.Api()
        env_cfg.commands.motion.motion_file_list = [
            str(pathlib.Path(api.artifact(name if ":" in name else name + ":latest").download()) / "motion.npz")
            for name in args_cli.registry_name
        ]
    if not env_cfg.commands.motion.motion_file_list:
        raise ValueError("Provide --motion_file_dir or --registry_name with valid motion files.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "box") and hasattr(env_cfg, "commands"):
        configure_box_assets_for_motion_files(env_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    runner_class, wrapper_class = runner_and_wrapper(agent_cfg)
    algo_cfg = agent_cfg.algorithm
    algo_class_name = algo_cfg.get("class_name") if isinstance(algo_cfg, dict) else algo_cfg.class_name
    is_amp_algorithm = algo_class_name == "AMPPPO"
    env = wrapper_class(env, clip_actions=agent_cfg.clip_actions)

    runner_cfg_dict = agent_cfg.to_dict()
    if is_amp_algorithm:
        amp_data_cfg = getattr(agent_cfg, "amp_data", None)
        if amp_data_cfg is None:
            raise ValueError("AMP algorithm requires `agent_cfg.amp_data`.")
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "amp"):
            env_amp_hist = getattr(env_cfg.observations.amp, "history_length", None)
            if env_amp_hist is not None:
                amp_data_cfg.history_length = int(env_amp_hist)

        if args_cli.amp_data_dir:
            amp_data_dir = os.path.abspath(args_cli.amp_data_dir)
            if not os.path.exists(amp_data_dir):
                raise ValueError(f"AMP data directory does not exist: {amp_data_dir}")
            amp_motion_files = _find_npz_files(amp_data_dir)
            if len(amp_motion_files) == 0:
                raise ValueError(
                    f"No .npz files found under AMP data directory (recursive search): {amp_data_dir}"
                )
            amp_data_cfg.motion_files = amp_motion_files
            print(f"[INFO] Using AMP expert obs files from --amp_data_dir: {amp_data_dir}")
            print(f"[INFO] AMP dataset motion file count: {len(amp_motion_files)}")

        # If not set in cfg, use the motion files selected by CLI/env cfg.
        if len(getattr(amp_data_cfg, "motion_files", [])) == 0:
            amp_data_cfg.motion_files = list(getattr(env_cfg.commands.motion, "motion_file_list", []))
        if len(getattr(amp_data_cfg, "motion_files", [])) == 0:
            raise ValueError("AMP algorithm requires non-empty `amp_data.motion_files`.")

        from rsl_rl.algorithms import AMPPPO  # fail early if the vendored package is not installed

        # Build expert dataset object once and pass it to algorithm kwargs.
        amp_data = amp_data_cfg.class_type(amp_data_cfg, env.unwrapped, device=agent_cfg.device)
        if list(env.unwrapped.observation_manager.active_terms["amp"]) != amp_data.observation_terms:
            raise ValueError("AMP expert observation terms/order do not match the environment AMP group.")
        runner_cfg_dict.pop("amp_data", None)
        runner_cfg_dict["algorithm"]["amp_data"] = amp_data
        runner_cfg_dict["algorithm"]["amp_reward_coef"] = float(getattr(agent_cfg, "amp_reward_coef", 0.5))
        runner_cfg_dict["algorithm"]["amp_tracking_weight"] = float(getattr(agent_cfg, "amp_tracking_weight", 0.7))
        runner_cfg_dict["algorithm"]["amp_discr_hidden_dims"] = list(
            getattr(agent_cfg, "amp_discr_hidden_dims", [256, 256])
        )
        runner_cfg_dict["algorithm"]["amp_obs_key"] = "amp"
        runner_cfg_dict["algorithm"]["amp_replay_buffer_size"] = int(
            runner_cfg_dict["algorithm"].get("amp_replay_buffer_size", 100000)
        )
    elif args_cli.amp_data_dir:
        print("[Warning] --amp_data_dir is ignored because current algorithm is not AMP.")

    # create runner from rsl-rl
    if getattr(agent_cfg, "runner_class", "") == "MotionOnPolicyRunner":
        runner = runner_class(env, runner_cfg_dict, log_dir=log_dir, device=agent_cfg.device,
                              registry_name=args_cli.registry_name)
    else:
        runner = runner_class(env, runner_cfg_dict, log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    # TODO(maji): auto load
    if agent_cfg.resume:
        # get path to previous checkpoint
        if args_cli.resume_root_dir is not None:
            resume_path = get_checkpoint_path(args_cli.resume_root_dir, agent_cfg.load_run, agent_cfg.load_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
