"""Expert history contract using real Isaac Lab math/config helpers, without Sim."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def isaac_config_modules(monkeypatch):
    pytest.importorskip("isaaclab.utils")
    # Isaac math imports this logger but the tested math functions never use it.
    monkeypatch.setitem(sys.modules, "omni.log", types.ModuleType("omni.log"))
    spec = importlib.util.find_spec("isaaclab_rl")
    if spec is None:
        pytest.skip("Isaac Lab RL configuration package is not installed")
    # Load actual config classes without the package's simulator wrapper import.
    package = types.ModuleType("isaaclab_rl.rsl_rl")
    package.__path__ = [str(Path(next(iter(spec.submodule_search_locations))) / "rsl_rl")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    config = importlib.import_module("isaaclab_rl.rsl_rl.rl_cfg")
    for name in ("RslRlOnPolicyRunnerCfg", "RslRlPpoActorCriticCfg", "RslRlPpoAlgorithmCfg"):
        setattr(package, name, getattr(config, name))


def test_real_config_and_gym_registration(isaac_config_modules):
    import gymnasium as gym
    from omnicontact.tasks.omnicontact.agents.amp_rsl_rl_cfg import AMPCarryBoxRunnerCfg
    from omnicontact.utils.rsl_rl import inference_runner_cfg

    cfg = AMPCarryBoxRunnerCfg()
    cfg.validate()
    assert cfg.to_dict()["algorithm"]["class_name"] == "AMPPPO"
    assert inference_runner_cfg(cfg)["algorithm"]["class_name"] == "PPO"
    assert "amp_rsl_rl_cfg" in gym.spec("OmniContact-AMP").kwargs["rsl_rl_cfg_entry_point"]
    assert {task for task in gym.registry if task.startswith("OmniContact")} == {
        "OmniContact", "OmniContact-play", "OmniContact-AMP", "OmniContact-AMP-play",
        "OmniContact-Transformer", "OmniContact-Transformer-play",
        "OmniContact-AMP-Transformer", "OmniContact-AMP-Transformer-play",
    }
    for task in ("OmniContact-Transformer", "OmniContact-Transformer-play",
                 "OmniContact-AMP-Transformer", "OmniContact-AMP-Transformer-play"):
        module, name = gym.spec(task).kwargs["rsl_rl_cfg_entry_point"].split(":")
        transformer_cfg = getattr(importlib.import_module(module), name)()
        transformer_cfg.validate()
        assert transformer_cfg.policy.class_name == "ActorCriticTransformer"
        assert transformer_cfg.algorithm.class_name == ("AMPPPO" if "AMP" in task else "PPO")
        assert inference_runner_cfg(transformer_cfg)["policy"]["class_name"] == "ActorCriticTransformer"


def test_npz_expert_history_does_not_cross_motion_boundaries(isaac_config_modules, tmp_path):
    from omnicontact.tasks.omnicontact.amp_motion_dataset import AMPCarryBoxMotionDataset, AMPCarryBoxMotionDatasetCfg

    files = []
    for index, offset in enumerate((0, 100)):
        frames = 3
        xyz = np.zeros((frames, 1, 3), dtype=np.float32)
        quat = np.zeros((frames, 1, 4), dtype=np.float32)
        quat[..., 0] = 1
        path = tmp_path / f"motion_{index}.npz"
        np.savez(path, joint_pos=np.arange(offset, offset + frames, dtype=np.float32).reshape(-1, 1),
                 body_pos_w=xyz, body_quat_w=quat, body_lin_vel_w=xyz, body_ang_vel_w=xyz,
                 ee_pos_w=xyz, ee_quat_w=quat, object_pos_w=xyz[:, 0], object_quat_w=quat[:, 0],
                 contact_info=np.zeros((frames, 4), dtype=np.float32))
        files.append(str(path))
    robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros(1, 1)))
    cfg = AMPCarryBoxMotionDatasetCfg(motion_files=files, body_names=["pelvis"], anchor_name="pelvis",
                                    amp_obs_terms=["joint_pos", "projected_gravity"], history_length=2)
    dataset = AMPCarryBoxMotionDataset(cfg, SimpleNamespace(scene={"robot": robot}))
    assert dataset.index_t.tolist() == [0, 1, 3, 4]
    assert dataset.index_tp1.tolist() == [1, 2, 4, 5]
    state, following = dataset.build_transition(torch.tensor([0, 1, 3]), torch.tensor([1, 2, 4]))
    torch.testing.assert_close(state[:, :2], torch.tensor([[0., 0.], [0., 1.], [100., 100.]]))
    torch.testing.assert_close(following[:, :2], torch.tensor([[0., 1.], [1., 2.], [100., 101.]]))
    torch.testing.assert_close(state[:, 2:], torch.tensor([[0., 0., -1., 0., 0., -1.]]).expand(3, -1))
    assert dataset.observation_dims == [2, 6]
