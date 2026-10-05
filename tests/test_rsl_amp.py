"""CPU contract tests for the native PPO/AMP training and inference path."""

import copy
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.algorithms import AMPPPO
from rsl_rl.modules import ActorCritic, ActorCriticTransformer
from rsl_rl.modules.amp import AMPNormalizer
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage.amp_replay_buffer import AMPReplayBuffer
from omnicontact.utils.rsl_rl import OmniContactVecEnvWrapper, inference_runner_cfg, runner_and_wrapper


class ExpertDataset:
    history_length = 2
    observation_dims = [4, 2]
    observation_dim = 6

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            state = torch.randn(mini_batch_size, 6)
            yield state, state + 0.1


class SplitEnv:
    """Isaac Lab API double; env 0 resets every third step."""

    def __init__(self):
        self.unwrapped = self
        self.num_envs = 4
        self.device = "cpu"
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.action_manager = SimpleNamespace(total_action_dim=2)
        self.cfg = SimpleNamespace(is_finite_horizon=False)
        self.count = 0

    def observations(self):
        return {"policy_command": torch.ones(4, 2), "policy_proprio": torch.randn(4, 3),
                "critic_command": torch.full((4, 2), 2.0), "critic_proprio": torch.randn(4, 4),
                "amp": torch.randn(4, 6)}

    def reset(self):
        self.count = 0
        return self.observations(), {}

    def step(self, actions):
        self.last_actions = actions
        self.count += 1
        self.episode_length_buf += 1
        done = torch.tensor([self.count % 3 == 0, False, False, False])
        self.episode_length_buf[done] = 0
        return self.observations(), torch.ones(4), torch.zeros(4, dtype=torch.bool), done, {}


def configuration(amp=True, transformer=False):
    cfg = dict(num_steps_per_env=4, save_interval=1, empirical_normalization=True, logger="tensorboard",
               policy=dict(class_name="ActorCritic", actor_hidden_dims=[16], critic_hidden_dims=[16],
                           activation="elu", init_noise_std=0.5),
               algorithm=dict(class_name="PPO", num_learning_epochs=2, num_mini_batches=2,
                              learning_rate=1e-3, schedule="adaptive", desired_kl=0.01))
    if amp:
        cfg["algorithm"].update(class_name="AMPPPO", amp_data=ExpertDataset(), amp_discr_hidden_dims=[16],
                                amp_replay_buffer_size=32, amp_tracking_weight=0.9, amp_reward_coef=0.5)
    if transformer:
        cfg["policy"].update(class_name="ActorCriticTransformer", transformer_embed_dim=16,
                             transformer_num_heads=2, transformer_num_layers=1, transformer_ff_dim=32)
    return cfg


def make_algorithm(**kwargs):
    return AMPPPO(ActorCritic(5, 6, 2, actor_hidden_dims=[8], critic_hidden_dims=[8]),
                  amp_data=ExpertDataset(), amp_discr_hidden_dims=[8], **kwargs)


def test_wrapper_preserves_order_history_and_timeouts():
    base = SplitEnv()
    env = OmniContactVecEnvWrapper(base, clip_actions=0.5)
    obs, extras = env.get_observations()
    torch.testing.assert_close(obs[:, :2], torch.ones(4, 2))
    torch.testing.assert_close(extras["observations"]["critic"][:, :2], torch.full((4, 2), 2.0))
    assert env.get_observations()[0] is obs
    for _ in range(3):
        _, _, dones, info = env.step(torch.ones(4, 2))
    assert base.last_actions.max() == 0.5
    assert dones.tolist() == [1, 0, 0, 0]
    assert info["time_outs"].tolist() == [True, False, False, False]
    base.cfg.is_finite_horizon = True
    assert "time_outs" not in env.step(torch.ones(4, 2))[3]


def test_amp_history_layout_and_reward_formula():
    alg = make_algorithm(amp_normalize_obs=False, amp_tracking_weight=0.9, amp_reward_coef=0.5)
    state = torch.arange(6.0).expand(4, -1)
    following = state + 10
    torch.testing.assert_close(alg.discriminator_input(state, following)[0],
                               torch.tensor([0., 1., 2., 3., 4., 5., 12., 13., 15.]))
    for param in alg.discriminator.parameters():
        param.data.zero_()
    alg.discriminator.network[-1].bias.data.fill_(1)
    alg.init_storage("rl", 4, 1, [5], [6], [2])
    alg.set_amp_observations(state)
    with torch.inference_mode():
        alg.act(torch.ones(4, 5), torch.ones(4, 6))
        alg.process_env_step(torch.ones(4), torch.tensor([0, 1, 0, 0]), {"observations": {"amp": following}})
    torch.testing.assert_close(alg.storage.rewards[0, :, 0], torch.tensor([0.95, 0.9, 0.95, 0.95]))
    assert alg.amp_replay.size == 3
    torch.testing.assert_close(alg._amp_obs, following)


def test_replay_wrap_and_oversized_insert():
    replay = AMPReplayBuffer(1, 3)
    replay.insert(torch.arange(2.).view(-1, 1), torch.arange(2.).view(-1, 1) + 1)
    replay.insert(torch.arange(2., 7.).view(-1, 1), torch.arange(3., 8.).view(-1, 1))
    assert replay.size == 3
    assert sorted(replay.states.flatten().tolist()) == [4, 5, 6]
    states, following = replay.sample(20)
    torch.testing.assert_close(following, states + 1)


def test_normalizer_save_and_batch_moments():
    normalizer = AMPNormalizer(2)
    values = torch.tensor([[1., 2.], [3., 6.], [5., 10.]])
    normalizer.update(values[:2])
    normalizer.update(values[2:])
    torch.testing.assert_close(normalizer.mean, values.mean(0), atol=0.001, rtol=0.001)
    torch.testing.assert_close(normalizer.var, values.var(0, unbiased=False), atol=0.001, rtol=0.001)
    restored = AMPNormalizer(2)
    restored.load_state_dict(normalizer.state_dict())
    torch.testing.assert_close(restored(values), normalizer(values))


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("transformer", [False, True])
def test_runner_learn_resume_and_inference(tmp_path, amp, transformer):
    cfg = configuration(amp, transformer)
    original = copy.copy(cfg["algorithm"])
    runner = OnPolicyRunner(OmniContactVecEnvWrapper(SplitEnv()), cfg, str(tmp_path))
    if transformer:
        assert runner.alg.policy.actor.observation_dims == (2, 3)
        assert runner.alg.policy.critic.observation_dims == (2, 4)
    policy_before = next(runner.alg.policy.parameters()).detach().clone()
    if amp:
        discriminator_before = next(runner.alg.discriminator.parameters()).detach().clone()
    runner.learn(2)
    assert cfg["algorithm"] == original
    assert not torch.equal(policy_before, next(runner.alg.policy.parameters()))
    if amp:
        assert not torch.equal(discriminator_before, next(runner.alg.discriminator.parameters()))
        assert runner.alg.amp_normalizer.count > 1
        assert "amp_data" not in runner.cfg["algorithm"]
    checkpoint = tmp_path / "model_1.pt"
    assert checkpoint.exists()

    resumed = OnPolicyRunner(OmniContactVecEnvWrapper(SplitEnv()), configuration(amp, transformer))
    resumed.load(checkpoint)
    assert resumed.current_learning_iteration == 2
    if amp:
        torch.testing.assert_close(resumed.alg.amp_normalizer.mean, runner.alg.amp_normalizer.mean)
        assert resumed.alg.discriminator_optimizer.state_dict()["state"]
        for restored, saved in zip(resumed.alg.discriminator.parameters(), runner.alg.discriminator.parameters()):
            torch.testing.assert_close(restored, saved)
    resumed.learn(1)

    inference_cfg = inference_runner_cfg(SimpleNamespace(to_dict=lambda: configuration(amp, transformer)))
    inference = OnPolicyRunner(OmniContactVecEnvWrapper(SplitEnv()), inference_cfg)
    inference.load(checkpoint, load_optimizer=False)
    observations = torch.randn(4, 5)
    torch.testing.assert_close(inference.get_inference_policy()(observations), runner.get_inference_policy()(observations))
    assert all(torch.isfinite(p).all() for p in resumed.alg.policy.parameters())
    runner.writer.close()


def test_missing_amp_data_and_history_mismatch_fail():
    with pytest.raises(ValueError, match="expert dataset"):
        OnPolicyRunner(OmniContactVecEnvWrapper(SplitEnv()), {**configuration(), "algorithm": {"class_name": "AMPPPO"}})
    alg = make_algorithm()
    with pytest.raises(ValueError, match="shape"):
        alg.set_amp_observations(torch.zeros(4, 7))


def test_native_runner_selection_and_inference_config():
    cfg = SimpleNamespace(runner_class="OnPolicyRunner", to_dict=lambda: configuration())
    assert runner_and_wrapper(cfg) == (OnPolicyRunner, OmniContactVecEnvWrapper)
    inference = inference_runner_cfg(cfg)
    assert inference["algorithm"]["class_name"] == "PPO"
    assert not any(k.startswith("amp_") for k in inference["algorithm"])


@pytest.mark.parametrize("transformer", [False, True])
def test_motion_onnx_export_matches_actor_and_clamps_frame(tmp_path, transformer):
    import onnx
    from onnx.reference import ReferenceEvaluator
    from omnicontact.utils.exporter import export_motion_policy_as_onnx
    from rsl_rl.modules import EmpiricalNormalization

    motion = SimpleNamespace(**{key: torch.randn(3, width) for key, width in (
        ("joint_pos", 2), ("joint_vel", 2), ("body_pos_w", 3), ("body_quat_w", 4),
        ("body_lin_vel_w", 3), ("body_ang_vel_w", 3), ("object_pos_w", 3),
        ("object_quat_w", 4), ("contact_info", 4), ("table1_pos_w", 3),
    )})
    command = SimpleNamespace(motion=motion, cfg=SimpleNamespace(table2_name=None))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    if transformer:
        policy = ActorCriticTransformer(
            5, 6, 2, actor_obs_group_dims=(2, 3), critic_obs_group_dims=(2, 4),
            transformer_embed_dim=16, transformer_num_heads=2, transformer_num_layers=1, transformer_ff_dim=32,
        )
    else:
        policy = ActorCritic(5, 6, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
    normalizer = EmpiricalNormalization([5])
    normalizer(torch.randn(12, 5))
    normalizer.eval()
    export_motion_policy_as_onnx(env, policy, str(tmp_path), normalizer)
    model = onnx.load(tmp_path / "policy.onnx")
    onnx.checker.check_model(model)
    evaluate = ReferenceEvaluator(model)
    obs = torch.randn(1, 5)
    outputs = evaluate.run(None, {"obs": obs.numpy(), "time_step": torch.tensor([[100.]]).numpy()})
    torch.testing.assert_close(torch.from_numpy(outputs[0]), policy.act_inference(normalizer(obs)))
    torch.testing.assert_close(torch.from_numpy(outputs[1]), motion.joint_pos[-1:])
    assert "contact_info" in [output.name for output in model.graph.output]


def test_transformer_uses_each_group_and_preserves_ppo_train_eval_outputs():
    torch.manual_seed(7)
    policy = ActorCriticTransformer(
        5, 6, 2, actor_obs_group_dims=(2, 3), critic_obs_group_dims=(2, 4),
        transformer_embed_dim=16, transformer_num_heads=2, transformer_num_layers=2, transformer_ff_dim=32,
    )
    observations = torch.randn(4, 5, requires_grad=True)
    critic_obs = torch.randn(4, 6, requires_grad=True)
    output = policy.act_inference(observations)
    value = policy.evaluate(critic_obs)
    (output.square().sum() + value.square().sum()).backward()
    assert torch.count_nonzero(observations.grad) == observations.numel()
    assert torch.count_nonzero(critic_obs.grad) == critic_obs.numel()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in list(policy.actor.parameters()) + list(policy.critic.parameters()))
    policy.eval()
    torch.testing.assert_close(policy.act_inference(observations), output)
    torch.testing.assert_close(policy.evaluate(critic_obs), value)
    assert output.shape == (4, 2)
    assert value.shape == (4, 1)


def test_transformer_rejects_invalid_observation_layout_and_attention_size():
    with pytest.raises(ValueError, match="input widths"):
        ActorCriticTransformer(5, 6, 2, actor_obs_group_dims=(2, 4))
    with pytest.raises(ValueError, match="divisible"):
        ActorCriticTransformer(5, 6, 2, transformer_embed_dim=15, transformer_num_heads=2)


def _distributed_amp_worker(rank, rendezvous, result_dir):
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(rank)
        alg = make_algorithm(multi_gpu_cfg={"global_rank": rank, "world_size": 2},
                             num_learning_epochs=1, num_mini_batches=1)
        alg.init_storage("rl", 4, 2, [5], [6], [2])
        alg.broadcast_parameters()
        for iteration in range(2):
            alg.set_amp_observations(torch.randn(4, 6))
            with torch.inference_mode():
                for _ in range(2):
                    alg.act(torch.randn(4, 5), torch.randn(4, 6))
                    # First rollout: one rank has no valid AMP transitions. All
                    # ranks must skip discriminator collectives without hanging.
                    dones = torch.full((4,), int(iteration == 0 and rank == 1))
                    alg.process_env_step(torch.ones(4), dones, {"observations": {"amp": torch.randn(4, 6)}})
                alg.compute_returns(torch.randn(4, 6))
            alg.update()
        torch.save({"discriminator": alg.discriminator.state_dict(), "normalizer": alg.amp_normalizer.state_dict()},
                   Path(result_dir) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo backend unavailable")
def test_distributed_amp_synchronizes_gradients_and_moments(tmp_path):
    torch.multiprocessing.spawn(_distributed_amp_worker,
                                args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)), nprocs=2, join=True)
    first = torch.load(tmp_path / "rank_0.pt", weights_only=True)
    second = torch.load(tmp_path / "rank_1.pt", weights_only=True)
    for component in first:
        for key in first[component]:
            torch.testing.assert_close(first[component][key], second[component][key], atol=0, rtol=0)
    assert first["normalizer"]["count"] > 1
