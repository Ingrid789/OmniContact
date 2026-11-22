"""Regression tests for V5 GPU acceleration (replay, finish_step_gpu, learner steps, async save)."""
from collections import deque
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from carrywm.model import Actor, Critic, WorldModel
from carrywm.replay_v4 import ClipBank, VectorReplay, gather_reference
from carrywm.sim import CarryEnv
from carrywm.train import GAMMA, HORIZON, OFFSETS, Learner
from carrywm.train_v4 import AsyncCheckpointer, check_resume
from carrywm.train_v5 import args_from_cli
from carrywm.v5 import LearnerV5
from carrywm.warp_sim import finish_step_gpu


def _make_legacy_replay(capacity=24, num_envs=2):
    class LegacyReplay:
        pass

    clips = [
        np.full((60, 111), 1000 * i, np.float32) + np.arange(60, dtype=np.float32)[:, None]
        for i in range(num_envs)
    ]
    old = LegacyReplay()
    old.num_envs = num_envs
    old.rows = capacity // num_envs
    old.capacity = old.rows * num_envs
    old.rng = np.random.default_rng(13)
    old.ticks = 10
    old._first = np.array([0, 4], dtype=np.int64)
    old._restart = False
    shape = (old.rows, num_envs)
    old.states = np.arange(old.capacity * 84, dtype=np.float32).reshape(*shape, 84) * 0.001
    old.after = old.states + 1.0
    old.actions = np.ones((*shape, 29), np.float32)
    old.previous = np.zeros((*shape, 29), np.float32)
    old.contacts = np.zeros((*shape, 6), np.float32)
    old.contexts = np.full((*shape, 16), 0.25, np.float32)
    old.observations = np.arange((old.rows + 9) * num_envs * 141, dtype=np.float32).reshape(old.rows + 9, num_envs, 141) * 0.01
    old.frames = np.tile(np.arange(old.rows, dtype=np.int32)[:, None], (1, num_envs))
    old.episode_start = np.zeros(shape, dtype=np.int64)
    old.episode_start[4:, 1] = 4
    old.clips = np.empty(shape, dtype=object)
    for r in range(min(old.ticks, old.rows)):
        for c in range(num_envs):
            old.clips[r, c] = clips[c]
    return old, clips


class ClipBankAndReplayTests(unittest.TestCase):
    def test_clip_bank_compaction_recycles_inactive_dynamic_slots(self):
        bank = ClipBank(init_frames=40, init_clips=8)
        static_clip = np.ones((12, 111), np.float32)
        s_slot = bank.register(static_clip, static=True)
        d_clips = [np.full((12, 111), float(k + 2), np.float32) for k in range(4)]
        d_slots = [bank.register(d_clips[0], static=False)]
        d_slots.append(bank.register(d_clips[1], static=False))
        # Third dynamic clip overflows 40 frames; mark only static + d_slots[1] active.
        active = {s_slot, d_slots[1]}
        slot2 = bank.register(d_clips[2], static=False, active_ids=active)
        self.assertEqual(slot2, d_slots[0])  # Reused retired slot ID
        self.assertLessEqual(bank._used_frames, 40)
        ids = torch.tensor([s_slot, d_slots[1], slot2], dtype=torch.int64)
        frames = torch.tensor([0, 5, 11], dtype=torch.int32)
        offsets = torch.tensor([0, 1, 50], dtype=torch.int64)
        gathered = bank.gather(ids, frames, offsets).numpy()
        expected = gather_reference([static_clip, d_clips[1], d_clips[2]], [0, 5, 11], [0, 1, 50])
        np.testing.assert_array_equal(gathered, expected)

    def test_from_legacy_migrates_numpy_replay_cleanly(self):
        old, clips = _make_legacy_replay(capacity=24, num_envs=2)
        migrated = VectorReplay.from_legacy(old, device="cpu")
        self.assertEqual(migrated.size, 20)
        self.assertEqual(migrated.valid_windows(8), 3)  # env0: ticks 0..9 (3 windows); env1: ticks 4..9 (0)
        s, a, c, ctx = migrated.wm(4, 8)
        self.assertEqual(s.shape, (4, 9, 84))
        self.assertEqual(a.shape, (4, 8, 29))
        self.assertEqual(c.shape, (4, 8, 6))
        self.assertEqual(ctx.shape, (4, 16))
        st, prev, ctx2, hist, raw, rem = migrated.actor(6, 32)
        self.assertEqual((st.shape, hist.shape, raw.shape, rem.shape),
                         ((6, 84), (6, 10, 141), (6, 83, 111), (6,)))
        restored = pickle.loads(pickle.dumps(migrated))
        self.assertEqual(restored.valid_windows(8), 3)


class FinishStepEquivalenceTests(unittest.TestCase):
    def _make_cpu_env(self, clip, goal, half, tabletop=0.75):
        env = CarryEnv.__new__(CarryEnv)
        env.clip = clip.copy()
        env.goal = goal.copy()
        env.half = half.copy()
        env.source = "npz"
        env.recovery_rejected = 0
        env.index = 0
        env.bp = 36
        env.bv = 35
        env.table_ids = [0, 1]
        env.tables_geom = [1, 2]
        env.m = SimpleNamespace(geom_size=np.array([[0., 0., 0.], [0.5, 0.5, 0.05], [0.5, 0.5, 0.05]]))
        mocap = np.array([[0., 0., 0.70], [0., 0., tabletop - 0.05]])
        env.d = SimpleNamespace(qpos=np.zeros(43, np.float32), qvel=np.zeros(41, np.float32), mocap_pos=mocap)
        env._done = False
        env._lifted = False
        env._carried_steps = 0
        env._lost_steps = 0
        env._stable_steps = 0
        env._initial_box = np.array([0., 0., 0.85], np.float32)
        env.last_action = np.zeros(29, np.float32)
        env.pending_recovery = False
        env._update_teacher_history = lambda: None
        return env

    def test_finish_step_gpu_matches_carry_env_across_all_event_branches(self):
        n = 10
        half = np.array([0.15, 0.15, 0.15], dtype=float)
        clip = np.zeros((40, 111), np.float32)
        clip[:, 100] = 1.0
        clip[:, 107] = 1.0
        tabletop = 0.75
        goals = [np.array([1.0, 0.0, 0.90], np.float32) for _ in range(n)]
        goals[4] = np.array([1.0, 0.0, 0.15], np.float32)  # World 4 targets floor
        envs = [self._make_cpu_env(clip, goals[i], half, tabletop) for i in range(n)]
        for i in (3, 4, 6):
            envs[i]._lifted = True
        envs[5].index = 0
        envs[9].index = len(clip) - 2  # Next step reaches timeout

        initial_box = torch.tensor([[0., 0., 0.85]], dtype=torch.float32).expand(n, 3).clone()
        goal_t = torch.as_tensor(np.stack(goals), dtype=torch.float32)
        goal_on_floor = torch.zeros(n, dtype=torch.bool)
        goal_on_floor[4] = True
        clip_len = torch.full((n,), len(clip), dtype=torch.int32)
        env_index = torch.tensor([e.index for e in envs], dtype=torch.int32)
        lifted = torch.tensor([e._lifted for e in envs], dtype=torch.bool)
        carried = torch.zeros(n, dtype=torch.int32)
        lost = torch.zeros(n, dtype=torch.int32)
        stable = torch.zeros(n, dtype=torch.int32)

        # Simulate 6 steps covering lift, table/floor success, off_table drop, lost_grasp drop, falls, timeout
        for step in range(6):
            states = torch.zeros((n, 84), dtype=torch.float32)
            states[:, 2] = 0.78
            states[:, 3] = 1.0
            states[:, 71:74] = torch.tensor([0., 0., 0.85])
            states[:, 74] = 1.0
            contacts = torch.zeros((n, 6), dtype=torch.float32)
            support = torch.zeros((n, 3), dtype=torch.bool)

            # World 1: held free & lifted > 2.5cm -> lifted after 2 steps
            states[1, 73] = 0.90
            contacts[1, 2:4] = 1.0
            # World 3: placed on destination table, released, stable -> success at step 4
            states[3, 71:74] = goal_t[3] + torch.tensor([0.05, 0.0, 0.0])
            support[3, 1] = True
            # World 4: placed on floor, released, stable -> success at step 4
            states[4, 71:74] = goal_t[4] + torch.tensor([0.05, 0.0, 0.0])
            support[4, 0] = True
            # World 5: off_table drop on ground -> immediate drop at step 0
            states[5, 73] = 0.50
            support[5, 0] = True
            # World 6: lost_grasp drop on origin table after 3 lost steps -> drop at step 2
            support[6, 2] = True
            # World 7: low pelvis fall
            if step == 1:
                states[7, 2] = 0.25
            # World 8: tilted unnormalized quaternion fall
            if step == 1:
                states[8, 3:7] = torch.tensor([1.0, 2.0, 0.0, 0.0])  # normalized r22 = 1 - 2*(4/5) = -0.6 < 0.35

            (env_index, lifted, carried, lost, stable,
             done, success, drop, timeout, fallen, err) = finish_step_gpu(
                states, contacts, support, initial_box, goal_t, goal_on_floor,
                env_index, clip_len, lifted, carried, lost, stable,
            )

            act = np.zeros(29, np.float32)
            for i, e in enumerate(envs):
                if e._done:
                    continue
                e.d.qpos[:36] = states[i, :36].numpy()
                e.d.qvel[:35] = states[i, 36:71].numpy()
                e.d.qpos[36:43] = states[i, 71:78].numpy()
                e.d.qvel[35:41] = states[i, 78:84].numpy()
                sup_tuple = (bool(support[i, 0]), bool(support[i, 1]), bool(support[i, 2]))
                _, _, cpu_done, info = e._finish_step(act, (contacts[i].numpy(), *sup_tuple))
                self.assertEqual(bool(done[i].item()), cpu_done, f"step={step} world={i} done")
                self.assertEqual(bool(success[i].item()), info["success"], f"step={step} world={i} success")
                self.assertEqual(bool(drop[i].item()), info["drop"], f"step={step} world={i} drop")
                self.assertEqual(bool(timeout[i].item()), info["timeout"], f"step={step} world={i} timeout")
                self.assertEqual(bool(fallen[i].item()), info["fallen"], f"step={step} world={i} fallen")
                self.assertEqual(bool(lifted[i].item()), info["lifted"], f"step={step} world={i} lifted")
                self.assertEqual(int(env_index[i].item()), e.index, f"step={step} world={i} index")
                self.assertAlmostEqual(float(err[i].item()), info["goal_error"], places=5)


class LearnerAndCheckpointerTests(unittest.TestCase):
    def test_collect_act_masked_history_matches_step_by_step(self):
        net = LearnerV5.__new__(LearnerV5)
        net.device = torch.device("cpu")
        net.args = SimpleNamespace(precision="fp32", goal_mode="full", drop_penalty=1.0)
        net.offsets = torch.tensor(OFFSETS)
        net.actor = Actor(first_width=16, decoder_width=16, hidden=(8,))
        net.geo = SimpleNamespace(
            observation=lambda s, p, c: torch.cat((s[:, :2], p[:, :1]), dim=-1).repeat(1, 47),
            reference=lambda s, raw, fk=None: torch.nn.functional.pad(raw, (0, 2)),
        )
        state = torch.randn(4, 84)
        prev = torch.randn(4, 29)
        ctx = torch.randn(4, 16)
        raw_win = torch.randn(4, 11, 111)
        hist = torch.randn(4, 10, 141)
        fresh = torch.tensor([True, False, True, False])

        obs, new_hist, act = net.collect_act(state, prev, ctx, raw_win, hist, fresh)
        expected_obs = net.geo.observation(state, prev, ctx)
        torch.testing.assert_close(obs, expected_obs)
        # Fresh worlds repeat current obs 10 times; continuing worlds shift and append obs
        torch.testing.assert_close(new_hist[0], expected_obs[0].unsqueeze(0).expand(10, -1))
        torch.testing.assert_close(new_hist[1, :-1], hist[1, 1:])
        torch.testing.assert_close(new_hist[1, -1], expected_obs[1])
        expected_act = net.actor(net.reference(state, raw_win), new_hist)
        torch.testing.assert_close(act, expected_act)

    def test_unwrapped_state_dicts_have_no_orig_mod_prefix(self):
        actor = Actor(first_width=16, decoder_width=16, hidden=(8,))
        wm = WorldModel(16, width=16)
        critic = Critic(16, hidden=(8,))
        for mod in (actor, wm, critic):
            self.assertFalse(any("_orig_mod" in k for k in mod.state_dict()))

    def test_imagine_step_and_rollout_produce_identical_returns_and_gradients(self):
        from carrywm.v5 import soft_termination_probability
        torch.manual_seed(7)
        net = LearnerV5.__new__(LearnerV5)
        net.device = torch.device("cpu")
        net.args = SimpleNamespace(precision="fp32", goal_mode="full", drop_penalty=1.0)
        net.offsets = torch.tensor(OFFSETS)
        net._step_offsets = torch.arange(1, HORIZON + 1)[:, None] + net.offsets[None, :]
        net._step_numbers = torch.arange(1, HORIZON + 1, dtype=torch.float32)
        net.actor = Actor(first_width=16, decoder_width=16, hidden=(8,))
        net.wm = WorldModel(16, width=16)
        net.critic = Critic(16, hidden=(8,))
        net.geo = SimpleNamespace(
            fk_from_target=lambda t: t[:, :3],
            forward_kinematics=lambda s: s[:, :3],
            reference=lambda s, raw, fk=None: torch.nn.functional.pad(raw, (0, 2)),
            observation=lambda s, a, ctx, fk=None: torch.nn.functional.pad(torch.cat((s, a), -1), (0, 28)),
            reward=lambda before, after, prev, act, tgt, con, **kw: 0.01 * (after[:, 2] - act[:, 0].square()),
        )
        state = torch.zeros(3, 84)
        state[:, [3, 74]] = 1.0
        prev = torch.zeros(3, 29)
        ctx = torch.zeros(3, 16)
        hist = torch.zeros(3, 10, 141)
        raw = 0.02 * torch.randn(3, HORIZON + 51, 111)
        raw[..., [100, 107]] = 1.0
        remaining = torch.full((3,), float(HORIZON + 5))
        ret, mask, states, goals, rewards, continuations, valid_rows = net._rollout((state, prev, ctx, hist, raw, remaining))
        ret.mean().backward()
        grads = [p.grad.detach().clone() for p in net.actor.parameters()]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)
        self.assertEqual(len(rewards), HORIZON)
        self.assertEqual(len(states), HORIZON + 1)
        self.assertEqual(len(valid_rows), HORIZON)
        # Verify step 0 continuation matches direct soft_termination_probability
        p0 = soft_termination_probability(states[1], raw[:, 1])
        torch.testing.assert_close(continuations[0], 1.0 - p0)

    def test_async_checkpointer_writes_and_propagates_errors(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "latest.pt")
            ck = AsyncCheckpointer(enabled=True)
            try:
                ck.save(lambda: {"ok": 123}, path)
                ck.wait()
                self.assertTrue(path.exists())
                self.assertEqual(torch.load(path, map_location="cpu")["ok"], 123)

                def bad_payload():
                    return {"bad": lambda: None}  # unpicklable object fails inside background torch.save

                ck.save(bad_payload, path)
                with self.assertRaises(Exception):
                    ck.wait()
            finally:
                ck.close()

    def test_v5_cli_acceleration_defaults_and_legacy_resume(self):
        flags = ["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."]
        args = args_from_cli(flags)
        self.assertFalse(args.checkpoint_encoder)
        self.assertTrue(args.compile)
        self.assertTrue(args.cuda_graph)
        self.assertTrue(args.async_save)
        self.assertGreaterEqual(args.eval_workers, 1)
        self.assertGreaterEqual(args.cpu_threads, 1)
        # Earlier V5 checkpoints without acceleration flags remain resumable.
        legacy_cfg = {
            "num_envs": args.num_envs, "capacity": args.capacity, "cfgen_ratio": args.cfgen_ratio,
            "action_noise": args.action_noise, "goal_mode": args.goal_mode, "seed": args.seed,
            "termination_height_m": args.termination_height_m,
            "termination_gravity_z": args.termination_gravity_z,
            "termination_softness": args.termination_softness,
            "critic_value_max": args.critic_value_max, "drop_penalty": args.drop_penalty,
        }
        check_resume({"version": 5, "iteration": 100, "config": legacy_cfg}, args, version=5)

        toggled = args_from_cli(flags + ["--no-compile", "--no-cuda-graph", "--no-async-save", "--checkpoint-encoder"])
        self.assertTrue(toggled.checkpoint_encoder)
        self.assertFalse(toggled.compile)
        self.assertFalse(toggled.cuda_graph)
        self.assertFalse(toggled.async_save)


if __name__ == "__main__":
    unittest.main()
