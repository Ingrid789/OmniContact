import unittest

from carrywm import train_v4


class UpdateClockTests(unittest.TestCase):
    def test_schedule_repeats_without_drift(self):
        clock = train_v4.UpdateClock(1024)
        for _ in range(2000):
            self.assertEqual([clock.tick() for _ in range(5)], [0, 0, 0, 0, 1])
        self.assertEqual(clock.credit, 0)
        self.assertEqual(train_v4.WM_SAMPLES // (5 * 1024), 16)

    def test_no_noise_branch_policy_and_physics_batches(self):
        flags = ['--sim-root', '.', '--motion-dir', '.', '--output-dir', '.']
        args = train_v4.args_from_cli(flags)
        self.assertEqual((args.num_envs, args.collect_steps, args.wm_batch_size), (1024, 5, 1024))
        self.assertEqual((args.batch_size, args.actor_microbatch, args.critic_chunk), (2048, 2048, 16384))
        self.assertEqual((args.cfgen_goal_noise_std_m, args.cfgen_ratio, args.action_noise), (0., .1, .1))
        self.assertEqual(args.warmup_steps, 8192)
        for invalid in (['--wm-batch-size', '2048'], ['--batch-size', '4096'], ['--cfgen-goal-noise-std-m', '.005']):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                train_v4.args_from_cli(flags + invalid)
