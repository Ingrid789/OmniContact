import pickle
import unittest

import numpy as np

from carrywm.replay_v4 import VectorReplay, gather_reference


class VectorReplayTests(unittest.TestCase):
    def setUp(self):
        self.replay = VectorReplay(24, seed=7, num_envs=2)
        self.clips = [np.full((100, 111), 1000 * i, np.float32)
                      + np.arange(100, dtype=np.float32)[:, None] for i in range(2)]

    def append(self, tick, reset=(False, False)):
        marker = tick + 1000 * np.arange(2)
        def matrix(value, width):
            return np.broadcast_to(np.asarray(value)[:, None], (2, width))
        self.replay.append(matrix(marker, 84), matrix(marker + 1, 84), matrix(marker - 1, 29),
                           matrix(marker, 29), matrix(marker, 6), matrix(marker, 16),
                           matrix(marker, 141), np.full(2, tick), self.clips, np.asarray(reset))

    def test_windows_follow_one_world_and_episode_including_terminal_state(self):
        for t in range(7):
            self.append(t)
        self.assertFalse(self.replay.has_window(8))
        self.append(7, (False, True))
        self.assertEqual(self.replay.valid_windows(8), 1)
        states, actions, contacts, _ = self.replay.wm(8, 8)
        np.testing.assert_array_equal(states[:, :, 0], np.tile(np.arange(9), (8, 1)))
        np.testing.assert_array_equal(actions[:, :, 0], states[:, :-1, 0])
        np.testing.assert_array_equal(contacts[:, :, 0], actions[:, :, 0])
        # Post-checkpoint simulator reset must invalidate all cross-session windows.
        self.replay = pickle.loads(pickle.dumps(self.replay))
        self.replay.begin_new_episodes()
        self.append(8)
        self.assertEqual(self.replay.valid_windows(8), 1)

    def test_ring_wrap_keeps_actor_history_and_future_clip(self):
        for t in range(25):
            self.append(t, (t == 16, False))
        self.assertEqual(self.replay.size, 24)
        state, previous, _, history, raw, remaining = self.replay.actor(300, 32)
        for marker, prev, hist, goal, left in zip(state[:, 0], previous[:, 0], history[:, :, 0],
                                                  raw[:, :, 0], remaining):
            world, t = divmod(int(marker), 1000)
            first = 16 if world == 0 and t >= 16 else 0
            self.assertGreaterEqual(t, 13)
            self.assertEqual(prev, marker - 1)
            np.testing.assert_array_equal(hist, world * 1000 + np.maximum(t - np.arange(9, -1, -1), first))
            np.testing.assert_array_equal(goal, world * 1000 + np.minimum(99, t + np.arange(83)))
            self.assertEqual(left, 99 - t)

    def test_reference_gather_preserves_row_order_and_caps_at_own_end(self):
        a, b = self.clips[0][:3], self.clips[1][:2]
        result = gather_reference([b, a, b], np.array([0, 1, 1]), [0, 1, 50])
        np.testing.assert_array_equal(result[:, :, 0], [[1000, 1001, 1001], [1, 2, 2], [1001] * 3])


if __name__ == "__main__":
    unittest.main()
