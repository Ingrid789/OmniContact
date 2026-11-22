"""Uniform replay of real transitions from a vector of independent physics worlds."""
import numpy as np


def gather_reference(clips, frames, offsets):
    """Gather different-length shared CPU clips without allocating padding for all worlds."""
    frames = np.asarray(frames)
    times = frames[:, None] + np.asarray(offsets)[None, :]
    groups = {}
    for row, clip in enumerate(clips):
        groups.setdefault(id(clip), []).append(row)
    raw = np.empty((*times.shape, clips[0].shape[-1]), np.float32)
    for rows in groups.values():
        clip = clips[rows[0]]
        raw[rows] = clip[np.minimum(times[rows], len(clip) - 1)]
    return raw


class VectorReplay:
    """Time-major ring. A terminal transition retains its actual final state.

    Store nine extra observation rows so actor history remains exact at ring eviction.
    Raw reference arrays are immutable and shared between all rows of their episode.
    """

    def __init__(self, capacity, seed, num_envs):
        if num_envs < 1 or capacity < num_envs:
            raise ValueError("Replay capacity must fit at least one complete vector tick")
        self.num_envs = num_envs
        self.rows = capacity // num_envs
        self.capacity = self.rows * num_envs
        self.rng = np.random.default_rng(seed)
        self.ticks = 0
        self._first = np.zeros(num_envs, dtype=np.int64)
        self._restart = True
        self._windows = {}

        shape = (self.rows, num_envs)
        # Zero unused pages so partially populated replay checkpoints are deterministic.
        def field(width):
            return np.zeros((*shape, width), np.float32)

        self.states = field(84)
        self.after = field(84)
        self.actions = field(29)
        self.previous = field(29)
        self.contacts = field(6)
        self.contexts = field(16)
        self.observations = np.zeros((self.rows + 9, num_envs, 141), np.float32)
        self.frames = np.zeros(shape, np.int32)
        self.episode_start = np.zeros(shape, np.int64)
        self.clips = np.empty(shape, dtype=object)

    @property
    def size(self):
        return min(self.ticks, self.rows) * self.num_envs

    def begin_new_episodes(self):
        """Next append follows a full simulator reset (also after checkpoint resume)."""
        self._restart = True

    def append(self, state, after, previous, actions, contacts, contexts, obs, frames, clips, fresh):
        tick = self.ticks
        row = tick % self.rows
        self._first[:] = np.where(self._restart | fresh, tick, self._first)
        self._restart = False
        self.episode_start[row] = self._first
        self.frames[row] = frames
        self.clips[row] = clips
        self.states[row] = state
        self.after[row] = after
        self.previous[row] = previous
        self.actions[row] = actions
        self.contacts[row] = contacts
        self.contexts[row] = contexts
        self.observations[tick % len(self.observations)] = obs
        self.ticks += 1
        self._windows.clear()

    def _valid(self, steps):
        if steps < 1:
            raise ValueError("Window length must be positive")
        if steps not in self._windows:
            start = max(0, self.ticks - self.rows)
            ends = np.arange(start + steps - 1, self.ticks, dtype=np.int64)
            # Start and end belong to the same episode iff the last reset is at
            # or before the candidate start. Each column is one fixed world ID.
            valid = self.episode_start[ends % self.rows] <= (ends - steps + 1)[:, None]
            self._windows[steps] = np.flatnonzero(valid)
        return self._windows[steps]

    def valid_windows(self, steps):
        return len(self._valid(steps))

    def has_window(self, steps):
        return self.valid_windows(steps) > 0

    def _starts(self, batch, steps):
        valid = self._valid(steps)
        if not len(valid):
            raise ValueError(f"No episode with {steps} contiguous transitions")
        local, world = np.divmod(self.rng.choice(valid, size=batch), self.num_envs)
        return local + max(0, self.ticks - self.rows), world

    def wm(self, batch, steps):
        tick, world = self._starts(batch, steps)
        row = tick % self.rows
        rows = (tick[:, None] + np.arange(steps)) % self.rows
        indices = rows, world[:, None]
        states = np.concatenate((self.states[row, world][:, None], self.after[indices]), axis=1)
        return states, self.actions[indices], self.contacts[indices], self.contexts[row, world]

    def actor(self, batch, horizon):
        tick, world = self._starts(batch, 1)
        row = tick % self.rows
        clips = self.clips[row, world]
        frames = self.frames[row, world]
        history_ticks = np.maximum(tick[:, None] - np.arange(9, -1, -1),
                                   self.episode_start[row, world][:, None])
        history = self.observations[history_ticks % len(self.observations), world[:, None]]
        raw = gather_reference(clips, frames, np.arange(horizon + 51))
        remaining = np.asarray([len(clip) - 1 for clip in clips], np.float32) - frames
        return (self.states[row, world], self.previous[row, world], self.contexts[row, world],
                history, raw, remaining.astype(np.float32))
