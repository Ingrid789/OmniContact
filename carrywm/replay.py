"""Episode-aware CPU replay. Reference clips are stored once per segment."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def array(x):
    return np.array(x, dtype=np.float32, copy=True)


@dataclass
class Segment:
    clip: np.ndarray
    context: np.ndarray
    source: str
    name: str
    states: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    previous: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    contacts: list = field(default_factory=list)
    events: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

    def append(self, frame, obs, previous, action, contact, next_state, event):
        self.frames.append(int(frame))
        self.observations.append(array(obs))
        self.previous.append(array(previous))
        self.actions.append(array(action))
        self.contacts.append(array(contact))
        self.states.append(array(next_state))
        self.events.append(dict(event))

    def __len__(self):
        return len(self.actions)


class Replay:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.segments = deque()
        self.size = 0

    def begin(self, env):
        seg = Segment(array(env.clip), array(env.context), env.source, env.name,
                      states=[array(env.state)])
        self.segments.append(seg)
        return seg

    def added(self):
        self.size += 1
        while self.size > self.capacity and len(self.segments) > 1:
            self.size -= len(self.segments.popleft())

    def _starts(self, batch, steps):
        segments = [s for s in self.segments if len(s) >= steps]
        if not segments:
            raise ValueError(f"No segment with {steps} consecutive transitions")
        counts = np.asarray([len(s) - steps + 1 for s in segments], dtype=np.int64)
        edges = np.cumsum(counts)
        flat = self.rng.integers(edges[-1], size=batch)
        idx = np.searchsorted(edges, flat, side="right")
        start = flat - np.r_[0, edges[:-1]][idx]
        return [(segments[i], int(j)) for i, j in zip(idx, start)]

    def wm(self, batch, steps):
        starts = self._starts(batch, steps)
        return tuple(np.asarray(items, dtype=np.float32) for items in zip(*[
            (seg.states[t:t + steps + 1], seg.actions[t:t + steps],
             seg.contacts[t:t + steps], seg.context) for seg, t in starts
        ]))

    def actor(self, batch, horizon):
        starts = self._starts(batch, 1)
        data = []
        for seg, i in starts:
            frame = seg.frames[i]
            right = len(seg.clip) - 1
            history = np.stack([seg.observations[max(i-j, 0)] for j in range(9, -1, -1)])
            # The final value at horizon also reads a full sparse reference window.
            ix = np.minimum(frame + np.arange(horizon + 51), right)
            data.append((seg.states[i], seg.previous[i], seg.context, history,
                         seg.clip[ix], max(0, right-frame)))
        return tuple(np.asarray(items, dtype=np.float32) for items in zip(*data))
