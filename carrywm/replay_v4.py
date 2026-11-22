"""Uniform GPU/CPU replay of real transitions and flat reference bank."""
import hashlib
import numpy as np
import torch


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


class ClipBank:
    """Contiguous tensor storage for static and dynamic reference clips.

    Slot IDs remain stable across frame-buffer compaction so replay ring entries
    never require remapping. Dynamic slots no longer referenced by any active
    replay row are recycled automatically when space is needed.
    """

    def __init__(self, device="cpu", raw_dim=111, init_frames=65536, init_clips=2048):
        self.device = torch.device(device)
        self.raw_dim = raw_dim
        self.frames_buf = torch.zeros((init_frames, raw_dim), dtype=torch.float32, device=self.device)
        self.starts = torch.zeros(init_clips, dtype=torch.int64, device=self.device)
        self.lengths = torch.zeros(init_clips, dtype=torch.int64, device=self.device)
        self._slot_starts = []
        self._slot_lengths = []
        self._slot_static = []
        self._free_slots = []
        self._used_frames = 0
        self._id_to_slot = {}
        self._slot_ref = {}
        self._static_hashes = {}

    @staticmethod
    def _fingerprint(clip_np):
        h = hashlib.blake2b(memoryview(clip_np), digest_size=16).digest()
        return (clip_np.shape, h)

    def _grow_clips(self, min_slots):
        cap = len(self.starts)
        if min_slots <= cap:
            return
        new_cap = max(cap * 2, min_slots)
        new_starts = torch.zeros(new_cap, dtype=torch.int64, device=self.device)
        new_lengths = torch.zeros(new_cap, dtype=torch.int64, device=self.device)
        n = len(self._slot_starts)
        if n:
            new_starts[:n] = self.starts[:n]
            new_lengths[:n] = self.lengths[:n]
        self.starts, self.lengths = new_starts, new_lengths

    def _compact_or_grow(self, needed_frames, active_ids=None):
        if active_ids is not None:
            keep = set(active_ids)
            for slot in range(len(self._slot_starts)):
                if self._slot_lengths[slot] > 0 and not self._slot_static[slot] and slot not in keep:
                    self._slot_lengths[slot] = 0
                    self.lengths[slot] = 0
                    self._free_slots.append(slot)
                    old_ref = self._slot_ref.pop(slot, None)
                    if old_ref is not None:
                        self._id_to_slot.pop(id(old_ref), None)

        alive = [(slot, self._slot_starts[slot], self._slot_lengths[slot])
                 for slot in range(len(self._slot_starts)) if self._slot_lengths[slot] > 0]
        total_alive = sum(length for _, _, length in alive)
        cap = len(self.frames_buf)
        if total_alive + needed_frames <= cap:
            new_cap = cap
        else:
            new_cap = max(cap * 2, total_alive + needed_frames)
        new_buf = torch.zeros((new_cap, self.raw_dim), dtype=torch.float32, device=self.device)
        cursor = 0
        for slot, old_start, length in alive:
            new_buf[cursor:cursor + length].copy_(self.frames_buf[old_start:old_start + length])
            self._slot_starts[slot] = cursor
            self.starts[slot] = cursor
            cursor += length
        self.frames_buf = new_buf
        self._used_frames = cursor

    def register(self, clip, *, static=False, active_ids=None):
        key = id(clip)
        slot = self._id_to_slot.get(key)
        if slot is not None:
            if static and not self._slot_static[slot]:
                self._slot_static[slot] = True
            return slot

        arr = np.ascontiguousarray(clip, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.raw_dim:
            raise ValueError(f"Expected [T, {self.raw_dim}] clip, got {arr.shape}")

        if static:
            fp = self._fingerprint(arr)
            slot = self._static_hashes.get(fp)
            if slot is not None:
                self._id_to_slot[key] = slot
                self._slot_ref[slot] = clip
                return slot

        length = int(arr.shape[0])
        if self._used_frames + length > len(self.frames_buf):
            self._compact_or_grow(length, active_ids=active_ids)

        if self._free_slots:
            slot = self._free_slots.pop()
            self._slot_starts[slot] = self._used_frames
            self._slot_lengths[slot] = length
            self._slot_static[slot] = bool(static)
        else:
            slot = len(self._slot_starts)
            if slot >= len(self.starts):
                self._grow_clips(slot + 1)
            self._slot_starts.append(self._used_frames)
            self._slot_lengths.append(length)
            self._slot_static.append(bool(static))

        start = self._used_frames
        src = torch.from_numpy(arr)
        self.frames_buf[start:start + length].copy_(src)
        self.starts[slot] = start
        self.lengths[slot] = length
        self._used_frames = start + length
        self._id_to_slot[key] = slot
        self._slot_ref[slot] = clip
        if static:
            self._static_hashes[self._fingerprint(arr)] = slot
        return slot

    def register_many(self, clips, *, static=False, active_ids_fn=None):
        out = []
        cached_active = None
        for clip in clips:
            key = id(clip)
            slot = self._id_to_slot.get(key)
            if slot is None:
                if cached_active is None and active_ids_fn is not None and not static:
                    # Defer active scan until a dynamic allocation actually needs compaction.
                    arr_len = len(clip)
                    if self._used_frames + arr_len > len(self.frames_buf):
                        cached_active = active_ids_fn()
                slot = self.register(clip, static=static, active_ids=cached_active)
                if cached_active is not None:
                    cached_active.add(slot)
            elif static and not self._slot_static[slot]:
                self._slot_static[slot] = True
            out.append(slot)
        return out

    def gather(self, clip_ids, frames, offsets):
        starts = self.starts[clip_ids]
        lengths = self.lengths[clip_ids]
        clamped = torch.minimum(frames[:, None].to(torch.int64) + offsets[None, :], lengths[:, None] - 1)
        return self.frames_buf[starts[:, None] + clamped]

    def remaining(self, clip_ids, frames):
        return (self.lengths[clip_ids] - 1 - frames.to(torch.int64)).clamp_min(0).float()

    def _copy_state(self, target_device):
        clone = ClipBank.__new__(ClipBank)
        clone.device = torch.device(target_device)
        clone.raw_dim = self.raw_dim
        clone.frames_buf = self.frames_buf[:self._used_frames].detach().to(clone.device).clone()
        n = len(self._slot_starts)
        clone.starts = self.starts[:n].detach().to(clone.device).clone()
        clone.lengths = self.lengths[:n].detach().to(clone.device).clone()
        clone._slot_starts = list(self._slot_starts)
        clone._slot_lengths = list(self._slot_lengths)
        clone._slot_static = list(self._slot_static)
        clone._free_slots = list(self._free_slots)
        clone._used_frames = self._used_frames
        clone._static_hashes = dict(self._static_hashes)
        if clone.device == self.device:
            clone._id_to_slot = dict(self._id_to_slot)
            clone._slot_ref = dict(self._slot_ref)
        else:
            clone._id_to_slot = {}
            clone._slot_ref = {}
        return clone

    def snapshot_cpu(self):
        clone = self._copy_state("cpu")
        clone._id_to_slot = {}
        clone._slot_ref = {}
        return clone

    def to(self, device):
        target = torch.device(device)
        if target == self.device:
            return self
        self.device = target
        self.frames_buf = self.frames_buf.to(target)
        self.starts = self.starts.to(target)
        self.lengths = self.lengths.to(target)
        return self

    def __getstate__(self):
        n = len(self._slot_starts)
        return {
            "raw_dim": self.raw_dim,
            "frames_buf": self.frames_buf[:self._used_frames].detach().cpu().clone(),
            "starts": self.starts[:n].detach().cpu().clone(),
            "lengths": self.lengths[:n].detach().cpu().clone(),
            "_slot_starts": list(self._slot_starts),
            "_slot_lengths": list(self._slot_lengths),
            "_slot_static": list(self._slot_static),
            "_free_slots": list(self._free_slots),
            "_used_frames": self._used_frames,
            "_static_hashes": dict(self._static_hashes),
        }

    def __setstate__(self, state):
        self.device = torch.device("cpu")
        self.raw_dim = state["raw_dim"]
        self.frames_buf = state["frames_buf"]
        self.starts = state["starts"]
        self.lengths = state["lengths"]
        self._slot_starts = state["_slot_starts"]
        self._slot_lengths = state["_slot_lengths"]
        self._slot_static = state["_slot_static"]
        self._free_slots = state["_free_slots"]
        self._used_frames = state["_used_frames"]
        self._static_hashes = state.get("_static_hashes", {})
        self._id_to_slot = {}
        self._slot_ref = {}


class VectorReplay:
    """Time-major ring on GPU or CPU. A terminal transition retains its actual final state.

    Store nine extra observation rows so actor history remains exact at ring eviction.
    Raw reference clips live in ClipBank and are shared across all rows of each episode.
    """

    def __init__(self, capacity, seed, num_envs, device="cpu"):
        if num_envs < 1 or capacity < num_envs:
            raise ValueError("Replay capacity must fit at least one complete vector tick")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.num_envs = num_envs
        self.rows = capacity // num_envs
        self.capacity = self.rows * num_envs
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device=self.device).manual_seed(int(seed) & 0xFFFFFFFF)
        self.ticks = 0
        self._first = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._restart = True
        self._windows = {}
        self.clip_bank = ClipBank(device=self.device)

        shape = (self.rows, num_envs)

        def field(width):
            return torch.zeros((*shape, width), dtype=torch.float32, device=self.device)

        self.states = field(84)
        self.after = field(84)
        self.actions = field(29)
        self.previous = field(29)
        self.contacts = field(6)
        self.contexts = field(16)
        self.observations = torch.zeros((self.rows + 9, num_envs, 141), dtype=torch.float32, device=self.device)
        self.frames = torch.zeros(shape, dtype=torch.int32, device=self.device)
        self.episode_start = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.clip_ids = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self._hist_offsets = torch.arange(9, -1, -1, dtype=torch.int64, device=self.device)
        self._actor_offsets = torch.arange(83, dtype=torch.int64, device=self.device)
        self._saved_cuda_gen_state = None

    @property
    def size(self):
        return min(self.ticks, self.rows) * self.num_envs

    def begin_new_episodes(self):
        """Next append follows a full simulator reset (also after checkpoint resume)."""
        self._restart = True

    def _active_clip_ids(self):
        populated = min(self.ticks, self.rows)
        if populated == 0:
            return set()
        if populated == self.rows:
            ids = self.clip_ids.reshape(-1)
        else:
            ids = self.clip_ids[:populated].reshape(-1)
        return set(torch.unique(ids).cpu().tolist())

    def register_clips(self, clips, *, static=False, extra_active_ids=None):
        if static:
            active_fn = None
        else:
            def active_fn():
                ids = self._active_clip_ids()
                if extra_active_ids is not None:
                    extra = extra_active_ids
                    if torch.is_tensor(extra):
                        ids.update(torch.unique(extra).cpu().tolist())
                    else:
                        ids.update(int(x) for x in extra)
                return ids
        slots = self.clip_bank.register_many(
            clips, static=static, active_ids_fn=active_fn
        )
        return torch.as_tensor(slots, dtype=torch.int64, device=self.device)

    def _to_device(self, x, dtype):
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=dtype)
        arr = np.asarray(x)
        if not arr.flags.writeable:
            arr = arr.copy()
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def append(self, state, after, previous, actions, contacts, contexts, obs, frames, clips, fresh):
        clip_ids = self.register_clips(clips, static=False)
        self.append_gpu(
            self._to_device(state, torch.float32),
            self._to_device(after, torch.float32),
            self._to_device(previous, torch.float32),
            self._to_device(actions, torch.float32),
            self._to_device(contacts, torch.float32),
            self._to_device(contexts, torch.float32),
            self._to_device(obs, torch.float32),
            self._to_device(frames, torch.int32),
            clip_ids,
            self._to_device(fresh, torch.bool),
        )

    def append_gpu(self, state, after, previous, actions, contacts, contexts, obs, frames, clip_ids, fresh):
        tick = self.ticks
        row = tick % self.rows
        if self._restart:
            self._first.fill_(tick)
            self._restart = False
        else:
            self._first.masked_fill_(fresh, tick)
        self.episode_start[row] = self._first
        self.frames[row] = frames
        self.clip_ids[row] = clip_ids
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
            if self.ticks - start < steps:
                self._windows[steps] = torch.empty(0, dtype=torch.int64, device=self.device)
            else:
                ends = torch.arange(start + steps - 1, self.ticks, dtype=torch.int64, device=self.device)
                # Start and end belong to the same episode iff the last reset is at
                # or before the candidate start. Each column is one fixed world ID.
                valid = self.episode_start[ends % self.rows] <= (ends - steps + 1)[:, None]
                self._windows[steps] = torch.nonzero(valid.reshape(-1), as_tuple=True)[0]
        return self._windows[steps]

    def valid_windows(self, steps):
        return int(self._valid(steps).numel())

    def has_window(self, steps):
        return self.valid_windows(steps) > 0

    def _starts(self, batch, steps):
        valid = self._valid(steps)
        n = int(valid.numel())
        if n == 0:
            raise ValueError(f"No episode with {steps} contiguous transitions")
        idx = torch.randint(n, (batch,), device=self.device, generator=self._gen)
        chosen = valid[idx]
        local = chosen // self.num_envs
        world = chosen % self.num_envs
        return local + max(0, self.ticks - self.rows), world

    def wm(self, batch, steps):
        tick, world = self._starts(batch, steps)
        row = tick % self.rows
        rows = (tick[:, None] + torch.arange(steps, dtype=torch.int64, device=self.device)[None, :]) % self.rows
        w = world[:, None]
        states = torch.cat((self.states[row, world].unsqueeze(1), self.after[rows, w]), dim=1)
        return states, self.actions[rows, w], self.contacts[rows, w], self.contexts[row, world]

    def gather_reference(self, clip_ids, frames, offsets):
        return self.clip_bank.gather(clip_ids, frames, offsets)

    def actor(self, batch, horizon):
        tick, world = self._starts(batch, 1)
        row = tick % self.rows
        clip_ids = self.clip_ids[row, world]
        frames = self.frames[row, world]
        history_ticks = torch.maximum(
            tick[:, None] - self._hist_offsets[None, :],
            self.episode_start[row, world][:, None],
        )
        history = self.observations[history_ticks % len(self.observations), world[:, None]]
        if horizon + 51 != len(self._actor_offsets):
            offsets = torch.arange(horizon + 51, dtype=torch.int64, device=self.device)
        else:
            offsets = self._actor_offsets
        raw = self.clip_bank.gather(clip_ids, frames, offsets)
        remaining = self.clip_bank.remaining(clip_ids, frames)
        return (self.states[row, world], self.previous[row, world], self.contexts[row, world],
                history, raw, remaining)

    def to(self, device):
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if target == self.device:
            return self
        self.device = target
        for attr in ("_first", "states", "after", "actions", "previous", "contacts",
                     "contexts", "observations", "frames", "episode_start", "clip_ids",
                     "_hist_offsets", "_actor_offsets"):
            setattr(self, attr, getattr(self, attr).to(target))
        self.clip_bank.to(target)
        self._windows = {k: v.to(target) for k, v in self._windows.items()}
        saved = getattr(self, "_saved_cuda_gen_state", None)
        if target.type == "cuda" and saved is not None:
            self._gen = torch.Generator(device=target)
            self._gen.set_state(saved)
            self._saved_cuda_gen_state = None
        else:
            if self._gen.device.type == "cpu":
                gen_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self._gen).item())
            else:
                gen_seed = int(self.rng.integers(0, 2**31 - 1))
            self._gen = torch.Generator(device=target).manual_seed(gen_seed)
        return self

    def snapshot_cpu(self):
        snap = VectorReplay.__new__(VectorReplay)
        snap.device = torch.device("cpu")
        snap.num_envs = self.num_envs
        snap.rows = self.rows
        snap.capacity = self.capacity
        snap.seed = self.seed
        snap.rng = np.random.default_rng()
        snap.rng.bit_generator.state = self.rng.bit_generator.state
        gen_state = self._gen.get_state().cpu().clone()
        if self._gen.device.type == "cpu":
            snap._gen = torch.Generator(device="cpu")
            snap._gen.set_state(gen_state)
            snap._saved_cuda_gen_state = None
        else:
            snap._gen = torch.Generator(device="cpu").manual_seed(int(self.rng.integers(0, 2**31 - 1)))
            snap._saved_cuda_gen_state = gen_state
        snap.ticks = self.ticks
        snap._restart = self._restart
        snap._windows = {}
        snap.clip_bank = self.clip_bank.snapshot_cpu()
        for attr in ("_first", "states", "after", "actions", "previous", "contacts",
                     "contexts", "observations", "frames", "episode_start", "clip_ids",
                     "_hist_offsets", "_actor_offsets"):
            setattr(snap, attr, getattr(self, attr).detach().cpu().clone())
        return snap

    def __getstate__(self):
        snap = self.snapshot_cpu()
        if snap._saved_cuda_gen_state is not None:
            gen_state = snap._saved_cuda_gen_state
            gen_device = "cuda"
        else:
            gen_state = snap._gen.get_state()
            gen_device = "cpu"
        return {
            "num_envs": snap.num_envs,
            "rows": snap.rows,
            "capacity": snap.capacity,
            "seed": snap.seed,
            "rng_state": snap.rng.bit_generator.state,
            "gen_state": gen_state,
            "gen_device": gen_device,
            "ticks": snap.ticks,
            "_first": snap._first,
            "_restart": snap._restart,
            "clip_bank": snap.clip_bank,
            "states": snap.states,
            "after": snap.after,
            "actions": snap.actions,
            "previous": snap.previous,
            "contacts": snap.contacts,
            "contexts": snap.contexts,
            "observations": snap.observations,
            "frames": snap.frames,
            "episode_start": snap.episode_start,
            "clip_ids": snap.clip_ids,
        }

    def __setstate__(self, state):
        self.device = torch.device("cpu")
        self.num_envs = state["num_envs"]
        self.rows = state["rows"]
        self.capacity = state["capacity"]
        self.seed = state["seed"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]
        if state.get("gen_device", "cpu") == "cuda":
            self._saved_cuda_gen_state = state["gen_state"]
            seed = int.from_bytes(state["gen_state"][:8].numpy().tobytes(), "little") & 0x7FFFFFFF
            self._gen = torch.Generator(device="cpu").manual_seed(seed)
        else:
            self._saved_cuda_gen_state = None
            self._gen = torch.Generator(device="cpu")
            self._gen.set_state(state["gen_state"])
        self.ticks = state["ticks"]
        self._first = state["_first"]
        self._restart = state["_restart"]
        self._windows = {}
        self.clip_bank = state["clip_bank"]
        for attr in ("states", "after", "actions", "previous", "contacts", "contexts",
                     "observations", "frames", "episode_start", "clip_ids"):
            setattr(self, attr, state[attr])
        self._hist_offsets = torch.arange(9, -1, -1, dtype=torch.int64)
        self._actor_offsets = torch.arange(83, dtype=torch.int64)

    @classmethod
    def from_legacy(cls, old, device="cpu"):
        """Convert an older NumPy-backed VectorReplay checkpoint to the GPU/CPU tensor layout."""
        if isinstance(old, cls):
            return old.to(device)
        seed = int(old.rng.integers(0, 2**31 - 1))
        rep = cls(old.capacity, seed, old.num_envs, device="cpu")
        rep.rng.bit_generator.state = old.rng.bit_generator.state
        rep.ticks = int(old.ticks)
        rep._first = torch.as_tensor(old._first, dtype=torch.int64)
        rep._restart = bool(old._restart)
        for attr in ("states", "after", "actions", "previous", "contacts",
                     "contexts", "observations", "frames", "episode_start"):
            getattr(rep, attr).copy_(torch.as_tensor(getattr(old, attr)))
        populated = min(rep.ticks, rep.rows)
        for r in range(populated):
            row_clips = old.clips[r]
            for c in range(rep.num_envs):
                clip = row_clips[c]
                if clip is not None:
                    rep.clip_ids[r, c] = rep.clip_bank.register(clip, static=False)
        return rep.to(device)
