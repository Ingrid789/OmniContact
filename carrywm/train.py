"""Train CF-Track in a learned differentiable model of a MuJoCo carry scene."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import fcntl
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .model import Actor, Critic, WorldModel, CarryGeometry, world_loss
from .replay import Replay, array
from .sim import CarryEnv


OFFSETS = (0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50)
WM_STEPS = 8
HORIZON = 32
GAMMA = 0.99
LAMBDA = 0.95


def args_from_cli(argv=None, *, configure=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-root", required=True, type=Path)
    p.add_argument("--motion-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--resume", type=Path, help="Trusted local checkpoint written by this program")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--iterations", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--wm-batch-size", type=int, default=512)
    p.add_argument("--auto-batch", action="store_true")
    p.add_argument("--target-memory-fraction", type=float, default=.85)
    p.add_argument("--max-batch-size", type=int, default=4096)
    p.add_argument("--collect-steps", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=512)
    p.add_argument("--wm-pretrain", type=int, default=64)
    p.add_argument("--warmup-teacher", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--teacher-probability", type=float, default=0.)
    p.add_argument("--cfgen-ratio", type=float, default=.1)
    p.add_argument("--action-noise", type=float, default=.1)
    p.add_argument("--capacity", type=int, default=65536)
    p.add_argument("--goal-mode",choices=("full","flow"),default="full",
                   help="Keep 49 original reference fields only for a constant-width ablation")
    p.add_argument("--encoder-width", type=int, default=2048)
    p.add_argument("--decoder-width", type=int, default=2048)
    p.add_argument("--wm-width", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--drop-penalty", type=float, default=1.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=1, help="per primary reference source")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    if configure is not None:
        configure(p)
    a = p.parse_args(argv)
    for key in ("iterations", "batch_size", "wm_batch_size", "max_batch_size",
                "collect_steps", "capacity", "encoder_width", "decoder_width",
                "wm_width", "save_every", "log_every", "eval_episodes"):
        if getattr(a, key) < 1:
            p.error(f"{key} must be positive")
    if not 0 < a.target_memory_fraction < 1:
        p.error("target-memory-fraction must be between zero and one")
    if not (0 <= a.cfgen_ratio <= 1 and 0 <= a.teacher_probability <= 1):
        p.error("sampling probabilities must be in [0, 1]")
    if a.warmup_steps < WM_STEPS or a.capacity < WM_STEPS:
        p.error("warmup-steps and capacity must cover at least eight transitions")
    return a


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


@contextmanager
def frozen(*models):
    params = [p for m in models for p in m.parameters()]
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


def update(optimizer, model, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def lambda_returns(rewards, next_values, continues, gamma=GAMMA, lam=LAMBDA):
    running = next_values[-1]
    targets = []
    for r, v, c in zip(reversed(rewards.unbind()), reversed(next_values.unbind()),
                       reversed(continues.unbind())):
        running = r + gamma * c * ((1 - lam) * v + lam * running)
        targets.append(running)
    return torch.stack(targets[::-1])


class Learner:
    def __init__(self, args, context_dim=16):
        self.args = args
        self.device = torch.device(args.device)
        self.offsets = torch.tensor(OFFSETS, device=self.device)
        self.geo = CarryGeometry(args.sim_root).to(self.device)
        self.actor = Actor(first_width=args.encoder_width, decoder_width=args.decoder_width).to(self.device)
        self.wm = WorldModel(context_dim, width=args.wm_width).to(self.device)
        self.critic = Critic(context_dim).to(self.device)
        self.models = [self.wm, self.actor, self.critic]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=args.lr) for m in self.models]

    def autocast(self):
        if self.args.precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def reference(self, state, raw, fk=None):
        goal = self.geo.reference(state, raw, fk=fk)
        if self.args.goal_mode == "flow":
            goal = torch.cat((goal[..., :49], torch.zeros_like(goal[..., 49:])), dim=-1)
        return goal

    def wm_update(self, replay, batch_size):
        s, a, c, ctx = (tensor(v, self.device) for v in replay.wm(batch_size, WM_STEPS))
        self.wm.update_stats(s[:, :-1].reshape(-1, 84), a.reshape(-1, 29),
                             ctx[:, None].expand(-1, WM_STEPS, -1).reshape(-1, ctx.shape[-1]))
        state = s[:, 0]
        loss = torch.zeros((), device=self.device)
        for j in range(WM_STEPS):
            with self.autocast():
                state, logits = self.wm(state, a[:, j], ctx)
            state = state.float()
            loss = loss + world_loss(state, s[:, j+1], logits.float(), c[:, j])
        loss = loss / WM_STEPS
        update(self.optimizers[0], self.wm, loss)
        return {"wm/loss": float(loss.detach()),
                "wm/box_8step_m": (state[:, 71:74]-s[:, -1, 71:74]).norm(dim=-1).mean().item()}

    def _value(self, states, goals, contexts):
        with self.autocast():
            return self.critic(states, goals, contexts).reshape(-1).float()

    def _rollout(self, inputs, *, update_stats=False, checkpoint_encoder=False):
        """Closed-loop V2/V3 imagination; caller owns freezing and optimizer steps."""
        state, prev, ctx, hist, raw, remaining = inputs
        batch_size = len(state)
        target_fk = self.geo.fk_from_target(raw[:, 1:HORIZON+1].reshape(-1, 111))
        target_fk = target_fk.reshape(batch_size, HORIZON, *target_fk.shape[1:])
        fk = self.geo.forward_kinematics(state)
        goal = self.reference(state, raw[:, self.offsets], fk=fk)
        if update_stats:
            self.actor.update_stats(goal.detach(), hist)
        states, goals, rewards, continuations, weights = [], [], [], [], []
        mask = torch.ones_like(remaining)
        slip_frames = torch.zeros_like(remaining)
        ret = torch.zeros_like(remaining)
        for j in range(HORIZON):
            states.append(state.detach())
            goals.append(goal.detach())
            weights.append(mask)
            with self.autocast():
                action = self.actor(goal, hist, checkpoint_encoder=checkpoint_encoder).float().clamp(-10, 10)
                after, logits = self.wm(state, action, ctx)
            after = after.float()
            contacts = logits.float().sigmoid()
            next_fk = self.geo.forward_kinematics(after)
            target = raw[:, j+1]
            reward = self.geo.reward(state, after, prev, action, target, contacts,
                                     prev_fk=fk, fk=next_fk, target_fk=target_fk[:, j]).reshape(-1)
            # Consecutive inferred hand loss and a substantially descended box
            # distinguish a confirmed drop from brief loss of a contact.
            expects_hold = target[:, 37:39].mean(-1) > .5
            lost = expects_hold & (contacts[:, 2:4].mean(-1) < .5)
            slip_frames = torch.where(lost, slip_frames+1, 0.)
            dropped = (slip_frames >= 3) & (after[:, 73] < target[:, 106]-.15)
            fallen = (after[:, 2] - target[:, 99]).abs() > .5
            bad_box = (after[:, 73] - target[:, 106]).abs() > .5
            cont = (~(dropped | fallen | bad_box) & (remaining > j+1)).float()
            reward = reward - self.args.drop_penalty * dropped.float()
            rewards.append(reward.detach())
            continuations.append(cont)
            ret = ret + (GAMMA**j) * mask * reward
            mask = mask * cont
            obs = self.geo.observation(after, action, ctx, fk=next_fk)
            hist = torch.cat([hist[:, 1:], obs[:, None]], dim=1)
            state, prev, fk = after, action, next_fk
            goal = self.reference(state, raw[:, j+1+self.offsets], fk=fk)
        terminal = self._value(state, goal, ctx)
        ret = ret + (GAMMA**HORIZON)*mask*terminal
        states.append(state.detach())
        goals.append(goal.detach())
        return ret, mask, states, goals, rewards, continuations, weights

    def policy_update(self, replay, batch_size):
        inputs = tuple(tensor(v, self.device) for v in replay.actor(batch_size, HORIZON))
        ctx = inputs[2]
        with frozen(self.wm, self.critic):
            ret, mask, states, goals, rewards, continuations, weights = self._rollout(
                inputs, update_stats=True)
            self.optimizers[1].zero_grad(set_to_none=True)
            (-ret.mean()).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1., error_if_nonfinite=True)
        # Critic targets are fixed before any critic update; use detached rollouts.
        contexts = ctx.expand(HORIZON+1, -1, -1).reshape(-1, ctx.shape[-1])
        sf = torch.stack(states).flatten(0, 1)
        gf = torch.stack(goals).flatten(0, 1)
        # Bound critic activation memory independently of the 32 imagined steps.
        chunk = min(2048, batch_size * 4)
        with torch.no_grad():
            old = torch.cat([self._value(sf[i:i+chunk], gf[i:i+chunk], contexts[i:i+chunk])
                             for i in range(0, len(sf), chunk)]).view(HORIZON+1,-1)
            targets = lambda_returns(torch.stack(rewards), old[1:], torch.stack(continuations)).flatten()
        valid = torch.stack(weights).flatten()
        denominator = valid.sum().clamp(min=1)
        total_loss = torch.zeros((), device=self.device)
        self.optimizers[2].zero_grad(set_to_none=True)
        for i in range(0, len(targets), chunk):
            end = min(i+chunk, len(targets))
            value = self._value(sf[i:end], gf[i:end], contexts[i:end])
            loss = ((value-targets[i:end]).square()*valid[i:end]).sum()/denominator
            loss.backward()
            total_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1., error_if_nonfinite=True)
        # Hold both sets of parameters fixed until both backward passes succeed.
        for opt in self.optimizers[1:]:
            opt.step()
            opt.zero_grad(set_to_none=True)
        return {"actor/return": ret.detach().mean().item(), "critic/loss": total_loss.item(),
                "actor/alive32": mask.mean().item()}


class Collector:
    def __init__(self, env, learner, replay, rng):
        self.env, self.net, self.replay, self.rng = env, learner, replay, rng
        self.seg = self.hist = None
        self.episodes = Counter()
        self.steps = 0

    def reset(self, source=None):
        if source is None:
            self.env.reset()
        else:
            self.env.reset(source=source)
        self.seg = self.replay.begin(self.env)
        self.hist = None
        self.prev = np.zeros(29, np.float32)
        self.episodes[f"{self.env.source}/started"] += 1

    @torch.no_grad()
    def collect(self, n, teacher_prob=0., noise=0., stop_at_done=False, update_normalizers=False):
        geo, device = self.net.geo, self.net.device
        collected = []
        teacher_count = 0
        for _ in range(n):
            if self.seg is None:
                self.reset()
            e, seg = self.env, self.seg
            frame = e.index
            state, ctx = tensor(e.state, device)[None], tensor(e.context, device)[None]
            obs = geo.observation(state, tensor(self.prev, device)[None], ctx)
            if self.hist is None:
                self.hist = obs[:, None].expand(-1, 10, -1).clone()
            else:
                self.hist = torch.cat((self.hist[:, 1:], obs[:, None]), dim=1)
            using_teacher = teacher_prob > 0 and self.rng.random() < teacher_prob
            if not using_teacher or update_normalizers:
                idx = np.minimum(frame+np.asarray(OFFSETS), len(seg.clip)-1)
                goal = self.net.reference(state, tensor(seg.clip[idx], device)[None])
            if using_teacher:
                action = array(e.teacher_action())
                teacher_count += 1
            else:
                with self.net.autocast():
                    action = self.net.actor(goal, self.hist)[0].float().cpu().numpy()
            if noise:
                action = action + self.rng.normal(0, noise, 29)
            action = np.clip(action, -10, 10).astype(np.float32)
            after, contact, done, info = e.step(action)
            action = array(info.get("executed_action", action))
            if update_normalizers:
                # Real executed transitions seed the first imagined update.
                self.net.actor.update_stats(goal, self.hist)
                self.net.wm.update_stats(state, tensor(action, device)[None], ctx)
            ev = {k: bool(info.get(k, False)) for k in ("success", "drop", "timeout", "recovered", "fallen")}
            ev.update({k: bool(info.get(k, False)) for k in ("grounded", "supported", "lifted")})
            ev["done"] = bool(done)
            seg.append(frame, obs[0].cpu().numpy(), self.prev, action, contact, after, ev)
            collected.append((seg, len(seg)-1))
            self.prev = action
            self.replay.added()
            self.steps += 1
            if done:
                for k in ("success", "drop", "timeout", "recovered", "fallen", "done"):
                    self.episodes[f"{seg.source}/{k}"] += int(ev[k])
                self.seg = None
                if stop_at_done:
                    break
        # Evaluate the original reward components on measured transitions in one batch.
        states, next_states, previous, actions, targets, contacts = zip(*[
            (s.states[i], s.states[i+1], s.previous[i], s.actions[i],
             s.clip[min(s.frames[i]+1,len(s.clip)-1)], s.contacts[i]) for s,i in collected])
        args = [tensor(np.asarray(x), device) for x in (states,next_states,previous,actions,targets,contacts)]
        terms = geo.reward_terms(*args)
        names = list(terms)
        rows = torch.stack([terms[k].reshape(-1) for k in names], dim=1).cpu().numpy()
        for (s,i), row in zip(collected,rows):
            entry = dict(zip(names,row.tolist()))
            entry["drop_event"] = -self.net.args.drop_penalty * s.events[i]["drop"]
            s.rewards.append(entry)
        return {"sim/teacher_fraction":teacher_count/len(collected),
                "sim/reward":float(rows.sum(1).mean()-self.net.args.drop_penalty*
                                   np.mean([s.events[i]["drop"] for s,i in collected]))}


@torch.no_grad()
def validate(net, args):
    env = CarryEnv(args.sim_root, args.motion_dir, seed=args.seed+811, split="val",
                   cfgen_ratio=args.cfgen_ratio)
    results = {}
    replay = Replay(max(65536,args.capacity), args.seed+811)
    collector = Collector(env, net, replay, np.random.default_rng(args.seed+811))
    for source in ("npz","cfgen"):
        for _ in range(args.eval_episodes):
            collector.reset(source)
            for _ in range(2):  # One recovery attempt, separately counted and scored.
                seg = collector.seg
                while collector.seg is not None:
                    collector.collect(64, stop_at_done=True)
                if not seg.events[-1]["drop"] or seg.source=="recovery":
                    break
                if not getattr(env, "pending_recovery", False):
                    break
                collector.reset()
                if collector.seg.source != "recovery":
                    break
    results.update({f"eval/{k}":v for k,v in collector.episodes.items()})
    for source in ("npz","cfgen","recovery"):
        n = collector.episodes[f"{source}/done"]
        if n:
            for key in ("success","drop"):
                results[f"eval/{source}/{key}_rate"] = collector.episodes[f"{source}/{key}"]/n
    from .validation import validate_dynamics, aggregate_dynamics
    evaluated = []
    by_source = {}
    for seg in replay.segments:
        if not len(seg):
            continue
        indices = [*seg.frames, min(seg.frames[-1] + 1, len(seg.clip) - 1)]
        args_wm = (seg.states, seg.actions, seg.context, seg.contacts,
                   seg.clip[indices], [x["done"] for x in seg.events])
        held = [tensor(np.asarray(x), net.device) for x in args_wm]
        held[-1] = held[-1].bool()
        wm_metrics = validate_dynamics(net.wm, *held, events=seg.events)
        evaluated.append(wm_metrics)
        by_source.setdefault(seg.source, []).append(wm_metrics)
    results.update({f"eval/dynamics/{k}": v for k,v in aggregate_dynamics(evaluated).items()})
    for source, episodes in by_source.items():
        results.update({f"eval/{source}/dynamics/{k}": v for k,v in aggregate_dynamics(episodes).items()})
    return results


def atomic_save(payload, path):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def train(args, *, version=2, learner_type=Learner, wm_iteration=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train_locked(args, version, learner_type, wm_iteration)


def _train_locked(args, version, learner_type, wm_iteration):
    device = torch.device(args.device)
    gpu = device.type=="cuda"
    if gpu:
        torch.cuda.set_device(device if device.index is not None else 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.precision=="bf16" and gpu and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 unavailable on requested device, use --precision fp32")
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    if not args.resume and any((args.output_dir / n).exists() for n in ("latest.pt","metrics.jsonl","config.json")):
        raise FileExistsError("output directory already contains a run; choose another or pass --resume")
    env = CarryEnv(args.sim_root, args.motion_dir, seed=args.seed, split="train", cfgen_ratio=args.cfgen_ratio)
    net = learner_type(args)
    replay = Replay(args.capacity, args.seed)
    collector = Collector(env, net, replay, rng)
    first, elapsed, old_steps = 1, 0., 0
    batch, wm_batch = args.batch_size, args.wm_batch_size
    peak_alloc = peak_reserved = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_version = ck["version"]
        if saved_version != version:
            raise ValueError(f"Cannot resume V{saved_version} using V{version}")
        if version == 3 and ck["iteration"] >= args.iterations:
            raise ValueError("--iterations must exceed the saved global iteration")
        for model, state in zip(net.models,ck["models"]):
            model.load_state_dict(state)
        for opt, state in zip(net.optimizers,ck["optimizers"]):
            opt.load_state_dict(state)
        replay = collector.replay = ck["replay"]
        first,elapsed,old_steps = ck["iteration"]+1,ck["elapsed"],ck["real_steps"]
        if version == 2:
            batch,wm_batch = ck["batch"],ck["wm_batch"]
        collector.episodes.update(ck["episodes"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        if gpu and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"],device)
        if hasattr(env,"rng") and ck.get("env_rng"):
            env.rng.bit_generator.state = ck["env_rng"]
        peak_alloc,peak_reserved = ck.get("peak_bytes",(0,0))
        del ck
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config["hardware"] = torch.cuda.get_device_name(device) if gpu else str(device)
    config["torch"] = str(torch.__version__)
    config["version"] = version
    (args.output_dir/("resume-config.json" if args.resume else "config.json")).write_text(
        json.dumps(config,indent=2)+"\n",encoding="utf-8")
    def record(data):
        with (args.output_dir/"metrics.jsonl").open("a", encoding="utf-8") as log:
            msg=json.dumps(data,allow_nan=False)
            log.write(msg+"\n")
        print(msg,flush=True)
    def save(iteration):
        atomic_save(dict(version=version,
                         iteration=iteration,elapsed=elapsed+time.perf_counter()-start,
                         real_steps=collector.steps+old_steps,batch=batch,wm_batch=wm_batch,
                         models=[m.state_dict() for m in net.models],
                         optimizers=[o.state_dict() for o in net.optimizers],replay=replay,
                         rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state(device) if gpu else None,
                         env_rng=env.rng.bit_generator.state if hasattr(env,"rng") else None,
                         episodes=collector.episodes,config=config,peak_bytes=(peak_alloc,peak_reserved)),
                    args.output_dir/"latest.pt")
    if first==1 and not args.resume:
        todo=args.warmup_steps
        while todo:
            n=min(todo,128)
            collector.collect(n,float(args.warmup_teacher),args.action_noise,
                              update_normalizers=version == 3)
            todo-=n
        if version == 3 and not any(len(s) >= WM_STEPS for s in replay.segments):
            raise RuntimeError("Warmup did not produce an eight-step window; inspect reset conditions")
        for _ in range(args.wm_pretrain):
            net.wm_update(replay,wm_batch)
        sync(device)
        record({"stage":"warmup","real_steps":collector.steps,"seconds":time.perf_counter()-start,
                "warmup/teacher":args.warmup_teacher, "warmup/wm_updates":args.wm_pretrain,
                "warmup/actor_updates":0, "warmup/critic_updates":0,
                "warmup/valid_windows":sum(max(0,len(s)-WM_STEPS+1) for s in replay.segments)})
    for it in range(first,args.iterations+1):
        sync(device)
        tick=time.perf_counter()
        if gpu:
            torch.cuda.reset_peak_memory_stats(device)
        stats = collector.collect(args.collect_steps,args.teacher_probability,args.action_noise)
        sync(device)
        after_collect=time.perf_counter()
        if wm_iteration is None:
            stats.update(net.wm_update(replay,wm_batch))
        else:
            stats.update(wm_iteration(net,replay,wm_batch))
        sync(device)
        after_wm=time.perf_counter()
        while True:
            try:
                stats.update(net.policy_update(replay,batch))
                break
            except torch.cuda.OutOfMemoryError:
                if not args.auto_batch or not gpu or batch<=1:
                    raise
            # Leave the exception scope before releasing tensors in its traceback.
            for opt in net.optimizers:
                opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            batch=max(1,batch//2)
            args.max_batch_size=min(args.max_batch_size,batch)
        sync(device)
        after_policy=time.perf_counter()
        stats.update({"iteration":it,"batch_size":batch,"real_steps":collector.steps+old_steps,
                      "timing/collect_sec":after_collect-tick,"timing/wm_sec":after_wm-after_collect,
                      "timing/policy_sec":after_policy-after_wm})
        if gpu:
            alloc=torch.cuda.max_memory_allocated(device)
            reserved=torch.cuda.max_memory_reserved(device)
            peak_alloc=max(peak_alloc,alloc)
            peak_reserved=max(peak_reserved,reserved)
            free,total=torch.cuda.mem_get_info(device)
            stats.update({"gpu/total_bytes":total,"gpu/peak_allocated_bytes":peak_alloc,
                          "gpu/peak_reserved_bytes":peak_reserved,
                          "gpu/peak_process_reserved_pct":100*peak_reserved/total})
            if args.auto_batch and it%5==1:
                # Exclude other processes' occupied memory from this job's budget.
                available=free+torch.cuda.memory_reserved(device)
                budget=min(args.target_memory_fraction*total,available-.05*total)
                if alloc>0 and budget>alloc*1.14:
                    new=min(args.max_batch_size, int(batch*min(1.5,.97*budget/alloc)))
                    batch=max(batch,new)
        if args.eval_every and (it%args.eval_every==0 or it==args.iterations):
            stats.update(validate(net,args))
        if it%args.save_every==0 or it==args.iterations or (version==3 and it==first):
            save(it)
        if it%args.log_every==0 or it in (first,args.iterations) or (
            args.eval_every and it%args.eval_every==0):
            stats["wall_seconds"]=elapsed+time.perf_counter()-start
            stats.update({f"collect/{k}":v for k,v in collector.episodes.items()})
            record(stats)
    sync(device)
    summary={"completed_iterations":max(first-1,args.iterations),
             "wall_seconds":elapsed+time.perf_counter()-start,
             "real_steps":collector.steps+old_steps,"peak_allocated_bytes":peak_alloc,
             "peak_reserved_bytes":peak_reserved,"hardware":config["hardware"]}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    train(args_from_cli())
