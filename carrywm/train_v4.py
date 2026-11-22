"""Train the carry-box policy on 1024 on-policy MuJoCo Warp worlds."""
import fcntl
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import train as base
from . import train_v3
from .v3 import LearnerV3

NUM_ENVS = 1024
ACTOR_BATCH = 2048
REPLAY_RATIO = 16
WM_SAMPLES = train_v3.WM_UPDATES * train_v3.WM_BATCH * base.WM_STEPS


class UpdateClock:
    """Integer accounting for a fraction of an update per vector control step."""

    def __init__(self, num_envs):
        self.increment = num_envs * REPLAY_RATIO
        self.credit = 0

    def tick(self):
        updates, self.credit = divmod(self.credit + self.increment, WM_SAMPLES)
        return updates


def configure(parser):
    train_v3.configure(parser)
    parser.description = __doc__
    parser.add_argument('--num-envs', type=int, default=NUM_ENVS)
    parser.set_defaults(batch_size=ACTOR_BATCH, actor_microbatch=ACTOR_BATCH,
                        critic_chunk=16384, warmup_steps=NUM_ENVS * base.WM_STEPS,
                        cfgen_goal_noise_std_m=0., collect_steps=5, eval_episodes=3)


def args_from_cli(argv=None, *, configure_parser=configure):
    args = base.args_from_cli(argv, configure=configure_parser)
    if (args.num_envs != NUM_ENVS or args.collect_steps != 5 or args.wm_batch_size != train_v3.WM_BATCH
            or args.batch_size != ACTOR_BATCH or args.auto_batch):
        raise ValueError('The vector trainer uses 5 × 1024 real transitions, 10 × 1024 WM windows and 2048 actor rollouts')
    if (args.cfgen_goal_noise_std_m != 0 or args.warmup_teacher or args.teacher_probability != 0
            or args.wm_pretrain != 0 or not 1 <= args.actor_microbatch <= ACTOR_BATCH
            or args.critic_chunk < 1):
        raise ValueError('The vector trainer uses on-policy warmup, zero CF-Gen goal noise and positive physical minibatches')
    if args.capacity // args.num_envs < base.WM_STEPS or torch.device(args.device).type != 'cuda':
        raise ValueError('The vector trainer requires CUDA and room for at least eight transitions per world')
    return args


def train(args, *, version=4, learner_type=LearnerV3):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(args, version, learner_type)


def check_resume(ck, args, version):
    if ck['version'] != version:
        raise ValueError(f"Cannot resume V{ck['version']} using V{version}")
    if ck['iteration'] >= args.iterations:
        raise ValueError('--iterations must exceed the saved iteration')
    keys = ('num_envs', 'capacity', 'cfgen_ratio', 'action_noise', 'goal_mode', 'seed')
    if version == 5:
        keys += ('termination_height_m', 'termination_gravity_z', 'termination_softness', 'critic_value_max', 'drop_penalty')
    for key in keys:
        if ck['config'][key] != getattr(args, key):
            raise ValueError(f'--resume requires the saved {key} value')


def _run(args, version, learner_type):
    from .collector_v4 import VectorCollector
    from .replay_v4 import VectorReplay
    from .warp_sim import WarpCarryWorlds

    start = time.monotonic()
    output = args.output_dir
    if not args.resume and any((output / name).exists() for name in ('latest.pt', 'config.json', 'metrics.jsonl')):
        raise FileExistsError(f'{output} already contains a run. Select a new directory or resume its checkpoint.')
    device = torch.device(args.device)
    torch.cuda.set_device(device.index if device.index is not None else 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('The selected device does not support bfloat16')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = learner_type(args)
    clock = UpdateClock(args.num_envs)
    first, elapsed, old_steps, old_ticks = 1, 0., 0, 0
    peak_alloc = peak_reserved = 0
    episodes = {}
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        check_resume(ck, args, version)
        for model, state in zip(net.models, ck['models']):
            model.load_state_dict(state)
        for optimizer, state in zip(net.optimizers, ck['optimizers']):
            optimizer.load_state_dict(state)
        replay = ck['replay']
        first, elapsed = ck['iteration'] + 1, ck['elapsed']
        old_steps, old_ticks = ck['real_steps'], ck['control_ticks']
        peak_alloc, peak_reserved = ck['peak_bytes']
        clock.credit = ck['credit']
        rng.bit_generator.state = ck['rng']
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state(ck['cuda_rng'], device)
        episodes = ck['episodes']
        del ck
    else:
        replay = VectorReplay(args.capacity, args.seed, args.num_envs)

    # Recreate physical environments on restart. A fresh segment in every world
    # prevents a sampled WM sequence from straddling the restart boundary.
    env = WarpCarryWorlds(args.sim_root, args.motion_dir, num_envs=args.num_envs,
                          seed=args.seed + old_ticks, device=device, cfgen_ratio=args.cfgen_ratio)
    collector = VectorCollector(env, net, replay, rng)
    collector.episodes.update(episodes)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(version=version, replay_ratio=REPLAY_RATIO, wm_horizon=base.WM_STEPS,
                  actor_horizon=base.HORIZON, physics='mujoco_warp', torch=str(torch.__version__),
                  hardware=torch.cuda.get_device_name(device))
    (output / ('resume-config.json' if args.resume else 'config.json')).write_text(json.dumps(config, indent=2) + '\n')

    def record(data):
        line = json.dumps(data, allow_nan=False)
        with (output / 'metrics.jsonl').open('a') as log:
            log.write(line + '\n')
        print(line, flush=True)

    def save(iteration):
        base.atomic_save(dict(version=version, iteration=iteration, elapsed=elapsed + time.monotonic() - start,
                              real_steps=old_steps + collector.steps, control_ticks=old_ticks + collector.steps // args.num_envs,
                              models=[m.state_dict() for m in net.models], optimizers=[o.state_dict() for o in net.optimizers],
                              replay=replay, credit=clock.credit, rng=rng.bit_generator.state,
                              torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                              episodes=collector.episodes, config=config, peak_bytes=(peak_alloc, peak_reserved)), output / 'latest.pt')

    def warmup(minimum, *, update_normalizers, reason):
        initial_steps = collector.steps
        while collector.steps - initial_steps < minimum or not replay.has_window(base.WM_STEPS):
            collector.one_tick(noise=args.action_noise, update_normalizers=update_normalizers)
            if collector.steps - initial_steps >= max(args.capacity, minimum) and not replay.has_window(base.WM_STEPS):
                raise RuntimeError('Warmup produced no continuous eight-step trajectory in any world')
        base.sync(device)
        record({'stage': 'warmup', 'reason': reason, 'real_steps': old_steps + collector.steps, 'warmup/valid_windows': replay.valid_windows(base.WM_STEPS),
                'warmup/actor_updates': 0, 'warmup/critic_updates': 0, 'warmup/wm_updates': 0,
                'seconds': time.monotonic() - start})

    if args.resume:
        warmup(base.WM_STEPS * args.num_envs, update_normalizers=False, reason='new_physical_episodes_after_resume')
    else:
        warmup(args.warmup_steps, update_normalizers=True, reason='initial_replay')

    for iteration in range(first, args.iterations + 1):
        base.sync(device)
        tick = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        before = collector.steps
        totals = {}
        updates = 0
        while not updates:
            for key, value in collector.one_tick(noise=args.action_noise).items():
                totals[key] = totals.get(key, 0.) + value
            updates = clock.tick()
            if updates and not replay.has_window(base.WM_STEPS):
                # Short episodes can evict every valid window. Don't count the
                # extra collection as an update with replay ratio 16.
                warmup(base.WM_STEPS * args.num_envs, update_normalizers=False, reason='no_continuous_window')
                before = collector.steps
                totals.clear()
                updates = 0
        new_steps = collector.steps - before
        if updates != 1 or new_steps != args.collect_steps * args.num_envs:
            raise RuntimeError('Inconsistent vector-step count for one learning round')
        stats = {key: value / (new_steps // args.num_envs) for key, value in totals.items()}
        base.sync(device)
        after_collect = time.monotonic()
        stats.update(train_v3.wm_iteration(net, replay, args.wm_batch_size))
        base.sync(device)
        after_wm = time.monotonic()
        stats.update(net.policy_update(replay, args.batch_size))
        base.sync(device)
        after_policy = time.monotonic()
        peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        stats.update(iteration=iteration, real_steps=old_steps + collector.steps, batch_size=args.batch_size,
                     **{'sim/new_transitions': new_steps, 'sim/num_envs': args.num_envs,
                        'replay/ratio': WM_SAMPLES / new_steps, 'timing/collect_sec': after_collect - tick,
                        'timing/wm_sec': after_wm - after_collect, 'timing/policy_sec': after_policy - after_wm,
                        'gpu/peak_allocated_bytes': peak_alloc, 'gpu/peak_reserved_bytes': peak_reserved})
        evaluate = args.eval_every and (iteration % args.eval_every == 0 or iteration == args.iterations)
        if evaluate:
            stats.update(base.validate(net, args))
        if iteration in (first, args.iterations) or iteration % args.save_every == 0:
            save(iteration)
        if evaluate or iteration in (first, args.iterations) or iteration % args.log_every == 0:
            stats['wall_seconds'] = elapsed + time.monotonic() - start
            stats.update({f'collect/{k}': v for k, v in collector.episodes.items()})
            record(stats)
    summary = dict(completed_iterations=args.iterations, real_steps=old_steps + collector.steps,
                   wall_seconds=elapsed + time.monotonic() - start, peak_allocated_bytes=peak_alloc,
                   peak_reserved_bytes=peak_reserved, hardware=config['hardware'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    train(args_from_cli())
