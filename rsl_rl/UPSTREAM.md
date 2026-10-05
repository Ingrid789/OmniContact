# OmniContact RSL-RL fork

Source: https://github.com/leggedrobotics/rsl_rl

Base tag: `v2.3.3`

Local package version: `2.3.3+omnicontact.1`.

This directory was cloned from upstream and is vendored as ordinary source files
so a checkout of OmniContact includes the AMP changes. It is not a git submodule.
The upstream BSD-3-Clause license and notices are retained in `LICENSE`.

Local additions:

- `rsl_rl/algorithms/amp_ppo.py`: native PPO subclass with motion prior rewards,
  least-squares discriminator training and gradient penalty.
- `rsl_rl/modules/amp.py`: discriminator and serializable torch running moments.
- `rsl_rl/modules/actor_critic_transformer.py`: observation-group Transformer
  actor and critic using the native Gaussian policy interface.
- `rsl_rl/modules/actor_critic.py`: shared action-distribution initialization and
  input dimensions used for MLP/Transformer export.
- `rsl_rl/storage/amp_replay_buffer.py`: bounded transition replay.
- `rsl_rl/runners/on_policy_runner.py`: AMP construction, rollout initialization,
  checkpoint state, portable loading, initial observation normalization and
  support for learning without a log directory, and Transformer group dimensions.
- `rsl_rl/algorithms/__init__.py`: public `AMPPPO` export.
- `pyproject.toml`: local version suffix distinguishing this fork from stock
  RSL-RL, which does not provide this integration.

PPO updates and rollout storage remain the upstream implementation. Tests are in
the parent repository's `tests/` directory. Isaac Lab task-specific observations
and motion loading remain in `source/omnicontact`.
