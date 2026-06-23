<h1 align="center">
  <span style="color:#ef5b5b">OmniContact</span>: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation
</h1>

<p align="center">
  <a href="https://ingrid789.github.io/IngridYu/">Runyi Yu</a><sup>1,2,*</sup>,
  <a href="https://github.com/XiaoyiLin-code">Xiaoyi Lin</a><sup>1,3,*</sup>,
  <a href="https://astrorix.github.io/">Ji Ma</a><sup>1</sup>,
  <a href="https://wyhuai.github.io/info/">Yinhuai Wang</a><sup>2,✉</sup>,
  <a href="https://chubbyemo.github.io/">Koukou Luo</a><sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=3dhUvVYAAAAJ&hl=zh-CN&oi=ao">Jiahao Ji</a><sup>1</sup>,
  <a href="https://why618188.github.io/">Huayi Wang</a><sup>1,4</sup>,
  <a href="https://wenjiawang0312.github.io/">Wenjia Wang</a><sup>1,4</sup>,
  <a href="mailto:zhang-rh25@mails.tsinghua.edu.cn">Runhan Zhang</a><sup>1</sup>,
  <a href="https://ece.hkust.edu.hk/pingtan">Ping Tan</a><sup>2</sup>,
  <a href="https://www.linkedin.com/in/ting-wu-25332618/">Ting Wu</a><sup>1</sup>,
  <a href="https://www.linkedin.com/in/tristan-ruoli-dai-b2369330/">Ruoli Dai</a><sup>1</sup>,
  <a href="https://cqf.io/">Qifeng Chen</a><sup>2,✉</sup>,
  <a href="https://www.leihan.org/">Lei Han</a><sup>1,✉</sup>
</p>

<p align="center">
  <sup>1</sup>Noitom Robotics&nbsp;&nbsp;
  <sup>2</sup>The Hong Kong University of Science and Technology&nbsp;&nbsp;
  <sup>3</sup>Wuhan University&nbsp;&nbsp;
  <sup>4</sup>The University of Hong Kong
</p>

<p align="center">
  <sup>*</sup>Equal contributors&nbsp;&nbsp;&nbsp;
  <sup>✉</sup>Corresponding authors
</p>

<p align="center">
  <a href="https://omnicontact.github.io/"><img src="https://img.shields.io/badge/Project-Page-2ea44f" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2510.xxxxx"><img src="https://img.shields.io/badge/arXiv-2510.xxxxx-b31b1b" alt="arXiv"></a>
  <a href="https://omnicontact.github.io/policy-viewer.html?v=policy-hide-push-ghostbox-20260604a"><img src="https://img.shields.io/badge/Live%20Demo-MuJoCo%20Policy%20Viewer-orange" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey" alt="License: CC BY-NC-SA 4.0">
</p>

---

<p align="center">
  <img src="docs/teaser.gif" alt="OmniContact teaser" width="90%">
</p>

OmniContact is a contact-flow framework for long-horizon humanoid loco-manipulation. The system has two main pieces:

- **CFgen** generates task-space contact-flow references for skills such as carry, push, slide, relocate, kick, and chained meta-skills.
- **CFtrack** is the low-level policy that tracks either CFgen references or full `.npz` human-object interaction motions.

This repository provides two MuJoCo execution paths:

- `deploy_omnicontact/run_skill_omnicontact.py`: direct scripted execution for CFgen or NPZmotion tracking.
- `deploy_omnicontact/deploy_omnicontact.py`: interactive hot-switch execution with an Xbox joystick, designed to mirror the state switching pattern used by sim2real deployment.

## ⚙️ Setup

Create and activate the Python environment:

```bash
conda create -n omnicontact python=3.11 -y
conda activate omnicontact
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy onnx onnxruntime mujoco pyyaml scipy pygame
```

Run commands from the repository root:

```bash
cd /path/to/OmniContact_sim2sim
```

The direct runner resolves relative model paths under `policy/omnicontact/model/`, so `--policy policy.onnx` maps to `policy/omnicontact/model/policy.onnx`.

## ▶️ Direct Runner

Use `run_skill_omnicontact.py` for scripted runs that start OmniContact immediately.

### 🧭 CFgen Reference

```bash
python deploy_omnicontact/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy policy.onnx \
  --task carrybox \
  --init-pos 1.0 0.0 \
  --goal-pos 2.5 0.5
```

Supported single-skill tasks:

| Family | Tasks |
| --- | --- |
| Locomotion | `loco` |
| Carry | `carrybox` |
| Push | `pushbox`, `pushbox-in`, `pushbox-two` |
| Slide | `slidebox`, `slidebox-left`, `slidebox-right` |
| Ball | `relocateball`, `kickball` |

`--task pushbox` is treated as an alias for `pushbox-in`.

### 🔗 Skill Chaining

```bash
python deploy_omnicontact/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy policy.onnx \
  --task-chaining carry-push \
  --init-pos 1.0 0.0 \
  --goal-pos 2.5 0.5
```

Common chain presets include `push-carry`, `carry-push`, `push-relocate`, `carry-carry`, `carry-carry-carry`, and `carryheart`.

<details>
<summary>Chain XML mapping</summary>

| Chain | Scene XML |
| --- | --- |
| `push-carry` | `g1_description/omnicontact_pushcarry_box.xml` |
| `carry-push` | `g1_description/omnicontact_pushcarry_box.xml` |
| `push-relocate` | `g1_description/omnicontact_pushrelocate_ball.xml` |
| `carry-carry` | `g1_description/omnicontact_stack_2box.xml` |
| `carry-carry-carry` | `g1_description/omnicontact_stack_3box.xml` |
| `carryheart` | `g1_description/omnicontact_heart_10box.xml` |

</details>

<br>

Extra object initialization can be provided for chained tasks:

```bash
python deploy_omnicontact/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy policy.onnx \
  --task-chaining push-carry \
  --init-pos 1.0 0.0 \
  --init-pos-extra 2.2 -0.8 \
  --goal-pos 2.5 0.5
```

### 🎞️ NPZ Motion Tracking

Use `--reference-source NPZmotion` to track a full `.npz` motion from `data/`.

```bash
python deploy_omnicontact/run_skill_omnicontact.py \
  --reference-source NPZmotion \
  --policy policy.onnx \
  --npz-dir data/relocateball/relocateball_motion_3_with_contact.npz \
  --start-frame 0
```

The runner infers XML assets from common data folders. Pass `--xml-path` to override automatic XML selection.

<details>
<summary>NPZ XML mapping</summary>

| NPZ path prefix | Scene XML |
| --- | --- |
| `data/loco` | `g1_description/omnicontact_carry_box.xml` |
| `data/carrybox` | `g1_description/omnicontact_carry_box.xml` |
| `data/pushbox` | `g1_description/omnicontact_push_box_npz.xml` |
| `data/slidebox` | `g1_description/omnicontact_slide_box_npz.xml` |
| `data/relocateball` | `g1_description/omnicontact_relocate_ball.xml` |

</details>

<br>

```bash
python deploy_omnicontact/run_skill_omnicontact.py \
  --reference-source NPZmotion \
  --xml-path g1_description/omnicontact_relocate_ball.xml \
  --policy policy.onnx \
  --npz-dir data/relocateball/relocateball_motion_3_with_contact.npz
```

<details>
<summary>Useful runner options</summary>

| Option | Meaning |
| --- | --- |
| `--headless` | Run without opening the MuJoCo GLFW viewer. |
| `--max-steps N` | Stop after `N` simulation steps. Values `<=0` mean unlimited. |
| `--stop-when-done` | Stop when CFgen reaches the end and switches back to locomotion. |
| `--start-frame N` | Inclusive start frame for NPZ motion tracking. |
| `--end-frame N` | Exclusive end frame for NPZ motion tracking. `-1` uses all remaining frames. |
| `--no-reset-env` | Disable startup environment reset. |
| `--disable-replan` | Disable CFgen replan support. |

</details>

<br>

## 🎮 Interactive Hot-Switch Deploy

`deploy_omnicontact/deploy_omnicontact.py` keeps the full FSM alive and allows hot-switching between policies with an Xbox joystick:

```text
Passive / default state -> DefaultPose -> LocoMode -> OmniContact
```

This mirrors the sim2real pattern: bring the robot to a stable default pose, switch into locomotion, then switch into the OmniContact policy at the desired moment without restarting the controller.

### ▶️ Example

```bash
python deploy_omnicontact/deploy_omnicontact.py \
  --task pushbox \
  --init-pos 3.0 0.0 0.55 \
  --goal-pos 3.0 0.0 0.15 \
  --box-half-dims 0.15 0.15 0.15
```

`deploy_omnicontact.py` automatically selects the MuJoCo XML from `--task`. Pass `--xml-path` only when you want to override this automatic mapping.

<details>
<summary>Deploy XML mapping</summary>

| Task | Scene XML |
| --- | --- |
| `carrybox` | `g1_description/omnicontact_carry_box.xml` |
| `pushbox`, `pushbox-in`, `pushbox-two` | `g1_description/omnicontact_push_box.xml` |
| `slidebox`, `slidebox-left`, `slidebox-right` | `g1_description/omnicontact_slide_box.xml` |
| `relocateball` | `g1_description/omnicontact_relocate_ball.xml` |
| `kickball`, `kickbox` | `g1_description/omnicontact_kick_ball.xml` |
| `push-carry`, `carry-push` | `g1_description/omnicontact_pushcarry_box.xml` |
| `push-relocate` | `g1_description/omnicontact_pushrelocate_ball.xml` |
| `carry-carry` | `g1_description/omnicontact_stack_2box.xml` |
| `carry-carry-carry` | `g1_description/omnicontact_stack_3box.xml` |
| `carryheart` | `g1_description/omnicontact_heart_10box.xml` |

</details>

<br>

### 🕹️ Xbox Joystick Controls

The joystick mapping is defined in `common/joystick.py`.

| Input | Runtime switch |
| --- | --- |
| `START` | `POS_RESET` / `DefaultPose` |
| `R1 + A` | `LocoMode` |
| `L1 + A` | OmniContact policy |
| `R1 + B` | Skill cooldown / auxiliary skill mode |
| `L3` | Passive |
| `SELECT` | Stop the deploy loop |

When switching to OmniContact with `L1 + A`, the script resets the active object and table references from `--init-pos`, `--goal-pos`, and `--box-half-dims`, then enters the OmniContact FSM state.

The interactive deploy visualizes:

- wrist, torso, and ankle reference mocaps;
- contact-state color changes;
- table/object references;
- ghost robot visualization when the loaded XML contains `g1_ghost.xml`.

If the ghost robot is not visible in the MuJoCo viewer, enable `group 1` rendering, since the ghost geoms are assigned to visual group 1.

## 📚 Citation

```bibtex
@article{yu2026omnicontact,
  title={OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation},
  author={Yu, Runyi and Lin, Xiaoyi and Ma, Ji and Wang, Yinhuai and Luo, Koukou and Ji, Jiahao and Wang, Huayi and Wang, Wenjia and Zhang, Runhan and Tan, Ping and Wu, Ting and Dai, Ruoli and Chen, Qifeng and Han, Lei},
  journal={arXiv preprint arXiv:coming soon},
  year={2026}
}
```
