<h1 align="center">
  OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation
</h1>

<p align="center"><strong>🎉 Accepted to CORL 2026 🎉</strong></p>

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
  <sup>2</sup>HKUST&nbsp;&nbsp;
  <sup>3</sup>Wuhan University&nbsp;&nbsp;
  <sup>4</sup>The University of Hong Kong
</p>

<p align="center">
  <sup>*</sup>Equal contributors&nbsp;&nbsp;&nbsp;
  <sup>✉</sup>Corresponding authors
</p>

<p align="center">
  <a href="https://github.com/Ingrid789/OmniContact_sim2sim"><img src="https://img.shields.io/badge/Code in Mujoco-CFGen-blue" alt="CF-Gen Mujoco Code"></a>
  <a href="https://omnicontact.github.io/"><img src="https://img.shields.io/badge/Project-Page-2ea44f" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.26201"><img src="https://img.shields.io/badge/arXiv-2606.26201-b31b1b" alt="arXiv"></a>
  <a href="https://omnicontact.github.io/policy-viewer.html?v=policy-hide-push-ghostbox-20260604a"><img src="https://img.shields.io/badge/Live%20Demo-MuJoCo%20Policy%20Viewer-orange" alt="Live Demo"></a>
  <a href="https://huggingface.co/datasets/lightcone02/OmniContact-Dataset"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-yellow" alt="Dataset"></a>
</p>

---

OmniContact chains humanoid locomotion and manipulation skills through contact
flow for long-horizon tasks. **CFgen** generates task-space references, while
**CFtrack** tracks these references or mocap HOI motions.

This repository provides **CFtrack training in Isaac Lab** for the Unitree G1,
with MLP or Transformer policies.

The **CF-Gen implementation and sim2sim code** are available in
[OmniContact_sim2sim](https://github.com/Ingrid789/OmniContact_sim2sim).

## ⚙️ Setup

Run the following commands from the repository roota.

Create and activate the Python environment:

```bash
conda create -n omnicontact_tracker python=3.10 -y
conda activate omnicontact_tracker
```

Install **PyTorch 2.7.0** and **Isaac Sim 4.5.0**:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
```

Install **Isaac Lab 2.1.1** and its extensions:

```bash
git clone --branch v2.1.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
../IsaacLab/isaaclab.sh --install none
```

Use Isaac Lab **2.1.1** in your active Python environment, including VS Code.
See the [installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/pip_installation.html) for details.

Install the local packages:

```bash
python -m pip install -e ./rsl_rl -e ./source/omnicontact
```

The bundled RSL-RL fork provides the AMP and Transformer implementations.
See [fork details](rsl_rl/UPSTREAM.md) for upstream attribution and local changes.

## 📦 Data Preparation

Download the assets from [Google Drive](https://drive.google.com/drive/folders/16eHSL6t-5u77Vhgpnjt95m_UGz8YxIMJ)
and merge them into `assets/`, keeping existing Python files.

The resulting layout should be:

```text
omnicontact/
└── assets/
    ├── data/
    │   ├── box/
    │   ├── loco/
    │   └── soccer/
    ├── g1/
    │   ├── g1_29dof.usd
    │   ├── g1_29dof.urdf
    │   └── meshes/
    ├── objects/
    └── unitree_description/
```

## 🧩 Tasks

| Training task | Algorithm | Policy |
| --- | --- | --- |
| `OmniContact` | PPO | MLP |
| `OmniContact-AMP` | PPO + AMP | MLP |
| `OmniContact-Transformer` | PPO | Transformer |
| `OmniContact-AMP-Transformer` | PPO + AMP | Transformer |

For playback, append `-play` to the training task name.
Use the same task family and policy architecture as the checkpoint.

## 🚀 Train

Start with a small run:

```bash
python scripts/rsl_rl/train.py \
  --task OmniContact-AMP --num_envs 64 --max_iterations 10 \
  --motion_file_dir assets/data/box/case2_push/push \
  --disable_wandb --headless
```

- Increase `--num_envs` and `--max_iterations` for full training.
- `--motion_file_dir` accepts an NPZ file or a directory searched recursively.
- AMP uses the tracking motions as expert data by default;
  `--amp_data_dir /path/to/expert/motions` selects a separate dataset.
- `--disable_wandb` uses TensorBoard. Logs and checkpoints are saved under
  `logs/rsl_rl/<experiment>/<run>/`.

To resume, add `--resume --load_run '<run>' --checkpoint model_4999.pt`.

## ▶️ Play and export

```bash
python scripts/rsl_rl/play.py \
  --task OmniContact-AMP-play --num_envs 1 \
  --motion_file_dir assets/data/box/case2_push/push \
  --checkpoint /path/to/model_4999.pt \
  --disable_wandb --start_frame 0 --stop_on_reset
```

Playback also exports `exported/policy.onnx` beside the checkpoint.
Reference motions are required for playback. Select a motion file or subset
under `assets/data/` that matches the checkpoint's task.

## 🧪 Tests

Run in the same environment without launching Isaac Sim:

```bash
python -m pytest tests -q
```

Tests cover motion loading and asset labels, PPO/AMP, checkpoints and ONNX export.
The distributed test requires local socket access.

## 🔗 Sim2Sim

The **CF-Gen implementation and sim2sim code** are available in
[OmniContact_sim2sim](https://github.com/Ingrid789/OmniContact_sim2sim).
