<h1 align="center">
  <span style="color:#ef5b5b">OmniContact</span>: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation
</h1>

<p align="center">
  <a href="https://ingrid789.github.io/IngridYu/">Runyi Yu</a><sup>1,2,*</sup>,
  <a href="https://github.com/XiaoyiLin-code">Xiaoyi Lin</a><sup>1,3,*</sup>,
  <a href="https://astrorix.github.io/">Ji Ma</a><sup>1</sup>,
  <a href="https://wyhuai.github.io/info/">Yinhuai Wang</a><sup>2,✉</sup>,
  <a href="https://www.linkedin.com/in/koukou-luo-801275295/">Koukou Luo</a><sup>2</sup>,
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
  <a href="https://arxiv.org/abs/2510.11072"><img src="https://img.shields.io/badge/arXiv-2510.11072-b31b1b" alt="arXiv"></a>
  <a href="https://omnicontact.github.io/policy-viewer.html?v=policy-hide-push-ghostbox-20260604a"><img src="https://img.shields.io/badge/Live%20Demo-MuJoCo%20Policy%20Viewer-orange" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey" alt="License: CC BY-NC-SA 4.0">
</p>

---

OmniContact is a long-horizon human-object interaction system built from two parts:

- **CFgen**: a rule-based contact-flow trajectory synthesis method. It generates reference trajectories for meta-skills such as carry, push, slide, relocate, and kick.
- **CFtrack**: the low-level tracking policy. It tracks either CFgen-generated references or a complete HOI `.npz` motion.

In this repository, **CFgen + CFtrack** is the full OmniContact pipeline, while **CFtrack** alone can replay a precomputed HOI `.npz` trajectory from `data/`.

## 1. Install Environment

Create and activate the conda environment:

```bash
conda create -n nair_sim python=3.11 -y
conda activate nair_sim
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy onnx onnxruntime mujoco pyyaml scipy
```

Install this repository:

```bash
cd /home/lenovo/NR/omnicontact_sim2sim-real
pip install -e .
```

Optional, for real-robot deployment:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

## 2. Inference Meta-Skill With CFgen

Use `reference-source=CFgen` to synthesize a single meta-skill reference with CFgen and track it with CFtrack.

Example: carry a box from `(2, 0)` to `(5, 1)`:

```bash
python deploy_carrybox_manager/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy actorcritic_transformer_45k.onnx \
  --task carrybox \
  --init-pos 2 0 \
  --goal-pos 5 1
```

Other CFgen meta-skills:

```bash
# Push box
python deploy_carrybox_manager/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy actorcritic_transformer_45k.onnx \
  --task pushbox-two \
  --init-pos 2 0 \
  --goal-pos 5 1

# Slide box left
python deploy_carrybox_manager/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy actorcritic_transformer_45k.onnx \
  --task slidebox-left \
  --init-pos 2 0 \
  --goal-pos 5 1

# Kick ball
python deploy_carrybox_manager/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy actorcritic_transformer_45k.onnx \
  --task kickball \
  --init-pos 2 0 \
  --goal-pos 5 1
```

The object half dimensions are loaded from the selected MuJoCo XML automatically. Use `--box-half-dims HX HY HZ` only when you want to override the XML dimensions.

## 3. Inference Skill Chaining With CFgen

Skill chaining uses CFgen to plan a sequence of meta-skills and CFtrack to execute each stage.

Example: carry first, then push:

```bash
python deploy_carrybox_manager/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy actorcritic_transformer_45k.onnx \
  --task-chaining carry-push \
  --init-pos 2 0 \
  --goal-pos 5 1
```

Supported chains:

```bash
--task-chaining push-carry
--task-chaining carry-push
--task-chaining carry-carry
--task-chaining carry-carry-carry
```

For chained tasks, the push-box and carry-box half dimensions are loaded separately from the task XML. For example, `push-carry` and `carry-push` use `g1_description/omnicontact_pushcarry_box.xml`, where `push_box` and `carry_box` have distinct sizes.

Optional extra-object initialization:

```bash
python deploy_carrybox_manager/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy actorcritic_transformer_45k.onnx \
  --task-chaining push-carry \
  --init-pos 2 0 \
  --init-pos-extra 3 -1 \
  --goal-pos 5 1
```

## 4. Inference With CFtrack

Use `reference-source=NPZmotion` to run CFtrack directly on a complete HOI `.npz` trajectory. This bypasses CFgen and tracks the provided motion.

Example using a motion under `data/`:

```bash
python deploy_carrybox_manager/run_skill_omnicontact.py \
  --reference-source NPZmotion \
  --xml-path g1_description/omnicontact_kick_ball.xml \
  --policy actorcritic_transformer_18k.onnx \
  --npz-dir data/punt_data_slow2/ball10_092_with_contact_auto_seg01.npz \
  --start-frame 0
```

You can also pass a directory; the runner will use the first `.npz` file in that directory:

```bash
python deploy_carrybox_manager/run_skill_omnicontact.py \
  --reference-source NPZmotion \
  --xml-path g1_description/omnicontact_kick_ball.xml \
  --policy actorcritic_transformer_18k.onnx \
  --npz-dir data/punt_data_slow2 \
  --start-frame 0 \
  --end-frame 1000
```

Common options:

```bash
--headless          # run without the MuJoCo viewer
--max-steps N       # stop after N simulation steps
--stop-when-done    # exit when the policy reports task completion
--randomize         # apply random horizontal disturbance to the active object
```
