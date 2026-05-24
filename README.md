# HSI / Pre-Grasp Refusal

Simulator scaffold for **Pre-Grasp Refusal: Selective Risk Control at the Gripper-Closure Decision Point for VLA Manipulation**.

The robot decides, immediately before gripper closure, whether to:
- **ACCEPT_CLOSE** — commit to grasping
- **REFUSE_CLOSE** — abort and re-approach

**Key finding:** In the language wrong-object benchmark, frozen DINOv2 + language (`dino_language`) achieves **0.074 FAR** and **zero wrong-object false accepts**, beating oracle geometry (0.291 FAR) by a large margin. No-oracle camera + language (`visual_language`) also beats all oracle geometry baselines at matched acceptance.

---

## Results at a Glance

### Headline: Language Wrong-Object Benchmark

*Main claim: no-oracle learned heads beat oracle geometry at matched acceptance.*

| Method | Oracle? | FAR | Wrong-Obj FAR | Acceptance |
|--------|:-------:|----:|-------------:|----------:|
| **`dino_language`** (Phase 3, DINOv2) | **no** | **`0.0741`** | **`0.0000`** | `0.1350` |
| `visual_language` (camera summaries, 600/400) | no | `0.0900` | `0.0000` | `0.5000` |
| `visual_proxy_language` (CSV-only proxy) | no | `0.1700` | `0.0000` | `0.5000` |
| `oracle_geometry_upper_bound` (DINOv2 holdout) | yes | `0.2907` | `0.2907` | `0.5029` |
| `distance_only` (oracle, 600/400 holdout) | yes | `0.3700` | `0.3700` | `0.5000` |
| `lateral_error_only` (oracle) | yes | `0.3750` | `0.3750` | `0.5000` |
| Always Close | no | `0.4561–0.5125` | `0.2877–0.3600` | `1.0000` |

*Rows marked `yes` use simulator pose/geometry unavailable on a real robot — oracle upper bounds only.*

### Online: Language Wrong-Object

| Policy | Oracle? | Episodes | Ep. Success | FAR | Wrong-Obj Accepted | Acceptance |
|--------|:-------:|--------:|------------:|----:|------------------:|-----------:|
| Always Close | no | `112` | `0.3393` | `0.4795` | `21` | `1.0000` |
| **Visual + Language Refusal** | **no** | `112` | `0.4107` | `0.1154` | **`0`** | `0.4444` |
| Full Language Refusal | yes | `112` | `0.3839` | `0.0444` | `0` | `0.3516` |

---

### State-Only Holdout Baselines (400 events)

| Method | FAR | Accepted Success | Acceptance |
|--------|----:|-----------------:|-----------:|
| Always Close | `0.2475` | `0.7525` | `1.0000` |
| Matched Random | `0.2475` | `0.7525` | `0.8345` |
| Distance Only | `0.0988` | `0.9012` | `0.8350` |
| Lateral Error Only | `0.0988` | `0.9012` | `0.8350` |
| Geometry Learned Head | `0.0961` | `0.9039` | `0.8325` |
| Full Learned Head | `0.0988` | `0.9012` | `0.8350` |
| Action/State Head | `0.2000` | `0.8000` | `0.0875` |

### Online: State-Only

| Policy | Episodes | Ep. Success | FAR | Acceptance | Time/Success | Attempts/Success |
|--------|--------:|------------:|----:|-----------:|-------------:|-----------------:|
| Always Close | `448` | `0.5246` | `0.2610` | `1.0000` | `9.53 s` | `1.35` |
| Refusal + Reapproach | `448` | `0.5670` | `0.1119` | `0.8266` | `8.82 s` | `1.13` |

### Approach-Noise Robustness (state-only)

| Noise Std | Always-Close FAR | Refusal FAR | Acceptance |
|:---------:|----------------:|------------:|-----------:|
| `0.01` easier | `0.0075` | `0.0050` | `0.9975` |
| `0.02` main | `0.2475` | `0.0988` | `0.8350` |
| `0.03` harder | `0.5325` | `0.1355` | `0.5350` |

---

### VLA-Sim Feature Groups (single object, 600 train / 400 holdout)

| Feature Group | FAR | Accepted Success | Acceptance |
|:-------------:|----:|-----------------:|-----------:|
| `visual` | `0.1104` | `0.8896` | `0.7925` |
| `robot_visual` | `0.1351` | `0.8649` | `0.8325` |
| `full` | `0.1529` | `0.8471` | `0.8500` |
| `robot_state` | `0.1628` | `0.8372` | `0.8600` |
| `vla_action_uncertainty` | `0.2500` | `0.7500` | `0.1100` |

### Online: Single-Object VLA-Sim

| Policy | Episodes | Ep. Success | FAR | Acceptance | Time/Success | Attempts/Success |
|--------|--------:|------------:|----:|-----------:|-------------:|-----------------:|
| Always Close | `80` | `0.5375` | `0.2182` | `1.0000` | `9.30 s` | `1.28` |
| Visual Refusal + Reapproach | `80` | `0.5875` | `0.1132` | `0.8833` | `8.51 s` | `1.13` |

---

### Phase 3: DINOv2 Feature-Group Ablation (300 train / 171 holdout)

| Feature Group | Oracle? | FAR | Accepted Success | Wrong-Obj FAR | Acceptance |
|:-------------:|:-------:|----:|-----------------:|-------------:|-----------:|
| **`dino_language`** | **no** | **`0.0741`** | `0.9259` | **`0.0000`** | `0.1350` |
| `geometry_dino_language` | yes | `0.0789` | `0.9211` | `0.0000` | `0.1900` |
| `visual_language` | no | `0.1000` | `0.9000` | `0.0000` | `0.5500` |
| `robot_state` | yes | `0.2609` | `0.7391` | `0.0000` | `0.1150` |
| `visual` | no | `0.5000` | `0.5000` | `0.0000` | `0.0300` |
| `dino` (no language) | no | `1.0000` | `0.0000` | `0.0000` | `0.0100` |
| `language` (no vision) | no | n/a | n/a | n/a | `0.0000` |

### DINOv2 Holdout Proxy Baselines (171 events, matched to 0.50 acceptance)

| Method | Oracle? | FAR | Wrong-Obj FAR | Acceptance |
|--------|:-------:|----:|-------------:|-----------:|
| Always Close | no | `0.4561` | `0.3216` | `1.0000` |
| Estimated Geometry Proxy | no | `0.4070` | `0.3140` | `0.5029` |
| Oracle Geometry Upper Bound | yes | `0.2907` | `0.2907` | `0.5029` |

---

### Robustness Sweeps (`robot_visual` checkpoint, main600 trained)

| Shift | FAR | Accepted Success | Acceptance | Refusal |
|-------|----:|-----------------:|-----------:|--------:|
| Partial occlusion | `0.0476` | `0.9524` | `0.8400` | `0.1600` |
| Camera shift | `0.0676` | `0.9324` | `0.7400` | `0.2600` |
| Object shift | `0.0411` | `0.9589` | `0.7300` | `0.2700` |
| Clutter | `0.0612` | `0.9388` | `0.4900` | `0.5100` |
| Approach noise 0.03 | `0.1163` | `0.8837` | `0.4300` | `0.5700` |
| Lighting shift | `0.0000` | `1.0000` | `0.0700` | `0.9300` |

*Lighting shift causes near-total refusal — camera features are brittle to illumination changes.*

---

## Project Structure

| File | Role |
|------|------|
| `hsi_pregrasp_refusal/trigger.py` | Deterministic pre-grasp trigger |
| `hsi_pregrasp_refusal/features.py` | Feature column definitions (incl. DINOv2/CLIP groups) |
| `hsi_pregrasp_refusal/model.py` | MLP `RefusalHead` + `TargetAwareRefusalHead` with aux heads |
| `hsi_pregrasp_refusal/calibration.py` | Threshold calibration to target FAR |
| `hsi_pregrasp_refusal/metrics.py` | Selective close/refuse metrics |
| `hsi_pregrasp_refusal/vision.py` | Camera summaries + `LearnedEmbeddingExtractor` (DINOv2/CLIP) |
| `hsi_pregrasp_refusal/vla.py` | SmolVLA adapter + sampled action-uncertainty features |
| `hsi_pregrasp_refusal/sim_analysis.py` | Language/proxy-baseline/failure-type summaries |
| `hsi_pregrasp_refusal/isaaclab_vla_scene.py` | Camera-enabled lift scene + simulator variants |
| `hsi_pregrasp_refusal/state_machine.py` | Shared Franka lift state machine |
| `scripts/collect_lift_cube_pregrasp.py` | State-only event collector |
| `scripts/collect_vla_lift_pregrasp.py` | Camera/VLA event collector (wrong-object, DINOv2, etc.) |
| `scripts/train_refusal_head.py` | Train + calibrate refusal head |
| `scripts/evaluate_refusal.py` | Evaluate a trained checkpoint |
| `scripts/run_feature_group_ablation.py` | Feature-group training/evaluation |
| `scripts/run_matched_acceptance_ci.py` | Bootstrap CI at matched acceptance |
| `scripts/run_simulation_proxy_baselines.py` | Always-close / geometry-proxy / oracle baselines |
| `scripts/run_online_vla_refusal_eval.py` | Online refusal + reapproach evaluation |
| `scripts/run_offline_baselines.py` | Offline scalar and matched-random baselines |
| `tests/test_refusal_core.py` | Lightweight unit tests |

---

## Environment

```bash
conda run -n env_isaaclab python -c "import sys; print(sys.executable)"
```

- Python: `/home/chyanin/miniconda3/envs/env_isaaclab/bin/python 3.11.14`
- PyTorch: `2.7.0+cu128`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- NVIDIA driver: `580.159.03` — driver `595.71.05` crashes the RTX renderer on camera-enabled jobs
- Isaac Sim graphics API: Vulkan
- Prefix headless commands with `TERM=xterm` if terminal reset warnings appear

Extra packages for SmolVLA runtime:

```
lerobot draccus accelerate mergedeep typing-inspect mypy-extensions datasets dill
multiprocess xxhash pyarrow httpx diffusers pyserial deepdiff orderly-set cachebox av num2words
```

---

## Experiments

### 1. State-Only Scaffold

**Purpose:** baseline with pure robot-state features — no cameras, no VLA.

<details>
<summary>Collection commands</summary>

```bash
# Stage 0 smoke (30 events, visible GUI)
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --num_envs 16 --num_events 30 --approach_noise_std 0.02 \
  --output logs/hsi_pregrasp/stage0_events.csv

# Main 800-event headless
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --headless --num_envs 64 --num_events 800 \
  --approach_noise_std 0.02 --seed 42 \
  --output logs/hsi_pregrasp/main800_events.csv

# Independent 400-event holdout
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --headless --num_envs 64 --num_events 400 \
  --approach_noise_std 0.02 --seed 99 \
  --output logs/hsi_pregrasp/holdout400_events.csv

# Approach-noise robustness shards
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --headless --num_envs 64 --num_events 400 --approach_noise_std 0.01 --seed 201 \
  --output logs/hsi_pregrasp/shift_noise001_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --headless --num_envs 64 --num_events 400 --approach_noise_std 0.03 --seed 202 \
  --output logs/hsi_pregrasp/shift_noise003_events.csv
```

</details>

<details>
<summary>Training and evaluation commands</summary>

```bash
# Train on main800
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/train_refusal_head.py \
  --input logs/hsi_pregrasp/main800_events.csv \
  --output logs/hsi_pregrasp/main800_refusal_head.pt \
  --target_false_accept_risk 0.10 --epochs 300 --seed 42 --device cpu

# Evaluate on holdout
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/evaluate_refusal.py \
  --input logs/hsi_pregrasp/holdout400_events.csv \
  --checkpoint logs/hsi_pregrasp/main800_refusal_head.pt --device cpu

# Offline baselines (distance-only, random, geometry-head, full-head, action/state)
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_offline_baselines.py \
  --input logs/hsi_pregrasp/main800_events.csv \
  --eval-input logs/hsi_pregrasp/holdout400_events.csv \
  --output logs/hsi_pregrasp/offline_baselines.json \
  --seed 42 --matched_acceptance_rate 0.835

# Online refusal evaluation
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/run_online_refusal_eval.py \
  --headless --checkpoint logs/hsi_pregrasp/main800_refusal_head.pt \
  --num_envs 64 --num_episodes 400 --approach_noise_std 0.02 \
  --retry_noise_std 0.02 --seed 123 \
  --output logs/hsi_pregrasp/online_refusal_events.csv \
  --summary logs/hsi_pregrasp/online_refusal_summary.json
```

</details>

Dataset summary:

| Dataset | Events | Successes | Failures | Always-Close FAR |
|---------|-------:|----------:|---------:|-----------------:|
| Stage 0 smoke | `30` | `21` | `9` | `0.3000` |
| Main | `800` | `588` | `212` | `0.2650` |
| Holdout | `400` | `301` | `99` | `0.2475` |

---

### 2. SmolVLA Smoke Test

```bash
# Download
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/download_smolvla_assets.py \
  --output-dir logs/hsi_pregrasp/vla/smolvla_base

# Runtime smoke
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_smolvla_smoke.py \
  --checkpoint logs/hsi_pregrasp/vla/smolvla_base \
  --num_samples 8 --output logs/hsi_pregrasp/vla/smolvla_smoke_summary.json
```

| Check | Value |
|-------|-------|
| Checkpoint size | `865 MB` |
| Output shape | `[8, 6]` |
| Mean action variance | `0.0065883` |
| Mean VLA inference | `353 ms/event` |
| Underlying VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |

---

### 3. Camera + VLA — Single Object (600/400 scale)

**Purpose:** camera-enabled VLA-sim with `robot_state`, `visual`, `vla_action_uncertainty`, `robot_visual`, `full` feature groups.

<details>
<summary>Collection commands</summary>

```bash
# Proposal-scale main (600 events, vla_samples=2)
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --num_envs 8 --num_events 600 \
  --approach_noise_std 0.02 --vla_samples 2 --seed 142 \
  --max_steps 80000 --stall_steps 5000 \
  --output logs/hsi_pregrasp/vla/main600_vla_events.csv

# Proposal-scale holdout (400 events, vla_samples=2)
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --num_envs 8 --num_events 400 \
  --approach_noise_std 0.02 --vla_samples 2 --seed 299 \
  --max_steps 60000 --stall_steps 5000 \
  --output logs/hsi_pregrasp/vla/holdout400_vla_events.csv
```

</details>

<details>
<summary>Training and evaluation commands</summary>

```bash
# Feature-group ablation
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_feature_group_ablation.py \
  --input logs/hsi_pregrasp/vla/main600_vla_events.csv \
  --eval-input logs/hsi_pregrasp/vla/holdout400_vla_events.csv \
  --output logs/hsi_pregrasp/vla/feature_group_ablations_main600.json \
  --checkpoint-dir logs/hsi_pregrasp/vla/checkpoints_main600 \
  --epochs 300 --seed 42 --device cpu

# No-oracle matched-acceptance CI
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_matched_acceptance_ci.py \
  --input logs/hsi_pregrasp/vla/holdout400_vla_events.csv \
  --title "No-Oracle Holdout400 Camera-Only" \
  --reference-checkpoint visual=logs/hsi_pregrasp/vla/checkpoints_main600/visual_refusal_head.pt \
  --checkpoint vla_uncertainty=logs/hsi_pregrasp/vla/checkpoints_main600/vla_action_uncertainty_refusal_head.pt \
  --scalar oracle_distance_only=ee_object_distance:low \
  --include-always-close --include-matched-random \
  --bootstrap 2000 --seed 314 --device cpu \
  --output-json logs/hsi_pregrasp/vla/no_oracle_holdout400_matched_ci.json \
  --output-md logs/hsi_pregrasp/vla/no_oracle_holdout400_matched_ci.md

# Online visual refusal
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/run_online_vla_refusal_eval.py \
  --headless --checkpoint logs/hsi_pregrasp/vla/checkpoints_main600/visual_refusal_head.pt \
  --num_envs 8 --num_episodes 80 --approach_noise_std 0.02 \
  --retry_noise_std 0.02 --skip_vla --seed 501 --max_steps 12000 \
  --output logs/hsi_pregrasp/vla/online_visual_main600_events.csv \
  --summary logs/hsi_pregrasp/vla/online_visual_main600_summary.json
```

</details>

Dataset summary:

| Dataset | Events | Successes | Failures | Mean VLA ms/Event |
|---------|-------:|----------:|---------:|------------------:|
| `main600_vla_events.csv` | `600` | `452` | `148` | `331.3` |
| `holdout400_vla_events.csv` | `400` | `288` | `112` | `330.3` |

---

### 4. Language Wrong-Object Benchmark — Camera Summaries (600/400 scale)

**Purpose:** scene has default cube + 4 colored distractors; policy always grasps default cube; language sometimes asks for a colored cube → automatic wrong-object labels. First simulator condition where no-oracle visual+language beats oracle geometry.

<details>
<summary>Collection commands (4 shards merged into 600/400)</summary>

```bash
# Train shards A + B (300 events each; merge into main600_camera)
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --device cuda:0 --num_envs 16 --num_events 300 \
  --variant distractors --num_distractors 4 --skip_vla \
  --language_mode multi_object_default_policy --language_default_prob 0.65 \
  --seed 922 --approach_noise_std 0.02 --label_horizon_s 1.0 \
  --rendering_mode performance --max_steps 50000 --stall_steps 2500 \
  --output logs/hsi_pregrasp/vla/language_wrong_object_train300_camera_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --device cuda:0 --num_envs 16 --num_events 300 \
  --variant distractors --num_distractors 4 --skip_vla \
  --language_mode multi_object_default_policy --language_default_prob 0.65 \
  --seed 924 --approach_noise_std 0.02 --label_horizon_s 1.0 \
  --rendering_mode performance --max_steps 50000 --stall_steps 2500 \
  --output logs/hsi_pregrasp/vla/language_wrong_object_train300b_camera_events.csv

# Holdout shards A + B (200 events each; merge into holdout400_camera)
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --device cuda:0 --num_envs 16 --num_events 200 \
  --variant distractors --num_distractors 4 --skip_vla \
  --language_mode multi_object_default_policy --language_default_prob 0.65 \
  --seed 923 --approach_noise_std 0.02 --label_horizon_s 1.0 \
  --rendering_mode performance --max_steps 50000 --stall_steps 2500 \
  --output logs/hsi_pregrasp/vla/language_wrong_object_holdout200_camera_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --device cuda:0 --num_envs 16 --num_events 200 \
  --variant distractors --num_distractors 4 --skip_vla \
  --language_mode multi_object_default_policy --language_default_prob 0.65 \
  --seed 925 --approach_noise_std 0.02 --label_horizon_s 1.0 \
  --rendering_mode performance --max_steps 50000 --stall_steps 2500 \
  --output logs/hsi_pregrasp/vla/language_wrong_object_holdout200b_camera_events.csv
```

Merge: shards A+B → `language_wrong_object_main600_camera_events.csv` and `language_wrong_object_holdout400_camera_events.csv`.

</details>

<details>
<summary>Training and evaluation commands</summary>

```bash
# Feature-group ablation
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_feature_group_ablation.py \
  --input logs/hsi_pregrasp/vla/language_wrong_object_main600_camera_events.csv \
  --eval-input logs/hsi_pregrasp/vla/language_wrong_object_holdout400_camera_events.csv \
  --output logs/hsi_pregrasp/vla/language_wrong_object_camera_main600_ablation.json \
  --checkpoint-dir logs/hsi_pregrasp/vla/checkpoints_language_camera_main600 \
  --feature-groups robot_state,visual,language,visual_language,robot_visual,robot_visual_language,full,full_language,vla_action_uncertainty \
  --epochs 200 --seed 74 --device cpu

# Matched-acceptance CI (reference: visual_language at 0.50 acceptance)
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_matched_acceptance_ci.py \
  --input logs/hsi_pregrasp/vla/language_wrong_object_holdout400_camera_events.csv \
  --title "Language Wrong-Object Camera Main600 Holdout400" \
  --reference-checkpoint visual_language=logs/hsi_pregrasp/vla/checkpoints_language_camera_main600/visual_language_refusal_head.pt \
  --checkpoint language=logs/hsi_pregrasp/vla/checkpoints_language_camera_main600/language_refusal_head.pt \
  --checkpoint full_language=logs/hsi_pregrasp/vla/checkpoints_language_camera_main600/full_language_refusal_head.pt \
  --scalar distance_only=ee_object_distance:low \
  --scalar lateral_error_only=ee_object_lateral_error:low \
  --include-always-close --include-matched-random \
  --bootstrap 2000 --seed 44 --device cpu \
  --output-json logs/hsi_pregrasp/vla/language_wrong_object_camera_main600_matched_ci.json \
  --output-md logs/hsi_pregrasp/vla/language_wrong_object_camera_main600_matched_ci.md

# CSV-only visual-proxy ablation
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_feature_group_ablation.py \
  --input logs/hsi_pregrasp/vla/language_wrong_object_main600_camera_events.csv \
  --eval-input logs/hsi_pregrasp/vla/language_wrong_object_holdout400_camera_events.csv \
  --output logs/hsi_pregrasp/vla/language_wrong_object_visual_proxy_ablation_main600.json \
  --output-md logs/hsi_pregrasp/vla/language_wrong_object_visual_proxy_ablation_main600.md \
  --checkpoint-dir logs/hsi_pregrasp/vla/checkpoints_language_visual_proxy_main600 \
  --feature-groups language,visual_proxy,visual_proxy_language,visual_language \
  --epochs 300 --seed 42 --device cpu

# Proxy geometry baselines
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_simulation_proxy_baselines.py \
  --input logs/hsi_pregrasp/vla/language_wrong_object_holdout400_camera_events.csv \
  --target-acceptance-rate 0.50 \
  --output-json logs/hsi_pregrasp/vla/language_wrong_object_camera_main600_proxy_baselines.json \
  --output-md logs/hsi_pregrasp/vla/language_wrong_object_camera_main600_proxy_baselines.md

# Online: always-close baseline
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/run_online_vla_refusal_eval.py \
  --headless --device cuda:0 --decision_mode always_close \
  --num_envs 16 --num_episodes 100 \
  --variant distractors --num_distractors 4 --skip_vla \
  --language_mode multi_object_default_policy --language_default_prob 0.65 \
  --seed 1022 --approach_noise_std 0.02 --retry_noise_std 0.02 \
  --label_horizon_s 1.0 --rendering_mode performance --max_steps 50000 \
  --output logs/hsi_pregrasp/vla/online_language_wrong_object_main600_always_close_events.csv \
  --summary logs/hsi_pregrasp/vla/online_language_wrong_object_main600_always_close_summary.json

# Online: no-oracle visual+language refusal
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/run_online_vla_refusal_eval.py \
  --headless --device cuda:0 --decision_mode model \
  --checkpoint logs/hsi_pregrasp/vla/checkpoints_language_camera_main600/visual_language_refusal_head.pt \
  --num_envs 16 --num_episodes 100 \
  --variant distractors --num_distractors 4 --skip_vla \
  --language_mode multi_object_default_policy --language_default_prob 0.65 \
  --seed 1023 --approach_noise_std 0.02 --retry_noise_std 0.02 \
  --max_refusals_per_episode 3 --label_horizon_s 1.0 \
  --rendering_mode performance --max_steps 50000 \
  --output logs/hsi_pregrasp/vla/online_language_wrong_object_main600_visual_language_events.csv \
  --summary logs/hsi_pregrasp/vla/online_language_wrong_object_main600_visual_language_summary.json
```

</details>

Dataset summary:

| Dataset | Events | Successes | Failures | Wrong-Object Failures |
|---------|-------:|----------:|---------:|---------------------:|
| `main600_camera` | `600` | `280` | `320` | `171` |
| `holdout400_camera` | `400` | `195` | `205` | `116` |
| `vla40` (SmolVLA subset) | `40` | `22` | `18` | `9` |

Matched-acceptance CI on the 400-event holdout (matched to `visual_language` acceptance of `0.50`):

| Method | Oracle? | FAR, 95% CI | Wrong-Obj FAR, 95% CI | Acceptance |
|--------|:-------:|------------:|---------------------:|-----------:|
| `visual_language` | no | `0.0900` [`0.0503`, `0.1337`] | `0.0000` [`0.0000`, `0.0000`] | `0.5000` |
| `language` | no | `0.2450` [`0.1872`, `0.3069`] | `0.0000` [`0.0000`, `0.0000`] | `0.5000` |
| `distance_only` | yes | `0.3700` [`0.3028`, `0.4369`] | `0.3700` [`0.3022`, `0.4352`] | `0.5000` |
| `lateral_error_only` | yes | `0.3750` [`0.3073`, `0.4466`] | `0.3750` [`0.3092`, `0.4410`] | `0.5000` |
| `robot_visual_language` | yes | `0.0400` [`0.0153`, `0.0688`] | `0.0050` [`0.0000`, `0.0157`] | `0.5000` |
| `full_language` | yes | `0.0350` [`0.0109`, `0.0625`] | `0.0100` [`0.0000`, `0.0254`] | `0.5000` |
| Matched Random | no | `0.5128` [`0.4650`, `0.5650`] | `0.3604` [`0.3150`, `0.4100`] | `0.5000` |

---

### 5. Phase 3: DINOv2 Learned Embeddings

**Purpose:** replace camera summary statistics with frozen DINOv2-S CLS tokens (384-dim per camera, 1536 total columns). `dino_language` achieves the best no-oracle FAR of all groups.

New code:
- `vision.py`: `LearnedEmbeddingExtractor` — frozen DINOv2-S or CLIP, columns named `dinov2_{cam}_dim{i}` and `dinov2_global_dim{i}`
- `features.py`: `DINO_EMBED_DIM=384`; 18 new feature groups including `dino`, `dino_language`, `geometry_dino_language`
- `model.py`: `TargetAwareRefusalHead` — shared encoder + auxiliary `wrong_object`, `occlusion`, `geometric` binary heads
- `collect_vla_lift_pregrasp.py`: `--embedding_model {none,dinov2,clip,both}` and `--embedding_device` flags
- `train_refusal_head.py`: `--aux_heads`, `--aux_loss_weight`, `--weight_decay` flags

<details>
<summary>Collection commands</summary>

```bash
# DINOv2 train (300 events)
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --device cuda:0 --num_envs 16 --num_events 300 \
  --variant distractors --num_distractors 4 --skip_vla \
  --language_mode multi_object_default_policy --language_default_prob 0.65 \
  --seed 930 --approach_noise_std 0.02 --label_horizon_s 1.0 \
  --rendering_mode performance --max_steps 50000 --stall_steps 2500 \
  --embedding_model dinov2 --embedding_device cuda \
  --output logs/hsi_pregrasp/vla/language_wrong_object_dino_train300_events.csv

# DINOv2 holdout (200 target, 171 collected)
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --device cuda:0 --num_envs 16 --num_events 200 \
  --variant distractors --num_distractors 4 --skip_vla \
  --language_mode multi_object_default_policy --language_default_prob 0.65 \
  --seed 931 --approach_noise_std 0.02 --label_horizon_s 1.0 \
  --rendering_mode performance --max_steps 40000 --stall_steps 2500 \
  --embedding_model dinov2 --embedding_device cuda \
  --output logs/hsi_pregrasp/vla/language_wrong_object_dino_holdout200_events.csv
```

</details>

<details>
<summary>Training and evaluation commands</summary>

```bash
# Feature-group ablation
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_feature_group_ablation.py \
  --input logs/hsi_pregrasp/vla/language_wrong_object_dino_train300_events.csv \
  --eval-input logs/hsi_pregrasp/vla/language_wrong_object_dino_holdout200_events.csv \
  --output logs/hsi_pregrasp/vla/dino_ablation_main300.json \
  --output-md logs/hsi_pregrasp/vla/dino_ablation_main300.md \
  --checkpoint-dir logs/hsi_pregrasp/vla/checkpoints_dino_main300 \
  --feature-groups robot_state,visual,language,visual_language,dino,dino_language,geometry_dino_language \
  --epochs 300 --seed 42 --device cuda

# Proxy geometry baselines at 0.50 acceptance
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_simulation_proxy_baselines.py \
  --input logs/hsi_pregrasp/vla/language_wrong_object_dino_holdout200_events.csv \
  --target-acceptance-rate 0.50 \
  --output-json logs/hsi_pregrasp/vla/dino_holdout158_proxy_baselines.json \
  --output-md logs/hsi_pregrasp/vla/dino_holdout158_proxy_baselines.md
```

</details>

Dataset summary:

| Dataset | Events | Successes | Failures | Wrong-Object Failures | DINO Cols | Nonzero Rows |
|---------|-------:|----------:|---------:|---------------------:|----------:|-------------:|
| `dino_train300` | `300` | `138` | `162` | `111` | `1536` | `300` |
| `dino_holdout` | `171` | `93` | `78` | `47` | `1536` | `171` |

**Key takeaways:**
- `dino_language` beats oracle geometry by `0.217` FAR while accepting zero wrong-object closes.
- `dino` without language collapses to 1% acceptance — the language conditioning is essential.
- All language-conditioned heads achieve 0.000 wrong-object FAR, confirming the 600/400 finding on richer features.

---

### 6. VLA Robustness Sweeps

**Purpose:** evaluate main600 checkpoints on shifted 100-event datasets.

<details>
<summary>Collection commands</summary>

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --num_envs 8 --num_events 100 --variant partial_occlusion \
  --num_distractors 1 --skip_vla --seed 606 --approach_noise_std 0.02 \
  --max_steps 20000 --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_partial_occlusion100_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --num_envs 8 --num_events 100 --variant clutter \
  --num_distractors 3 --skip_vla --seed 601 --approach_noise_std 0.02 \
  --max_steps 20000 --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_clutter100_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --num_envs 8 --num_events 100 --variant lighting_shift \
  --skip_vla --seed 602 --approach_noise_std 0.02 \
  --max_steps 20000 --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_lighting100_full_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --num_envs 8 --num_events 100 --variant camera_shift \
  --skip_vla --seed 603 --approach_noise_std 0.02 \
  --max_steps 20000 --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_camera_shift100_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --num_envs 8 --num_events 100 --variant object_shift \
  --skip_vla --seed 604 --approach_noise_std 0.02 \
  --max_steps 20000 --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_object_shift100_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless --num_envs 8 --num_events 100 --variant single \
  --skip_vla --seed 605 --approach_noise_std 0.03 \
  --max_steps 20000 --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_approach_noise003_100_events.csv
```

</details>

Offline robustness (all `robot_visual` unless otherwise shown):

| Feature Group | Shift | Events | FAR | Accepted Success | Acceptance | Refusal |
|:-------------:|-------|-------:|----:|-----------------:|-----------:|--------:|
| `robot_visual` | partial occlusion | `100` | `0.0476` | `0.9524` | `0.8400` | `0.1600` |
| `robot_visual` | camera shift | `100` | `0.0676` | `0.9324` | `0.7400` | `0.2600` |
| `robot_visual` | object shift | `100` | `0.0411` | `0.9589` | `0.7300` | `0.2700` |
| `robot_visual` | clutter | `100` | `0.0612` | `0.9388` | `0.4900` | `0.5100` |
| `robot_visual` | approach 0.03 | `100` | `0.1163` | `0.8837` | `0.4300` | `0.5700` |
| `robot_visual` | lighting | `100` | `0.0000` | `1.0000` | `0.0700` | `0.9300` |
| `robot_state` | clutter | `100` | `0.1148` | `0.8852` | `0.6100` | `0.3900` |
| `visual` | clutter | `100` | `0.2558` | `0.7442` | `0.4300` | `0.5700` |
| `full` | camera shift | `100` | `0.0312` | `0.9688` | `0.6400` | `0.3600` |

Online robustness smoke (40 episodes/shift, `robot_visual` checkpoint):

| Shift | Policy | Success | FAR | Acceptance | Time/Success |
|-------|--------|--------:|----:|-----------:|-------------:|
| camera shift | robot+visual | `0.7250` | `0.0000` | `0.7436` | `6.90 s` |
| approach 0.03 | robot+visual | `0.3750` | `0.1667` | `0.6207` | `13.33 s` |
| clutter | robot+visual | `0.2000` | `0.0000` | `0.3200` | `25.00 s` |
| object shift | robot+visual | `0.4750` | `0.1739` | `0.6765` | `10.53 s` |
| partial occlusion | robot+visual | `0.4000` | `0.2381` | `0.7000` | `12.50 s` |

---

## Dataset Index

| CSV File | Events | Split | Notes |
|----------|-------:|-------|-------|
| `logs/hsi_pregrasp/stage0_events.csv` | `30` | train | state-only smoke |
| `logs/hsi_pregrasp/main800_events.csv` | `800` | train | state-only main |
| `logs/hsi_pregrasp/holdout400_events.csv` | `400` | holdout | state-only |
| `logs/hsi_pregrasp/shift_noise001_events.csv` | `400` | shift | noise=0.01 |
| `logs/hsi_pregrasp/shift_noise003_events.csv` | `400` | shift | noise=0.03 |
| `logs/hsi_pregrasp/vla/main600_vla_events.csv` | `600` | train | VLA single-object |
| `logs/hsi_pregrasp/vla/holdout400_vla_events.csv` | `400` | holdout | VLA single-object |
| `logs/hsi_pregrasp/vla/language_wrong_object_main600_camera_events.csv` | `600` | train | camera wrong-object |
| `logs/hsi_pregrasp/vla/language_wrong_object_holdout400_camera_events.csv` | `400` | holdout | camera wrong-object |
| `logs/hsi_pregrasp/vla/language_wrong_object_vla40_events.csv` | `40` | subset | wrong-object + VLA uncertainty |
| `logs/hsi_pregrasp/vla/language_wrong_object_dino_train300_events.csv` | `300` | train | DINOv2 wrong-object |
| `logs/hsi_pregrasp/vla/language_wrong_object_dino_holdout200_events.csv` | `171` | holdout | DINOv2 wrong-object |
| `logs/hsi_pregrasp/vla/physical_wrong_object_debug100_events.csv` | `100` | debug | physical wrong-object |
| `logs/hsi_pregrasp/vla/physical_partial_occlusion_debug100_events.csv` | `100` | debug | physical partial occlusion |
| `logs/hsi_pregrasp/vla/robust_partial_occlusion100_events.csv` | `100` | shift | partial occlusion |
| `logs/hsi_pregrasp/vla/robust_clutter100_events.csv` | `100` | shift | clutter |
| `logs/hsi_pregrasp/vla/robust_lighting100_full_events.csv` | `100` | shift | lighting |
| `logs/hsi_pregrasp/vla/robust_camera_shift100_events.csv` | `100` | shift | camera pose shift |
| `logs/hsi_pregrasp/vla/robust_object_shift100_events.csv` | `100` | shift | object position shift |
| `logs/hsi_pregrasp/vla/robust_approach_noise003_100_events.csv` | `100` | shift | harder noise |

---

## Result Source Audit

Verified `2026-05-24`:

| Result | Source Artifact | Checked Value |
|--------|----------------|--------------|
| State-only offline holdout | `logs/hsi_pregrasp/offline_baselines.json` | distance FAR `0.0988`; always-close FAR `0.2475` |
| State-only online | `online_refusal_summary.json` | refusal FAR `0.1119`; episodes `448` |
| State-only robustness | `shift_noise001/003_events.csv` + evaluator | easy FAR `0.0050`; hard FAR `0.1355` |
| VLA main dataset | `main600_vla_events.csv` | events `600`; success `452`; mean VLA `331.3 ms` |
| VLA holdout dataset | `holdout400_vla_events.csv` | events `400`; success `288`; mean VLA `330.3 ms` |
| VLA feature groups | `feature_group_ablations_main600.json` | visual FAR `0.1104`; robot_visual FAR `0.1351` |
| Online VLA single-object | `online_visual_main600_summary.json` | FAR `0.1132`; acceptance `0.8833`; episodes `80` |
| Camera wrong-object matched CI | `language_wrong_object_camera_main600_matched_ci.json` | visual_language FAR `0.0900`; distance FAR `0.3700` |
| Camera wrong-object proxy baselines | `language_wrong_object_camera_main600_proxy_baselines.json` | proxy FAR `0.4800`; oracle FAR `0.3900` |
| Online wrong-object always-close | `online_language_wrong_object_main600_always_close_summary.json` | FAR `0.4795`; wrong accepted `21`; episodes `112` |
| Online wrong-object visual+language | `online_language_wrong_object_main600_visual_language_summary.json` | FAR `0.1154`; wrong accepted `0`; episodes `112` |
| DINOv2 feature-group ablation | `dino_ablation_main300.json` | dino_language FAR `0.0741`; acceptance `0.1350` |
| DINOv2 holdout proxy baselines | `dino_holdout158_proxy_baselines.json` | oracle FAR `0.2907`; always-close FAR `0.4561`; N=`171` |
| VLA robustness offline | `robustness_main600_table.md` | clutter robot_visual FAR `0.0612` |
| VLA robustness online | `online_robust_main600_table.md` | camera_shift robot_visual FAR `0.0000` |

---

## Caveats

- IsaacLab simulation only — no real robot results yet.
- Motion policy is a deterministic state machine + injected approach noise; SmolVLA provides action-uncertainty features only, not trajectory control.
- SmolVLA exposes a 6-D continuous action output — no gripper-close log-probability is available.
- Lighting shift causes near-total refusal for camera-heavy checkpoints (known gap).
- Calibration is empirical selective-risk; no formal conformal guarantee.
- Online reapproach is a simulator noise resample, not a real retreat-resense-replan stack.
- DINOv2 holdout targeted 200 events but collected 171 before the run stopped; results are from the 171-event set.

---

## What Is Still Needed

| Task | Est. Time |
|------|----------:|
| Scale physical wrong-object/occlusion debug to 300/200 events | `0.5–1 day` |
| Target-aware multi-object controller (actually lifts requested colored cube) | `1–3 days` |
| Full VLA-uncertainty wrong-object 600/400 collection | `1–3 days` |
| Shift-aware online threshold calibration | `0.5–1 day` |
| Real robot 30-event smoke | `0.5 day` |
| Real robot 150-event pilot | `1 day` |
| Real robot 600–800 event main dataset | `2–4 days` |
| Real robot online final evaluation | `1–2 days` |
| Paper figures, tables, video, writing | `3–7 days` |

Simulator-only version: complete for single-object, robustness, and language wrong-object benchmarks.
Full proposal with real robot and paper-ready figures: ~3–5 weeks.

---

## Timing Reference

| Run | Approx. Time |
|-----|-------------|
| 800-event state-only headless | ~1 min |
| 400-event state-only holdout | ~30–35 sec |
| 600-event VLA main (vla_samples=2) | ~78 min |
| 400-event VLA holdout (vla_samples=2) | ~56 min |
| 300-event language wrong-object shard (skip_vla, camera) | ~20–40 min |
| Online wrong-object evaluation (~100 episodes) | ~10–11 min |
| DINOv2 300-event train collection | ~20–40 min |
| MLP training + calibration after CSV | ~1–2 sec on CPU |
