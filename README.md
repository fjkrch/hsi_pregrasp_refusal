# HSI / Pre-Grasp Refusal

Runnable IsaacLab scaffold for the project proposal:

`Pre-Grasp Refusal: Selective Risk Control at the Gripper-Closure Decision Point for VLA Manipulation`

The goal is to decide, immediately before gripper closure, whether the robot should:

- `ACCEPT_CLOSE`: commit to closing the gripper
- `REFUSE_CLOSE`: avoid the close attempt and later re-approach

This implementation uses IsaacLab simulation with a deterministic Franka cube-lift state machine as the surrogate
policy, plus a camera-enabled VLA-sim stage that runs the staged SmolVLA checkpoint on real IsaacLab RGB observations.
The same trigger, event schema, refusal head, threshold calibration, and metrics are reused across robot-state,
visual, VLA-action-uncertainty, and combined feature groups.

## Latest Status

Completed so far:

- 30-event visible smoke test
- 800-event headless main simulator dataset
- 400-event independent headless holdout dataset
- refusal-head training/calibration
- offline baselines: always-close, matched-random, distance-only, lateral-error-only, geometry-head, action/state-head,
  and full-head
- online simulator refusal/reapproach evaluation against an always-close online baseline
- simulator robustness sweep over easier and harder approach-noise shifts
- SmolVLA checkpoint downloaded and runtime smoke-tested with dummy three-camera inputs
- camera-enabled IsaacLab lift scene with table, wrist, and overhead RGB cameras at `256x256`
- SmolVLA adapter for IsaacLab camera tensors plus a 6-D simulator proxy state
- VLA event collection, feature-group ablations, online VLA refusal evaluation, and distractor smoke test
- no-oracle matched-acceptance holdout analysis with bootstrap confidence intervals
- matched-acceptance clutter analysis with bootstrap confidence intervals
- language-conditioned wrong-object simulator pilot with automatic labels

Latest state-only holdout result summary:

| Method | False-Accept Risk | Accepted Success | Acceptance Rate |
| --- | ---: | ---: | ---: |
| Always Close | `0.2475` | `0.7525` | `1.0000` |
| Matched Random Refusal | `0.2475` | `0.7525` | `0.8345` |
| Distance Only | `0.0988` | `0.9012` | `0.8350` |
| Lateral Error Only | `0.0988` | `0.9012` | `0.8350` |
| Geometry Learned Head | `0.0961` | `0.9039` | `0.8325` |
| Full Learned Head | `0.0988` | `0.9012` | `0.8350` |
| Action/State Learned Head | `0.2000` | `0.8000` | `0.0875` |

Current conclusion: in the simplified IsaacLab setup, pre-grasp geometry is the dominant signal. The next experiments
must add online re-approach, visual clutter, distractors, VLA uncertainty, or real robot data to go beyond a simple
distance/lateral-error threshold.

Latest online simulator result:

| Online Policy | Episodes | Episode Success | False-Accept Risk | Acceptance Rate | Refusal Rate | Robot Time / Success | Attempts / Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Always Close | `448` | `0.5246` | `0.2610` | `1.0000` | `0.0000` | `9.5319 s` | `1.3532` |
| Refusal + Reapproach | `448` | `0.5670` | `0.1119` | `0.8266` | `0.1734` | `8.8189 s` | `1.1260` |

Online result interpretation:

- The refusal policy sharply reduces risky accepted closures.
- Online task success improves modestly because refused attempts get a re-approach instead of an immediate bad close.
- False-accept risk is slightly above the `0.10` calibration target online (`0.1119`), so the next online run should tune
  threshold/retry behavior after adding richer visual or VLA signals.

Latest robustness shift result:

| Shift Dataset | Noise Std | Always-Close False-Accept Risk | Refusal False-Accept Risk | Acceptance Rate | Refusal Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easier shift | `0.01` | `0.0075` | `0.0050` | `0.9975` | `0.0025` |
| Main/holdout setting | `0.02` | `0.2475` | `0.0988` | `0.8350` | `0.1650` |
| Harder shift | `0.03` | `0.5325` | `0.1355` | `0.5350` | `0.4650` |

Latest VLA asset/runtime status:

| Item | Status |
| --- | --- |
| SmolVLA repo | `lerobot/smolvla_base` |
| Local path | `logs/hsi_pregrasp/vla/smolvla_base` |
| Checkpoint size | `865 MB` |
| Runtime | LeRobot `SmolVLAPolicy` imports and loads |
| Smoke input | three dummy `3x256x256` camera tensors, one 6-D state, language task |
| Smoke output | 6-D continuous action |
| Mean action variance over 8 samples | `0.0065883` |
| Action std norm over 8 samples | `0.1988218` |

Latest camera-enabled VLA-sim datasets:

| Run | Events | Successes | Failures | Failure Rate | VLA Samples | Mean VLA ms/Event | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Visible VLA smoke | `10` | `10` | `0` | `0.0000` | `2` | `353.4094` | `logs/hsi_pregrasp/vla/stage0_vla_events.csv` |
| Pilot partial | `95` | `67` | `28` | `0.2947` | `4` | `674.6589` | `logs/hsi_pregrasp/vla/pilot150_vla_events.csv` |
| Proposal-scale main | `600` | `452` | `148` | `0.2467` | `2` | `331.2642` | `logs/hsi_pregrasp/vla/main600_vla_events.csv` |
| Proposal-scale holdout | `400` | `288` | `112` | `0.2800` | `2` | `330.3354` | `logs/hsi_pregrasp/vla/holdout400_vla_events.csv` |
| Distractor smoke | `12` | `8` | `4` | `0.3333` | `1` | not summarized | `logs/hsi_pregrasp/vla/distractor12_vla_events.csv` |
| Language wrong-object train pilot | `63` | `30` | `33` | `0.5238` | skipped | `0.0000` | `logs/hsi_pregrasp/vla/language_wrong_object_main63_events.csv` |
| Language wrong-object holdout pilot | `40` | `22` | `18` | `0.4500` | skipped | `0.0000` | `logs/hsi_pregrasp/vla/language_wrong_object_smoke40_events.csv` |
| Partial-occlusion robustness | `100` | `82` | `18` | `0.1800` | skipped | `0.0000` | `logs/hsi_pregrasp/vla/robust_partial_occlusion100_events.csv` |
| Clutter robustness | `100` | `70` | `30` | `0.3000` | skipped | `0.0000` | `logs/hsi_pregrasp/vla/robust_clutter100_events.csv` |
| Lighting robustness | `100` | `72` | `28` | `0.2800` | skipped | `0.0000` | `logs/hsi_pregrasp/vla/robust_lighting100_full_events.csv` |
| Camera-shift robustness | `100` | `74` | `26` | `0.2600` | skipped | `0.0000` | `logs/hsi_pregrasp/vla/robust_camera_shift100_events.csv` |
| Object-shift robustness | `100` | `78` | `22` | `0.2200` | skipped | `0.0000` | `logs/hsi_pregrasp/vla/robust_object_shift100_events.csv` |
| Approach-shift robustness | `100` | `38` | `62` | `0.6200` | skipped | `0.0000` | `logs/hsi_pregrasp/vla/robust_approach_noise003_100_events.csv` |

Latest VLA-sim feature-group holdout results, trained on the 600-event main set and evaluated on the 400-event holdout:

| Feature Group | False-Accept Risk | Accepted Success | Acceptance Rate |
| --- | ---: | ---: | ---: |
| `robot_state` | `0.1628` | `0.8372` | `0.8600` |
| `visual` | `0.1104` | `0.8896` | `0.7925` |
| `vla_action_uncertainty` | `0.2500` | `0.7500` | `0.1100` |
| `robot_visual` | `0.1351` | `0.8649` | `0.8325` |
| `full` | `0.1529` | `0.8471` | `0.8500` |

Latest online VLA-sim result:

| Online Policy | Episodes | Episode Success | False-Accept Risk | Acceptance Rate | Refusal Rate | Mean VLA ms/Event |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always Close, matched seed | `80` | `0.5375` | `0.2182` | `1.0000` | `0.0000` | `0.0000` |
| Visual Refusal + Reapproach | `80` | `0.5875` | `0.1132` | `0.8833` | `0.1167` | `0.0000` |

Latest VLA robustness results, evaluating main600 checkpoints on shifted simulator data:

| Feature Group | Shift | Events | False-Accept Risk | Accepted Success | Acceptance Rate | Refusal Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `robot_visual` | partial occlusion | `100` | `0.0476` | `0.9524` | `0.8400` | `0.1600` |
| `robot_state` | clutter | `100` | `0.1148` | `0.8852` | `0.6100` | `0.3900` |
| `visual` | clutter | `100` | `0.2558` | `0.7442` | `0.4300` | `0.5700` |
| `robot_visual` | clutter | `100` | `0.0612` | `0.9388` | `0.4900` | `0.5100` |
| `full` | clutter | `100` | `0.1207` | `0.8793` | `0.5800` | `0.4200` |
| `robot_state` | lighting | `100` | `0.1125` | `0.8875` | `0.8000` | `0.2000` |
| `robot_visual` | lighting | `100` | `0.0000` | `1.0000` | `0.0700` | `0.9300` |
| `full` | camera shift | `100` | `0.0312` | `0.9688` | `0.6400` | `0.3600` |
| `robot_visual` | object shift | `100` | `0.0411` | `0.9589` | `0.7300` | `0.2700` |
| `robot_visual` | approach noise `0.03` | `100` | `0.1163` | `0.8837` | `0.4300` | `0.5700` |

Current VLA-sim conclusion: proposal-scale simulation is complete for the main single-object setting. Visual features
are the best learned feature group on the independent holdout, but oracle distance is still slightly stronger on that
same holdout. Matched clutter analysis shows distance still separates failures too well. A new language-conditioned
wrong-object pilot finally makes geometry lose at matched acceptance, but it is small-scale and should be expanded
before final claims. Sampled SmolVLA action uncertainty alone is weak. Lighting shift exposes a clear robustness gap
because camera-heavy checkpoints over-refuse.

## Research Question

Which signals are sufficient for a VLA robot to decide whether to physically commit to gripper closure?

In this simulator scaffold, the first signal set is:

- end-effector/object distance
- lateral and vertical pre-grasp error
- object height
- gripper width
- action delta norm
- action rotation distance
- close-command flag
- state-machine state and wait time

## Project Structure

- `hsi_pregrasp_refusal/trigger.py`: deterministic pre-grasp trigger
- `hsi_pregrasp_refusal/features.py`: feature columns used by the collector and model
- `hsi_pregrasp_refusal/model.py`: small MLP refusal head predicting `p(close will fail)`
- `hsi_pregrasp_refusal/calibration.py`: threshold calibration for target false-accept risk
- `hsi_pregrasp_refusal/metrics.py`: selective close/refuse metrics
- `hsi_pregrasp_refusal/vision.py`: RGB tensor conversion and simple visual summaries
- `hsi_pregrasp_refusal/vla.py`: SmolVLA adapter and sampled action-uncertainty summaries
- `hsi_pregrasp_refusal/isaaclab_vla_scene.py`: camera-enabled lift scene and simulator variants
- `hsi_pregrasp_refusal/state_machine.py`: shared Franka lift state machine
- `scripts/collect_lift_cube_pregrasp.py`: IsaacLab data collector
- `scripts/collect_vla_lift_pregrasp.py`: camera/VLA IsaacLab event collector
- `scripts/train_refusal_head.py`: train/calibrate refusal head
- `scripts/evaluate_refusal.py`: evaluate a trained checkpoint
- `scripts/run_offline_baselines.py`: scalar and matched-random offline baselines
- `scripts/run_matched_acceptance_ci.py`: matched-acceptance comparisons with bootstrap confidence intervals
- `scripts/run_online_vla_refusal_eval.py`: camera/VLA online refusal and reapproach evaluation
- `scripts/run_feature_group_ablation.py`: train/evaluate feature-group heads
- `scripts/make_results_table.py`: generate markdown result tables from JSON summaries
- `tests/test_refusal_core.py`: lightweight unit tests

## Environment

Use the IsaacLab conda environment:

```bash
conda run -n env_isaaclab python -c "import sys; print(sys.executable); print(sys.version)"
```

Expected environment from the current machine:

- Python: `/home/chyanin/miniconda3/envs/env_isaaclab/bin/python`
- Python version: `3.11.14`
- PyTorch: `2.7.0+cu128`
- GPU used by IsaacLab: NVIDIA GeForce RTX 4060 Laptop GPU

If IsaacLab prints terminal reset warnings in this shell, prefix commands with `TERM=xterm`.

Extra packages installed for SmolVLA runtime:

- `lerobot`
- `draccus`
- `accelerate`
- `mergedeep`
- `typing-inspect`
- `mypy-extensions`
- `datasets`
- `dill`
- `multiprocess`
- `xxhash`
- `pyarrow`
- `httpx`
- `diffusers`
- `pyserial`
- `deepdiff`
- `orderly-set`
- `cachebox`
- `av`
- `num2words`

## Stage 0 Visible Smoke Test

This was run without `--headless` first, as a visual sanity check:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --num_envs 16 \
  --num_events 30 \
  --approach_noise_std 0.02 \
  --output logs/hsi_pregrasp/stage0_events.csv
```

Then train and evaluate:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/train_refusal_head.py \
  --input logs/hsi_pregrasp/stage0_events.csv \
  --output logs/hsi_pregrasp/refusal_head.pt \
  --target_false_accept_risk 0.10 \
  --epochs 300 \
  --seed 7 \
  --device cpu

conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/evaluate_refusal.py \
  --input logs/hsi_pregrasp/stage0_events.csv \
  --checkpoint logs/hsi_pregrasp/refusal_head.pt \
  --device cpu
```

Stage 0 output files:

- `logs/hsi_pregrasp/stage0_events.csv`
- `logs/hsi_pregrasp/refusal_head.pt`

Stage 0 results:

| Metric | Value |
| --- | ---: |
| Events | `30` |
| Always-close successes | `21` |
| Always-close failures | `9` |
| Always-close success rate | `0.7000` |
| Always-close false-accept risk | `0.3000` |
| Calibrated threshold `tau` | `0.2980891466` |
| Accepted closures | `21 / 30` |
| Refused closures | `9 / 30` |
| Failed accepted closures | `0` |
| Refusal-model false-accept risk | `0.0000` |
| Accepted grasp success | `1.0000` |
| Acceptance rate | `0.7000` |
| Refusal rate | `0.3000` |

Stage 0 is only a smoke test. It is useful for checking that the trigger, labels, model, and metrics work.

## Full-Scale Headless Run

The main simulator run used 800 events, matching the proposal's target main-dataset scale.

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --headless \
  --num_envs 64 \
  --num_events 800 \
  --approach_noise_std 0.02 \
  --seed 42 \
  --output logs/hsi_pregrasp/main800_events.csv
```

Train/calibrate:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/train_refusal_head.py \
  --input logs/hsi_pregrasp/main800_events.csv \
  --output logs/hsi_pregrasp/main800_refusal_head.pt \
  --target_false_accept_risk 0.10 \
  --epochs 300 \
  --seed 42 \
  --device cpu
```

Evaluate on the full 800-event dataset:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/evaluate_refusal.py \
  --input logs/hsi_pregrasp/main800_events.csv \
  --checkpoint logs/hsi_pregrasp/main800_refusal_head.pt \
  --device cpu
```

Main output files:

- `logs/hsi_pregrasp/main800_events.csv`
- `logs/hsi_pregrasp/main800_refusal_head.pt`

Raw main-dataset distribution:

| Metric | Value |
| --- | ---: |
| Events | `800` |
| Always-close successes | `588` |
| Always-close failures | `212` |
| Always-close success rate | `0.7350` |
| Always-close false-accept risk | `0.2650` |

Feature summary:

| Feature | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| `ee_object_distance` | `0.0062578` | `0.0233437` | `0.0682844` |
| `ee_object_lateral_error` | `0.0004703` | `0.0219136` | `0.0678901` |
| `ee_object_height_error` | `0.0050135` | `0.0067811` | `0.0091815` |
| `gripper_width` | `0.0799986` | `0.0799999` | `0.0800000` |
| `action_delta_pos_norm` | `0.0055439` | `0.0069528` | `0.0099827` |

Calibration result:

| Metric | Value |
| --- | ---: |
| Threshold `tau` | `0.9997993112` |
| Target false-accept risk | `0.1000` |
| Calibration events | `160` |
| Accepted calibration events | `135` |
| Calibration false-accept risk | `0.0963` |
| Calibration acceptance rate | `0.8438` |
| Risk estimator | `empirical` |

Train/calibration/test split metrics from `main800_events.csv`:

| Split | Events | Accepted | Refused | Failed Accepted | False-Accept Risk | Accepted Success | Acceptance Rate | Refusal Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | `480` | `385` | `95` | `35` | `0.0909` | `0.9091` | `0.8021` | `0.1979` |
| Calibration | `160` | `135` | `25` | `13` | `0.0963` | `0.9037` | `0.8438` | `0.1563` |
| Test | `160` | `127` | `33` | `11` | `0.0866` | `0.9134` | `0.7938` | `0.2063` |

Evaluation over all 800 events using the calibrated threshold:

| Metric | Value |
| --- | ---: |
| Accepted closures | `647 / 800` |
| Refused closures | `153 / 800` |
| Successful accepted closures | `588` |
| Failed accepted closures | `59` |
| False-accept risk | `0.0912` |
| Accepted grasp success | `0.9088` |
| Acceptance rate | `0.8088` |
| Refusal rate | `0.1913` |

## Independent Holdout Run

An independent 400-event holdout run was collected with a different seed, then evaluated with the checkpoint and
threshold trained on `main800_events.csv`.

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --headless \
  --num_envs 64 \
  --num_events 400 \
  --approach_noise_std 0.02 \
  --seed 99 \
  --output logs/hsi_pregrasp/holdout400_events.csv

conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/evaluate_refusal.py \
  --input logs/hsi_pregrasp/holdout400_events.csv \
  --checkpoint logs/hsi_pregrasp/main800_refusal_head.pt \
  --device cpu
```

Holdout output file:

- `logs/hsi_pregrasp/holdout400_events.csv`

Holdout raw distribution:

| Metric | Value |
| --- | ---: |
| Events | `400` |
| Always-close successes | `301` |
| Always-close failures | `99` |
| Always-close success rate | `0.7525` |
| Always-close false-accept risk | `0.2475` |

Holdout refusal-model result:

| Metric | Value |
| --- | ---: |
| Accepted closures | `334 / 400` |
| Refused closures | `66 / 400` |
| Successful accepted closures | `301` |
| Failed accepted closures | `33` |
| False-accept risk | `0.0988` |
| Accepted grasp success | `0.9012` |
| Acceptance rate | `0.8350` |
| Refusal rate | `0.1650` |

## Result Summary

The refusal head reduced false-accept risk while keeping most closures:

| Dataset | Always-Close False-Accept Risk | Refusal False-Accept Risk | Acceptance Rate |
| --- | ---: | ---: | ---: |
| Stage 0, 30 events | `0.3000` | `0.0000` | `0.7000` |
| Main, 800 events | `0.2650` | `0.0912` | `0.8088` |
| Holdout, 400 events | `0.2475` | `0.0988` | `0.8350` |

Interpretation:

- Always closing fails roughly one quarter of the time in this noisy pre-grasp simulator setup.
- The calibrated refusal head holds false-accept risk near the target `0.10`.
- The model still accepts about `81-84%` of closure attempts on the larger datasets.
- The independent holdout result is close to calibration, which is a good sign for this simulator scaffold.

## Offline Baselines Run

The following additional baselines were run on the current simulator CSVs:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_offline_baselines.py \
  --input logs/hsi_pregrasp/main800_events.csv \
  --eval-input logs/hsi_pregrasp/holdout400_events.csv \
  --output logs/hsi_pregrasp/offline_baselines.json \
  --seed 42 \
  --matched_acceptance_rate 0.835
```

Feature-subset learned heads:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/train_refusal_head.py \
  --input logs/hsi_pregrasp/main800_events.csv \
  --output logs/hsi_pregrasp/geometry_refusal_head.pt \
  --features ee_object_distance,ee_object_lateral_error,ee_object_height_error,object_height,gripper_width \
  --target_false_accept_risk 0.10 \
  --epochs 300 \
  --seed 42 \
  --device cpu

conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/train_refusal_head.py \
  --input logs/hsi_pregrasp/main800_events.csv \
  --output logs/hsi_pregrasp/action_state_refusal_head.pt \
  --features action_delta_pos_norm,action_delta_rot_distance,close_commanded,sm_state,sm_wait_time \
  --target_false_accept_risk 0.10 \
  --epochs 300 \
  --seed 42 \
  --device cpu
```

Holdout results for supported offline baselines:

| Method | Holdout False-Accept Risk | Accepted Success | Acceptance Rate | Refusal Rate | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Always Close | `0.2475` | `0.7525` | `1.0000` | `0.0000` | Baseline with no safety gate |
| Matched Random Refusal | `0.2475` | `0.7525` | `0.8345` | `0.1655` | Randomly refuses at same rate as learned model |
| Distance Only | `0.0988` | `0.9012` | `0.8350` | `0.1650` | Scalar threshold on `ee_object_distance` |
| Lateral Error Only | `0.0988` | `0.9012` | `0.8350` | `0.1650` | Scalar threshold on `ee_object_lateral_error` |
| Geometry Learned Head | `0.0961` | `0.9039` | `0.8325` | `0.1675` | MLP over distance/lateral/height/object/gripper state |
| Full Learned Head | `0.0988` | `0.9012` | `0.8350` | `0.1650` | MLP over all current simulator features |
| Action/State Learned Head | `0.2000` | `0.8000` | `0.0875` | `0.9125` | Over-refuses and does not meet target on holdout |

Conclusion from current simulator baselines:

- The strongest signal in this simplified IsaacLab setup is geometric pre-grasp alignment.
- Distance-only and lateral-error-only baselines are already very strong because injected failures mostly come from XY approach noise.
- Action/state-only is weak here because the state machine timing is almost identical for success and failure.
- To make the learned full head meaningfully different from simple geometry thresholds, the next experiments need visual clutter, distractors, occlusion, VLA uncertainty, or real robot noise.

## Online Refusal/Reapproach Run

Online refusal was run with the trained full-head checkpoint:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/run_online_refusal_eval.py \
  --headless \
  --checkpoint logs/hsi_pregrasp/main800_refusal_head.pt \
  --num_envs 64 \
  --num_episodes 400 \
  --approach_noise_std 0.02 \
  --retry_noise_std 0.02 \
  --seed 123 \
  --output logs/hsi_pregrasp/online_refusal_events.csv \
  --summary logs/hsi_pregrasp/online_refusal_summary.json
```

The matching always-close online baseline was run by forcing every pre-grasp event to accept:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/run_online_refusal_eval.py \
  --headless \
  --checkpoint logs/hsi_pregrasp/main800_refusal_head.pt \
  --num_envs 64 \
  --num_episodes 400 \
  --approach_noise_std 0.02 \
  --retry_noise_std 0.02 \
  --seed 123 \
  --max_refusals_per_episode 0 \
  --output logs/hsi_pregrasp/online_always_close_events.csv \
  --summary logs/hsi_pregrasp/online_always_close_summary.json
```

Online output files:

- `logs/hsi_pregrasp/online_refusal_events.csv`
- `logs/hsi_pregrasp/online_refusal_summary.json`
- `logs/hsi_pregrasp/online_always_close_events.csv`
- `logs/hsi_pregrasp/online_always_close_summary.json`

Online comparison:

| Metric | Always Close | Refusal + Reapproach |
| --- | ---: | ---: |
| Completed episodes | `448` | `448` |
| Successful episodes | `235` | `254` |
| Pre-grasp events | `318` | `346` |
| Accepted closures | `318` | `286` |
| Refused closures | `0` | `60` |
| Successful accepted closures | `235` | `254` |
| Failed accepted closures | `83` | `32` |
| False-accept risk | `0.2610` | `0.1119` |
| Accepted grasp success | `0.7390` | `0.8881` |
| Acceptance rate | `1.0000` | `0.8266` |
| Refusal rate | `0.0000` | `0.1734` |
| Task success after reapproach | `0.5246` | `0.5670` |
| Robot time per success | `9.5319 s` | `8.8189 s` |
| Attempts per success | `1.3532` | `1.1260` |

## Robustness Shift Sweep

Two extra headless simulator datasets were collected by changing the injected approach-noise level:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --headless \
  --num_envs 64 \
  --num_events 400 \
  --approach_noise_std 0.01 \
  --seed 201 \
  --output logs/hsi_pregrasp/shift_noise001_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \
  --headless \
  --num_envs 64 \
  --num_events 400 \
  --approach_noise_std 0.03 \
  --seed 202 \
  --output logs/hsi_pregrasp/shift_noise003_events.csv
```

Both were evaluated with the threshold trained on `main800_events.csv`:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/evaluate_refusal.py \
  --input logs/hsi_pregrasp/shift_noise001_events.csv \
  --checkpoint logs/hsi_pregrasp/main800_refusal_head.pt \
  --device cpu

conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/evaluate_refusal.py \
  --input logs/hsi_pregrasp/shift_noise003_events.csv \
  --checkpoint logs/hsi_pregrasp/main800_refusal_head.pt \
  --device cpu
```

Shift results:

| Dataset | Events | Successes | Failures | Always-Close False-Accept Risk | Refusal False-Accept Risk | Accepted Success | Acceptance Rate | Refusal Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shift_noise001_events.csv` | `400` | `397` | `3` | `0.0075` | `0.0050` | `0.9950` | `0.9975` | `0.0025` |
| `shift_noise003_events.csv` | `400` | `187` | `213` | `0.5325` | `0.1355` | `0.8645` | `0.5350` | `0.4650` |

Interpretation:

- Under easier approach noise, the task is almost solved and refusal rarely triggers.
- Under harder approach noise, refusal cuts false accepts from `0.5325` to `0.1355`, but misses the `0.10` target.
- The harder-shift result is a useful stress test and motivates richer visual/VLA features or shift-aware calibration.

## SmolVLA Download And Smoke Test

The lightweight proposal-compatible VLA backbone was staged locally:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/download_smolvla_assets.py \
  --output-dir logs/hsi_pregrasp/vla/smolvla_base
```

Downloaded files:

- `logs/hsi_pregrasp/vla/smolvla_base/config.json`
- `logs/hsi_pregrasp/vla/smolvla_base/model.safetensors`
- `logs/hsi_pregrasp/vla/smolvla_base/policy_preprocessor.json`
- `logs/hsi_pregrasp/vla/smolvla_base/policy_postprocessor.json`
- `logs/hsi_pregrasp/vla/smolvla_base/download_manifest.json`

Checkpoint/runtime check:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/check_vla_runtime.py \
  --local-dir logs/hsi_pregrasp/vla/smolvla_base
```

Status after installing runtime dependencies:

| Check | Value |
| --- | --- |
| `config_exists` | `true` |
| `weights_exist` | `true` |
| `preprocessor_exists` | `true` |
| `postprocessor_exists` | `true` |
| `lerobot_installed` | `true` |
| `ready_for_runtime` | `true` |

SmolVLA config summary:

- input state: `observation.state`, shape `(6,)`
- visual inputs: `observation.images.camera1`, `camera2`, `camera3`, each shape `(3, 256, 256)`
- output action: shape `(6,)`
- underlying VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

Smoke test:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_smolvla_smoke.py \
  --checkpoint logs/hsi_pregrasp/vla/smolvla_base \
  --num_samples 8 \
  --output logs/hsi_pregrasp/vla/smolvla_smoke_summary.json
```

Smoke-test result:

| Metric | Value |
| --- | ---: |
| Device | `cuda` |
| Samples | `8` |
| Output action shape | `[8, 6]` |
| Mean action variance | `0.0065883` |
| Action std norm | `0.1988218` |

Mean action:

```text
[-0.0874, 0.2510, 0.1159, 0.0581, 0.1647, -0.1211]
```

Action standard deviation:

```text
[0.0843, 0.0966, 0.1073, 0.0390, 0.0307, 0.0955]
```

This proves that the real SmolVLA checkpoint can be loaded and queried locally. The next section records the completed
camera-enabled IsaacLab integration that replaces dummy images with simulator RGB camera tensors.

## Camera-Enabled VLA-Sim Stage

New implementation pieces:

- Three RGB cameras are added to the Franka cube-lift task: table, wrist, and overhead/angled views.
- Camera tensors are resized/normalized into SmolVLA's `camera1`, `camera2`, and `camera3` schema at `3x256x256`.
- A consistent 6-D simulator proxy state is built from TCP/object geometry for SmolVLA's `observation.state`.
- Refusal features now support `robot_state`, `visual`, `vla_action_uncertainty`, `vla_only`, `robot_visual`, and `full`.
- The VLA signal is sampled action uncertainty: per-dimension action mean/std/variance/range, total variance,
  std norm, and an entropy-style Gaussian proxy.
- Simulator variants are wired for `single`, `distractors`, `clutter`, `partial_occlusion`, `wrong_object`,
  `lighting_shift`, `camera_shift`, and `object_shift`.

Important VLA note: the staged SmolVLA checkpoint exposes a 6-D continuous action output in this setup. It does not
expose a true gripper-close log-probability, so this project logs sampled action uncertainty as the available VLA signal.

Visible VLA smoke, run before headless experiments:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --num_envs 4 \
  --num_events 10 \
  --approach_noise_std 0.02 \
  --vla_samples 2 \
  --output logs/hsi_pregrasp/vla/stage0_vla_events.csv
```

Result:

| Metric | Value |
| --- | ---: |
| Events | `10` |
| Successes | `10` |
| Failures | `0` |
| Failure rate | `0.0000` |
| Mean VLA inference | `353.4094 ms/event` |

Headless VLA pilot:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 150 \
  --approach_noise_std 0.02 \
  --vla_samples 4 \
  --output logs/hsi_pregrasp/vla/pilot150_vla_events.csv
```

The pilot stopped writing new rows at 95 events while the simulator was still alive, so it was interrupted and kept as a
valid partial pilot. The collector now has `--max_steps` and `--stall_steps` guards plus row flushing for long headless
runs.

Pilot result:

| Metric | Value |
| --- | ---: |
| Events kept | `95` |
| Successes | `67` |
| Failures | `28` |
| Failure rate | `0.2947` |
| Mean VLA inference | `674.6589 ms/event` |

Independent VLA holdout:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 80 \
  --approach_noise_std 0.02 \
  --vla_samples 2 \
  --max_steps 10000 \
  --stall_steps 2000 \
  --output logs/hsi_pregrasp/vla/holdout80_vla_events.csv
```

Holdout result:

| Metric | Value |
| --- | ---: |
| Events | `80` |
| Successes | `53` |
| Failures | `27` |
| Failure rate | `0.3375` |
| Mean VLA inference | `327.5958 ms/event` |

Proposal-scale VLA main dataset:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 600 \
  --approach_noise_std 0.02 \
  --vla_samples 2 \
  --seed 142 \
  --max_steps 80000 \
  --stall_steps 5000 \
  --output logs/hsi_pregrasp/vla/main600_vla_events.csv
```

Proposal-scale VLA holdout:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 400 \
  --approach_noise_std 0.02 \
  --vla_samples 2 \
  --seed 299 \
  --max_steps 60000 \
  --stall_steps 5000 \
  --output logs/hsi_pregrasp/vla/holdout400_vla_events.csv
```

Proposal-scale dataset results:

| Dataset | Events | Successes | Failures | Failure Rate | Mean VLA ms/Event | Missing Feature Columns | Feature NaNs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `main600_vla_events.csv` | `600` | `452` | `148` | `0.2467` | `331.2642` | `0` | `0` |
| `holdout400_vla_events.csv` | `400` | `288` | `112` | `0.2800` | `330.3354` | `0` | `0` |

Feature-group ablation command:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_feature_group_ablation.py \
  --input logs/hsi_pregrasp/vla/main600_vla_events.csv \
  --eval-input logs/hsi_pregrasp/vla/holdout400_vla_events.csv \
  --output logs/hsi_pregrasp/vla/feature_group_ablations_main600.json \
  --checkpoint-dir logs/hsi_pregrasp/vla/checkpoints_main600 \
  --epochs 300 \
  --seed 42 \
  --device cpu
```

Feature-group holdout results:

| Feature Group | False-Accept Risk | Accepted Success | Acceptance Rate |
| --- | ---: | ---: | ---: |
| `robot_state` | `0.1628` | `0.8372` | `0.8600` |
| `visual` | `0.1104` | `0.8896` | `0.7925` |
| `vla_action_uncertainty` | `0.2500` | `0.7500` | `0.1100` |
| `robot_visual` | `0.1351` | `0.8649` | `0.8325` |
| `full` | `0.1529` | `0.8471` | `0.8500` |

Scalar and offline baseline command:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_offline_baselines.py \
  --input logs/hsi_pregrasp/vla/main600_vla_events.csv \
  --eval-input logs/hsi_pregrasp/vla/holdout400_vla_events.csv \
  --output logs/hsi_pregrasp/vla/offline_vla_baselines_main600.json \
  --seed 42 \
  --matched_acceptance_rate 0.7925
```

Selected holdout baseline results:

| Method | False-Accept Risk | Accepted Success | Acceptance Rate |
| --- | ---: | ---: | ---: |
| Always Close | `0.2800` | `0.7200` | `1.0000` |
| Matched Random | `0.2799` | `0.7201` | `0.7921` |
| Distance Only | `0.1377` | `0.8623` | `0.8350` |
| Lateral Error Only | `0.1429` | `0.8571` | `0.8400` |
| VLA Action Variance Only | `0.3333` | `0.6667` | `0.0150` |
| VLA Entropy Proxy Only | `0.2750` | `0.7250` | `0.1000` |
| Table Camera Center Mean Only | `0.1111` | `0.8889` | `0.0225` |

No-oracle matched-acceptance holdout comparison:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_matched_acceptance_ci.py \
  --input logs/hsi_pregrasp/vla/holdout400_vla_events.csv \
  --title "No-Oracle Holdout400 Camera-Only Matched-Acceptance Comparison" \
  --reference-checkpoint visual=logs/hsi_pregrasp/vla/checkpoints_main600/visual_refusal_head.pt \
  --checkpoint vla_uncertainty=logs/hsi_pregrasp/vla/checkpoints_main600/vla_action_uncertainty_refusal_head.pt \
  --scalar table_center=camera1_center_mean:low \
  --scalar wrist_center=camera2_center_mean:low \
  --scalar overhead_center=camera3_center_mean:low \
  --scalar oracle_distance_only=ee_object_distance:low \
  --include-always-close \
  --include-matched-random \
  --bootstrap 2000 \
  --seed 314 \
  --device cpu \
  --output-json logs/hsi_pregrasp/vla/no_oracle_holdout400_matched_ci.json \
  --output-md logs/hsi_pregrasp/vla/no_oracle_holdout400_matched_ci.md
```

The no-oracle comparison is the valid camera-only result. The `oracle_distance_only` row is retained only to close the
loose end from the proposal notes and should not be counted as a no-oracle baseline.

| Method | Oracle Geometry? | Acceptance Mode | False-Accept Risk, 95% CI | Accepted Success, 95% CI | Acceptance, 95% CI |
| --- | --- | --- | ---: | ---: | ---: |
| `visual` | `no` | checkpoint threshold | `0.1104` [`0.0757`, `0.1438`] | `0.8896` [`0.8526`, `0.9228`] | `0.7925` [`0.7525`, `0.8300`] |
| `matched_random` | `no` | matched target acceptance | `0.2799` [`0.2587`, `0.2997`] | `0.7201` [`0.7003`, `0.7413`] | `0.7925` [`0.7925`, `0.7925`] |
| `vla_uncertainty` | `no` | matched target acceptance | `0.2776` [`0.2278`, `0.3301`] | `0.7224` [`0.6708`, `0.7695`] | `0.7925` [`0.7525`, `0.8300`] |
| `table_center` | `no` | matched target acceptance | `0.2713` [`0.2229`, `0.3231`] | `0.7287` [`0.6785`, `0.7745`] | `0.7925` [`0.7525`, `0.8300`] |
| `wrist_center` | `no` | matched target acceptance | `0.2050` [`0.1636`, `0.2500`] | `0.7950` [`0.7515`, `0.8385`] | `0.7925` [`0.7525`, `0.8325`] |
| `overhead_center` | `no` | matched target acceptance | `0.2618` [`0.2134`, `0.3121`] | `0.7382` [`0.6930`, `0.7870`] | `0.7925` [`0.7525`, `0.8325`] |
| `oracle_distance_only` | `yes` | matched target acceptance | `0.0915` [`0.0617`, `0.1254`] | `0.9085` [`0.8758`, `0.9381`] | `0.7925` [`0.7525`, `0.8325`] |

Interpretation: after removing oracle pose/distance from the learned head and no-oracle baselines, the visual checkpoint
is a clean win over matched random on `holdout400`. It is not yet a clean win over oracle distance, which means the
current simulator still allows geometry to explain too much of the task.

Language-conditioned wrong-object pilot:

This is a small simulator-only pilot for the proposal's multi-object/language milestone. The scene contains the default
cube plus colored distractors. The state-machine policy is intentionally language-blind and still grasps the default
cube. If the instruction asks for a colored cube and the default cube is lifted, the simulator automatically labels the
event as `wrong_object_default_cube_lifted`, so no manual labels are used.

Collection commands:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 40 \
  --variant distractors \
  --num_distractors 4 \
  --skip_vla \
  --language_mode multi_object_default_policy \
  --language_default_prob 0.65 \
  --seed 712 \
  --approach_noise_std 0.02 \
  --output logs/hsi_pregrasp/vla/language_wrong_object_smoke40_events.csv

conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 16 \
  --num_events 120 \
  --variant distractors \
  --num_distractors 4 \
  --skip_vla \
  --language_mode multi_object_default_policy \
  --language_default_prob 0.65 \
  --seed 721 \
  --approach_noise_std 0.02 \
  --label_horizon_s 1.0 \
  --output logs/hsi_pregrasp/vla/language_wrong_object_main120_events.csv
```

The second run was stopped after `63` usable events and renamed to
`logs/hsi_pregrasp/vla/language_wrong_object_main63_events.csv` because the camera-enabled simulator was slow. This is
a pilot result, not a final-scale dataset.

Language pilot distributions:

| Dataset | Events | Successes | Failures | Wrong-Object Failures |
| --- | ---: | ---: | ---: | ---: |
| `language_wrong_object_main63_events.csv` | `63` | `30` | `33` | `16` |
| `language_wrong_object_smoke40_events.csv` | `40` | `22` | `18` | `11` |

Feature-group command:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_feature_group_ablation.py \
  --input logs/hsi_pregrasp/vla/language_wrong_object_main63_events.csv \
  --eval-input logs/hsi_pregrasp/vla/language_wrong_object_smoke40_events.csv \
  --output logs/hsi_pregrasp/vla/language_wrong_object_ablation_main63.json \
  --checkpoint-dir logs/hsi_pregrasp/vla/checkpoints_language_main63 \
  --feature-groups robot_state,visual,language,visual_language,robot_visual_language \
  --epochs 200 \
  --seed 73 \
  --device cpu
```

Matched-acceptance command:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_matched_acceptance_ci.py \
  --input logs/hsi_pregrasp/vla/language_wrong_object_smoke40_events.csv \
  --title "Language Wrong-Object Pilot Matched-Acceptance Comparison" \
  --reference-checkpoint visual_language=logs/hsi_pregrasp/vla/checkpoints_language_main63/visual_language_refusal_head.pt \
  --checkpoint language=logs/hsi_pregrasp/vla/checkpoints_language_main63/language_refusal_head.pt \
  --checkpoint visual=logs/hsi_pregrasp/vla/checkpoints_language_main63/visual_refusal_head.pt \
  --checkpoint robot_state=logs/hsi_pregrasp/vla/checkpoints_language_main63/robot_state_refusal_head.pt \
  --checkpoint robot_visual_language=logs/hsi_pregrasp/vla/checkpoints_language_main63/robot_visual_language_refusal_head.pt \
  --scalar oracle_distance_only=ee_object_distance:low \
  --include-always-close \
  --include-matched-random \
  --bootstrap 2000 \
  --seed 812 \
  --device cpu \
  --output-json logs/hsi_pregrasp/vla/language_wrong_object_matched_ci.json \
  --output-md logs/hsi_pregrasp/vla/language_wrong_object_matched_ci.md
```

Matched-acceptance language pilot result:

| Method | Oracle Geometry? | False-Accept Risk, 95% CI | Accepted Success, 95% CI | Acceptance, 95% CI |
| --- | --- | ---: | ---: | ---: |
| `visual_language` | `no` | `0.1667` [`0.0000`, `0.4167`] | `0.8333` [`0.6000`, `1.0000`] | `0.3000` [`0.1750`, `0.4500`] |
| `matched_random` | `no` | `0.4507` [`0.2500`, `0.6667`] | `0.5493` [`0.3333`, `0.7500`] | `0.3000` [`0.3000`, `0.3000`] |
| `language` | `no` | `0.2500` [`0.0000`, `0.5385`] | `0.7500` [`0.4665`, `1.0000`] | `0.3000` [`0.1500`, `0.4500`] |
| `visual` | `no` | `0.3333` [`0.0909`, `0.6154`] | `0.6667` [`0.3750`, `0.9167`] | `0.3000` [`0.1750`, `0.4500`] |
| `robot_state` | `yes` | `0.4167` [`0.1429`, `0.7143`] | `0.5833` [`0.2857`, `0.8750`] | `0.3000` [`0.1750`, `0.4500`] |
| `robot_visual_language` | `yes` | `0.0000` [`0.0000`, `0.0000`] | `1.0000` [`1.0000`, `1.0000`] | `0.3000` [`0.1500`, `0.4500`] |
| `oracle_distance_only` | `yes` | `0.2500` [`0.0000`, `0.5008`] | `0.7500` [`0.5000`, `1.0000`] | `0.3000` [`0.1744`, `0.4500`] |

Interpretation: this pilot is the first completed simulator condition where geometry loses at matched acceptance.
No-oracle `visual_language` beats robot-state geometry and distance-only on the 40-event holdout. The confidence
intervals are wide because the holdout is small; scale this to at least `300/200` or `600/400` before making a final
paper claim.

Online VLA visual refusal, using the main600 visual checkpoint:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/run_online_vla_refusal_eval.py \
  --headless \
  --checkpoint logs/hsi_pregrasp/vla/checkpoints_main600/visual_refusal_head.pt \
  --num_envs 8 \
  --num_episodes 80 \
  --approach_noise_std 0.02 \
  --retry_noise_std 0.02 \
  --vla_samples 2 \
  --skip_vla \
  --seed 501 \
  --max_steps 12000 \
  --output logs/hsi_pregrasp/vla/online_visual_main600_events.csv \
  --summary logs/hsi_pregrasp/vla/online_visual_main600_summary.json
```

Matching online always-close baseline:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/run_online_vla_refusal_eval.py \
  --headless \
  --checkpoint logs/hsi_pregrasp/vla/checkpoints_main600/visual_refusal_head.pt \
  --num_envs 8 \
  --num_episodes 80 \
  --approach_noise_std 0.02 \
  --retry_noise_std 0.02 \
  --vla_samples 2 \
  --skip_vla \
  --seed 501 \
  --max_refusals_per_episode 0 \
  --max_steps 12000 \
  --output logs/hsi_pregrasp/vla/online_visual_main600_always_close_events.csv \
  --summary logs/hsi_pregrasp/vla/online_visual_main600_always_close_summary.json
```

Online VLA comparison:

| Metric | Always Close | Visual Refusal |
| --- | ---: | ---: |
| Completed episodes | `80` | `80` |
| Successful episodes | `43` | `47` |
| Episode success | `0.5375` | `0.5875` |
| Pre-grasp events | `55` | `60` |
| Accepted closures | `55` | `53` |
| Refused closures | `0` | `7` |
| Failed accepted closures | `12` | `6` |
| False-accept risk | `0.2182` | `0.1132` |
| Accepted grasp success | `0.7818` | `0.8868` |
| Acceptance rate | `1.0000` | `0.8833` |
| Refusal rate | `0.0000` | `0.1167` |
| Robot time per success | `9.3023 s` | `8.5106 s` |
| Attempts per success | `1.2791` | `1.1277` |
| Mean VLA inference | `0.0000 ms/event` | `0.0000 ms/event` |

The online visual run used `--skip_vla` because the selected checkpoint uses only camera summary features; this avoids
unneeded repeated SmolVLA sampling during online evaluation.

Distractor smoke:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 4 \
  --num_events 12 \
  --variant distractors \
  --num_distractors 2 \
  --approach_noise_std 0.02 \
  --vla_samples 1 \
  --max_steps 3000 \
  --stall_steps 1000 \
  --output logs/hsi_pregrasp/vla/distractor12_vla_events.csv
```

Distractor smoke result: `12` events, `8` successes, `4` failures.

Robustness smoke commands:

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 100 \
  --variant partial_occlusion \
  --num_distractors 1 \
  --approach_noise_std 0.02 \
  --vla_samples 2 \
  --skip_vla \
  --seed 606 \
  --max_steps 20000 \
  --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_partial_occlusion100_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 100 \
  --variant clutter \
  --num_distractors 3 \
  --approach_noise_std 0.02 \
  --vla_samples 2 \
  --skip_vla \
  --seed 601 \
  --max_steps 20000 \
  --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_clutter100_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 100 \
  --variant lighting_shift \
  --approach_noise_std 0.02 \
  --vla_samples 2 \
  --skip_vla \
  --seed 602 \
  --max_steps 20000 \
  --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_lighting100_full_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 100 \
  --variant camera_shift \
  --approach_noise_std 0.02 \
  --vla_samples 2 \
  --skip_vla \
  --seed 603 \
  --max_steps 20000 \
  --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_camera_shift100_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 100 \
  --variant object_shift \
  --approach_noise_std 0.02 \
  --vla_samples 2 \
  --skip_vla \
  --seed 604 \
  --max_steps 20000 \
  --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_object_shift100_events.csv

TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p \
  source/hsi_pregrasp_refusal/scripts/collect_vla_lift_pregrasp.py \
  --headless \
  --num_envs 8 \
  --num_events 100 \
  --variant single \
  --approach_noise_std 0.03 \
  --vla_samples 2 \
  --skip_vla \
  --seed 605 \
  --max_steps 20000 \
  --stall_steps 3000 \
  --output logs/hsi_pregrasp/vla/robust_approach_noise003_100_events.csv
```

Robustness dataset distributions:

| Dataset | Events | Successes | Failures | Failure Rate |
| --- | ---: | ---: | ---: | ---: |
| `robust_partial_occlusion100_events.csv` | `100` | `82` | `18` | `0.1800` |
| `robust_clutter100_events.csv` | `100` | `70` | `30` | `0.3000` |
| `robust_lighting100_full_events.csv` | `100` | `72` | `28` | `0.2800` |
| `robust_camera_shift100_events.csv` | `100` | `74` | `26` | `0.2600` |
| `robust_object_shift100_events.csv` | `100` | `78` | `22` | `0.2200` |
| `robust_approach_noise003_100_events.csv` | `100` | `38` | `62` | `0.6200` |

Wrong-object / stronger non-geometric variant status:

- Code support was added for `--variant wrong_object`, which places one distractor close to the target grasp corridor
  plus additional clutter.
- Two pilot collections were attempted with `80-120` requested events and `--skip_vla`, but they stalled after only
  `6-8` usable events because the occluder often blocked the state machine before a close event could be labeled.
- Those partial CSVs were deleted and are not used in any result table.
- This means the current completed simulation still has a geometry-oracle gap: clutter and partial occlusion are useful
  robustness checks, but they do not yet create enough aligned-but-failed wrong-object events.

Robustness evaluation with main600 checkpoints:

| Feature Group | Shift | Events | False-Accept Risk | Accepted Success | Acceptance Rate | Refusal Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `robot_state` | partial occlusion | `100` | `0.0899` | `0.9101` | `0.8900` | `0.1100` |
| `visual` | partial occlusion | `100` | `0.1034` | `0.8966` | `0.2900` | `0.7100` |
| `robot_visual` | partial occlusion | `100` | `0.0476` | `0.9524` | `0.8400` | `0.1600` |
| `full` | partial occlusion | `100` | `0.1183` | `0.8817` | `0.9300` | `0.0700` |
| `robot_state` | clutter | `100` | `0.1148` | `0.8852` | `0.6100` | `0.3900` |
| `visual` | clutter | `100` | `0.2558` | `0.7442` | `0.4300` | `0.5700` |
| `robot_visual` | clutter | `100` | `0.0612` | `0.9388` | `0.4900` | `0.5100` |
| `full` | clutter | `100` | `0.1207` | `0.8793` | `0.5800` | `0.4200` |
| `robot_state` | lighting | `100` | `0.1125` | `0.8875` | `0.8000` | `0.2000` |
| `visual` | lighting | `100` | n/a | n/a | `0.0000` | `1.0000` |
| `robot_visual` | lighting | `100` | `0.0000` | `1.0000` | `0.0700` | `0.9300` |
| `full` | lighting | `100` | n/a | n/a | `0.0000` | `1.0000` |
| `robot_state` | camera shift | `100` | `0.1685` | `0.8315` | `0.8900` | `0.1100` |
| `visual` | camera shift | `100` | `0.0500` | `0.9500` | `0.4000` | `0.6000` |
| `robot_visual` | camera shift | `100` | `0.0676` | `0.9324` | `0.7400` | `0.2600` |
| `full` | camera shift | `100` | `0.0312` | `0.9688` | `0.6400` | `0.3600` |
| `robot_state` | object shift | `100` | `0.0824` | `0.9176` | `0.8500` | `0.1500` |
| `visual` | object shift | `100` | `0.0455` | `0.9545` | `0.4400` | `0.5600` |
| `robot_visual` | object shift | `100` | `0.0411` | `0.9589` | `0.7300` | `0.2700` |
| `full` | object shift | `100` | `0.0556` | `0.9444` | `0.7200` | `0.2800` |
| `robot_state` | approach noise `0.03` | `100` | `0.1739` | `0.8261` | `0.4600` | `0.5400` |
| `visual` | approach noise `0.03` | `100` | `0.2292` | `0.7708` | `0.4800` | `0.5200` |
| `robot_visual` | approach noise `0.03` | `100` | `0.1163` | `0.8837` | `0.4300` | `0.5700` |
| `full` | approach noise `0.03` | `100` | `0.3214` | `0.6786` | `0.5600` | `0.4400` |

Matched-acceptance clutter comparison:

```bash
conda run -n env_isaaclab python source/hsi_pregrasp_refusal/scripts/run_matched_acceptance_ci.py \
  --input logs/hsi_pregrasp/vla/robust_clutter100_events.csv \
  --title "Clutter100 Matched-Acceptance Comparison" \
  --reference-checkpoint robot_visual=logs/hsi_pregrasp/vla/checkpoints_main600/robot_visual_refusal_head.pt \
  --checkpoint visual=logs/hsi_pregrasp/vla/checkpoints_main600/visual_refusal_head.pt \
  --checkpoint robot_state=logs/hsi_pregrasp/vla/checkpoints_main600/robot_state_refusal_head.pt \
  --scalar table_center=camera1_center_mean:low \
  --scalar wrist_center=camera2_center_mean:low \
  --scalar overhead_center=camera3_center_mean:low \
  --scalar oracle_distance_only=ee_object_distance:low \
  --include-always-close \
  --include-matched-random \
  --bootstrap 2000 \
  --seed 315 \
  --device cpu \
  --output-json logs/hsi_pregrasp/vla/clutter100_matched_ci.json \
  --output-md logs/hsi_pregrasp/vla/clutter100_matched_ci.md
```

The target acceptance is the default robot+visual clutter acceptance, `0.4900`. This table is the matched-acceptance
version of the clutter result, so improvements cannot be explained only by refusing more often.

| Method | Oracle Geometry? | False-Accept Risk, 95% CI | Accepted Success, 95% CI | Acceptance, 95% CI |
| --- | --- | ---: | ---: | ---: |
| `robot_visual` | `yes` | `0.0612` [`0.0000`, `0.1395`] | `0.9388` [`0.8636`, `1.0000`] | `0.4900` [`0.3900`, `0.5900`] |
| `matched_random` | `no` | `0.3008` [`0.2041`, `0.3878`] | `0.6992` [`0.6122`, `0.7959`] | `0.4900` [`0.4900`, `0.4900`] |
| `visual` | `no` | `0.2449` [`0.1250`, `0.3684`] | `0.7551` [`0.6250`, `0.8704`] | `0.4900` [`0.4000`, `0.5900`] |
| `robot_state` | `yes` | `0.0000` [`0.0000`, `0.0000`] | `1.0000` [`1.0000`, `1.0000`] | `0.4900` [`0.4000`, `0.5900`] |
| `table_center` | `no` | `0.2857` [`0.1731`, `0.4211`] | `0.7143` [`0.5814`, `0.8333`] | `0.4900` [`0.3900`, `0.5800`] |
| `wrist_center` | `no` | `0.2653` [`0.1455`, `0.3878`] | `0.7347` [`0.6000`, `0.8600`] | `0.4900` [`0.3900`, `0.5900`] |
| `overhead_center` | `no` | `0.3673` [`0.2400`, `0.5106`] | `0.6327` [`0.5000`, `0.7636`] | `0.4900` [`0.3900`, `0.5900`] |
| `oracle_distance_only` | `yes` | `0.0000` [`0.0000`, `0.0000`] | `1.0000` [`1.0000`, `1.0000`] | `0.4900` [`0.3900`, `0.5800`] |

Interpretation:

- The robot+visual checkpoint is strongest on the clutter smoke, reducing accepted failures but refusing about half of
  events.
- However, the matched clutter analysis still shows robot-state and distance-only geometry perfectly separating the
  accepted subset at this acceptance point. This clutter task is not yet the desired non-geometric failure test.
- Lighting shift breaks the camera-heavy checkpoints by causing near-total refusal.
- Camera and object shifts are handled well by at least one learned checkpoint, with false-accept risk below `0.10`.
- The harder approach-noise shift remains above the `0.10` false-accept target for all tested feature groups.

Online robustness smoke tests used the main600 `robot_visual` checkpoint with `40` episodes per shift and `--skip_vla`
because that checkpoint does not use VLA action-uncertainty columns:

| Shift | Policy | Episodes | Success | False-Accept Risk | Acceptance | Refusal | Time / Success | Attempts / Success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `partial_occlusion` | always close | `40` | `0.6250` | `0.1935` | `1.0000` | `0.0000` | `8.0000` | `1.2400` |
| `partial_occlusion` | robot+visual | `40` | `0.4000` | `0.2381` | `0.7000` | `0.3000` | `12.5000` | `1.3125` |
| `clutter` | always close | `40` | `0.2750` | `0.2143` | `1.0000` | `0.0000` | `18.1818` | `1.2727` |
| `clutter` | robot+visual | `40` | `0.2000` | `0.0000` | `0.3200` | `0.6800` | `25.0000` | `1.0000` |
| `camera_shift` | always close | `40` | `0.5000` | `0.2857` | `1.0000` | `0.0000` | `10.0000` | `1.4000` |
| `camera_shift` | robot+visual | `40` | `0.7250` | `0.0000` | `0.7436` | `0.2564` | `6.8966` | `1.0000` |
| `object_shift` | always close | `40` | `0.5500` | `0.1852` | `1.0000` | `0.0000` | `9.0909` | `1.2273` |
| `object_shift` | robot+visual | `40` | `0.4750` | `0.1739` | `0.6765` | `0.3235` | `10.5263` | `1.2105` |
| `approach003` | always close | `40` | `0.2500` | `0.4444` | `1.0000` | `0.0000` | `20.0000` | `1.8000` |
| `approach003` | robot+visual | `40` | `0.3750` | `0.1667` | `0.6207` | `0.3793` | `13.3333` | `1.2000` |

Online robustness interpretation:

- Robot+visual helps strongly for camera shift and harder approach noise.
- It reduces false accepts in clutter but loses task success because it refuses too often.
- It hurts partial occlusion and object shift in this 40-episode smoke, so online thresholds still need shift-aware
  tuning before final claims.

Diagnostic threshold tuning for the robot+visual checkpoint:

| Dataset | Tuned Threshold | Tuned False-Accept Risk | Tuned Acceptance | Default False-Accept Risk | Default Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `holdout400` | `0.999997` | `0.0975` | `0.7950` | `0.1351` | `0.8325` |
| `partial_occlusion100` | `1.000000` | `0.0476` | `0.8400` | `0.0476` | `0.8400` |
| `lighting100` | `1.000000` | `0.0000` | `0.0700` | `0.0000` | `0.0700` |
| `clutter100` | `1.000000` | `0.0612` | `0.4900` | `0.0612` | `0.4900` |
| `camera_shift100` | `1.000000` | `0.0676` | `0.7400` | `0.0676` | `0.7400` |
| `object_shift100` | `1.000000` | `0.0411` | `0.7300` | `0.0411` | `0.7300` |
| `approach003_100` | `0.999968` | `0.0952` | `0.4200` | `0.1163` | `0.4300` |
| `combined_robustness` | `1.000000` | `0.0980` | `0.6630` | `0.0980` | `0.6630` |

Report assets generated by `scripts/make_report_plots.py`:

- `logs/hsi_pregrasp/vla/report_assets/feature_groups_main600.png`
- `logs/hsi_pregrasp/vla/report_assets/robustness_false_accept_main600.png`
- `logs/hsi_pregrasp/vla/report_assets/online_robust_false_accept_main600.png`
- `logs/hsi_pregrasp/vla/report_assets/threshold_tuning_robot_visual_main600.png`

GPU check during VLA collection:

```bash
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi pmon -c 1
```

Observed result: the RTX 4060 Laptop GPU was active at `100%` overall GPU utilization, with the Python IsaacLab process
around `83%` SM, `52%` memory utilization, and about `5.7 GB` VRAM used. CPU usage is still expected because Isaac Sim
uses CPU work for simulation, camera readback, preprocessing, CSV writing, and Python orchestration.

## Timing

Observed on this machine:

- 30-event visible smoke collection: about `1.5 minutes`, including Isaac Sim GUI startup
- 800-event headless collection: about `1 minute`
- 400-event headless holdout collection: about `30-35 seconds`
- 400-episode online headless evaluation: about `30 seconds` per policy
- MLP training and calibration after CSV collection: about `1-2 seconds` on CPU
- camera-enabled visible VLA smoke: about `353 ms/event` of SmolVLA inference with `vla_samples=2`, plus Isaac startup
- camera-enabled VLA pilot: about `675 ms/event` of SmolVLA inference with `vla_samples=4`
- camera-enabled VLA holdout/online: about `328-333 ms/event` of SmolVLA inference with `vla_samples=2`
- proposal-scale VLA main collection: `600` events completed in about `78 minutes`
- proposal-scale VLA holdout collection: `400` events completed in about `56 minutes`
- feature-group ablation training over the main CSV: seconds on CPU once the CSV exists

The expensive part is simulation/data collection and repeated SmolVLA inference, not refusal-head training.

Estimated full camera/VLA collection time:

| Planned Run | VLA Samples | Events | Inference-Only Estimate | Practical Estimate |
| --- | ---: | ---: | ---: | ---: |
| Main dataset | `2` | `600-800` | `3.3-4.4 minutes` | `20-90 minutes` with Isaac startup/rendering/stalls |
| Main dataset | `4` | `600-800` | `6.8-9.0 minutes` | `1-3 hours` if rendering/stalls occur |
| Holdout | `2` | `400` | `2.2 minutes` | `10-45 minutes` |
| Online evaluation per policy | `2` | `80 episodes` | `2-4 minutes` inference | `10-45 minutes` depending attempts |

These are estimates from short runs. Long IsaacLab camera jobs can stall or slow down, so use `--max_steps`,
`--stall_steps`, and frequent CSV checks.

## Current Caveats

- This is IsaacLab simulation, not the real robot yet.
- The motion policy is still a deterministic state machine with injected approach noise. SmolVLA is used for frozen
  action-uncertainty features, not to control the Franka trajectory.
- SmolVLA currently provides sampled 6-D action uncertainty, not a true gripper-close log-probability.
- The early VLA pilot is only `95` events because the requested 150-event run stopped writing rows and was interrupted;
  the later main run completed `600` events cleanly.
- Visual features are simple image summary statistics, not learned visual embeddings yet.
- Lighting shift currently causes severe over-refusal for camera-heavy checkpoints.
- The calibration procedure is empirical selective-risk calibration. Do not claim a formal conformal guarantee yet.
- The current online reapproach is a simple simulator retry that resamples approach noise; it is not yet a real VLA/robot
  retreat, resense, and replan stack.

## What Is Still Needed From The Proposal

Completed in this scaffold:

- Stage 0 simulator smoke test
- 800-event main simulator dataset
- 400-event independent simulator holdout
- full refusal-head training and calibration
- always-close, matched-random, scalar geometry, geometry-head, action/state-head, and full-head offline baselines
- online refusal/reapproach simulator evaluation
- online always-close baseline
- approach-noise robustness shift sweep
- SmolVLA checkpoint download and runtime smoke test
- camera-enabled VLA visible smoke on IsaacLab observations
- SmolVLA adapter for table/wrist/overhead RGB cameras and a 6-D simulator proxy state
- VLA action-uncertainty feature logging
- visual feature logging
- feature-group training/evaluation for `robot_state`, `visual`, `vla_action_uncertainty`, `robot_visual`, and `full`
- offline VLA-sim scalar baselines
- online VLA-sim refusal/reapproach evaluation against an always-close baseline
- distractor variant smoke test
- proposal-scale VLA main dataset of `600` events
- proposal-scale VLA holdout dataset of `400` events
- VLA-sim robustness sweeps for partial occlusion, clutter, lighting shift, camera shift, object shift, and harder
  approach noise
- online robustness smoke tests for partial occlusion, clutter, camera shift, object shift, and harder approach noise
- diagnostic threshold tuning for the robot+visual checkpoint
- no-oracle matched-acceptance holdout table with confidence intervals
- matched-acceptance clutter table with confidence intervals
- multi-object/language wrong-object pilot with automatic simulator labels
- report-ready plot assets for feature groups, robustness, online robustness, and threshold tuning

Not yet completed:

- learned visual embeddings or RGB-D features
- final-scale non-geometric wrong-object dataset where distance/pose can no longer separate failures
- target-aware multi-object controller that can actually choose and lift the requested colored cube
- true VLA gripper-close probability; unavailable in the current 6-D SmolVLA action interface
- real robot smoke/pilot/main calibration/final test
- final paper polish around plots/tables

Blocked inputs:

- Real robot experiments need hardware access and a data-logging/labeling interface.
- RGB-D or stronger learned vision needs raw image/depth logging or a new embedding feature collector; current event CSVs
  contain visual summaries but not raw frames.
- A true close log-probability needs a VLA policy interface that exposes gripper-close probability/logits.

Data labeling:

- No manual labels are needed for simulator data. The label is generated automatically from lift success after the close
  event.
- Real robot labels should be logged with the same event schema once hardware access exists. Those labels will come from
  observed grasp/lift outcome, not manual frame-by-frame annotation.

Estimated remaining time:

| Work | Estimated Time |
| --- | ---: |
| Scale language wrong-object pilot to `300/200` or `600/400` | `0.5-1 day` |
| Tune physical wrong-object/occlusion condition and collect enough failures | `0.5-1.5 days` |
| Add target-aware multi-object controller for colored-cube lift | `1-3 days` |
| Add learned visual embeddings or RGB-D features | `1-3 days` |
| Tune/calibrate online thresholds with a held-out robustness calibration protocol | `0.5-1 day` |
| Real robot 30-event smoke test | `0.5 day` |
| Real robot 150-event pilot | `1 day` |
| Real robot 600-800 event main dataset | `2-4 days` |
| Real robot online final evaluation with frozen threshold | `1-2 days` |
| Paper figures, tables, video, and writing | `3-7 days` |

Practical total estimate:

- Simulator-only version: mostly complete
- VLA-in-simulation version: mostly complete for the implemented simulator variants; about `1-3 working days` remain for
  stronger vision and final threshold protocol polish
- Full proposal with real robot data and paper-ready figures: about `3-5 weeks`

## Next Steps

1. Scale the language-conditioned wrong-object pilot to `300/200` or `600/400` events.
2. Tune the physical `wrong_object` / occlusion simulator so it produces enough close events where distance is good but
   the target lift still fails.
3. Add a target-aware multi-object controller that can actually choose and lift the requested colored cube.
4. Add learned visual embeddings or RGB-D features if camera summary features remain too brittle.
5. Tune/calibrate online thresholds with a held-out robustness calibration protocol.
6. Move to real robot calibration with the same event schema once hardware/logging access exists.
7. Freeze the threshold before final test runs and report false-accept risk, acceptance rate, task success after
   re-approach, robot time per success, and attempts per success.
# hsi_pregrasp_refusal
