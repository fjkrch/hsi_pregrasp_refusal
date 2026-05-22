"""Run online pre-grasp refusal with camera/VLA features in IsaacLab."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
for path in [
    PROJECT_ROOT,
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run online HSI VLA/camera pre-grasp refusal in IsaacLab.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--output", type=str, default="logs/hsi_pregrasp/vla/online_vla_refusal_events.csv")
parser.add_argument("--summary", type=str, default="logs/hsi_pregrasp/vla/online_vla_refusal_summary.json")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--label_horizon_s", type=float, default=1.2)
parser.add_argument("--lift_height", type=float, default=0.04)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--near_distance_threshold", type=float, default=0.08)
parser.add_argument("--open_width_threshold", type=float, default=0.04)
parser.add_argument("--approach_noise_std", type=float, default=0.02)
parser.add_argument("--approach_height_bias", type=float, default=0.0)
parser.add_argument("--retry_noise_std", type=float, default=0.02)
parser.add_argument("--max_refusals_per_episode", type=int, default=3)
parser.add_argument(
    "--variant",
    choices=[
        "single",
        "distractors",
        "clutter",
        "partial_occlusion",
        "wrong_object",
        "lighting_shift",
        "camera_shift",
        "object_shift",
    ],
    default="single",
)
parser.add_argument("--num_distractors", type=int, default=0)
parser.add_argument("--camera_width", type=int, default=256)
parser.add_argument("--camera_height", type=int, default=256)
parser.add_argument(
    "--disable_cameras",
    action="store_true",
    default=False,
    help="Do not add/render cameras. Visual and VLA columns are filled with zeros; useful when RTX rendering is unavailable.",
)
parser.add_argument("--vla_checkpoint", type=str, default="logs/hsi_pregrasp/vla/smolvla_base")
parser.add_argument("--vla_task", type=str, default="pick up the cube")
parser.add_argument("--vla_samples", type=int, default=2)
parser.add_argument("--vla_device", type=str, default="cuda")
parser.add_argument("--skip_vla", action="store_true", default=False)
parser.add_argument(
    "--decision_mode",
    choices=["model", "always_close"],
    default="model",
    help="model uses the refusal checkpoint; always_close accepts every pre-grasp event as a simulation baseline.",
)
parser.add_argument(
    "--language_mode",
    choices=["fixed", "multi_object_default_policy"],
    default="fixed",
    help=(
        "fixed keeps one task string. multi_object_default_policy samples colored-cube commands while the "
        "state-machine policy still grasps the default cube, creating automatic wrong-object failures."
    ),
)
parser.add_argument(
    "--language_default_prob",
    type=float,
    default=0.65,
    help="Probability that multi_object_default_policy asks for the default cube instead of a colored distractor.",
)
parser.add_argument("--max_steps", type=int, default=30000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


if args_cli.disable_cameras and not args_cli.skip_vla:
    parser.error("--disable_cameras requires --skip_vla because SmolVLA needs RGB observations.")

app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=not args_cli.disable_cameras)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from hsi_pregrasp_refusal.features import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    LANGUAGE_FEATURE_COLUMNS,
    PREGRASP_FEATURE_COLUMNS,
    language_feature_row,
)
from hsi_pregrasp_refusal.isaaclab_vla_scene import (  # noqa: E402
    collect_smolvla_images,
    configure_vla_lift_scene,
    reset_distractors,
)
from hsi_pregrasp_refusal.model import RefusalHead  # noqa: E402
from hsi_pregrasp_refusal.sim_analysis import (  # noqa: E402
    failure_type,
    instruction_for_target,
    metrics_by_failure_type,
    sample_language_target,
)
from hsi_pregrasp_refusal.state_machine import (  # noqa: E402
    PickAndLiftSm,
    new_episode_noise,
    quat_distance,
    tensor_item,
)
from hsi_pregrasp_refusal.trigger import detect_pregrasp  # noqa: E402
from hsi_pregrasp_refusal.vision import visual_feature_row, zero_visual_feature_row  # noqa: E402
from hsi_pregrasp_refusal.vla import SmolVLAAdapter, build_smolvla_state, zero_vla_feature_row  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


wp.init()


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _robot_feature_tensors(
    *,
    tcp_position: torch.Tensor,
    tcp_orientation: torch.Tensor,
    object_position: torch.Tensor,
    gripper_width: torch.Tensor,
    actions: torch.Tensor,
    close_commanded: torch.Tensor,
    pick_sm: PickAndLiftSm,
) -> dict[str, torch.Tensor]:
    ee_object_delta = tcp_position - object_position
    action_delta_pos = actions[:, :3] - tcp_position
    return {
        "ee_object_distance": torch.linalg.norm(ee_object_delta, dim=-1),
        "ee_object_lateral_error": torch.linalg.norm(ee_object_delta[:, :2], dim=-1),
        "ee_object_height_error": ee_object_delta[:, 2],
        "object_height": object_position[:, 2],
        "gripper_width": gripper_width,
        "action_delta_pos_norm": torch.linalg.norm(action_delta_pos, dim=-1),
        "action_delta_rot_distance": quat_distance(tcp_orientation, actions[:, 3:7]),
        "close_commanded": close_commanded.float(),
        "sm_state": pick_sm.sm_state.float(),
        "sm_wait_time": pick_sm.sm_wait_time,
    }


def _row_from_feature_tensors(feature_tensors: dict[str, torch.Tensor], env_id: int) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for column in PREGRASP_FEATURE_COLUMNS:
        value = feature_tensors[column]
        if column in {"close_commanded", "sm_state"}:
            row[column] = int(value[env_id].detach().cpu().item())
        else:
            row[column] = tensor_item(value, env_id)
    return row


@torch.inference_mode()
def _score_row(
    model: RefusalHead,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    row: dict[str, object],
    device: torch.device,
) -> float:
    features = np.asarray([float(row[column]) for column in feature_columns], dtype=np.float32)
    features = (features - feature_mean) / np.maximum(feature_std, 1e-6)
    tensor = torch.as_tensor(features[None, :], dtype=torch.float32, device=device)
    return float(model.predict_failure_probability(tensor).detach().cpu().item())


def main():
    output_path = Path(args_cli.output)
    summary_path = Path(args_cli.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    rng = random.Random(args_cli.seed)

    checkpoint_device = torch.device("cpu")
    model = None
    checkpoint = None
    feature_columns: list[str] = []
    threshold = 0.0
    feature_mean = np.asarray([], dtype=np.float32)
    feature_std = np.asarray([], dtype=np.float32)
    if args_cli.decision_mode == "model":
        if args_cli.checkpoint is None:
            raise ValueError("--checkpoint is required when --decision_mode=model")
        checkpoint = _load_checkpoint(args_cli.checkpoint, checkpoint_device)
        feature_columns = list(checkpoint["feature_columns"])
        model = RefusalHead(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims=tuple(int(value) for value in checkpoint["hidden_dims"]),
            dropout=float(checkpoint["dropout"]),
        ).to(checkpoint_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        threshold = float(checkpoint["threshold"])
        feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

    env_cfg: LiftEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    requested_distractors = args_cli.num_distractors
    if args_cli.language_mode == "multi_object_default_policy":
        requested_distractors = max(requested_distractors, 4)
    effective_distractors = configure_vla_lift_scene(
        env_cfg,
        variant=args_cli.variant,
        camera_width=args_cli.camera_width,
        camera_height=args_cli.camera_height,
        num_distractors=requested_distractors,
        add_cameras=not args_cli.disable_cameras,
    )
    dt = env_cfg.sim.dt * env_cfg.decimation
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    reset_distractors(
        env,
        torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device),
        variant=args_cli.variant,
        num_distractors=effective_distractors,
    )

    vla = None
    if not args_cli.skip_vla:
        vla = SmolVLAAdapter(
            args_cli.vla_checkpoint,
            device=args_cli.vla_device,
            task=args_cli.vla_task,
            image_size=(args_cli.camera_height, args_cli.camera_width),
        )

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    pick_sm = PickAndLiftSm(dt, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01)

    robot = env.unwrapped.scene["robot"]
    finger_joint_ids, _ = robot.find_joints(["panda_finger.*"])
    previous_close_commanded = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    episode_noise = new_episode_noise(
        env.unwrapped.num_envs,
        env.unwrapped.device,
        args_cli.approach_noise_std,
        height_bias=args_cli.approach_height_bias,
    )
    refusals_in_episode = torch.zeros(env.unwrapped.num_envs, dtype=torch.int32, device=env.unwrapped.device)
    success_seen = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    episode_start_step = torch.zeros(env.unwrapped.num_envs, dtype=torch.int64, device=env.unwrapped.device)
    language_targets = [
        sample_language_target(
            rng,
            mode=args_cli.language_mode,
            default_probability=args_cli.language_default_prob,
        )
        for _ in range(env.unwrapped.num_envs)
    ]
    pending: dict[int, dict[str, object]] = {}
    label_horizon_steps = max(1, int(round(args_cli.label_horizon_s / dt)))
    step = 0
    event_id = 0

    totals = {
        "completed_episodes": 0,
        "successful_episodes": 0,
        "pregrasp_events": 0,
        "accepted_closures": 0,
        "refused_closures": 0,
        "successful_accepted_closures": 0,
        "failed_accepted_closures": 0,
        "forced_accepts": 0,
        "wrong_object_events": 0,
        "wrong_object_accepted_closures": 0,
        "total_robot_time_s": 0.0,
        "total_vla_inference_ms": 0.0,
    }
    all_rows: list[dict[str, object]] = []
    all_success: list[bool] = []
    all_accepted: list[bool] = []
    all_wrong_object: list[bool] = []

    fieldnames = [
        "event_id",
        "env_id",
        "step",
        "time_s",
        "label_horizon_s",
        "variant",
        "language_mode",
        "language_instruction",
        "language_target",
        "decision",
        "risk",
        "threshold",
        "forced_accept",
        "refusals_before_event",
        "vla_samples",
        "vla_inference_ms",
        *ALL_FEATURE_COLUMNS,
        *LANGUAGE_FEATURE_COLUMNS,
        "close_accepted",
        "close_was_refused",
        "wrong_object",
        "failure_type",
        "grasp_success",
        "label_reason",
    ]

    def refresh_language(reset_ids: torch.Tensor):
        for reset_id in reset_ids.detach().cpu().tolist():
            language_targets[reset_id] = sample_language_target(
                rng,
                mode=args_cli.language_mode,
                default_probability=args_cli.language_default_prob,
            )

    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        while (
            simulation_app.is_running()
            and totals["completed_episodes"] < args_cli.num_episodes
            and step < args_cli.max_steps
        ):
            with torch.inference_mode():
                step_result = env.step(actions)
                terminated = step_result[2]
                truncated = step_result[3]
                dones = torch.logical_or(terminated, truncated)
                step += 1

                ee_frame_sensor = env.unwrapped.scene["ee_frame"]
                tcp_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
                tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
                object_data = env.unwrapped.scene["object"].data
                object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
                desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

                sm_object_position = object_position + episode_noise
                actions = pick_sm.compute(
                    torch.cat([tcp_position, tcp_orientation], dim=-1),
                    torch.cat([sm_object_position, desired_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )

                gripper_width = robot.data.joint_pos[:, finger_joint_ids].sum(dim=-1)
                close_commanded = actions[:, -1] < 0.0
                feature_tensors = _robot_feature_tensors(
                    tcp_position=tcp_position,
                    tcp_orientation=tcp_orientation,
                    object_position=object_position,
                    gripper_width=gripper_width,
                    actions=actions,
                    close_commanded=close_commanded,
                    pick_sm=pick_sm,
                )
                pregrasp_mask = detect_pregrasp(
                    close_commanded,
                    gripper_width,
                    feature_tensors["ee_object_distance"],
                    previous_close_commanded=previous_close_commanded,
                    open_width_threshold=args_cli.open_width_threshold,
                    near_distance_threshold=args_cli.near_distance_threshold,
                )
                pregrasp_ids = pregrasp_mask.nonzero(as_tuple=False).squeeze(-1)

                for env_id in pregrasp_ids.detach().cpu().tolist():
                    images = {} if args_cli.disable_cameras else collect_smolvla_images(env, env_id)
                    visual_row = visual_feature_row(images) if images else zero_visual_feature_row()
                    vla_row = zero_vla_feature_row()
                    vla_inference_ms = 0.0
                    language_target = language_targets[env_id]
                    instruction = instruction_for_target(language_target, args_cli.vla_task)
                    if vla is not None:
                        vla.task = instruction
                        vla_state = build_smolvla_state(tcp_position[env_id], object_position[env_id], actions[env_id, :3])
                        summary = vla.sample_actions(images, vla_state, num_samples=args_cli.vla_samples)
                        vla_row = summary.feature_row()
                        vla_inference_ms = summary.inference_ms
                        totals["total_vla_inference_ms"] += vla_inference_ms
                    base_row = {
                        "event_id": event_id,
                        "env_id": env_id,
                        "step": step,
                        "time_s": step * dt,
                        "label_horizon_s": args_cli.label_horizon_s,
                        "variant": args_cli.variant,
                        "language_mode": args_cli.language_mode,
                        "language_instruction": instruction,
                        "language_target": language_target,
                        "threshold": threshold,
                        "refusals_before_event": int(refusals_in_episode[env_id].detach().cpu().item()),
                        "vla_samples": 0 if vla is None else args_cli.vla_samples,
                        "vla_inference_ms": vla_inference_ms,
                        **_row_from_feature_tensors(feature_tensors, env_id),
                        **visual_row,
                        **vla_row,
                        **language_feature_row(language_target),
                        "close_accepted": "",
                        "close_was_refused": "",
                        "wrong_object": int(language_target != "default"),
                        "failure_type": "",
                        "grasp_success": "",
                        "label_reason": "",
                    }
                    if args_cli.decision_mode == "always_close":
                        risk = 0.0
                    else:
                        assert model is not None
                        risk = _score_row(model, feature_columns, feature_mean, feature_std, base_row, checkpoint_device)
                    forced_accept = refusals_in_episode[env_id] >= args_cli.max_refusals_per_episode
                    accepted = args_cli.decision_mode == "always_close" or risk <= threshold or bool(
                        forced_accept.detach().cpu().item()
                    )
                    base_row["risk"] = risk
                    base_row["decision"] = "accept" if accepted else "refuse"
                    base_row["forced_accept"] = int(bool(forced_accept.detach().cpu().item()))
                    base_row["close_accepted"] = int(accepted)
                    base_row["close_was_refused"] = int(not accepted)
                    base_row["failure_type"] = failure_type(base_row)
                    totals["pregrasp_events"] += 1
                    totals["wrong_object_events"] += int(language_target != "default")
                    if accepted:
                        totals["accepted_closures"] += 1
                        totals["forced_accepts"] += int(base_row["forced_accept"])
                        totals["wrong_object_accepted_closures"] += int(language_target != "default")
                        pending[env_id] = {
                            **base_row,
                            "_target_step": step + label_horizon_steps,
                            "_object_z_at_event": tensor_item(object_position[:, 2], env_id),
                        }
                    else:
                        totals["refused_closures"] += 1
                        base_row["grasp_success"] = 0
                        base_row["label_reason"] = "refused_wrong_object" if language_target != "default" else "refused"
                        base_row["failure_type"] = failure_type(base_row)
                        writer.writerow(base_row)
                        stream.flush()
                        all_rows.append(dict(base_row))
                        all_success.append(False)
                        all_accepted.append(False)
                        all_wrong_object.append(language_target != "default")
                        refusals_in_episode[env_id] += 1
                        episode_noise[env_id : env_id + 1] = new_episode_noise(
                            1,
                            env.unwrapped.device,
                            args_cli.retry_noise_std,
                            height_bias=args_cli.approach_height_bias,
                        )
                        pick_sm.retry_idx(torch.tensor([env_id], device=env.unwrapped.device))
                        retreat_position = object_position[env_id] + episode_noise[env_id]
                        retreat_position[2] += 0.1
                        actions[env_id, :3] = retreat_position
                        actions[env_id, 3:7] = desired_orientation[env_id]
                        actions[env_id, -1] = 1.0
                        close_commanded[env_id] = False
                    event_id += 1

                done_ids = set(dones.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist())
                for env_id, row in list(pending.items()):
                    reached_horizon = step >= int(row["_target_step"])
                    env_done = env_id in done_ids
                    if not reached_horizon and not env_done:
                        continue
                    lifted = tensor_item(object_position[:, 2], env_id) - float(row["_object_z_at_event"])
                    default_cube_lifted = lifted >= args_cli.lift_height
                    target_is_default = row.get("language_target", "default") == "default"
                    grasp_success = int(default_cube_lifted and target_is_default)
                    row["grasp_success"] = grasp_success
                    if not target_is_default and default_cube_lifted:
                        row["label_reason"] = "wrong_object_default_cube_lifted"
                    else:
                        row["label_reason"] = "done" if env_done else "horizon"
                    row["failure_type"] = failure_type(row)
                    row.pop("_target_step")
                    row.pop("_object_z_at_event")
                    writer.writerow(row)
                    stream.flush()
                    all_rows.append(dict(row))
                    all_success.append(bool(grasp_success))
                    all_accepted.append(True)
                    all_wrong_object.append(bool(row.get("wrong_object", 0)))
                    if grasp_success:
                        totals["successful_accepted_closures"] += 1
                        success_seen[env_id] = True
                    else:
                        totals["failed_accepted_closures"] += 1
                    pending.pop(env_id)

                if dones.any():
                    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    for env_id in reset_ids.detach().cpu().tolist():
                        elapsed = (step - int(episode_start_step[env_id].detach().cpu().item())) * dt
                        totals["completed_episodes"] += 1
                        totals["total_robot_time_s"] += float(elapsed)
                        totals["successful_episodes"] += int(success_seen[env_id].detach().cpu().item())
                    pick_sm.reset_idx(reset_ids)
                    episode_noise[reset_ids] = new_episode_noise(
                        len(reset_ids),
                        env.unwrapped.device,
                        args_cli.approach_noise_std,
                        height_bias=args_cli.approach_height_bias,
                    )
                    refusals_in_episode[reset_ids] = 0
                    success_seen[reset_ids] = False
                    episode_start_step[reset_ids] = step
                    close_commanded[reset_ids] = False
                    reset_distractors(
                        env,
                        reset_ids,
                        variant=args_cli.variant,
                        num_distractors=effective_distractors,
                    )
                    refresh_language(reset_ids)
                    for env_id in reset_ids.detach().cpu().tolist():
                        pending.pop(env_id, None)

                previous_close_commanded = close_commanded.clone()

    accepted = totals["accepted_closures"]
    pregrasp = totals["pregrasp_events"]
    successful_episodes = totals["successful_episodes"]
    completed = totals["completed_episodes"]
    summary = {
        **totals,
        "threshold": threshold,
        "checkpoint": str(args_cli.checkpoint) if args_cli.checkpoint is not None else None,
        "decision_mode": args_cli.decision_mode,
        "feature_columns": feature_columns,
        "num_envs": args_cli.num_envs,
        "target_episodes": args_cli.num_episodes,
        "variant": args_cli.variant,
        "language_mode": args_cli.language_mode,
        "language_default_prob": args_cli.language_default_prob,
        "effective_distractors": effective_distractors,
        "vla_samples": 0 if vla is None else args_cli.vla_samples,
        "approach_noise_std": args_cli.approach_noise_std,
        "retry_noise_std": args_cli.retry_noise_std,
        "max_refusals_per_episode": args_cli.max_refusals_per_episode,
        "false_accept_risk": _safe_div(totals["failed_accepted_closures"], accepted),
        "accepted_grasp_success": _safe_div(totals["successful_accepted_closures"], accepted),
        "accepted_success": _safe_div(totals["successful_accepted_closures"], accepted),
        "wrong_object_false_accept_rate": _safe_div(totals["wrong_object_accepted_closures"], accepted),
        "acceptance_rate": accepted / pregrasp if pregrasp else 0.0,
        "refusal_rate": totals["refused_closures"] / pregrasp if pregrasp else 0.0,
        "task_success_after_reapproach": successful_episodes / completed if completed else 0.0,
        "robot_time_per_success_s": _safe_div(totals["total_robot_time_s"], successful_episodes),
        "attempts_per_success": _safe_div(accepted, successful_episodes),
        "mean_vla_inference_ms_per_pregrasp": _safe_div(totals["total_vla_inference_ms"], pregrasp),
    }
    if all_rows:
        summary["failure_type_metrics"] = metrics_by_failure_type(
            np.asarray(all_accepted, dtype=bool),
            np.asarray(all_success, dtype=bool),
            all_rows,
            wrong_object=np.asarray(all_wrong_object, dtype=bool),
        )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
