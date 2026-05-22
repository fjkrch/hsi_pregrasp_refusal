"""Run online pre-grasp refusal with retreat and re-approach in IsaacLab.

This script turns the offline refusal head into an online gate:

1. the state-machine policy proposes a close command,
2. the refusal head scores the pre-grasp event,
3. accepted events close the gripper,
4. refused events keep the gripper open, retreat above the object, resample the approach offset, and retry.

The current re-approach is intentionally simple because this is still the simulator scaffold. In the real VLA/robot
version, the same decision point should call the real retreat/resense/replan stack.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
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


parser = argparse.ArgumentParser(description="Run online HSI pre-grasp refusal in IsaacLab.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0", help="IsaacLab task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Refusal head checkpoint.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=400, help="Number of completed episodes to evaluate.")
parser.add_argument("--output", type=str, default="logs/hsi_pregrasp/online_refusal_events.csv")
parser.add_argument("--summary", type=str, default="logs/hsi_pregrasp/online_refusal_summary.json")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--label_horizon_s", type=float, default=1.2, help="Seconds after close trigger before labeling.")
parser.add_argument("--lift_height", type=float, default=0.04, help="Minimum object lift height for success.")
parser.add_argument("--seed", type=int, default=123, help="Random seed for IsaacLab and injected approach noise.")
parser.add_argument("--near_distance_threshold", type=float, default=0.08, help="Trigger distance threshold in meters.")
parser.add_argument("--open_width_threshold", type=float, default=0.04, help="Trigger gripper width threshold in meters.")
parser.add_argument("--approach_noise_std", type=float, default=0.02, help="Per-episode XY approach noise.")
parser.add_argument("--approach_height_bias", type=float, default=0.0, help="Z bias added to the grasp target.")
parser.add_argument("--retry_noise_std", type=float, default=0.02, help="XY approach noise resampled after each refusal.")
parser.add_argument("--max_refusals_per_episode", type=int, default=3, help="Force accept after this many refusals.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from hsi_pregrasp_refusal.features import PREGRASP_FEATURE_COLUMNS  # noqa: E402
from hsi_pregrasp_refusal.model import RefusalHead  # noqa: E402
from hsi_pregrasp_refusal.trigger import detect_pregrasp  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


wp.init()


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def retry_idx(self, env_ids: Sequence[int] | torch.Tensor):
        self.sm_state[env_ids] = 1
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                wp.from_torch(ee_pose.contiguous(), wp.transform),
                wp.from_torch(object_pose.contiguous(), wp.transform),
                wp.from_torch(des_object_pose.contiguous(), wp.transform),
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _quat_distance(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    alignment = torch.sum(current * desired, dim=-1).abs().clamp(max=1.0)
    return 1.0 - alignment


def _new_episode_noise(num_envs: int, device: torch.device | str, std: float) -> torch.Tensor:
    noise = torch.zeros((num_envs, 3), device=device)
    if std > 0.0:
        noise[:, :2] = torch.randn((num_envs, 2), device=device) * std
    noise[:, 2] = args_cli.approach_height_bias
    return noise


def _tensor_item(tensor: torch.Tensor, env_id: int) -> float:
    return float(tensor[env_id].detach().cpu().item())


@torch.inference_mode()
def _score_failure(
    model: RefusalHead,
    feature_columns: list[str],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    feature_tensors: dict[str, torch.Tensor],
    env_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    features = torch.stack([feature_tensors[column][env_ids].to(device=device) for column in feature_columns], dim=-1)
    features = (features - feature_mean) / torch.clamp(feature_std, min=1e-6)
    return model.predict_failure_probability(features)


def main():
    output_path = Path(args_cli.output)
    summary_path = Path(args_cli.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    checkpoint_device = torch.device("cpu")
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
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=checkpoint_device)
    feature_std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32, device=checkpoint_device)

    env_cfg: LiftEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    dt = env_cfg.sim.dt * env_cfg.decimation

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    pick_sm = PickAndLiftSm(dt, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01)

    robot = env.unwrapped.scene["robot"]
    finger_joint_ids, _ = robot.find_joints(["panda_finger.*"])

    previous_close_commanded = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    episode_noise = _new_episode_noise(env.unwrapped.num_envs, env.unwrapped.device, args_cli.approach_noise_std)
    refusals_in_episode = torch.zeros(env.unwrapped.num_envs, dtype=torch.int32, device=env.unwrapped.device)
    success_seen = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    episode_start_step = torch.zeros(env.unwrapped.num_envs, dtype=torch.int64, device=env.unwrapped.device)

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
        "total_robot_time_s": 0.0,
    }

    fieldnames = [
        "event_id",
        "env_id",
        "step",
        "time_s",
        "decision",
        "risk",
        "threshold",
        "forced_accept",
        "refusals_before_event",
        *PREGRASP_FEATURE_COLUMNS,
        "grasp_success",
        "label_reason",
    ]

    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        while simulation_app.is_running() and totals["completed_episodes"] < args_cli.num_episodes:
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
                ee_object_delta = tcp_position - object_position
                ee_object_distance = torch.linalg.norm(ee_object_delta, dim=-1)
                ee_object_lateral_error = torch.linalg.norm(ee_object_delta[:, :2], dim=-1)
                ee_object_height_error = ee_object_delta[:, 2]
                action_delta_pos = actions[:, :3] - tcp_position
                action_delta_pos_norm = torch.linalg.norm(action_delta_pos, dim=-1)
                action_delta_rot_distance = _quat_distance(tcp_orientation, actions[:, 3:7])
                close_commanded = actions[:, -1] < 0.0

                feature_tensors = {
                    "ee_object_distance": ee_object_distance,
                    "ee_object_lateral_error": ee_object_lateral_error,
                    "ee_object_height_error": ee_object_height_error,
                    "object_height": object_position[:, 2],
                    "gripper_width": gripper_width,
                    "action_delta_pos_norm": action_delta_pos_norm,
                    "action_delta_rot_distance": action_delta_rot_distance,
                    "close_commanded": close_commanded.float(),
                    "sm_state": pick_sm.sm_state.float(),
                    "sm_wait_time": pick_sm.sm_wait_time,
                }

                pregrasp_mask = detect_pregrasp(
                    close_commanded,
                    gripper_width,
                    ee_object_distance,
                    previous_close_commanded=previous_close_commanded,
                    open_width_threshold=args_cli.open_width_threshold,
                    near_distance_threshold=args_cli.near_distance_threshold,
                )
                pregrasp_ids = pregrasp_mask.nonzero(as_tuple=False).squeeze(-1)

                if pregrasp_ids.numel() > 0:
                    risks = _score_failure(
                        model,
                        feature_columns,
                        feature_mean,
                        feature_std,
                        feature_tensors,
                        pregrasp_ids,
                        checkpoint_device,
                    ).to(device=env.unwrapped.device)
                    forced_accept = refusals_in_episode[pregrasp_ids] >= args_cli.max_refusals_per_episode
                    accept_mask = (risks <= threshold) | forced_accept
                    refuse_mask = ~accept_mask

                    for local_idx, env_id in enumerate(pregrasp_ids.detach().cpu().tolist()):
                        risk = float(risks[local_idx].detach().cpu().item())
                        forced = bool(forced_accept[local_idx].detach().cpu().item())
                        accepted = bool(accept_mask[local_idx].detach().cpu().item())
                        totals["pregrasp_events"] += 1
                        base_row = {
                            "event_id": event_id,
                            "env_id": env_id,
                            "step": step,
                            "time_s": step * dt,
                            "decision": "accept" if accepted else "refuse",
                            "risk": risk,
                            "threshold": threshold,
                            "forced_accept": int(forced),
                            "refusals_before_event": int(refusals_in_episode[env_id].detach().cpu().item()),
                            "ee_object_distance": _tensor_item(ee_object_distance, env_id),
                            "ee_object_lateral_error": _tensor_item(ee_object_lateral_error, env_id),
                            "ee_object_height_error": _tensor_item(ee_object_height_error, env_id),
                            "object_height": _tensor_item(object_position[:, 2], env_id),
                            "gripper_width": _tensor_item(gripper_width, env_id),
                            "action_delta_pos_norm": _tensor_item(action_delta_pos_norm, env_id),
                            "action_delta_rot_distance": _tensor_item(action_delta_rot_distance, env_id),
                            "close_commanded": int(close_commanded[env_id].detach().cpu().item()),
                            "sm_state": int(pick_sm.sm_state[env_id].detach().cpu().item()),
                            "sm_wait_time": _tensor_item(pick_sm.sm_wait_time, env_id),
                            "grasp_success": "",
                            "label_reason": "",
                        }
                        if accepted:
                            totals["accepted_closures"] += 1
                            totals["forced_accepts"] += int(forced)
                            pending[env_id] = {
                                **base_row,
                                "_target_step": step + label_horizon_steps,
                                "_object_z_at_event": _tensor_item(object_position[:, 2], env_id),
                            }
                        else:
                            totals["refused_closures"] += 1
                            writer.writerow(base_row)
                        event_id += 1

                    refused_ids = pregrasp_ids[refuse_mask]
                    if refused_ids.numel() > 0:
                        refusals_in_episode[refused_ids] += 1
                        episode_noise[refused_ids] = _new_episode_noise(
                            len(refused_ids), env.unwrapped.device, args_cli.retry_noise_std
                        )
                        pick_sm.retry_idx(refused_ids)
                        retreat_position = object_position[refused_ids] + episode_noise[refused_ids]
                        retreat_position[:, 2] += 0.1
                        actions[refused_ids, :3] = retreat_position
                        actions[refused_ids, 3:7] = desired_orientation[refused_ids]
                        actions[refused_ids, -1] = 1.0
                        close_commanded[refused_ids] = False

                done_ids = set(dones.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist())
                for env_id, row in list(pending.items()):
                    reached_horizon = step >= int(row["_target_step"])
                    env_done = env_id in done_ids
                    if not reached_horizon and not env_done:
                        continue
                    lifted = _tensor_item(object_position[:, 2], env_id) - float(row["_object_z_at_event"])
                    grasp_success = int(lifted >= args_cli.lift_height)
                    row["grasp_success"] = grasp_success
                    row["label_reason"] = "done" if env_done else "horizon"
                    row.pop("_target_step")
                    row.pop("_object_z_at_event")
                    writer.writerow(row)
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
                    episode_noise[reset_ids] = _new_episode_noise(
                        len(reset_ids), env.unwrapped.device, args_cli.approach_noise_std
                    )
                    refusals_in_episode[reset_ids] = 0
                    success_seen[reset_ids] = False
                    episode_start_step[reset_ids] = step
                    close_commanded[reset_ids] = False
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
        "checkpoint": str(args_cli.checkpoint),
        "num_envs": args_cli.num_envs,
        "target_episodes": args_cli.num_episodes,
        "approach_noise_std": args_cli.approach_noise_std,
        "retry_noise_std": args_cli.retry_noise_std,
        "max_refusals_per_episode": args_cli.max_refusals_per_episode,
        "false_accept_risk": totals["failed_accepted_closures"] / accepted if accepted else None,
        "accepted_grasp_success": totals["successful_accepted_closures"] / accepted if accepted else None,
        "acceptance_rate": accepted / pregrasp if pregrasp else 0.0,
        "refusal_rate": totals["refused_closures"] / pregrasp if pregrasp else 0.0,
        "task_success_after_reapproach": successful_episodes / completed if completed else 0.0,
        "robot_time_per_success_s": totals["total_robot_time_s"] / successful_episodes if successful_episodes else None,
        "attempts_per_success": accepted / successful_episodes if successful_episodes else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
