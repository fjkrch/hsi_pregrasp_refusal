"""Collect pre-grasp close events from the IsaacLab Franka cube lift task.

This is the Stage 0 data path for the HSI / pre-grasp refusal project. It uses IsaacLab's deterministic lift state
machine as a surrogate policy, logs the first close command while the gripper is still open and near the object, then
labels whether that closure lifted the object after a short horizon.

Example:
    ./isaaclab.sh -p source/hsi_pregrasp_refusal/scripts/collect_lift_cube_pregrasp.py \\
        --headless --num_envs 32 --num_events 30 --approach_noise_std 0.02 \\
        --output logs/hsi_pregrasp/stage0_events.csv
"""

from __future__ import annotations

import argparse
import csv
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


parser = argparse.ArgumentParser(description="Collect pre-grasp events from Isaac-Lift-Cube-Franka-IK-Abs-v0.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0", help="IsaacLab task id.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments.")
parser.add_argument("--num_events", type=int, default=30, help="Number of labeled pre-grasp events to write.")
parser.add_argument("--output", type=str, default="logs/hsi_pregrasp/stage0_events.csv", help="Output CSV path.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--label_horizon_s", type=float, default=1.2, help="Seconds after close trigger before labeling.")
parser.add_argument("--lift_height", type=float, default=0.04, help="Minimum object lift height for success.")
parser.add_argument("--seed", type=int, default=7, help="Random seed for IsaacLab and injected approach noise.")
parser.add_argument("--near_distance_threshold", type=float, default=0.08, help="Trigger distance threshold in meters.")
parser.add_argument("--open_width_threshold", type=float, default=0.04, help="Trigger gripper width threshold in meters.")
parser.add_argument(
    "--approach_noise_std",
    type=float,
    default=0.0,
    help="Per-episode XY noise added to the state-machine target. Use 0.01-0.03 m to create pilot failures.",
)
parser.add_argument(
    "--approach_height_bias",
    type=float,
    default=0.0,
    help="Z bias added to the state-machine grasp target. Positive values grasp above the object.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from hsi_pregrasp_refusal.features import PREGRASP_FEATURE_COLUMNS  # noqa: E402
from hsi_pregrasp_refusal.trigger import detect_pregrasp  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times before switching states."""

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
    """Simple task-space state machine for pick and lift."""

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

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
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


def main():
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args_cli.seed)

    env_cfg: LiftEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        position_threshold=0.01,
    )

    robot = env.unwrapped.scene["robot"]
    finger_joint_ids, _ = robot.find_joints(["panda_finger.*"])
    previous_close_commanded = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    episode_noise = _new_episode_noise(env.unwrapped.num_envs, env.unwrapped.device, args_cli.approach_noise_std)

    fieldnames = [
        "event_id",
        "env_id",
        "step",
        "time_s",
        "label_horizon_s",
        *PREGRASP_FEATURE_COLUMNS,
        "close_accepted",
        "close_was_refused",
        "grasp_success",
        "label_reason",
    ]

    pending: dict[int, dict[str, object]] = {}
    label_horizon_steps = max(1, int(round(args_cli.label_horizon_s / (env_cfg.sim.dt * env_cfg.decimation))))
    step = 0
    next_event_id = 0
    written_events = 0

    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        while simulation_app.is_running() and written_events < args_cli.num_events:
            with torch.inference_mode():
                dones = env.step(actions)[-2]
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

                pregrasp_mask = detect_pregrasp(
                    close_commanded,
                    gripper_width,
                    ee_object_distance,
                    previous_close_commanded=previous_close_commanded,
                    open_width_threshold=args_cli.open_width_threshold,
                    near_distance_threshold=args_cli.near_distance_threshold,
                )

                for env_id in pregrasp_mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist():
                    if env_id in pending:
                        continue
                    pending[env_id] = {
                        "event_id": next_event_id,
                        "env_id": env_id,
                        "step": step,
                        "time_s": step * env_cfg.sim.dt * env_cfg.decimation,
                        "label_horizon_s": args_cli.label_horizon_s,
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
                        "close_accepted": 1,
                        "close_was_refused": 0,
                        "_target_step": step + label_horizon_steps,
                        "_object_z_at_event": _tensor_item(object_position[:, 2], env_id),
                    }
                    next_event_id += 1

                done_ids = set(dones.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist())
                for env_id, row in list(pending.items()):
                    reached_horizon = step >= int(row["_target_step"])
                    env_done = env_id in done_ids
                    if not reached_horizon and not env_done:
                        continue
                    lifted = _tensor_item(object_position[:, 2], env_id) - float(row["_object_z_at_event"])
                    row["grasp_success"] = int(lifted >= args_cli.lift_height)
                    row["label_reason"] = "done" if env_done else "horizon"
                    row.pop("_target_step")
                    row.pop("_object_z_at_event")
                    writer.writerow(row)
                    written_events += 1
                    pending.pop(env_id)
                    if written_events >= args_cli.num_events:
                        break

                if dones.any():
                    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    pick_sm.reset_idx(reset_ids)
                    episode_noise[reset_ids] = _new_episode_noise(
                        len(reset_ids), env.unwrapped.device, args_cli.approach_noise_std
                    )
                    close_commanded[reset_ids] = False
                previous_close_commanded = close_commanded.clone()

    print(f"[INFO] Wrote {written_events} labeled pre-grasp events to {output_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
