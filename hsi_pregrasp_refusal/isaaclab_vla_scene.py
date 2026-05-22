"""IsaacLab scene helpers for camera-enabled VLA pre-grasp experiments.

Import this module only after IsaacLab's AppLauncher has started.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


CAMERA_SCENE_TO_SMOLVLA = {
    "vla_table_cam": "camera1",
    "vla_wrist_cam": "camera2",
    "vla_overhead_cam": "camera3",
}

VARIANTS = [
    "single",
    "distractors",
    "clutter",
    "partial_occlusion",
    "wrong_object",
    "lighting_shift",
    "camera_shift",
    "object_shift",
]


def configure_vla_lift_scene(
    env_cfg,
    *,
    variant: str,
    camera_width: int = 256,
    camera_height: int = 256,
    num_distractors: int = 0,
    add_cameras: bool = True,
) -> int:
    """Mutate a lift env config with optional VLA cameras and distractor objects."""
    if variant not in VARIANTS:
        valid = ", ".join(VARIANTS)
        raise ValueError(f"Unknown variant {variant!r}. Valid variants: {valid}")
    if add_cameras:
        _add_cameras(env_cfg, variant=variant, width=camera_width, height=camera_height)
    effective_distractors = num_distractors
    if variant in {"distractors", "lighting_shift", "camera_shift", "object_shift"}:
        effective_distractors = max(effective_distractors, 2)
    if variant == "clutter":
        effective_distractors = max(effective_distractors, 3)
    if variant == "partial_occlusion":
        effective_distractors = max(effective_distractors, 1)
    if variant == "wrong_object":
        effective_distractors = max(effective_distractors, 3)
    if effective_distractors > 0:
        _add_distractors(env_cfg, effective_distractors)
    if variant == "lighting_shift":
        env_cfg.scene.light.spawn.intensity = 1200.0
        env_cfg.scene.light.spawn.color = (0.9, 0.82, 0.72)
    if variant == "object_shift":
        env_cfg.events.reset_object_position.params["pose_range"] = {
            "x": (-0.16, 0.16),
            "y": (-0.32, 0.32),
            "z": (0.0, 0.0),
        }
    if add_cameras:
        env_cfg.num_rerenders_on_reset = 3
        env_cfg.sim.render.antialiasing_mode = "DLAA"
    return effective_distractors


def collect_smolvla_images(env, env_id: int) -> dict[str, torch.Tensor]:
    """Read one environment's three RGB cameras as SmolVLA camera keys."""
    images: dict[str, torch.Tensor] = {}
    for scene_name, smolvla_name in CAMERA_SCENE_TO_SMOLVLA.items():
        rgb = env.unwrapped.scene[scene_name].data.output["rgb"][env_id]
        images[smolvla_name] = rgb.detach().clone()
    return images


def reset_distractors(env, env_ids: torch.Tensor, *, variant: str, num_distractors: int):
    """Place distractors around the target object for selected environments."""
    if num_distractors <= 0:
        return
    if env_ids is None or env_ids.numel() == 0:
        return
    device = env.unwrapped.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    origins = env.unwrapped.scene.env_origins[env_ids]
    object_pos = env.unwrapped.scene["object"].data.root_pos_w[env_ids] - origins

    for distractor_idx in range(num_distractors):
        name = f"distractor_{distractor_idx + 1}"
        if name not in env.unwrapped.scene.keys():
            continue
        asset = env.unwrapped.scene[name]
        count = env_ids.numel()
        offsets = torch.zeros((count, 3), device=device)
        if variant == "wrong_object" and distractor_idx == 0:
            offsets[:, 0] = 0.045 + (torch.rand(count, device=device) - 0.5) * 0.014
            offsets[:, 1] = (torch.rand(count, device=device) - 0.5) * 0.032
        elif variant == "wrong_object":
            offsets[:, 0] = (torch.rand(count, device=device) - 0.5) * 0.09
            offsets[:, 1] = (torch.rand(count, device=device) - 0.5) * 0.09
        elif variant == "partial_occlusion":
            offsets[:, 0] = 0.035 + (torch.rand(count, device=device) - 0.5) * 0.015
            offsets[:, 1] = (torch.rand(count, device=device) - 0.5) * 0.035
        elif variant == "clutter":
            offsets[:, 0] = (torch.rand(count, device=device) - 0.5) * 0.12
            offsets[:, 1] = (torch.rand(count, device=device) - 0.5) * 0.12
        else:
            angle = (distractor_idx / max(1, num_distractors)) * 6.2831853
            radius = 0.09 + 0.03 * (distractor_idx % 2)
            offsets[:, 0] = radius * torch.cos(torch.full((count,), angle, device=device))
            offsets[:, 1] = radius * torch.sin(torch.full((count,), angle, device=device))
            offsets[:, :2] += (torch.rand((count, 2), device=device) - 0.5) * 0.04
        offsets[:, 2] = -0.015
        positions = object_pos + offsets + origins
        orientations = torch.zeros((count, 4), device=device)
        orientations[:, 0] = 1.0
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((count, 6), device=device), env_ids=env_ids)


def _add_cameras(env_cfg, *, variant: str, width: int, height: int):
    table_pos = (1.0, 0.0, 0.4)
    table_rot = (0.35355, -0.61237, -0.61237, 0.35355)
    overhead_pos = (1.25, 0.95, 0.95)
    overhead_rot = (-0.1393, 0.2025, 0.8185, -0.5192)
    if variant == "camera_shift":
        table_pos = (0.88, -0.18, 0.46)
        overhead_pos = (1.08, 1.18, 1.05)

    camera_spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 2.0),
    )
    env_cfg.scene.vla_table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/vla_table_cam",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=camera_spawn,
        offset=CameraCfg.OffsetCfg(pos=table_pos, rot=table_rot, convention="ros"),
    )
    env_cfg.scene.vla_wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/vla_wrist_cam",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=camera_spawn,
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        ),
    )
    env_cfg.scene.vla_overhead_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/vla_overhead_cam",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=camera_spawn,
        offset=CameraCfg.OffsetCfg(pos=overhead_pos, rot=overhead_rot, convention="ros"),
    )


def _add_distractors(env_cfg, count: int):
    cube_properties = RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    assets = [
        "blue_block.usd",
        "red_block.usd",
        "green_block.usd",
        "yellow_block.usd",
    ]
    for idx in range(count):
        setattr(
            env_cfg.scene,
            f"distractor_{idx + 1}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Distractor_{idx + 1}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5 + 0.04 * idx, 0.08, 0.04], rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/{assets[idx % len(assets)]}",
                    scale=(0.8, 0.8, 0.8),
                    rigid_props=cube_properties,
                    semantic_tags=[("class", "distractor")],
                ),
            ),
        )
