# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless check of the student depth camera: intrinsics, geometry, and framing over a rollout.

Launches one environment with the depth camera (``presets=distill``), optionally driven by a PPO
teacher checkpoint (zero actions otherwise), and at selected policy steps saves the depth image with
known scene points projected onto it.  The JSON summary records:

- the camera's read-back intrinsic matrix against the one derived from the D435 crop;
- rendered depth at projected table-top points against their camera-frame depth.  A wrong field of
  view, mount pose, frame convention, or ray-versus-planar depth each breaks this agreement;
- the apple centre's projected pixel and rendered depth, to confirm the apple stays in frame.

Usage:
    python scripts/check_depth_camera.py --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
        --checkpoint <teacher model.pt> --out_dir <dir> presets=distill
"""

import argparse
import importlib.metadata as metadata
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

# The `isaaclab` CLI registers downstream task packages before running a script; this runs directly.
for _entry_point in metadata.entry_points(group="isaaclab.tasks"):
    _entry_point.load()

parser = argparse.ArgumentParser(description="Check the student depth camera headlessly.")
parser.add_argument("--task", type=str, default="CrossEmbodimentCl-G1-Wuji-Table-Direct", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Teacher agent config entry point.")
parser.add_argument("--out_dir", type=str, required=True, help="Directory for images and the JSON summary.")
parser.add_argument(
    "--frame_steps",
    type=str,
    default="0,30,60,90,120,180,240,300,400,470",
    help="Comma-separated policy steps at which to capture the depth image.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args

_OCCLUSION_MARGIN_M = 0.02
"""A table point whose rendered depth is this much nearer than expected is hidden behind something."""
_TILE_SCALE = 2


def _table_top_points(env) -> torch.Tensor:
    """World positions [m] of a grid over the table top, shape ``(P, 3)``."""
    table_x, table_y = env.cfg.table_cfg.init_state.pos[:2]
    size_x, size_y = env.cfg.table_cfg.spawn.size[:2]
    xs = torch.linspace(table_x - 0.45 * size_x, table_x + 0.45 * size_x, 9, device=env.device)
    ys = torch.linspace(table_y - 0.45 * size_y, table_y + 0.45 * size_y, 9, device=env.device)
    grid = torch.cartesian_prod(xs, ys)
    heights = torch.full((grid.shape[0], 1), env.table_top_height, device=env.device)
    return torch.cat((grid, heights), dim=-1) + env.scene.env_origins[0]


def _project(camera, points_w: torch.Tensor) -> torch.Tensor:
    """Project world points into the camera image: returns ``(P, 3)`` of pixel u, pixel v, and camera z [m].

    Projects with the intrinsic matrix directly: ``isaaclab.utils.math.project_points`` returns
    ``(1, P, 3)`` for a ``(P, 3)`` input, contrary to its docstring.
    """
    data = camera.data
    position, orientation = data.pos_w.torch[0], data.quat_w_ros.torch[0]
    points_camera = quat_apply_inverse(orientation.expand(points_w.shape[0], 4), points_w - position)
    pixels = points_camera @ data.intrinsic_matrices.torch[0].T
    return torch.cat((pixels[:, :2] / pixels[:, 2:3], points_camera[:, 2:3]), dim=-1)


def _pixels(uvz: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    columns, rows = uvz[:, 0].floor().long(), uvz[:, 1].floor().long()
    in_view = (uvz[:, 2] > 0.0) & (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    return rows.clamp(0, height - 1), columns.clamp(0, width - 1), in_view


def _inspect_frame(env, step: int, max_depth_m: float) -> tuple[dict, Image.Image, np.ndarray]:
    camera = env.depth_camera
    depth = camera.data.output["distance_to_image_plane"].torch[0, :, :, 0]
    height, width = depth.shape

    table_uvz = _project(camera, _table_top_points(env))
    rows, columns, in_view = _pixels(table_uvz, height, width)
    rendered = depth[rows, columns]
    unoccluded = in_view & (rendered > table_uvz[:, 2] - _OCCLUSION_MARGIN_M)
    table_error = (rendered - table_uvz[:, 2])[unoccluded].abs()

    apple_uvz = _project(camera, env.apple.data.root_pos_w.torch[:1])
    apple_row, apple_column, apple_in_view = _pixels(apple_uvz, height, width)
    goal_uvz = _project(camera, (env.goal_position[0] + env.scene.env_origins[0])[None])

    record = {
        "step": step,
        "apple_height_m": float(env.apple.data.root_pos_w.torch[0, 2] - env.scene.env_origins[0, 2]),
        "apple_pixel_uv": [float(apple_uvz[0, 0]), float(apple_uvz[0, 1])],
        "apple_in_view": bool(apple_in_view[0]),
        "apple_center_camera_z_m": float(apple_uvz[0, 2]),
        # Negative by about the apple's radius when the apple's near surface is what the pixel sees.
        "apple_rendered_minus_center_z_m": float(depth[apple_row[0], apple_column[0]] - apple_uvz[0, 2]),
        "goal_pixel_uv": [float(goal_uvz[0, 0]), float(goal_uvz[0, 1])],
        "table_points_in_view": int(in_view.sum()),
        "table_points_unoccluded": int(unoccluded.sum()),
        "table_depth_abs_error_median_m": float(table_error.median()) if table_error.numel() else None,
        "table_depth_abs_error_max_m": float(table_error.max()) if table_error.numel() else None,
        "pixels_missing_frac": float((depth <= 0.0).float().mean()),
        "pixels_nearer_than_10cm_frac": float(((depth > 0.0) & (depth < 0.10)).float().mean()),
        "depth_min_valid_m": float(depth[depth > 0.0].min()) if (depth > 0.0).any() else None,
        "depth_max_m": float(depth.max()),
    }

    image = (depth.clamp(0.0, max_depth_m) / max_depth_m * 255.0).byte().cpu().numpy()
    tile = Image.fromarray(image, mode="L").convert("RGB").resize(
        (width * _TILE_SCALE, height * _TILE_SCALE), Image.NEAREST
    )
    draw = ImageDraw.Draw(tile)
    for index in range(table_uvz.shape[0]):
        if not in_view[index]:
            continue
        u, v = (table_uvz[index, :2] * _TILE_SCALE).tolist()
        color = (0, 220, 0) if unoccluded[index] else (255, 150, 0)
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), outline=color)
    for uvz, color in ((apple_uvz[0], (255, 0, 0)), (goal_uvz[0], (60, 120, 255))):
        u, v = (uvz[:2] * _TILE_SCALE).tolist()
        draw.line((u - 5, v, u + 5, v), fill=color)
        draw.line((u, v - 5, u, v + 5), fill=color)
    draw.rectangle((0, 0, tile.width, 12), fill=(0, 0, 0))
    draw.text((2, 0), f"step {step}  apple h={record['apple_height_m'] * 100:.1f}cm", fill=(255, 255, 0))
    return record, tile, depth.cpu().numpy()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Roll out one environment and inspect the depth camera at the requested steps."""
    if getattr(env_cfg, "depth_camera", None) is None:
        raise ValueError("The environment has no depth camera; select one with presets=distill.")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    frame_steps = sorted({int(step) for step in args_cli.frame_steps.split(",")})
    os.makedirs(args_cli.out_dir, exist_ok=True)

    from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env_cfg import STUDENT_DEPTH_CROP

    with launch_simulation(env_cfg, args_cli):
        env = RslRlVecEnvWrapper(
            create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        )
        base_env = env.unwrapped
        policy = None
        if args_cli.checkpoint:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(retrieve_file_path(args_cli.checkpoint))
            policy = runner.get_inference_policy(device=base_env.device)

        expected_intrinsics = torch.tensor(STUDENT_DEPTH_CROP.output.matrix(), device=base_env.device).reshape(3, 3)
        obs = env.get_observations()
        records, tiles = [], []
        for step in range(frame_steps[-1] + 1):
            if step in frame_steps:
                record, tile, depth = _inspect_frame(base_env, step, env_cfg.student_depth_max_m)
                camera_obs = obs["camera"]
                record["camera_obs_shape"] = list(camera_obs.shape)
                record["camera_obs_range"] = [float(camera_obs.min()), float(camera_obs.max())]
                records.append(record)
                tiles.append(tile)
                tile.save(os.path.join(args_cli.out_dir, f"depth_step_{step:03d}.png"))
                np.save(os.path.join(args_cli.out_dir, f"depth_step_{step:03d}.npy"), depth)
                print(json.dumps(record))
            with torch.inference_mode():
                if policy is None:
                    actions = torch.zeros((env.num_envs, env.num_actions), device=base_env.device)
                else:
                    actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if policy is not None:
                    policy.reset(dones)

        camera_data = base_env.depth_camera.data
        read_back_intrinsics = camera_data.intrinsic_matrices.torch[0]
        summary = {
            "checkpoint": args_cli.checkpoint,
            "intrinsics_expected": expected_intrinsics.tolist(),
            "intrinsics_read_back": read_back_intrinsics.tolist(),
            "intrinsics_max_abs_error": float((read_back_intrinsics - expected_intrinsics).abs().max()),
            "camera_pos_env_m": (camera_data.pos_w.torch[0] - base_env.scene.env_origins[0]).tolist(),
            "camera_quat_w_ros_xyzw": camera_data.quat_w_ros.torch[0].tolist(),
            "frames": records,
        }
        env.close()

    columns = 4
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (tiles[0].width * columns, tiles[0].height * rows))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    sheet.save(os.path.join(args_cli.out_dir, "contact_sheet.png"))
    with open(os.path.join(args_cli.out_dir, "summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2)
    print(f"[INFO] Wrote {args_cli.out_dir}/summary.json and contact_sheet.png")


if __name__ == "__main__":
    main()
