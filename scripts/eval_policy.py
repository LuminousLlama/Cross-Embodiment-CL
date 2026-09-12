# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless, deterministic evaluation of an RSL-RL checkpoint.

Based on IsaacLab's ``isaaclab_rl/entrypoints/backends/play_rsl_rl.py``: keeps its CLI and
preset parsing, environment creation, runner construction, ``runner.load``, and
``runner.get_inference_policy`` (the deterministic mean action), but drops JIT/ONNX export,
video, real-time sleeping, and the infinite loop in favor of a fixed number of episodes.

Every scalar in ``extras["log"]`` (see ``G1WujiTableEnv._update_episode_metrics``) is
aggregated as a weighted mean, weighted by the number of environments that finished on that
step; since the env only adds its ``*_ep*`` tags on steps where ``reset_ids`` is non-empty,
and those are already averaged over just the finished environments, this weighting turns the
per-step averages into a single per-episode mean across the whole run. The always-present
``*_step`` tags are aggregated the same way, over only the steps that had a completed episode.

Usage:
    uv run python scripts/eval_policy.py --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
        --checkpoint <model.pt> presets=eval [--seed S] [--episodes N]
"""

import argparse
import importlib.metadata as metadata
import json
import os
import subprocess
import sys

import torch
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from PIL import Image, ImageDraw
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_frontend_args,
    create_isaaclab_env,
    request_determinism,
    resolve_checkpoint_selector,
    resolve_play_task_name,
    show_run_summary,
    startup_screen,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config


def _parse_xyz(value: str) -> tuple[float, float, float]:
    """Parse a comma-separated ``"x,y,z"`` CLI argument into a float tuple."""
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'x,y,z', got {value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _log_scalar(log: dict, tag: str) -> float | None:
    """Extract a scalar float from an ``extras["log"]`` entry, or ``None`` if the tag is absent."""
    if tag not in log:
        return None
    value = log[tag]
    return value.item() if torch.is_tensor(value) else float(value)


def _write_contact_sheet(frames: list[tuple[int, object, float | None, float | None]], out_path: str) -> None:
    """Tile captured rollout frames into one labeled contact sheet, 4 columns wide.

    Each tile is downscaled to 480x270 and labeled with its policy step, apple height, and
    (when available) the max per-group contact force, so a reviewer can judge grasp quality
    from a single image.
    """
    tile_width, tile_height, columns = 480, 270, 4
    rows = -(-len(frames) // columns)
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (step, frame, height_cm, force_max) in enumerate(frames):
        column, row = index % columns, index // columns
        x, y = column * tile_width, row * tile_height
        sheet.paste(Image.fromarray(frame).resize((tile_width, tile_height)), (x, y))
        label = f"step {step}"
        if height_cm is not None:
            label += f"  h={height_cm:.1f}cm"
        if force_max is not None:
            label += f"  f={force_max:.2f}N"
        draw.rectangle((x, y, x + tile_width, y + 14), fill=(0, 0, 0))
        draw.text((x + 2, y + 1), label, fill=(255, 255, 0))
    sheet.save(out_path)


# Import task packages registered by downstream projects (e.g. Cross_Embodiment_CL), which the
# `isaaclab play`/`isaaclab train` CLI dispatcher normally loads before running an entrypoint
# script; this script is invoked directly, so it has to do the same registration itself.
for _entry_point in metadata.entry_points(group="isaaclab.tasks"):
    _entry_point.load()

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from RSL-RL, headless.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--train_env_cfg",
    action="store_true",
    default=False,
    help="Play with the training environment configuration as-is, skipping play-mode overrides.",
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
parser.add_argument("--episodes", type=int, default=64, help="Number of completed episodes to collect.")
parser.add_argument(
    "--out", type=str, default=None, help="Output JSON path. Defaults to <checkpoint dir>/eval_<checkpoint stem>.json."
)
parser.add_argument(
    "--frames_dir",
    type=str,
    default=None,
    help="Directory to save headless PNG frames of the deterministic rollout plus a tiled contact sheet. "
    "Disabled by default; when set, forces 1 environment, Newton visual shapes, and keypoint markers.",
)
parser.add_argument("--frame_every", type=int, default=15, help="Save a frame every N policy steps.")
parser.add_argument("--max_frames", type=int, default=16, help="Maximum number of frames to save.")
parser.add_argument(
    "--cam_eye",
    type=_parse_xyz,
    default=_parse_xyz("0.85,-0.55,0.50"),
    help="Headless capture camera eye position, as 'x,y,z' in the env frame.",
)
parser.add_argument(
    "--cam_lookat",
    type=_parse_xyz,
    default=_parse_xyz("0.35,-0.05,0.12"),
    help="Headless capture camera look-at target, as 'x,y,z' in the env frame.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
args_cli.task = resolve_play_task_name(args_cli.task)

# an external callback lets downstream code register its environments; it returns
# the arguments it did not consume
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# hand the arguments consumed by neither this parser nor the callback over to Hydra
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate an RSL-RL agent for a fixed number of episodes."""
    with startup_screen(args_cli, num_stages=3) as screen:
        show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            task_name = args_cli.task.split(":")[-1]
            train_task_name = task_name.replace("-Play", "")

            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
            if args_cli.frames_dir:
                # Frame capture needs a single, visible env: the apple's render mesh (Newton
                # visual shapes) and the goal/current keypoint markers, drawn by a headless
                # Newton GL visualizer at the requested camera pose.
                env_cfg.scene.num_envs = 1
                env_cfg.sim.physics.load_visual_shapes = True
                env_cfg.debug.keypoint_markers = True
                env_cfg.sim.visualizer_cfgs = [
                    NewtonGLVisualizerCfg(
                        headless=True,
                        window_width=960,
                        window_height=540,
                        eye=args_cli.cam_eye,
                        lookat=args_cli.cam_lookat,
                    )
                ]
            # Warp reads its determinism mode at module build time, so request it before the env exists.
            request_determinism(args_cli, env_cfg)

            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

            # note: certain randomizations occur in the environment initialization so we set the seed here
            env_cfg.seed = agent_cfg.seed
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            if args_cli.checkpoint in CHECKPOINT_SELECTORS:
                resume_path = resolve_checkpoint_selector(
                    log_root_path,
                    args_cli.checkpoint,
                    library="rsl_rl",
                    task=train_task_name,
                    checkpoint_pattern=r"model_.*\.pt",
                    metadata={"agent": args_cli.agent},
                )
            elif args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
                resume_path = get_checkpoint_path(
                    os.path.dirname(args_cli.checkpoint),
                    os.path.basename(args_cli.checkpoint),
                    agent_cfg.load_checkpoint,
                )
            elif args_cli.checkpoint:
                resume_path = retrieve_file_path(args_cli.checkpoint)
            else:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

            log_dir = os.path.dirname(resume_path)
            env_cfg.log_dir = log_dir

            # A fresh process restarts the curriculum from its easy end: force full apple weight
            # for evaluation regardless of the training-default curriculum start.
            if hasattr(env_cfg, "apple_weight_curriculum_start"):
                env_cfg.apple_weight_curriculum_start = 1.0
                print("[INFO] Forcing apple_weight_curriculum_start=1.0 for evaluation.")

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )

            screen.stage("Loading policy")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(resume_path)

            # The inference policy uses the deterministic mean action, not a sampled one.
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            screen.close()
            obs = env.get_observations()

            metric_sums: dict[str, float] = {}
            metric_weights: dict[str, float] = {}
            completed_episodes = 0
            policy_step = 0
            frames: list[tuple[int, object, float | None, float | None]] = []
            if args_cli.frames_dir:
                os.makedirs(args_cli.frames_dir, exist_ok=True)
            print(f"[INFO] Collecting {args_cli.episodes} episodes...")
            while completed_episodes < args_cli.episodes:
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, extras = env.step(actions)
                    policy.reset(dones)
                policy_step += 1
                log = extras.get("log", {})
                num_done = int(dones.sum().item())
                if num_done > 0:
                    for tag, value in log.items():
                        scalar = value.item() if torch.is_tensor(value) else float(value)
                        metric_sums[tag] = metric_sums.get(tag, 0.0) + scalar * num_done
                        metric_weights[tag] = metric_weights.get(tag, 0.0) + num_done
                    completed_episodes += num_done
                capture_frame = (
                    args_cli.frames_dir
                    and len(frames) < args_cli.max_frames
                    and policy_step % args_cli.frame_every == 0
                )
                if capture_frame:
                    frame = env.unwrapped.sim.visualizers[0].render_rgb_array()
                    height_cm = _log_scalar(log, "Task/object_height_step")
                    force_max = _log_scalar(log, "Contact/force_max_step")
                    Image.fromarray(frame).save(os.path.join(args_cli.frames_dir, f"frame_{policy_step}.png"))
                    frames.append((policy_step, frame, None if height_cm is None else height_cm * 100.0, force_max))

            env.close()

            if args_cli.frames_dir and frames:
                _write_contact_sheet(frames, os.path.join(args_cli.frames_dir, "contact_sheet.png"))

    metrics = {tag: metric_sums[tag] / metric_weights[tag] for tag in metric_sums}

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    out_path = args_cli.out or os.path.join(log_dir, f"eval_{os.path.splitext(os.path.basename(resume_path))[0]}.json")
    result = {
        "checkpoint": resume_path,
        "commit": commit,
        "num_envs": env_cfg.scene.num_envs,
        "episodes": completed_episodes,
        "seed": env_cfg.seed,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(f"[INFO] Wrote {out_path}")

    print(f"{'episodes':40s} {completed_episodes}")
    remaining_tags = sorted(tag for tag in metrics if tag != "Task/success")
    if "Task/success" in metrics:
        print(f"{'Task/success':40s} {metrics['Task/success']:.4f}")
    for tag in remaining_tags:
        print(f"{tag:40s} {metrics[tag]:.4f}")


if __name__ == "__main__":
    main()
