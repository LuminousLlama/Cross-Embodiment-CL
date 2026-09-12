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
            print(f"[INFO] Collecting {args_cli.episodes} episodes...")
            while completed_episodes < args_cli.episodes:
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, extras = env.step(actions)
                    policy.reset(dones)
                num_done = int(dones.sum().item())
                if num_done > 0:
                    for tag, value in extras.get("log", {}).items():
                        scalar = value.item() if torch.is_tensor(value) else float(value)
                        metric_sums[tag] = metric_sums.get(tag, 0.0) + scalar * num_done
                        metric_weights[tag] = metric_weights.get(tag, 0.0) + num_done
                    completed_episodes += num_done

            env.close()

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
