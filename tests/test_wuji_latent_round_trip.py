# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless end-to-end validation for the Wuji latent action contract."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import Cross_Embodiment_CL.tasks  # noqa: E402, F401


@pytest.mark.integration
def test_wuji_latent_round_trip_ping_pong() -> None:
    """Project two simulated poses and verify their latent commands move the hand."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    resolve_presets(env_cfg)
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        env.reset(seed=42)
        robot = unwrapped.robot
        joint_ids = unwrapped.wuji_joint_ids
        limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
        lower_limits, upper_limits = limits[..., 0], limits[..., 1]
        generator = torch.Generator(device=unwrapped.device).manual_seed(42)
        sampled_positions = lower_limits + (upper_limits - lower_limits) * (
            0.2 + 0.6 * torch.rand((2, len(joint_ids)), device=unwrapped.device, generator=generator)
        )

        projections = []
        for sampled_position in sampled_positions:
            sampled_position = sampled_position.unsqueeze(0)
            # Author a valid simulated pose, then use its read-back value as
            # the source for the Wuji-to-MANO manifold projection.
            robot.write_joint_position_to_sim_index(position=sampled_position, joint_ids=joint_ids)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(sampled_position), joint_ids=joint_ids)
            unwrapped.sim.step(render=False)
            source_position = robot.data.joint_pos.torch[:, joint_ids].clone()
            projections.append(
                unwrapped.wuji_action_pipeline.project_joint_positions(
                    source_position, lower_limits, upper_limits, steps=512
                )
            )

        policy_actions = torch.zeros((1, unwrapped.cfg.action_space), device=unwrapped.device)
        settled_errors = []
        for projection in (projections[1], projections[0], projections[1]):
            policy_actions[:, -projection.latent_action.shape[1] :] = projection.latent_action
            initial_error = torch.linalg.vector_norm(
                robot.data.joint_pos.torch[:, joint_ids] - projection.joint_target, dim=-1
            )
            for _ in range(240):
                env.step(policy_actions)
            settled_error = torch.linalg.vector_norm(
                robot.data.joint_pos.torch[:, joint_ids] - projection.joint_target, dim=-1
            )
            settled_errors.append(settled_error)
            assert torch.isfinite(settled_error).all()
            assert torch.all(settled_error < initial_error)

        projection_error = torch.stack([projection.error for projection in projections])
        assert torch.isfinite(projection_error).all()
        print(
            "Wuji latent round trip [rad]: "
            f"projection_l2={projection_error.flatten().cpu().tolist()}, "
            f"settled_tracking_l2={torch.cat(settled_errors).cpu().tolist()}"
        )
    finally:
        env.close()
