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
    # The default preset opens the Newton viewer; keep this runtime test headless.
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = True
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        env.reset(seed=42)
        robot = unwrapped.robot
        joint_ids = unwrapped.wuji_joint_ids
        assert torch.allclose(unwrapped.goal_position, torch.tensor([[0.35, -0.05, 0.24]], device=unwrapped.device))
        assert unwrapped.goal_keypoint_marker is not None
        assert unwrapped.object_keypoint_marker is not None
        assert unwrapped.cfg.episode_length_s == 8.0
        assert unwrapped.local_cube_keypoints.shape == (8, 3)
        assert unwrapped.cfg.object_max_horizontal_displacement == 0.20
        assert unwrapped.cfg.success_keypoint_error_threshold == 0.10
        assert unwrapped.cfg.observation_space == 117
        assert unwrapped.cfg.state_space == 117
        observations = unwrapped._get_observations()
        assert set(observations) == {"policy", "critic"}
        assert observations["policy"].shape == (1, 117)
        assert torch.isfinite(observations["policy"]).all()
        assert torch.equal(observations["policy"], observations["critic"])
        assert unwrapped.torso_contact_sensor.data.normal_force_matrix_w.torch.shape == (1, 1, 1, 3)
        terminated, timed_out = unwrapped._get_dones()
        assert terminated.shape == (1,)
        assert timed_out.shape == (1,)
        rewards = unwrapped._get_rewards()
        assert rewards.shape == (1,)
        assert torch.isfinite(rewards).all()
        unwrapped.episode_length_buf.fill_(unwrapped.max_episode_length - 1)
        _, _, _, _, extras = env.step(torch.zeros((1, unwrapped.cfg.action_space), device=unwrapped.device))
        expected_log_keys = {
            "Metrics/step_reach_reward",
            "Metrics/step_goal_reward",
            "Metrics/step_contact_reward",
            "Metrics/reach_return",
            "Metrics/goal_return",
            "Metrics/contact_return",
            "Metrics/min_hand_distance",
            "Metrics/final_keypoint_error",
            "Metrics/min_keypoint_error",
            "Metrics/max_object_height",
            "Metrics/contact_gate_fraction",
            "Metrics/success",
            "Control/arm_tracking_error",
            "Control/wuji_tracking_error",
            "Terminations/timeout",
        }
        assert expected_log_keys <= extras["log"].keys()
        assert extras["log"]["Metrics/success"].item() == 0.0
        assert extras["log"]["Terminations/timeout"].item() == 1.0
        arm_actions = torch.zeros((1, unwrapped.cfg.action_space), device=unwrapped.device)
        arm_actions[:, 0] = 1.0
        arm_default = robot.data.default_joint_pos.torch[:, unwrapped.arm_joint_ids].clone()
        arm_limits = robot.data.soft_joint_pos_limits.torch[:, unwrapped.arm_joint_ids]
        arm_desired = torch.clamp(
            arm_default + unwrapped.arm_action_scale.unsqueeze(0) * arm_actions[:, : len(unwrapped.arm_joint_ids)],
            arm_limits[..., 0],
            arm_limits[..., 1],
        )
        unwrapped._pre_physics_step(arm_actions)
        assert torch.allclose(
            unwrapped.arm_joint_targets,
            0.25 * arm_desired + 0.75 * arm_default,
        )
        wuji_default = robot.data.default_joint_pos.torch[:, joint_ids].clone()
        wuji_limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
        wuji_desired = unwrapped.wuji_action_pipeline.latent_action_to_joint_target(
            arm_actions[:, len(unwrapped.arm_joint_ids) :], wuji_limits[..., 0], wuji_limits[..., 1]
        )
        assert torch.allclose(unwrapped.wuji_joint_targets, 0.1 * wuji_desired + 0.9 * wuji_default)
        env.reset(seed=42)
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
            # Keep the production eight-second timeout while allowing the
            # diagnostic's three four-second legs to retain physical state.
            unwrapped.episode_length_buf.zero_()
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


@pytest.mark.integration
def test_wuji_multi_env_reset_initializes_ema_targets() -> None:
    """Resetting all environments must preserve the per-joint target layout."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    resolve_presets(env_cfg)
    env_cfg.scene.num_envs = 2
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        observations, _ = env.reset(seed=42)
        unwrapped = env.unwrapped
        default_joint_pos = unwrapped.robot.data.default_joint_pos.torch
        assert observations["policy"].shape == (2, 117)
        assert torch.equal(unwrapped.arm_joint_targets, default_joint_pos[:, unwrapped.arm_joint_ids])
        assert torch.equal(unwrapped.wuji_joint_targets, default_joint_pos[:, unwrapped.wuji_joint_ids])
    finally:
        env.close()
