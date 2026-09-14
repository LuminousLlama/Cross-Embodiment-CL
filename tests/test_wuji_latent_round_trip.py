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

from isaaclab.utils.math import unscale_transform  # noqa: E402

from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import Cross_Embodiment_CL.tasks  # noqa: E402, F401


@pytest.mark.integration
@pytest.mark.parametrize("physics_preset", ["newton_mjwarp", "isaacsim_physx"])
def test_observation_uses_raw_joint_positions_and_command_limits(physics_preset: str) -> None:
    """Expose raw joint angles and the physical arm/effective Wuji command limits."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    resolve_presets(env_cfg, selected=(physics_preset,))
    env_cfg.scene.num_envs = 4
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = False
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        env.reset(seed=42)
        unwrapped = env.unwrapped
        robot = unwrapped.robot
        observations = unwrapped._get_observations()

        assert observations["policy"].shape == (4, 171)
        assert observations["student"].shape == (4, 141)
        assert torch.equal(observations["policy"][:, :141], observations["student"])
        assert torch.equal(observations["student"][:, : robot.num_joints], robot.data.joint_pos.torch)

        shoulder_roll_joint_id = unwrapped.arm_joint_ids[1]
        expected_shoulder_roll = torch.full_like(
            robot.data.default_joint_pos.torch[:, shoulder_roll_joint_id], -torch.pi / 4
        )
        assert torch.allclose(robot.data.default_joint_pos.torch[:, shoulder_roll_joint_id], expected_shoulder_roll)
        assert torch.allclose(robot.data.joint_pos.torch[:, shoulder_roll_joint_id], expected_shoulder_roll)

        arm_limits = robot.data.soft_joint_pos_limits.torch[:, unwrapped.arm_joint_ids]
        wuji_asset_limits = robot.data.soft_joint_pos_limits.torch[:, unwrapped.wuji_joint_ids]
        wuji_command_limits = torch.stack(
            (
                torch.maximum(wuji_asset_limits[..., 0], unwrapped._wuji_command_lower_floor),
                wuji_asset_limits[..., 1],
            ),
            dim=-1,
        )
        expected_command_limits = torch.cat(
            (arm_limits.flatten(start_dim=1), wuji_command_limits.flatten(start_dim=1)), dim=-1
        )
        observed_command_limits = observations["student"][:, -expected_command_limits.shape[-1] :]
        assert torch.equal(observed_command_limits, expected_command_limits)

        zero_floor = unwrapped._wuji_command_lower_floor == 0.0
        assert (wuji_asset_limits[..., 0][:, zero_floor] < 0.0).any()
        assert torch.all(wuji_command_limits[..., 0][:, zero_floor] >= 0.0)
    finally:
        env.close()


@pytest.mark.integration
def test_nonfinite_state_returns_zero_terminal_reward() -> None:
    """A state rejected by the done guard must not leak a NaN reward to the trainer."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    resolve_presets(env_cfg)
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = False
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        env.reset(seed=42)
        unwrapped = env.unwrapped
        unwrapped.apple.data.root_pos_w.torch[0, 0] = torch.nan

        terminated, _ = unwrapped._get_dones()
        reward = unwrapped._get_rewards()

        assert terminated.item()
        assert unwrapped._termination_nonfinite.item()
        assert torch.equal(reward, torch.zeros_like(reward))
    finally:
        env.close()


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
        assert unwrapped.goal_keypoint_marker is not None
        assert unwrapped.object_keypoint_marker is not None
        assert unwrapped.adr_spawn_area_marker is not None
        assert unwrapped.local_cube_keypoints.shape == (8, 3)
        observations = unwrapped._get_observations()
        assert set(observations) == {"policy", "critic", "student"}
        assert observations["policy"].shape == (1, 171)
        assert observations["student"].shape == (1, 141)
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
            "Reward/reach_step",
            "Reward/goal_step",
            "Reward/contact_step",
            "Reward/reach_ep_return",
            "Reward/goal_ep_return",
            "Reward/contact_ep_return",
            "Reach/hand_distance_farthest_ep_min",
            "Task/keypoint_error_ep_final",
            "Task/keypoint_error_ep_min",
            "Task/object_height_ep_max",
            "Contact/gate_frac_ep",
            "Task/success",
            "Control/arm_tracking_error_ep",
            "Control/wuji_tracking_error_ep",
            "Control/arm_target_rate_step",
            "Control/arm_joint_velocity_step",
            "Control/arm_computed_effort_step",
            "Control/arm_applied_effort_step",
            "Control/arm_effort_saturation_frac_step",
            "Control/arm_anti_windup_frac_step",
            "Contact/penetration_elbow_torso_step",
            "Contact/elbow_torso_penetrating_frac_step",
            "Contact/penetration_elbow_torso_ep_max",
            "Terminations/timeout",
        }
        for joint_name in unwrapped._ARM_JOINT_NAMES:
            expected_log_keys.update(
                {
                    f"Control/arm_target_rate_{joint_name}_step",
                    f"Control/arm_tracking_error_{joint_name}_step",
                    f"Control/arm_joint_velocity_{joint_name}_step",
                    f"Control/arm_computed_effort_{joint_name}_step",
                    f"Control/arm_applied_effort_{joint_name}_step",
                    f"Control/arm_effort_saturation_{joint_name}_frac_step",
                    f"Control/arm_anti_windup_{joint_name}_frac_step",
                }
            )
        assert expected_log_keys <= extras["log"].keys()
        assert extras["log"]["Task/success"].item() == 0.0
        assert extras["log"]["Terminations/timeout"].item() == 1.0
        arm_actions = torch.zeros((1, unwrapped.cfg.action_space), device=unwrapped.device)
        arm_actions[:, 0] = 1.0
        arm_actions[:, 1] = -1.0
        arm_default = robot.data.default_joint_pos.torch[:, unwrapped.arm_joint_ids].clone()
        arm_limits = robot.data.soft_joint_pos_limits.torch[:, unwrapped.arm_joint_ids]
        arm_desired = unscale_transform(
            arm_actions[:, : len(unwrapped.arm_joint_ids)], arm_limits[..., 0], arm_limits[..., 1]
        )
        assert torch.equal(arm_desired[:, 0], arm_limits[:, 0, 1])
        assert torch.equal(arm_desired[:, 1], arm_limits[:, 1, 0])
        max_step = unwrapped.cfg.arm_joint_velocity_limit * unwrapped.step_dt
        expected_arm_target = arm_default + torch.clamp(
            unwrapped.cfg.arm_action_ema_alpha * (arm_desired - arm_default), -max_step, max_step
        )
        max_lead = (
            robot.data.joint_effort_limits.torch[:, unwrapped.arm_joint_ids]
            + robot.data.joint_damping.torch[:, unwrapped.arm_joint_ids] * unwrapped.cfg.arm_joint_velocity_limit
        ) / robot.data.joint_stiffness.torch[:, unwrapped.arm_joint_ids].clamp_min(1.0e-6)
        arm_position = robot.data.joint_pos.torch[:, unwrapped.arm_joint_ids]
        expected_arm_target.clamp_(min=arm_position - max_lead, max=arm_position + max_lead)
        unwrapped._pre_physics_step(arm_actions)
        assert torch.allclose(unwrapped.arm_joint_targets, expected_arm_target)
        wuji_default = robot.data.default_joint_pos.torch[:, joint_ids].clone()
        wuji_limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
        wuji_desired = unwrapped.wuji_action_pipeline.latent_action_to_joint_target(
            arm_actions[:, len(unwrapped.arm_joint_ids) :],
            torch.maximum(wuji_limits[..., 0], unwrapped._wuji_command_lower_floor),
            wuji_limits[..., 1],
        )
        wuji_max_step = unwrapped.cfg.hand_joint_velocity_limit * unwrapped.step_dt
        expected_wuji_target = wuji_default + torch.clamp(
            unwrapped.cfg.wuji_action_ema_alpha * (wuji_desired - wuji_default),
            -wuji_max_step,
            wuji_max_step,
        )
        wuji_max_lead = (
            robot.data.joint_effort_limits.torch[:, joint_ids]
            + robot.data.joint_damping.torch[:, joint_ids] * unwrapped.cfg.hand_joint_velocity_limit
        ) / robot.data.joint_stiffness.torch[:, joint_ids].clamp_min(1.0e-6)
        wuji_position = robot.data.joint_pos.torch[:, joint_ids]
        expected_wuji_target.clamp_(min=wuji_position - wuji_max_lead, max=wuji_position + wuji_max_lead)
        expected_wuji_target.clamp_(min=torch.maximum(wuji_limits[..., 0], unwrapped._wuji_command_lower_floor))
        assert torch.allclose(unwrapped.wuji_joint_targets, expected_wuji_target)
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
        assert observations["policy"].shape == (2, 171)
        assert torch.equal(unwrapped.arm_joint_targets, default_joint_pos[:, unwrapped.arm_joint_ids])
        assert torch.equal(unwrapped.wuji_joint_targets, default_joint_pos[:, unwrapped.wuji_joint_ids])
    finally:
        env.close()
