# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal DirectRLEnv used solely to inspect the G1-Wuji table scene."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul, scale_transform

from Cross_Embodiment_CL.models import WujiLatentActionPipeline

from .env_cfg import G1WujiTableEnvCfg


class G1WujiTableEnv(DirectRLEnv):
    """G1-Wuji scene with normalized arm and latent-hand position actions."""

    cfg: G1WujiTableEnvCfg

    _ARM_JOINT_NAMES = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    _WUJI_JOINT_NAMES = tuple(f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5))
    _HAND_POINT_BODY_NAMES = ("right_palm_link",) + tuple(f"right_finger{finger}_tip_link" for finger in range(1, 6))
    _THUMB_CONTACT_BODY_NAME = "right_finger1_tip_link"
    _OBSERVATION_DIM = 117

    def __init__(self, cfg: G1WujiTableEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self.arm_joint_ids, _ = self.robot.find_joints(self._ARM_JOINT_NAMES, preserve_order=True)
        self.wuji_joint_ids, _ = self.robot.find_joints(self._WUJI_JOINT_NAMES, preserve_order=True)
        self.waist_joint_ids, _ = self.robot.find_joints("waist_.*_joint")
        self.hand_point_body_ids, hand_point_names = self.robot.find_bodies(
            self._HAND_POINT_BODY_NAMES, preserve_order=True
        )
        if tuple(hand_point_names) != self._HAND_POINT_BODY_NAMES:
            raise ValueError(
                f"Expected hand point bodies {self._HAND_POINT_BODY_NAMES}, found {tuple(hand_point_names)}."
            )
        if self.cfg.action_space != len(self.arm_joint_ids) + WujiLatentActionPipeline.latent_dim:
            raise ValueError(
                f"{type(self.cfg).__name__} declares action_space={self.cfg.action_space}, but the configured "
                "right arm plus Wuji latent action require "
                f"{len(self.arm_joint_ids) + WujiLatentActionPipeline.latent_dim}."
            )
        if self.cfg.observation_space != self._OBSERVATION_DIM or self.cfg.state_space != self._OBSERVATION_DIM:
            raise ValueError(
                f"{type(self.cfg).__name__} must declare matching policy and critic observation spaces of "
                f"{self._OBSERVATION_DIM}."
            )
        self.wuji_action_pipeline = WujiLatentActionPipeline(self.device)
        if not 0.0 < self.cfg.arm_action_ema_alpha <= 1.0:
            raise ValueError("arm_action_ema_alpha must be in (0, 1].")
        if not 0.0 < self.cfg.wuji_action_ema_alpha <= 1.0:
            raise ValueError("wuji_action_ema_alpha must be in (0, 1].")
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.arm_joint_targets = torch.zeros((self.num_envs, len(self.arm_joint_ids)), device=self.device)
        self.wuji_joint_targets = torch.zeros((self.num_envs, len(self.wuji_joint_ids)), device=self.device)
        self.arm_action_scale = torch.full((len(self.arm_joint_ids),), 0.5, device=self.device)
        self.waist_joint_targets = torch.zeros((self.num_envs, len(self.waist_joint_ids)), device=self.device)
        self.goal_position = torch.tensor(self.cfg.goal_position, device=self.device).repeat(self.num_envs, 1)
        self.goal_rotation = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device).repeat(self.num_envs, 1)
        self.object_start_position = torch.tensor(self.cfg.apple_cfg.init_state.pos, device=self.device).repeat(
            self.num_envs, 1
        )
        self.table_top_height = self.cfg.table_cfg.init_state.pos[2] + 0.5 * self.cfg.table_cfg.spawn.size[2]
        self.local_cube_keypoints = self._make_cube_keypoints(self.cfg.keypoint_extent)
        self._init_episode_metrics()
        self.goal_keypoint_marker: VisualizationMarkers | None = None
        self.object_keypoint_marker: VisualizationMarkers | None = None
        if self.cfg.debug_vis:
            self.goal_keypoint_marker = VisualizationMarkers(self.cfg.goal_keypoint_marker_cfg)
            self.object_keypoint_marker = VisualizationMarkers(self.cfg.object_keypoint_marker_cfg)
            self._update_keypoint_markers()

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.apple = RigidObject(self.cfg.apple_cfg)
        self.contact_sensors: dict[str, ContactSensor] = {}
        for body_name in self._HAND_POINT_BODY_NAMES:
            sensor_cfg = self.cfg.contact_sensor_cfg.replace(
                prim_path=f"/World/envs/env_[^/]+/G1Wuji/wujihand/{body_name}"
            )
            sensor = ContactSensor(sensor_cfg)
            self.contact_sensors[body_name] = sensor
        self.torso_contact_sensor = ContactSensor(self.cfg.torso_contact_sensor_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.0))
        source, destination = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing)[0]
        plan = cloner.clone_plan_from_env_0(
            source, destination, self.scene.num_envs, positions, global_paths=("/World/ground",)
        )
        cloner.replicate(plan)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["apple"] = self.apple
        self.scene.sensors.update(self.contact_sensors)
        self.scene.sensors["torso_contact"] = self.torso_contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Map arm deltas and Wuji latent actions onto physical joint targets."""
        self.actions[:] = torch.clamp(actions, -1.0, 1.0)
        arm_limits = self.robot.data.soft_joint_pos_limits.torch[:, self.arm_joint_ids]
        arm_lower, arm_upper = arm_limits[..., 0], arm_limits[..., 1]
        arm_actions = self.actions[:, : len(self.arm_joint_ids)]
        arm_targets = (
            self.robot.data.default_joint_pos.torch[:, self.arm_joint_ids]
            + self.arm_action_scale.unsqueeze(0) * arm_actions
        )
        arm_targets = torch.clamp(arm_targets, min=arm_lower, max=arm_upper)
        self.arm_joint_targets.lerp_(arm_targets, self.cfg.arm_action_ema_alpha)

        wuji_limits = self.robot.data.soft_joint_pos_limits.torch[:, self.wuji_joint_ids]
        wuji_targets = self.wuji_action_pipeline.latent_action_to_joint_target(
            self.actions[:, len(self.arm_joint_ids) :], wuji_limits[..., 0], wuji_limits[..., 1]
        )
        self.wuji_joint_targets.lerp_(wuji_targets, self.cfg.wuji_action_ema_alpha)

    def _apply_action(self) -> None:
        """Apply arm and Wuji targets while holding all waist joints at zero."""
        self.robot.actuators.target_command.set_position_index(
            value=self.arm_joint_targets,
            joint_ids=self.arm_joint_ids,
        )
        self.robot.actuators.target_command.set_position_index(
            value=self.wuji_joint_targets,
            joint_ids=self.wuji_joint_ids,
        )
        self.robot.actuators.target_command.set_position_index(
            value=self.waist_joint_targets,
            joint_ids=self.waist_joint_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return the identical 117-D privileged state for policy and critic.

        The observation contains normalized full-robot state and commanded arm/hand targets,
        followed by apple and goal state in the fixed robot-base frame and filtered contact forces.
        """
        joint_limits = self.robot.data.soft_joint_pos_limits.torch
        joint_position = torch.clamp(
            scale_transform(
                self.robot.data.joint_pos.torch,
                joint_limits[..., 0],
                joint_limits[..., 1],
            ),
            -1.0,
            1.0,
        )
        joint_velocity_limits = torch.clamp_min(self.robot.data.soft_joint_vel_limits.torch, 1.0e-6)
        joint_velocity = torch.clamp(self.robot.data.joint_vel.torch / joint_velocity_limits, -1.0, 1.0)

        arm_limits = joint_limits[:, self.arm_joint_ids]
        wuji_limits = joint_limits[:, self.wuji_joint_ids]
        arm_target = torch.clamp(
            scale_transform(self.arm_joint_targets, arm_limits[..., 0], arm_limits[..., 1]), -1.0, 1.0
        )
        wuji_target = torch.clamp(
            scale_transform(self.wuji_joint_targets, wuji_limits[..., 0], wuji_limits[..., 1]), -1.0, 1.0
        )

        object_position = self.apple.data.root_pos_w.torch
        object_rotation = self.apple.data.root_quat_w.torch
        object_position_relative_to_base = object_position - self.robot.data.root_pos_w.torch
        object_rotation_6d = matrix_from_quat(object_rotation)[..., :, :2].reshape(self.num_envs, -1)
        object_velocity = torch.cat(
            (self.apple.data.root_lin_vel_w.torch, self.apple.data.root_ang_vel_w.torch), dim=-1
        )

        goal_position = self.goal_position + self.scene.env_origins
        object_rotation_relative_to_goal = quat_mul(quat_inv(self.goal_rotation), object_rotation)
        goal_rotation_error_6d = matrix_from_quat(object_rotation_relative_to_goal)[..., :, :2].reshape(
            self.num_envs, -1
        )

        contact_force = torch.stack(
            [
                torch.linalg.vector_norm(sensor.data.normal_force_matrix_w.torch[:, 0, 0], dim=-1)
                for sensor in self.contact_sensors.values()
            ],
            dim=-1,
        )
        contact_force = torch.log1p(torch.clamp(contact_force, max=self.cfg.contact_force_observation_max))

        observation = torch.cat(
            (
                joint_position,
                joint_velocity,
                arm_target,
                wuji_target,
                object_position_relative_to_base,
                object_rotation_6d,
                object_velocity,
                goal_position - object_position,
                goal_rotation_error_6d,
                contact_force,
            ),
            dim=-1,
        )
        if observation.shape[-1] != self._OBSERVATION_DIM:
            raise RuntimeError(
                f"Expected {self._OBSERVATION_DIM}-D observation, received {observation.shape[-1]}."
            )
        return {"policy": observation, "critic": observation}

    def _get_rewards(self) -> torch.Tensor:
        """Reward reaching, thumb-opposed contact, and the upright object pose."""
        self.extras.pop("log", None)
        hand_points = self.robot.data.body_pos_w.torch[:, self.hand_point_body_ids]
        object_position = self.apple.data.root_pos_w.torch
        hand_dist = torch.linalg.vector_norm(hand_points - object_position.unsqueeze(1), dim=-1).max(dim=1).values
        reach_reward = torch.exp(-self.cfg.reach_reward_scale * hand_dist)

        contact_forces = {
            name: torch.linalg.vector_norm(sensor.data.normal_force_matrix_w.torch[:, 0, 0], dim=-1)
            for name, sensor in self.contact_sensors.items()
        }
        thumb_contact = contact_forces[self._THUMB_CONTACT_BODY_NAME] > self.cfg.contact_force_threshold
        other_contact_count = torch.stack(
            [
                force > self.cfg.contact_force_threshold
                for name, force in contact_forces.items()
                if name != self._THUMB_CONTACT_BODY_NAME
            ],
            dim=1,
        ).sum(dim=1)
        contact_gate = thumb_contact & (other_contact_count >= 1)

        current_keypoints = self._transform_keypoints(object_position, self.apple.data.root_quat_w.torch)
        goal_keypoints = self._transform_keypoints(
            self.goal_position + self.scene.env_origins, self.goal_rotation
        )
        keypoint_error = torch.linalg.vector_norm(current_keypoints - goal_keypoints, dim=-1).mean(dim=1)
        goal_reward = (
            self.cfg.goal_reward_scale * torch.exp(-self.cfg.goal_reward_alpha * keypoint_error) * contact_gate
        )
        contact_reward = 0.01 * contact_gate
        arm_tracking_error = torch.abs(
            self.robot.data.joint_pos.torch[:, self.arm_joint_ids] - self.arm_joint_targets
        ).mean(dim=-1)
        wuji_tracking_error = torch.abs(
            self.robot.data.joint_pos.torch[:, self.wuji_joint_ids] - self.wuji_joint_targets
        ).mean(dim=-1)
        self._update_episode_metrics(
            reach_reward,
            goal_reward,
            contact_reward,
            hand_dist,
            keypoint_error,
            object_position[:, 2],
            contact_gate,
            arm_tracking_error,
            wuji_tracking_error,
        )
        self._update_keypoint_markers(current_keypoints=current_keypoints, goal_keypoints=goal_keypoints)
        return reach_reward + goal_reward + contact_reward

    def _init_episode_metrics(self) -> None:
        """Allocate per-environment buffers for completed-episode diagnostics."""
        self._episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("reach_return", "goal_return", "contact_return")
        }
        self._episode_contact_gate_steps = torch.zeros(self.num_envs, device=self.device)
        self._episode_arm_tracking_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_wuji_tracking_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_min_hand_distance = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_min_keypoint_error = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_max_object_height = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self._termination_torso_apple = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._termination_below_table = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._termination_workspace_exit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _update_episode_metrics(
        self,
        reach_reward: torch.Tensor,
        goal_reward: torch.Tensor,
        contact_reward: torch.Tensor,
        hand_distance: torch.Tensor,
        keypoint_error: torch.Tensor,
        object_height: torch.Tensor,
        contact_gate: torch.Tensor,
        arm_tracking_error: torch.Tensor,
        wuji_tracking_error: torch.Tensor,
    ) -> None:
        """Accumulate task diagnostics and publish scalar metrics through RSL-RL extras."""
        self._episode_reward_sums["reach_return"] += reach_reward
        self._episode_reward_sums["goal_return"] += goal_reward
        self._episode_reward_sums["contact_return"] += contact_reward
        self._episode_contact_gate_steps += contact_gate
        self._episode_arm_tracking_error_sum += arm_tracking_error
        self._episode_wuji_tracking_error_sum += wuji_tracking_error
        self._episode_min_hand_distance = torch.minimum(self._episode_min_hand_distance, hand_distance)
        self._episode_min_keypoint_error = torch.minimum(self._episode_min_keypoint_error, keypoint_error)
        self._episode_max_object_height = torch.maximum(self._episode_max_object_height, object_height)

        log = {
            "Metrics/step_reach_reward": reach_reward.mean(),
            "Metrics/step_goal_reward": goal_reward.mean(),
            "Metrics/step_contact_reward": contact_reward.mean(),
            "Metrics/step_hand_distance": hand_distance.mean(),
            "Metrics/step_keypoint_error": keypoint_error.mean(),
            "Metrics/step_object_height": object_height.mean(),
            "Metrics/step_contact_gate_fraction": contact_gate.float().mean(),
            "Control/step_arm_tracking_error": arm_tracking_error.mean(),
            "Control/step_wuji_tracking_error": wuji_tracking_error.mean(),
            "Control/step_action_saturation_fraction": (self.actions.abs() >= 0.999).float().mean(),
        }

        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            episode_steps = self.episode_length_buf[reset_ids].clamp_min(1).float()
            log.update(
                {
                    f"Metrics/{name}": values[reset_ids].mean()
                    for name, values in self._episode_reward_sums.items()
                }
            )
            log.update(
                {
                    "Metrics/min_hand_distance": self._episode_min_hand_distance[reset_ids].mean(),
                    "Metrics/final_keypoint_error": keypoint_error[reset_ids].mean(),
                    "Metrics/min_keypoint_error": self._episode_min_keypoint_error[reset_ids].mean(),
                    "Metrics/max_object_height": self._episode_max_object_height[reset_ids].mean(),
                    "Metrics/contact_gate_fraction": (
                        self._episode_contact_gate_steps[reset_ids] / episode_steps
                    ).mean(),
                    "Metrics/success": (
                        keypoint_error[reset_ids] < self.cfg.success_keypoint_error_threshold
                    ).float().mean(),
                    "Control/arm_tracking_error": (
                        self._episode_arm_tracking_error_sum[reset_ids] / episode_steps
                    ).mean(),
                    "Control/wuji_tracking_error": (
                        self._episode_wuji_tracking_error_sum[reset_ids] / episode_steps
                    ).mean(),
                    "Terminations/torso_apple": self._termination_torso_apple[reset_ids].float().mean(),
                    "Terminations/below_table": self._termination_below_table[reset_ids].float().mean(),
                    "Terminations/workspace_exit": self._termination_workspace_exit[reset_ids].float().mean(),
                    "Terminations/timeout": self.reset_time_outs[reset_ids].float().mean(),
                }
            )
        self.extras["log"] = log

    def _make_cube_keypoints(self, extent: float) -> torch.Tensor:
        """Create the eight local virtual-cube corners [m]."""
        coordinates = torch.tensor((-extent, extent), device=self.device)
        return torch.cartesian_prod(coordinates, coordinates, coordinates)

    def _transform_keypoints(self, position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """Transform local virtual keypoints into world coordinates [m]."""
        local_keypoints = self.local_cube_keypoints.unsqueeze(0).expand(position.shape[0], -1, -1)
        rotated_keypoints = quat_apply(rotation.unsqueeze(1).expand(-1, local_keypoints.shape[1], -1), local_keypoints)
        return position.unsqueeze(1) + rotated_keypoints

    def _update_keypoint_markers(
        self, current_keypoints: torch.Tensor | None = None, goal_keypoints: torch.Tensor | None = None
    ) -> None:
        """Update debug-only pose keypoint discs without adding scene physics."""
        if self.goal_keypoint_marker is None or self.object_keypoint_marker is None:
            return
        if current_keypoints is None:
            current_keypoints = self._transform_keypoints(
                self.apple.data.root_pos_w.torch, self.apple.data.root_quat_w.torch
            )
        if goal_keypoints is None:
            goal_keypoints = self._transform_keypoints(self.goal_position + self.scene.env_origins, self.goal_rotation)
        self.goal_keypoint_marker.visualize(translations=goal_keypoints.reshape(-1, 3))
        self.object_keypoint_marker.visualize(translations=current_keypoints.reshape(-1, 3))

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate on torso-to-apple contact or when the apple leaves the tabletop workspace."""
        torso_contact = (
            torch.linalg.vector_norm(self.torso_contact_sensor.data.normal_force_matrix_w.torch[:, 0, 0], dim=-1)
            > 0.0
        )
        object_position = self.apple.data.root_pos_w.torch
        table_top_height = self.scene.env_origins[:, 2] + self.table_top_height
        object_below_table = object_position[:, 2] < table_top_height
        object_start_position = self.object_start_position + self.scene.env_origins
        object_too_far = (
            torch.linalg.vector_norm(object_position[:, :2] - object_start_position[:, :2], dim=-1)
            > self.cfg.object_max_horizontal_displacement
        )
        self._termination_torso_apple = torso_contact
        self._termination_below_table = object_below_table
        self._termination_workspace_exit = object_too_far
        terminated = torso_contact | object_below_table | object_too_far
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Restore the authored robot, table, and apple poses without randomization."""
        super()._reset_idx(env_ids)

        robot_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        robot_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=robot_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=self.robot.data.default_joint_pos.torch[env_ids], env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=self.robot.data.default_joint_vel.torch[env_ids], env_ids=env_ids
        )
        self.robot.actuators.target_command.set_position_index(
            value=self.robot.data.default_joint_pos.torch[env_ids], env_ids=env_ids
        )
        # Start each episode's EMA at the reset target, rather than blending
        # its first policy command with a prior episode's target.
        if hasattr(self, "arm_joint_targets"):
            default_joint_pos = self.robot.data.default_joint_pos.torch[env_ids]
            self.arm_joint_targets[env_ids] = default_joint_pos[:, self.arm_joint_ids]
            self.wuji_joint_targets[env_ids] = default_joint_pos[:, self.wuji_joint_ids]
        if hasattr(self, "_episode_reward_sums"):
            for values in self._episode_reward_sums.values():
                values[env_ids] = 0.0
            self._episode_contact_gate_steps[env_ids] = 0.0
            self._episode_arm_tracking_error_sum[env_ids] = 0.0
            self._episode_wuji_tracking_error_sum[env_ids] = 0.0
            self._episode_min_hand_distance[env_ids] = torch.inf
            self._episode_min_keypoint_error[env_ids] = torch.inf
            self._episode_max_object_height[env_ids] = -torch.inf
        self.robot.actuators.target_command.set_position_index(
            value=torch.zeros((len(env_ids), len(self.waist_joint_ids)), device=self.device),
            joint_ids=self.waist_joint_ids,
            env_ids=env_ids,
        )
        table_pose = self.table.data.default_root_pose.torch[env_ids].clone()
        table_pose[:, :3] += self.scene.env_origins[env_ids]
        self.table.write_root_pose_to_sim_index(root_pose=table_pose, env_ids=env_ids)
        self.table.write_root_velocity_to_sim_index(
            root_velocity=self.table.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        apple_pose = self.apple.data.default_root_pose.torch[env_ids].clone()
        apple_pose[:, :3] += self.scene.env_origins[env_ids]
        self.apple.write_root_pose_to_sim_index(root_pose=apple_pose, env_ids=env_ids)
        self.apple.write_root_velocity_to_sim_index(
            root_velocity=self.apple.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        self._update_keypoint_markers()
