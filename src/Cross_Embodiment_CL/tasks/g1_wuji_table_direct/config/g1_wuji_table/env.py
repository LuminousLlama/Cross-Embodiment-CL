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
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

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

    def __init__(self, cfg: G1WujiTableEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self.arm_joint_ids, _ = self.robot.find_joints(self._ARM_JOINT_NAMES, preserve_order=True)
        self.wuji_joint_ids, _ = self.robot.find_joints(self._WUJI_JOINT_NAMES, preserve_order=True)
        self.waist_joint_ids, _ = self.robot.find_joints("waist_.*_joint")
        if self.cfg.action_space != len(self.arm_joint_ids) + WujiLatentActionPipeline.latent_dim:
            raise ValueError(
                f"{type(self.cfg).__name__} declares action_space={self.cfg.action_space}, but the configured "
                "right arm plus Wuji latent action require "
                f"{len(self.arm_joint_ids) + WujiLatentActionPipeline.latent_dim}."
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
        self.goal_marker: VisualizationMarkers | None = None
        if self.cfg.goal_marker_debug_vis:
            self.goal_marker = VisualizationMarkers(self.cfg.goal_marker_cfg)
            self.goal_marker.visualize(translations=self.goal_position + self.scene.env_origins)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.apple = RigidObject(self.cfg.apple_cfg)

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
        self.robot.set_joint_position_target_index(
            target=self.arm_joint_targets,
            joint_ids=self.arm_joint_ids,
        )
        self.robot.set_joint_position_target_index(
            target=self.wuji_joint_targets,
            joint_ids=self.wuji_joint_ids,
        )
        self.robot.set_joint_position_target_index(
            target=self.waist_joint_targets,
            joint_ids=self.waist_joint_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return an empty policy observation until the observation design is added."""
        return {"policy": torch.empty((self.num_envs, 0), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        """Return neutral rewards until task rewards are designed."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Only reset on the long inspection timeout."""
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        self.robot.set_joint_position_target_index(
            target=self.robot.data.default_joint_pos.torch[env_ids], env_ids=env_ids
        )
        # Start each episode's EMA at the reset target, rather than blending
        # its first policy command with a prior episode's target.
        if hasattr(self, "arm_joint_targets"):
            self.arm_joint_targets[env_ids] = self.robot.data.default_joint_pos.torch[env_ids, self.arm_joint_ids]
            self.wuji_joint_targets[env_ids] = self.robot.data.default_joint_pos.torch[env_ids, self.wuji_joint_ids]
        self.robot.set_joint_position_target_index(
            target=torch.zeros((len(env_ids), len(self.waist_joint_ids)), device=self.device),
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
