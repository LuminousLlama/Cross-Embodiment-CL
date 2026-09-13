# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal DirectRLEnv used solely to inspect the G1-Wuji table scene."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import Camera, ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul, scale_transform

from Cross_Embodiment_CL.models import WujiLatentActionPipeline

from .depth_camera import normalize_depth
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
    # Side-swing joints of the four fingers, whose range is symmetric about 0 rad.
    _WUJI_SIDE_SWING_JOINT_NAMES = tuple(f"right_finger{finger}_joint2" for finger in range(2, 6))
    _HAND_POINT_BODY_NAMES = ("right_palm_link",) + tuple(f"right_finger{finger}_tip_link" for finger in range(1, 6))
    # Apple contact is sensed per group, over every hand body that owns a collision shape.  The
    # *_tip_link frames above own none, so a sensor on one reads 0 N forever; fingertip contact
    # lands on link4.  Fingers 2-5 have no link1 collider either.
    _CONTACT_BODY_GROUPS = {
        "palm": ("right_palm_link",),
        "finger1": (
            "right_finger1_link1",
            "right_finger1_link2",
            "right_finger1_link2_softbody",
            "right_finger1_link3",
            "right_finger1_link4",
        ),
        **{
            f"finger{finger}": tuple(f"right_finger{finger}_link{link}" for link in (2, 3, 4))
            for finger in range(2, 6)
        },
    }
    _THUMB_CONTACT_GROUP = "finger1"
    _OBSERVATION_DIM = 117
    # The deployable proprioceptive prefix of the privileged observation: joint positions and
    # velocities plus the commanded arm and hand targets.
    _STUDENT_OBSERVATION_DIM = 87

    def __init__(self, cfg: G1WujiTableEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self.arm_joint_ids, _ = self.robot.find_joints(self._ARM_JOINT_NAMES, preserve_order=True)
        self.wuji_joint_ids, _ = self.robot.find_joints(self._WUJI_JOINT_NAMES, preserve_order=True)
        # Commanded hand targets never bend a joint backwards past 0 rad; the thumb base's own limit already sits
        # just above it, and the four fingers' side-swing joints keep their symmetric range.  Only the command is
        # restricted: the simulated joints keep their full limits, since the real hand can be pushed backwards.
        self._wuji_command_lower_floor = torch.tensor(
            [-torch.inf if name in self._WUJI_SIDE_SWING_JOINT_NAMES else 0.0 for name in self._WUJI_JOINT_NAMES],
            device=self.device,
        )
        self.waist_joint_ids, _ = self.robot.find_joints("waist_.*_joint")
        self.hand_point_body_ids, hand_point_names = self.robot.find_bodies(
            self._HAND_POINT_BODY_NAMES, preserve_order=True
        )
        if tuple(hand_point_names) != self._HAND_POINT_BODY_NAMES:
            raise ValueError(
                f"Expected hand point bodies {self._HAND_POINT_BODY_NAMES}, found {tuple(hand_point_names)}."
            )
        for group_name, sensor in self.contact_sensors.items():
            if sensor.num_sensors != len(self._CONTACT_BODY_GROUPS[group_name]):
                raise ValueError(
                    f"Contact group '{group_name}' expects bodies {self._CONTACT_BODY_GROUPS[group_name]}, "
                    f"but its sensor resolved {sensor.num_sensors}."
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
        if self.cfg.arm_joint_velocity_limit <= 0.0 or self.cfg.hand_joint_velocity_limit <= 0.0:
            raise ValueError("arm_joint_velocity_limit and hand_joint_velocity_limit must be positive.")
        if not 0.0 <= self.cfg.gravity_curriculum_start <= 1.0:
            raise ValueError("gravity_curriculum_start must be in [0, 1].")
        if self.cfg.gravity_curriculum_steps <= 0:
            raise ValueError("gravity_curriculum_steps must be positive.")
        if self.cfg.contact_debug_interval <= 0:
            raise ValueError("contact_debug_interval must be positive.")
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.arm_joint_targets = torch.zeros((self.num_envs, len(self.arm_joint_ids)), device=self.device)
        self.wuji_joint_targets = torch.zeros((self.num_envs, len(self.wuji_joint_ids)), device=self.device)
        # Targets at the start of the current policy step, which _apply_action interpolates from.
        self._arm_joint_targets_start = torch.zeros_like(self.arm_joint_targets)
        self._wuji_joint_targets_start = torch.zeros_like(self.wuji_joint_targets)
        self._action_substep = 0
        self.arm_action_scale = torch.full((len(self.arm_joint_ids),), 0.5, device=self.device)
        self.waist_joint_targets = torch.zeros((self.num_envs, len(self.waist_joint_ids)), device=self.device)
        self.goal_position = torch.tensor(self.cfg.goal_position, device=self.device).repeat(self.num_envs, 1)
        self.goal_rotation = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device).repeat(self.num_envs, 1)
        self.object_start_position = torch.tensor(self.cfg.apple_cfg.init_state.pos, device=self.device).repeat(
            self.num_envs, 1
        )
        self._gravity_frac = 1.0
        self._last_applied_gravity_frac: float | None = None
        self._apply_gravity_curriculum(force=True)
        self.table_top_height = self.cfg.table_cfg.init_state.pos[2] + 0.5 * self.cfg.table_cfg.spawn.size[2]
        self.local_cube_keypoints = self._make_cube_keypoints(self.cfg.keypoint_extent)
        self._init_episode_metrics()
        self._init_penetration_probe()
        self.goal_keypoint_marker: VisualizationMarkers | None = None
        self.object_keypoint_marker: VisualizationMarkers | None = None
        if self.cfg.debug.keypoint_markers:
            self.goal_keypoint_marker = VisualizationMarkers(self.cfg.goal_keypoint_marker_cfg)
            self.object_keypoint_marker = VisualizationMarkers(self.cfg.object_keypoint_marker_cfg)
            self._update_keypoint_markers()

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.apple = RigidObject(self.cfg.apple_cfg)
        # One multi-body sensor per group; its force matrix is (envs, bodies, 1 apple, 3).
        self.contact_sensors: dict[str, ContactSensor] = {}
        for group_name, body_names in self._CONTACT_BODY_GROUPS.items():
            sensor_cfg = self.cfg.contact_sensor_cfg.replace(
                prim_path=f"/World/envs/env_[^/]+/G1Wuji/wujihand/({'|'.join(body_names)})"
            )
            self.contact_sensors[group_name] = ContactSensor(sensor_cfg)
        self.torso_contact_sensor = ContactSensor(self.cfg.torso_contact_sensor_cfg)
        # The student's head depth camera exists only when a preset configures one (presets=distill).
        self.depth_camera = Camera(self.cfg.depth_camera) if self.cfg.depth_camera is not None else None

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
        self.scene.sensors.update({f"{name}_contact": sensor for name, sensor in self.contact_sensors.items()})
        self.scene.sensors["torso_contact"] = self.torso_contact_sensor
        if self.depth_camera is not None:
            self.scene.sensors["depth_camera"] = self.depth_camera
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Map arm deltas and Wuji latent actions onto speed-capped physical joint targets."""
        self.actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._arm_joint_targets_start.copy_(self.arm_joint_targets)
        self._wuji_joint_targets_start.copy_(self.wuji_joint_targets)
        self._action_substep = 0
        arm_limits = self.robot.data.soft_joint_pos_limits.torch[:, self.arm_joint_ids]
        arm_lower, arm_upper = arm_limits[..., 0], arm_limits[..., 1]
        arm_actions = self.actions[:, : len(self.arm_joint_ids)]
        arm_targets = (
            self.robot.data.default_joint_pos.torch[:, self.arm_joint_ids]
            + self.arm_action_scale.unsqueeze(0) * arm_actions
        )
        arm_targets = torch.clamp(arm_targets, min=arm_lower, max=arm_upper)
        self._advance_joint_targets(
            self.arm_joint_targets,
            arm_targets,
            self.arm_joint_ids,
            self.cfg.arm_action_ema_alpha,
            self.cfg.arm_joint_velocity_limit,
        )

        wuji_limits = self.robot.data.soft_joint_pos_limits.torch[:, self.wuji_joint_ids]
        wuji_command_lower = torch.maximum(wuji_limits[..., 0], self._wuji_command_lower_floor)
        wuji_targets = self.wuji_action_pipeline.latent_action_to_joint_target(
            self.actions[:, len(self.arm_joint_ids) :], wuji_command_lower, wuji_limits[..., 1]
        )
        self._advance_joint_targets(
            self.wuji_joint_targets,
            wuji_targets,
            self.wuji_joint_ids,
            self.cfg.wuji_action_ema_alpha,
            self.cfg.hand_joint_velocity_limit,
        )
        # The anti-windup clamp follows the measured position, which contact can push backwards; never command that.
        self.wuji_joint_targets.clamp_(min=wuji_command_lower)
        self._apply_gravity_curriculum()

    def _apply_gravity_curriculum(self, force: bool = False) -> None:
        """Ramp the whole scene's configured gravity from ``gravity_curriculum_start`` to full strength.

        The update is sent only when the fraction changes by at least 0.005, avoiding a model-property
        notification every policy step while keeping the 600-iteration ramp smooth.  Isaac Lab's current
        Newton, OvPhysX, and Isaac Sim PhysX managers expose different runtime gravity APIs, so this selects
        their capability rather than changing task dynamics by backend name.
        """
        start = self.cfg.gravity_curriculum_start
        ramp = min(1.0, self.common_step_counter / self.cfg.gravity_curriculum_steps)
        self._gravity_frac = start + (1.0 - start) * ramp
        if (
            not force
            and self._last_applied_gravity_frac is not None
            and abs(self._gravity_frac - self._last_applied_gravity_frac) < 0.005
        ):
            return

        gravity = (0.0, 0.0, self.cfg.sim.gravity[2] * self._gravity_frac)
        physics_manager = self.sim.physics_manager
        if hasattr(physics_manager, "get_model"):
            from newton import ModelFlags

            model = physics_manager.get_model()
            if model is None:
                raise RuntimeError("Newton model is not initialized; cannot apply the gravity curriculum.")
            model.set_gravity(gravity)
            physics_manager.add_model_change(ModelFlags.MODEL_PROPERTIES)
        elif hasattr(physics_manager, "set_gravity"):
            physics_manager.set_gravity(gravity)
        else:
            import carb

            sim_utils.SimulationContext.instance().physics_sim_view.set_gravity(carb.Float3(*gravity))
        self._last_applied_gravity_frac = self._gravity_frac

    def _advance_joint_targets(
        self,
        targets: torch.Tensor,
        commanded: torch.Tensor,
        joint_ids: Sequence[int],
        ema_alpha: float,
        velocity_limit: float,
    ) -> None:
        """Take one policy-rate EMA step toward ``commanded``, capped at ``velocity_limit`` [rad/s].

        Uncapped this is exactly ``targets.lerp_(commanded, ema_alpha)``.  Capping the per-step change at
        ``velocity_limit * step_dt`` is what bounds joint speed on every backend, since solver velocity
        limits are not portable.

        The target is then kept within ``(effort_limit + damping * velocity_limit) / stiffness`` of the
        measured joint position, using the drive gains and effort limits the actuator cfg gave the solver.
        That is the lead at which the drive saturates, plus the lead it needs to cruise at the cap against its
        own damping.  A target further ahead adds no torque, but while a joint is blocked it would keep winding
        up and then snap the joint back at full effort once released.
        """
        max_step = velocity_limit * self.step_dt
        targets.add_(torch.clamp(ema_alpha * (commanded - targets), -max_step, max_step))

        data = self.robot.data
        max_lead = (
            data.joint_effort_limits.torch[:, joint_ids] + data.joint_damping.torch[:, joint_ids] * velocity_limit
        ) / torch.clamp_min(data.joint_stiffness.torch[:, joint_ids], 1.0e-6)
        joint_pos = data.joint_pos.torch[:, joint_ids]
        targets.clamp_(min=joint_pos - max_lead, max=joint_pos + max_lead)

    def _apply_action(self) -> None:
        """Apply arm and Wuji targets while holding all waist joints at zero."""
        # Spread each policy step's capped target change evenly over its physics substeps, so the drives
        # track a ramp rather than a staircase, whose jumps overshoot the cap between policy steps.  If the
        # backend folds decimation into one call, this applies a partial step, which still respects the cap.
        self._action_substep += 1
        fraction = min(self._action_substep / self.cfg.decimation, 1.0)
        self.robot.actuators.target_command.set_position_index(
            value=torch.lerp(self._arm_joint_targets_start, self.arm_joint_targets, fraction),
            joint_ids=self.arm_joint_ids,
        )
        self.robot.actuators.target_command.set_position_index(
            value=torch.lerp(self._wuji_joint_targets_start, self.wuji_joint_targets, fraction),
            joint_ids=self.wuji_joint_ids,
        )
        # Velocity feedforward, disabled for now: the ramp's slope as the drive velocity target stops damping
        # from braking the ramp into a damping / stiffness * speed lag, but at the capped speeds that lag is
        # small, and the real Wuji client accepts position targets only.  The slope decays to zero at the
        # target with the EMA, so it needs no stopping logic.
        # self.robot.actuators.target_command.set_velocity_index(
        #     value=(self.arm_joint_targets - self._arm_joint_targets_start) / self.step_dt,
        #     joint_ids=self.arm_joint_ids,
        # )
        # self.robot.actuators.target_command.set_velocity_index(
        #     value=(self.wuji_joint_targets - self._wuji_joint_targets_start) / self.step_dt,
        #     joint_ids=self.wuji_joint_ids,
        # )
        self.robot.actuators.target_command.set_position_index(
            value=self.waist_joint_targets,
            joint_ids=self.waist_joint_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return the identical 117-D privileged state for policy and critic, plus the student's proprioception.

        The observation contains normalized full-robot state and commanded arm/hand targets,
        followed by apple and goal state in the fixed robot-base frame and filtered contact forces.
        The ``student`` group is the 87-D robot-state prefix alone, which a real robot can measure.  With a depth
        camera configured, ``camera`` adds its normalized depth image with shape ``(N, 1, H, W)``.
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
        # Arm and hand speeds are scaled by the task caps rather than the looser solver limits.
        joint_velocity_limits = self.robot.data.soft_joint_vel_limits.torch.clone()
        joint_velocity_limits[:, self.arm_joint_ids] = self.cfg.arm_joint_velocity_limit
        joint_velocity_limits[:, self.wuji_joint_ids] = self.cfg.hand_joint_velocity_limit
        joint_velocity_limits.clamp_min_(1.0e-6)
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

        contact_force = torch.log1p(
            torch.clamp(self._contact_group_forces(), max=self.cfg.contact_force_observation_max)
        )

        proprioception = torch.cat((joint_position, joint_velocity, arm_target, wuji_target), dim=-1)
        observation = torch.cat(
            (
                proprioception,
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
        if proprioception.shape[-1] != self._STUDENT_OBSERVATION_DIM:
            raise RuntimeError(
                f"Expected {self._STUDENT_OBSERVATION_DIM}-D student observation, received {proprioception.shape[-1]}."
            )
        observations = {"policy": observation, "critic": observation, "student": proprioception}
        if self.depth_camera is not None:
            # (N, H, W, 1) planar depth [m] to (N, 1, H, W), the layout RSL-RL's CNN expects.
            depth = self.depth_camera.data.output["distance_to_image_plane"].torch
            observations["camera"] = normalize_depth(depth, self.cfg.student_depth_max_m).permute(0, 3, 1, 2)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Reward reaching, thumb-opposed contact, and the upright object pose.

        ``reward_mode`` (see :attr:`G1WujiTableEnvCfg.reward_mode`) switches between today's
        shaped reward and the ADEPT-style minimal reward; the shaped branch below is unchanged.
        """
        self.extras.pop("log", None)
        hand_points = self.robot.data.body_pos_w.torch[:, self.hand_point_body_ids]
        object_position = self.apple.data.root_pos_w.torch
        hand_point_dist = torch.linalg.vector_norm(hand_points - object_position.unsqueeze(1), dim=-1)
        hand_dist = hand_point_dist.max(dim=1).values
        nearest_hand_dist = hand_point_dist.min(dim=1).values
        reach_reward = torch.exp(-self.cfg.reach_reward_scale * hand_dist)

        contact_force_stack = self._contact_group_forces()

        current_keypoints = self._transform_keypoints(object_position, self.apple.data.root_quat_w.torch)
        goal_keypoints = self._transform_keypoints(
            self.goal_position + self.scene.env_origins, self.goal_rotation
        )
        keypoint_error = torch.linalg.vector_norm(current_keypoints - goal_keypoints, dim=-1).mean(dim=1)
        position_error = torch.linalg.vector_norm(
            object_position - (self.goal_position + self.scene.env_origins), dim=-1
        )
        # Angle between the object and goal quaternions, using |dot| for double cover.
        rotation_dot = (self.apple.data.root_quat_w.torch * self.goal_rotation).sum(dim=-1).abs().clamp(max=1.0)
        rotation_error = torch.rad2deg(2.0 * torch.acos(rotation_dot))

        if self.cfg.reward_mode == "shaped":
            contact_group_count = (contact_force_stack > self.cfg.contact_force_threshold).sum(dim=1)
            contact_gate = contact_group_count >= self.cfg.contact_min_bodies
            # Gating the pose reward on contact is the original design and it is kept, because an
            # ungated version is a trap: keypoint_error can only rise when the apple is disturbed,
            # so touching it is net negative.  Measured ungated, the policy hovered with its
            # nearest hand point 1.8 cm clear of the apple and touched in ~1% of steps.  Gated,
            # experimenting with contact costs nothing and the lift only competes once the apple
            # is held.  The gate itself had to be repaired first -- see contact_force_threshold.
            goal_reward = (
                self.cfg.goal_reward_scale * torch.exp(-self.cfg.goal_reward_alpha * keypoint_error) * contact_gate
            )
            # Graded in force rather than gated.  Paying contact_reward_scale * contact_gate is
            # zero until two fingers already touch, so it supplies no gradient toward touching
            # at all; this term rises with any contact and bridges reach -> grasp.  tanh bounds
            # it so pressing the apple into the table cannot out-earn lifting it.
            lift_reward = self.cfg.lift_reward_scale * self._lift_fraction(object_position)
            # Contact only counts while the apple is not being crushed downward, which is what
            # closes off the press exploit without removing the gradient toward touching.
            rest_height = self.object_start_position[:, 2] + self.scene.env_origins[:, 2]
            held = object_position[:, 2] > rest_height - self.cfg.press_tolerance
            contact_reward = (
                self.cfg.contact_reward_scale
                * torch.tanh(contact_force_stack / self.cfg.contact_force_reference).mean(dim=1)
                * held
            )
            goal_alpha_step = 0.0
            reward = reach_reward + goal_reward + contact_reward + lift_reward
        elif self.cfg.reward_mode == "adept":
            # Grasp gate: the thumb and at least one other finger (never the palm) each past
            # adept_gate_force, per the ADEPT paper's minimal reward. No dense contact term, no
            # lift term, and no press guard -- the gate alone stands in for all three.
            thumb_index = list(self.contact_sensors).index(self._THUMB_CONTACT_GROUP)
            other_finger_indices = [
                index
                for index, name in enumerate(self.contact_sensors)
                if name != "palm" and name != self._THUMB_CONTACT_GROUP
            ]
            thumb_force = contact_force_stack[:, thumb_index]
            other_finger_force = contact_force_stack[:, other_finger_indices]
            contact_gate = (thumb_force > self.cfg.adept_gate_force) & (
                other_finger_force > self.cfg.adept_gate_force
            ).any(dim=1)
            # Keypoint-error sharpness ramps so the goal term starts forgiving (easy to earn once
            # gated) and sharpens into a tighter pose requirement as training progresses.
            alpha_ramp = min(1.0, self.common_step_counter / self.cfg.adept_goal_alpha_steps)
            goal_alpha_step = self.cfg.adept_goal_alpha_start + (
                self.cfg.adept_goal_alpha_end - self.cfg.adept_goal_alpha_start
            ) * alpha_ramp
            goal_reward = self.cfg.goal_reward_scale * torch.exp(-goal_alpha_step * keypoint_error) * contact_gate
            contact_reward = self.cfg.adept_contact_reward_scale * contact_gate.float()
            # Optional dense assist, off by default (adept_lift_reward_scale=0.0): reuses the
            # shaped mode's lift_fraction so a policy far from the goal -- where the alpha-sharpened,
            # gate-only goal term gives no gradient -- still has a signal toward lifting.
            lift_reward = self.cfg.adept_lift_reward_scale * self._lift_fraction(object_position)
            reward = reach_reward + goal_reward + contact_reward + lift_reward
        else:
            raise ValueError(f"Unknown reward_mode '{self.cfg.reward_mode}'; expected 'shaped' or 'adept'.")

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
            lift_reward,
            hand_dist,
            keypoint_error,
            position_error,
            rotation_error,
            object_position[:, 2],
            contact_gate,
            arm_tracking_error,
            wuji_tracking_error,
            contact_force_stack,
            nearest_hand_dist,
            goal_alpha_step,
        )
        self._update_keypoint_markers(current_keypoints=current_keypoints, goal_keypoints=goal_keypoints)
        return reward

    def _lift_fraction(self, object_position: torch.Tensor) -> torch.Tensor:
        """Return the apple's height progress from rest to goal, clamped to ``[0, 1]``.

        Rest height is the authored spawn height; the apple settles a touch below it, so the
        clamp makes "sitting untouched" score exactly zero. Shared by the shaped mode's dense
        lift term and the optional ``adept_lift_reward_scale`` term.
        """
        rest_height = self.object_start_position[:, 2] + self.scene.env_origins[:, 2]
        goal_height = self.goal_position[:, 2] + self.scene.env_origins[:, 2]
        return torch.clamp((object_position[:, 2] - rest_height) / (goal_height - rest_height), 0.0, 1.0)

    def _contact_group_forces(self) -> torch.Tensor:
        """Return each contact group's apple normal force [N], summed over the group's bodies.

        Summing makes a finger that holds with several phalanges read as its total load.

        Returns:
            Force magnitudes with shape ``(num_envs, num_groups)`` in ``_CONTACT_BODY_GROUPS`` order.
        """
        return torch.stack(
            [
                torch.linalg.vector_norm(sensor.data.normal_force_matrix_w.torch[:, :, 0], dim=-1).sum(dim=1)
                for sensor in self.contact_sensors.values()
            ],
            dim=-1,
        )

    def _init_penetration_probe(self) -> None:
        """Index the MJWarp geoms of the hand, apple, table, and other robot bodies for the contact diagnostic.

        Isaac Lab contact sensors report forces but not penetration depth, so depth is read from the MJWarp
        contact buffer.  UNTESTED on PhysX: there no MJWarp solver exists, the probe stays off, and the
        penetration tags are simply not logged.
        """
        self._mjw_data = None
        try:
            import mujoco

            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        solver = getattr(NewtonManager, "_solver", None)
        mj_model = getattr(solver, "mj_model", None)
        if mj_model is None:
            return
        labels = [
            f"{mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, int(mj_model.geom_bodyid[geom])) or ''}/"
            f"{mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''}"
            for geom in range(mj_model.ngeom)
        ]
        # "g1_simplified" is the USD sub-asset name for the arm/torso, mirroring "wujihand" for the hand
        # (see the robot prim paths in env_cfg.py); it covers the non-hand robot geoms used for self-contact.
        self._hand_geoms, self._apple_geoms, self._table_geoms, self._other_robot_geoms = (
            torch.tensor([key in label for label in labels], device=self.device)
            for key in ("wujihand", "Apple", "Table", "g1_simplified")
        )
        if not (
            self._hand_geoms.any()
            and self._apple_geoms.any()
            and self._table_geoms.any()
            and self._other_robot_geoms.any()
        ):
            raise ValueError(
                "Penetration probe could not find the hand, apple, table, and robot geoms in the MJWarp model."
            )
        self._mjw_data = solver.mjw_data
        # One-time record of the apple's built collision-shape count, so runs log the real hull
        # count produced by whatever mesh_approximation_name/max_hull_vertices Hydra selected.
        approximation = self.cfg.apple_cfg.spawn.collision_props.mesh_collision_property.mesh_approximation_name
        print(f"[apple-collision] approximation={approximation} shapes={int(self._apple_geoms.sum())}")

    def _contact_penetration(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return each environment's deepest hand-apple, apple-table, and hand self-contact penetration [m].

        Self-contact covers hand-hand and hand-to-other-robot-geom (arm/torso) pairs; a pair the solver
        filters out of collision (e.g. palm <-> proximal) never appears in the buffer, so it never
        contributes here.  All three are zero without contact.
        """
        contact = self._mjw_data.contact
        dist = wp.to_torch(contact.dist)
        geoms = wp.to_torch(contact.geom).long().clamp_min(0)
        worlds = wp.to_torch(contact.worldid).long().clamp(0, self.num_envs - 1)
        # Entries past the detected-contact count are stale; comparing on device avoids a host sync each step.
        detected = torch.arange(dist.shape[0], device=self.device) < wp.to_torch(self._mjw_data.nacon)
        depth = torch.where(detected, (-dist).clamp_min(0.0), 0.0)
        first, second = geoms[:, 0], geoms[:, 1]
        apple_first, apple_second = self._apple_geoms[first], self._apple_geoms[second]
        penetrations = []
        for other in (self._hand_geoms, self._table_geoms):
            pair = (apple_first & other[second]) | (apple_second & other[first])
            penetrations.append(
                torch.zeros(self.num_envs, device=self.device).scatter_reduce_(
                    0, worlds, torch.where(pair, depth, 0.0), reduce="amax"
                )
            )
        hand_first, hand_second = self._hand_geoms[first], self._hand_geoms[second]
        other_first, other_second = self._other_robot_geoms[first], self._other_robot_geoms[second]
        self_pair = (hand_first & hand_second) | (hand_first & other_second) | (hand_second & other_first)
        penetrations.append(
            torch.zeros(self.num_envs, device=self.device).scatter_reduce_(
                0, worlds, torch.where(self_pair, depth, 0.0), reduce="amax"
            )
        )
        return penetrations[0], penetrations[1], penetrations[2]

    def _contact_demand_metrics(self) -> dict[str, torch.Tensor]:
        """Return sampled MJWarp contact demand for sizing solver capacities.

        The probe reads MJWarp only when explicitly enabled. ``nacon`` and
        ``ncollision`` are global counts, while ``nefc`` and the contact
        ``worldid`` buffer identify the busiest cloned environment.
        """
        if (
            not self.cfg.contact_debug
            or self._mjw_data is None
            or self.common_step_counter % self.cfg.contact_debug_interval != 0
        ):
            return {}

        contact = self._mjw_data.contact
        worlds = wp.to_torch(contact.worldid).long().clamp(0, self.num_envs - 1)
        detected = torch.arange(worlds.shape[0], device=self.device) < wp.to_torch(self._mjw_data.nacon)
        contacts_per_world = torch.zeros(self.num_envs, device=self.device).scatter_add_(
            0, worlds, detected.to(dtype=torch.float32)
        )
        constraint_rows = wp.to_torch(self._mjw_data.nefc).float()
        overflow = wp.to_torch(self._mjw_data.overflow)
        return {
            "Debug/contact_count_total_step": wp.to_torch(self._mjw_data.nacon).float().squeeze(),
            "Debug/contact_count_busiest_step": contacts_per_world.max(),
            "Debug/contact_count_p99_step": torch.quantile(contacts_per_world, 0.99),
            "Debug/constraint_rows_busiest_step": constraint_rows.max(),
            "Debug/constraint_rows_p99_step": torch.quantile(constraint_rows, 0.99),
            "Debug/broadphase_candidate_count_total_step": wp.to_torch(self._mjw_data.ncollision).float().squeeze(),
            "Debug/overflow_frac_step": overflow.ne(0).float().mean(),
            "Debug/overflow_mask_step": overflow.max().float(),
        }

    def _init_episode_metrics(self) -> None:
        """Allocate per-environment buffers for completed-episode diagnostics."""
        self._episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("reach", "goal", "contact", "lift")
        }
        self._episode_contact_gate_steps = torch.zeros(self.num_envs, device=self.device)
        self._episode_arm_tracking_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_wuji_tracking_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_min_hand_distance = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_min_keypoint_error = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_max_object_height = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self._episode_max_hand_penetration = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_self_penetration = torch.zeros(self.num_envs, device=self.device)
        self._termination_torso_apple = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._termination_below_table = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._termination_workspace_exit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._termination_nonfinite = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Last known-finite apple linear speed [m/s] and height [m] per environment, used only to
        # describe a non-finite episode in the diagnostic print below; updated at the end of _get_dones.
        self._prev_apple_speed = torch.zeros(self.num_envs, device=self.device)
        self._prev_apple_height = torch.zeros(self.num_envs, device=self.device)
        self._nonfinite_diag_count = 0

    def _update_episode_metrics(
        self,
        reach_reward: torch.Tensor,
        goal_reward: torch.Tensor,
        contact_reward: torch.Tensor,
        lift_reward: torch.Tensor,
        hand_distance_farthest: torch.Tensor,
        keypoint_error: torch.Tensor,
        position_error: torch.Tensor,
        rotation_error: torch.Tensor,
        object_height: torch.Tensor,
        contact_gate: torch.Tensor,
        arm_tracking_error: torch.Tensor,
        wuji_tracking_error: torch.Tensor,
        contact_force_stack: torch.Tensor,
        hand_distance_nearest: torch.Tensor,
        goal_alpha_step: float,
    ) -> None:
        """Accumulate task diagnostics and publish scalar metrics through RSL-RL extras.

        Tags read ``Group/<quantity>_<qualifier>_<timescale>``, so TensorBoard's alphabetical
        order keeps a quantity's variants together.  The timescale comes last: ``step`` is the
        current step averaged over environments; ``ep`` (mean), ``ep_min``, ``ep_max``,
        ``ep_final``, and ``ep_return`` (sum) summarise the episodes that reset this step.

        ``contact_gate`` and ``goal_alpha_step`` are the mode-appropriate grasp gate and goal
        sharpness from :meth:`G1WujiTableEnv._get_rewards`: the ``Contact/gate_frac_*`` tags
        below read whichever gate the active ``reward_mode`` used, and ``goal_alpha_step`` is 0.0
        under ``reward_mode="shaped"``, which has no ramp.
        """
        self._episode_reward_sums["reach"] += reach_reward
        self._episode_reward_sums["goal"] += goal_reward
        self._episode_reward_sums["contact"] += contact_reward
        self._episode_reward_sums["lift"] += lift_reward
        self._episode_contact_gate_steps += contact_gate
        self._episode_arm_tracking_error_sum += arm_tracking_error
        self._episode_wuji_tracking_error_sum += wuji_tracking_error
        self._episode_min_hand_distance = torch.minimum(self._episode_min_hand_distance, hand_distance_farthest)
        self._episode_min_keypoint_error = torch.minimum(self._episode_min_keypoint_error, keypoint_error)
        self._episode_max_object_height = torch.maximum(self._episode_max_object_height, object_height)

        log = {
            "Task/keypoint_error_step": keypoint_error.mean(),
            "Task/object_height_step": object_height.mean(),
            # Distances from the apple centre to the six hand points (palm and fingertips).
            "Reach/hand_distance_farthest_step": hand_distance_farthest.mean(),
            "Reach/hand_distance_nearest_step": hand_distance_nearest.mean(),
            "Contact/force_max_step": contact_force_stack.max(dim=1).values.mean(),
            "Contact/force_thumb_step": contact_force_stack[
                :, list(self.contact_sensors).index(self._THUMB_CONTACT_GROUP)
            ].mean(),
            "Contact/gate_frac_step": contact_gate.float().mean(),
            "Contact/groups_over_threshold_step": (
                contact_force_stack > self.cfg.contact_force_threshold
            ).float().sum(dim=1).mean(),
            "Contact/touch_frac_any_step": (contact_force_stack > 0.0).any(dim=1).float().mean(),
            "Reward/reach_step": reach_reward.mean(),
            "Reward/goal_step": goal_reward.mean(),
            "Reward/contact_step": contact_reward.mean(),
            "Reward/lift_step": lift_reward.mean(),
            "Control/action_saturation_frac_step": (self.actions.abs() >= 0.999).float().mean(),
            "Control/arm_tracking_error_step": arm_tracking_error.mean(),
            "Control/wuji_tracking_error_step": wuji_tracking_error.mean(),
            "Curriculum/gravity_frac_step": self._gravity_frac,
            "Curriculum/goal_alpha_step": goal_alpha_step,
        }
        log.update(
            {
                f"Contact/touch_frac_{name}_step": (contact_force_stack[:, index] > 0.0).float().mean()
                for index, name in enumerate(self.contact_sensors)
            }
        )
        if self._mjw_data is not None:
            # Deepest contact per environment [m], averaged over environments.
            hand_penetration, table_penetration, self_penetration = self._contact_penetration()
            self._episode_max_hand_penetration = torch.maximum(self._episode_max_hand_penetration, hand_penetration)
            self._episode_max_self_penetration = torch.maximum(self._episode_max_self_penetration, self_penetration)
            log["Contact/penetration_hand_step"] = hand_penetration.mean()
            log["Contact/penetration_table_step"] = table_penetration.mean()
            log["Contact/penetration_self_step"] = self_penetration.mean()
            log["Contact/self_penetrating_frac_step"] = (self_penetration > 0.0005).float().mean()
        log.update(self._contact_demand_metrics())

        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            episode_steps = self.episode_length_buf[reset_ids].clamp_min(1).float()
            log.update(
                {
                    f"Reward/{name}_ep_return": values[reset_ids].mean()
                    for name, values in self._episode_reward_sums.items()
                }
            )
            log.update(
                {
                    "Task/success": (
                        keypoint_error[reset_ids] < self.cfg.success_keypoint_error_threshold
                    ).float().mean(),
                    "Task/keypoint_error_ep_final": keypoint_error[reset_ids].mean(),
                    "Task/position_error_ep_final": position_error[reset_ids].mean(),
                    "Task/rotation_error_ep_final": rotation_error[reset_ids].mean(),
                    "Task/keypoint_error_ep_min": self._episode_min_keypoint_error[reset_ids].mean(),
                    "Task/object_height_ep_max": self._episode_max_object_height[reset_ids].mean(),
                    "Reach/hand_distance_farthest_ep_min": self._episode_min_hand_distance[reset_ids].mean(),
                    "Contact/gate_frac_ep": (self._episode_contact_gate_steps[reset_ids] / episode_steps).mean(),
                    "Control/arm_tracking_error_ep": (
                        self._episode_arm_tracking_error_sum[reset_ids] / episode_steps
                    ).mean(),
                    "Control/wuji_tracking_error_ep": (
                        self._episode_wuji_tracking_error_sum[reset_ids] / episode_steps
                    ).mean(),
                    "Terminations/torso_apple": self._termination_torso_apple[reset_ids].float().mean(),
                    "Terminations/below_table": self._termination_below_table[reset_ids].float().mean(),
                    "Terminations/workspace_exit": self._termination_workspace_exit[reset_ids].float().mean(),
                    "Terminations/nonfinite": self._termination_nonfinite[reset_ids].float().mean(),
                    "Terminations/timeout": self.reset_time_outs[reset_ids].float().mean(),
                }
            )
            if self._mjw_data is not None:
                log["Contact/penetration_hand_ep_max"] = self._episode_max_hand_penetration[reset_ids].mean()
                log["Contact/penetration_self_ep_max"] = self._episode_max_self_penetration[reset_ids].mean()
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
        """Terminate on torso-to-apple contact, workspace exit, or a non-finite simulation state.

        A single bad environment (observed as MJWarp state going NaN with no prior overflow warning)
        would otherwise poison the shared policy observation tensor and kill a multi-hour run through
        RSL-RL's ``check_nan``.  Resetting the offending environments here is what actually cleans the
        state; nothing upstream is allowed to paper over it with ``nan_to_num``.
        """
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

        joint_pos = self.robot.data.joint_pos.torch
        joint_vel = self.robot.data.joint_vel.torch
        object_rotation = self.apple.data.root_quat_w.torch
        object_lin_vel = self.apple.data.root_lin_vel_w.torch
        object_ang_vel = self.apple.data.root_ang_vel_w.torch
        arm_or_hand_joints_nonfinite = ~torch.isfinite(joint_pos).all(dim=-1) | ~torch.isfinite(joint_vel).all(dim=-1)
        apple_pose_nonfinite = ~torch.isfinite(object_position).all(dim=-1) | ~torch.isfinite(object_rotation).all(
            dim=-1
        )
        apple_vel_nonfinite = ~torch.isfinite(object_lin_vel).all(dim=-1) | ~torch.isfinite(object_ang_vel).all(
            dim=-1
        )
        nonfinite = arm_or_hand_joints_nonfinite | apple_pose_nonfinite | apple_vel_nonfinite

        self._termination_torso_apple = torso_contact
        self._termination_below_table = object_below_table
        self._termination_workspace_exit = object_too_far
        self._termination_nonfinite = nonfinite
        terminated = torso_contact | object_below_table | object_too_far | nonfinite
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if nonfinite.any() and self._nonfinite_diag_count < 50:
            self._nonfinite_diag_count += 1
            bad_envs = nonfinite.nonzero(as_tuple=False).squeeze(-1)
            bad_groups = [
                name
                for name, group_nonfinite in (
                    ("arm_or_hand_joints", arm_or_hand_joints_nonfinite),
                    ("apple_pose", apple_pose_nonfinite),
                    ("apple_vel", apple_vel_nonfinite),
                )
                if group_nonfinite.any()
            ]
            print(
                f"[nonfinite-guard] step={self.common_step_counter} bad_envs={bad_envs.numel()} "
                f"groups={bad_groups} prev_apple_speed_max={self._prev_apple_speed[bad_envs].max().item():.4f} "
                f"prev_apple_height_max={self._prev_apple_height[bad_envs].max().item():.4f}"
            )

        # Cache the last known-finite apple speed and height for the diagnostic above; a non-finite
        # environment keeps whatever it last had until its reset overwrites the underlying sim state.
        apple_speed = torch.linalg.vector_norm(object_lin_vel, dim=-1)
        finite_speed = torch.isfinite(apple_speed)
        self._prev_apple_speed = torch.where(finite_speed, apple_speed, self._prev_apple_speed)
        finite_height = torch.isfinite(object_position[:, 2])
        self._prev_apple_height = torch.where(finite_height, object_position[:, 2], self._prev_apple_height)

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
            self._episode_max_hand_penetration[env_ids] = 0.0
            self._episode_max_self_penetration[env_ids] = 0.0
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
