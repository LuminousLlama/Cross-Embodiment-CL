# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal DirectRLEnv used solely to inspect the G1-Wuji table scene."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.events import randomize_rigid_body_mass, randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import Camera, CameraCfg, ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    scale_transform,
    unscale_transform,
)
from isaaclab.utils.warp import ProxyArray

from Cross_Embodiment_CL.models import WujiLatentActionPipeline

from .adr import AdaptiveDomainRandomization
from .depth_camera import normalize_depth, randomize_depth_measurement, resize_and_pad_depth, warp_depth_intrinsics
from .env_cfg import STUDENT_DEPTH_LETTERBOX, G1WujiTableEnvCfg
from .virtual_force import (
    VirtualForceOutput,
    VirtualForcePipeline,
    VirtualForcePipelineConfig,
    WujiForceSystemIdModel,
)


def contact_term(forces: torch.Tensor, mode: str, threshold: float, reference: float) -> torch.Tensor:
    """Per-group dense contact term for the shaped reward, averaged over the group dimension.

    ``mode="force"`` grades by force, ``tanh(forces / reference)``, so squeezing harder pays more
    until it saturates.  ``mode="binary"`` is a per-group contact indicator, ``forces >
    threshold``, so the policy is paid for how many groups touch, never for how hard it squeezes.
    """
    if mode == "force":
        return torch.tanh(forces / reference).mean(dim=1)
    if mode == "binary":
        return (forces > threshold).float().mean(dim=1)
    raise ValueError(f"Unknown contact_reward_mode '{mode}'; expected 'force' or 'binary'.")


def sample_spawn_offsets(
    n: int, strength: float, box_x: float, box_y: float, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-env apple spawn offsets [m] from the ADR box's far corner toward its near corner.

    ``dx`` and ``dy`` are independent ``U(-box_x * strength, 0)`` / ``U(-box_y * strength, 0)`` draws, so
    ``strength=0`` returns zeros and the authored apple pose stays the box's far corner.
    """
    dx = -torch.rand(n, device=device) * box_x * strength
    dy = -torch.rand(n, device=device) * box_y * strength
    return dx, dy


def apply_nominal_spawn_mask(
    dx: torch.Tensor, dy: torch.Tensor, prob: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero ``dx``/``dy`` for envs that keep the exact authored apple pose (zero spawn offset).

    Each entry independently keeps the authored pose with probability ``prob``; ``prob=0`` returns
    ``dx``/``dy`` unchanged without drawing any random numbers, so the default is byte-identical.
    """
    if prob == 0.0:
        return dx, dy
    mask = torch.rand(dx.shape[0], device=dx.device, generator=generator) < prob
    return dx.masked_fill(mask, 0.0), dy.masked_fill(mask, 0.0)


def scaled_uniform(
    size: int | tuple[int, ...],
    half_width: float,
    strength: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample ``U(-half_width * strength, half_width * strength)``; ``strength=0`` returns zeros."""
    width = half_width * strength
    return (torch.rand(size, device=device) * 2.0 - 1.0) * width


def sample_latency_steps(
    n: int, max_steps: int, strength: float, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Sample per-env action-delay steps [policy steps].

    ``floor(U(0, max_steps * strength + 1))``, clamped to ``[0, round(max_steps * strength)]``, so
    ``strength=0`` always returns zero delay and ``strength=1`` returns delays up to ``max_steps``.
    """
    cap = round(max_steps * strength)
    delay = torch.floor(torch.rand(n, device=device) * (max_steps * strength + 1.0))
    return delay.clamp_(0, cap).long()


class ActionDelayBuffer:
    """Ring buffer of the last ``capacity`` per-env actions, returning the action from ``delay`` steps ago."""

    def __init__(self, capacity: int, num_envs: int, action_dim: int, device: torch.device | str) -> None:
        self.capacity = capacity
        self._buffer = torch.zeros((capacity, num_envs, action_dim), device=device)
        self._step = 0

    def push(self, actions: torch.Tensor) -> None:
        """Record this step's actions and advance the ring pointer."""
        self._buffer[self._step % self.capacity] = actions
        self._step += 1

    def get(self, delay: torch.Tensor) -> torch.Tensor:
        """Return each env's action from ``delay`` steps ago; ``delay=0`` is the most recent push."""
        row = (self._step - 1 - delay) % self.capacity
        return self._buffer[row, torch.arange(self._buffer.shape[1], device=self._buffer.device)]

    def reset(self, env_ids: torch.Tensor) -> None:
        """Zero-fill the given envs' history, e.g. on episode reset."""
        self._buffer[:, env_ids] = 0.0


class PendingPhysicsRandomization:
    """Batches per-env Newton physics-model writes (friction, mass) across resets.

    Writing friction/mass on every ``_reset_idx`` call registers a Newton model-change flag that
    ``NewtonManager.step`` notifies on the next physics step; that notification recomputes MJWarp
    model arrays across every world, not just the resetting envs. With 2048 envs and 480-step
    episodes, resets land on almost every step, so batching all pending envs into one write every
    ``update_every_steps`` env steps turns that near-per-step cost into a once-per-rollout cost.
    """

    def __init__(self, num_envs: int, update_every_steps: int, device: torch.device | str) -> None:
        self.update_every_steps = update_every_steps
        self._pending = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def mark(self, env_ids: torch.Tensor) -> None:
        """Flag the given envs as due for their next batched physics-parameter write."""
        self._pending[env_ids] = True

    def due(self, step: int) -> bool:
        """Whether ``step`` is a batch-write step."""
        return step % self.update_every_steps == 0

    def take(self) -> torch.Tensor:
        """Return the pending env ids and clear them."""
        env_ids = self._pending.nonzero(as_tuple=False).squeeze(-1)
        self._pending[env_ids] = False
        return env_ids


class _StudentDepthCamera(Camera):
    """Camera whose metric-depth buffer can be postprocessed before any consumer reads it."""

    def __init__(self, cfg: CameraCfg) -> None:
        self.depth_postprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None
        super().__init__(cfg)

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        super()._update_buffers_impl(env_mask)
        if self.depth_postprocessor is None:
            return
        depth = self._data.output["distance_to_image_plane"].torch
        randomized = self.depth_postprocessor(depth.permute(0, 3, 1, 2))
        depth.copy_(randomized.permute(0, 2, 3, 1))


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
    # PhysX requires every filtered-contact expression to resolve to one body per environment.
    # Keep this list explicit instead of using ``G1Wuji/.*`` so the same configuration also works
    # with Newton's many-to-many contact implementation.  These are all active rigid bodies in the
    # fixed-base G1 and Wuji layers, including non-colliding tip frames (which simply report zero).
    _G1_BODY_NAMES = (
        "pelvis",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    )
    _WUJI_BODY_NAMES = (
        "right_palm_link",
        "right_finger1_link1",
        "right_finger1_link2",
        "right_finger1_link2_softbody",
        "right_finger1_link3",
        "right_finger1_link4",
        "right_finger1_tip_link",
        *tuple(
            f"right_finger{finger}_{suffix}"
            for finger in range(2, 6)
            for suffix in ("link1", "link2", "link3", "link4", "tip_link")
        ),
    )
    _THUMB_CONTACT_GROUP = "finger1"
    _OBSERVATION_DIM = 171
    # The deployable proprioceptive prefix of the privileged observation: joint positions and
    # velocities, commanded arm and hand targets, and their position limits.
    _STUDENT_OBSERVATION_DIM = 141

    def __init__(self, cfg: G1WujiTableEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        self._student_depth_preview: torch.Tensor | None = None
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
        self._virtual_force_sensor_body_ids: dict[str, torch.Tensor] = {}
        if self.cfg.virtual_force.enabled:
            for group_name, sensor in self.contact_sensors.items():
                sensor_names = getattr(sensor, "sensor_names", None)
                if sensor_names is None:
                    sensor_names = sensor.body_names
                if sensor_names is None or len(sensor_names) != sensor.num_sensors:
                    raise ValueError(f"Virtual force sensor '{group_name}' did not expose its sensing body names.")
                self._virtual_force_sensor_body_ids[group_name] = torch.as_tensor(
                    [self.robot.body_names.index(name) for name in sensor_names],
                    dtype=torch.long,
                    device=self.device,
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
        self.virtual_force_pipeline: VirtualForcePipeline | None = None
        self.virtual_force_output: VirtualForceOutput | None = None
        if self.cfg.virtual_force.enabled:
            self._initialize_virtual_force_pipeline()
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
        if self.cfg.contact_reward_mode not in ("force", "binary"):
            raise ValueError(
                f"contact_reward_mode must be 'force' or 'binary', got '{self.cfg.contact_reward_mode}'."
            )
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.arm_joint_targets = torch.zeros((self.num_envs, len(self.arm_joint_ids)), device=self.device)
        self.wuji_joint_targets = torch.zeros((self.num_envs, len(self.wuji_joint_ids)), device=self.device)
        # Targets at the start of the current policy step, which _apply_action interpolates from.
        self._arm_joint_targets_start = torch.zeros_like(self.arm_joint_targets)
        self._wuji_joint_targets_start = torch.zeros_like(self.wuji_joint_targets)
        self._arm_anti_windup_active = torch.zeros_like(self.arm_joint_targets, dtype=torch.bool)
        self._action_substep = 0
        self.waist_joint_targets = torch.zeros((self.num_envs, len(self.waist_joint_ids)), device=self.device)
        self.goal_position = torch.tensor(self.cfg.goal_position, device=self.device).repeat(self.num_envs, 1)
        self.goal_rotation = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device).repeat(self.num_envs, 1)
        self.object_start_position = torch.tensor(self.cfg.apple_cfg.init_state.pos, device=self.device).repeat(
            self.num_envs, 1
        )
        self.adr: AdaptiveDomainRandomization | None = None
        if self.cfg.adr_enabled:
            self.adr = AdaptiveDomainRandomization(
                max_level=self.cfg.adr_max_level,
                success_threshold=self.cfg.adr_success_threshold,
                level=self.cfg.adr_initial_level,
            )
            self._adr_successful_episodes = 0
            self._adr_completed_episodes = 0
        self._gravity_frac = 1.0
        self._last_applied_gravity_frac: float | None = None
        self._apply_gravity_curriculum(force=True)
        self._adr_extra_active = bool(self.cfg.adr_enabled and self.cfg.adr_extra_enabled)
        if self._adr_extra_active:
            num_joints = self.robot.data.joint_pos.torch.shape[-1]
            self._adr_joint_pos_obs_bias = torch.zeros((self.num_envs, num_joints), device=self.device)
            self._adr_joint_vel_obs_bias = torch.zeros_like(self._adr_joint_pos_obs_bias)
            self._adr_object_pos_obs_bias = torch.zeros((self.num_envs, 3), device=self.device)
            self._adr_hand_target_scale = torch.ones(self.num_envs, device=self.device)
            self._adr_action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._adr_action_buffer = ActionDelayBuffer(
                capacity=self.cfg.adr_action_latency_max_steps + 1,
                num_envs=self.num_envs,
                action_dim=self.cfg.action_space,
                device=self.device,
            )
            self._adr_friction_terms = self._make_adr_friction_terms()
            self._adr_mass_asset_cfg = SceneEntityCfg("apple")
            self._adr_mass_term = randomize_rigid_body_mass(
                EventTermCfg(
                    func=randomize_rigid_body_mass,
                    mode="reset",
                    params={"asset_cfg": self._adr_mass_asset_cfg, "operation": "scale"},
                ),
                self,
            )
            self._adr_physics_pending = PendingPhysicsRandomization(
                self.num_envs, self.cfg.adr_physics_update_every_steps, device=self.device
            )
        self._adr_camera_active = bool(self._adr_extra_active and self.depth_camera is not None)
        if self._adr_camera_active:
            nonnegative_camera_cfg = {
                "adr_camera_position_range": self.cfg.adr_camera_position_range,
                "adr_camera_rotation_range_deg": self.cfg.adr_camera_rotation_range_deg,
                "adr_camera_principal_point_offset": self.cfg.adr_camera_principal_point_offset,
                "adr_depth_bias": self.cfg.adr_depth_bias,
                "adr_depth_noise_std_at_1m": self.cfg.adr_depth_noise_std_at_1m,
                "adr_depth_boundary_threshold": self.cfg.adr_depth_boundary_threshold,
            }
            invalid = {name: value for name, value in nonnegative_camera_cfg.items() if value < 0.0}
            if invalid:
                raise ValueError(f"Camera ADR ranges must be non-negative, received {invalid}.")
            if not 0.0 <= self.cfg.adr_depth_boundary_corruption_prob <= 1.0:
                raise ValueError(
                    "adr_depth_boundary_corruption_prob must be in [0, 1], received "
                    f"{self.cfg.adr_depth_boundary_corruption_prob}."
                )
            for name, half_width in (
                ("adr_camera_focal_scale", self.cfg.adr_camera_focal_scale),
                ("adr_depth_scale", self.cfg.adr_depth_scale),
            ):
                if not 0.0 <= half_width < 1.0:
                    raise ValueError(f"{name} must be in [0, 1), received {half_width}.")

            camera_data = self.depth_camera.data
            self._adr_camera_nominal_pos_w = camera_data.pos_w.torch.clone()
            self._adr_camera_nominal_quat_w = camera_data.quat_w_world.torch.clone()
            self._adr_camera_focal_scale = torch.ones(self.num_envs, device=self.device)
            self._adr_camera_principal_point_offset = torch.zeros((self.num_envs, 2), device=self.device)
            self._adr_depth_scale = torch.ones(self.num_envs, device=self.device)
            self._adr_depth_bias = torch.zeros(self.num_envs, device=self.device)
            self.depth_camera.depth_postprocessor = self._randomize_depth_camera_frame
        self.table_top_height = self.cfg.table_cfg.init_state.pos[2] + 0.5 * self.cfg.table_cfg.spawn.size[2]
        self.local_cube_keypoints = self._make_cube_keypoints(self.cfg.keypoint_extent)
        self._init_episode_metrics()
        self._init_penetration_probe()
        self.goal_keypoint_marker: VisualizationMarkers | None = None
        self.object_keypoint_marker: VisualizationMarkers | None = None
        self.adr_spawn_area_marker: VisualizationMarkers | None = None
        if self.cfg.debug.keypoint_markers:
            self.goal_keypoint_marker = VisualizationMarkers(self.cfg.goal_keypoint_marker_cfg)
            self.object_keypoint_marker = VisualizationMarkers(self.cfg.object_keypoint_marker_cfg)
            self._update_keypoint_markers()
        if self.cfg.debug.adr_spawn_area_marker or self.cfg.adr_debug_spawn_area_vis:
            self.adr_spawn_area_marker = VisualizationMarkers(self.cfg.adr_spawn_area_marker_cfg)
            marker_thickness = self.cfg.adr_spawn_area_marker_cfg.markers["area"].size[2]
            apple_x, apple_y = self.cfg.apple_cfg.init_state.pos[:2]
            marker_center = torch.tensor(
                (
                    apple_x - 0.5 * self.cfg.adr_spawn_box_x,
                    apple_y - 0.5 * self.cfg.adr_spawn_box_y,
                    self.table_top_height + 0.5 * marker_thickness,
                ),
                device=self.device,
            )
            marker_positions = self.scene.env_origins + marker_center
            marker_scales = marker_positions.new_tensor(
                (self.cfg.adr_spawn_box_x, self.cfg.adr_spawn_box_y, 1.0)
            ).expand(self.num_envs, -1)
            self.adr_spawn_area_marker.visualize(
                translations=marker_positions,
                scales=marker_scales,
                environment_ids=torch.arange(self.num_envs, device=self.device),
            )

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.apple = RigidObject(self.cfg.apple_cfg)
        # One multi-body sensor per group; its force matrix is (envs, bodies, 1 apple, 3).
        self.contact_sensors: dict[str, ContactSensor] = {}
        contact_sensor_template = self.cfg.contact_sensor_cfg
        if self.cfg.virtual_force.enabled:
            if self.cfg.virtual_force.max_contact_data_count_per_prim < 1:
                raise ValueError("virtual_force.max_contact_data_count_per_prim must be positive.")
            # Keep the apple first: reward and termination logic retain filter index zero. The force
            # pipeline sums all filters so that table and robot self-contact are observable too.
            contact_sensor_template = contact_sensor_template.replace(
                filter_prim_paths_expr=[
                    "/World/envs/env_[^/]+/Apple",
                    "/World/envs/env_[^/]+/Table",
                    *(f"/World/envs/env_[^/]+/G1Wuji/g1_simplified/{name}" for name in self._G1_BODY_NAMES),
                    *(f"/World/envs/env_[^/]+/G1Wuji/wujihand/{name}" for name in self._WUJI_BODY_NAMES),
                ],
                track_contact_points=True,
                track_friction_forces=True,
                max_contact_data_count_per_prim=self.cfg.virtual_force.max_contact_data_count_per_prim,
            )
        for group_name, body_names in self._CONTACT_BODY_GROUPS.items():
            sensor_cfg = contact_sensor_template.replace(
                prim_path=f"/World/envs/env_[^/]+/G1Wuji/wujihand/({'|'.join(body_names)})"
            )
            self.contact_sensors[group_name] = ContactSensor(sensor_cfg)
        self.torso_contact_sensor = ContactSensor(self.cfg.torso_contact_sensor_cfg)
        # The student's head depth camera exists only when a preset configures one (presets=distill).
        self.depth_camera = _StudentDepthCamera(self.cfg.depth_camera) if self.cfg.depth_camera is not None else None

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

    def _make_adr_friction_terms(self) -> dict[str, tuple[randomize_rigid_body_material, float]]:
        """Instantiate one ``randomize_rigid_body_material`` term per friction-randomized asset.

        Newton-only, like :meth:`_init_penetration_probe`: on any other backend this returns an
        empty dict and friction stays unrandomized. The Newton implementation caches its friction
        range at construction and ignores the range passed to its ``__call__``, so
        :meth:`_apply_adr_friction` writes its shape ``mu`` binding directly instead of calling it.
        """
        if "newton" not in self.sim.physics_manager.__name__.lower():
            return {}
        hand_asset_cfg = SceneEntityCfg("robot", body_names=["right_palm_link", "right_finger.*"])
        hand_asset_cfg.resolve(self.scene)
        terms: dict[str, tuple[randomize_rigid_body_material, float]] = {}
        asset_cfgs = (("apple", SceneEntityCfg("apple")), ("table", SceneEntityCfg("table")), ("hand", hand_asset_cfg))
        for name, asset_cfg in asset_cfgs:
            term_cfg = EventTermCfg(func=randomize_rigid_body_material, mode="reset", params={"asset_cfg": asset_cfg})
            term = randomize_rigid_body_material(term_cfg, self)
            nominal = float(wp.to_torch(term._impl._friction_binding)[0, term._impl._shape_indices[0]].item())
            terms[name] = (term, nominal)
        return terms

    def _apply_adr_friction(self, env_ids: torch.Tensor, strength: float) -> None:
        """Blend each Newton-backed asset's shape friction from nominal toward a sampled value.

        ``mu = nominal + strength * (U(adr_friction_range) - nominal)``; static equals dynamic, since
        Newton's ``shape_material_mu`` is a single coefficient.
        """
        low, high = self.cfg.adr_friction_range
        for term, nominal in self._adr_friction_terms.values():
            impl = term._impl
            friction_view = wp.to_torch(impl._friction_binding)
            shape_idx = impl._shape_indices.to(self.device)
            sample = torch.rand(len(env_ids), len(shape_idx), device=self.device) * (high - low) + low
            friction_view[env_ids[:, None], shape_idx] = nominal + strength * (sample - nominal)
            impl._newton_manager.add_model_change(impl._notify_shape_properties)

    def _apply_adr_mass(self, env_ids: torch.Tensor, strength: float) -> None:
        """Scale the apple's mass relative to its default mass, ``U(1 - m*strength, 1 + m*strength)``."""
        half_width = self.cfg.adr_object_mass_scale * strength
        self._adr_mass_term(
            self,
            env_ids,
            asset_cfg=self._adr_mass_asset_cfg,
            mass_distribution_params=(1.0 - half_width, 1.0 + half_width),
            operation="scale",
            recompute_inertia=True,
        )

    def _randomize_depth_camera(self, env_ids: torch.Tensor, strength: float) -> None:
        """Sample fixed-per-episode student-camera extrinsics, intrinsics, and calibration."""
        count = len(env_ids)
        if self.cfg.adr_camera_position_enabled or self.cfg.adr_camera_rotation_enabled:
            position_range = self.cfg.adr_camera_position_range if self.cfg.adr_camera_position_enabled else 0.0
            rotation_range = (
                math.radians(self.cfg.adr_camera_rotation_range_deg)
                if self.cfg.adr_camera_rotation_enabled
                else 0.0
            )
            translation = scaled_uniform((count, 3), position_range, strength, device=self.device)
            rotation = scaled_uniform((count, 3), rotation_range, strength, device=self.device)
            delta_quat = quat_from_euler_xyz(rotation[:, 0], rotation[:, 1], rotation[:, 2])
            camera_pos_w, camera_quat_w = combine_frame_transforms(
                self._adr_camera_nominal_pos_w[env_ids],
                self._adr_camera_nominal_quat_w[env_ids],
                translation,
                delta_quat,
            )
            self.depth_camera.set_world_poses(
                positions=camera_pos_w,
                orientations=camera_quat_w,
                env_ids=env_ids,
                convention="world",
            )

        self._adr_camera_focal_scale[env_ids] = 1.0 + scaled_uniform(
            count,
            self.cfg.adr_camera_focal_scale if self.cfg.adr_camera_focal_enabled else 0.0,
            strength,
            device=self.device,
        )
        self._adr_camera_principal_point_offset[env_ids] = scaled_uniform(
            (count, 2),
            self.cfg.adr_camera_principal_point_offset if self.cfg.adr_camera_principal_point_enabled else 0.0,
            strength,
            device=self.device,
        )
        self._adr_depth_scale[env_ids] = 1.0 + scaled_uniform(
            count,
            self.cfg.adr_depth_scale if self.cfg.adr_depth_scale_enabled else 0.0,
            strength,
            device=self.device,
        )
        self._adr_depth_bias[env_ids] = scaled_uniform(
            count,
            self.cfg.adr_depth_bias if self.cfg.adr_depth_bias_enabled else 0.0,
            strength,
            device=self.device,
        )

    def _randomize_depth_camera_frame(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply the sampled student-only intrinsic and depth-measurement DR to one rendered frame."""
        if self.cfg.adr_camera_focal_enabled or self.cfg.adr_camera_principal_point_enabled:
            depth = warp_depth_intrinsics(
                depth,
                self._adr_camera_focal_scale,
                self._adr_camera_principal_point_offset,
            )
        measurement_enabled = (
            self.cfg.adr_depth_scale_enabled
            or self.cfg.adr_depth_bias_enabled
            or self.cfg.adr_depth_pixel_noise_enabled
            or self.cfg.adr_depth_boundary_corruption_enabled
        )
        if not measurement_enabled:
            return depth
        return randomize_depth_measurement(
            depth,
            self._adr_depth_scale,
            self._adr_depth_bias,
            noise_std_at_1m_m=(
                self.cfg.adr_depth_noise_std_at_1m if self.cfg.adr_depth_pixel_noise_enabled else 0.0
            ),
            boundary_corruption_prob=(
                self.cfg.adr_depth_boundary_corruption_prob
                if self.cfg.adr_depth_boundary_corruption_enabled
                else 0.0
            ),
            boundary_threshold_m=self.cfg.adr_depth_boundary_threshold,
            strength=self.adr.strength,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Map normalized arm positions and Wuji latent actions onto speed-capped physical joint targets."""
        self.actions[:] = torch.clamp(actions, -1.0, 1.0)
        applied_actions = self.actions
        if self._adr_extra_active:
            # The delayed action feeds the joint-target computation below; other uses of
            # self.actions (e.g. the saturation-fraction log) keep reading the undelayed action.
            self._adr_action_buffer.push(self.actions)
            applied_actions = self._adr_action_buffer.get(self._adr_action_delay_steps)
        self._arm_joint_targets_start.copy_(self.arm_joint_targets)
        self._wuji_joint_targets_start.copy_(self.wuji_joint_targets)
        self._action_substep = 0
        arm_limits = self.robot.data.soft_joint_pos_limits.torch[:, self.arm_joint_ids]
        arm_lower, arm_upper = arm_limits[..., 0], arm_limits[..., 1]
        arm_actions = applied_actions[:, : len(self.arm_joint_ids)]
        arm_targets = unscale_transform(arm_actions, arm_lower, arm_upper)
        self._arm_anti_windup_active.copy_(
            self._advance_joint_targets(
                self.arm_joint_targets,
                arm_targets,
                self.arm_joint_ids,
                self.cfg.arm_action_ema_alpha,
                self.cfg.arm_joint_velocity_limit,
            )
        )

        wuji_limits = self.robot.data.soft_joint_pos_limits.torch[:, self.wuji_joint_ids]
        wuji_command_lower = self._wuji_command_lower_limits(wuji_limits)
        wuji_latent_action = applied_actions[:, len(self.arm_joint_ids) :]
        if self._adr_extra_active:
            # Apply the per-env hand-target scale to the decoded target before the joint-limit clamp
            # that latent_action_to_joint_target would otherwise apply first.
            raw_wuji_targets = self.wuji_action_pipeline.retarget_mano_pose(
                self.wuji_action_pipeline.decode_latent_action(wuji_latent_action)
            )
            wuji_targets = torch.clamp(
                raw_wuji_targets * self._adr_hand_target_scale.unsqueeze(-1), wuji_command_lower, wuji_limits[..., 1]
            )
        else:
            wuji_targets = self.wuji_action_pipeline.latent_action_to_joint_target(
                wuji_latent_action, wuji_command_lower, wuji_limits[..., 1]
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

    def _wuji_command_lower_limits(self, wuji_limits: torch.Tensor) -> torch.Tensor:
        """Return the effective Wuji command lower limits [rad]."""
        return torch.maximum(wuji_limits[..., 0], self._wuji_command_lower_floor)

    def _apply_gravity_curriculum(self, force: bool = False) -> None:
        """Ramp the whole scene's configured gravity from ``gravity_curriculum_start`` to full strength.

        The update is sent only when the fraction changes by at least 0.005, avoiding a model-property
        notification every policy step while keeping the 600-iteration ramp smooth.  Isaac Lab's current
        Newton, OvPhysX, and Isaac Sim PhysX managers expose different runtime gravity APIs, so this selects
        their capability rather than changing task dynamics by backend name.  When :attr:`G1WujiTableEnvCfg.
        adr_enabled` and :attr:`~.G1WujiTableEnvCfg.adr_drives_gravity` are both set, the ADR schedule's
        strength replaces the step-based ramp.
        """
        if self.cfg.adr_enabled and self.cfg.adr_drives_gravity:
            self._gravity_frac = self.adr.strength
        else:
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
    ) -> torch.Tensor:
        """Advance targets and return where the measured-position anti-windup clamp activated.

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
        lower, upper = joint_pos - max_lead, joint_pos + max_lead
        anti_windup_active = (targets < lower) | (targets > upper)
        targets.clamp_(min=lower, max=upper)
        return anti_windup_active

    def _arm_control_metrics(self) -> dict[str, torch.Tensor]:
        """Return aggregate and per-joint arm command/drive diagnostics averaged over environments."""
        joint_ids = self.arm_joint_ids
        data = self.robot.data
        target_rate = torch.abs(self.arm_joint_targets - self._arm_joint_targets_start) / self.step_dt
        tracking_error = torch.abs(self.arm_joint_targets - data.joint_pos.torch[:, joint_ids])
        joint_velocity = torch.abs(data.joint_vel.torch[:, joint_ids])
        computed_effort = torch.abs(self.robot.actuators.computed_effort.torch[:, joint_ids])
        applied_effort = torch.abs(self.robot.actuators.applied_effort.torch[:, joint_ids])
        effort_limits = data.joint_effort_limits.torch[:, joint_ids]
        effort_saturation = computed_effort >= effort_limits - 1.0e-6

        log = {
            "Control/arm_target_rate_step": target_rate.mean(),
            "Control/arm_joint_velocity_step": joint_velocity.mean(),
            "Control/arm_computed_effort_step": computed_effort.mean(),
            "Control/arm_applied_effort_step": applied_effort.mean(),
            "Control/arm_effort_saturation_frac_step": effort_saturation.float().mean(),
            "Control/arm_anti_windup_frac_step": self._arm_anti_windup_active.float().mean(),
        }
        for index, joint_name in enumerate(self._ARM_JOINT_NAMES):
            log.update(
                {
                    f"Control/arm_target_rate_{joint_name}_step": target_rate[:, index].mean(),
                    f"Control/arm_tracking_error_{joint_name}_step": tracking_error[:, index].mean(),
                    f"Control/arm_joint_velocity_{joint_name}_step": joint_velocity[:, index].mean(),
                    f"Control/arm_computed_effort_{joint_name}_step": computed_effort[:, index].mean(),
                    f"Control/arm_applied_effort_{joint_name}_step": applied_effort[:, index].mean(),
                    f"Control/arm_effort_saturation_{joint_name}_frac_step": (
                        effort_saturation[:, index].float().mean()
                    ),
                    f"Control/arm_anti_windup_{joint_name}_frac_step": (
                        self._arm_anti_windup_active[:, index].float().mean()
                    ),
                }
            )
        return log

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

    def _initialize_virtual_force_pipeline(self) -> None:
        """Create the Wuji virtual force model after articulation ordering is known."""
        cfg = self.cfg.virtual_force
        pipeline_cfg = VirtualForcePipelineConfig(
            torque_scale_range=cfg.torque_scale_range,
            torque_bias_range_nm=cfg.torque_bias_range_nm,
            torque_noise_std_nm=cfg.torque_noise_std_nm,
            torque_lpf_alpha=cfg.torque_lpf_alpha,
            latency_steps_range=cfg.latency_steps_range,
            packet_dropout_probability=cfg.packet_dropout_probability,
            torque_clip_abs_nm=cfg.torque_clip_abs_nm,
            torque_deadband_nm=cfg.torque_deadband_nm,
            contact_force_sign=cfg.contact_force_sign,
            updates_per_step=cfg.updates_per_step,
        )
        system_id_model = None
        if cfg.system_id_model_path is not None:
            system_id_model = WujiForceSystemIdModel(
                cfg.system_id_model_path,
                actuator_joint_names=list(self._WUJI_JOINT_NAMES),
                num_envs=self.num_envs,
                device=self.device,
                sample_residual=cfg.system_id_sample_residual,
                random_seed=cfg.system_id_seed,
            )
        self.virtual_force_pipeline = VirtualForcePipeline(
            num_envs=self.num_envs,
            config=pipeline_cfg,
            device=self.device,
            system_id_model=system_id_model,
        )
        self.virtual_force_output = self.virtual_force_pipeline.output

    def _update_virtual_force_pipeline(self) -> None:
        """Project the latest public contact-sensor data into the 20 Wuji actuators.

        Isaac Lab 3.0 exposes per-counterpart total force and an average contact point on both
        PhysX and Newton. A force applied at that average point preserves a single contact exactly;
        for multiple non-collinear contacts on one body/counterpart pair it approximates their net
        moment because the public API does not expose every application point.
        """
        pipeline = self.virtual_force_pipeline
        if pipeline is None:
            return

        contact_forces: list[torch.Tensor] = []
        contact_moments: list[torch.Tensor] = []
        jacobian_rows: list[torch.Tensor] = []
        body_link_positions = self.robot.data.body_link_pos_w.torch
        body_link_jacobians = self.robot.data.body_link_jacobian_w.torch
        for group_name, sensor in self.contact_sensors.items():
            data = sensor.data
            normal_force = data.normal_force_matrix_w
            friction_force = data.friction_force_matrix_w
            contact_position = data.contact_pos_w
            if normal_force is None or friction_force is None or contact_position is None:
                raise RuntimeError(
                    f"Virtual force sensor '{group_name}' did not provide force, friction, and contact-point data."
                )
            normal_force_torch = normal_force.torch
            friction_force_torch = friction_force.torch
            contact_position_torch = contact_position.torch
            if normal_force_torch.shape != friction_force_torch.shape:
                raise RuntimeError(
                    f"Virtual force sensor '{group_name}' normal/friction shapes differ: "
                    f"{normal_force_torch.shape} and {friction_force_torch.shape}."
                )
            if contact_position_torch.shape != normal_force_torch.shape:
                raise RuntimeError(
                    f"Virtual force sensor '{group_name}' contact-point shape {contact_position_torch.shape} "
                    f"does not match force shape {normal_force_torch.shape}."
                )

            body_ids = self._virtual_force_sensor_body_ids[group_name]
            if normal_force_torch.shape[1] != len(body_ids):
                raise RuntimeError(
                    f"Virtual force sensor '{group_name}' returned {normal_force_torch.shape[1]} bodies, "
                    f"expected {len(body_ids)}."
                )
            force_by_partner = normal_force_torch + friction_force_torch
            body_origins = body_link_positions[:, body_ids].unsqueeze(2)
            moment_arms = torch.nan_to_num(contact_position_torch - body_origins)
            moment_by_partner = torch.linalg.cross(moment_arms, force_by_partner, dim=-1)
            contact_forces.append(force_by_partner.sum(dim=2))
            contact_moments.append(moment_by_partner.sum(dim=2))

            jacobian_body_ids = body_ids - 1
            if bool((jacobian_body_ids < 0).any()):
                raise RuntimeError("Virtual force sensing cannot use the fixed articulation root body.")
            jacobian_rows.append(body_link_jacobians[:, jacobian_body_ids, :, :][:, :, :, self.wuji_joint_ids])

        forces_w = torch.cat(contact_forces, dim=1)
        moments_w = torch.cat(contact_moments, dim=1)
        jacobians_w = torch.cat(jacobian_rows, dim=1)
        self.virtual_force_output = pipeline.step(
            contact_forces_w=forces_w,
            contact_linear_jacobians_w=jacobians_w[:, :, :3],
            contact_moments_w=moments_w,
            contact_angular_jacobians_w=jacobians_w[:, :, 3:],
            joint_position=self.robot.data.joint_pos.torch[:, self.wuji_joint_ids],
            joint_velocity=self.robot.data.joint_vel.torch[:, self.wuji_joint_ids],
            joint_command=self.wuji_joint_targets,
        )

    def get_virtual_force_output(self) -> VirtualForceOutput | None:
        """Return the latest virtual force packet and its diagnostic fields."""
        return self.virtual_force_output

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return the identical 171-D privileged state for policy and critic, plus student proprioception.

        The observation contains raw joint positions [rad], normalized joint velocities and commanded arm/hand
        targets, raw arm and effective Wuji command limits [rad], followed by apple and goal state in the fixed
        robot-base frame and filtered contact forces.  The ``student`` group is the 141-D deployable prefix.  With
        a depth camera configured, ``camera`` adds its normalized depth image with shape ``(N, 1, H, W)``.
        """
        joint_limits = self.robot.data.soft_joint_pos_limits.torch
        joint_position = self.robot.data.joint_pos.torch
        joint_velocity_raw = self.robot.data.joint_vel.torch
        if self._adr_extra_active:
            strength = self.adr.strength
            joint_position = (
                joint_position
                + self._adr_joint_pos_obs_bias
                + torch.randn_like(joint_position) * (self.cfg.adr_joint_pos_obs_noise * strength)
            )
            joint_velocity_raw = (
                joint_velocity_raw
                + self._adr_joint_vel_obs_bias
                + torch.randn_like(joint_velocity_raw) * (self.cfg.adr_joint_vel_obs_noise * strength)
            )
        # Arm and hand speeds are scaled by the task caps rather than the looser solver limits.
        joint_velocity_limits = self.robot.data.soft_joint_vel_limits.torch.clone()
        joint_velocity_limits[:, self.arm_joint_ids] = self.cfg.arm_joint_velocity_limit
        joint_velocity_limits[:, self.wuji_joint_ids] = self.cfg.hand_joint_velocity_limit
        joint_velocity_limits.clamp_min_(1.0e-6)
        joint_velocity = torch.clamp(joint_velocity_raw / joint_velocity_limits, -1.0, 1.0)

        arm_limits = joint_limits[:, self.arm_joint_ids]
        wuji_limits = joint_limits[:, self.wuji_joint_ids]
        wuji_command_limits = torch.stack(
            (self._wuji_command_lower_limits(wuji_limits), wuji_limits[..., 1]), dim=-1
        )
        arm_target = torch.clamp(
            scale_transform(self.arm_joint_targets, arm_limits[..., 0], arm_limits[..., 1]), -1.0, 1.0
        )
        wuji_target = torch.clamp(
            scale_transform(self.wuji_joint_targets, wuji_limits[..., 0], wuji_limits[..., 1]), -1.0, 1.0
        )

        object_position = self.apple.data.root_pos_w.torch
        if self._adr_extra_active:
            # Observation-only noise: reward and success computations use the true apple position.
            object_position = (
                object_position
                + self._adr_object_pos_obs_bias
                + torch.randn_like(object_position) * (self.cfg.adr_object_pos_obs_noise * self.adr.strength)
            )
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

        proprioception = torch.cat(
            (
                joint_position,
                joint_velocity,
                arm_target,
                wuji_target,
                arm_limits.flatten(start_dim=1),
                wuji_command_limits.flatten(start_dim=1),
            ),
            dim=-1,
        )
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
            # (N, H, W, 1) full-width planar depth [m] to the padded (N, 1, 224, 224) student input.
            depth = self.depth_camera.data.output["distance_to_image_plane"].torch
            depth = depth.permute(0, 3, 1, 2)
            depth = resize_and_pad_depth(depth, STUDENT_DEPTH_LETTERBOX)
            student_depth = normalize_depth(depth, self.cfg.student_depth_max_m)
            observations["camera"] = student_depth
            if self.cfg.debug.student_depth_preview:
                self._publish_student_depth_preview(student_depth)
        if self.virtual_force_output is not None:
            observations["force"] = self.virtual_force_output.observed_actuator_torque_nm
        return observations

    def _publish_student_depth_preview(self, student_depth: torch.Tensor) -> None:
        """Expose the exact normalized student image through the camera panel's preferred depth key."""
        preview = student_depth.permute(0, 2, 3, 1)
        if self._student_depth_preview is None:
            self._student_depth_preview = torch.empty_like(preview, memory_format=torch.contiguous_format)
            self.depth_camera.data.output["depth"] = ProxyArray(wp.from_torch(self._student_depth_preview))
        self._student_depth_preview.copy_(preview)

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
            # adr_enabled ramps the sharpness with DR strength instead of the fixed constant.
            goal_alpha = self.cfg.goal_reward_alpha
            goal_alpha_step = 0.0
            if self.cfg.adr_enabled:
                goal_alpha = self.cfg.goal_reward_alpha + (
                    self.cfg.adr_goal_alpha_end - self.cfg.goal_reward_alpha
                ) * self.adr.strength
                goal_alpha_step = goal_alpha
            goal_reward = self.cfg.goal_reward_scale * torch.exp(-goal_alpha * keypoint_error) * contact_gate
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
                * contact_term(
                    contact_force_stack,
                    self.cfg.contact_reward_mode,
                    self.cfg.contact_force_threshold,
                    self.cfg.contact_force_reference,
                )
                * held
            )
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

        # DirectRLEnv computes dones before rewards, so the termination mask already identifies any
        # environment whose non-finite simulator state would otherwise produce a non-finite reward.
        reward = torch.where(self._termination_nonfinite, torch.zeros_like(reward), reward)

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
        if self.cfg.adr_enabled and self.common_step_counter % self.cfg.adr_update_every_steps == 0:
            self.adr.update(self._adr_successful_episodes, self._adr_completed_episodes)
            self._adr_successful_episodes = 0
            self._adr_completed_episodes = 0
        if self._adr_extra_active and self._adr_physics_pending.due(self.common_step_counter):
            pending_ids = self._adr_physics_pending.take()
            if len(pending_ids) > 0:
                strength = self.adr.strength
                self._apply_adr_friction(pending_ids, strength)
                self._apply_adr_mass(pending_ids, strength)
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
        (
            self._hand_geoms,
            self._apple_geoms,
            self._table_geoms,
            self._other_robot_geoms,
            self._right_elbow_geoms,
            self._torso_geoms,
        ) = (
            torch.tensor([key in label for label in labels], device=self.device)
            for key in ("wujihand", "Apple", "Table", "g1_simplified", "right_elbow_link", "torso_link")
        )
        if not (
            self._hand_geoms.any()
            and self._apple_geoms.any()
            and self._table_geoms.any()
            and self._other_robot_geoms.any()
            and self._right_elbow_geoms.any()
            and self._torso_geoms.any()
        ):
            raise ValueError(
                "Penetration probe could not find the hand, apple, table, arm, and torso geoms in the MJWarp model."
            )
        self._mjw_data = solver.mjw_data
        # One-time record of the apple's built collision-shape count, so runs log the real hull
        # count produced by whatever mesh_approximation_name/max_hull_vertices Hydra selected.
        approximation = self.cfg.apple_cfg.spawn.collision_props.mesh_collision_property.mesh_approximation_name
        print(f"[apple-collision] approximation={approximation} shapes={int(self._apple_geoms.sum())}")

    def _contact_penetration(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-environment hand, table, hand-self, and elbow-torso penetration depths [m].

        Self-contact covers hand-hand and hand-to-other-robot-geom (arm/torso) pairs; a pair the solver
        filters out of collision (e.g. palm <-> proximal) never appears in the buffer, so it never
        contributes here.  Every reported depth is zero without contact.
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
        elbow_first, elbow_second = self._right_elbow_geoms[first], self._right_elbow_geoms[second]
        torso_first, torso_second = self._torso_geoms[first], self._torso_geoms[second]
        elbow_torso_pair = (elbow_first & torso_second) | (elbow_second & torso_first)
        penetrations.append(
            torch.zeros(self.num_envs, device=self.device).scatter_reduce_(
                0, worlds, torch.where(elbow_torso_pair, depth, 0.0), reduce="amax"
            )
        )
        return penetrations[0], penetrations[1], penetrations[2], penetrations[3]

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
        self._episode_max_elbow_torso_penetration = torch.zeros(self.num_envs, device=self.device)
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
        if self.cfg.adr_enabled:
            log["Curriculum/adr_level"] = float(self.adr.level)
            log["Curriculum/adr_success_rate"] = self.adr.last_success_rate
            log["Curriculum/adr_extra_active"] = float(self.cfg.adr_extra_enabled)
        log.update(
            {
                f"Contact/touch_frac_{name}_step": (contact_force_stack[:, index] > 0.0).float().mean()
                for index, name in enumerate(self.contact_sensors)
            }
        )
        log.update(self._arm_control_metrics())
        if self._mjw_data is not None:
            # Deepest contact per environment [m], averaged over environments.
            hand_penetration, table_penetration, self_penetration, elbow_torso_penetration = (
                self._contact_penetration()
            )
            self._episode_max_hand_penetration = torch.maximum(self._episode_max_hand_penetration, hand_penetration)
            self._episode_max_self_penetration = torch.maximum(self._episode_max_self_penetration, self_penetration)
            self._episode_max_elbow_torso_penetration = torch.maximum(
                self._episode_max_elbow_torso_penetration, elbow_torso_penetration
            )
            log["Contact/penetration_hand_step"] = hand_penetration.mean()
            log["Contact/penetration_table_step"] = table_penetration.mean()
            log["Contact/penetration_self_step"] = self_penetration.mean()
            log["Contact/self_penetrating_frac_step"] = (self_penetration > 0.0005).float().mean()
            log["Contact/penetration_elbow_torso_step"] = elbow_torso_penetration.mean()
            log["Contact/elbow_torso_penetrating_frac_step"] = (
                elbow_torso_penetration > 0.0005
            ).float().mean()
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
            episode_success = keypoint_error[reset_ids] < self.cfg.success_keypoint_error_threshold
            if self.cfg.adr_enabled:
                self._adr_successful_episodes += int(episode_success.sum().item())
                self._adr_completed_episodes += len(reset_ids)
            log.update(
                {
                    "Task/success": episode_success.float().mean(),
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
                log["Contact/penetration_elbow_torso_ep_max"] = self._episode_max_elbow_torso_penetration[
                    reset_ids
                ].mean()
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
        self._update_virtual_force_pipeline()
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
        """Restore the authored robot, table, and apple poses; ``adr_enabled`` also samples the apple's
        spawn xy within the current-strength ADR box and moves that env's goal xy to match."""
        super()._reset_idx(env_ids)
        if getattr(self, "virtual_force_pipeline", None) is not None:
            self.virtual_force_pipeline.reset(torch.as_tensor(env_ids, dtype=torch.long, device=self.device))
            self.virtual_force_output = self.virtual_force_pipeline.output

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
            self._episode_max_elbow_torso_penetration[env_ids] = 0.0
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
        if self.cfg.adr_enabled:
            # Per-env apple spawn offset from the ADR box's far corner, at the current DR strength; the
            # goal xy tracks the same spawn xy, so the object always starts directly under its goal.
            dx, dy = sample_spawn_offsets(
                len(env_ids),
                self.adr.strength,
                self.cfg.adr_spawn_box_x,
                self.cfg.adr_spawn_box_y,
                device=self.device,
            )
            dx, dy = apply_nominal_spawn_mask(dx, dy, self.cfg.adr_nominal_spawn_prob)
            apple_pose[:, 0] += dx
            apple_pose[:, 1] += dy
            self.object_start_position[env_ids, :2] = apple_pose[:, :2]
            self.goal_position[env_ids, :2] = apple_pose[:, :2]
        apple_pose[:, :3] += self.scene.env_origins[env_ids]
        self.apple.write_root_pose_to_sim_index(root_pose=apple_pose, env_ids=env_ids)
        self.apple.write_root_velocity_to_sim_index(
            root_velocity=self.apple.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        if self._adr_extra_active:
            strength = self.adr.strength
            num_dof = self._adr_joint_pos_obs_bias.shape[-1]
            self._adr_joint_pos_obs_bias[env_ids] = scaled_uniform(
                (len(env_ids), num_dof), self.cfg.adr_joint_pos_obs_bias, strength, device=self.device
            )
            self._adr_joint_vel_obs_bias[env_ids] = scaled_uniform(
                (len(env_ids), num_dof), self.cfg.adr_joint_vel_obs_bias, strength, device=self.device
            )
            self._adr_object_pos_obs_bias[env_ids] = scaled_uniform(
                (len(env_ids), 3), self.cfg.adr_object_pos_obs_bias, strength, device=self.device
            )
            self._adr_hand_target_scale[env_ids] = 1.0 + scaled_uniform(
                len(env_ids), self.cfg.adr_hand_target_scale, strength, device=self.device
            )
            self._adr_action_delay_steps[env_ids] = sample_latency_steps(
                len(env_ids), self.cfg.adr_action_latency_max_steps, strength, device=self.device
            )
            self._adr_action_buffer.reset(env_ids)
            if self._adr_camera_active:
                self._randomize_depth_camera(env_ids, strength)
            # Friction/mass are physics-model writes; batch them (see PendingPhysicsRandomization)
            # instead of writing per reset, to avoid a Newton model-change notification per step.
            self._adr_physics_pending.mark(env_ids)
        self._update_keypoint_markers()
