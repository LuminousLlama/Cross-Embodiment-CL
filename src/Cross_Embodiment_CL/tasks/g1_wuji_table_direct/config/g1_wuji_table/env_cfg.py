# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the visual G1-Wuji table scene."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from pathlib import Path

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonShapeCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg

from isaaclab_tasks.utils import PresetCfg, preset

from .depth_camera import D435_DEPTH_848X480, fit_depth_letterbox

_G1_CONFIG_PATH = Path(__file__).resolve().parents[6] / "assets/g1/g1.py"
# TODO make a objects CFG file
_APPLE_USD_PATH = Path(__file__).resolve().parents[6] / "assets/objects/YcbApple/textured_collision.usda"
# TODO can we not just do from cross embodiment import assets
_g1_config_spec = importlib.util.spec_from_file_location("cross_embodiment_cl_g1_config", _G1_CONFIG_PATH)
if _g1_config_spec is None or _g1_config_spec.loader is None:
    raise ImportError(f"Unable to load G1 configuration from {_G1_CONFIG_PATH}.")
_g1_config = importlib.util.module_from_spec(_g1_config_spec)
_g1_config_spec.loader.exec_module(_g1_config)
G1_WUJI_CFG = _g1_config.G1_WUJI_CFG

_FRICTION = 0.5
"""Static and dynamic friction of the hand, table, and apple (whose USD material authors the same value).

MJWarp resolves a contact's friction as the max of its two shapes and reads only dynamic friction, so friction
is authored with static equal to dynamic and PhysX materials combine by max.
"""
_TABLE_NEAR_EDGE_X = 0.15
"""Table edge nearest the robot [m], retained from the original 0.5 m center and 0.7 m length."""
_TABLE_LENGTH_X = 0.483
"""Table length [m] along +X, looking away from the robot."""
_TABLE_WIDTH_Y = 0.805
"""Table width [m] across the robot."""


def _find_undeclared_config_fields(obj: object, path: str = "env", seen: set[int] | None = None) -> list[str]:
    """Return paths dynamically attached to config dataclasses."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return []
    seen.add(id(obj))

    if hasattr(obj, "__dataclass_fields__"):
        declared = set(obj.__dataclass_fields__)
        unknown = [f"{path}.{name}" for name in vars(obj) if name not in declared]
        for name in declared:
            unknown.extend(_find_undeclared_config_fields(getattr(obj, name), f"{path}.{name}", seen))
        return unknown
    if isinstance(obj, Mapping):
        unknown = []
        for name, value in obj.items():
            unknown.extend(_find_undeclared_config_fields(value, f"{path}.{name}", seen))
        return unknown
    if isinstance(obj, (list, tuple)):
        unknown = []
        for index, value in enumerate(obj):
            unknown.extend(_find_undeclared_config_fields(value, f"{path}[{index}]", seen))
        return unknown
    return []


def _mjwarp_physics_cfg(load_visual_shapes: bool) -> NewtonCfg:
    """Build the MJWarp physics config; the run presets differ only in visual-shape loading.

    Variants are built rather than derived with ``replace``, which forwards the auto-derived
    ``class_type`` that ``NewtonCfg`` rejects.
    """
    return NewtonCfg(
        # MJWarp defaults to explicit Euler and one substep.  This articulated
        # hand scene needs the documented dexterous-manipulation baseline;
        # keep it backend-local so the verified PhysX dynamics are unchanged.
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            # The R007 deterministic and random-action probes peaked at 48
            # rows and 68 contacts per world, respectively. A 128-row budget
            # leaves at least 1.5x headroom and ran 2048 resumed-R007 worlds
            # without an overflow; rows include drives and limits as well as
            # contacts, so this remains independent of nconmax.
            njmax=128,
            nconmax=128,
            iterations=100,
            ls_iterations=50,
            tolerance=1e-6,
            cone="elliptic",
            impratio=10.0,
        ),
        num_substeps=2,
        # Friction of every shape without an authored physics material, which includes the whole hand and G1.
        # MJWarp gives each contact the larger of its two shapes' friction; the PhysX default material in
        # ``SimulationCfg.physics_material`` combines by max to match.
        # ke/kd set the MJWarp contact solref to (2 / kd, kd / 2 * sqrt(1 / ke)) = (0.01 s, 1.0).  The default
        # (0.02 s) let fingers closing at the hand speed cap sink ~3 mm into each other and the palm; 0.01 s
        # measured ~1 mm and stays above twice the 1/240 s substep.  UNTESTED on PhysX, which ignores these.
        default_shape_cfg=NewtonShapeCfg(mu=_FRICTION, ke=1.0e4, kd=200.0),
        # Solver debug mode performs a device-to-host readback after every
        # simulation step. Keep it disabled for training; enable it only for
        # a focused Newton solver investigation.
        debug_mode=False,
        use_cuda_graph=True,
        load_visual_shapes=load_visual_shapes,
    )


@configclass
class G1WujiTablePhysicsCfg(PresetCfg):
    """PhysX and Newton backend presets for the G1-Wuji visual scene."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    # OvPhysX runs the same PhysX solver without Kit, which is the only way to
    # reach PhysX on a machine that has no Isaac Sim installation.
    #
    # The stock GPU buffer capacities are sized for far heavier scenes than one fixed-base
    # arm, one apple and one table.  Measured at 2048 environments, the values below cut
    # PhysX's GPU footprint by ~1.7 GB with byte-identical rollout metrics and no capacity
    # warnings.  Raise them again if this scene ever gains objects.
    ovphysx: OvPhysxCfg = OvPhysxCfg(
        gpu_max_rigid_contact_count=2**20,
        gpu_found_lost_aggregate_pairs_capacity=2**22,
        gpu_collision_stack_size=2**24,
        gpu_total_aggregate_pairs_capacity=2**19,
    )
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    # The YCB asset intentionally separates an invisible collision mesh
    # from its render-only textured mesh.  Always import the latter so
    # either Newton visualizer can display the same apple as Kit/PhysX.
    newton_mjwarp: NewtonCfg = _mjwarp_physics_cfg(load_visual_shapes=True)
    # Headless runs (``presets=train|eval``) never draw the scene, and Newton clones render meshes per
    # environment, so those presets drop them.  Visual shapes do not collide, so dynamics match the default.
    train: NewtonCfg = _mjwarp_physics_cfg(load_visual_shapes=False)
    eval: NewtonCfg = train
    # The student's depth camera must see the apple, whose visible mesh is visual-only.  An explicit
    # False would win over the camera's request for visual shapes, so this preset sets True.
    distill: NewtonCfg = _mjwarp_physics_cfg(load_visual_shapes=True)
    force_distill: NewtonCfg = distill
    default: NewtonCfg = newton_mjwarp


# Run presets live on whole sections, never on a scalar field: Isaac Lab reads ``env.a.b=value`` on a
# preset node as a preset name, so a scalar preset would reject ``env.a.b=True`` as an unknown preset.
@configclass
class G1WujiTableSceneCfg(PresetCfg):
    """Environment-count presets; every other scene setting is shared."""

    default: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    train: InteractiveSceneCfg = default.replace(num_envs=2048)
    debug: InteractiveSceneCfg = default.replace(num_envs=4)
    eval: InteractiveSceneCfg = default.replace(num_envs=16)
    distill: InteractiveSceneCfg = default.replace(num_envs=1024)
    force_distill: InteractiveSceneCfg = distill


@configclass
class G1WujiTableResetCfg:
    """Task-reset controls independent of ADR."""

    object_on_table: bool = False
    """When true, fix the apple root at :attr:`G1WujiTableEnvCfg.object_rest_height` on every reset."""


@configclass
class G1WujiTableDebugCfg:
    """Diagnostic drawing, off for training."""

    keypoint_markers: bool = False
    """Whether to draw the goal (green) and current (red) object-pose keypoint markers."""
    adr_spawn_area_marker: bool = False
    """Whether to draw the configured full-strength ADR spawn area as a visual-only square."""


@configclass
class G1WujiTableDebugPresetCfg(PresetCfg):
    """Diagnostics per run preset: markers whenever a viewer is open, so off for headless ``train`` and ``eval``."""

    default: G1WujiTableDebugCfg = G1WujiTableDebugCfg(keypoint_markers=True, adr_spawn_area_marker=True)
    train: G1WujiTableDebugCfg = G1WujiTableDebugCfg()
    eval: G1WujiTableDebugCfg = train
    distill: G1WujiTableDebugCfg = train
    force_distill: G1WujiTableDebugCfg = train


@configclass
class G1WujiTableDepthPreviewCfg:
    """Student-depth viewer output independent of the debug-marker bundle."""

    enabled: bool = False
    """Whether the camera panel shows the finalized normalized 224x224 student observation."""


@configclass
class G1WujiTableDepthPreviewPresetCfg(PresetCfg):
    """Enable the policy-input preview only for the composable ``depth_view`` overlay."""

    default: G1WujiTableDepthPreviewCfg = G1WujiTableDepthPreviewCfg()
    depth_view: G1WujiTableDepthPreviewCfg = default.replace(enabled=True)


STUDENT_DEPTH_SIZE = 224
"""Side [px] of the student's square depth image."""
STUDENT_DEPTH_LETTERBOX = fit_depth_letterbox(D435_DEPTH_848X480, STUDENT_DEPTH_SIZE)
"""Full-width D435 depth stream resized and padded to the student's square input."""
_STUDENT_DEPTH_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/G1Wuji/g1_simplified/torso_link/d435_link/depth_camera"
"""Prim path of the student's D435 depth imager."""
_STUDENT_DEPTH_CAMERA_STREAM_PATTERN = "/World/envs/env_[^/]+/G1Wuji/g1_simplified/torso_link/d435_link/depth_camera"
"""Resolved camera-prim regex required by the Newton visualizer stream lookup."""


@configclass
class G1WujiTableDepthCameraPresetCfg(PresetCfg):
    """The student's head depth camera: absent by default, so PPO training renders nothing."""

    default: CameraCfg | None = None
    distill: CameraCfg = CameraCfg(
        # The G1 rev 1.0 head D435 frame on torso_link (x forward), at the depth imager's origin.
        prim_path=_STUDENT_DEPTH_CAMERA_PRIM_PATH,
        # Sensors update lazily, so the camera renders once per policy step, when the observation reads it.
        update_period=0.0,
        # Otherwise the camera keeps its spawn pose instead of following the torso.
        update_latest_camera_pose=True,
        # Planar z-depth, as a RealSense reports, rather than distance along the ray.
        data_types=["distance_to_image_plane"],
        width=STUDENT_DEPTH_LETTERBOX.content.width,
        height=STUDENT_DEPTH_LETTERBOX.content.height,
        # Render only the valid full-width content; the observation path pads it to 224x224.
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            STUDENT_DEPTH_LETTERBOX.content.matrix(),
            width=STUDENT_DEPTH_LETTERBOX.content.width,
            height=STUDENT_DEPTH_LETTERBOX.content.height,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(convention="world"),
        renderer_cfg=NewtonWarpRendererCfg(enable_textures=False),
    )
    force_distill: CameraCfg = distill
    depth_view: CameraCfg = distill


@configclass
class G1WujiTableVirtualForceCfg:
    """Virtual Wuji actuator-torque packet used for diagnostics before student integration."""

    enabled: bool = False
    """Whether to collect hand contact wrenches and update the virtual torque model."""
    torque_scale_range: tuple[float, float] = (1.0, 1.0)
    """Episode-constant multiplicative range applied to ideal contact torque."""
    torque_bias_range_nm: tuple[float, float] = (0.0, 0.0)
    """Episode-constant additive actuator-torque bias range [N m]."""
    torque_noise_std_nm: float = 0.0
    """Per-sensor-update actuator-torque noise standard deviation [N m]."""
    torque_lpf_alpha: float = 0.2
    """EMA weight of the current torque sample, matching the deployed estimator."""
    latency_steps_range: tuple[int, int] = (0, 0)
    """Episode-constant packet latency range in virtual 120 Hz sensor updates."""
    packet_dropout_probability: float = 0.0
    """Probability that the latest virtual sensor packet is invalid."""
    torque_clip_abs_nm: float | None = None
    """Optional symmetric actuator-torque clipping magnitude [N m]."""
    torque_deadband_nm: float = 0.1
    """Observed actuator torques at or below this magnitude are reported as zero [N m]."""
    contact_force_sign: float = 1.0
    """Sign converting the common contact-sensor force convention to generalized torque."""
    updates_per_step: int = 2
    """Virtual 120 Hz updates per 60 Hz policy step."""
    max_contact_data_count_per_prim: int = 16
    """Detailed contact capacity per sensing body and environment."""
    system_id_model_path: str | None = None
    """Optional ``wuji_force_model_v1`` NPZ mapping simulated contact torque to real estimator output."""
    system_id_sample_residual: bool = True
    """Whether the fitted system-ID model samples its correlated AR(1) residual."""
    system_id_seed: int = 0
    """Random seed for the system-ID residual process."""


@configclass
class G1WujiTableVirtualForcePresetCfg(PresetCfg):
    """Keep virtual force disabled unless a force-student preset explicitly enables it."""

    default: G1WujiTableVirtualForceCfg = G1WujiTableVirtualForceCfg()
    force_distill: G1WujiTableVirtualForceCfg = default.replace(enabled=True)


@configclass
class G1WujiTableAdrCfg:
    """Adaptive domain-randomization schedule and term configuration."""

    enabled: bool = False
    """Whether the success-gated adaptive domain-randomization schedule is active."""
    drives_gravity: bool = True
    """Whether ADR strength drives whole-scene gravity from ``gravity_start`` to full strength."""
    gravity_start: float = 0.0
    """Whole-scene gravity fraction at ADR strength zero when :attr:`drives_gravity` is enabled."""
    max_level: int = 50
    """Number of levels the adaptive schedule can advance through."""
    success_threshold: float = 0.40
    """Rollout success rate above which the schedule advances one level."""
    initial_level: int = 0
    """DR level at the start of training."""
    update_every_steps: int = 480
    """Env steps between schedule updates; matches one 480-step episode horizon."""
    spawn_box_x: float = 0.11
    """Full-strength apple spawn-box size along x [m]."""
    spawn_box_y: float = 0.20
    """Full-strength apple spawn-box size along y [m]."""
    spawn_enabled: bool = True
    """Whether ADR strength controls centered continuous apple spawn offsets."""
    # Robot-position DR is intentionally unavailable for now. Keep the former
    # configuration here as a record until the feature is reconsidered.
    # robot_position_enabled: bool = False
    # robot_position_range: float = 0.03
    goal_alpha_end: float = 30.0
    """Goal-reward keypoint-error sharpness at full ADR strength."""
    goal_alpha_enabled: bool = True
    """Whether ADR strength increases the shaped goal-reward sharpness."""
    extra_enabled: bool = False
    """Whether the extra observation-noise, latency, hand-scale, friction, and mass terms are active."""
    sensor_noise_enabled: bool = True
    """Whether the extra ADR master enables observation noise and bias."""
    action_latency_enabled: bool = True
    """Whether the extra ADR master enables per-env action latency."""
    hand_target_scale_enabled: bool = True
    """Whether the extra ADR master enables per-env Wuji target scaling."""
    friction_enabled: bool = True
    """Whether the extra ADR master enables friction randomization."""
    mass_enabled: bool = True
    """Whether the extra ADR master enables apple mass randomization."""
    joint_pos_obs_bias: float = 0.01
    """Per-episode joint-position observation bias half-width [rad]."""
    joint_pos_obs_noise: float = 0.003
    """Per-step joint-position observation noise standard deviation [rad]."""
    joint_vel_obs_bias: float = 0.02
    """Per-episode joint-velocity bias half-width [rad/s]."""
    joint_vel_obs_noise: float = 0.03
    """Per-step joint-velocity noise standard deviation [rad/s]."""
    object_pos_obs_bias: float = 0.01
    """Per-episode apple-position observation bias half-width [m]."""
    object_pos_obs_noise: float = 0.005
    """Per-step apple-position observation noise standard deviation [m]."""
    action_latency_max_steps: int = 3
    """Full-strength maximum per-env action delay [policy steps]."""
    hand_target_scale: float = 0.10
    """Full-strength half-width of the per-env multiplicative Wuji hand-target scale."""
    friction_range: tuple[float, float] = (0.1, 0.4)
    """Full-strength friction range for the apple, table, and Wuji hand links."""
    object_mass_scale: float = 0.20
    """Full-strength half-width of the apple's per-env mass scale relative to its default mass."""
    camera_position_enabled: bool = False
    """Whether per-episode camera translation DR is enabled."""
    camera_position_range: float = 0.03
    """Full-strength camera translation half-width [m] along each camera-local axis."""
    camera_rotation_enabled: bool = False
    """Whether per-episode camera roll/pitch/yaw DR is enabled."""
    camera_rotation_range_deg: float = 3.0
    """Full-strength camera roll/pitch/yaw half-width [deg] about its nominal mounting pose."""
    camera_focal_enabled: bool = True
    """Whether per-episode focal-length DR is enabled."""
    camera_focal_scale: float = 0.01
    """Full-strength focal-length scale half-width around one."""
    camera_principal_point_enabled: bool = True
    """Whether per-episode principal-point DR is enabled."""
    camera_principal_point_offset: float = 2.0
    """Full-strength principal-point offset half-width [px] along each content-image axis."""
    depth_scale_enabled: bool = True
    """Whether per-episode metric-depth scale DR is enabled."""
    depth_scale: float = 0.01
    """Full-strength metric-depth scale half-width around one."""
    depth_bias_enabled: bool = True
    """Whether per-episode additive metric-depth bias DR is enabled."""
    depth_bias: float = 0.003
    """Full-strength additive metric-depth bias half-width [m]."""
    depth_pixel_noise_enabled: bool = True
    """Whether independent per-pixel Gaussian depth noise is enabled."""
    depth_noise_std_at_1m: float = 0.004
    """Full-strength per-pixel Gaussian depth-noise standard deviation at 1 m [m]."""
    depth_boundary_corruption_enabled: bool = True
    """Whether depth-discontinuity boundary corruption is enabled."""
    depth_boundary_corruption_prob: float = 0.05
    """Full-strength probability of corrupting a pixel beside a depth discontinuity."""
    depth_edge_dropout_enabled: bool = True
    """Whether a thin zero-depth outline is sampled on the foreground side of depth discontinuities."""
    depth_edge_dropout_prob: float = 0.85
    """Full-strength probability that each pixel in the thin depth-edge outline is invalid."""
    depth_boundary_threshold: float = 0.02
    """Neighboring metric-depth difference [m] that marks a silhouette boundary."""
    physics_update_every_steps: int = 32
    """Env steps between batched physics-model writes."""


@configclass
class G1WujiTableAdrPresetCfg(PresetCfg):
    """ADR presets for nominal, no-DR, and full-DR runs."""

    default: G1WujiTableAdrCfg = G1WujiTableAdrCfg()
    dr_none: G1WujiTableAdrCfg = default.replace(
        enabled=True,
        initial_level=50,
        max_level=50,
        spawn_enabled=False,
        # robot_position_enabled=False,
        goal_alpha_enabled=False,
        extra_enabled=False,
        sensor_noise_enabled=False,
        action_latency_enabled=False,
        hand_target_scale_enabled=False,
        friction_enabled=False,
        mass_enabled=False,
    )
    dr_full: G1WujiTableAdrCfg = default.replace(
        enabled=True,
        initial_level=50,
        max_level=50,
        spawn_enabled=True,
        # robot_position_enabled=True,
        goal_alpha_enabled=True,
        extra_enabled=True,
        sensor_noise_enabled=True,
        action_latency_enabled=True,
        hand_target_scale_enabled=True,
        friction_enabled=True,
        mass_enabled=True,
    )


@configclass
class G1WujiTableEnvCfg(DirectRLEnvCfg):
    """Configuration for a fixed G1-Wuji assembly facing a pelvis-height work table."""

    decimation = 2
    episode_length_s = 8.0

    # Normalized full-range joint-position targets for the 7 right-arm joints, followed by
    # the frozen 18-D Wuji latent action. The waist remains internally held.
    action_space = 25
    # The actor receives the deployable 171-D policy tensor. The asymmetric critic receives the 171-D clean
    # simulator tensor plus the fixed-width DR state appended by ``_get_critic_privileged_state``.
    observation_space = 171
    state_space = 247
    log_control_metrics: bool = False
    """Whether to emit all ``Control/*`` TensorBoard/extras topics."""
    contact_force_observation_max = 20.0
    """Maximum apple contact-force magnitude [N] before the log1p observation transform."""
    arm_action_ema_alpha = 0.25
    """Weight of the current arm target in the policy-rate EMA."""
    wuji_action_ema_alpha = 0.1
    """Weight of the current decoded Wuji target in the policy-rate EMA."""
    arm_joint_velocity_limit = 0.25
    """Maximum arm joint speed [rad/s], enforced by rate-limiting the post-EMA arm position targets.

    Solver velocity limits are not a portable clamp: MJWarp ignores ``joint_velocity_limit``, and a PhysX
    clamp at the cap would block catch-up motion that Newton allows.  So the cap is applied to the commanded
    targets on every backend, and the actuators keep the asset's looser authored limit.  The target may also
    lead the measured position only as far as the drive's cfg gains and effort limit make useful.  This cap
    is the arm joint-velocity observation scale.
    """
    hand_joint_velocity_limit = 0.5
    """Maximum Wuji finger joint speed [rad/s], enforced like :attr:`arm_joint_velocity_limit`."""
    action_delta_reward_scale = 0.001
    """Penalty scale for consecutive applied normalized-action changes.

    The regularizer is ``-scale * mean((a_t - a_{t-1})^2)`` after policy-action clipping and
    any configured action latency.
    """
    goal_position = (0.35, -0.05, 0.24)
    """Legacy nominal apple goal position [m] in the environment frame."""
    object_rest_height = 0.04
    """Original tabletop apple root height [m], used as the lift-progress baseline."""
    object_spawn_x_range = (0.25, 0.35)
    """Uniform apple reset x-coordinate range [m] in the environment frame."""
    object_spawn_y_range = (-0.30, -0.10)
    """Uniform apple reset y-coordinate range [m] in the environment frame."""
    object_spawn_height_above_table = 0.20
    """Apple and target-region ceiling above the normal tabletop apple root height [m]."""
    object_spawn_z_range = (object_rest_height, object_rest_height + object_spawn_height_above_table)
    """Uniform apple reset z-coordinate range [m] in the environment/world frame."""
    goal_spawn_x_range = (0.25, 0.35)
    """Uniform target-frame x-coordinate range [m] in the environment frame."""
    goal_spawn_y_range = (-0.30, -0.10)
    """Uniform target-frame y-coordinate range [m] in the environment frame."""
    goal_spawn_z_range = (0.10, object_spawn_z_range[1])
    """Uniform target-frame z-coordinate range [m] in the environment frame."""
    goal_roll_range = (-math.radians(30.0), math.radians(30.0))
    """Uniform target-frame roll range [rad] about the nominal world frame."""
    goal_pitch_range = (-math.radians(30.0), math.radians(30.0))
    """Uniform target-frame pitch range [rad] about the nominal world frame."""
    goal_yaw_range = (-math.radians(30.0), math.radians(30.0))
    """Uniform target-frame yaw range [rad] about the nominal world frame."""
    object_spawn_height_offset_cm = 5.0
    """Legacy vertical apple spawn offset [cm]; randomized reset uses :attr:`object_spawn_z_range`."""
    reset: G1WujiTableResetCfg = G1WujiTableResetCfg()
    """Task-reset controls; unlike ADR, these apply at every reset."""
    adr: G1WujiTableAdrCfg = G1WujiTableAdrPresetCfg()
    """ADR settings; select ``presets=dr_none`` or ``presets=dr_full`` to compose a run preset."""
    virtual_force: G1WujiTableVirtualForceCfg = G1WujiTableVirtualForcePresetCfg()
    """Virtual Wuji actuator-torque sensing, enabled only by ``presets=force_distill``."""
    adr_debug_spawn_area_vis: bool = False
    """Legacy opt-in alias for :attr:`debug.adr_spawn_area_marker`."""
    debug: G1WujiTableDebugCfg = G1WujiTableDebugPresetCfg()
    """Viewer diagnostics; headless ``train`` and ``eval`` turn them off."""
    depth_preview: G1WujiTableDepthPreviewCfg = G1WujiTableDepthPreviewPresetCfg()
    """Composable grayscale preview of the exact student depth input."""
    contact_debug: bool = False
    """Whether to sample MJWarp contact and constraint demand for capacity sizing; off during normal runs."""
    contact_debug_interval: int = 1
    """Number of policy steps between contact-demand samples when :attr:`contact_debug` is enabled."""
    depth_camera: CameraCfg | None = G1WujiTableDepthCameraPresetCfg()
    """The student's head depth camera.

    Adds the ``camera`` observation; ``distill`` and ``depth_view`` enable it.
    """
    student_depth_min_m: float = 0.01
    """Depths [m] below this usable range become invalid zero-depth pixels."""
    student_depth_max_m: float = 1.2
    """Depths [m] at or beyond this usable range become invalid zero-depth pixels."""
    goal_keypoint_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CrossEmbodiment/goal_keypoints",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    """Two-centimetre green spheres marking the eight fixed goal keypoints."""
    object_keypoint_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CrossEmbodiment/object_keypoints",
        markers={
            "object": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    """Two-centimetre red spheres marking the current object-frame keypoints."""
    goal_frame_marker_cfg = FRAME_MARKER_CFG.copy()
    goal_frame_marker_cfg.prim_path = "/Visuals/CrossEmbodiment/goal_frame"
    goal_frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    """Uniformly scaled axis-frame marker for the randomized target orientation."""
    adr_spawn_area_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CrossEmbodiment/adr_spawn_area",
        markers={
            "area": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.6, 1.0),
                    opacity=0.3,
                ),
            ),
        },
    )
    """Unit-width blue square scaled to the configured ADR spawn area at runtime."""
    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_[^/]+/G1Wuji/wujihand/right_palm_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_[^/]+/Apple"],
        max_contact_data_count_per_prim=64,
    )
    """Template for a hand-to-apple force sensor; the environment creates one per contact group.

    The groups are the palm and the five fingers, each covering every body that owns a collision shape.
    """
    torso_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_[^/]+/G1Wuji/g1_simplified/torso_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_[^/]+/Apple"],
    )
    """Filtered torso-to-apple sensor for the collision termination."""
    keypoint_extent = 0.15
    """Half-side length [m] of the virtual object-frame cube used for pose reward."""
    reward_mode: str = "adept"
    """Reward formulation used by :meth:`G1WujiTableEnv._get_rewards`.

    ``"shaped"`` (default) is reach + ungated keypoint-goal reward + optional lift + a flat
    thumb-and-finger contact bonus. ``"adept"`` is the ADEPT-style minimal
    reward: reach, plus a goal term gated on a two-body force threshold (thumb and any other
    finger, not the palm) whose keypoint-error sharpness ramps over training, plus a flat
    contact bonus under the same gate. It has no lift term and no press guard. Any other value
    raises ``ValueError``. ``"shaped"`` stays the default until ``"adept"`` is shown to train.
    """
    adept_gate_force: float = 0.3
    """``reward_mode="adept"`` per-body contact-force threshold [N] for the grasp gate.

    The gate requires the thumb (:attr:`G1WujiTableEnv._THUMB_CONTACT_GROUP`) and at least one
    other finger, excluding the palm, to each exceed this force.
    """
    adept_contact_reward_scale: float = 0.01
    """``reward_mode="adept"`` flat per-step reward while the grasp gate holds."""
    adept_goal_alpha_start: float = 15.0
    """``reward_mode="adept"`` goal-reward keypoint-error sharpness at ADR level zero."""
    adept_goal_alpha_end: float = 30.0
    """``reward_mode="adept"`` goal-reward keypoint-error sharpness at full ADR strength."""
    adept_lift_reward_scale: float = 0.0
    """``reward_mode="adept"`` optional dense per-step reward for height progress toward the goal.

    Reuses the shaped mode's ``lift_fraction`` (see :meth:`G1WujiTableEnv._lift_fraction`). 0.0
    (default) adds nothing, so pure ADEPT behaviour is byte-identical; the gated, alpha-sharpened
    goal term alone gives no gradient toward lifting while far from the goal, which this term
    supplies when set nonzero. :attr:`lift_reward_enabled` globally disables this term too.
    """
    reach_reward_scale = 10.0
    goal_reward_scale = 5.0
    goal_reward_alpha = 15.0
    lift_reward_enabled: bool = False
    """Whether to include dense lift reward in either reward formulation; disabled by default."""
    lift_reward_scale = 3.0
    """Per-step reward for carrying the apple the full way from its rest height to the goal."""
    gravity_curriculum_start: float = 0.0
    """Fraction of configured scene gravity at the start of training; 1.0 disables the curriculum."""
    gravity_curriculum_steps: int = 19_200
    """Env steps for whole-scene gravity to ramp from :attr:`gravity_curriculum_start` to full gravity."""
    contact_force_threshold = 0.3
    """Per-group normal force [N] counted as contact.

    The shaped contact bonus requires the thumb and at least one other finger to exceed this threshold.
    """
    contact_reward_scale = 0.5
    """Flat per-step bonus when the thumb and another finger both exceed the contact threshold."""
    success_keypoint_error_threshold = 0.10
    """Terminal mean virtual-keypoint error threshold [m] (10 cm) for the success metric."""
    workspace_termination_enabled: bool = False
    """Whether to terminate when the apple leaves its horizontal reset workspace."""
    object_max_horizontal_displacement = 0.20
    """Maximum horizontal displacement [m] from the authored apple reset pose."""

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics=G1WujiTablePhysicsCfg(),
        # PhysX's default material for shapes without one (Newton uses ``default_shape_cfg`` instead).
        # UNTESTED on PhysX: the max combine mode and resulting contact friction are only verified on Newton.
        physics_material=PhysxRigidBodyMaterialCfg(
            static_friction=_FRICTION, dynamic_friction=_FRICTION, friction_combine_mode="max"
        ),
        # The Newton viewer, except for the headless ``train`` and ``eval`` presets.  An explicit ``--viz``
        # still takes precedence.
        # ``depth_view`` is an overlay: it adds the streaming panel while preserving the selected
        # run preset's physics, environment count, and diagnostic markers.
        visualizer_cfgs=preset(
            default=[NewtonGLVisualizerCfg()],
            train=[],
            eval=[],
            distill=[],
            force_distill=[],
            depth_view=[
                NewtonGLVisualizerCfg(
                    streaming_view=True,
                    streaming_sensor_prim_path=_STUDENT_DEPTH_CAMERA_STREAM_PATTERN,
                    streaming_gt_types=("rgb",),
                    # The task publishes a grayscale rendering of the exact normalized student tensor
                    # under ``rgb``. The native metric render remains available under
                    # ``distance_to_image_plane`` for observation assembly and diagnostics.
                )
            ],
        ),
    )
    scene: InteractiveSceneCfg = G1WujiTableSceneCfg()

    robot_cfg = G1_WUJI_CFG.replace(prim_path="{ENV_REGEX_NS}/G1Wuji")
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LENGTH_X, _TABLE_WIDTH_Y, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # UNTESTED on PhysX: the max combine mode is only verified on Newton, which ignores it.
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=_FRICTION, dynamic_friction=_FRICTION, friction_combine_mode="max"
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.18, 0.08)),
        ),
        # The G1 asset's fixed pelvis is at z=0; the table top is therefore at pelvis height.
        # Keep the near edge at x=0.15 m so the robot-to-table gap is unchanged.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(_TABLE_NEAR_EDGE_X + 0.5 * _TABLE_LENGTH_X, 0.0, -0.02)),
    )
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_APPLE_USD_PATH),
            # This reviewed offline asset has one explicit 64-vertex convex hull. It avoids
            # Newton's runtime convexDecomposition, which expands the source apple into ~61
            # hulls and inflates hand contact and constraint demand at scale.
            # NewtonMeshCollisionPropertiesCfg (rather than the generic MeshCollisionPropertiesCfg)
            # The default mesh_approximation_name="convexHull" remains safe: Isaac Lab and
            # Newton process each collision-enabled Mesh prim independently.
            collision_props=sim_utils.CollisionPropertiesCfg(
                mesh_collision_property=sim_utils.NewtonMeshCollisionPropertiesCfg(
                    mesh_approximation_name="convexHull", max_hull_vertices=None
                )
            ),
        ),
        # The reset path applies ``object_spawn_height_offset_cm`` to this rest baseline.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.05, object_rest_height)),
    )

    def __post_init__(self) -> None:
        """Set a useful default camera for visual scene inspection."""
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.2, -2.2, 1.5), lookat=(0.3, 0.0, 0.15))

    def validate_config(self) -> None:
        """Reject undeclared CLI overrides before the environment is created."""
        unknown = _find_undeclared_config_fields(self)
        if unknown:
            paths = "\n".join(f"  - {path}" for path in unknown)
            raise ValueError(f"Undeclared environment config field(s):\n{paths}")
