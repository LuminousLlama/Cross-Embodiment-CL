# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the visual G1-Wuji table scene."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
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

from .depth_camera import D435_DEPTH_848X480, square_crop

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


STUDENT_DEPTH_SIZE = 224
"""Side [px] of the student's square depth image."""
STUDENT_DEPTH_CROP = square_crop(D435_DEPTH_848X480, STUDENT_DEPTH_SIZE)
"""Square crop of the D435 depth stream that the simulated camera reproduces."""
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
        width=STUDENT_DEPTH_SIZE,
        height=STUDENT_DEPTH_SIZE,
        # The Newton renderer draws square pixels about a centred principal point, which the crop guarantees.
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            STUDENT_DEPTH_CROP.output.matrix(),
            width=STUDENT_DEPTH_SIZE,
            height=STUDENT_DEPTH_SIZE,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(convention="world"),
        renderer_cfg=NewtonWarpRendererCfg(enable_textures=False),
    )
    depth_view: CameraCfg = distill


@configclass
class G1WujiTableAdrCfg:
    """Adaptive domain-randomization schedule and term configuration."""

    enabled: bool = False
    """Whether the success-gated adaptive domain-randomization schedule is active."""
    drives_gravity: bool = True
    """Whether ADR strength drives whole-scene gravity from ``gravity_start`` to full strength."""
    gravity_start: float = 0.1
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
    robot_position_enabled: bool = False
    """Whether ADR strength randomizes the robot root position independently of other DR terms."""
    robot_position_range: float = 0.03
    """Full-strength robot root-position randomization half-width [m] along each axis."""
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
        robot_position_enabled=False,
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
        robot_position_enabled=True,
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
    # Policy and critic have equal dimensions but distinct values when sensor-noise ADR is active:
    # policy receives noisy measurements, while critic receives clean simulator-derived values.
    observation_space = 171
    state_space = 171
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
    goal_position = (0.35, -0.05, 0.24)
    """Fixed apple goal position [m] in the environment frame."""
    adr: G1WujiTableAdrCfg = G1WujiTableAdrPresetCfg()
    """ADR settings; select ``presets=dr_none`` or ``presets=dr_full`` to compose a run preset."""
    adr_debug_spawn_area_vis: bool = False
    """Legacy opt-in alias for :attr:`debug.adr_spawn_area_marker`."""
    debug: G1WujiTableDebugCfg = G1WujiTableDebugPresetCfg()
    """Viewer diagnostics; headless ``train`` and ``eval`` turn them off."""
    contact_debug: bool = False
    """Whether to sample MJWarp contact and constraint demand for capacity sizing; off during normal runs."""
    contact_debug_interval: int = 1
    """Number of policy steps between contact-demand samples when :attr:`contact_debug` is enabled."""
    depth_camera: CameraCfg | None = G1WujiTableDepthCameraPresetCfg()
    """The student's head depth camera.

    Adds the ``camera`` observation; ``distill`` and ``depth_view`` enable it.
    """
    student_depth_max_m: float = 1.2
    """Depth [m] that the student's normalized depth image saturates at."""
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
    reward_mode: str = "shaped"
    """Reward formulation used by :meth:`G1WujiTableEnv._get_rewards`.

    ``"shaped"`` (default) is today's reach + thumb-and-finger contact-gated goal + lift + binary contact
    reward. ``"adept"`` is the ADEPT-style minimal
    reward: reach, plus a goal term gated on a two-body force threshold (thumb and any other
    finger, not the palm) whose keypoint-error sharpness ramps over training, plus a flat
    contact bonus under the same gate. It has no lift term and no press guard. Any other value
    raises ``ValueError``. ``"shaped"`` stays the default until ``"adept"`` is shown to train.
    """
    adept_gate_force: float = 1.0
    """``reward_mode="adept"`` per-body contact-force threshold [N] for the grasp gate.

    The gate requires the thumb (:attr:`G1WujiTableEnv._THUMB_CONTACT_GROUP`) and at least one
    other finger, excluding the palm, to each exceed this force.
    """
    adept_contact_reward_scale: float = 0.01
    """``reward_mode="adept"`` flat per-step reward while the grasp gate holds."""
    adept_goal_alpha_start: float = 15.0
    """``reward_mode="adept"`` goal-reward keypoint-error sharpness at ``common_step_counter=0``."""
    adept_goal_alpha_end: float = 30.0
    """``reward_mode="adept"`` goal-reward keypoint-error sharpness once the ramp completes."""
    adept_goal_alpha_steps: int = 32_000
    """Env steps over which the ``adept`` goal-reward sharpness ramps from start to end.

    PPO runs 32 steps per iteration, so 32 000 steps is iteration 1000.
    """
    adept_lift_reward_scale: float = 0.0
    """``reward_mode="adept"`` optional dense per-step reward for height progress toward the goal.

    Reuses the shaped mode's ``lift_fraction`` (see :meth:`G1WujiTableEnv._lift_fraction`). 0.0
    (default) adds nothing, so pure ADEPT behaviour is byte-identical; the gated, alpha-sharpened
    goal term alone gives no gradient toward lifting while far from the goal, which this term
    supplies when set nonzero.
    """
    reach_reward_scale = 10.0
    goal_reward_scale = 5.0
    goal_reward_alpha = 15.0
    lift_reward_scale = 3.0
    """Per-step reward for carrying the apple the full way from its rest height to the goal."""
    gravity_curriculum_start: float = 0.0
    """Fraction of configured scene gravity at the start of training; 1.0 disables the curriculum."""
    gravity_curriculum_steps: int = 19_200
    """Env steps for whole-scene gravity to ramp from :attr:`gravity_curriculum_start` to full gravity."""
    contact_force_threshold = 0.3
    """Per-group normal force [N] counted as contact.

    The shaped goal gate requires the thumb and at least one other finger to exceed this threshold.
    """
    contact_reward_scale = 0.5
    """Per-step scale of the binary per-contact-group grasp reward."""
    success_keypoint_error_threshold = 0.10
    """Terminal mean virtual-keypoint error threshold [m] for the success metric."""
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
            depth_view=[
                NewtonGLVisualizerCfg(
                    streaming_view=True,
                    streaming_sensor_prim_path=_STUDENT_DEPTH_CAMERA_STREAM_PATTERN,
                    streaming_gt_types=("depth",),
                    streaming_depth_min=0.1,
                    streaming_depth_max=1.2,
                )
            ],
        ),
    )
    scene: InteractiveSceneCfg = G1WujiTableSceneCfg()

    robot_cfg = G1_WUJI_CFG.replace(prim_path="{ENV_REGEX_NS}/G1Wuji")
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 1.0, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # UNTESTED on PhysX: the max combine mode is only verified on Newton, which ignores it.
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=_FRICTION, dynamic_friction=_FRICTION, friction_combine_mode="max"
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.18, 0.08)),
        ),
        # The G1 asset's fixed pelvis is at z=0; the table top is therefore at pelvis height.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.02)),
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
        # The apple mesh extends to z=-0.0367 m in its local frame. Start its
        # root just above the z=0 tabletop and let normal contact settle it.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.05, 0.04)),
    )

    def __post_init__(self) -> None:
        """Set a useful default camera for visual scene inspection."""
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.2, -2.2, 1.5), lookat=(0.3, 0.0, 0.15))
