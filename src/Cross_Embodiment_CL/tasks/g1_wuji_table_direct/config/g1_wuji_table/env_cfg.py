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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonShapeCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg

from isaaclab_tasks.utils import PresetCfg, preset

_G1_CONFIG_PATH = Path(__file__).resolve().parents[6] / "assets/g1/g1.py"
# TODO make a objects CFG file 
_APPLE_USD_PATH = Path(__file__).resolve().parents[6] / "assets/objects/YcbApple/textured.usda"
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
            # Random full-range hand motion has reached 581 constraint rows.
            # Constraints include drives and limits as well as contacts, so
            # this budget is intentionally independent of nconmax.
            njmax=1024,
            # The apple's convex decomposition has 61 hulls.  A deliberately
            # deep apple-palm overlap exceeds the 70-contact dexterous baseline
            # before its 144 constraint rows reach njmax.  Reserve headroom for
            # valid hand-object contacts without changing the PhysX scene.
            nconmax=512,
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


@configclass
class G1WujiTableDebugCfg:
    """Diagnostic drawing, off for training."""

    keypoint_markers: bool = False
    """Whether to draw the goal (green) and current (red) object-pose keypoint markers."""


@configclass
class G1WujiTableDebugPresetCfg(PresetCfg):
    """Diagnostics per run preset: markers whenever a viewer is open, so off for headless ``train`` and ``eval``."""

    default: G1WujiTableDebugCfg = G1WujiTableDebugCfg(keypoint_markers=True)
    train: G1WujiTableDebugCfg = G1WujiTableDebugCfg()
    eval: G1WujiTableDebugCfg = train


@configclass
class G1WujiTableEnvCfg(DirectRLEnvCfg):
    """Configuration for a fixed G1-Wuji assembly facing a pelvis-height work table."""

    decimation = 2
    episode_length_s = 8.0

    # Normalized joint-position deltas for the 7 right-arm joints, followed by
    # the frozen 18-D Wuji latent action. The waist remains internally held.
    action_space = 25
    # The policy and critic deliberately receive the identical privileged state.
    observation_space = 117
    state_space = 117
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
    debug: G1WujiTableDebugCfg = G1WujiTableDebugPresetCfg()
    """Diagnostics, e.g. ``env.debug.keypoint_markers=False``; headless ``train`` and ``eval`` turn them off."""
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
    reach_reward_scale = 10.0
    goal_reward_scale = 5.0
    goal_reward_alpha = 15.0
    lift_reward_scale = 3.0
    """Per-step reward for carrying the apple the full way from its rest height to the goal.

    Force-graded contact alone is farmable by mashing one body into the apple against the
    table: measured, that pinned the apple 1.5 cm *below* its rest height at ~50 N while
    ``tanh`` saturated.  Height cannot be farmed that way and is the behaviour actually wanted.
    """
    apple_weight_curriculum_start: float = 0.5
    """Fraction of the apple's true weight it feels at the start of training.

    An upward force makes up the rest of its weight.  1.0 disables the curriculum: no wrench
    is ever applied.  0.5 is the confirmed default (run R007, 100% no-noise eval success):
    a light apple early lets lift discovery happen at all, with :attr:`entropy_coef` at 0.005.
    ``eval_policy.py`` forces this back to 1.0, so evaluation always runs at full weight.
    """
    apple_weight_curriculum_steps: int = 19_200
    """Env steps over which the felt weight ramps linearly from ``apple_weight_curriculum_start`` to full.

    PPO runs 32 steps per iteration, so 19 200 steps is iteration 600: full weight well inside
    the 2000-iteration training budget (run R007).
    """
    press_tolerance = 0.005
    """Depth [m] below rest height past which the apple counts as pressed, not held."""
    contact_force_threshold = 0.1
    """Per-group normal force [N] counted as contact.

    The apple weighs 0.667 N and its contacts have friction 0.5, so a two-sided pinch needs about
    0.67 N per side.  The previous 1.0 N gate on two bodies was unreachable when friction was 2.0.
    """
    contact_min_bodies = 1
    """Contact groups (the palm or a finger) that must be in contact for the grasp gate.

    Deliberately not thumb-specific.  The thumb once read 0.0 N throughout, but only because its
    sensor sat on the collider-less ``right_finger1_tip_link`` frame.
    """
    contact_reward_scale = 0.5
    """Per-step scale of the dense grasp reward, the stepping stone between reach and lift."""
    contact_force_reference = 0.2
    """Force [N] at which one group's dense contact term reaches tanh(1) ~ 0.76.

    The dense term must be graded in force rather than gated: a binary gate pays nothing
    until it is already satisfied, so it supplies no gradient toward making contact.
    """
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
        visualizer_cfgs=preset(default=[NewtonGLVisualizerCfg()], train=[], eval=[]),
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
    apple_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_APPLE_USD_PATH),
            # The authored asset uses convexDecomposition, which CoACD expands into 61
            # hulls.  Against a 20-joint hand that inflates the contact and constraint
            # counts enough to make large environment counts impractical: an njmax high
            # enough to stay stable costs more memory than it is worth, and an njmax small
            # enough to be cheap diverges into NaN.  An apple is convex apart from its stem
            # dimple, so a single convex hull keeps grasp contacts faithful far more cheaply.
            collision_props=sim_utils.CollisionPropertiesCfg(
                mesh_collision_property=sim_utils.MeshCollisionPropertiesCfg(
                    mesh_approximation_name="convexHull"
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
