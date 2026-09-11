# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the visual G1-Wuji table scene."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

_G1_CONFIG_PATH = Path(__file__).resolve().parents[6] / "assets/g1/g1.py"
_APPLE_USD_PATH = Path(__file__).resolve().parents[6] / "assets/objects/YcbApple/textured.usda"
_g1_config_spec = importlib.util.spec_from_file_location("cross_embodiment_cl_g1_config", _G1_CONFIG_PATH)
if _g1_config_spec is None or _g1_config_spec.loader is None:
    raise ImportError(f"Unable to load G1 configuration from {_G1_CONFIG_PATH}.")
_g1_config = importlib.util.module_from_spec(_g1_config_spec)
_g1_config_spec.loader.exec_module(_g1_config)
G1_WUJI_CFG = _g1_config.G1_WUJI_CFG


@configclass
class G1WujiTablePhysicsCfg(PresetCfg):
    """PhysX and Newton backend presets for the G1-Wuji visual scene."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
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
        debug_mode=True,
        use_cuda_graph=True,
        # The YCB asset intentionally separates an invisible collision mesh
        # from its render-only textured mesh.  Always import the latter so
        # either Newton visualizer can display the same apple as Kit/PhysX.
        load_visual_shapes=True,
    )
    default: PhysxCfg = isaacsim_physx


@configclass
class G1WujiTableEnvCfg(DirectRLEnvCfg):
    """Configuration for a fixed G1-Wuji assembly facing a pelvis-height work table."""

    decimation = 2
    episode_length_s = 60.0

    # Normalized joint-position deltas for the 7 right-arm joints, followed by
    # the frozen 18-D Wuji latent action. The waist remains internally held.
    action_space = 25
    observation_space = 0
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics=G1WujiTablePhysicsCfg(),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)

    robot_cfg = G1_WUJI_CFG.replace(prim_path="{ENV_REGEX_NS}/G1Wuji")
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 1.0, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.18, 0.08)),
        ),
        # The G1 asset's fixed pelvis is at z=0; the table top is therefore at pelvis height.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.02)),
    )
    apple_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_APPLE_USD_PATH)),
        # The apple mesh extends to z=-0.0367 m in its local frame. Start its
        # root just above the z=0 tabletop and let normal contact settle it.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.05, 0.04)),
    )

    def __post_init__(self) -> None:
        """Set a useful default camera for visual scene inspection."""
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.2, -2.2, 1.5), lookat=(0.3, 0.0, 0.15))
