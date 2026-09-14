# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Open every reviewed YCB object in a Newton collision viewer.

The visual meshes and their authored collision hulls are both loaded.  Objects are
kinematic and arranged in a 4-by-3 grid so they stay put while inspecting them.

Usage:
    PYTHONPATH=$PWD/src ../../.venv/bin/python scripts/view_objects_newton.py
    PYTHONPATH=$PWD/src ../../.venv/bin/python scripts/view_objects_newton.py --headless --steps 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg, build_simulation_context
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.assets import RigidObject
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

_OBJECT_NAMES = (
    "YcbApple",
    "YcbBanana",
    "YcbFoamBrick",
    "YcbGelatinBox",
    "YcbHammer",
    "YcbMediumClamp",
    "YcbPhillipsScrewdriver",
    "YcbPottedMeatCan",
    "YcbStrawberry",
    "YcbTennisBall",
    "YcbTomatoSoupCan",
)
_GRID_COLUMNS = 4
_GRID_SPACING_M = 0.35
_GRID_HEIGHT_M = 0.20


def _grid_position(index: int) -> tuple[float, float, float]:
    """Return an object's fixed position in the inspection grid [m]."""
    return (
        (index % _GRID_COLUMNS) * _GRID_SPACING_M,
        (index // _GRID_COLUMNS) * _GRID_SPACING_M,
        _GRID_HEIGHT_M,
    )


def _object_cfg(object_name: str, position: tuple[float, float, float]) -> RigidObjectCfg:
    """Build the fixed Newton asset configuration for one reviewed object."""
    asset_path = Path(__file__).resolve().parents[1] / "assets/objects" / object_name / "textured_collision.usda"
    return RigidObjectCfg(
        prim_path=f"/World/{object_name}",
        spawn=UsdFileCfg(
            usd_path=str(asset_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                mesh_collision_property=sim_utils.NewtonMeshCollisionPropertiesCfg(
                    mesh_approximation_name="convexHull", max_hull_vertices=None
                )
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def main() -> None:
    """Run the Newton visualizer with all reviewed object meshes and hulls."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="Run the initialization smoke without a viewer.")
    parser.add_argument("--steps", type=int, default=100_000, help="Physics steps before exiting.")
    args = parser.parse_args()

    sim_cfg = SimulationCfg(
        dt=1 / 120,
        device="cuda:0",
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=128, nconmax=128, integrator="implicitfast", iterations=100),
            debug_mode=False,
            use_cuda_graph=False,
            load_visual_shapes=True,
        ),
        visualizer_cfgs=NewtonGLVisualizerCfg(headless=args.headless, show_collision=True),
        default_visualizer_cfg=VisualizerCfg(eye=(1.7, -1.7, 1.35), lookat=(0.5, 0.35, 0.15)),
    )

    print("Newton inspection grid (x, y, z) in metres:")
    for index, object_name in enumerate(_OBJECT_NAMES):
        print(f"  {object_name}: {_grid_position(index)}")

    with build_simulation_context(sim_cfg=sim_cfg, device="cuda:0") as sim:
        objects = [RigidObject(_object_cfg(name, _grid_position(index))) for index, name in enumerate(_OBJECT_NAMES)]
        sim.reset()
        for _ in range(args.steps):
            for object_ in objects:
                object_.write_data_to_sim()
            sim.step()
            for object_ in objects:
                object_.update(sim.cfg.dt)


if __name__ == "__main__":
    main()
