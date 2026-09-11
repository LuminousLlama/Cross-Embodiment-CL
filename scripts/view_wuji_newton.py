# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Open the Wuji hand with Newton collision overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg, build_simulation_context
import isaaclab.sim as sim_utils
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg


def main() -> None:
    """Run the Newton collision viewer for the Wuji hand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run the same smoke test without a viewer.")
    parser.add_argument("--steps", type=int, default=100_000, help="Physics steps before exiting.")
    args = parser.parse_args()

    asset_path = Path(__file__).parents[1] / "assets/hands/wuji_right_soft_simplified/wujihand.usda"
    lower_limit_rad = 2.7215493 * torch.pi / 180.0
    sim_cfg = SimulationCfg(
        dt=1 / 120,
        device="cuda:0",
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=200,
                # The restored primitive colliders require 104 contacts at startup.
                nconmax=200,
                integrator="implicitfast",
                iterations=100,
                ls_iterations=50,
                cone="elliptic",
                impratio=10.0,
            ),
            num_substeps=2,
            debug_mode=True,
            use_cuda_graph=False,
        ),
        visualizer_cfgs=NewtonGLVisualizerCfg(
            headless=args.headless,
            show_collision=True,
            show_joints=True,
            show_contacts=True,
        ),
    )
    initial_state = ArticulationCfg.InitialStateCfg(joint_pos={"right_finger1_joint1": float(lower_limit_rad)})
    hand_cfg = ArticulationCfg(
        prim_path="/World/Wuji",
        spawn=UsdFileCfg(usd_path=str(asset_path)),
        init_state=initial_state,
        actuators={},
    )

    with build_simulation_context(sim_cfg=sim_cfg, device="cuda:0") as sim:
        hand = Articulation(hand_cfg)
        sim.reset()
        for _ in range(args.steps):
            hand.write_data_to_sim()
            sim.step()
            hand.update(sim.cfg.dt)


if __name__ == "__main__":
    main()
