"""Sweep a standalone hand through full ROM with Newton MJWarp."""

import argparse

import torch
from hand_asset_specs import HAND_SPECS

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg, build_simulation_context
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

MAX_TARGET_SPEED = 0.5


def _actuators(spec):
    actuators = {
        "active": ImplicitActuatorCfg(
            joint_names_expr=[spec.active_joint_pattern],
            joint_effort_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        )
    }
    if spec.follower_joint_pattern:
        actuators["mimic_followers"] = ImplicitActuatorCfg(
            joint_names_expr=[spec.follower_joint_pattern],
            joint_effort_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        )
    return actuators


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True, choices=HAND_SPECS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    spec = HAND_SPECS[args.hand]
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device=args.device,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=256,
                nconmax=256,
                iterations=50,
                ls_iterations=25,
            ),
            num_substeps=2,
            debug_mode=False,
            use_cuda_graph=False,
        ),
        visualizer_cfgs=[],
    )
    hand_cfg = ArticulationCfg(
        prim_path="/World/Hand",
        spawn=UsdFileCfg(usd_path=str(spec.usd_path), variants={"Physics": "physics"}),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=spec.open_joint_pos),
        actuators=_actuators(spec),
    )

    with build_simulation_context(sim_cfg=sim_cfg, device=args.device) as sim:
        hand = Articulation(hand_cfg)
        sim.reset()
        joint_ids = [hand.joint_names.index(name) for name in spec.active_joint_names]
        armature = hand.data.joint_armature.torch[0]
        if not torch.allclose(armature, torch.full_like(armature, spec.armature), atol=1.0e-8, rtol=0.0):
            raise RuntimeError(f"Incorrect Newton armature: {armature.tolist()}")

        limits = hand.data.joint_pos_limits.torch[0, joint_ids]
        margin = torch.minimum(torch.full_like(limits[:, 0], 1.0e-4), 0.01 * (limits[:, 1] - limits[:, 0]))
        lower = limits[:, 0] + margin
        upper = limits[:, 1] - margin
        leg_duration = 1.5 * torch.max(upper - lower).item() / MAX_TARGET_SPEED
        steps_per_leg = max(2, round(leg_duration / sim.cfg.dt))
        max_tracking_error = 0.0
        max_joint_speed = 0.0

        def move(start: torch.Tensor, end: torch.Tensor) -> None:
            nonlocal max_tracking_error, max_joint_speed
            for step in range(1, steps_per_leg + 1):
                fraction = step / steps_per_leg
                blend = fraction * fraction * (3.0 - 2.0 * fraction)
                target = (start + blend * (end - start)).unsqueeze(0)
                hand.actuators.target_command.set_position_index(value=target, joint_ids=joint_ids)
                hand.write_data_to_sim()
                sim.step()
                hand.update(sim.cfg.dt)
                position = hand.data.joint_pos.torch[:, joint_ids]
                velocity = hand.data.joint_vel.torch[:, joint_ids]
                if not torch.isfinite(position).all() or not torch.isfinite(velocity).all():
                    raise RuntimeError(f"Non-finite joint state at ROM step {step}")
                max_tracking_error = max(max_tracking_error, torch.max(torch.abs(position - target)).item())
                max_joint_speed = max(max_joint_speed, torch.max(torch.abs(velocity)).item())

        current = hand.data.joint_pos.torch[0, joint_ids].clone()
        move(current, lower)
        move(lower, upper)
        move(upper, lower)
        print(f"PASS Newton MJWarp full ROM asset={spec.usd_path}")
        print(f"Bodies={hand.num_bodies}, joints={hand.num_joints}")
        print(f"Newton armature: {armature.tolist()}")
        print(f"Maximum moving-target error: {max_tracking_error:.6g} rad")
        print(f"Maximum measured joint speed: {max_joint_speed:.6g} rad/s")


if __name__ == "__main__":
    main()
