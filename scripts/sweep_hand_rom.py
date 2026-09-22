"""Sweep a standalone hand through its full range of motion in PhysX."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hand", required=True, choices=("inspire", "dex3"))
parser.add_argument("--cycles", type=int, default=1)
parser.add_argument("--seconds-per-leg", type=float, default=4.0)
parser.add_argument("--max-target-speed", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import omni.usd
import torch
from hand_asset_specs import HAND_SPECS

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg


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
    if args.cycles < 1 or args.seconds_per_leg <= 0.0 or args.max_target_speed <= 0.0:
        raise ValueError("Cycles must be positive and durations/speeds must be greater than zero")
    spec = HAND_SPECS[args.hand]
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=1.0 / 120.0))
    hand = Articulation(
        ArticulationCfg(
            prim_path="/World/Hand",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(spec.usd_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            ),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=spec.open_joint_pos),
            actuators=_actuators(spec),
        )
    )
    sim.reset()

    root_path = "/World/Hand" + spec.root_body_path.removeprefix(spec.root_prim)
    root_prim = omni.usd.get_context().get_stage().GetPrimAtPath(root_path)
    self_collision = root_prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get()
    if self_collision is not True:
        raise RuntimeError(f"PhysX self-collision must be enabled, got {self_collision!r}")

    joint_ids = [hand.joint_names.index(name) for name in spec.active_joint_names]
    limits = hand.data.joint_pos_limits.torch[0]
    active_limits = limits[joint_ids]
    margin = torch.minimum(
        torch.full_like(active_limits[:, 0], 1.0e-4),
        0.01 * (active_limits[:, 1] - active_limits[:, 0]),
    )
    lower = active_limits[:, 0] + margin
    upper = active_limits[:, 1] - margin
    minimum_duration = 1.5 * torch.max(upper - lower).item() / args.max_target_speed
    leg_duration = max(args.seconds_per_leg, minimum_duration)
    steps_per_leg = max(2, round(leg_duration / sim.get_physics_dt()))
    root_id = hand.body_names.index(spec.root_body_name)
    root_start = hand.data.body_pos_w.torch[0, root_id].clone()
    max_mimic_error = 0.0
    max_tracking_error = 0.0
    max_joint_speed = 0.0

    def move(start: torch.Tensor, end: torch.Tensor) -> None:
        nonlocal max_mimic_error, max_tracking_error, max_joint_speed
        for step in range(1, steps_per_leg + 1):
            fraction = step / steps_per_leg
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            target = (start + blend * (end - start)).unsqueeze(0)
            hand.actuators.target_command.set_position_index(value=target, joint_ids=joint_ids)
            hand.write_data_to_sim()
            sim.step()
            hand.update(sim.get_physics_dt())
            position = hand.data.joint_pos.torch[0]
            velocity = hand.data.joint_vel.torch[0]
            if not torch.isfinite(position).all() or not torch.isfinite(velocity).all():
                raise RuntimeError(f"Non-finite joint state at ROM step {step}")
            max_tracking_error = max(max_tracking_error, torch.max(torch.abs(position[joint_ids] - target[0])).item())
            max_joint_speed = max(max_joint_speed, torch.max(torch.abs(velocity[joint_ids])).item())
            for follower, (leader, multiplier) in spec.mimic_joints.items():
                error = abs(
                    position[hand.joint_names.index(follower)].item()
                    - multiplier * position[hand.joint_names.index(leader)].item()
                )
                max_mimic_error = max(max_mimic_error, error)

    current = hand.data.joint_pos.torch[0, joint_ids].clone()
    move(current, lower)
    for _ in range(args.cycles):
        move(lower, upper)
        move(upper, lower)

    root_displacement = torch.linalg.vector_norm(hand.data.body_pos_w.torch[0, root_id] - root_start).item()
    if root_displacement > 1.0e-5:
        raise RuntimeError(f"Fixed root moved {root_displacement:.6g} m")
    if max_mimic_error > 1.0e-3:
        raise RuntimeError(f"Mimic error exceeded tolerance: {max_mimic_error:.6g} rad")
    print(f"PASS PhysX full ROM asset={spec.usd_path}")
    print(f"Joints={hand.num_joints}, bodies={hand.num_bodies}, self-collision={self_collision}")
    print(f"Fixed-root displacement: {root_displacement:.6g} m")
    print(f"Maximum moving-target error: {max_tracking_error:.6g} rad")
    print(f"Maximum mimic error: {max_mimic_error:.6g} rad")
    print(f"Maximum measured joint speed: {max_joint_speed:.6g} rad/s")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
