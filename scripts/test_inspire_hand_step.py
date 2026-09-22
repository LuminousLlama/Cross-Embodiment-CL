"""Regress a simultaneous 0-to-70-degree Inspire finger target step."""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
from hand_asset_specs import INSPIRE

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

ASSET_PATH = Path(__file__).resolve().parents[1] / "assets/hands/inspire_hand/inspire_hand_right.usda"


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=1.0 / 120.0))
    hand = Articulation(
        ArticulationCfg(
            prim_path="/World/InspireHand",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ASSET_PATH),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            ),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=INSPIRE.open_joint_pos),
            actuators={
                "active": ImplicitActuatorCfg(
                    joint_names_expr=[INSPIRE.active_joint_pattern],
                    joint_effort_limit=None,
                    stiffness=None,
                    damping=None,
                    armature=None,
                ),
                "mimic_followers": ImplicitActuatorCfg(
                    joint_names_expr=[INSPIRE.follower_joint_pattern],
                    joint_effort_limit=None,
                    stiffness=None,
                    damping=None,
                    armature=None,
                ),
            },
        )
    )
    sim.reset()

    joint_names = hand.joint_names
    active_ids = [joint_names.index(name) for name in INSPIRE.active_joint_names]
    target = torch.tensor(
        [[math.radians(70.0), math.radians(70.0), math.radians(70.0), math.radians(70.0), -0.31, 0.15]],
        device=sim.device,
    )
    base_id = hand.body_names.index("hand_base_link")
    base_before = hand.data.body_pos_w.torch[0, base_id].clone()

    # Allow the conservative 0.7 rad/s asset velocity limit to reach and settle
    # at the 70-degree target after the intentionally discontinuous command.
    for step in range(480):
        hand.actuators.target_command.set_position_index(value=target, joint_ids=active_ids)
        hand.write_data_to_sim()
        sim.step()
        hand.update(sim.get_physics_dt())
        if not torch.isfinite(hand.data.joint_pos.torch).all():
            raise RuntimeError(f"Non-finite joint state at frame {step}")

    final_position = hand.data.joint_pos.torch[0]
    root_displacement = torch.linalg.vector_norm(hand.data.body_pos_w.torch[0, base_id] - base_before).item()
    mimic_errors = {
        follower: abs(
            final_position[joint_names.index(follower)].item()
            - multiplier * final_position[joint_names.index(leader)].item()
        )
        for follower, (leader, multiplier) in INSPIRE.mimic_joints.items()
    }
    finger_error = torch.max(torch.abs(final_position[active_ids[:4]] - target[0, :4])).item()

    if root_displacement > 1.0e-5:
        raise RuntimeError(f"Fixed base moved {root_displacement:.6g} m")
    if max(mimic_errors.values()) > 1.0e-3:
        raise RuntimeError(f"Mimic error exceeded tolerance: {mimic_errors}")
    if finger_error > 1.0e-3:
        raise RuntimeError(f"Finger target error exceeded tolerance: {finger_error:.6g} rad")

    print(f"PASS asset={ASSET_PATH}")
    print(f"Abrupt 0-to-70-degree step base displacement: {root_displacement:.6g} m")
    print(f"Maximum finger target error: {finger_error:.6g} rad")
    print(f"Maximum mimic error: {max(mimic_errors.values()):.6g} rad")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL {type(error).__name__}: {error}", flush=True)
        raise
    finally:
        simulation_app.close()
