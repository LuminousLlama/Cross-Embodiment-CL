"""Validate that an assembled G1 hand is one fixed-base PhysX articulation."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hand", required=True, choices=("inspire", "dex3"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
from hand_asset_specs import HAND_SPECS
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_PATHS = {
    "inspire": REPO_ROOT / "assets/g1/g1_with_hands/g1_inspire.usda",
    "dex3": REPO_ROOT / "assets/g1/g1_with_hands/g1_dex3.usda",
}
EXPECTED_JOINT_COUNTS = {"inspire": 22, "dex3": 17}
AUTHORED_POSE_BODIES = {
    "inspire": {
        "hand_base_link": "/World/inspire_hand_right/Geometry/inspire_hand_base/hand_base_link",
        "right_thumb_1": "/World/inspire_hand_right/Geometry/inspire_hand_base/hand_base_link/right_thumb_1",
        "right_thumb_2": (
            "/World/inspire_hand_right/Geometry/inspire_hand_base/hand_base_link/right_thumb_1/right_thumb_2"
        ),
        "right_thumb_3": (
            "/World/inspire_hand_right/Geometry/inspire_hand_base/hand_base_link/"
            "right_thumb_1/right_thumb_2/right_thumb_3"
        ),
        "right_thumb_4": (
            "/World/inspire_hand_right/Geometry/inspire_hand_base/hand_base_link/"
            "right_thumb_1/right_thumb_2/right_thumb_3/right_thumb_4"
        ),
    },
    "dex3": {
        "right_hand_palm_link": "/World/dex3_1_right/Geometry/right_hand_wrist_link/right_hand_palm_link",
        "right_hand_thumb_0_link": (
            "/World/dex3_1_right/Geometry/right_hand_wrist_link/right_hand_palm_link/right_hand_thumb_0_link"
        ),
        "right_hand_thumb_1_link": (
            "/World/dex3_1_right/Geometry/right_hand_wrist_link/right_hand_palm_link/"
            "right_hand_thumb_0_link/right_hand_thumb_1_link"
        ),
        "right_hand_thumb_2_link": (
            "/World/dex3_1_right/Geometry/right_hand_wrist_link/right_hand_palm_link/"
            "right_hand_thumb_0_link/right_hand_thumb_1_link/right_hand_thumb_2_link"
        ),
        "right_hand_middle_0_link": (
            "/World/dex3_1_right/Geometry/right_hand_wrist_link/right_hand_palm_link/right_hand_middle_0_link"
        ),
        "right_hand_middle_1_link": (
            "/World/dex3_1_right/Geometry/right_hand_wrist_link/right_hand_palm_link/"
            "right_hand_middle_0_link/right_hand_middle_1_link"
        ),
        "right_hand_index_0_link": (
            "/World/dex3_1_right/Geometry/right_hand_wrist_link/right_hand_palm_link/right_hand_index_0_link"
        ),
        "right_hand_index_1_link": (
            "/World/dex3_1_right/Geometry/right_hand_wrist_link/right_hand_palm_link/"
            "right_hand_index_0_link/right_hand_index_1_link"
        ),
    },
}


def main() -> None:
    asset_path = ASSET_PATHS[args.hand]
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=1.0 / 120.0))
    sim.stage.GetRootLayer().subLayerPaths.append(str(asset_path))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/g1_simplified",
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=HAND_SPECS[args.hand].open_joint_pos),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    joint_effort_limit=None,
                    stiffness=None,
                    damping=None,
                    armature=None,
                )
            },
        )
    )
    authored_poses = {}
    xform_cache = UsdGeom.XformCache()
    for body_name, prim_path in AUTHORED_POSE_BODIES[args.hand].items():
        transform = xform_cache.GetLocalToWorldTransform(sim.stage.GetPrimAtPath(prim_path))
        quat = transform.ExtractRotationQuat()
        authored_poses[body_name] = (
            torch.tensor(transform.ExtractTranslation(), device=args.device, dtype=torch.float32),
            torch.tensor([*quat.GetImaginary(), quat.GetReal()], device=args.device, dtype=torch.float32),
        )
    sim.reset()

    expected_joints = EXPECTED_JOINT_COUNTS[args.hand]
    if robot.num_joints != expected_joints:
        raise RuntimeError(f"Expected one {expected_joints}-joint G1+hand articulation, got {robot.num_joints} joints")
    pelvis_id = robot.body_names.index("pelvis")
    pelvis_start = robot.data.body_pos_w.torch[0, pelvis_id].clone()
    pose_errors = {}
    pose_failures = []
    for body_name, (authored_position, authored_orientation) in authored_poses.items():
        body_id = robot.body_names.index(body_name)
        position_error = torch.linalg.vector_norm(robot.data.body_pos_w.torch[0, body_id] - authored_position).item()
        orientation_dot = torch.abs(torch.dot(robot.data.body_quat_w.torch[0, body_id], authored_orientation)).clamp(
            max=1.0
        )
        orientation_error = (2.0 * torch.acos(orientation_dot)).item()
        pose_errors[body_name] = (position_error, orientation_error)
        if position_error > 5.0e-3 or orientation_error > 1.0e-2:
            pose_failures.append(
                f"{body_name} moved from its authored assembly pose when physics initialized: "
                f"position error={position_error:.6g} m, orientation error={orientation_error:.6g} rad; "
                f"authored position={authored_position.tolist()}, simulated position="
                f"{robot.data.body_pos_w.torch[0, body_id].tolist()}, authored orientation="
                f"{authored_orientation.tolist()}, simulated orientation="
                f"{robot.data.body_quat_w.torch[0, body_id].tolist()}"
            )
    if pose_failures:
        raise RuntimeError("\n".join(pose_failures))
    target = robot.data.joint_pos.torch.clone()
    joint_ids = list(range(robot.num_joints))
    for step in range(120):
        robot.actuators.target_command.set_position_index(value=target, joint_ids=joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        if not torch.isfinite(robot.data.joint_pos.torch).all():
            raise RuntimeError(f"Non-finite joint state at frame {step}")

    pelvis_displacement = torch.linalg.vector_norm(robot.data.body_pos_w.torch[0, pelvis_id] - pelvis_start).item()
    if pelvis_displacement > 1.0e-5:
        raise RuntimeError(f"Fixed pelvis moved {pelvis_displacement:.6g} m")
    print(f"PASS one G1+{args.hand} articulation: joints={robot.num_joints}, bodies={robot.num_bodies}", flush=True)
    print(f"Fixed-pelvis displacement: {pelvis_displacement:.6g} m", flush=True)
    max_position_error = max(error[0] for error in pose_errors.values())
    max_orientation_error = max(error[1] for error in pose_errors.values())
    print(f"Authored-pose error: position={max_position_error:.6g} m, orientation={max_orientation_error:.6g} rad")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL {type(error).__name__}: {error}", flush=True)
        raise
    finally:
        simulation_app.close()
