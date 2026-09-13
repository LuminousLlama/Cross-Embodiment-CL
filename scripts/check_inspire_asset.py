#!/usr/bin/env python3
# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless, CPU-only sanity check for the vendored G1 + Inspire right hand assembly.

Loads ``assets/g1/g1_with_hands/g1_inspire.usda`` (and, for comparison,
``g1_wuji.usda``) with Newton's USD importer exactly as Isaac Lab would --
``newton.ModelBuilder.add_usd`` followed by ``finalize`` -- on
``wp.get_device("cpu")``, no GPU/Isaac Sim app required. Prints and asserts:

- body names;
- per-body collision shape counts (Inspire fingertip frames must have 0,
  every real Inspire link must have >=1);
- the 12 hand joints' names/types/limits;
- the number of PhysxMimicJointAPI mimic constraints (must be 6) and gearing;
- the hand's total mass;
- the world-frame position of each fingertip and the palm/base at the
  default joint pose for both hands, and the angle between the mean
  wrist->fingertip direction of Inspire vs Wuji.

Usage:
    timeout 900 uv run python scripts/check_inspire_asset.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "g1" / "g1_with_hands"

# Fingertip frames/bodies, keyed by hand, as (prim_path, is_body).  Inspire's
# tips are plain Xform frames with no PhysicsRigidBodyAPI (is_body=False);
# Wuji's tips are real bodies.
WUJI_FINGERTIPS = [f"/World/wujihand/right_finger{i}_tip_link" for i in range(1, 6)]
INSPIRE_FINGERTIPS = [
    "/World/inspire_hand/right_index_2/index_tip",
    "/World/inspire_hand/right_middle_2/middle_tip",
    "/World/inspire_hand/right_ring_2/ring_tip",
    "/World/inspire_hand/right_little_2/little_tip",
    "/World/inspire_hand/right_thumb_4/thumb_tip",
]
INSPIRE_HAND_JOINTS = [
    "right_index_1_joint",
    "right_index_2_joint",
    "right_middle_1_joint",
    "right_middle_2_joint",
    "right_ring_1_joint",
    "right_ring_2_joint",
    "right_little_1_joint",
    "right_little_2_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
]
INSPIRE_MIMIC_JOINTS = [
    "right_index_2_joint",
    "right_middle_2_joint",
    "right_ring_2_joint",
    "right_little_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
]


def _load(usd_path: Path):
    import newton

    builder = newton.ModelBuilder()
    result = builder.add_usd(str(usd_path), floating=False)
    model = builder.finalize(device="cpu")
    return builder, model, result


def _body_world_pos(model, body_idx: int) -> np.ndarray:
    return model.body_q.numpy()[body_idx][:3]


def _frame_world_pos(usd_path: Path, prim_path: str) -> np.ndarray:
    """World-frame position of a frame-only (non-body) prim at its authored default pose."""
    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f"{prim_path} not found in {usd_path}"
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = xf.ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def _mean_direction(base: np.ndarray, tips: list[np.ndarray]) -> np.ndarray:
    dirs = []
    for tip in tips:
        v = tip - base
        dirs.append(v / np.linalg.norm(v))
    mean = np.mean(dirs, axis=0)
    return mean / np.linalg.norm(mean)


def check_inspire() -> dict:
    usd_path = _ASSET_DIR / "g1_inspire.usda"
    builder, model, result = _load(usd_path)
    path_body_map = result["path_body_map"]
    path_joint_map = result["path_joint_map"]

    print("\n=== g1_inspire.usda bodies ===")
    for path in sorted(path_body_map):
        print(" ", path)

    print("\n=== Inspire hand collision shape counts (collision-enabled only) ===")
    hand_body_paths = {p: i for p, i in path_body_map.items() if p.startswith("/World/inspire_hand/")}
    for path, idx in sorted(hand_body_paths.items()):
        count = sum(
            1
            for shape_idx, body_idx in enumerate(builder.shape_body)
            if body_idx == idx and builder.shape_flags[shape_idx] & int(newton_shape_collide_flag())
        )
        print(f"  {path}: {count}")
        assert count >= 1, f"real Inspire link {path} has no collision shape"

    for tip_path in INSPIRE_FINGERTIPS:
        assert tip_path not in path_body_map, f"fingertip {tip_path} unexpectedly imported as a rigid body"

    import newton

    print("\n=== Inspire hand joints ===")
    for jname in INSPIRE_HAND_JOINTS:
        jpath = f"/World/inspire_hand/joints/{jname}"
        jidx = path_joint_map[jpath]
        # joint_limit_lower/upper are indexed per-DOF, not per-joint: a 0-DOF
        # fixed joint earlier in the tree (here, AssemblerFixedJoint) shifts
        # every later joint's DOF index off of its joint index.
        dof = builder.joint_qd_start[jidx]
        lower = builder.joint_limit_lower[dof]
        upper = builder.joint_limit_upper[dof]
        jtype = newton.JointType(builder.joint_type[jidx]).name
        print(f"  {jname}: type={jtype} lower={lower:.4f} rad upper={upper:.4f} rad")

    mimic_count = len(builder.constraint_mimic_joint0)
    print(f"\n=== Mimic constraints: {mimic_count} ===")
    joint_label_by_idx = {idx: label for label, idx in path_joint_map.items()}
    for i in range(mimic_count):
        follower = joint_label_by_idx[builder.constraint_mimic_joint0[i]]
        leader = joint_label_by_idx[builder.constraint_mimic_joint1[i]]
        print(
            f"  {follower} = {builder.constraint_mimic_coef0[i]:.4f} + "
            f"{builder.constraint_mimic_coef1[i]:.4f} * {leader}"
        )
    assert mimic_count == 6, f"expected 6 mimic constraints, got {mimic_count}"
    for jname in INSPIRE_MIMIC_JOINTS:
        jpath = f"/World/inspire_hand/joints/{jname}"
        assert any(joint_label_by_idx[c] == jpath for c in builder.constraint_mimic_joint0), (
            f"{jpath} missing a mimic constraint"
        )

    hand_mass = sum(builder.body_mass[i] for i in hand_body_paths.values())
    print(f"\nInspire hand total mass: {hand_mass:.5f} kg")

    base_idx = path_body_map["/World/inspire_hand/inspire_hand_base"]
    base_pos = _body_world_pos(model, base_idx)
    wrist_idx = path_body_map["/World/g1_simplified/right_wrist_yaw_link"]
    wrist_pos = _body_world_pos(model, wrist_idx)
    print(f"Inspire palm/base world position: {base_pos}")
    print(f"G1 right_wrist_yaw_link world position: {wrist_pos}")

    tip_positions = [_frame_world_pos(usd_path, p) for p in INSPIRE_FINGERTIPS]
    for p, pos in zip(INSPIRE_FINGERTIPS, tip_positions, strict=True):
        print(f"  fingertip {p}: {pos}")

    return {
        "hand_mass": hand_mass,
        "wrist_pos": wrist_pos,
        "tip_positions": tip_positions,
    }


def check_wuji() -> dict:
    usd_path = _ASSET_DIR / "g1_wuji.usda"
    builder, model, result = _load(usd_path)
    path_body_map = result["path_body_map"]

    wrist_idx = path_body_map["/World/g1_simplified/right_wrist_yaw_link"]
    wrist_pos = _body_world_pos(model, wrist_idx)
    tip_positions = [_body_world_pos(model, path_body_map[p]) for p in WUJI_FINGERTIPS]

    print("\n=== g1_wuji.usda (comparison) ===")
    print(f"G1 right_wrist_yaw_link world position: {wrist_pos}")
    for p, pos in zip(WUJI_FINGERTIPS, tip_positions, strict=True):
        print(f"  fingertip {p}: {pos}")

    return {"wrist_pos": wrist_pos, "tip_positions": tip_positions}


def newton_shape_collide_flag():
    import newton

    return newton.ShapeFlags.COLLIDE_SHAPES


def main() -> None:
    wp.init()
    wp.set_device("cpu")

    inspire = check_inspire()
    wuji = check_wuji()

    inspire_dir = _mean_direction(inspire["wrist_pos"], inspire["tip_positions"])
    wuji_dir = _mean_direction(wuji["wrist_pos"], wuji["tip_positions"])
    cos_angle = float(np.clip(np.dot(inspire_dir, wuji_dir), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_angle))

    print("\n=== Inspire vs Wuji mean wrist->fingertip direction ===")
    print(f"  Inspire mean direction: {inspire_dir}")
    print(f"  Wuji mean direction:    {wuji_dir}")
    print(f"  Angle between them: {angle_deg:.1f} deg")
    if angle_deg > 45.0:
        print(
            "  WARNING: >45 deg -- the reused Wuji mount offset/rotation points the "
            "Inspire hand in a substantially different direction than Wuji."
        )

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
