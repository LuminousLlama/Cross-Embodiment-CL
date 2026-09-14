# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Wuji finger drives in the G1 cfg must equal the hand USD's authored drives and the official armature."""

import math
from pathlib import Path

import pytest
from pxr import Usd, UsdPhysics

from isaaclab.utils.string import resolve_matching_names_values

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env_cfg import G1_WUJI_CFG

HAND_USD = Path(__file__).resolve().parents[1] / "assets/hands/wuji_right_soft_simplified/wujihand.usda"
JOINT_NAMES = [f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]
# Official wuji-hand-description mjcf/right.xml: armature 0.0002; 0.0005 on each finger's joint1 and the thumb joint2.
OFFICIAL_ARMATURE = {
    name: 0.0005 if name.endswith("_joint1") or name == "right_finger1_joint2" else 0.0002 for name in JOINT_NAMES
}


def _authored_drives() -> dict[str, tuple[float, float, float]]:
    """Return each finger joint's USD drive as (effort limit [N·m], stiffness [N·m/rad], damping [N·m·s/rad])."""
    # USD authors angular drive gains per degree.
    per_radian = 180.0 / math.pi
    drives = {}
    # Hold the stage: traversing a temporary one fails once Python collects it.
    stage = Usd.Stage.Open(str(HAND_USD))
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in JOINT_NAMES:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            drives[prim.GetName()] = (
                drive.GetMaxForceAttr().Get(),
                drive.GetStiffnessAttr().Get() * per_radian,
                drive.GetDampingAttr().Get() * per_radian,
            )
    return drives


@pytest.mark.unit
def test_wuji_finger_drives_match_hand_usd() -> None:
    """Every finger joint's effort limit and gains copy the USD, and its armature is the official value."""
    actuator = G1_WUJI_CFG.actuators["wuji_fingers"]
    authored = _authored_drives()
    assert sorted(authored) == sorted(JOINT_NAMES)
    # Resolve the cfg's joint-name patterns the way the articulation does; a joint matching no key or two keys raises.
    effort_limit, stiffness, damping, armature = (
        dict(zip(*resolve_matching_names_values(values, JOINT_NAMES)[1:], strict=True))
        for values in (actuator.joint_effort_limit, actuator.stiffness, actuator.damping, actuator.armature)
    )
    for name, (usd_effort_limit, usd_stiffness, usd_damping) in authored.items():
        assert effort_limit[name] == pytest.approx(usd_effort_limit, rel=1e-5), name
        assert stiffness[name] == pytest.approx(usd_stiffness, rel=1e-5), name
        assert damping[name] == pytest.approx(usd_damping, rel=1e-5), name
        assert armature[name] == OFFICIAL_ARMATURE[name], name
