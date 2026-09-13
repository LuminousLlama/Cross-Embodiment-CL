# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression contract for the realistic G1 arm-drive experiment."""

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env_cfg import G1_WUJI_CFG


def test_realistic_arm_drive_gains() -> None:
    """The experiment uses the documented G1 arm gains without changing effort limits."""
    arms = G1_WUJI_CFG.actuators["arms"]
    assert arms.stiffness == 40.0
    assert arms.damping == 10.0
    assert arms.joint_effort_limit[".*_shoulder_.*"] == 25.0
    assert arms.joint_effort_limit[".*_wrist_pitch_joint"] == 5.0
