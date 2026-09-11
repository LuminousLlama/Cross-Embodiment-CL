# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab configurations for the local fixed-base G1 assemblies.

``G1_BASE_CFG`` describes the hand-agnostic simplified G1 upper body. Hand
variants copy it so shared G1 placement and passive-articulation settings stay
in one place.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_G1_ASSET_DIR = Path(__file__).resolve().parent


G1_BASE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/G1",
    spawn=sim_utils.UsdFileCfg(usd_path=str(_G1_ASSET_DIR / "g1_simplified/g1_simplified.usda")),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    # Unitree G1 joint limits and effort limits, with deliberately moderate
    # arm PD gains for an initially easy-to-control manipulation interface.
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            joint_effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            joint_velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness=5000.0,
            damping=5.0,
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            joint_effort_limit={
                ".*_shoulder_.*": 60.0,
                ".*_elbow_.*": 30.0,
                ".*_wrist_.*_joint": 5.0,
            },
            joint_velocity_limit=1.5,
            stiffness=300.0,
            damping=30.0,
            armature={
                ".*_shoulder_.*": 0.001,
                ".*_elbow_.*": 0.001,
                ".*_wrist_.*_joint": 0.001,
            },
        ),
        # Wuji's 20 actual revolute finger joints. These deliberately simple
        # provisional PD values are responsive and bounded for policy training;
        # system-identified values will replace them in a later pass. The fixed
        # legacy weld is not an actuator because it contributes no degree of
        # freedom.
        "wuji_fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_finger[1-5]_joint[1-4]"],
            joint_effort_limit=5.0,
            joint_velocity_limit=0.7,
            stiffness=20.0,
            damping=1.0,
            armature=0.001,
        ),
    },
)
"""Hand-agnostic fixed-base simplified G1 upper-body configuration."""


G1_WUJI_CFG = G1_BASE_CFG.copy()
G1_WUJI_CFG.spawn.usd_path = str(_G1_ASSET_DIR / "g1_with_hands/g1_wuji.usda")
G1_WUJI_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    # The thumb-base lower limit is 0.047 rad; this is its valid neutral pose.
    joint_pos={"right_finger1_joint1": 0.05},
    joint_vel={},
)
"""Fixed-base simplified G1 with the local right Wuji hand assembly."""
