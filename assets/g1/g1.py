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

import math
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
    # Unitree G1 joint limits and effort limits.  The arm gains use the
    # documented G1 simulation values for the realistic-arm-gains experiment.
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
            # Canonical Unitree G1 motor limits: unitree_mujoco g1_29dof.xml ctrlrange, which the G1 USD's
            # authored maxForce matches.
            joint_effort_limit={
                ".*_shoulder_.*": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            joint_velocity_limit=1.5,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.001,
                ".*_elbow_.*": 0.001,
                ".*_wrist_.*_joint": 0.001,
            },
        ),
        # Wuji's 20 actual revolute finger joints. The fixed legacy weld is not an
        # actuator because it contributes no degree of freedom. Effort limits,
        # stiffness, and damping copy the drives authored in the hand USD (which
        # authors gains per degree); its maxForce equals the official
        # wuji-hand-description MJCF actuatorfrcrange. Armature is that MJCF's
        # value: the USD authors none, so leaving it unset would load 0.
        "wuji_fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_finger[1-5]_joint[1-4]"],
            joint_effort_limit={
                "right_finger1_joint1": 0.4452,
                "right_finger1_joint2": 0.4259,
                "right_finger1_joint3": 0.1888,
                "right_finger1_joint4": 0.1468,
                "right_finger2_joint1": 0.6188,
                "right_finger2_joint2": 0.1822,
                "right_finger2_joint3": 0.2251,
                "right_finger2_joint4": 0.2170,
                "right_finger3_joint1": 0.6494,
                "right_finger3_joint2": 0.1827,
                "right_finger3_joint3": 0.2078,
                "right_finger3_joint4": 0.2018,
                "right_finger4_joint1": 0.6389,
                "right_finger4_joint2": 0.1832,
                "right_finger4_joint3": 0.2249,
                "right_finger4_joint4": 0.2044,
                "right_finger5_joint1": 0.6441,
                "right_finger5_joint2": 0.1798,
                "right_finger5_joint3": 0.2384,
                "right_finger5_joint4": 0.1866,
            },
            joint_velocity_limit=0.7,
            stiffness={
                "right_finger[1-5]_joint[12]": 2.0,
                "right_finger[1-5]_joint3": 1.0,
                "right_finger[1-5]_joint4": 0.8,
            },
            damping={
                "right_finger[1-5]_joint[12]": 0.05,
                "right_finger[1-5]_joint[34]": 0.03,
            },
            armature={
                "right_finger[1-5]_joint1": 0.0005,
                "right_finger1_joint2": 0.0005,
                "right_finger[2-5]_joint2": 0.0002,
                "right_finger[1-5]_joint[34]": 0.0002,
            },
        ),
    },
)
"""Hand-agnostic fixed-base simplified G1 upper-body configuration."""


G1_WUJI_CFG = G1_BASE_CFG.copy()
G1_WUJI_CFG.spawn.usd_path = str(_G1_ASSET_DIR / "g1_with_hands/g1_wuji.usda")
G1_WUJI_CFG.spawn.activate_contact_sensors = True
G1_WUJI_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    # The thumb-base lower limit is 0.047 rad; this is its valid neutral pose.
    joint_pos={
        "right_shoulder_roll_joint": -math.pi / 4,
        "right_finger1_joint1": 0.05,
    },
    joint_vel={},
)
"""Fixed-base simplified G1 with the local right Wuji hand assembly."""


G1_INSPIRE_CFG = G1_BASE_CFG.copy()
G1_INSPIRE_CFG.spawn.usd_path = str(_G1_ASSET_DIR / "g1_with_hands/g1_inspire.usda")
G1_INSPIRE_CFG.spawn.activate_contact_sensors = True
# "wuji_fingers" matches only Wuji joint names ("right_finger[1-5]_joint[1-4]"),
# none of which exist on the Inspire hand; drop it rather than keep a dead group.
del G1_INSPIRE_CFG.actuators["wuji_fingers"]
G1_INSPIRE_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    # right_thumb_1_joint's range is [-1.2, -0.3] rad and right_thumb_2_joint's is
    # [0.14, 0.56] rad (neither includes 0), so both need an explicit in-range
    # open pose; right_thumb_3_joint/right_thumb_4_joint are their mimic
    # followers (x0.4/x0.6) and need the matching in-range pose for the same
    # reason. Values are the sister repo's hardware-safe open pose
    # (Cross-Embodiment-RL-Bench assets/inspire_hand_rom.py INSPIRE_HAND_OPEN_JOINT_POS_RAD).
    joint_pos={
        "right_shoulder_roll_joint": -math.pi / 4,
        "right_thumb_1_joint": -0.30,
        "right_thumb_2_joint": 0.14,
        "right_thumb_3_joint": 0.056001,
        "right_thumb_4_joint": 0.084001,
    },
    joint_vel={},
)
G1_INSPIRE_CFG.actuators["inspire_fingers"] = ImplicitActuatorCfg(
    # Only the six driven joints get an actuator; the four finger `_2` joints
    # and the thumb `_3`/`_4` joints are PhysX mimic joints coupled to these
    # six (see assets/hands/inspire_right/inspire_hand_right.usda), so they
    # are intentionally left uncovered here -- the coupling is enforced by
    # the USD's physxMimicJoint, not by an actuator.
    joint_names_expr=[
        "right_index_1_joint",
        "right_middle_1_joint",
        "right_ring_1_joint",
        "right_little_1_joint",
        "right_thumb_1_joint",
        "right_thumb_2_joint",
    ],
    # Effort/velocity/gains/armature copy the sister repo's Inspire system-ID
    # actuator (Cross-Embodiment-RL-Bench assets/g1_arm.py:246-264): a shared
    # 30.0 N*m effort cap, the cross-embodiment 0.7 rad/s hardware velocity
    # ceiling, per-joint fitted stiffness/damping, and 0.001 armature.
    joint_effort_limit=30.0,
    joint_velocity_limit=0.7,
    stiffness={
        "right_index_1_joint": 23.784141540527344,
        "right_middle_1_joint": 28.284271240234375,
        "right_ring_1_joint": 3.535533905029297,
        "right_little_1_joint": 4.204482078552246,
        "right_thumb_1_joint": 2715.2900390625,
        "right_thumb_2_joint": 1442.4978332519531,
    },
    damping={
        "right_index_1_joint": 0.04204482212662697,
        "right_middle_1_joint": 0.01767767034471035,
        "right_ring_1_joint": 0.02500000037252903,
        "right_little_1_joint": 0.05000000074505806,
        "right_thumb_1_joint": 0.1767766922712326,
        "right_thumb_2_joint": 1.0606601536273956,
    },
    armature=0.001,
)
"""Fixed-base simplified G1 with the local right Inspire hand assembly."""
