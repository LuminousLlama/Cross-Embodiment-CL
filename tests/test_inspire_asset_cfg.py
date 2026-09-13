# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""String-level checks for G1_INSPIRE_CFG: no sim, no GPU."""

import importlib.util
from pathlib import Path

import pytest

from isaaclab.utils.string import resolve_matching_names

# assets/g1/g1.py is not an importable package (see env_cfg.py's identical load),
# so load it directly from its file path.
_G1_CONFIG_PATH = Path(__file__).resolve().parents[1] / "assets/g1/g1.py"
_g1_config_spec = importlib.util.spec_from_file_location("cross_embodiment_cl_g1_config", _G1_CONFIG_PATH)
if _g1_config_spec is None or _g1_config_spec.loader is None:
    raise ImportError(f"Unable to load G1 configuration from {_G1_CONFIG_PATH}.")
_g1_config = importlib.util.module_from_spec(_g1_config_spec)
_g1_config_spec.loader.exec_module(_g1_config)
G1_INSPIRE_CFG = _g1_config.G1_INSPIRE_CFG

# The 12 real Inspire hand joints (per inspire_hand_right.urdf): the 6 driven
# joints the "inspire_fingers" actuator must match, plus their 6 mimic
# followers, which must NOT match (they are driven by the USD's
# PhysxMimicJointAPI constraint, not an actuator).
DRIVEN_JOINT_NAMES = [
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
]
MIMIC_JOINT_NAMES = [
    "right_index_2_joint",
    "right_middle_2_joint",
    "right_ring_2_joint",
    "right_little_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
]


@pytest.mark.unit
def test_g1_inspire_usd_path_exists() -> None:
    """The assembly usda referenced by G1_INSPIRE_CFG.spawn is vendored on disk."""
    assert Path(G1_INSPIRE_CFG.spawn.usd_path).is_file()


@pytest.mark.unit
def test_g1_inspire_fingers_actuator_matches_exactly_six_driven_joints() -> None:
    """"inspire_fingers"'s joint_names_expr matches the 6 driven joints and no mimic follower."""
    actuator = G1_INSPIRE_CFG.actuators["inspire_fingers"]
    _, matched = resolve_matching_names(
        actuator.joint_names_expr, DRIVEN_JOINT_NAMES + MIMIC_JOINT_NAMES, raise_when_no_match=False
    )
    assert sorted(matched) == sorted(DRIVEN_JOINT_NAMES)


@pytest.mark.unit
def test_g1_inspire_drops_wuji_fingers_group() -> None:
    """The Wuji-only actuator group is removed rather than kept dead."""
    assert "wuji_fingers" not in G1_INSPIRE_CFG.actuators
