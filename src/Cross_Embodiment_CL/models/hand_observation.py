# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fixed semantic hand observation layout shared by task and policy normalization."""

HAND_SLOT_COUNT = 20
HAND_MASK_SLICE = slice(141, 161)
HAND_POSITION_SLICE = slice(10, 30)
HAND_VELOCITY_SLICE = slice(40, 60)
HAND_TARGET_SLICE = slice(67, 87)
HAND_LIMIT_SLICE = slice(101, 141)
STUDENT_OBSERVATION_DIM = 161
POLICY_OBSERVATION_DIM = 191
CRITIC_OBSERVATION_DIM = 267
CRITIC_HAND_POSITION_BIAS_SLICE = slice(209, 229)
CRITIC_HAND_VELOCITY_BIAS_SLICE = slice(239, 259)
