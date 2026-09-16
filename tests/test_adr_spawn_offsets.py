# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ADR apple spawn-offset sampler."""

import pytest
import torch

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env import (
    sample_goal_poses_outside_success_threshold,
    sample_spawn_offsets,
)


@pytest.mark.unit
def test_sample_spawn_offsets_is_zero_at_zero_strength():
    dx, dy = sample_spawn_offsets(n=1000, strength=0.0, box_x=0.11, box_y=0.20)

    assert torch.all(dx == 0.0)
    assert torch.all(dy == 0.0)


@pytest.mark.unit
def test_sample_spawn_offsets_stays_within_the_strength_scaled_box():
    dx, dy = sample_spawn_offsets(n=1000, strength=0.5, box_x=0.11, box_y=0.20)

    assert torch.all(dx <= 0.11 * 0.5 / 2) and torch.all(dx >= -0.11 * 0.5 / 2)
    assert torch.all(dy <= 0.20 * 0.5 / 2) and torch.all(dy >= -0.20 * 0.5 / 2)


@pytest.mark.unit
def test_sample_spawn_offsets_reaches_the_full_strength_box():
    dx, dy = sample_spawn_offsets(n=1000, strength=1.0, box_x=0.11, box_y=0.20)

    assert torch.all(dx <= 0.11 / 2) and torch.all(dx >= -0.11 / 2)
    assert torch.all(dy <= 0.20 / 2) and torch.all(dy >= -0.20 / 2)


@pytest.mark.unit
def test_goal_sampler_rejects_initially_successful_target_poses():
    object_positions = torch.zeros((128, 3))
    object_rotations = torch.tensor((0.0, 0.0, 0.0, 1.0)).repeat(128, 1)
    local_keypoints = torch.cartesian_prod(*(3 * [torch.tensor((-0.15, 0.15))]))
    goals, rotations = sample_goal_poses_outside_success_threshold(
        object_positions,
        object_rotations,
        local_keypoints,
        position_ranges=((0.0, 0.10), (0.0, 0.10), (0.0, 0.10)),
        euler_ranges=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        success_threshold=0.10,
    )
    keypoint_error = torch.linalg.vector_norm(
        goals.unsqueeze(1) - local_keypoints.unsqueeze(0) + local_keypoints.unsqueeze(0), dim=-1
    ).mean(dim=1)

    assert torch.all(keypoint_error > 0.10)
    assert torch.allclose(rotations, object_rotations)
