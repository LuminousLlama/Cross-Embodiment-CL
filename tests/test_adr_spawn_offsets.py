# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ADR apple spawn-offset sampler."""

import pytest
import torch

from isaaclab.utils.math import quat_apply

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env import (
    latch_first_success_steps,
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
    num_samples = 20_000
    object_positions = torch.stack(
        (
            torch.empty(num_samples).uniform_(0.25, 0.35),
            torch.empty(num_samples).uniform_(-0.30, -0.10),
            torch.empty(num_samples).uniform_(0.04, 0.24),
        ),
        dim=-1,
    )
    object_rotations = torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(num_samples, 1)
    local_keypoints = torch.cartesian_prod(*(3 * [torch.tensor((-0.15, 0.15))]))
    goals, rotations = sample_goal_poses_outside_success_threshold(
        object_positions,
        object_rotations,
        local_keypoints,
        position_ranges=((0.25, 0.35), (-0.30, -0.10), (0.10, 0.24)),
        success_threshold=0.10,
    )
    object_keypoints = quat_apply(
        object_rotations.unsqueeze(1).expand(-1, local_keypoints.shape[0], -1),
        local_keypoints.unsqueeze(0).expand(num_samples, -1, -1),
    ) + object_positions.unsqueeze(1)
    goal_keypoints = quat_apply(
        rotations.unsqueeze(1).expand(-1, local_keypoints.shape[0], -1),
        local_keypoints.unsqueeze(0).expand(num_samples, -1, -1),
    ) + goals.unsqueeze(1)
    keypoint_error = torch.linalg.vector_norm(object_keypoints - goal_keypoints, dim=-1).mean(dim=1)

    assert torch.all(keypoint_error > 0.10)
    assert torch.allclose(rotations, object_rotations)


@pytest.mark.unit
def test_goal_sampler_rejects_an_infeasible_position_range_without_hanging():
    object_positions = torch.zeros((4, 3))
    object_rotations = torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(4, 1)
    local_keypoints = torch.cartesian_prod(*(3 * [torch.tensor((-0.15, 0.15))]))

    with pytest.raises(RuntimeError, match="Unable to sample 4 target poses"):
        sample_goal_poses_outside_success_threshold(
            object_positions,
            object_rotations,
            local_keypoints,
            position_ranges=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            success_threshold=0.10,
        )


@pytest.mark.unit
def test_first_success_latch_records_once_and_keeps_unsuccessful_sentinel():
    first_steps = torch.tensor((-1, 3, -1))
    errors = torch.tensor((0.09, 0.01, 0.11))
    result = latch_first_success_steps(
        first_steps, errors, success_threshold=0.10, episode_steps=torch.tensor((7, 7, 7))
    )

    assert torch.equal(result, torch.tensor((7, 3, -1)))
