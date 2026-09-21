# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ADR object spawn-offset sampler."""

import numpy as np
import pytest
import torch

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env import (
    balanced_random_clone_strategy,
    latch_first_success_steps,
    per_object_success_metrics,
    sample_goal_poses_outside_success_threshold,
    sample_spawn_offsets,
)


@pytest.mark.unit
def test_object_clone_strategy_is_seeded_and_balanced():
    """Initialization must randomize placement without starving a selected object variant."""
    combinations = np.arange(3)[:, None]

    first = balanced_random_clone_strategy(combinations, 8, seed=7)
    second = balanced_random_clone_strategy(combinations, 8, seed=7)
    counts = np.bincount(first[:, 0], minlength=3)

    assert np.array_equal(first, second)
    assert counts.max() - counts.min() <= 1


@pytest.mark.unit
def test_per_object_success_metrics_groups_only_completed_object_episodes():
    """Per-object topics must use only that object's resets and omit absent variants."""
    metrics = per_object_success_metrics(
        episode_success=torch.tensor((True, False, True, False)),
        object_variant_ids=torch.tensor((0, 1, 0, 0)),
        active_objects=("YcbApple", "YcbBanana", "YcbHammer"),
    )

    assert metrics.keys() == {"Task/success_ycb_apple_ep", "Task/success_ycb_banana_ep"}
    assert metrics["Task/success_ycb_apple_ep"] == pytest.approx(2 / 3)
    assert metrics["Task/success_ycb_banana_ep"] == pytest.approx(0.0)


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


@pytest.mark.unit
def test_first_success_latch_records_once_and_keeps_unsuccessful_sentinel():
    first_steps = torch.tensor((-1, 3, -1))
    errors = torch.tensor((0.09, 0.01, 0.11))
    result = latch_first_success_steps(
        first_steps, errors, success_threshold=0.10, episode_steps=torch.tensor((7, 7, 7))
    )

    assert torch.equal(result, torch.tensor((7, 3, -1)))
