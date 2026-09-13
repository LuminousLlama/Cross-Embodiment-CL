# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ADR apple spawn-offset sampler."""

import pytest
import torch

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env import sample_spawn_offsets


@pytest.mark.unit
def test_sample_spawn_offsets_is_zero_at_zero_strength():
    dx, dy = sample_spawn_offsets(n=1000, strength=0.0, box_x=0.11, box_y=0.20)

    assert torch.all(dx == 0.0)
    assert torch.all(dy == 0.0)


@pytest.mark.unit
def test_sample_spawn_offsets_stays_within_the_strength_scaled_box():
    dx, dy = sample_spawn_offsets(n=1000, strength=0.5, box_x=0.11, box_y=0.20)

    assert torch.all(dx <= 0.0) and torch.all(dx >= -0.11 * 0.5)
    assert torch.all(dy <= 0.0) and torch.all(dy >= -0.20 * 0.5)


@pytest.mark.unit
def test_sample_spawn_offsets_reaches_the_full_strength_box():
    dx, dy = sample_spawn_offsets(n=1000, strength=1.0, box_x=0.11, box_y=0.20)

    assert torch.all(dx <= 0.0) and torch.all(dx >= -0.11)
    assert torch.all(dy <= 0.0) and torch.all(dy >= -0.20)
