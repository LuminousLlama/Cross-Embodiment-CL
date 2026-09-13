# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ADR nominal-pose spawn-mixing mask."""

import pytest
import torch

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env import apply_nominal_spawn_mask


@pytest.mark.unit
def test_apply_nominal_spawn_mask_is_unchanged_at_zero_probability():
    dx = -torch.rand(1000) * 0.11
    dy = -torch.rand(1000) * 0.20

    out_dx, out_dy = apply_nominal_spawn_mask(dx, dy, prob=0.0)

    assert torch.equal(out_dx, dx)
    assert torch.equal(out_dy, dy)


@pytest.mark.unit
def test_apply_nominal_spawn_mask_zeros_everything_at_probability_one():
    dx = -torch.rand(1000) * 0.11
    dy = -torch.rand(1000) * 0.20

    out_dx, out_dy = apply_nominal_spawn_mask(dx, dy, prob=1.0)

    assert torch.all(out_dx == 0.0)
    assert torch.all(out_dy == 0.0)


@pytest.mark.unit
def test_apply_nominal_spawn_mask_zeros_roughly_half_at_probability_half():
    generator = torch.Generator().manual_seed(0)
    dx = -torch.rand(10000) * 0.11
    dy = -torch.rand(10000) * 0.20

    out_dx, out_dy = apply_nominal_spawn_mask(dx, dy, prob=0.5, generator=generator)

    zero_frac = (out_dx == 0.0).float().mean()
    assert 0.45 < zero_frac < 0.55
    assert torch.equal(out_dx == 0.0, out_dy == 0.0)
