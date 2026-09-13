# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the extra-ADR pure helpers: scaled uniform sampling, action latency, and its buffer."""

import pytest
import torch

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env import (
    ActionDelayBuffer,
    sample_latency_steps,
    scaled_uniform,
)


@pytest.mark.unit
def test_scaled_uniform_is_zero_at_zero_strength():
    sample = scaled_uniform(size=1000, half_width=0.01, strength=0.0)

    assert torch.all(sample == 0.0)


@pytest.mark.unit
def test_scaled_uniform_stays_within_the_strength_scaled_bound():
    sample = scaled_uniform(size=1000, half_width=0.01, strength=1.0)

    assert torch.all(sample >= -0.01) and torch.all(sample <= 0.01)


@pytest.mark.unit
def test_sample_latency_steps_is_zero_at_zero_strength():
    delay = sample_latency_steps(n=1000, max_steps=3, strength=0.0)

    assert torch.all(delay == 0)


@pytest.mark.unit
def test_sample_latency_steps_stays_within_bounds_at_full_strength():
    delay = sample_latency_steps(n=1000, max_steps=3, strength=1.0)

    assert torch.all(delay >= 0) and torch.all(delay <= 3)


@pytest.mark.unit
def test_action_delay_buffer_returns_the_action_from_d_steps_ago():
    buffer = ActionDelayBuffer(capacity=4, num_envs=2, action_dim=1, device="cpu")
    for step in range(5):
        buffer.push(torch.full((2, 1), float(step)))

    # After 5 pushes (steps 0..4), the latest is step 4; delay=2 should return step 2.
    delay = torch.tensor([0, 2])
    result = buffer.get(delay)

    assert torch.equal(result, torch.tensor([[4.0], [2.0]]))


@pytest.mark.unit
def test_action_delay_buffer_reset_zero_fills_the_given_envs():
    buffer = ActionDelayBuffer(capacity=4, num_envs=2, action_dim=1, device="cpu")
    buffer.push(torch.full((2, 1), 5.0))

    buffer.reset(torch.tensor([0]))

    assert torch.equal(buffer.get(torch.tensor([0, 0])), torch.tensor([[0.0], [5.0]]))
