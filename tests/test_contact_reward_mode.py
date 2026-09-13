# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shaped reward's per-mode dense contact term."""

import pytest
import torch

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env import contact_term


def test_binary_mode_indicates_contact_without_grading_force():
    forces = torch.tensor([[0.05], [0.2], [50.0]])
    result = contact_term(forces, "binary", threshold=0.1, reference=0.2)
    assert torch.equal(result, torch.tensor([0.0, 1.0, 1.0]))


def test_force_mode_matches_tanh_of_force_over_reference():
    forces = torch.tensor([[0.05, 0.2], [50.0, 0.0]])
    result = contact_term(forces, "force", threshold=0.1, reference=0.2)
    expected = torch.tanh(forces / 0.2).mean(dim=1)
    assert torch.allclose(result, expected)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="contact_reward_mode"):
        contact_term(torch.zeros((1, 1)), "gated", threshold=0.1, reference=0.2)
