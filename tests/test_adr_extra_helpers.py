# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the extra-ADR pure helpers: scaled uniform sampling, action latency, and its buffer."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import DirectRLEnv

from Cross_Embodiment_CL.tasks.g1_hand_table_direct.config.g1_hand_table.env import (
    ActionDelayBuffer,
    G1HandTableEnv,
    PendingPhysicsRandomization,
    action_delta_regularization,
    sample_latency_steps,
    scaled_uniform,
    shaped_goal_and_contact_rewards,
    thumb_and_other_finger_gate,
)


@pytest.mark.unit
def test_contact_capacity_probe_does_not_enable_penetration(monkeypatch):
    """Capacity sizing must remain available with expensive penetration diagnostics disabled."""
    from isaaclab_newton.physics import NewtonManager

    data = object()
    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace(mj_model=object(), mjw_data=data))
    env = G1HandTableEnv.__new__(G1HandTableEnv)
    env.cfg = SimpleNamespace(log_penetration_metrics=False, contact_debug=False)
    env._init_penetration_probe()
    assert env._mjw_data is None

    env.cfg.contact_debug = True
    env._init_penetration_probe()
    assert env._mjw_data is data


@pytest.mark.unit
@pytest.mark.parametrize("indices", [[], [1, 3]])
def test_step_reset_reuses_indices_and_preserves_empty_reset(monkeypatch, indices):
    """Reset exactly the environments used by metrics without recompacting their mask."""
    env = G1HandTableEnv.__new__(G1HandTableEnv)
    env.cfg = SimpleNamespace(compute_final_obs=False)
    env.common_step_counter = env._step_reset_ids_step = 7
    env._step_reset_ids = torch.tensor(indices, dtype=torch.int32)
    calls = []
    monkeypatch.setattr(env, "_reset_idx", calls.append)

    assert env._reset_envs_from_buffer() is env._step_reset_ids
    assert len(calls) == bool(indices)
    if indices:
        assert calls[0] is env._step_reset_ids


@pytest.mark.unit
@pytest.mark.parametrize("final_obs, cached_step", [(True, 7), (False, 6)])
def test_step_reset_delegates_terminal_observations_and_stale_indices(monkeypatch, final_obs, cached_step):
    """Terminal observation handling and masks changed outside a step retain the base reset path."""
    env = G1HandTableEnv.__new__(G1HandTableEnv)
    env.cfg = SimpleNamespace(compute_final_obs=final_obs)
    env.common_step_counter = 7
    env._step_reset_ids_step = cached_step
    expected = object()
    monkeypatch.setattr(DirectRLEnv, "_reset_envs_from_buffer", lambda self: expected)

    assert env._reset_envs_from_buffer() is expected


@pytest.mark.unit
def test_scaled_uniform_is_zero_at_zero_strength():
    sample = scaled_uniform(size=1000, half_width=0.01, strength=0.0)

    assert torch.all(sample == 0.0)


@pytest.mark.unit
def test_scaled_uniform_stays_within_the_strength_scaled_bound():
    sample = scaled_uniform(size=1000, half_width=0.01, strength=1.0)

    assert torch.all(sample >= -0.01) and torch.all(sample <= 0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("forces", "expected"),
    [
        ([0.0, 0.4, 0.0, 0.0, 0.0, 0.0], False),  # thumb alone
        ([0.4, 0.4, 0.0, 0.0, 0.0, 0.0], False),  # palm plus thumb
        ([0.0, 0.0, 0.4, 0.4, 0.0, 0.0], False),  # two non-thumb fingers
        ([0.0, 0.4, 0.4, 0.0, 0.0, 0.0], True),  # thumb plus one other finger
        ([0.0, 0.3, 0.3, 0.0, 0.0, 0.0], False),  # equality is not contact
    ],
)
def test_thumb_and_other_finger_gate_requires_strict_two_finger_contact(forces, expected):
    """The grasp gate excludes palm/single-finger contact and uses a strict threshold."""
    result = thumb_and_other_finger_gate(
        torch.tensor([forces]), ("palm", "finger1", "finger2", "finger3", "finger4", "finger5"), 0.3, "finger1"
    )

    assert result.tolist() == [expected]


@pytest.mark.unit
def test_shaped_goal_is_ungated_and_contact_bonus_is_flat():
    """Contact changes only the binary bonus, not the goal reward."""
    keypoint_error = torch.tensor([0.05, 0.05])
    gate = torch.tensor([False, True])

    goal_reward, contact_reward = shaped_goal_and_contact_rewards(keypoint_error, gate, 5.0, 15.0, 0.5)

    assert torch.allclose(goal_reward, 5.0 * torch.exp(-15.0 * keypoint_error))
    assert torch.equal(contact_reward, torch.tensor([0.0, 0.5]))


@pytest.mark.unit
def test_action_delta_regularization_uses_applied_actions_and_is_disabled_at_zero_scale():
    """The penalty compares consecutive clipped/applied actions, with zero preserving the reward."""
    applied = torch.tensor([[1.0, -1.0, 0.5], [0.0, 0.25, -0.5]])
    previous = torch.tensor([[0.5, -0.5, 0.5], [0.0, 0.0, -1.0]])

    penalty = action_delta_regularization(applied, previous, 0.0001)
    assert torch.allclose(penalty, -0.0001 * torch.tensor([0.5 / 3.0, 0.3125 / 3.0]))
    assert torch.equal(action_delta_regularization(applied, previous, 0.0), torch.zeros(2))


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


@pytest.mark.unit
def test_pending_physics_randomization_accumulates_marks_across_resets():
    pending = PendingPhysicsRandomization(num_envs=4, update_every_steps=32, device="cpu")
    pending.mark(torch.tensor([0, 1]))
    pending.mark(torch.tensor([1, 2]))

    env_ids = pending.take()

    assert torch.equal(torch.sort(env_ids).values, torch.tensor([0, 1, 2]))


@pytest.mark.unit
def test_pending_physics_randomization_take_clears_pending_envs():
    pending = PendingPhysicsRandomization(num_envs=4, update_every_steps=32, device="cpu")
    pending.mark(torch.tensor([0]))

    pending.take()

    assert pending.take().numel() == 0


@pytest.mark.unit
def test_pending_physics_randomization_due_matches_update_cadence():
    pending = PendingPhysicsRandomization(num_envs=4, update_every_steps=32, device="cpu")

    assert pending.due(0) and pending.due(32)
    assert not pending.due(1) and not pending.due(31)
