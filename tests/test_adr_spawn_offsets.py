# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ADR object spawn-offset sampler."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from rsl_rl.utils.logger import Logger

from Cross_Embodiment_CL.models.episode_logging import CompletedEpisodeLogging
from Cross_Embodiment_CL.tasks.g1_hand_table_direct.config.g1_hand_table.env import (
    G1HandTableEnv,
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


@pytest.mark.unit
def test_training_success_weights_completed_episodes_and_clears_each_iteration():
    """Unequal reset batches must not turn one success in 100 episodes into 50%."""
    n = 100
    zeros = torch.zeros(n)
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            active_objects=("YcbApple",),
            success_keypoint_error_threshold=0.10,
            log_control_metrics=False,
            adr=SimpleNamespace(enabled=False),
            contact_force_threshold=1.0,
        ),
        _episode_reward_sums={name: zeros.clone() for name in ("reach", "goal", "contact", "lift", "action_delta")},
        _episode_contact_gate_steps=zeros.clone(),
        _episode_arm_tracking_error_sum=zeros.clone(),
        _episode_hand_tracking_error_sum=zeros.clone(),
        _episode_min_hand_distance=torch.full((n,), torch.inf),
        _episode_min_keypoint_error=torch.full((n,), torch.inf),
        _episode_max_object_height=zeros.clone(),
        _episode_first_success_step=torch.full((n,), -1, dtype=torch.long),
        _gravity_frac=1.0,
        contact_groups=("thumb", "finger"),
        _THUMB_CONTACT_GROUP="thumb",
        _mjw_data=None,
        _contact_demand_metrics=lambda: {},
        episode_length_buf=torch.ones(n, dtype=torch.long),
        object_variant_ids_tensor=torch.zeros(n, dtype=torch.long),
        _termination_torso_object=zeros.bool(),
        _termination_below_table=zeros.bool(),
        _termination_workspace_exit=zeros.bool(),
        _termination_nonfinite=zeros.bool(),
        reset_time_outs=torch.ones(n, dtype=torch.bool),
        extras={},
    )
    logger = Logger(None, {"num_steps_per_env": 3, "algorithm": {"rnd_cfg": None}}, {}, n, False, 1, 0, "cpu")
    logger.writer = Mock()

    class TransitionRecorder:
        def process_env_step(self, obs, rewards, dones, extras):
            self.storage.step += 1

    class Algorithm(CompletedEpisodeLogging, TransitionRecorder):
        storage = SimpleNamespace(step=0, num_transitions_per_env=3)

    algorithm = Algorithm()
    errors = torch.full((n,), 0.2)
    errors[0] = 0.05
    for reset_ids in ([0], [], list(range(1, n))):
        env.reset_buf = torch.zeros(n, dtype=torch.bool)
        env.reset_buf[reset_ids] = True
        G1HandTableEnv._update_episode_metrics(
            env,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            errors,
            zeros,
            zeros,
            zeros,
            zeros.bool(),
            zeros,
            zeros,
            torch.zeros(n, 2),
            zeros,
            0.0,
        )
        algorithm.process_env_step(None, zeros, env.reset_buf, env.extras)
        logger.process_env_step(zeros, env.reset_buf, env.extras)
    logger.log(0, 0, 2, 1.0, 1.0, {}, 0.001, torch.ones(1), None)
    success = [
        call.args[1].item() for call in logger.writer.add_scalar.call_args_list if call.args[0] == "Task/success"
    ]
    assert success == pytest.approx([1 / 100])
    counts = [
        call.args[1].item()
        for call in logger.writer.add_scalar.call_args_list
        if call.args[0] == "Task/success-percentage-completed-episode-count"
    ]
    assert counts == [100]
    logger.writer.reset_mock()
    algorithm.storage.step = 0
    algorithm.storage.num_transitions_per_env = 1
    env.reset_buf.zero_()
    G1HandTableEnv._update_episode_metrics(
        env,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        errors,
        zeros,
        zeros,
        zeros,
        zeros.bool(),
        zeros,
        zeros,
        torch.zeros(n, 2),
        zeros,
        0.0,
    )
    algorithm.process_env_step(None, zeros, env.reset_buf, env.extras)
    logger.process_env_step(zeros, env.reset_buf, env.extras)
    logger.log(1, 0, 2, 1.0, 1.0, {}, 0.001, torch.ones(1), None)
    assert not any(call.args[0] == "Task/success" for call in logger.writer.add_scalar.call_args_list)
    counts = [
        call.args[1].item()
        for call in logger.writer.add_scalar.call_args_list
        if call.args[0] == "Task/success-percentage-completed-episode-count"
    ]
    assert counts == [0]
