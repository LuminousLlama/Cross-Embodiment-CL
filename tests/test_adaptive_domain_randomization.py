# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the success-gated adaptive domain-randomization scheduler."""

import pytest

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.adr import (
    AdaptiveDomainRandomization,
)


@pytest.mark.unit
def test_adr_advances_once_above_threshold_and_caps_at_maximum():
    adr = AdaptiveDomainRandomization(max_level=2, success_threshold=0.40)

    assert adr.update(successful_episodes=4, completed_episodes=10) is False
    assert adr.level == 0
    assert adr.update(successful_episodes=5, completed_episodes=10) is True
    assert adr.level == 1
    assert adr.strength == 0.5
    assert adr.update(successful_episodes=10, completed_episodes=10) is True
    assert adr.update(successful_episodes=10, completed_episodes=10) is False
    assert adr.level == 2
    assert adr.strength == 1.0


@pytest.mark.unit
def test_adr_waits_when_a_rollout_completes_no_episodes():
    adr = AdaptiveDomainRandomization()

    adr.update(successful_episodes=1, completed_episodes=1)
    assert adr.update(successful_episodes=0, completed_episodes=0) is False
    assert adr.level == 1
    assert adr.last_success_rate == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_level": 0}, "max_level"),
        ({"success_threshold": 1.1}, "success_threshold"),
        ({"level": 51}, "level"),
    ],
)
def test_adr_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdaptiveDomainRandomization(**kwargs)


@pytest.mark.unit
def test_adr_rejects_invalid_episode_counts():
    adr = AdaptiveDomainRandomization()

    with pytest.raises(ValueError, match="0 <= successful <= completed"):
        adr.update(successful_episodes=2, completed_episodes=1)
