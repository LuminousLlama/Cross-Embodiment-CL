# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small success-gated scheduler for adaptive domain randomization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveDomainRandomization:
    """Advance a bounded DR level from one rollout's completed-episode success rate."""

    max_level: int = 50
    success_threshold: float = 0.40
    level: int = 0
    last_success_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.max_level <= 0:
            raise ValueError("max_level must be positive.")
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in [0, 1].")
        self.set_level(self.level)

    @property
    def strength(self) -> float:
        """Current normalized DR strength in [0, 1]."""
        return self.level / self.max_level

    def set_level(self, level: int) -> None:
        """Set the current level after validating its bounds."""
        if not 0 <= level <= self.max_level:
            raise ValueError(f"level must be in [0, {self.max_level}].")
        self.level = level

    def update(self, successful_episodes: int, completed_episodes: int) -> bool:
        """Consume one rollout summary and return whether the level advanced."""
        if completed_episodes < 0 or successful_episodes < 0 or successful_episodes > completed_episodes:
            raise ValueError("Episode counts must satisfy 0 <= successful <= completed.")
        if completed_episodes == 0:
            self.last_success_rate = 0.0
            return False

        self.last_success_rate = successful_episodes / completed_episodes
        if self.last_success_rate > self.success_threshold and self.level < self.max_level:
            self.level += 1
            return True
        return False
