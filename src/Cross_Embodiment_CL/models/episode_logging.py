# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL algorithms that publish the success denominator once per rollout."""

import torch
from rsl_rl.algorithms import PPO, Distillation
from tensordict import TensorDict


class CompletedEpisodeLogging:
    """Add a rollout completion count without changing algorithm updates."""

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record the transition and publish the count on the rollout's final step."""
        if self.storage.step == 0:
            self._success_completed_episode_count = 0
        log = extras.get("log", {})
        if "Task/success" in log:
            self._success_completed_episode_count += log["Task/success"].numel()
        super().process_env_step(obs, rewards, dones, extras)
        if self.storage.step == self.storage.num_transitions_per_env:
            extras.setdefault("log", {})["Task/success-percentage-completed-episode-count"] = (
                self._success_completed_episode_count
            )


class CompletedEpisodePPO(CompletedEpisodeLogging, PPO):
    """PPO with the completed-episode denominator logged every iteration."""


class CompletedEpisodeDistillation(CompletedEpisodeLogging, Distillation):
    """Distillation with the completed-episode denominator logged every iteration."""
