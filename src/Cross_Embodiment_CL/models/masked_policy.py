# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard RSL-RL models with normalization that respects absent hand joints."""

from __future__ import annotations

import torch
from rsl_rl.models import CNNModel, MLPModel
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict

from .hand_observation import (
    CRITIC_HAND_POSITION_BIAS_SLICE,
    CRITIC_HAND_VELOCITY_BIAS_SLICE,
    CRITIC_OBSERVATION_DIM,
    HAND_LIMIT_SLICE,
    HAND_MASK_SLICE,
    HAND_POSITION_SLICE,
    HAND_SLOT_COUNT,
    HAND_TARGET_SLICE,
    HAND_VELOCITY_SLICE,
    POLICY_OBSERVATION_DIM,
    STUDENT_OBSERVATION_DIM,
)


class HandObservationNormalization(EmpiricalNormalization):
    """Exclude absent slots from statistics and leave validity bits unchanged."""

    def __init__(self, shape: int, mask_start: int, slot_indices: torch.Tensor) -> None:
        super().__init__(shape)
        self.mask_start = mask_start
        self.mask_stop = mask_start + HAND_SLOT_COUNT
        self.register_buffer("slot_indices", slot_indices)
        self.register_buffer("feature_count", torch.zeros(1, shape, dtype=torch.long))
        mask_features = torch.zeros(shape, dtype=torch.bool)
        mask_features[mask_start : mask_start + HAND_SLOT_COUNT] = True
        self.register_buffer("mask_features", mask_features)

    def _valid_features(self, x: torch.Tensor) -> torch.Tensor:
        slot_mask = x[:, self.mask_start : self.mask_stop] > 0.5
        return (self.slot_indices < 0) | slot_mask[:, self.slot_indices.clamp_min(0)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self._mean) / (self._std + self.eps)
        normalized = torch.where(self._valid_features(x), normalized, 0.0)
        return torch.where(self.mask_features, x, normalized)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        if not self.training:
            return
        # Each coordinate has its own count: padding from one hand must not
        # change the statistics of joints that only exist on another hand.
        valid = self._valid_features(x) & ~self.mask_features
        batch_count = valid.sum(dim=0, keepdim=True)
        values = torch.where(valid, x, 0.0)
        batch_mean = values.sum(dim=0, keepdim=True) / batch_count.clamp_min(1)
        batch_var = torch.where(valid, (x - batch_mean).square(), 0.0).sum(dim=0, keepdim=True)
        batch_var /= batch_count.clamp_min(1)
        self.feature_count += batch_count
        rate = batch_count / self.feature_count.clamp_min(1)
        delta = batch_mean - self._mean
        self._mean += rate * delta
        self._var += rate * (batch_var - self._var + delta * (batch_mean - self._mean))
        self._std = self._var.clamp_min(0.0).sqrt()
        self.count += x.shape[0]


def _install_hand_normalizer(model: MLPModel, obs: TensorDict) -> None:
    dimensions = {
        "policy": POLICY_OBSERVATION_DIM,
        "critic": CRITIC_OBSERVATION_DIM,
        "student": STUDENT_OBSERVATION_DIM,
    }
    primary = [name for name in model.obs_groups if name in dimensions]
    if len(primary) != 1:
        raise ValueError("Hand policies require exactly one policy, critic, or student observation group.")
    group = primary[0]
    if obs[group].shape[-1] != dimensions[group]:
        raise ValueError(f"Unexpected {group} observation width: {obs[group].shape[-1]} != {dimensions[group]}.")
    if not model.obs_normalization:
        return
    offset = sum(obs[name].shape[-1] for name in model.obs_groups[: model.obs_groups.index(group)])
    slots = torch.full((model.obs_dim,), -1, dtype=torch.long)
    fields = [HAND_POSITION_SLICE, HAND_VELOCITY_SLICE, HAND_TARGET_SLICE]
    if group == "critic":
        fields += [CRITIC_HAND_POSITION_BIAS_SLICE, CRITIC_HAND_VELOCITY_BIAS_SLICE]
    for field in fields:
        slots[offset + field.start : offset + field.stop] = torch.arange(HAND_SLOT_COUNT)
    slots[offset + HAND_LIMIT_SLICE.start : offset + HAND_LIMIT_SLICE.stop] = torch.arange(
        HAND_SLOT_COUNT
    ).repeat_interleave(2)
    model.obs_normalizer = HandObservationNormalization(model.obs_dim, offset + HAND_MASK_SLICE.start, slots)


class HandMLPModel(MLPModel):
    """RSL-RL MLP with binary hand masks and zero padded inputs after normalization."""

    def __init__(self, obs: TensorDict, *args, **kwargs) -> None:
        super().__init__(obs, *args, **kwargs)
        _install_hand_normalizer(self, obs)


class HandCNNModel(CNNModel):
    """RSL-RL depth model using the same masked proprioception as the teacher."""

    def __init__(self, obs: TensorDict, *args, **kwargs) -> None:
        super().__init__(obs, *args, **kwargs)
        _install_hand_normalizer(self, obs)
