# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the G1-Wuji distillation runner configurations."""

import pytest
import torch
from rsl_rl.models import CNNModel, MLPModel
from tensordict import TensorDict

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.agents.rsl_rl_distillation_cfg import (
    G1WujiTableDepthDistillationRunnerCfg,
    G1WujiTableStateDistillationRunnerCfg,
)
from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.agents.rsl_rl_ppo_cfg import (
    G1WujiTablePPORunnerCfg,
)

_OBSERVATION_DIM = 171
_STUDENT_OBSERVATION_DIM = 141
_FORCE_OBSERVATION_DIM = 20
_DEPTH_SIZE = 224
_ACTION_DIM = 25
_DISTILLATION_CFGS = [G1WujiTableStateDistillationRunnerCfg, G1WujiTableDepthDistillationRunnerCfg]


def _observations(batch: int = 1) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.zeros(batch, _OBSERVATION_DIM),
            "student": torch.zeros(batch, _STUDENT_OBSERVATION_DIM),
            "force": torch.zeros(batch, _FORCE_OBSERVATION_DIM),
            "camera": torch.zeros(batch, 1, _DEPTH_SIZE, _DEPTH_SIZE),
        },
        batch_size=[batch],
    )


def _build_mlp(model_cfg, obs_groups: dict[str, list[str]], obs_set: str) -> MLPModel:
    return MLPModel(
        _observations(),
        obs_groups,
        obs_set,
        _ACTION_DIM,
        hidden_dims=model_cfg.hidden_dims,
        activation=model_cfg.activation,
        obs_normalization=model_cfg.obs_normalization,
        distribution_cfg=model_cfg.distribution_cfg.to_dict(),
    )


@pytest.mark.parametrize("cfg_class", _DISTILLATION_CFGS)
def test_ppo_actor_loads_strictly_into_teacher(cfg_class):
    """A PPO checkpoint's actor weights must load strictly into the distillation teacher."""
    ppo_cfg = G1WujiTablePPORunnerCfg()
    distillation_cfg = cfg_class()
    actor = _build_mlp(ppo_cfg.actor, ppo_cfg.obs_groups, "actor")
    teacher = _build_mlp(distillation_cfg.teacher, distillation_cfg.obs_groups, "teacher")
    teacher.load_state_dict(actor.state_dict(), strict=True)


@pytest.mark.parametrize("cfg_class", _DISTILLATION_CFGS)
def test_student_mlp_matches_teacher(cfg_class):
    """The student MLP deliberately has the teacher's capacity."""
    distillation_cfg = cfg_class()
    assert distillation_cfg.student.hidden_dims == distillation_cfg.teacher.hidden_dims
    assert distillation_cfg.student.activation == distillation_cfg.teacher.activation


def test_depth_student_reads_only_deployable_observations():
    """The depth student encodes proprioception, virtual force, and depth, never privileged state."""
    distillation_cfg = G1WujiTableDepthDistillationRunnerCfg()
    assert distillation_cfg.obs_groups == {
        "teacher": ["policy"],
        "student": ["student", "force", "camera"],
    }

    student_cfg = distillation_cfg.student
    student = CNNModel(
        _observations(),
        distillation_cfg.obs_groups,
        "student",
        _ACTION_DIM,
        hidden_dims=student_cfg.hidden_dims,
        activation=student_cfg.activation,
        obs_normalization=student_cfg.obs_normalization,
        distribution_cfg=student_cfg.distribution_cfg.to_dict(),
        cnn_cfg=student_cfg.cnn_cfg.to_dict(),
    )
    # 224 -> 55 -> 26 -> 12 px through the encoder, flattened, then 141-D proprioception + 20-D force.
    assert student.mlp[0].in_features == 64 * 12 * 12 + _STUDENT_OBSERVATION_DIM + _FORCE_OBSERVATION_DIM
    assert student(_observations(batch=2)).shape == (2, _ACTION_DIM)
