# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the G1-Wuji distillation runner configuration."""

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.agents.rsl_rl_distillation_cfg import (
    G1WujiTableStateDistillationRunnerCfg,
)
from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.agents.rsl_rl_ppo_cfg import (
    G1WujiTablePPORunnerCfg,
)

_OBSERVATION_DIM = 117
_ACTION_DIM = 25


def _build_model(model_cfg, obs_groups: dict[str, list[str]], obs_set: str) -> MLPModel:
    obs = TensorDict({"policy": torch.zeros(1, _OBSERVATION_DIM)}, batch_size=[1])
    return MLPModel(
        obs,
        obs_groups,
        obs_set,
        _ACTION_DIM,
        hidden_dims=model_cfg.hidden_dims,
        activation=model_cfg.activation,
        obs_normalization=model_cfg.obs_normalization,
        distribution_cfg=model_cfg.distribution_cfg.to_dict(),
    )


def test_ppo_actor_loads_strictly_into_teacher():
    """A PPO checkpoint's actor weights must load strictly into the distillation teacher."""
    ppo_cfg = G1WujiTablePPORunnerCfg()
    distillation_cfg = G1WujiTableStateDistillationRunnerCfg()
    actor = _build_model(ppo_cfg.actor, ppo_cfg.obs_groups, "actor")
    teacher = _build_model(distillation_cfg.teacher, distillation_cfg.obs_groups, "teacher")
    teacher.load_state_dict(actor.state_dict(), strict=True)


def test_student_mlp_matches_teacher():
    """The student MLP deliberately has the teacher's capacity."""
    distillation_cfg = G1WujiTableStateDistillationRunnerCfg()
    assert distillation_cfg.student.hidden_dims == distillation_cfg.teacher.hidden_dims
    assert distillation_cfg.student.activation == distillation_cfg.teacher.activation
