# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the G1-Wuji distillation runner configurations."""

import pytest
import torch
from rsl_rl.models import CNNModel, MLPModel
from tensordict import TensorDict

from isaaclab_tasks.utils import resolve_task_config

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.agents.rsl_rl_distillation_cfg import (
    G1WujiTableDepthDistillationBaseRunnerCfg,
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
_TASK = "CrossEmbodimentCl-G1-Wuji-Table-Direct"
_AGENT = "rsl_rl_distillation_cfg_entry_point"


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


@pytest.mark.parametrize(
    "distillation_cfg",
    [
        G1WujiTableStateDistillationRunnerCfg(),
        G1WujiTableDepthDistillationBaseRunnerCfg(),
    ],
)
def test_ppo_actor_loads_strictly_into_teacher(distillation_cfg):
    """A PPO checkpoint's actor weights must load strictly into the distillation teacher."""
    ppo_cfg = G1WujiTablePPORunnerCfg()
    actor = _build_mlp(ppo_cfg.actor, ppo_cfg.obs_groups, "actor")
    teacher = _build_mlp(distillation_cfg.teacher, distillation_cfg.obs_groups, "teacher")
    teacher.load_state_dict(actor.state_dict(), strict=True)


@pytest.mark.parametrize(
    ("preset", "student_groups", "low_dimensional_size"),
    [
        ("distill", ["student", "camera"], _STUDENT_OBSERVATION_DIM),
        (
            "force_distill",
            ["student", "force", "camera"],
            _STUDENT_OBSERVATION_DIM + _FORCE_OBSERVATION_DIM,
        ),
    ],
)
def test_depth_student_preset_selects_matching_deployable_observations(preset, student_groups, low_dimensional_size):
    """Each depth-student preset builds against exactly the observations its environment exposes."""
    env_cfg, distillation_cfg = resolve_task_config(_TASK, _AGENT, overrides=[f"presets={preset}"])
    assert distillation_cfg.obs_groups == {"teacher": ["policy"], "student": student_groups}
    assert env_cfg.virtual_force.enabled is (preset == "force_distill")

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
    # 224 -> 55 -> 26 -> 12 px through the encoder, followed by the selected low-dimensional inputs.
    assert student.mlp[0].in_features == 64 * 12 * 12 + low_dimensional_size
    assert student(_observations(batch=2)).shape == (2, _ACTION_DIM)
