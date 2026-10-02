# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the G1-hand distillation runner configurations."""

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import resolve_task_config

from Cross_Embodiment_CL.models.hand_observation import HAND_MASK_SLICE, HAND_POSITION_SLICE
from Cross_Embodiment_CL.tasks.g1_hand_table_direct.config.g1_hand_table.agents.rsl_rl_distillation_cfg import (
    G1HandTableDepthDistillationBaseRunnerCfg,
    G1HandTableStateDistillationRunnerCfg,
)
from Cross_Embodiment_CL.tasks.g1_hand_table_direct.config.g1_hand_table.agents.rsl_rl_ppo_cfg import (
    G1HandTablePPORunnerCfg,
)

_OBSERVATION_DIM = 191
_STUDENT_OBSERVATION_DIM = 161
_GOAL_OBSERVATION_DIM = 9
_FORCE_OBSERVATION_DIM = 20
_DEPTH_SIZE = 224
_ACTION_DIM = 25
_TASK = "CrossEmbodimentCl-G1-Hand-Table-Direct"
_AGENT = "rsl_rl_distillation_cfg_entry_point"


def _observations(batch: int = 1) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.zeros(batch, _OBSERVATION_DIM),
            "student": torch.zeros(batch, _STUDENT_OBSERVATION_DIM),
            "goal": torch.zeros(batch, _GOAL_OBSERVATION_DIM),
            "force": torch.zeros(batch, _FORCE_OBSERVATION_DIM),
            "camera": torch.zeros(batch, 1, _DEPTH_SIZE, _DEPTH_SIZE),
        },
        batch_size=[batch],
    )


def _build_mlp(model_cfg, obs_groups: dict[str, list[str]], obs_set: str) -> MLPModel:
    return resolve_callable(model_cfg.class_name)(
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
        G1HandTableStateDistillationRunnerCfg(),
        G1HandTableDepthDistillationBaseRunnerCfg(),
    ],
)
def test_ppo_actor_loads_strictly_into_teacher(distillation_cfg):
    """A PPO checkpoint's actor weights must load strictly into the distillation teacher."""
    ppo_cfg = G1HandTablePPORunnerCfg()
    actor = _build_mlp(ppo_cfg.actor, ppo_cfg.obs_groups, "actor")
    teacher = _build_mlp(distillation_cfg.teacher, distillation_cfg.obs_groups, "teacher")
    teacher.load_state_dict(actor.state_dict(), strict=True)


def test_changing_hand_masks_preserves_valid_joint_statistics_and_zero_padding():
    """A hand without a joint cannot dilute its normalization or inherit a ghost input."""
    cfg = G1HandTablePPORunnerCfg()
    cfg.actor.hidden_dims = [8]
    actor = _build_mlp(cfg.actor, cfg.obs_groups, "actor")
    normalizer = actor.obs_normalizer
    present = torch.zeros(2, _OBSERVATION_DIM)
    present[:, HAND_MASK_SLICE] = 1
    present[:, HAND_POSITION_SLICE.start] = torch.tensor([2.0, 4.0])
    normalizer.update(present)
    absent = present.clone()
    absent[:, HAND_MASK_SLICE.start] = 0
    absent[:, HAND_POSITION_SLICE.start] = 0
    normalizer.update(absent)
    reference = (torch.tensor([2.0, 4.0]) - 3.0) / (1.0 + normalizer.eps)
    torch.testing.assert_close(normalizer(present)[:, HAND_POSITION_SLICE.start], reference)
    normalized = normalizer(absent)
    assert torch.equal(normalized[:, HAND_MASK_SLICE], absent[:, HAND_MASK_SLICE])
    assert torch.count_nonzero(normalized[:, HAND_POSITION_SLICE.start]) == 0
    exported = torch.jit.script(actor.as_jit())
    observations = _observations(batch=2)
    observations["policy"] = absent
    torch.testing.assert_close(exported(absent), actor(observations))


@pytest.mark.parametrize(
    ("presets", "student_groups", "low_dimensional_size"),
    [
        (
            ["distill"],
            ["student", "goal", "camera"],
            _STUDENT_OBSERVATION_DIM + _GOAL_OBSERVATION_DIM,
        ),
        (
            ["distill", "force"],
            ["student", "goal", "force", "camera"],
            _STUDENT_OBSERVATION_DIM + _GOAL_OBSERVATION_DIM + _FORCE_OBSERVATION_DIM,
        ),
    ],
)
def test_depth_student_preset_selects_matching_deployable_observations(presets, student_groups, low_dimensional_size):
    """Each depth-student preset builds against exactly the observations its environment exposes."""
    env_cfg, distillation_cfg = resolve_task_config(_TASK, _AGENT, overrides=[f"presets={','.join(presets)}"])
    assert distillation_cfg.obs_groups == {"teacher": ["policy"], "student": student_groups}
    assert env_cfg.virtual_force.enabled is ("force" in presets)

    student_cfg = distillation_cfg.student
    student = resolve_callable(student_cfg.class_name)(
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


@pytest.mark.unit
@pytest.mark.parametrize("cfg_type", [G1HandTablePPORunnerCfg, G1HandTableStateDistillationRunnerCfg])
def test_success_denominator_uses_real_algorithm_rollout_boundaries(cfg_type):
    """PPO and distillation wiring must log the count only at each storage boundary."""
    cfg = cfg_type()
    cfg.num_steps_per_env = 3
    for name in ("actor", "critic", "student", "teacher"):
        if hasattr(cfg, name):
            getattr(cfg, name).hidden_dims = [8]
    obs = _observations(batch=2)
    obs["critic"] = torch.cat((obs["policy"], torch.zeros(2, 76)), dim=-1)
    env = SimpleNamespace(get_observations=lambda: obs, cfg={}, num_envs=2, num_actions=_ACTION_DIM)
    cfg = handle_deprecated_rsl_rl_cfg(cfg, version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, cfg.to_dict())
    for counts in ((1, 0, 2), (0, 0, 0)):
        runner.alg.storage.clear()
        for step, count in enumerate(counts):
            extras = {"log": {"Task/success": torch.ones(count)} if count else {}}
            with torch.inference_mode():
                runner.alg.act(obs)
                runner.alg.process_env_step(obs, torch.zeros(2), torch.zeros(2), extras)
            tag = "Task/success-percentage-completed-episode-count"
            if step == cfg.num_steps_per_env - 1:
                assert extras["log"][tag] == sum(counts)
            else:
                assert tag not in extras["log"]
