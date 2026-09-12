# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the G1-Wuji run presets, resolved the way ``isaaclab train``/``play`` resolve them."""

import pytest

from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import resolve_task_config

import Cross_Embodiment_CL.tasks  # noqa: F401

TASK = "CrossEmbodimentCl-G1-Wuji-Table-Direct"
AGENT = "rsl_rl_cfg_entry_point"


def _visualizer_types(env_cfg) -> list[str]:
    return [cfg.visualizer_type for cfg in env_cfg.sim.visualizer_cfgs]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "num_envs", "visual_shapes", "visualizers", "keypoint_markers"),
    [
        ([], 1, True, ["newton_gl"], True),
        (["presets=train"], 2048, False, [], False),
        (["presets=debug"], 4, True, ["newton_gl"], True),
        (["presets=eval"], 16, False, [], False),
    ],
)
def test_run_preset_bundles(overrides, num_envs, visual_shapes, visualizers, keypoint_markers):
    """The default and each run preset set physics, environment count, viewer, and diagnostics together."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=overrides)

    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert env_cfg.sim.physics.load_visual_shapes is visual_shapes
    assert env_cfg.scene.num_envs == num_envs
    assert env_cfg.scene.env_spacing == 3.0
    assert _visualizer_types(env_cfg) == visualizers
    assert env_cfg.debug.keypoint_markers is keypoint_markers


@pytest.mark.unit
def test_scalar_overrides_win_inside_run_presets():
    """Scalar overrides still reach fields inside a section a run preset selected."""
    env_cfg, _ = resolve_task_config(
        TASK,
        AGENT,
        overrides=[
            "presets=train",
            "env.sim.physics.load_visual_shapes=True",
            "env.scene.num_envs=2",
            "env.debug.keypoint_markers=True",
        ],
    )

    assert env_cfg.sim.physics.load_visual_shapes is True
    assert env_cfg.scene.num_envs == 2
    assert env_cfg.debug.keypoint_markers is True


@pytest.mark.unit
@pytest.mark.parametrize("physics_override", ["physics=isaacsim_physx", "env.sim.physics=isaacsim_physx"])
def test_physics_selectors_compose_with_debug_preset(physics_override):
    """``debug`` leaves physics at its default, so either selector swaps only the backend."""
    env_cfg, _ = resolve_task_config(
        TASK, AGENT, overrides=["presets=debug", physics_override, "env.debug.keypoint_markers=False"]
    )

    assert type(env_cfg.sim.physics) is PhysxCfg
    assert env_cfg.scene.num_envs == 4
    assert _visualizer_types(env_cfg) == ["newton_gl"]
    assert env_cfg.debug.keypoint_markers is False


@pytest.mark.unit
def test_physics_path_selector_wins_over_train_preset():
    """``env.sim.physics=NAME`` replaces the ``train`` physics config and keeps the rest of the preset."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["presets=train", "env.sim.physics=isaacsim_physx"])

    assert type(env_cfg.sim.physics) is PhysxCfg
    assert env_cfg.scene.num_envs == 2048
    assert _visualizer_types(env_cfg) == []


@pytest.mark.unit
@pytest.mark.parametrize("physics", ["ovphysx", "newton_mjwarp"])
@pytest.mark.parametrize("preset_name", ["train", "eval"])
def test_typed_physics_selector_conflicts_with_headless_presets(preset_name, physics):
    """A typed selector picking another physics config errors instead of silently losing."""
    with pytest.raises(ValueError, match="Conflicting global presets"):
        resolve_task_config(TASK, AGENT, overrides=[f"presets={preset_name}", f"physics={physics}"])
