# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the G1-Wuji run presets, resolved the way ``isaaclab train``/``play`` resolve them."""

import pytest
import torch

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
    ("overrides", "num_envs", "visual_shapes", "visualizers", "keypoint_markers", "spawn_area_marker"),
    [
        ([], 1, True, ["newton_gl"], True, True),
        (["presets=train"], 2048, False, [], False, False),
        (["presets=debug"], 4, True, ["newton_gl"], True, True),
        (["presets=eval"], 16, False, [], False, False),
        (["presets=distill"], 1024, True, [], False, False),
        (["presets=debug,depth_view"], 4, True, ["newton_gl"], True, True),
    ],
)
def test_run_preset_bundles(overrides, num_envs, visual_shapes, visualizers, keypoint_markers, spawn_area_marker):
    """The default and each run preset set physics, environment count, viewer, and diagnostics together."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=overrides)

    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert env_cfg.sim.physics.solver_cfg.nconmax == 128
    assert env_cfg.sim.physics.solver_cfg.njmax == 128
    assert env_cfg.sim.physics.load_visual_shapes is visual_shapes
    assert env_cfg.scene.num_envs == num_envs
    assert env_cfg.scene.env_spacing == 3.0
    assert _visualizer_types(env_cfg) == visualizers
    assert env_cfg.debug.keypoint_markers is keypoint_markers
    assert env_cfg.debug.adr_spawn_area_marker is spawn_area_marker


@pytest.mark.unit
def test_default_reset_pose_sampling_is_task_randomization_not_adr():
    """The apple and target ranges are fixed task settings, independent of ADR."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["presets=train,dr_none"])

    assert env_cfg.object_spawn_x_range == pytest.approx((0.25, 0.35))
    assert env_cfg.object_spawn_y_range == pytest.approx((-0.30, -0.10))
    assert env_cfg.object_spawn_height_above_table == pytest.approx(0.20)
    assert env_cfg.object_spawn_z_range == pytest.approx(
        (env_cfg.object_rest_height, env_cfg.object_rest_height + 0.20)
    )
    assert env_cfg.goal_spawn_x_range == pytest.approx((0.25, 0.35))
    assert env_cfg.goal_spawn_y_range == pytest.approx((-0.30, -0.10))
    assert env_cfg.goal_spawn_z_range == pytest.approx((0.10, env_cfg.object_spawn_z_range[1]))
    assert env_cfg.goal_roll_range == pytest.approx((-torch.pi / 6, torch.pi / 6))
    assert env_cfg.goal_pitch_range == pytest.approx((-torch.pi / 6, torch.pi / 6))
    assert env_cfg.goal_yaw_range == pytest.approx((-torch.pi / 6, torch.pi / 6))
    assert env_cfg.success_keypoint_error_threshold == pytest.approx(0.10)
    assert env_cfg.adr.spawn_enabled is False
    assert env_cfg.adept_gate_force == pytest.approx(0.3)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "has_camera"),
    [
        ([], False),
        (["presets=train"], False),
        (["presets=eval"], False),
        (["presets=distill"], True),
        (["presets=debug,depth_view"], True),
    ],
)
def test_depth_camera_only_in_student_presets(overrides, has_camera):
    """Only student presets render the depth camera, so PPO runs pay no rendering cost."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=overrides)

    assert (env_cfg.depth_camera is not None) is has_camera
    if has_camera:
        assert env_cfg.depth_camera.data_types == ["distance_to_image_plane"]
        assert (env_cfg.depth_camera.width, env_cfg.depth_camera.height) == (224, 224)
        assert env_cfg.depth_camera.update_latest_camera_pose is True


@pytest.mark.unit
def test_depth_view_preset_colorizes_the_student_camera():
    """The opt-in manual viewer streams the depth-only student camera, not an unavailable RGB output."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["presets=debug,depth_view"])

    visualizer = env_cfg.sim.visualizer_cfgs[0]
    assert visualizer.streaming_view is True
    assert visualizer.streaming_sensor_prim_path == (
        "/World/envs/env_[^/]+/G1Wuji/g1_simplified/torso_link/d435_link/depth_camera"
    )
    assert visualizer.streaming_gt_types == ("depth",)
    assert (visualizer.streaming_depth_min, visualizer.streaming_depth_max) == (0.1, 1.2)


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
            "env.debug.adr_spawn_area_marker=True",
        ],
    )

    assert env_cfg.sim.physics.load_visual_shapes is True
    assert env_cfg.scene.num_envs == 2
    assert env_cfg.debug.keypoint_markers is True
    assert env_cfg.debug.adr_spawn_area_marker is True


@pytest.mark.unit
def test_lift_reward_toggle_defaults_off_and_accepts_cli_override():
    """Dense lift is off by default but can be enabled with a scalar override."""
    default_cfg, _ = resolve_task_config(TASK, AGENT, overrides=[])
    enabled_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["env.lift_reward_enabled=True"])

    assert default_cfg.lift_reward_enabled is False
    assert enabled_cfg.lift_reward_enabled is True


@pytest.mark.unit
def test_train_and_dr_none_select_nominal_gravity_without_randomized_terms():
    """The no-DR preset keeps full gravity while disabling every stochastic ADR term."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["presets=train,dr_none"])

    adr = env_cfg.adr
    assert adr.enabled is True
    assert adr.initial_level == adr.max_level == 50
    assert adr.gravity_start + (1.0 - adr.gravity_start) * adr.initial_level / adr.max_level == 1.0
    assert not any(
        (
            adr.spawn_enabled,
            adr.robot_position_enabled,
            adr.goal_alpha_enabled,
            adr.extra_enabled,
            adr.sensor_noise_enabled,
            adr.action_latency_enabled,
            adr.hand_target_scale_enabled,
            adr.friction_enabled,
            adr.mass_enabled,
        )
    )


@pytest.mark.unit
def test_eval_and_dr_full_enable_all_randomized_terms_at_full_level():
    """The full-DR preset ramps gravity from zero at level zero to full strength."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["presets=eval,dr_full"])

    adr = env_cfg.adr
    assert adr.enabled is True
    assert adr.initial_level == adr.max_level == 50
    assert adr.gravity_start == 0.0
    assert adr.gravity_start + (1.0 - adr.gravity_start) * 0 / adr.max_level == 0.0
    assert adr.friction_range == (0.1, 0.4)
    assert all(
        (
            adr.spawn_enabled,
            adr.robot_position_enabled,
            adr.goal_alpha_enabled,
            adr.extra_enabled,
            adr.sensor_noise_enabled,
            adr.action_latency_enabled,
            adr.hand_target_scale_enabled,
            adr.friction_enabled,
            adr.mass_enabled,
        )
    )


@pytest.mark.unit
def test_adr_scalar_override_wins_after_dr_full_preset():
    """Specific ADR flags can still be disabled after selecting the full-DR preset."""
    env_cfg, _ = resolve_task_config(
        TASK,
        AGENT,
        overrides=["presets=dr_full", "env.adr.mass_enabled=False"],
    )

    assert env_cfg.adr.mass_enabled is False
    assert env_cfg.adr.extra_enabled is True


@pytest.mark.unit
def test_adr_spawn_area_marker_has_no_physics_properties():
    """The spawn-area square is display geometry, never a collider or rigid body."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=[])
    marker = env_cfg.adr_spawn_area_marker_cfg.markers["area"]

    assert marker.collision_props is None
    assert marker.rigid_props is None
    assert marker.mass_props is None
    assert marker.size == (1.0, 1.0, 0.001)


@pytest.mark.unit
def test_legacy_adr_spawn_area_marker_override_still_resolves():
    """Existing launch commands using the original root-level switch remain valid."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["env.adr_debug_spawn_area_vis=True"])

    assert env_cfg.adr_debug_spawn_area_vis is True


@pytest.mark.unit
def test_contact_debug_is_opt_in_and_configurable():
    """Capacity sampling is disabled by default and accepts scalar probe overrides."""
    default_cfg, _ = resolve_task_config(TASK, AGENT, overrides=[])
    configured_cfg, _ = resolve_task_config(
        TASK, AGENT, overrides=["env.contact_debug=True", "env.contact_debug_interval=16"]
    )

    assert default_cfg.contact_debug is False
    assert default_cfg.contact_debug_interval == 1
    assert configured_cfg.contact_debug is True
    assert configured_cfg.contact_debug_interval == 16


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
