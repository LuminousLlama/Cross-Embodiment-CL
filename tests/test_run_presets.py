# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the G1-Wuji run presets, resolved the way ``isaaclab train``/``play`` resolve them."""

import pytest

from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import resolve_task_config

import Cross_Embodiment_CL.tasks  # noqa: F401

TASK = "CrossEmbodimentCl-G1-Wuji-Table-Direct"
AGENT = "rsl_rl_cfg_entry_point"


def _visualizer_types(env_cfg) -> list[str]:
    return [cfg.visualizer_type for cfg in env_cfg.sim.visualizer_cfgs]


@pytest.mark.unit
def test_default_reset_pose_sampling_is_task_randomization_not_adr():
    """The apple and target ranges are fixed task settings, independent of ADR."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["presets=train,dr_none"])

    assert env_cfg.object_spawn_x_range == pytest.approx((0.25, 0.35))
    assert env_cfg.object_spawn_y_range == pytest.approx((-0.30, -0.10))
    assert env_cfg.object_spawn_height_above_table == pytest.approx(0.20)
    assert env_cfg.reset.object_on_table is False
    assert env_cfg.object_spawn_z_range == pytest.approx(
        (env_cfg.object_rest_height, env_cfg.object_rest_height + 0.20)
    )
    assert env_cfg.goal_spawn_x_range == pytest.approx((0.25, 0.35))
    assert env_cfg.goal_spawn_y_range == pytest.approx((-0.30, -0.10))
    assert env_cfg.goal_spawn_z_range == pytest.approx((0.10, env_cfg.object_spawn_z_range[1]))
    assert env_cfg.success_keypoint_error_threshold == pytest.approx(0.10)
    assert env_cfg.adr.spawn_enabled is False
    assert env_cfg.adept_gate_force == pytest.approx(0.3)
    assert env_cfg.goal_frame_marker_cfg.markers["frame"].scale == (0.1, 0.1, 0.1)


@pytest.mark.unit
def test_tabletop_object_reset_override_preserves_the_normal_spawn_range():
    """The convenience flag fixes only the apple height; task randomization remains configured."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["presets=debug,dr_none", "env.reset.object_on_table=True"])

    assert env_cfg.reset.object_on_table is True
    assert env_cfg.object_spawn_z_range == pytest.approx(
        (env_cfg.object_rest_height, env_cfg.object_rest_height + env_cfg.object_spawn_height_above_table)
    )


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
    disabled_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["env.lift_reward_enabled=False"])
    assert disabled_cfg.lift_reward_enabled is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("override", "field"),
    [
        ("env.definitely_fake_flag=False", "definitely_fake_flag"),
        ("env.adr.definitely_fake_flag=False", "adr.definitely_fake_flag"),
        ("env.sim.definitely_fake_flag=False", "sim.definitely_fake_flag"),
        ("env.adr.robot_position_enabled=True", "adr.robot_position_enabled"),
    ],
)
def test_undeclared_nested_override_fails_loudly(override, field):
    """Environment validation must reject fields Hydra attached dynamically."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=[override])

    with pytest.raises(ValueError, match=field):
        env_cfg.validate()


@pytest.mark.unit
def test_action_delta_regularizer_default_and_accepts_experiment_scale():
    """The default action-delta scale can still be overridden per experiment."""
    default_cfg, _ = resolve_task_config(TASK, AGENT, overrides=[])
    experiment_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["env.action_delta_reward_scale=0.0001"])

    assert default_cfg.action_delta_reward_scale == pytest.approx(0.001)
    assert experiment_cfg.action_delta_reward_scale == pytest.approx(0.0001)


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
            adr.goal_alpha_enabled,
            adr.extra_enabled,
            adr.sensor_noise_enabled,
            adr.action_latency_enabled,
            adr.hand_target_scale_enabled,
            adr.friction_enabled,
            adr.mass_enabled,
            adr.actuator_dynamics_enabled,
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
            adr.goal_alpha_enabled,
            adr.extra_enabled,
            adr.sensor_noise_enabled,
            adr.action_latency_enabled,
            adr.hand_target_scale_enabled,
            adr.friction_enabled,
            adr.mass_enabled,
            adr.actuator_dynamics_enabled,
        )
    )
    assert adr.actuator_stiffness_scale_range == pytest.approx((0.5, 1.5))
    assert adr.actuator_damping_scale_range == pytest.approx((0.5, 1.5))
    assert adr.actuator_armature_scale_range == pytest.approx((0.75, 1.25))
    assert adr.actuator_effort_limit_scale_range == pytest.approx((0.8, 1.2))
    assert adr.actuator_joint_friction_range == pytest.approx((0.0, 0.1))


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
def test_run_profile_composes_with_typed_physics_selector():
    """Backend selection remains independent instead of conflicting with the run profile."""
    env_cfg, _ = resolve_task_config(TASK, AGENT, overrides=["presets=train", "physics=isaacsim_physx"])
    assert type(env_cfg.sim.physics) is PhysxCfg
    assert env_cfg.scene.num_envs == 2048
    assert _visualizer_types(env_cfg) == []


@pytest.mark.unit
def test_distillation_overlays_compose_across_environment_and_agent():
    """Depth viewing, force input, and DR can be stacked without colliding on bundled sections."""
    env_cfg, agent_cfg = resolve_task_config(
        TASK,
        "rsl_rl_distillation_cfg_entry_point",
        overrides=["presets=distill,depth_view,force,dr_full"],
    )

    assert env_cfg.depth_camera is not None
    assert env_cfg.depth_preview.enabled
    assert env_cfg.virtual_force.enabled
    assert env_cfg.adr.enabled
    assert _visualizer_types(env_cfg) == ["newton_gl"]
    assert agent_cfg.obs_groups["student"] == ["student", "goal", "force", "camera"]


@pytest.mark.unit
def test_removed_force_distill_preset_fails_loudly():
    """The old bundled name is not retained as an alias after the clean preset cutover."""
    with pytest.raises(ValueError, match="force_distill"):
        resolve_task_config(TASK, AGENT, overrides=["presets=force_distill"])
