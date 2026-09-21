# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless end-to-end validation for the Wuji latent action contract."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, unscale_transform  # noqa: E402

from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import Cross_Embodiment_CL.tasks  # noqa: E402, F401
from Cross_Embodiment_CL.models.hand_registry import get_hand_spec  # noqa: E402
from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.depth_camera import (  # noqa: E402
    normalized_depth_to_grayscale,
)
from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env import (  # noqa: E402
    goal_pose_in_base_frame,
)


@pytest.mark.unit
def test_goal_pose_observation_is_expressed_in_the_robot_base_frame() -> None:
    """Keep the deployable goal invariant to the robot base's world translation and yaw."""
    half_pi = torch.tensor([torch.pi / 2])
    zero = torch.zeros_like(half_pi)
    base_rotation_w = quat_from_euler_xyz(zero, zero, half_pi)
    base_position_w = torch.tensor([[1.0, 2.0, 3.0]])
    goal_position_w = torch.tensor([[1.2, 2.3, 3.1]])

    observation = goal_pose_in_base_frame(
        goal_position_w,
        base_rotation_w,
        base_position_w,
        base_rotation_w,
    )

    expected = torch.tensor([[0.3, -0.2, 0.1, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    torch.testing.assert_close(observation, expected, atol=1.0e-6, rtol=0.0)


@pytest.mark.integration
@pytest.mark.parametrize("physics_preset", ["newton_mjwarp", "isaacsim_physx"])
@pytest.mark.parametrize("hand_type", ["wuji", "inspire", "dex3"])
def test_observation_uses_raw_joint_positions_and_command_limits(physics_preset: str, hand_type: str) -> None:
    """Joint order and padding stay correct across different simulator topologies."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=(physics_preset,))
    env_cfg.hand_type = hand_type
    env_cfg.scene.num_envs = 4
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = False
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        env.reset(seed=42)
        unwrapped = env.unwrapped
        robot = unwrapped.robot
        observations = unwrapped._get_observations()

        assert observations["policy"].shape == (4, 191)
        assert observations["student"].shape == (4, 161)
        assert observations["goal"].shape == (4, 9)
        assert torch.equal(observations["policy"][:, :161], observations["student"])
        assert torch.equal(observations["student"][:, 3:10], robot.data.joint_pos.torch[:, unwrapped.arm_joint_ids])
        slots = list(get_hand_spec(hand_type).slot_indices)
        absent = [slot for slot in range(20) if slot not in slots]
        assert torch.equal(
            observations["student"][:, 10:30][:, slots], robot.data.joint_pos.torch[:, unwrapped.hand_joint_ids]
        )
        assert torch.count_nonzero(observations["student"][:, 10:30][:, absent]) == 0
        assert torch.count_nonzero(observations["student"][:, 40:60][:, absent]) == 0
        assert torch.count_nonzero(observations["student"][:, 67:87][:, absent]) == 0
        assert torch.count_nonzero(observations["student"][:, 101:141].reshape(4, 20, 2)[:, absent]) == 0
        expected_mask = torch.zeros((4, 20), device=unwrapped.device)
        expected_mask[:, slots] = 1
        assert torch.equal(observations["student"][:, 141:161], expected_mask)

        shoulder_roll_joint_id = unwrapped.arm_joint_ids[1]
        expected_shoulder_roll = torch.full_like(
            robot.data.default_joint_pos.torch[:, shoulder_roll_joint_id], -torch.deg2rad(torch.tensor(110.0))
        )
        assert torch.allclose(robot.data.default_joint_pos.torch[:, shoulder_roll_joint_id], expected_shoulder_roll)
        assert torch.allclose(robot.data.joint_pos.torch[:, shoulder_roll_joint_id], expected_shoulder_roll)

        arm_limits = robot.data.soft_joint_pos_limits.torch[:, unwrapped.arm_joint_ids]
        wuji_asset_limits = robot.data.soft_joint_pos_limits.torch[:, unwrapped.hand_joint_ids]
        wuji_command_limits = torch.stack(
            (
                torch.maximum(wuji_asset_limits[..., 0], unwrapped._hand_command_lower_floor),
                wuji_asset_limits[..., 1],
            ),
            dim=-1,
        )
        expected_command_limits = torch.cat(
            (arm_limits.flatten(start_dim=1), wuji_command_limits.flatten(start_dim=1)), dim=-1
        )
        observed_command_limits = torch.cat(
            (
                observations["student"][:, 87:101],
                observations["student"][:, 101:141].reshape(4, 20, 2)[:, slots].flatten(1),
            ),
            dim=1,
        )
        assert torch.equal(observed_command_limits, expected_command_limits)

        zero_floor = unwrapped._hand_command_lower_floor == 0.0
        if hand_type == "wuji":
            assert (wuji_asset_limits[..., 0][:, zero_floor] < 0.0).any()
        assert torch.all(wuji_command_limits[..., 0][:, zero_floor] >= 0.0)
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("physics_preset", ["newton_mjwarp", "isaacsim_physx"])
def test_virtual_force_packet_uses_common_contact_and_jacobian_apis(physics_preset: str) -> None:
    """Catch backend-specific sensor wiring, body ordering, and packet-shape regressions."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=(physics_preset,))
    env_cfg.scene.num_envs = 2
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = False
    env_cfg.virtual_force.enabled = True
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        observations, _ = env.reset(seed=42)
        unwrapped = env.unwrapped
        observations, _, _, _, _ = env.step(torch.zeros((2, unwrapped.cfg.action_space), device=unwrapped.device))
        output = unwrapped.get_virtual_force_output()

        assert output is not None
        assert observations["student"].shape == (2, 161)
        assert observations["goal"].shape == (2, 9)
        assert observations["force"].shape == (2, 20)
        assert output.ideal_actuator_torque_nm.shape == (2, 20)
        assert output.observed_actuator_torque_nm.shape == (2, 20)
        assert output.valid.shape == (2,)
        assert torch.isfinite(output.ideal_actuator_torque_nm).all()
        assert torch.isfinite(output.observed_actuator_torque_nm).all()
        assert sum(sensor.num_sensors for sensor in unwrapped.contact_sensors.values()) == 18
        for sensor in unwrapped.force_sensors.values():
            assert sensor.data.contact_pos_w is not None
            assert sensor.data.friction_force_matrix_w is not None
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("hand_type", ["wuji", "inspire", "dex3"])
@pytest.mark.parametrize("physics_preset", ["newton_mjwarp", "isaacsim_physx"])
def test_every_hand_body_sensor_reports_real_target_contact_and_clears(hand_type: str, physics_preset: str) -> None:
    """Physical probes catch missing body bindings, wrong filters, stale data, and cross-env leaks."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Hand-Table-Direct", "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=(physics_preset,))
    env_cfg.hand_type = hand_type
    env_cfg.scene.num_envs = 2
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = False
    env_cfg.adr.enabled = False
    env_cfg.virtual_force.enabled = hand_type == "wuji"
    env_cfg.object_cfg.spawn = sim_utils.SphereCfg(
        radius=0.002,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=env_cfg.object_cfg.spawn.collision_props,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
    )
    env = gym.make("CrossEmbodimentCl-G1-Hand-Table-Direct", cfg=env_cfg)
    try:
        env.reset(seed=42)
        task = env.unwrapped
        spec = get_hand_spec(hand_type)
        authored = Usd.Stage.Open(task.robot.cfg.spawn.usd_path)
        transforms = UsdGeom.XformCache()
        pose = torch.zeros((2, 7), device=task.device)
        pose[:, 3] = 1
        away = task.scene.env_origins + pose.new_tensor([0.0, 0.0, 2.0])

        def probe_step(position):
            pose[:, :3] = away
            pose[0, :3] = position
            task.apple.write_root_pose_to_sim_index(root_pose=pose)
            task.apple.write_root_velocity_to_sim_index(root_velocity=torch.zeros((2, 6), device=task.device))
            task.scene.write_data_to_sim()
            task.sim.step(render=False)
            task.scene.update(task.physics_dt)

        maxima = {}
        for group, bodies in spec.contact_body_groups.items():
            for index, body_path in enumerate(bodies):
                sensor_name = f"{group}__{index}"
                sensor = task.contact_sensors[sensor_name]
                assert sensor.cfg.prim_path.endswith(f"/G1Wuji/{body_path}")
                body = authored.GetPrimAtPath(authored.GetDefaultPrim().GetPath().AppendPath(body_path))

                def belongs_to_body(prim, body=body):
                    parent = prim.GetParent()
                    while parent.IsValid() and parent != body:
                        if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                            return False
                        parent = parent.GetParent()
                    return parent == body

                colliders = [
                    prim
                    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                    if prim.HasAPI(UsdPhysics.CollisionAPI)
                    and prim.GetAttribute("physics:collisionEnabled").Get() is not False
                    and belongs_to_body(prim)
                ]
                assert colliders, f"No authored collider on declared sensor body {body_path}"
                local_points = []
                for collider in colliders:
                    if collider.IsA(UsdGeom.Mesh):
                        points = UsdGeom.Mesh(collider).GetPointsAttr().Get()
                        # Surface overlap avoids the degenerate sphere-at-hull-center case.
                        candidates = [Gf.Vec3d(points[i]) for i in range(0, len(points), max(1, len(points) // 8))]
                        candidates.append(sum((Gf.Vec3d(point) for point in points), Gf.Vec3d()) / len(points))
                    else:
                        assert collider.IsA(UsdGeom.Capsule), f"Unsupported probe collider {collider.GetPath()}"
                        radius = collider.GetAttribute("radius").Get()
                        axis = {"X": 0, "Y": 1, "Z": 2}[collider.GetAttribute("axis").Get()]
                        radial, second_radial, axial = Gf.Vec3d(), Gf.Vec3d(), Gf.Vec3d()
                        radial[(axis + 1) % 3] = radius
                        second_radial[(axis + 2) % 3] = radius
                        axial[axis] = radius + collider.GetAttribute("height").Get() / 2
                        candidates = [radial, -radial, second_radial, -second_radial, axial, -axial, Gf.Vec3d()]
                    for point in candidates:
                        local = (
                            transforms.GetLocalToWorldTransform(body)
                            .GetInverse()
                            .Transform(transforms.GetLocalToWorldTransform(collider).Transform(point))
                        )
                        local_points.append(pose.new_tensor(tuple(local)).unsqueeze(0))
                body_id = task.robot.body_names.index(body.GetName())
                maximum = 0.0
                for attempt in range(4 * len(local_points)):
                    position = (
                        task.robot.data.body_pos_w.torch[0, body_id]
                        + quat_apply(
                            task.robot.data.body_quat_w.torch[0, body_id].unsqueeze(0), local_points[attempt // 4]
                        )[0]
                    )
                    probe_step(position)
                    expected_groups = []
                    for expected_group, expected_bodies in spec.contact_body_groups.items():
                        names = [f"{expected_group}__{i}" for i in range(len(expected_bodies))]
                        magnitudes = [
                            torch.linalg.vector_norm(
                                task.contact_sensors[name].data.normal_force_matrix_w.torch, dim=-1
                            ).sum(dim=(1, 2))
                            for name in names
                        ]
                        expected_groups.append(torch.stack(magnitudes).sum(dim=0))
                    expected_groups = torch.stack(expected_groups, dim=1)
                    assert torch.isfinite(expected_groups).all()
                    assert torch.count_nonzero(expected_groups[1]) == 0
                    torch.testing.assert_close(task._contact_group_forces(), expected_groups)
                    expected_observation = pose.new_zeros((2, 6))
                    for column, name in enumerate(spec.contact_body_groups):
                        expected_observation[:, ("palm", "thumb", "index", "middle", "ring", "little").index(name)] = (
                            expected_groups[:, column].clamp_max(task.cfg.contact_force_observation_max).log1p()
                        )
                    torch.testing.assert_close(task._get_observations()["policy"][:, -6:], expected_observation)
                    maximum = max(maximum, torch.linalg.vector_norm(sensor.data.normal_force_matrix_w.torch[0]).item())
                    if hand_type == "wuji" and sensor.data.normal_force_matrix_w.torch[0].abs().sum() > 1.0e-5:
                        detailed = task.force_sensors[sensor_name].data
                        assert torch.isfinite(detailed.normal_force_matrix_w.torch).all()
                        assert torch.linalg.vector_norm(detailed.normal_force_matrix_w.torch[0]) > 1.0e-5
                        assert torch.isfinite(detailed.friction_force_matrix_w.torch).all()
                    if maximum > 1.0e-5:
                        break
                assert maximum > 1.0e-5, f"No physical target contact detected on {hand_type}/{body_path}"
                maxima[sensor_name] = maximum
                for _ in range(3):
                    probe_step(away[0])
                assert torch.count_nonzero(task._contact_group_forces()) == 0, f"Stale contact after {body_path}"
        print(f"{hand_type}/{physics_preset} real contact peak [N]: {maxima}")
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("hand_type", ["wuji", "inspire", "dex3"])
def test_depth_view_publishes_the_exact_final_student_image(hand_type: str) -> None:
    """The viewer-facing camera output is the finalized student tensor in NHWC layout."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=("depth_view",))
    env_cfg.hand_type = hand_type
    env_cfg.sim.visualizer_cfgs = []
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        env.reset(seed=42)
        unwrapped = env.unwrapped
        observations = unwrapped._get_observations()
        preview = unwrapped.depth_camera.data.output["rgb"].torch
        native = unwrapped.depth_camera.data.output["distance_to_image_plane"].torch

        assert observations["camera"].shape == (1, 1, 224, 224)
        assert observations["student"].shape == (1, 161)
        assert torch.equal(observations["policy"][:, :161], observations["student"])
        assert "force" not in observations
        assert preview.shape == (1, 224, 224, 3)
        assert native.shape == (1, 127, 224, 1)
        expected = normalized_depth_to_grayscale(observations["camera"].permute(0, 2, 3, 1)).expand(-1, -1, -1, 3)
        assert torch.equal(preview, expected)
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("hand_type", ["wuji", "inspire", "dex3"])
def test_sensor_noise_is_actor_only_and_critic_gets_dr_state(hand_type: str) -> None:
    """Sensor noise stays actor-only while the critic receives clean state plus DR metadata."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=("newton_mjwarp",))
    env_cfg.hand_type = hand_type
    env_cfg.scene.num_envs = 2
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = False
    env_cfg.adr.enabled = True
    env_cfg.adr.extra_enabled = True
    env_cfg.adr.sensor_noise_enabled = True
    env_cfg.adr.initial_level = env_cfg.adr.max_level
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        env.reset(seed=42)
        unwrapped = env.unwrapped
        observations = unwrapped._get_observations()
        assert torch.equal(
            observations["critic"][:, 3:10],
            unwrapped.robot.data.joint_pos.torch[:, unwrapped.arm_joint_ids],
        )
        assert observations["policy"].shape == (2, 191)
        assert observations["critic"].shape == (2, 267)
        assert torch.allclose(observations["critic"][:, 191], torch.ones(2, device=unwrapped.device))
        assert torch.count_nonzero(observations["critic"][:, 194:197]) == 0
        expected_root_position = unwrapped.robot.data.default_root_pose.torch[:, :3] + unwrapped.scene.env_origins
        torch.testing.assert_close(unwrapped.robot.data.root_pos_w.torch, expected_root_position)
        assert torch.equal(observations["policy"][:, :161], observations["student"])
        spec = get_hand_spec(hand_type)
        absent = [slot for slot in range(20) if slot not in spec.slot_indices]
        for group in ("policy", "critic", "student"):
            assert torch.count_nonzero(observations[group][:, 10:30][:, absent]) == 0
            assert torch.count_nonzero(observations[group][:, 40:60][:, absent]) == 0
            assert torch.equal(observations[group][:, 141:161], observations["student"][:, 141:161])
        assert not torch.equal(observations["policy"], observations["critic"])
    finally:
        env.close()


@pytest.mark.integration
def test_nonfinite_state_returns_zero_terminal_reward() -> None:
    """A state rejected by the done guard must not leak a NaN reward to the trainer."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg)
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = False
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        env.reset(seed=42)
        unwrapped = env.unwrapped
        unwrapped.object.data.root_pos_w.torch[0, 0] = torch.nan

        terminated, _ = unwrapped._get_dones()
        reward = unwrapped._get_rewards()

        assert terminated.item()
        assert unwrapped._termination_nonfinite.item()
        assert torch.equal(reward, torch.zeros_like(reward))
    finally:
        env.close()


@pytest.mark.integration
def test_wuji_latent_round_trip_ping_pong() -> None:
    """Project two simulated poses and verify their latent commands move the hand."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg)
    # The default preset opens the Newton viewer; keep this runtime test headless.
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = True
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        env.reset(seed=42)
        robot = unwrapped.robot
        joint_ids = unwrapped.hand_joint_ids
        assert unwrapped.goal_keypoint_marker is not None
        assert unwrapped.object_keypoint_marker is not None
        assert unwrapped.adr_spawn_area_marker is not None
        assert unwrapped.local_cube_keypoints.shape == (8, 3)
        observations = unwrapped._get_observations()
        assert set(observations) == {"policy", "critic", "student", "goal"}
        assert observations["policy"].shape == (1, 191)
        assert observations["student"].shape == (1, 161)
        assert observations["goal"].shape == (1, 9)
        assert torch.isfinite(observations["policy"]).all()
        assert observations["critic"].shape == (1, 267)
        assert torch.equal(observations["critic"][:, :191], observations["policy"])
        assert unwrapped.torso_contact_sensor.data.normal_force_matrix_w.torch.shape == (1, 1, 1, 3)
        terminated, timed_out = unwrapped._get_dones()
        assert terminated.shape == (1,)
        assert timed_out.shape == (1,)
        rewards = unwrapped._get_rewards()
        assert rewards.shape == (1,)
        assert torch.isfinite(rewards).all()
        _, _, _, _, default_extras = env.step(torch.zeros((1, unwrapped.cfg.action_space), device=unwrapped.device))
        assert not any(key.startswith("Control/") for key in default_extras["log"])
        unwrapped.cfg.log_control_metrics = True
        unwrapped.episode_length_buf.fill_(unwrapped.max_episode_length - 1)
        _, _, _, _, extras = env.step(torch.zeros((1, unwrapped.cfg.action_space), device=unwrapped.device))
        expected_log_keys = {
            "Reward/reach_step",
            "Reward/goal_step",
            "Reward/contact_step",
            "Reward/reach_ep_return",
            "Reward/goal_ep_return",
            "Reward/contact_ep_return",
            "Reach/hand_distance_farthest_ep_min",
            "Task/keypoint_error_ep_final",
            "Task/keypoint_error_ep_min",
            "Task/object_height_ep_max",
            "Contact/gate_frac_ep",
            "Task/success",
            "Control/action_saturation_frac_step",
            "Control/arm_tracking_error_step",
            "Control/hand_tracking_error_step",
            "Control/arm_tracking_error_ep",
            "Control/hand_tracking_error_ep",
            "Control/arm_target_rate_step",
            "Control/arm_joint_velocity_step",
            "Control/arm_computed_effort_step",
            "Control/arm_applied_effort_step",
            "Control/arm_effort_saturation_frac_step",
            "Control/arm_anti_windup_frac_step",
            "Contact/penetration_elbow_torso_step",
            "Contact/elbow_torso_penetrating_frac_step",
            "Contact/penetration_elbow_torso_ep_max",
            "Terminations/timeout",
        }
        for joint_name in unwrapped._ARM_JOINT_NAMES:
            expected_log_keys.update(
                {
                    f"Control/arm_target_rate_{joint_name}_step",
                    f"Control/arm_tracking_error_{joint_name}_step",
                    f"Control/arm_joint_velocity_{joint_name}_step",
                    f"Control/arm_computed_effort_{joint_name}_step",
                    f"Control/arm_applied_effort_{joint_name}_step",
                    f"Control/arm_effort_saturation_{joint_name}_frac_step",
                    f"Control/arm_anti_windup_{joint_name}_frac_step",
                }
            )
        assert expected_log_keys <= extras["log"].keys()
        assert extras["log"]["Task/success"].item() == 0.0
        assert extras["log"]["Terminations/timeout"].item() == 1.0
        arm_actions = torch.zeros((1, unwrapped.cfg.action_space), device=unwrapped.device)
        arm_actions[:, 0] = 1.0
        arm_actions[:, 1] = -1.0
        arm_default = robot.data.default_joint_pos.torch[:, unwrapped.arm_joint_ids].clone()
        arm_limits = robot.data.soft_joint_pos_limits.torch[:, unwrapped.arm_joint_ids]
        arm_desired = unscale_transform(
            arm_actions[:, : len(unwrapped.arm_joint_ids)], arm_limits[..., 0], arm_limits[..., 1]
        )
        assert torch.equal(arm_desired[:, 0], arm_limits[:, 0, 1])
        assert torch.equal(arm_desired[:, 1], arm_limits[:, 1, 0])
        max_step = unwrapped.cfg.arm_joint_velocity_limit * unwrapped.step_dt
        expected_arm_target = arm_default + torch.clamp(
            unwrapped.cfg.arm_action_ema_alpha * (arm_desired - arm_default), -max_step, max_step
        )
        max_lead = (
            robot.data.joint_effort_limits.torch[:, unwrapped.arm_joint_ids]
            + robot.data.joint_damping.torch[:, unwrapped.arm_joint_ids] * unwrapped.cfg.arm_joint_velocity_limit
        ) / robot.data.joint_stiffness.torch[:, unwrapped.arm_joint_ids].clamp_min(1.0e-6)
        arm_position = robot.data.joint_pos.torch[:, unwrapped.arm_joint_ids]
        expected_arm_target.clamp_(min=arm_position - max_lead, max=arm_position + max_lead)
        unwrapped._pre_physics_step(arm_actions)
        assert torch.allclose(unwrapped.arm_joint_targets, expected_arm_target)
        wuji_default = robot.data.default_joint_pos.torch[:, joint_ids].clone()
        wuji_limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
        wuji_desired = unwrapped.hand_action_pipeline.latent_action_to_joint_target(
            arm_actions[:, len(unwrapped.arm_joint_ids) :],
            torch.maximum(wuji_limits[..., 0], unwrapped._hand_command_lower_floor),
            wuji_limits[..., 1],
        )
        wuji_max_step = unwrapped.cfg.hand_joint_velocity_limit * unwrapped.step_dt
        expected_wuji_target = wuji_default + torch.clamp(
            unwrapped.cfg.hand_action_ema_alpha * (wuji_desired - wuji_default),
            -wuji_max_step,
            wuji_max_step,
        )
        wuji_max_lead = (
            robot.data.joint_effort_limits.torch[:, joint_ids]
            + robot.data.joint_damping.torch[:, joint_ids] * unwrapped.cfg.hand_joint_velocity_limit
        ) / robot.data.joint_stiffness.torch[:, joint_ids].clamp_min(1.0e-6)
        wuji_position = robot.data.joint_pos.torch[:, joint_ids]
        expected_wuji_target.clamp_(min=wuji_position - wuji_max_lead, max=wuji_position + wuji_max_lead)
        expected_wuji_target.clamp_(min=torch.maximum(wuji_limits[..., 0], unwrapped._hand_command_lower_floor))
        assert torch.allclose(unwrapped.hand_joint_targets, expected_wuji_target)
        env.reset(seed=42)
        limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
        lower_limits, upper_limits = limits[..., 0], limits[..., 1]
        generator = torch.Generator(device=unwrapped.device).manual_seed(42)
        sampled_positions = lower_limits + (upper_limits - lower_limits) * (
            0.2 + 0.6 * torch.rand((2, len(joint_ids)), device=unwrapped.device, generator=generator)
        )

        projections = []
        for sampled_position in sampled_positions:
            sampled_position = sampled_position.unsqueeze(0)
            # Author a valid simulated pose, then use its read-back value as
            # the source for the Wuji-to-MANO manifold projection.
            robot.write_joint_position_to_sim_index(position=sampled_position, joint_ids=joint_ids)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(sampled_position), joint_ids=joint_ids)
            unwrapped.sim.step(render=False)
            source_position = robot.data.joint_pos.torch[:, joint_ids].clone()
            projections.append(
                unwrapped.hand_action_pipeline.project_joint_positions(
                    source_position, lower_limits, upper_limits, steps=512
                )
            )

        policy_actions = torch.zeros((1, unwrapped.cfg.action_space), device=unwrapped.device)
        settled_errors = []
        for projection in (projections[1], projections[0], projections[1]):
            # Keep the production eight-second timeout while allowing the
            # diagnostic's three four-second legs to retain physical state.
            unwrapped.episode_length_buf.zero_()
            policy_actions[:, -projection.latent_action.shape[1] :] = projection.latent_action
            initial_error = torch.linalg.vector_norm(
                robot.data.joint_pos.torch[:, joint_ids] - projection.joint_target, dim=-1
            )
            for _ in range(240):
                env.step(policy_actions)
            settled_error = torch.linalg.vector_norm(
                robot.data.joint_pos.torch[:, joint_ids] - projection.joint_target, dim=-1
            )
            settled_errors.append(settled_error)
            assert torch.isfinite(settled_error).all()
            assert torch.all(settled_error < initial_error)

        projection_error = torch.stack([projection.error for projection in projections])
        assert torch.isfinite(projection_error).all()
        print(
            "Wuji latent round trip [rad]: "
            f"projection_l2={projection_error.flatten().cpu().tolist()}, "
            f"settled_tracking_l2={torch.cat(settled_errors).cpu().tolist()}"
        )
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("hand_type", ["wuji", "inspire", "dex3"])
@pytest.mark.parametrize("physics_preset", ["newton_mjwarp", "isaacsim_physx"])
def test_wuji_multi_env_reset_initializes_ema_targets(hand_type: str, physics_preset: str) -> None:
    """Reset clears controller history and stepping preserves the fixed pelvis."""
    env_cfg = load_cfg_from_registry("CrossEmbodimentCl-G1-Wuji-Table-Direct", "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=(physics_preset,))
    env_cfg.hand_type = hand_type
    env_cfg.scene.num_envs = 2
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.debug.keypoint_markers = False
    env = gym.make("CrossEmbodimentCl-G1-Wuji-Table-Direct", cfg=env_cfg)
    try:
        observations, _ = env.reset(seed=42)
        unwrapped = env.unwrapped
        default_joint_pos = unwrapped.robot.data.default_joint_pos.torch
        assert observations["policy"].shape == (2, 191)
        assert torch.equal(unwrapped.arm_joint_targets, default_joint_pos[:, unwrapped.arm_joint_ids])
        assert torch.equal(unwrapped.hand_joint_targets, default_joint_pos[:, unwrapped.hand_joint_ids])
        spec = get_hand_spec(hand_type)
        authored = Usd.Stage.Open(unwrapped.robot.cfg.spawn.usd_path)
        joints = {
            prim.GetName(): prim
            for prim in authored.Traverse()
            if prim.GetName() in spec.joint_names + spec.follower_joint_names
        }
        for name, prim in joints.items():
            joint_id = unwrapped.robot.joint_names.index(name)
            expected = torch.deg2rad(
                torch.tensor(prim.GetAttribute("state:angular:physics:position").Get(), device=unwrapped.device)
            )
            torch.testing.assert_close(unwrapped.robot.data.joint_pos.torch[:, joint_id], expected.expand(2))
            if hand_type != "wuji":
                # New hands inherit USD drive gains: catch lost inheritance or
                # degree/radian conversion errors in either backend.
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                for field, attribute in (("stiffness", drive.GetStiffnessAttr()), ("damping", drive.GetDampingAttr())):
                    expected_gain = torch.rad2deg(torch.tensor(attribute.Get(), device=unwrapped.device))
                    actual = getattr(unwrapped.robot.data, f"joint_{field}").torch[:, joint_id]
                    torch.testing.assert_close(actual, expected_gain.expand(2))
        follower_ids = [unwrapped.robot.joint_names.index(name) for name in spec.follower_joint_names]
        assert torch.count_nonzero(unwrapped.robot.data.joint_stiffness.torch[:, follower_ids]) == 0
        assert torch.count_nonzero(unwrapped.robot.data.joint_damping.torch[:, follower_ids]) == 0
        root_position = unwrapped.robot.data.root_pos_w.torch.clone()
        actions = torch.linspace(-1, 1, 25, device=unwrapped.device).repeat(2, 1)
        for _ in range(32):
            observations, reward, _, _, _ = env.step(actions)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in observations.values())
        torch.testing.assert_close(unwrapped.robot.data.root_pos_w.torch, root_position, atol=1e-6, rtol=0)
        if spec.follower_joint_names:
            # Hold the final physical targets while the soft mimic constraint settles.
            unwrapped._apply_action()
            for _ in range(240):
                unwrapped.scene.write_data_to_sim()
                unwrapped.sim.step(render=False)
                unwrapped.scene.update(unwrapped.physics_dt)
        for name in spec.follower_joint_names:
            prim = joints[name]
            leader = prim.GetRelationship("newton:mimicJoint").GetTargets()[0].name
            multiplier = prim.GetAttribute("newton:mimicCoef1").Get()
            offset_degrees = prim.GetAttribute("newton:mimicCoef0").Get() or 0.0
            position = unwrapped.robot.data.joint_pos.torch
            expected = multiplier * position[:, unwrapped.robot.joint_names.index(leader)]
            expected += torch.deg2rad(torch.tensor(offset_degrees, device=unwrapped.device))
            follower_position = position[:, unwrapped.robot.joint_names.index(name)]
            torch.testing.assert_close(
                follower_position,
                expected,
                atol=0.002,
                rtol=0,
                msg=f"{name} settled mimic residual [rad]: {(follower_position - expected).tolist()}",
            )
        env.reset(seed=42)
        assert torch.equal(unwrapped.hand_joint_targets, default_joint_pos[:, unwrapped.hand_joint_ids])
    finally:
        env.close()
