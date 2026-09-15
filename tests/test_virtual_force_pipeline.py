# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Torch contracts for the virtual Wuji force pipeline."""

import numpy as np
import pytest
import torch

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.virtual_force import (
    VirtualForcePipeline,
    VirtualForcePipelineConfig,
    WujiForceSystemIdModel,
    contact_wrenches_to_actuator_torque,
)


@pytest.mark.unit
def test_contact_wrench_projects_force_and_link_origin_moment() -> None:
    """Catch loss of either the linear-force or angular-moment contribution."""
    torque = contact_wrenches_to_actuator_torque(
        torch.tensor([[[4.0, 0.0, 0.0]]]),
        torch.tensor([[[[2.0], [0.0], [0.0]]]]),
        torch.tensor([[[0.0, 0.0, 5.0]]]),
        torch.tensor([[[[0.0], [0.0], [3.0]]]]),
    )
    torch.testing.assert_close(torque, torch.tensor([[23.0]]))


@pytest.mark.unit
def test_pipeline_applies_120hz_updates_before_packet_latency() -> None:
    """Two held 120 Hz updates per policy step must advance both EMA and latency."""
    pipeline = VirtualForcePipeline(
        num_envs=1,
        config=VirtualForcePipelineConfig(
            torque_lpf_alpha=0.5,
            latency_steps_range=(1, 1),
            updates_per_step=2,
        ),
        device="cpu",
    )
    jacobian = torch.zeros(1, 1, 3, 20)
    jacobian[0, 0, 0, 0] = 1.0

    def step(value: float):
        return pipeline.step(
            contact_forces_w=torch.tensor([[[value, 0.0, 0.0]]]),
            contact_linear_jacobians_w=jacobian,
            contact_moments_w=torch.zeros(1, 1, 3),
            contact_angular_jacobians_w=torch.zeros_like(jacobian),
            joint_position=torch.zeros(1, 20),
            joint_velocity=torch.zeros(1, 20),
            joint_command=torch.zeros(1, 20),
        )

    first = step(1.0)
    second = step(3.0)
    torch.testing.assert_close(first.observed_actuator_torque_nm[0, 0], torch.tensor(1.0))
    # Held samples 3, 3 produce filtered values 2 and 2.5; one sensor-tick latency returns 2.
    torch.testing.assert_close(second.observed_actuator_torque_nm[0, 0], torch.tensor(2.0))
    assert second.observed_actuator_torque_nm.shape == (1, 20)
    assert bool(second.valid[0])


@pytest.mark.unit
def test_wuji_system_id_applies_baseline_coupling_and_residual_mean(tmp_path) -> None:
    """Validate the checked-in NPZ schema and actuator/model order conversion."""
    names = [f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]
    coefficients = np.zeros((5, 4, 13), dtype=np.float32)
    coefficients[:, :, 0] = np.arange(1, 21, dtype=np.float32).reshape(5, 4)
    model_path = tmp_path / "model.npz"
    np.savez_compressed(
        model_path,
        schema_version=np.asarray("wuji_force_model_v1"),
        joint_names=np.asarray(names),
        baseline_feature_mean=np.zeros((5, 12), dtype=np.float32),
        baseline_feature_scale=np.ones((5, 12), dtype=np.float32),
        baseline_output_by_feature=coefficients,
        coupling_output_by_input=np.tile(2.0 * np.eye(4, dtype=np.float32), (5, 1, 1)),
        coupling_fitted=np.asarray(True),
        delay_steps=np.asarray(0),
        residual_mean_nm=np.full(20, 0.5, dtype=np.float32),
        residual_covariance_nm2=np.zeros((20, 20), dtype=np.float32),
        residual_lag1_correlation=np.zeros(20, dtype=np.float32),
    )
    model = WujiForceSystemIdModel(
        model_path,
        actuator_joint_names=names,
        num_envs=1,
        device="cpu",
        sample_residual=False,
    )
    output = model.step(
        torch.ones(1, 20),
        joint_position_actuator_order=torch.zeros(1, 20),
        joint_velocity_actuator_order=torch.zeros(1, 20),
        command_actuator_order=torch.zeros(1, 20),
    )
    torch.testing.assert_close(output, torch.arange(1, 21, dtype=torch.float32).reshape(1, 20) + 2.5)
