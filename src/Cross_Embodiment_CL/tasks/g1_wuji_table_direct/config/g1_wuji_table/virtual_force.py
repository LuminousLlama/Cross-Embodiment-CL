# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Virtual Wuji actuator-torque sensing for sim-to-real distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

WUJI_JOINT_COUNT = 20


@dataclass(frozen=True)
class VirtualForcePipelineConfig:
    """Sensor effects applied to ideal contact torque.

    Ranges are sampled once per episode. Noise and packet dropout are sampled
    once per virtual sensor update.
    """

    torque_scale_range: tuple[float, float] = (1.0, 1.0)
    torque_bias_range_nm: tuple[float, float] = (0.0, 0.0)
    torque_noise_std_nm: float = 0.0
    torque_lpf_alpha: float = 0.2
    latency_steps_range: tuple[int, int] = (0, 0)
    packet_dropout_probability: float = 0.0
    torque_clip_abs_nm: float | None = None
    contact_force_sign: float = 1.0
    updates_per_step: int = 1

    def validate(self) -> None:
        """Validate the virtual sensor configuration."""
        for name, bounds in (
            ("torque_scale_range", self.torque_scale_range),
            ("torque_bias_range_nm", self.torque_bias_range_nm),
            ("latency_steps_range", self.latency_steps_range),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be an ordered (low, high) pair.")
        if self.torque_scale_range[0] <= 0.0:
            raise ValueError("torque_scale_range must stay positive.")
        if self.latency_steps_range[0] < 0:
            raise ValueError("latency_steps_range cannot be negative.")
        if self.torque_noise_std_nm < 0.0:
            raise ValueError("torque_noise_std_nm cannot be negative.")
        if not 0.0 < self.torque_lpf_alpha <= 1.0:
            raise ValueError("torque_lpf_alpha must be in (0, 1].")
        if not 0.0 <= self.packet_dropout_probability <= 1.0:
            raise ValueError("packet_dropout_probability must be in [0, 1].")
        if self.torque_clip_abs_nm is not None and self.torque_clip_abs_nm <= 0.0:
            raise ValueError("torque_clip_abs_nm must be positive when supplied.")
        if self.contact_force_sign not in (-1.0, 1.0):
            raise ValueError("contact_force_sign must be either -1 or +1.")
        if self.updates_per_step < 1:
            raise ValueError("updates_per_step must be positive.")


class VirtualForceOutput(NamedTuple):
    """One virtual 20-actuator Wuji torque packet."""

    ideal_actuator_torque_nm: torch.Tensor
    observed_actuator_torque_nm: torch.Tensor
    valid: torch.Tensor


def contact_wrenches_to_actuator_torque(
    contact_forces_w: torch.Tensor,
    contact_linear_jacobians_w: torch.Tensor,
    contact_moments_w: torch.Tensor | None = None,
    contact_angular_jacobians_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project per-link contact wrenches into actuator torque [N m].

    Forces and moments have shape ``[N, S, 3]``. Jacobians have shape
    ``[N, S, 3, A]``, where ``A`` is the number of physical actuators.
    """
    if contact_forces_w.ndim != 3 or contact_forces_w.shape[-1] != 3:
        raise ValueError("contact_forces_w must have shape [N, S, 3].")
    if contact_linear_jacobians_w.shape[:3] != contact_forces_w.shape:
        raise ValueError("contact_linear_jacobians_w must have shape [N, S, 3, A].")
    torque = torch.einsum("nsia,nsi->na", contact_linear_jacobians_w, contact_forces_w)
    if (contact_moments_w is None) != (contact_angular_jacobians_w is None):
        raise ValueError("contact_moments_w and contact_angular_jacobians_w must be provided together.")
    if contact_moments_w is not None:
        if contact_moments_w.shape != contact_forces_w.shape:
            raise ValueError("contact_moments_w must have shape [N, S, 3].")
        if contact_angular_jacobians_w.shape != contact_linear_jacobians_w.shape:
            raise ValueError("contact_angular_jacobians_w must match the linear Jacobian shape.")
        torque = torque + torch.einsum("nsia,nsi->na", contact_angular_jacobians_w, contact_moments_w)
    return torque


class WujiForceSystemIdModel:
    """Map simulated contact torque to a fitted real-estimator packet."""

    def __init__(
        self,
        path: str | Path,
        *,
        actuator_joint_names: list[str],
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        sample_residual: bool = True,
        random_seed: int = 0,
    ) -> None:
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
        if str(values.get("schema_version", "")) != "wuji_force_model_v1":
            raise ValueError(f"{source} is not a wuji_force_model_v1 artifact.")
        if not bool(np.asarray(values.get("coupling_fitted", False)).item()):
            raise ValueError(f"{source} does not contain a fitted contact coupling.")

        model_names = np.asarray(values["joint_names"]).astype(str).tolist()
        if len(model_names) != WUJI_JOINT_COUNT or len(set(model_names)) != WUJI_JOINT_COUNT:
            raise ValueError("Force system-ID model must contain 20 unique joints.")
        missing = sorted(set(actuator_joint_names) - set(model_names))
        extra = sorted(set(model_names) - set(actuator_joint_names))
        if missing or extra:
            raise ValueError(f"Force system-ID/model actuator mismatch: missing={missing}, extra={extra}.")
        actuator_to_model = [model_names.index(name) for name in actuator_joint_names]

        self.source = source
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.sample_residual = bool(sample_residual)
        self.actuator_to_model = torch.as_tensor(actuator_to_model, dtype=torch.long, device=self.device)
        self.model_to_actuator = torch.argsort(self.actuator_to_model)

        def tensor(name: str, shape: tuple[int, ...]) -> torch.Tensor:
            array = np.asarray(values[name], dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{source}: {name} must be finite with shape {shape}.")
            return torch.as_tensor(array, dtype=dtype, device=self.device)

        self.feature_mean = tensor("baseline_feature_mean", (5, 12))
        self.feature_scale = tensor("baseline_feature_scale", (5, 12))
        self.coefficients = tensor("baseline_output_by_feature", (5, 4, 13))
        self.coupling = tensor("coupling_output_by_input", (5, 4, 4))
        self.residual_mean = tensor("residual_mean_nm", (WUJI_JOINT_COUNT,))
        covariance = tensor("residual_covariance_nm2", (WUJI_JOINT_COUNT, WUJI_JOINT_COUNT))
        self.lag1 = tensor("residual_lag1_correlation", (WUJI_JOINT_COUNT,)).clamp(-0.999, 0.999)

        innovation_scale = torch.sqrt((1.0 - self.lag1.square()).clamp_min(0.0))
        innovation_covariance = innovation_scale[:, None] * covariance * innovation_scale[None, :]
        eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (innovation_covariance + innovation_covariance.T))
        self.innovation_factor = eigenvectors @ torch.diag(torch.sqrt(eigenvalues.clamp_min(0.0)))
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(random_seed))
        self.residual_state = self.residual_mean.expand(self.num_envs, -1).clone()

        self.delay_steps = int(np.asarray(values["delay_steps"]).item())
        if self.delay_steps < 0:
            raise ValueError("Force system-ID delay must be non-negative.")
        self.contact_history = torch.zeros(
            self.delay_steps + 1,
            self.num_envs,
            WUJI_JOINT_COUNT,
            dtype=dtype,
            device=self.device,
        )
        self.history_index = 0

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset model history for selected environments."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return
        self.contact_history[:, env_ids] = 0.0
        self.residual_state[env_ids] = self.residual_mean

    def step(
        self,
        contact_torque_actuator_order: torch.Tensor,
        *,
        joint_position_actuator_order: torch.Tensor,
        joint_velocity_actuator_order: torch.Tensor,
        command_actuator_order: torch.Tensor,
    ) -> torch.Tensor:
        """Return one modeled real Wuji torque packet in actuator order [N m]."""
        q = joint_position_actuator_order[:, self.model_to_actuator].reshape(-1, 5, 4)
        dq = joint_velocity_actuator_order[:, self.model_to_actuator].reshape(-1, 5, 4)
        command = command_actuator_order[:, self.model_to_actuator].reshape(-1, 5, 4)
        normalized = (torch.cat((q, dq, command), dim=-1) - self.feature_mean) / self.feature_scale
        design = torch.cat(
            (torch.ones(normalized.shape[:2] + (1,), dtype=self.dtype, device=self.device), normalized), dim=-1
        )
        baseline = torch.einsum("nfi,foi->nfo", design, self.coefficients)

        contact_model_order = contact_torque_actuator_order[:, self.model_to_actuator]
        self.contact_history[self.history_index].copy_(contact_model_order)
        delayed_index = (self.history_index - self.delay_steps) % len(self.contact_history)
        delayed_contact = self.contact_history[delayed_index].reshape(-1, 5, 4)
        self.history_index = (self.history_index + 1) % len(self.contact_history)
        modeled = baseline + torch.einsum("nfi,foi->nfo", delayed_contact, self.coupling)
        modeled = modeled.reshape(-1, WUJI_JOINT_COUNT)

        if self.sample_residual:
            standard_normal = torch.randn(
                self.num_envs,
                WUJI_JOINT_COUNT,
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
            innovation = standard_normal @ self.innovation_factor.T
            centered = self.residual_state - self.residual_mean
            self.residual_state = self.residual_mean + centered * self.lag1 + innovation
            modeled = modeled + self.residual_state
        else:
            modeled = modeled + self.residual_mean
        return modeled[:, self.actuator_to_model]


class VirtualForcePipeline:
    """Stateful virtual sensor for the 20 physical Wuji actuators."""

    def __init__(
        self,
        *,
        num_envs: int,
        config: VirtualForcePipelineConfig,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        system_id_model: WujiForceSystemIdModel | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.system_id_model = system_id_model
        shape = (self.num_envs, WUJI_JOINT_COUNT)
        self.torque_scale = torch.ones(shape, device=self.device, dtype=self.dtype)
        self.torque_bias_nm = torch.zeros_like(self.torque_scale)
        self.filtered_torque_nm = torch.zeros_like(self.torque_scale)
        self.latency_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        history_length = int(config.latency_steps_range[1]) + 1
        self.torque_history = torch.zeros(history_length, *shape, device=self.device, dtype=self.dtype)
        self.history_index = 0
        self.needs_initialization = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.output = VirtualForceOutput(
            torch.zeros(shape, device=self.device, dtype=self.dtype),
            torch.zeros(shape, device=self.device, dtype=self.dtype),
            torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        )
        self.reset(torch.arange(self.num_envs, device=self.device))

    def _uniform(self, bounds: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        low, high = bounds
        if low == high:
            return torch.full(shape, float(low), device=self.device, dtype=self.dtype)
        return low + torch.rand(shape, device=self.device, dtype=self.dtype) * (high - low)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset sensor state and resample episode-constant effects."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return
        shape = (env_ids.numel(), WUJI_JOINT_COUNT)
        self.torque_scale[env_ids] = self._uniform(self.config.torque_scale_range, shape)
        self.torque_bias_nm[env_ids] = self._uniform(self.config.torque_bias_range_nm, shape)
        low, high = self.config.latency_steps_range
        self.latency_steps[env_ids] = torch.randint(int(low), int(high) + 1, (env_ids.numel(),), device=self.device)
        self.filtered_torque_nm[env_ids] = 0.0
        self.torque_history[:, env_ids] = 0.0
        self.needs_initialization[env_ids] = True
        self.output.ideal_actuator_torque_nm[env_ids] = 0.0
        self.output.observed_actuator_torque_nm[env_ids] = 0.0
        self.output.valid[env_ids] = False
        if self.system_id_model is not None:
            self.system_id_model.reset(env_ids)

    def step(
        self,
        *,
        contact_forces_w: torch.Tensor,
        contact_linear_jacobians_w: torch.Tensor,
        contact_moments_w: torch.Tensor,
        contact_angular_jacobians_w: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        joint_command: torch.Tensor,
    ) -> VirtualForceOutput:
        """Advance the sensor model and return the latest packet."""
        ideal_torque = contact_wrenches_to_actuator_torque(
            contact_forces_w * self.config.contact_force_sign,
            contact_linear_jacobians_w,
            contact_moments_w * self.config.contact_force_sign,
            contact_angular_jacobians_w,
        )
        if ideal_torque.shape != (self.num_envs, WUJI_JOINT_COUNT):
            raise ValueError(
                f"Expected ideal Wuji torque shape {(self.num_envs, WUJI_JOINT_COUNT)}, got {ideal_torque.shape}."
            )

        observed_torque = ideal_torque
        valid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        for _ in range(self.config.updates_per_step):
            measured_torque = ideal_torque * self.torque_scale + self.torque_bias_nm
            if self.config.torque_noise_std_nm > 0.0:
                measured_torque = measured_torque + torch.randn_like(measured_torque) * self.config.torque_noise_std_nm
            if self.config.torque_clip_abs_nm is not None:
                measured_torque = measured_torque.clamp(-self.config.torque_clip_abs_nm, self.config.torque_clip_abs_nm)

            alpha = self.config.torque_lpf_alpha
            filtered = alpha * measured_torque + (1.0 - alpha) * self.filtered_torque_nm
            initialized = self.needs_initialization
            filtered = torch.where(initialized.unsqueeze(-1), measured_torque, filtered)
            self.filtered_torque_nm.copy_(filtered)
            self.torque_history[:, initialized] = filtered[initialized].unsqueeze(0)
            self.needs_initialization[initialized] = False

            newest_index = self.history_index
            self.torque_history[newest_index].copy_(filtered)
            selected_indices = torch.remainder(newest_index - self.latency_steps, self.torque_history.shape[0])
            env_indices = torch.arange(self.num_envs, device=self.device)
            observed_torque = self.torque_history[selected_indices, env_indices]
            self.history_index = (newest_index + 1) % self.torque_history.shape[0]

            valid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
            if self.config.packet_dropout_probability > 0.0:
                valid = torch.rand(self.num_envs, device=self.device) >= self.config.packet_dropout_probability
                observed_torque = torch.where(valid.unsqueeze(-1), observed_torque, torch.zeros_like(observed_torque))

            if self.system_id_model is not None:
                observed_torque = self.system_id_model.step(
                    observed_torque,
                    joint_position_actuator_order=joint_position,
                    joint_velocity_actuator_order=joint_velocity,
                    command_actuator_order=joint_command,
                )

        self.output = VirtualForceOutput(ideal_torque, observed_torque, valid)
        return self.output
