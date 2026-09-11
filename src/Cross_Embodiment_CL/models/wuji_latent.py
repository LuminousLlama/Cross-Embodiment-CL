# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen 18-D Wuji policy-action to joint-target conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets/models/wuji_latent"
_AUTOENCODER_PATH = _ASSET_DIR / "mano_pose_autoencoder_63d.pt"
_RETARGETER_PATH = _ASSET_DIR / "retargeting_nn_wuji_hand_grab_dexpilot_63d.pth"

MANO_POSE_DIM = 63
WUJI_JOINT_DIM = 20
WUJI_LATENT_DIM = 18


class _PoseAutoencoder(nn.Module):
    """The architecture stored in the packaged MANO autoencoder checkpoint."""

    def __init__(self, latent_dim: int, hidden_dims: tuple[int, int, int], bounded_latent: bool) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims
        encoder_layers: list[nn.Module] = [
            nn.Linear(MANO_POSE_DIM, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, latent_dim),
        ]
        if bounded_latent:
            encoder_layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3),
            nn.ReLU(),
            nn.Linear(h3, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, MANO_POSE_DIM),
        )


class _WujiRetargeter(nn.Module):
    """The architecture stored in the packaged MANO-to-Wuji checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = MANO_POSE_DIM
        for _ in range(3):
            layers.extend((nn.Linear(in_dim, 512), nn.ReLU()))
            in_dim = 512
        layers.append(nn.Linear(in_dim, WUJI_JOINT_DIM))
        self.model = nn.Sequential(*layers)


@dataclass(frozen=True)
class WujiLatentProjection:
    """Projection of a physical Wuji pose onto the 18-D latent manifold."""

    mano_pose: torch.Tensor
    latent_action: torch.Tensor
    joint_target: torch.Tensor
    error: torch.Tensor


class WujiLatentActionPipeline:
    """Convert normalized 18-D policy actions to physical Wuji joint targets.

    The packaged models define a 63-D MANO pose, an 18-D bounded latent
    action, and the 20-D Wuji joint order ``finger1_joint1`` through
    ``finger5_joint4``. Joint targets are physical positions [rad].
    """

    latent_dim = WUJI_LATENT_DIM
    joint_dim = WUJI_JOINT_DIM

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        autoencoder_checkpoint = self._load_checkpoint(_AUTOENCODER_PATH)
        autoencoder_config = autoencoder_checkpoint["model_config"]
        if (
            int(autoencoder_config["input_dim"]) != MANO_POSE_DIM
            or int(autoencoder_config["latent_dim"]) != WUJI_LATENT_DIM
            or not bool(autoencoder_config["bounded_latent"])
        ):
            raise ValueError("The packaged MANO autoencoder does not define the expected bounded 63-D to 18-D mapping.")

        hidden_dims = tuple(int(value) for value in autoencoder_config["hidden_dims"])
        self.autoencoder = _PoseAutoencoder(WUJI_LATENT_DIM, hidden_dims, True).to(self.device)
        self.autoencoder.load_state_dict(autoencoder_checkpoint["model_state_dict"], strict=True)
        self.autoencoder.eval()
        self._freeze(self.autoencoder)

        normalizer = autoencoder_checkpoint["normalizer"]
        self.mano_mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32, device=self.device)
        self.mano_std = torch.as_tensor(normalizer["std"], dtype=torch.float32, device=self.device)
        if self.mano_mean.shape != (MANO_POSE_DIM,) or self.mano_std.shape != (MANO_POSE_DIM,):
            raise ValueError(
                "The packaged MANO normalizer must contain 63 values for both mean and standard deviation."
            )
        if torch.any(self.mano_std <= 0.0):
            raise ValueError("The packaged MANO normalizer contains a non-positive standard deviation.")

        self.retargeter = _WujiRetargeter().to(self.device)
        self.retargeter.load_state_dict(self._load_checkpoint(_RETARGETER_PATH), strict=True)
        self.retargeter.eval()
        self._freeze(self.retargeter)

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[str, object]:
        if not path.is_file():
            raise FileNotFoundError(f"Required Wuji latent-model artifact is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Expected a dictionary checkpoint at {path}.")
        return checkpoint

    @staticmethod
    def _freeze(model: nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def _check_last_dimension(self, tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.ndim != 2 or tensor.shape[1] != expected:
            raise ValueError(f"{name} must have shape [batch, {expected}], got {tuple(tensor.shape)}.")

    def encode_mano_pose(self, mano_pose: torch.Tensor) -> torch.Tensor:
        """Encode physical MANO joint rotations [rad] into bounded policy actions."""
        self._check_last_dimension(mano_pose, MANO_POSE_DIM, "mano_pose")
        normalized_pose = (mano_pose.to(self.device) - self.mano_mean) / self.mano_std
        return self.autoencoder.encoder(normalized_pose)

    def decode_latent_action(self, latent_action: torch.Tensor) -> torch.Tensor:
        """Decode bounded policy actions into physical MANO joint rotations [rad]."""
        self._check_last_dimension(latent_action, WUJI_LATENT_DIM, "latent_action")
        decoded_pose = self.autoencoder.decoder(torch.clamp(latent_action.to(self.device), -1.0, 1.0))
        return decoded_pose * self.mano_std + self.mano_mean

    def retarget_mano_pose(self, mano_pose: torch.Tensor) -> torch.Tensor:
        """Retarget physical MANO joint rotations [rad] to physical Wuji positions [rad]."""
        self._check_last_dimension(mano_pose, MANO_POSE_DIM, "mano_pose")
        return self.retargeter.model(mano_pose.to(self.device))

    def latent_action_to_joint_target(
        self, latent_action: torch.Tensor, lower_limits: torch.Tensor, upper_limits: torch.Tensor
    ) -> torch.Tensor:
        """Decode and retarget a policy action, then clamp it to live Wuji limits [rad]."""
        self._check_last_dimension(lower_limits, WUJI_JOINT_DIM, "lower_limits")
        self._check_last_dimension(upper_limits, WUJI_JOINT_DIM, "upper_limits")
        if torch.any(upper_limits <= lower_limits):
            raise ValueError("Each Wuji upper joint limit must exceed its lower limit.")
        return torch.clamp(
            self.retarget_mano_pose(self.decode_latent_action(latent_action)), lower_limits, upper_limits
        )

    def project_joint_positions(
        self,
        joint_position: torch.Tensor,
        lower_limits: torch.Tensor,
        upper_limits: torch.Tensor,
        *,
        steps: int = 512,
        learning_rate: float = 0.05,
    ) -> WujiLatentProjection:
        """Fit the frozen latent manifold to physical Wuji positions [rad].

        This supplies the otherwise unavailable Wuji-to-MANO leg for
        round-trip validation. The result is a projection, not an inverse:
        arbitrary 20-D joint poses need not lie on the 18-D action manifold.
        """
        self._check_last_dimension(joint_position, WUJI_JOINT_DIM, "joint_position")
        self._check_last_dimension(lower_limits, WUJI_JOINT_DIM, "lower_limits")
        self._check_last_dimension(upper_limits, WUJI_JOINT_DIM, "upper_limits")
        if steps < 1 or learning_rate <= 0.0:
            raise ValueError("steps and learning_rate must be positive.")

        target = torch.clamp(joint_position.to(self.device), lower_limits, upper_limits)
        span = upper_limits - lower_limits
        unconstrained_latent = torch.zeros((target.shape[0], WUJI_LATENT_DIM), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam((unconstrained_latent,), lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            latent_action = torch.tanh(unconstrained_latent)
            joint_target = self.latent_action_to_joint_target(latent_action, lower_limits, upper_limits)
            loss = torch.mean(torch.square((joint_target - target) / span))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # This explicit decode-then-encode sequence is the deployed policy
            # contract exercised by the round-trip test.
            mano_pose = self.decode_latent_action(torch.tanh(unconstrained_latent))
            latent_action = self.encode_mano_pose(mano_pose)
            joint_target = self.latent_action_to_joint_target(latent_action, lower_limits, upper_limits)
            error = torch.linalg.vector_norm(joint_target - target, dim=-1)
        return WujiLatentProjection(mano_pose, latent_action, joint_target, error)
