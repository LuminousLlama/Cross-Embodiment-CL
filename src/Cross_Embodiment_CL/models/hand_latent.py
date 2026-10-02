# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen shared 18-D policy-action to hand joint-target conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .hand_registry import get_hand_spec

_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets/models/hand_latent"
_AUTOENCODER_PATH = _ASSET_DIR / "mano_pose_autoencoder_63d.pt"

MANO_POSE_DIM = 63
HAND_LATENT_DIM = 18


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


class _HandRetargeter(nn.Module):
    """The architecture described by each packaged retargeter manifest."""

    def __init__(self, joint_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = MANO_POSE_DIM
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU()))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, joint_dim))
        self.model = nn.Sequential(*layers)


@dataclass(frozen=True)
class HandLatentProjection:
    """Projection of a physical hand pose onto the 18-D latent manifold."""

    mano_pose: torch.Tensor
    latent_action: torch.Tensor
    joint_target: torch.Tensor
    error: torch.Tensor


class HandLatentActionPipeline:
    """Convert shared 18-D policy actions to a selected hand's joint targets.

    The packaged models define a 63-D MANO pose and an 18-D bounded latent
    action. Hand definitions map manifest output names to the selected hand's
    independent joint order. Joint targets are physical positions [rad].
    """

    latent_dim = HAND_LATENT_DIM

    def __init__(self, hand_type: str, device: torch.device | str) -> None:
        self.hand_spec = get_hand_spec(hand_type)
        self.joint_names = self.hand_spec.joint_names
        self.joint_dim = len(self.joint_names)
        self.device = torch.device(device)
        autoencoder_checkpoint = self._load_checkpoint(_AUTOENCODER_PATH)
        autoencoder_config = autoencoder_checkpoint["model_config"]
        if (
            int(autoencoder_config["input_dim"]) != MANO_POSE_DIM
            or int(autoencoder_config["latent_dim"]) != HAND_LATENT_DIM
            or not bool(autoencoder_config["bounded_latent"])
        ):
            raise ValueError("The packaged MANO autoencoder does not define the expected bounded 63-D to 18-D mapping.")

        hidden_dims = tuple(int(value) for value in autoencoder_config["hidden_dims"])
        self.autoencoder = _PoseAutoencoder(HAND_LATENT_DIM, hidden_dims, True).to(self.device)
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

        artifact_dir = _ASSET_DIR.parent / "retargeting" / self.hand_spec.retargeter_directory
        manifests = tuple(artifact_dir.glob("*.yaml"))
        if len(manifests) != 1:
            raise ValueError(f"Expected exactly one retargeter manifest in {artifact_dir}.")
        with manifests[0].open() as stream:
            metadata = yaml.safe_load(stream)
        if metadata["model_type"] != "RetargetingNN" or int(metadata["mano_dim"]) != MANO_POSE_DIM:
            raise ValueError(f"Unsupported retargeter contract in {manifests[0]}.")
        if metadata.get("output_units") != "rad":
            raise ValueError(f"Retargeter must declare physical radian outputs in {manifests[0]}.")
        if int(metadata["robot_dim"]) != self.joint_dim:
            raise ValueError(f"Retargeter output dimension disagrees with {hand_type} independent joints.")
        model_names = tuple(metadata["robot_joint_names"])
        if "target_joint_names" in metadata and tuple(metadata["target_joint_names"]) != model_names:
            raise ValueError("Retargeter robot and target joint orders disagree.")
        self._model_order = self.hand_spec.model_order(model_names)
        self.retargeter = _HandRetargeter(
            self.joint_dim, int(metadata["hidden_dim"]), int(metadata["hidden_layers"])
        ).to(self.device)
        self.retargeter.load_state_dict(self._load_checkpoint(artifact_dir / metadata["state_dict_path"]), strict=True)
        self.retargeter.eval()
        self._freeze(self.retargeter)

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[str, object]:
        if not path.is_file():
            raise FileNotFoundError(f"Required hand latent-model artifact is missing: {path}")
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
        self._check_last_dimension(latent_action, HAND_LATENT_DIM, "latent_action")
        decoded_pose = self.autoencoder.decoder(torch.clamp(latent_action.to(self.device), -1.0, 1.0))
        return decoded_pose * self.mano_std + self.mano_mean

    def retarget_mano_pose(self, mano_pose: torch.Tensor) -> torch.Tensor:
        """Retarget physical MANO joint rotations [rad] to independent hand positions [rad]."""
        self._check_last_dimension(mano_pose, MANO_POSE_DIM, "mano_pose")
        return self.retargeter.model(mano_pose.to(self.device))[:, self._model_order]

    def latent_action_to_joint_target(
        self, latent_action: torch.Tensor, lower_limits: torch.Tensor, upper_limits: torch.Tensor
    ) -> torch.Tensor:
        """Decode and retarget a policy action, then clamp it to live hand limits [rad]."""
        self._check_last_dimension(lower_limits, self.joint_dim, "lower_limits")
        self._check_last_dimension(upper_limits, self.joint_dim, "upper_limits")
        if torch.any(upper_limits <= lower_limits):
            raise ValueError("Each hand upper joint limit must exceed its lower limit.")
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
    ) -> HandLatentProjection:
        """Fit the frozen latent manifold to independent hand positions [rad].

        This supplies the otherwise unavailable hand-to-MANO leg for
        round-trip validation. The result is a projection, not an inverse:
        arbitrary joint poses need not lie on the 18-D action manifold.
        """
        self._check_last_dimension(joint_position, self.joint_dim, "joint_position")
        self._check_last_dimension(lower_limits, self.joint_dim, "lower_limits")
        self._check_last_dimension(upper_limits, self.joint_dim, "upper_limits")
        if steps < 1 or learning_rate <= 0.0:
            raise ValueError("steps and learning_rate must be positive.")

        target = torch.clamp(joint_position.to(self.device), lower_limits, upper_limits)
        span = upper_limits - lower_limits
        unconstrained_latent = torch.zeros((target.shape[0], HAND_LATENT_DIM), device=self.device, requires_grad=True)
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
        return HandLatentProjection(mano_pose, latent_action, joint_target, error)
