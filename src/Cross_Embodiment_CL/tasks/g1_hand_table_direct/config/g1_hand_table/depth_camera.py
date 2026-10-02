# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinhole geometry shared by the simulated student depth camera and real D435 preprocessing.

The simulator renders a reduced-resolution image with the D435's full aspect ratio.  Both that image
and a real D435 frame are letterboxed into the student's square input, preserving the wide horizontal
field of view instead of discarding the source image's sides.

Pixel coordinates are continuous and edge-based: an image spans ``[0, W] x [0, H]`` and a centred
principal point is ``(W / 2, H / 2)``, matching Isaac Lab's camera intrinsics.  librealsense and
OpenCV place pixel centres on integers, so add 0.5 to their ``ppx`` and ``ppy``.

This module deliberately imports nothing from Isaac Lab, so deployment code can use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Pinhole intrinsics of a ``width`` x ``height`` image, in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> list[float]:
        """Return the row-major 3x3 intrinsic matrix ``[fx, 0, cx, 0, fy, cy, 0, 0, 1]``."""
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]


def centered_intrinsics(width: int, height: int, vertical_fov_deg: float) -> PinholeIntrinsics:
    """Square-pixel intrinsics with a centred principal point and the given vertical field of view."""
    focal = 0.5 * height / math.tan(math.radians(0.5 * vertical_fov_deg))
    return PinholeIntrinsics(width, height, focal, focal, 0.5 * width, 0.5 * height)


D435_DEPTH_848X480 = centered_intrinsics(848, 480, vertical_fov_deg=58.0)
"""Generic D435 depth intrinsics from the datasheet: the 58 deg vertical field of view, centred.

848x480 is Intel's recommended D435 depth mode.  This is a placeholder until the deployment camera is
calibrated: replace it with the measured intrinsics of that exact stream, unaligned to color.
"""


@dataclass(frozen=True)
class DepthLetterbox:
    """Geometry for fitting one pinhole image inside a padded output canvas."""

    source: PinholeIntrinsics
    """Intrinsics of the full-resolution input image."""
    content: PinholeIntrinsics
    """Intrinsics of the resized, unpadded image that the simulator renders directly."""
    output_width: int
    output_height: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int

    @property
    def output(self) -> PinholeIntrinsics:
        """Return the content intrinsics expressed in the padded output coordinates."""
        return PinholeIntrinsics(
            width=self.output_width,
            height=self.output_height,
            fx=self.content.fx,
            fy=self.content.fy,
            cx=self.content.cx + self.pad_left,
            cy=self.content.cy + self.pad_top,
        )


def fit_depth_letterbox(intrinsics: PinholeIntrinsics, output_size: int) -> DepthLetterbox:
    """Fit a full pinhole image within an ``output_size`` square without stretching or cropping.

    The longer source dimension fills the output.  The shorter resized dimension is rounded to the
    nearest pixel, so an odd number of padding pixels is placed with the extra pixel on the bottom or
    right.  The focal length uses the uniform fit scale because Newton renders a centred square-pixel
    pinhole; the at-most-half-pixel aspect-ratio rounding is handled by the fixed padding boundary.

    Raises:
        ValueError: If ``output_size`` is invalid or the source pixels are not square.
    """
    if output_size < 1:
        raise ValueError(f"Output size must be positive, received {output_size}.")
    if not math.isclose(intrinsics.fx, intrinsics.fy, rel_tol=1.0e-3):
        raise ValueError(f"Square pixels are required, but fx={intrinsics.fx} and fy={intrinsics.fy}.")

    scale = min(output_size / intrinsics.width, output_size / intrinsics.height)
    content_width = max(1, round(intrinsics.width * scale))
    content_height = max(1, round(intrinsics.height * scale))
    pad_left = (output_size - content_width) // 2
    pad_top = (output_size - content_height) // 2
    content = PinholeIntrinsics(
        width=content_width,
        height=content_height,
        fx=intrinsics.fx * scale,
        fy=intrinsics.fy * scale,
        cx=0.5 * content_width + (intrinsics.cx - 0.5 * intrinsics.width) * scale,
        cy=0.5 * content_height + (intrinsics.cy - 0.5 * intrinsics.height) * scale,
    )
    return DepthLetterbox(
        source=intrinsics,
        content=content,
        output_width=output_size,
        output_height=output_size,
        pad_left=pad_left,
        pad_right=output_size - content_width - pad_left,
        pad_top=pad_top,
        pad_bottom=output_size - content_height - pad_top,
    )


def resize_and_pad_depth(depth_m: torch.Tensor, letterbox: DepthLetterbox) -> torch.Tensor:
    """Resize full-frame metric depth if needed and pad it to the student input resolution.

    Nearest sampling never blends foreground and background depths across an edge, which would invent
    surfaces that exist in neither the real scene nor the simulator.  The real frame is sampled at the
    exact rays of the simulated content camera instead of using independent rounded x/y resize scales.
    Zero padding uses the same representation as a missing D435/Newton depth return.

    Args:
        depth_m: Metric depth with shape ``(N, H, W)`` or ``(N, 1, H, W)``. Its spatial shape must
            match either the source D435 frame or the simulator's resized content frame.
        letterbox: Shared real/sim resize and padding geometry.

    Returns:
        Depth with shape ``(N, 1, output_height, output_width)``.

    Raises:
        ValueError: If the tensor rank, channel count, or spatial shape does not match the contract.
    """
    if depth_m.dim() == 3:
        depth_m = depth_m.unsqueeze(1)
    if depth_m.dim() != 4 or depth_m.shape[1] != 1:
        raise ValueError(f"Expected depth shaped (N,H,W) or (N,1,H,W), received {tuple(depth_m.shape)}.")

    spatial_shape = tuple(depth_m.shape[-2:])
    source_shape = (letterbox.source.height, letterbox.source.width)
    content_shape = (letterbox.content.height, letterbox.content.width)
    if spatial_shape == source_shape:
        dtype = depth_m.dtype
        columns = torch.arange(letterbox.content.width, device=depth_m.device, dtype=dtype) + 0.5
        rows = torch.arange(letterbox.content.height, device=depth_m.device, dtype=dtype) + 0.5
        source_columns = (
            columns - letterbox.content.cx
        ) / letterbox.content.fx * letterbox.source.fx + letterbox.source.cx
        source_rows = (rows - letterbox.content.cy) / letterbox.content.fy * letterbox.source.fy + letterbox.source.cy
        grid_x = 2.0 * source_columns / letterbox.source.width - 1.0
        grid_y = 2.0 * source_rows / letterbox.source.height - 1.0
        grid_rows, grid_columns = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack((grid_columns, grid_rows), dim=-1)
        depth_m = F.grid_sample(
            depth_m,
            grid.unsqueeze(0).expand(depth_m.shape[0], -1, -1, -1),
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
    elif spatial_shape != content_shape:
        raise ValueError(f"Expected depth spatial shape {source_shape} or {content_shape}, received {spatial_shape}.")

    return F.pad(
        depth_m,
        (letterbox.pad_left, letterbox.pad_right, letterbox.pad_top, letterbox.pad_bottom),
        mode="constant",
        value=0.0,
    )


def warp_depth_intrinsics(
    depth_m: torch.Tensor,
    focal_scale: torch.Tensor,
    principal_point_offset_px: torch.Tensor,
) -> torch.Tensor:
    """Reproject batched pinhole depth for per-episode intrinsic perturbations.

    The input and output have the same resolution. ``focal_scale`` multiplies both focal lengths, while
    ``principal_point_offset_px`` shifts ``(cx, cy)`` from the image centre. Nearest sampling preserves
    foreground/background discontinuities and rays outside the nominal image are zero-filled.

    Args:
        depth_m: Metric depth shaped ``(N, 1, H, W)``.
        focal_scale: Per-image focal-length scale shaped ``(N,)``.
        principal_point_offset_px: Per-image ``(cx, cy)`` offsets [px], shaped ``(N, 2)``.

    Returns:
        Reprojected metric depth with the same shape as ``depth_m``.
    """
    if depth_m.dim() != 4 or depth_m.shape[1] != 1:
        raise ValueError(f"Expected depth shaped (N,1,H,W), received {tuple(depth_m.shape)}.")
    batch_size, _, height, width = depth_m.shape
    if focal_scale.shape != (batch_size,):
        raise ValueError(f"Expected focal_scale shaped ({batch_size},), received {tuple(focal_scale.shape)}.")
    if principal_point_offset_px.shape != (batch_size, 2):
        raise ValueError(
            "Expected principal_point_offset_px shaped "
            f"({batch_size},2), received {tuple(principal_point_offset_px.shape)}."
        )
    if torch.any(focal_scale <= 0.0):
        raise ValueError("focal_scale must be positive.")

    dtype = depth_m.dtype
    columns = torch.arange(width, device=depth_m.device, dtype=dtype) + 0.5
    rows = torch.arange(height, device=depth_m.device, dtype=dtype) + 0.5
    center_x = 0.5 * width
    center_y = 0.5 * height
    scale = focal_scale.to(device=depth_m.device, dtype=dtype)
    offset = principal_point_offset_px.to(device=depth_m.device, dtype=dtype)
    source_columns = (columns.view(1, 1, width) - (center_x + offset[:, 0].view(batch_size, 1, 1))) / scale.view(
        batch_size, 1, 1
    ) + center_x
    source_rows = (rows.view(1, height, 1) - (center_y + offset[:, 1].view(batch_size, 1, 1))) / scale.view(
        batch_size, 1, 1
    ) + center_y
    grid_x = (2.0 * source_columns / width - 1.0).expand(-1, height, -1)
    grid_y = (2.0 * source_rows / height - 1.0).expand(-1, -1, width)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(depth_m, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def _shift_depth(depth_m: torch.Tensor, direction: int) -> torch.Tensor:
    """Shift depth by one pixel without wrapping; directions are left, right, up, down."""
    if direction == 0:
        return F.pad(depth_m[..., :-1], (1, 0, 0, 0))
    if direction == 1:
        return F.pad(depth_m[..., 1:], (0, 1, 0, 0))
    if direction == 2:
        return F.pad(depth_m[..., :-1, :], (0, 0, 1, 0))
    if direction == 3:
        return F.pad(depth_m[..., 1:, :], (0, 0, 0, 1))
    raise ValueError(f"Expected direction in [0, 3], received {direction}.")


def randomize_depth_measurement(
    depth_m: torch.Tensor,
    depth_scale: torch.Tensor,
    depth_bias_m: torch.Tensor,
    noise_std_at_1m_m: float,
    boundary_corruption_prob: float,
    edge_dropout_prob: float,
    boundary_threshold_m: float,
    strength: float,
) -> torch.Tensor:
    """Apply student-only D435-style measurement corruption to metric depth.

    Calibration scale/bias are fixed per episode by the caller. Gaussian noise is sampled per pixel
    with standard deviation proportional to ``depth^2``. Selected depth-discontinuity pixels copy a
    valid four-connected neighbor to mimic one-pixel foreground/background silhouette errors. A thin,
    mostly-invalid outline is sampled on the foreground side of those same boundaries.

    Args:
        depth_m: Metric depth shaped ``(N, 1, H, W)``; zero denotes invalid input depth.
        depth_scale: Per-image multiplicative calibration shaped ``(N,)``.
        depth_bias_m: Per-image calibration bias [m], shaped ``(N,)``.
        noise_std_at_1m_m: Full-strength Gaussian standard deviation at 1 m [m].
        boundary_corruption_prob: Full-strength corruption probability for depth-edge pixels.
        edge_dropout_prob: Full-strength invalid probability for each foreground depth-edge pixel.
        boundary_threshold_m: Minimum neighboring depth jump [m] considered a boundary.
        strength: ADR strength in ``[0, 1]``.

    Returns:
        Randomized metric depth with the same shape as ``depth_m``.
    """
    if depth_m.dim() != 4 or depth_m.shape[1] != 1:
        raise ValueError(f"Expected depth shaped (N,1,H,W), received {tuple(depth_m.shape)}.")
    batch_size = depth_m.shape[0]
    if depth_scale.shape != (batch_size,) or depth_bias_m.shape != (batch_size,):
        raise ValueError(
            f"Expected depth_scale and depth_bias_m shaped ({batch_size},), received "
            f"{tuple(depth_scale.shape)} and {tuple(depth_bias_m.shape)}."
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], received {strength}.")
    for name, probability in (
        ("boundary_corruption_prob", boundary_corruption_prob),
        ("edge_dropout_prob", edge_dropout_prob),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], received {probability}.")
    if noise_std_at_1m_m < 0.0 or boundary_threshold_m < 0.0:
        raise ValueError("Depth-noise standard deviation and boundary threshold must be non-negative.")

    clean = torch.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if strength == 0.0:
        return clean

    valid = clean > 0.0
    scale = depth_scale.to(device=depth_m.device, dtype=depth_m.dtype).view(batch_size, 1, 1, 1)
    bias = depth_bias_m.to(device=depth_m.device, dtype=depth_m.dtype).view(batch_size, 1, 1, 1)
    randomized = torch.where(valid, clean * scale + bias, torch.zeros_like(clean)).clamp_min_(0.0)

    if noise_std_at_1m_m > 0.0:
        noise_std = noise_std_at_1m_m * strength * randomized.square()
        randomized.add_(torch.randn_like(randomized) * noise_std)
        randomized.clamp_min_(0.0)

    boundary_prob = boundary_corruption_prob * strength
    if boundary_prob > 0.0:
        boundary_source = randomized.clone()
        neighbors = torch.stack([_shift_depth(boundary_source, direction) for direction in range(4)])
        eligible = (neighbors > 0.0) & (torch.abs(neighbors - boundary_source.unsqueeze(0)) > boundary_threshold_m)
        selected = eligible.any(dim=0) & (torch.rand_like(randomized) < boundary_prob)
        # Choose only across a real depth jump, never a same-surface or zero-depth neighbor.
        scores = torch.rand_like(neighbors).masked_fill_(~eligible, -1.0)
        chosen = neighbors.gather(0, scores.argmax(dim=0, keepdim=True)).squeeze(0)
        randomized = torch.where(selected, chosen, randomized)

    dropout_prob = edge_dropout_prob * strength
    if dropout_prob > 0.0:
        clean_neighbors = torch.stack([_shift_depth(clean, direction) for direction in range(4)])
        neighbor_in_bounds = torch.stack(
            [_shift_depth(torch.ones_like(clean), direction) > 0.0 for direction in range(4)]
        )
        neighbor_valid = clean_neighbors > 0.0
        depth_edge = (
            valid.unsqueeze(0)
            & neighbor_in_bounds
            & (~neighbor_valid | (clean_neighbors - clean.unsqueeze(0) > boundary_threshold_m))
        ).any(dim=0)
        dropout = depth_edge & (torch.rand_like(randomized) < dropout_prob)
        randomized.masked_fill_(dropout & valid, 0.0)
    return randomized


def normalize_depth(depth_m: torch.Tensor, min_depth_m: float, max_depth_m: float) -> torch.Tensor:
    """Scale usable metric depth to ``[0, 1]`` and map out-of-range values to zero.

    Zero, non-finite, too-close, and too-far input depths remain or become zero. Valid values retain
    the student's existing ``depth / max_depth_m`` encoding.
    """
    if not math.isfinite(min_depth_m) or not math.isfinite(max_depth_m) or not 0.0 <= min_depth_m < max_depth_m:
        raise ValueError(
            f"Expected finite depth bounds satisfying 0 <= min < max, received {min_depth_m} and {max_depth_m}."
        )
    depth = torch.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (depth >= min_depth_m) & (depth < max_depth_m)
    return torch.where(valid, depth / max_depth_m, torch.zeros_like(depth))


def normalized_depth_to_grayscale(depth: torch.Tensor) -> torch.Tensor:
    """Render normalized policy depth as near-white, far-dark uint8 grayscale.

    Zero is reserved for invalid depth and stays black. This display transform does not modify the
    floating-point observation passed to the policy.
    """
    valid = depth > 0.0
    contrast = torch.where(valid, 1.0 - depth.clamp(max=1.0), torch.zeros_like(depth))
    return contrast.mul(255.0).round().to(torch.uint8)
