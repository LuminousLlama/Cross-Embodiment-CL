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


def normalize_depth(depth_m: torch.Tensor, max_depth_m: float) -> torch.Tensor:
    """Clip metric depth to ``[0, max_depth_m]`` and scale it to ``[0, 1]``.

    A missing return reads 0 on both the D435 and the Newton renderer, and stays 0; non-finite values
    are treated the same way.
    """
    depth = torch.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    return depth.clamp(0.0, max_depth_m) / max_depth_m
