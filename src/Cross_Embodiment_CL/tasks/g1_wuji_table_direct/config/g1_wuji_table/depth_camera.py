# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinhole geometry shared by the simulated student depth camera and real D435 preprocessing.

The simulator renders the student's square depth image directly.  A real D435 frame is cropped to
the largest square centred on its principal point and resized to the same size.  Both derive their
geometry from one set of intrinsics, so each pixel is the same ray in either image.

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
class SquareCrop:
    """A square crop of a source image and the intrinsics of that crop after resizing."""

    left: int
    top: int
    size: int
    """Side of the square in source pixels."""
    output: PinholeIntrinsics
    """Intrinsics of the cropped square once resized to ``output.width`` pixels."""


def square_crop(intrinsics: PinholeIntrinsics, output_size: int) -> SquareCrop:
    """Return the largest square centred on the principal point, resized to ``output_size`` pixels.

    Cropping keeps every kept pixel's ray and shifts the principal point by the crop origin; a uniform
    resize scales all four intrinsics.  A stretch would make ``fx != fy``, which the Newton renderer
    cannot represent, so the non-square sides are discarded instead.

    Raises:
        ValueError: If the source pixels are not square.
    """
    if not math.isclose(intrinsics.fx, intrinsics.fy, rel_tol=1.0e-3):
        raise ValueError(f"Square pixels are required, but fx={intrinsics.fx} and fy={intrinsics.fy}.")
    half = math.floor(
        min(intrinsics.cx, intrinsics.width - intrinsics.cx, intrinsics.cy, intrinsics.height - intrinsics.cy)
    )
    if half < 1:
        raise ValueError(f"The principal point ({intrinsics.cx}, {intrinsics.cy}) leaves no square to crop.")
    size = 2 * half
    left = round(intrinsics.cx - half)
    top = round(intrinsics.cy - half)
    scale = output_size / size
    output = PinholeIntrinsics(
        width=output_size,
        height=output_size,
        fx=intrinsics.fx * scale,
        fy=intrinsics.fy * scale,
        cx=(intrinsics.cx - left) * scale,
        cy=(intrinsics.cy - top) * scale,
    )
    return SquareCrop(left=left, top=top, size=size, output=output)


def crop_and_resize_depth(depth_m: torch.Tensor, crop: SquareCrop) -> torch.Tensor:
    """Crop a real metric depth image and resize it to the student resolution.

    Nearest-exact sampling never blends foreground and background depths across an edge, which would
    invent surfaces that exist in neither the real scene nor the simulator.

    Args:
        depth_m: Metric depth with shape ``(N, H, W)`` or ``(N, 1, H, W)``.
        crop: The crop computed for this stream's intrinsics.

    Returns:
        Depth with shape ``(N, 1, S, S)``, where ``S`` is the crop's output size.
    """
    if depth_m.dim() == 3:
        depth_m = depth_m.unsqueeze(1)
    window = depth_m[..., crop.top : crop.top + crop.size, crop.left : crop.left + crop.size]
    return F.interpolate(window, size=(crop.output.height, crop.output.width), mode="nearest-exact")


def normalize_depth(depth_m: torch.Tensor, max_depth_m: float) -> torch.Tensor:
    """Clip metric depth to ``[0, max_depth_m]`` and scale it to ``[0, 1]``.

    A missing return reads 0 on both the D435 and the Newton renderer, and stays 0; non-finite values
    are treated the same way.
    """
    depth = torch.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    return depth.clamp(0.0, max_depth_m) / max_depth_m
