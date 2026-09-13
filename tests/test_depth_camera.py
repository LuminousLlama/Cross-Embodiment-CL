# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the student depth-camera geometry shared by simulation and the real D435."""

import math

import pytest
import torch

from Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.depth_camera import (
    D435_DEPTH_848X480,
    PinholeIntrinsics,
    crop_and_resize_depth,
    normalize_depth,
    square_crop,
)


def _render_tilted_plane(intrinsics: PinholeIntrinsics) -> torch.Tensor:
    """Depth of the plane ``z + 0.3 * y = 1`` sampled at pixel centres, shape ``(1, H, W)``."""
    v = torch.arange(intrinsics.height, dtype=torch.float64) + 0.5
    ray_y = (v - intrinsics.cy) / intrinsics.fy
    depth = 1.0 / (1.0 + 0.3 * ray_y)
    return depth[None, :, None].expand(1, intrinsics.height, intrinsics.width)


def test_datasheet_d435_crop_matches_the_signed_off_geometry():
    """848x480 at 58 deg crops columns 184-663 and becomes a centred 224x224 camera with f = 202.1."""
    assert D435_DEPTH_848X480.fy == pytest.approx(240.0 / math.tan(math.radians(29.0)))
    crop = square_crop(D435_DEPTH_848X480, 224)
    assert (crop.left, crop.top, crop.size) == (184, 0, 480)
    assert crop.output.fx == pytest.approx(202.05, abs=0.01)
    assert (crop.output.cx, crop.output.cy) == (112.0, 112.0)
    assert 2.0 * math.degrees(math.atan(112.0 / crop.output.fy)) == pytest.approx(58.0)


@pytest.mark.parametrize(
    "intrinsics",
    [D435_DEPTH_848X480, PinholeIntrinsics(848, 480, fx=431.2, fy=431.2, cx=424.37, cy=239.62)],
)
def test_cropped_real_depth_matches_a_direct_render(intrinsics):
    """Cropping and resizing a full frame reproduces what a camera with the output intrinsics renders."""
    crop = square_crop(intrinsics, 224)
    resized = crop_and_resize_depth(_render_tilted_plane(intrinsics), crop)
    direct = _render_tilted_plane(crop.output)
    # Nearest sampling picks a source pixel up to half a source pixel from the output ray.
    assert resized.shape == (1, 1, 224, 224)
    assert torch.allclose(resized[:, 0], direct, atol=1.0e-3)


def test_square_crop_rejects_non_square_pixels():
    with pytest.raises(ValueError, match="Square pixels"):
        square_crop(PinholeIntrinsics(848, 480, fx=430.0, fy=440.0, cx=424.0, cy=240.0), 224)


def test_normalize_depth_keeps_missing_returns_at_zero():
    depth = torch.tensor([0.0, 0.6, 2.0, float("nan"), float("inf")])
    assert torch.equal(normalize_depth(depth, 1.2), torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0]))
