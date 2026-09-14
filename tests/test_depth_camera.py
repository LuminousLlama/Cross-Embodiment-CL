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
    fit_depth_letterbox,
    normalize_depth,
    randomize_depth_measurement,
    resize_and_pad_depth,
    warp_depth_intrinsics,
)


def _render_tilted_plane(intrinsics: PinholeIntrinsics) -> torch.Tensor:
    """Depth of ``z + 0.2 * x + 0.3 * y = 1`` at pixel centres, shape ``(1, H, W)``."""
    u = torch.arange(intrinsics.width, dtype=torch.float64) + 0.5
    v = torch.arange(intrinsics.height, dtype=torch.float64) + 0.5
    ray_x = (u - intrinsics.cx) / intrinsics.fx
    ray_y = (v - intrinsics.cy) / intrinsics.fy
    return 1.0 / (1.0 + 0.2 * ray_x[None, None, :] + 0.3 * ray_y[None, :, None])


def test_datasheet_d435_letterbox_retains_the_full_horizontal_view():
    """The full 848x480 D435 view becomes 224x127 content inside the 224x224 student input."""
    assert D435_DEPTH_848X480.fy == pytest.approx(240.0 / math.tan(math.radians(29.0)))
    letterbox = fit_depth_letterbox(D435_DEPTH_848X480, 224)

    assert (letterbox.content.width, letterbox.content.height) == (224, 127)
    assert (letterbox.pad_left, letterbox.pad_right, letterbox.pad_top, letterbox.pad_bottom) == (0, 0, 48, 49)
    assert (letterbox.content.cx, letterbox.content.cy) == (112.0, 63.5)
    assert (letterbox.output.cx, letterbox.output.cy) == (112.0, 111.5)

    source_hfov = 2.0 * math.degrees(math.atan(D435_DEPTH_848X480.width / (2.0 * D435_DEPTH_848X480.fx)))
    content_hfov = 2.0 * math.degrees(math.atan(letterbox.content.width / (2.0 * letterbox.content.fx)))
    assert content_hfov == pytest.approx(source_hfov)
    assert content_hfov == pytest.approx(88.8, abs=0.1)


def test_letterboxed_real_depth_matches_the_simulated_content():
    """Real-frame resize and simulated low-resolution rendering represent the same wide pinhole view."""
    letterbox = fit_depth_letterbox(D435_DEPTH_848X480, 224)
    output = resize_and_pad_depth(_render_tilted_plane(D435_DEPTH_848X480), letterbox)
    content = output[
        :,
        :,
        letterbox.pad_top : letterbox.pad_top + letterbox.content.height,
        letterbox.pad_left : letterbox.pad_left + letterbox.content.width,
    ]
    direct = _render_tilted_plane(letterbox.content)

    assert output.shape == (1, 1, 224, 224)
    # Nearest sampling picks a source pixel up to half a source pixel from the output ray. The rounded
    # 127-pixel height adds less than 0.2% aspect error relative to the exact 126.79-pixel fit.
    assert torch.allclose(content[:, 0], direct, atol=1.5e-3)
    assert torch.count_nonzero(output[:, :, : letterbox.pad_top]) == 0
    assert torch.count_nonzero(output[:, :, -letterbox.pad_bottom :]) == 0


def test_simulated_content_is_only_padded():
    """The simulator already renders at the fitted resolution, so observation assembly must not resample it."""
    letterbox = fit_depth_letterbox(D435_DEPTH_848X480, 224)
    content = torch.rand(2, 1, letterbox.content.height, letterbox.content.width)
    output = resize_and_pad_depth(content, letterbox)

    assert torch.equal(output[:, :, letterbox.pad_top : letterbox.pad_top + letterbox.content.height], content)


def test_depth_letterbox_rejects_non_square_pixels():
    with pytest.raises(ValueError, match="Square pixels"):
        fit_depth_letterbox(PinholeIntrinsics(848, 480, fx=430.0, fy=440.0, cx=424.0, cy=240.0), 224)


def test_normalize_depth_keeps_missing_returns_at_zero():
    depth = torch.tensor([0.0, 0.6, 2.0, float("nan"), float("inf")])
    assert torch.equal(normalize_depth(depth, 1.2), torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0]))


def test_intrinsic_warp_identity_and_principal_point_shift():
    depth = torch.zeros((1, 1, 5, 7))
    depth[0, 0, 2, 2] = 0.5

    identity = warp_depth_intrinsics(depth, torch.ones(1), torch.zeros((1, 2)))
    shifted = warp_depth_intrinsics(depth, torch.ones(1), torch.tensor([[1.0, 0.0]]))

    assert torch.equal(identity, depth)
    assert shifted[0, 0, 2, 3] == 0.5
    assert torch.count_nonzero(shifted) == 1


def test_depth_calibration_preserves_missing_returns():
    depth = torch.tensor([[[[0.0, 0.5], [1.0, 2.0]]]])
    randomized = randomize_depth_measurement(
        depth,
        depth_scale=torch.tensor([1.1]),
        depth_bias_m=torch.tensor([0.01]),
        noise_std_at_1m_m=0.0,
        missing_return_prob=0.0,
        boundary_corruption_prob=0.0,
        boundary_threshold_m=0.02,
        strength=1.0,
    )

    torch.testing.assert_close(randomized, torch.tensor([[[[0.0, 0.56], [1.11, 2.21]]]]))


def test_depth_noise_is_per_pixel_depth_dependent_and_never_revives_missing_returns():
    depth = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    torch.manual_seed(7)
    expected_noise = torch.randn_like(depth) * (0.004 * depth.square())
    torch.manual_seed(7)
    randomized = randomize_depth_measurement(
        depth,
        depth_scale=torch.ones(1),
        depth_bias_m=torch.zeros(1),
        noise_std_at_1m_m=0.004,
        missing_return_prob=0.0,
        boundary_corruption_prob=0.0,
        boundary_threshold_m=0.02,
        strength=1.0,
    )

    torch.testing.assert_close(randomized, (depth + expected_noise).clamp_min(0.0))
    assert randomized[0, 0, 0, 0] == 0.0


def test_missing_returns_drop_only_valid_pixels():
    depth = torch.tensor([[[[0.0, 0.5], [1.0, 2.0]]]])
    randomized = randomize_depth_measurement(
        depth,
        depth_scale=torch.ones(1),
        depth_bias_m=torch.zeros(1),
        noise_std_at_1m_m=0.0,
        missing_return_prob=1.0,
        boundary_corruption_prob=0.0,
        boundary_threshold_m=0.02,
        strength=1.0,
    )

    assert torch.count_nonzero(randomized) == 0


def test_boundary_corruption_is_confined_to_depth_discontinuities():
    depth = torch.tensor([[[[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]]]])
    torch.manual_seed(3)
    randomized = randomize_depth_measurement(
        depth,
        depth_scale=torch.ones(1),
        depth_bias_m=torch.zeros(1),
        noise_std_at_1m_m=0.0,
        missing_return_prob=0.0,
        boundary_corruption_prob=1.0,
        boundary_threshold_m=0.02,
        strength=1.0,
    )

    assert torch.equal(randomized[..., 0], depth[..., 0])
    assert torch.equal(randomized[..., 3], depth[..., 3])
    assert set(randomized.flatten().tolist()) <= {0.0, 0.5, 1.0}
    assert not torch.equal(randomized[..., 1:3], depth[..., 1:3])
