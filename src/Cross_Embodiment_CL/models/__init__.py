# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen learned models used by Cross Embodiment environments."""

from .wuji_latent import WujiLatentActionPipeline, WujiLatentProjection

__all__ = ["WujiLatentActionPipeline", "WujiLatentProjection"]
