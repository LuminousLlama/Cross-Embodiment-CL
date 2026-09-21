# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bound OpenUSD concurrency while the task's collider-rich scene is imported."""

from pxr import Work

_previous_concurrency: int | None = None


def limit_openusd_scene_import() -> None:
    """Limit OpenUSD scene-import work to one thread, preserving the prior limit."""
    global _previous_concurrency

    if _previous_concurrency is None:
        _previous_concurrency = Work.GetConcurrencyLimit()
    Work.SetConcurrencyLimit(1)
    if Work.GetConcurrencyLimit() != 1:
        raise RuntimeError("OpenUSD concurrency was initialized before the G1-Wuji scene-import guard.")


def restore_openusd_concurrency() -> None:
    """Restore the OpenUSD concurrency limit saved before scene import."""
    global _previous_concurrency

    if _previous_concurrency is None:
        return
    Work.SetConcurrencyLimit(_previous_concurrency)
    _previous_concurrency = None


# This module is imported before Isaac Lab can initialize OpenUSD's worker arena.
limit_openusd_scene_import()
