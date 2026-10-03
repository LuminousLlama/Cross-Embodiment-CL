# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Surgical, in-memory workaround for RSL-RL 5.4.1's unordered episode metrics."""

import inspect
import textwrap

from rsl_rl.utils.logger import Logger

_UPSTREAM_LOG = Logger.log
_UNORDERED_LOOP = "for key in {k for ep_info in self.ep_extras for k in ep_info}:"
_ORDERED_LOOP = "for key in sorted({k for ep_info in self.ep_extras for k in ep_info}):"


def install_alphabetical_logging() -> None:
    """Sort episode metric names without copying the logger or editing site-packages.

    The replacement affects all RSL-RL loggers in this process. Keep the upstream
    method intact except for its key iterator; reject source changes that need review.
    """
    if Logger.log is not _UPSTREAM_LOG:
        if getattr(Logger.log, "_cl_alphabetical_logging", False):
            return
        raise RuntimeError("RSL-RL Logger.log was already replaced by another hook; review alphabetical_logging.py.")

    source = textwrap.dedent(inspect.getsource(_UPSTREAM_LOG))
    if source.count(_UNORDERED_LOOP) != 1:
        raise RuntimeError("RSL-RL Logger.log changed; review or remove the workaround in alphabetical_logging.py.")
    source = source.replace(_UNORDERED_LOOP, _ORDERED_LOOP)
    namespace = {}
    exec(compile(source, f"{__file__}:Logger.log", "exec"), _UPSTREAM_LOG.__globals__, namespace)
    ordered_log = namespace["log"]
    ordered_log._cl_alphabetical_logging = True
    Logger.log = ordered_log
