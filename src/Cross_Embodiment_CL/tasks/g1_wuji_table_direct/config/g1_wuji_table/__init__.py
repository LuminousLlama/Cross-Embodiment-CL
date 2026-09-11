# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the minimal G1-Wuji Direct scene."""

import gymnasium as gym

gym.register(
    id="CrossEmbodimentCl-G1-Wuji-Table-Direct",
    entry_point=f"{__name__}.env:G1WujiTableEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:G1WujiTableEnvCfg"},
)
