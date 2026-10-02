# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the G1-hand tabletop Direct scene."""

import gymnasium as gym

from . import agents

gym.register(
    id="CrossEmbodimentCl-G1-Hand-Table-Direct",
    entry_point=f"{__name__}.env:G1HandTableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1HandTableRunPresetCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1HandTablePPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1HandTableDepthDistillationRunnerCfg"
        ),
        "rsl_rl_state_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1HandTableStateDistillationRunnerCfg"
        ),
    },
)
