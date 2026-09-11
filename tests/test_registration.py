# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the generated task registrations."""

import gymnasium as gym

import Cross_Embodiment_CL.tasks  # noqa: F401


def test_task_registrations():
    """The generated tasks must expose valid environment and agent entry points."""
    expected = {
        "CrossEmbodimentCl-Balance-Cartpole-Direct": {
            "entry_point": "Cross_Embodiment_CL.tasks.balance_direct.config.cartpole.env:BalanceEnv",
            "env_cfg_entry_point": "Cross_Embodiment_CL.tasks.balance_direct.config.cartpole.env_cfg:BalanceEnvCfg",
        },
        "CrossEmbodimentCl-Balance-Marl-Cartpole-Direct": {
            "entry_point": "Cross_Embodiment_CL.tasks.balance_marl_direct.config.cartpole.env:BalanceMarlEnv",
            "env_cfg_entry_point": "Cross_Embodiment_CL.tasks.balance_marl_direct.config.cartpole.env_cfg:BalanceMarlEnvCfg",
            "default_agent": "skrl",
        },
        "CrossEmbodimentCl-G1-Wuji-Table-Direct": {
            "entry_point": "Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env:G1WujiTableEnv",
            "env_cfg_entry_point": (
                "Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.env_cfg:G1WujiTableEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "Cross_Embodiment_CL.tasks.g1_wuji_table_direct.config.g1_wuji_table.agents."
                "rsl_rl_ppo_cfg:G1WujiTablePPORunnerCfg"
            ),
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "rsl_rl_cfg_entry_point" in expected_values:
            assert spec.kwargs["rsl_rl_cfg_entry_point"] == expected_values["rsl_rl_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
