# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project task registrations."""

import gymnasium as gym

import Cross_Embodiment_CL.tasks  # noqa: F401


def test_task_registrations():
    """Tasks must expose valid environment and agent entry points."""
    expected = {
        "CrossEmbodimentCl-Balance-Cartpole-Direct": {
            "entry_point": "Cross_Embodiment_CL.tasks.balance_direct.config.cartpole.env:BalanceEnv",
            "env_cfg_entry_point": "Cross_Embodiment_CL.tasks.balance_direct.config.cartpole.env_cfg:BalanceEnvCfg",
        },
        "CrossEmbodimentCl-Balance-Marl-Cartpole-Direct": {
            "entry_point": "Cross_Embodiment_CL.tasks.balance_marl_direct.config.cartpole.env:BalanceMarlEnv",
            "env_cfg_entry_point": (
                "Cross_Embodiment_CL.tasks.balance_marl_direct.config.cartpole.env_cfg:BalanceMarlEnvCfg"
            ),
            "default_agent": "skrl",
        },
        "CrossEmbodimentCl-G1-Hand-Table-Direct": {
            "entry_point": "Cross_Embodiment_CL.tasks.g1_hand_table_direct.config.g1_hand_table.env:G1HandTableEnv",
            "env_cfg_entry_point": (
                "Cross_Embodiment_CL.tasks.g1_hand_table_direct.config.g1_hand_table.env_cfg:G1HandTableRunPresetCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "Cross_Embodiment_CL.tasks.g1_hand_table_direct.config.g1_hand_table.agents."
                "rsl_rl_ppo_cfg:G1HandTablePPORunnerCfg"
            ),
            "rsl_rl_distillation_cfg_entry_point": (
                "Cross_Embodiment_CL.tasks.g1_hand_table_direct.config.g1_hand_table.agents."
                "rsl_rl_distillation_cfg:G1HandTableDepthDistillationRunnerCfg"
            ),
            "rsl_rl_state_distillation_cfg_entry_point": (
                "Cross_Embodiment_CL.tasks.g1_hand_table_direct.config.g1_hand_table.agents."
                "rsl_rl_distillation_cfg:G1HandTableStateDistillationRunnerCfg"
            ),
        },
    }

    agent_keys = (
        "rsl_rl_cfg_entry_point",
        "rsl_rl_distillation_cfg_entry_point",
        "rsl_rl_state_distillation_cfg_entry_point",
    )
    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        for key in agent_keys:
            if key in expected_values:
                assert spec.kwargs[key] == expected_values[key]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
