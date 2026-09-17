# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO baseline for the G1-Wuji tabletop task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class G1WujiTablePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Untuned state-based PPO baseline for the 171-D G1-Wuji observation."""

    num_steps_per_env = 32
    max_iterations = 10_000
    save_interval = 250
    experiment_name = "g1_wuji_table_direct"
    # The actor retains the deployable policy tensor. The critic gets the clean simulator tensor with the
    # exact per-environment DR state appended by the environment.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[2048, 1024, 512],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[2048, 1024, 512],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.001 collapses exploration before the apple is ever lifted; 0.005 preserves exploration
        # while the gravity curriculum ramps up.
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
