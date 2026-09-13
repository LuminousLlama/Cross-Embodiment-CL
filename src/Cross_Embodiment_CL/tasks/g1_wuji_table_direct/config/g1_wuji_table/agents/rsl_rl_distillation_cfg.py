# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL DAgger distillation of the privileged PPO teacher into a deployable student.

The student drives every rollout and the frozen teacher labels each visited state; the loss is the
MSE between the two mean actions.  Launch with ``--agent <entry point> --checkpoint <teacher.pt>``:
a PPO checkpoint loads into the teacher only.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg, RslRlMLPModelCfg

from .rsl_rl_ppo_cfg import G1WujiTablePPORunnerCfg

_TEACHER_ACTOR = G1WujiTablePPORunnerCfg().actor


@configclass
class G1WujiTableStateDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Plumbing check: a student with the teacher's own 117-D privileged observation.

    With identical inputs and architecture the student should reach the teacher's success rate, so a
    shortfall here is a pipeline bug rather than an observability limit.
    """

    num_steps_per_env = 32
    max_iterations = 300
    save_interval = 50
    experiment_name = "g1_wuji_table_distill"
    obs_groups = {"teacher": ["policy"], "student": ["policy"]}
    # Must match the PPO actor exactly: the checkpoint's actor_state_dict is loaded strictly.
    teacher = RslRlMLPModelCfg(
        hidden_dims=_TEACHER_ACTOR.hidden_dims,
        activation=_TEACHER_ACTOR.activation,
        obs_normalization=_TEACHER_ACTOR.obs_normalization,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    student = RslRlMLPModelCfg(
        hidden_dims=_TEACHER_ACTOR.hidden_dims,
        activation=_TEACHER_ACTOR.activation,
        obs_normalization=True,
        # The loss only fits the mean, so this std never trains: it is fixed rollout noise that
        # widens the visited states slightly without derailing the student's own trajectories.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.05),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        # Storage yields one batch per rollout step; one optimizer step per batch keeps the
        # accumulated graph to a single step, which matters once the student encodes images.
        gradient_length=1,
        max_grad_norm=1.0,
    )
