# G1-Wuji D435 Student Distillation

## Status

The privileged PPO teacher R007 was behavior-cloned into a depth student with RSL-RL DAgger. The
student received the former 87-D proprioception and one normalized 224x224 D435 depth image; the teacher
used the former 117-D privileged observation. The current environment uses 141-D proprioception and a
171-D privileged observation, so that checkpoint predates and is incompatible with the current observation contract.

New distillation runs retain the D435's full-width view: the 848x480 source is resized to 224x127 and
padded with 48 zero rows above and 49 below. The simulator renders the same 224x127 pinhole content
before padding, while the CNN interface remains 224x224.

The stopped `depth_r0` run's `model_600.pt` checkpoint was evaluated locally with deterministic
actions, full apple weight, two environments, and 64 completed episodes. It achieved 63/64 success
(98.44%), 2.71 cm final position error, 5.49 deg final rotation error, 1.58 mm maximum hand-apple
penetration, and 10.60 N thumb force. The sole failure was a workspace exit.

This clears the Stage-1 behavior-cloning bar of at least 90% deterministic success. It does not by
itself show that the policy materially uses depth: the fixed scene and former 87-D controller state may
permit a largely proprioceptive trajectory. A zero-depth camera ablation is the next validation if
the visual-policy claim needs to be established.

## Reproduce

Use the commands in `COMMANDS.MD`. For a linked worktree, run the sibling main checkout's prepared
venv with `PYTHONPATH=$PWD/src`, ensuring project imports resolve from the active worktree.
