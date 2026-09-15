# G1-Wuji D435 Student Distillation

## Status

The privileged PPO teacher R007 was behavior-cloned into a depth student with RSL-RL DAgger. The
student received the former 87-D proprioception and one normalized 224x224 D435 depth image; the teacher
used the former 117-D privileged observation. The current student uses 141-D proprioception, 20-D virtual
force, and depth alongside the teacher's 171-D privileged observation, so that checkpoint predates and is
incompatible with the current observation contract.

New distillation runs retain the D435's full-width view: the 848x480 source is resized to 224x127 and
padded with 48 zero rows above and 49 below. The simulator renders the same 224x127 pinhole content
before padding, while the CNN interface remains 224x224.

The `depth_view` preset's camera panel displays that finalized normalized 224x224 tensor, including
the zero-padded rows and any enabled camera/depth randomization, rather than the native 224x127 render buffer.

The stopped `depth_r0` run's `model_600.pt` checkpoint was evaluated locally with deterministic
actions, full apple weight, two environments, and 64 completed episodes. It achieved 63/64 success
(98.44%), 2.71 cm final position error, 5.49 deg final rotation error, 1.58 mm maximum hand-apple
penetration, and 10.60 N thumb force. The sole failure was a workspace exit.

This clears the Stage-1 behavior-cloning bar of at least 90% deterministic success. It does not by
itself show that the policy materially uses depth: the fixed scene and former 87-D controller state may
permit a largely proprioceptive trajectory. A zero-depth camera ablation is the next validation if
the visual-policy claim needs to be established.

## Virtual Wuji force packet

The environment now has a backend-agnostic virtual force pipeline for the 20 physical Wuji joints.
It uses Isaac Lab 3.0's common `ContactSensor` force, friction, and contact-point matrices together
with the articulation's public `body_link_jacobian_w`; it does not call the old PhysX tensor view or
Newton-private buffers. The implementation has been exercised headlessly on both Newton MJWarp and
Isaac Sim PhysX.

`presets=distill` enables the pipeline. Other presets leave it disabled, so PPO training and ordinary
evaluation do not pay for detailed contact reporting. When enabled, observations contain a separate
`force` tensor with shape `(num_envs, 20)` in `right_finger1_joint1` through
`right_finger5_joint4` order. The depth-distillation student consumes this tensor alongside its unchanged
141-D proprioceptive `student` group and camera image, giving it 161 low-dimensional inputs. Code that
needs the diagnostic packet can call
`env.unwrapped.get_virtual_force_output()` to obtain ideal contact torque, observed/modelled torque,
and packet validity.

The sensor model supports episode-randomized scale, bias and latency plus noise, EMA filtering,
dropout, clipping, and an optional `wuji_force_model_v1` system-identification artifact. Configure
these under `env.virtual_force`; set `env.virtual_force.system_id_model_path=<model.npz>` to enable the
fitted real-estimator mapping. No benchmark fit is selected by default: the existing artifact was fit
against the previous simulator/controller and should be replayed or refit against traces from this
environment before it becomes a training default.

Two fidelity limits are explicit. First, the common API reports a total force and one averaged contact
point per sensing-body/counterpart pair, so multiple non-collinear contacts on the same pair only
approximate the true net moment. Second, the task receives the final contact snapshot at the 60 Hz
policy boundary. The pipeline holds that sample for two virtual updates to preserve the real
estimator's 120 Hz clock, but it cannot reconstruct the intermediate physics-step contact sample.

## Reproduce

Use the commands in `COMMANDS.MD`. For a linked worktree, run the sibling main checkout's prepared
venv with `PYTHONPATH=$PWD/src`, ensuring project imports resolve from the active worktree.
