# G1 + Hand Table Commands

Use these commands directly in the primary checkout. In a linked worktree, prefix them with `./worktree-local`.

Select one homogeneous hand per run with `env.hand_type=wuji|inspire|dex3` (default `wuji`),
independently of `--agent` and `physics`. All hands use 25 actions and the same padded observations.
Checkpoints from the earlier unmasked observation schema must be retrained. The `force` overlay is Wuji-only.
The environment defaults to seed 42; override it with `env.seed=<seed>` (or `--seed` for training).

Only `YcbApple` is active by default in every run profile. Other reviewed objects require an explicit CLI
selection; one selected object is assigned per environment at initialization and retained across resets.
For example, append:

```bash
env.active_objects=YcbApple,YcbBanana,YcbTomatoSoupCan
```

Each asset uses its own canonical root height. `env.object_spawn_z_range=[0.04,0.24]` and
`env.object_rest_height=0.04` retain their apple-reference meaning; other assets add their
canonical height offset. `env.reset.object_on_table=True` fixes the height at that rest value.
Object XY and goal XYZ/roll/pitch/yaw are sampled on every reset, including `dr_none`;
the object retains its authored starting orientation. Goal ranges are shared across the bank.

For multi-object PhysX runs, append `physics=isaacsim_physx` explicitly. The installed OVPhysX
runtime does not support this heterogeneous clone plan; `physics=physx` may select OVPhysX automatically.

## Train

Teacher PPO training uses `dr_curriculum`, starting at level 0 and advancing toward 50
when success exceeds 40%. Student distillation uses `dr_full`, starting at maximum
domain randomization (level 50) with the same DR terms.
Gravity ramps from 0g to 1g; append `env.adr.gravity_start=1.0` to keep full gravity throughout.
`Task/success` is the successful fraction of episodes completed within each learning iteration;
`Task/success-percentage-completed-episode-count` is its denominator (zero when no episodes finish).
Iterations without completions omit `Task/success`. ADR retains its independent update window.

Penetration diagnostics are off by default. Append `env.log_penetration_metrics=true`
to opt in to MJWarp penetration metrics; this scans the contact buffer every policy step.
Contact-capacity diagnostics are independent: `env.contact_debug=true`.

### PPO

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --num_envs <envs> \
  --seed <seed> \
  --max_iterations <iterations> \
  --run_name <run-name> \
  presets=train,dr_curriculum
```

### Resume PPO

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --checkpoint <absolute-checkpoint> \
  --num_envs <envs> \
  --seed <seed> \
  --max_iterations <additional-iterations> \
  --run_name <run-name> \
  env.adr.initial_level=<resume_level>
  presets=train,dr_curriculum
```

### Depth-student distillation

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-teacher-checkpoint> \
  --num_envs <envs> \
  --seed <seed> \
  --max_iterations <iterations> \
  --run_name <run-name> \
  presets=distill,dr_full
```

### Depth-and-force-student distillation

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-teacher-checkpoint> \
  --num_envs <envs> \
  --seed <seed> \
  --max_iterations <iterations> \
  --run_name <run-name> \
  presets=distill,force,dr_full
```

## Play

### PPO checkpoint

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --checkpoint <absolute-checkpoint> \
  --num_envs <envs> \
  --real-time \
  presets=debug
```

### Depth-student checkpoint

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-student-checkpoint> \
  --num_envs <envs> \
  --real-time \
  presets=debug,depth_view
```

### Depth-and-force-student checkpoint

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-student-checkpoint> \
  --num_envs <envs> \
  --real-time \
  presets=debug,depth_view,force
```

## Evaluate

Evaluation uses `dr_full`. Results default to `<checkpoint-directory>/eval/<checkpoint-name>.json`.
Add `--enable-train-time-noise` to sample the saved training-time action distribution.

### PPO checkpoint

```bash
uv run python scripts/eval_policy.py \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --checkpoint <absolute-checkpoint> \
  --num_envs <envs> \
  --episodes <episodes> \
  presets=eval,dr_full
```

### Depth-student checkpoint

```bash
uv run python scripts/eval_policy.py \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-student-checkpoint> \
  --num_envs <envs> \
  --episodes <episodes> \
  presets=distill,dr_full
```

### Depth-and-force-student checkpoint

```bash
uv run python scripts/eval_policy.py \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-student-checkpoint> \
  --num_envs <envs> \
  --episodes <episodes> \
  presets=distill,force,dr_full
```

## Inspect

### Rebuild Inspire hand USD

```bash
uv run python scripts/make_hand_colliders.py --hand inspire
uv run isaaclab -p scripts/export_hand_asset.py --hand inspire --visualizer none
```

### Inspire hand full-ROM sweep

```bash
uv run isaaclab -p scripts/sweep_hand_rom.py --hand inspire --visualizer none --device cpu
uv run python scripts/sweep_hand_newton_rom.py --hand inspire
```

### Rebuild and sweep Dex3-1 hand

```bash
uv run python scripts/make_hand_colliders.py --hand dex3
uv run isaaclab -p scripts/export_hand_asset.py --hand dex3 --visualizer none
uv run isaaclab -p scripts/sweep_hand_rom.py --hand dex3 --visualizer none --device cpu
uv run python scripts/sweep_hand_newton_rom.py --hand dex3
```

Abrupt 0-to-70-degree regression:

```bash
uv run isaaclab -p scripts/test_inspire_hand_step.py --visualizer none --device cpu
```

### G1 hand assembly validation

```bash
uv run isaaclab -p scripts/test_g1_hand_assembly.py --hand inspire --visualizer none --device cpu
uv run isaaclab -p scripts/test_g1_hand_assembly.py --hand dex3 --visualizer none --device cpu
```

### Scene with zero actions

```bash
uv run isaaclab zero_agent \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --num_envs 1 --device cuda \
  presets=debug,dr_none env.hand_type=inspire env.reset.object_on_table=True
```

Repeat with `env.hand_type=dex3` and `env.hand_type=wuji`; select `physics=isaacsim_physx`
to inspect PhysX separately. Check fixed pelvis, wrist attachment, legal open fingers (including
Inspire thumb), visible apple/table, and green/red goal/object markers.
For Inspire, followers should move with their drivers without independent follower targets.
Zero actions decode the zero MANO latent and command arm midranges; they do not hold the reset pose.

### Integrated latent motion and depth view

```bash
uv run isaaclab random_agent \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --num_envs 1 --device cuda --max_steps 600 \
  presets=debug,depth_view,dr_none env.hand_type=inspire env.reset.object_on_table=True
```

Repeat for Dex3 and Wuji. Verify bounded arm/hand motion, Inspire follower coupling,
apple contacts, and the depth preview matching the hand camera. These checks use full gravity.

### Bounded headless scene smoke

```bash
uv run isaaclab zero_agent \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  --num_envs 4 --device cuda --viz none --max_steps 64 \
  presets=eval,dr_none env.hand_type=dex3 env.reset.object_on_table=True
```

Repeat for the other hands and `physics=isaacsim_physx`. Numerical reset/step, semantic
padding, command-limit, and fixed-pelvis checks are in the existing runtime test:

```bash
uv run pytest tests/test_hand_latent_round_trip.py -q \
  -k 'observation_uses_raw_joint_positions or multi_env_reset'
```

### Physical hand-contact sensor check

This headless check replaces the apple with a small sphere and contacts every declared hand body,
then checks force activation, clearing after separation, environment isolation, and observation wiring
on both backends. It also exercises Wuji's optional detailed force sensors.

```bash
uv run pytest tests/test_hand_latent_round_trip.py -q -s \
  -k every_hand_body_sensor
```

### Student depth input

```bash
uv run isaaclab zero_agent \
  --task CrossEmbodimentCl-G1-Hand-Table-Direct \
  presets=debug,depth_view
```

## Common ADR overrides

Append overrides after the preset selector.

### Full DR with a success-gated curriculum

Replace the existing preset selector with:

```bash
presets=<run-profile>,dr_curriculum
```

### Keep gravity at 1g throughout ADR

```bash
env.adr.gravity_start=1.0
```

### Gravity-only ADR from 0g to 1g

Replace the existing preset selector with:

```bash
presets=<run-profile>,dr_none env.adr.initial_level=0
```

### Gravity-only ADR from a nonzero offset to 1g

Replace the existing preset selector with:

```bash
presets=<run-profile>,dr_none env.adr.initial_level=0 env.adr.gravity_start=<gravity-fraction>
```

## Miscellaneous

### TensorBoard

```bash
uv run tensorboard \
  --logdir logs/rsl_rl/g1_hand_table_direct \
  --port <port> \
  --bind_all
```
