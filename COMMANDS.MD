# G1 + Wuji Table Commands

Use these commands directly in the primary checkout. In a linked worktree, prefix them with `./worktree-local`.

All 11 reviewed YCB objects are active by default and are assigned one per environment at initialization. Select a
subset by appending, for example:

```bash
env.active_objects=[YcbApple,YcbBanana,YcbTomatoSoupCan]
```

## Train

Training and distillation use full domain randomization with `dr_full`.

### PPO

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --num_envs <envs> \
  --seed <seed> \
  --max_iterations <iterations> \
  --run_name <run-name> \
  presets=train,dr_full
```

### Resume PPO

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --checkpoint <absolute-checkpoint> \
  --num_envs <envs> \
  --seed <seed> \
  --max_iterations <additional-iterations> \
  --run_name <run-name> \
  presets=train,dr_full
```

### Depth-student distillation

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
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
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
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
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --checkpoint <absolute-checkpoint> \
  --num_envs <envs> \
  --real-time \
  presets=debug
```

### Depth-student checkpoint

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-student-checkpoint> \
  --num_envs <envs> \
  --real-time \
  presets=debug,depth_view
```

### Depth-and-force-student checkpoint

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
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
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --checkpoint <absolute-checkpoint> \
  --num_envs <envs> \
  --episodes <episodes> \
  presets=eval,dr_full
```

### Depth-student checkpoint

```bash
uv run python scripts/eval_policy.py \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-student-checkpoint> \
  --num_envs <envs> \
  --episodes <episodes> \
  presets=distill,dr_full
```

### Depth-and-force-student checkpoint

```bash
uv run python scripts/eval_policy.py \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-student-checkpoint> \
  --num_envs <envs> \
  --episodes <episodes> \
  presets=distill,force,dr_full
```

## Inspect

### Scene with zero actions

```bash
uv run isaaclab zero_agent \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  presets=debug
```

### Student depth input

```bash
uv run isaaclab zero_agent \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  presets=debug,depth_view
```

## Common ADR overrides

Append overrides after the preset selector.

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
  --logdir logs/rsl_rl/g1_wuji_table_direct \
  --port <port> \
  --bind_all
```
