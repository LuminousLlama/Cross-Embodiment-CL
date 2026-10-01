# G1 + Wuji Table Commands

Use these commands directly in the primary checkout. In a linked worktree, prefix them with `./worktree-local`.

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
