# G1-Wuji table task: teacher + depth student

Teacher 71.5% eval, student 62.1% eval. All numbers are deterministic
`eval_policy.py` at ADR level 50 with full DR.

## Teacher

Four stages, each resuming the previous checkpoint. 8192 envs throughout.

```bash
# 1. Scratch, ADR from level 0.  ->  model_2499, eval 23%
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --num_envs 8192 --seed 0 --max_iterations 2500 \
  --run_name teacher_adr0 \
  presets=train,dr_full env.adr.initial_level=0

# 2. Resume at ADR 44.  ->  model_3248, eval 46%
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --checkpoint <stage1>/model_2499.pt \
  --num_envs 8192 --seed 0 --max_iterations 750 \
  --run_name teacher_adr44 \
  presets=train,dr_full env.adr.initial_level=44

# 3. Resume at ADR 50.  ->  model_5247, eval 55%
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --checkpoint <stage2>/model_3248.pt \
  --num_envs 8192 --seed 0 --max_iterations 2000 \
  --run_name teacher_adr50 \
  presets=train,dr_full env.adr.initial_level=50

# 4. Resume with gamma 0.995.  ->  model_7246, eval 71.5%   <- final teacher
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --checkpoint <stage3>/model_5247.pt \
  --num_envs 8192 --seed 0 --max_iterations 2000 \
  --run_name teacher_gamma995 \
  presets=train,dr_full env.adr.initial_level=50 \
  agent.algorithm.gamma=0.995
```

Fixed PPO settings: `learning_rate=5e-4`, `lam=0.95`, `entropy_coef=0.001`,
`num_steps_per_env=32`, `num_learning_epochs=5`, `num_mini_batches=4`.

`env.adr.initial_level=0` is required on stage 1. The `dr_full` preset pins
`initial_level=50`; starting there from scratch stalls at ~0.05 success.

## Student

Single 5000-iteration DAgger run off the final teacher. 2048 envs.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <teacher>/model_7246.pt \
  --num_envs 2048 --seed 0 --max_iterations 5000 \
  --run_name student_mse_tableonly \
  presets=distill,dr_full env.reset.object_on_table=True
```

Fixed distillation settings: `loss_type=mse`, `learning_rate=1e-3`,
`num_learning_epochs=2`, `num_steps_per_env=32`.

`env.reset.object_on_table=True` is required: the default spawns the apple up
to 20 cm above the table and costs ~21 points of student success. `huber` is
within noise of `mse`. 3096 envs OOMs on a 24 GB card.

## Eval

```bash
# teacher
uv run python scripts/eval_policy.py \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --checkpoint <teacher>/model_7246.pt \
  --num_envs 48 --episodes 240 \
  presets=eval,dr_full

# student
uv run python scripts/eval_policy.py \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <student>/model_4999.pt \
  --num_envs 48 --episodes 240 \
  presets=distill,dr_full env.reset.object_on_table=True
```

Student eval needs `presets=distill`; `presets=eval` has no `camera`
observation and errors out.

Training-time `Task/success` is inflated for distillation because the student
keeps updating during the rollout. Only `eval_policy.py` numbers count.

## Results

| checkpoint | success | pos err | rot err | below table |
|---|---|---|---|---|
| teacher `model_7246` | 71.5% | 5.7 cm | 30.3 deg | 1.6% |
| student `model_4999` | 62.1% | 1.9 cm | 30.1 deg | 0.0% |
