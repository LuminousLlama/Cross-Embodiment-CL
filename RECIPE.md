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

Two stages off the final teacher. Stage 2 fixes the train/eval gap.

```bash
# 1. Default rollouts, 2048 envs.  ->  model_4999, eval 62%
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <teacher>/model_7246.pt \
  --num_envs 2048 --seed 0 --max_iterations 5000 \
  --run_name student_mse_tableonly \
  presets=distill,dr_full env.reset.object_on_table=True

# 2. Full-episode rollouts, 384 envs (48 GB GPU).  ->  model_5398, eval 72%
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <stage1>/model_4999.pt \
  --num_envs 384 --seed 0 --max_iterations 400 \
  --run_name student_honest \
  presets=distill,dr_full env.reset.object_on_table=True \
  agent.num_steps_per_env=480 agent.init_at_random_ep_len=True \
  agent.algorithm.gradient_length=11
```

`env.reset.object_on_table=True` is required: the default spawns the apple up
to 20 cm above the table and costs ~21 points of student success.

Stage 1 training `Task/success` reads ~85% against 62% eval: with 32-step
rollouts the student updates ~15 times per episode on that episode's own labels.
Stage 2 rolls out one full episode (480 steps) per update, so training success
matches eval (70.8% vs 72.2%). Keep random initial episode lengths (synced
resets make the per-timestep batches phase-ordered and the student forgets) and
set `gradient_length` so each step sees ~4k samples (with 1 it overfits).

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

Use `--train_env_cfg --num_envs 512 --episodes 2048` for precise evals; 240
episodes is ±3 points.

## Results

| checkpoint | success | pos err | rot err | below table |
|---|---|---|---|---|
| teacher `model_7246` | 71.5% | 5.7 cm | 30.3 deg | 1.6% |
| student `model_4999` | 62.2% | 3.2 cm | 36.9 deg | 0.6% |
| student `model_5398` | 72.2% | 3.3 cm | 31.4 deg | 1.0% |
