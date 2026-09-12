# COMMANDS

reference for commands we commonly run


## Isaac sim kit GUI 
```
uv run isaacsim
```

## Launch with full kit experience
```
uv run isaaclab zero_agent --task CrossEmbodimentCl-G1-Wuji-Table-Direct --viz kit --experience isaacsim.exp.full.kit
```


## Train the G1 + Wuji tabletop task

Newton (MJWarp) backend, 2048 environments, 5000 iterations:

```
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --num_envs 2048 --device cuda --max_iterations 5000 \
  --run_name g1_wuji_table_2048 \
  physics=newton_mjwarp
```

Swap `physics=newton_mjwarp` for `physics=ovphysx` to train the same task on PhysX
without Kit, or `physics=physx` to let the backend resolve automatically (it prefers
Isaac Sim PhysX where that is installed and falls back to OvPhysX where it is not).
OvPhysX needs its extra: `uv sync --extra ovphysx`.

Useful overrides:

```
# Headless training never draws the scene, and Newton clones render meshes per
# environment, so dropping them is a large memory saving at no cost.
env.sim.physics.load_visual_shapes=False

# Newton solver capacity. Memory grows quickly with njmax; lower it when environment
# count matters more than constraint headroom, raise it if the sim diverges into NaN.
env.sim.physics.solver_cfg.njmax=256 env.sim.physics.solver_cfg.nconmax=128
```

Play a trained checkpoint (`--viz newton` renders without Isaac Sim; use `--viz kit`
only where Isaac Sim is installed):

```
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --num_envs 16 --device cuda --viz newton --real-time \
  --checkpoint logs/rsl_rl/g1_wuji_table_direct/<run>/model_<n>.pt \
  physics=newton_mjwarp
```

# Play checkpoint
```
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --num_envs 16 --device cuda --viz newton --real-time \
  --checkpoint /path/to/checkpoint \
  physics=newton_mjwarp
```



## Tensorboard 
```
uv run tensorboard --logdir <path>

# on remote server 
uv run tensorboard --logdir <path> 
```


## Reading TensorBoard logs without launching TensorBoard

`scripts/analyze_run.py` caches a run's scalars to `<run>/analysis/scalars.csv` and
queries them; `scripts/run_summary.py` prints the key tags of one or two runs as a
binned table, which is what comparing experiments needs.

```
scripts/analyze_run.py tags   logs/rsl_rl/g1_wuji_table_direct/<run>
scripts/analyze_run.py curve  logs/rsl_rl/g1_wuji_table_direct/<run> --tag Metrics/success --bins 10
scripts/run_summary.py logs/rsl_rl/g1_wuji_table_direct/<run_a> logs/rsl_rl/g1_wuji_table_direct/<run_b>
```
