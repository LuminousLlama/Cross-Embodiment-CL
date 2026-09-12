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


## Run presets

The G1 + Wuji task bundles its launch settings into run presets, selected with
`presets=NAME` (defined in `g1_wuji_table/env_cfg.py`):

| preset  | physics          | envs | visualizer  | keypoint markers | link-contact prints | Newton visual shapes |
|---------|------------------|------|-------------|------------------|---------------------|----------------------|
| (none)  | `isaacsim_physx` | 1    | none        | off              | off                 | on                   |
| `train` | `newton_mjwarp`  | 2048 | none        | off              | off                 | off                  |
| `debug` | `newton_mjwarp`  | 4    | `newton_gl` | on               | on                  | on                   |
| `eval`  | `newton_mjwarp`  | 16   | `newton_gl` | on               | off                 | on                   |

Anything given explicitly still wins over the preset: `--num_envs 64`, `--viz none`
(headless eval), `--viz kit`, `env.debug.link_contacts=False`,
`env.sim.physics.load_visual_shapes=True`.  To change the backend inside a preset, use the path
selector, e.g. `env.sim.physics=isaacsim_physx`.  Don't combine a preset with `physics=`: that
errors with "Conflicting global presets" whenever the two pick different physics configs.

## Train the G1 + Wuji tabletop task

```
uv run isaaclab train --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --max_iterations 5000 --run_name g1_wuji_table_2048 \
  presets=train
```

Add `env.sim.physics=ovphysx` to train the same task on PhysX without Kit
(needs `uv sync --extra ovphysx`), or `env.sim.physics=physx` to let the backend resolve
automatically (Isaac Sim PhysX where installed, otherwise OvPhysX).

Useful override:

```
# Newton solver capacity. Memory grows quickly with njmax; lower it when environment
# count matters more than constraint headroom, raise it if the sim diverges into NaN.
env.sim.physics.solver_cfg.njmax=256 env.sim.physics.solver_cfg.nconmax=128
```

## Debug the scene

Zero actions, 4 environments, Newton viewer, keypoint markers and per-link contact prints:

```
uv run isaaclab zero_agent --task CrossEmbodimentCl-G1-Wuji-Table-Direct presets=debug
```

## Play / evaluate a checkpoint

The `newton_gl` viewer renders without Isaac Sim; pass `--viz kit` only where Isaac Sim
is installed, or `--viz none` for a headless run.

```
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct --real-time \
  --checkpoint logs/rsl_rl/g1_wuji_table_direct/<run>/model_<n>.pt \
  presets=eval
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
