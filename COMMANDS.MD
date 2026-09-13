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

| preset  | physics          | envs | visualizer  | keypoint markers | Newton visual shapes |
|---------|------------------|------|-------------|------------------|----------------------|
| (none)  | `newton_mjwarp`  | 1    | `newton_gl` | on               | on                   |
| `train` | `newton_mjwarp`  | 2048 | none        | off              | off                  |
| `debug` | `newton_mjwarp`  | 4    | `newton_gl` | on               | on                   |
| `eval`  | `newton_mjwarp`  | 16   | none        | off              | off                  |
| `distill` | `newton_mjwarp` | 1024 | none        | off              | on, student camera   |
| `depth_view` | `newton_mjwarp` | 1 | `newton_gl` | off | on, depth stream |

Anything given explicitly still wins over the preset: `--num_envs 64`, `--viz none`
(headless eval), `--viz kit`, `env.debug.keypoint_markers=False`,
`env.sim.physics.load_visual_shapes=True`.  Without a preset a viewer opens, so pass `--viz none`
for a headless run on the server.  `physics=isaacsim_physx` swaps the backend with no preset or
with `debug`; `train` and `eval` pick their own headless physics config, so combining them with
`physics=` errors with "Conflicting global presets" -- use the path selector
`env.sim.physics=isaacsim_physx` there.

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

Zero actions, 4 environments, Newton viewer and keypoint markers:

```
uv run isaaclab zero_agent --task CrossEmbodimentCl-G1-Wuji-Table-Direct presets=debug
```

## Inspect the student depth camera

`depth_view` is an interactive, one-environment Newton GL preset. Its stream panel colorizes the
depth-only D435 output over 0.1--1.2 m; it does not change the student's depth tensor.

In a linked worktree, reuse the sibling checkout's prepared venv and put this worktree's source
first on `PYTHONPATH`:

```
PYTHONPATH=$PWD/src ../Cross-Embodiment-CL/.venv/bin/isaaclab zero_agent \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct presets=depth_view
```

## Depth-student distillation

The deployable student consumes 87-D proprioception plus one normalized 224x224 D435 depth image.
Use the depth distillation runner to play a student checkpoint in the camera viewer:

```
PYTHONPATH=$PWD/src ../Cross-Embodiment-CL/.venv/bin/isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <student-checkpoint.pt> --num_envs 1 presets=depth_view
```

Evaluate deterministically, headlessly, at full gravity with an absolute checkpoint path:

```
PYTHONPATH=$PWD/src ../Cross-Embodiment-CL/.venv/bin/python scripts/eval_policy.py \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint <absolute-student-checkpoint.pt> presets=distill --num_envs 2 --episodes 64
```

## Evaluate a checkpoint

Headless, for eval-time success analysis: 16 environments, no viewer, markers, or visual shapes.

```
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct \
  --checkpoint logs/rsl_rl/g1_wuji_table_direct/<run>/model_<n>.pt \
  presets=eval
```

## Watch a checkpoint

No preset: 1 environment in the `newton_gl` viewer with keypoint markers (add `presets=debug`
for 4).  The Newton viewer renders without Isaac Sim; pass `--viz kit` only where Isaac Sim is
installed.

```
uv run isaaclab play --rl_library rsl_rl \
  --task CrossEmbodimentCl-G1-Wuji-Table-Direct --real-time \
  --checkpoint logs/rsl_rl/g1_wuji_table_direct/<run>/model_<n>.pt
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
scripts/analyze_run.py curve  logs/rsl_rl/g1_wuji_table_direct/<run> --tag Task/success --bins 10
scripts/run_summary.py logs/rsl_rl/g1_wuji_table_direct/<run_a> logs/rsl_rl/g1_wuji_table_direct/<run_b>
```
