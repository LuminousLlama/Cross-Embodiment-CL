---
name: cross-embodiment-remote-training
summary: Launches, queues, monitors, and retrieves reproducible headless Isaac Lab training runs on configured remote servers.
audience: developer
status: experimental
---

# Remote Training Operations

## When To Use

Use for remote checkpoint evaluation, training, resume, queued experiments, status checks, crash recovery, and artifact retrieval. Resolve the primary checkout with `git worktree list --porcelain`, then read `<primary>/.local_untracked/remote-servers.md`; never use or create a linked-worktree copy. If the primary profile does not exist, copy its `remote-servers.md.template` and stop before attempting a remote launch.

## Delegation

Delegate mechanical remote operations to a `gpt-5.6-luna` subagent for token efficiency. Give it the exact server, repository, checkpoint, command, GPU assignment, and requested status cadence. Keep experiment design, changes to training settings, and decisions after failures with the parent agent.

## Operating Rules

- Use the configured SSH alias. Probe with `ssh -o BatchMode=yes -o ConnectTimeout=10` before every launch.
- Run native repositories with `export PATH="$HOME/.local/bin:$PATH"` followed by `uv run isaaclab`. When the request names a configured container, run the same command through `docker exec <container>`.
- Training is headless unless the request explicitly asks for a viewer.
- Select the requested repository path from the server profile before any repository command. Never infer a repository from the account home directory.
- Use that repository's main checkout. Never create remote worktrees, reset, clean, stash, or overwrite user-owned changes.
- Inspect `git status`, `tmux ls`, GPU processes, and the target checkpoint before a launch. Do not kill or replace a run not explicitly assigned to this request.
- Use a unique numbered tmux session named `NN-description`; check that its numeric prefix cannot be confused with an existing session.
- Record branch/commit, checkpoint, exact command, server, GPU assignment, session name, and run directory in the queue record.
- Treat a run as healthy only after the process remains live and the run directory contains TensorBoard events plus a checkpoint. Treat tmux alone as insufficient evidence.
- Always copy a completed or crashed remote training run back to the primary local checkout. Never put retrieved runs in a linked worktree.

## Launch Recipes

Fill the repository path, tmux session name, and exact `uv run isaaclab train ...` or evaluation command from the request and local server profile. Include checkpoint, seed, environment count, and overrides. First establish the SSH connection with the configured batch-mode probe; run the following commands on the remote host.

### Native repository

1. Change to the selected repository path, inspect `git status`, list tmux sessions, inspect the configured GPUs, and verify any resume checkpoint.
2. Launch the command in detached tmux. The session owns the training process.
3. Capture the session pane after startup. Verify the task, device, environment count, checkpoint/resume path, and first iteration.

```bash
tmux new-session -d -s NN-description \
  "cd /path/to/repository && export PATH=\$HOME/.local/bin:\$PATH && exec uv run isaaclab train ..."
```

### Docker container

1. Inspect the selected repository, the named container, tmux sessions, configured GPUs, and any resume checkpoint. Confirm that the selected repository path is mounted inside the selected container.
   Use `docker inspect -f '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}' <container>` to map the host repository path to its container path. If the repository is not mounted, stop and report the configuration problem; do not create a new container or alter mounts.
2. From inside the container, verify the Isaac Lab dependency path required by the repository (usually `../IsaacLab`) exists before launching. If `uv run` reports a missing editable dependency or the container has no usable Isaac Lab installation, stop and report that the container is not runnable; do not install packages or mutate the container as part of a smoke test.
3. Launch `docker exec` from the host inside detached tmux. This keeps the container training process attached to the session.
4. Capture the session pane and verify the same startup evidence as a native run. Check container/process PIDs and GPU use as well as tmux while monitoring.

```bash
tmux new-session -d -s NN-description \
  "exec docker exec container-name bash -lc 'cd /path/in/container && export PATH=\$HOME/.local/bin:\$PATH && exec uv run isaaclab train ...'"
```

Do not run `docker exec` outside tmux: stopping the SSH command can leave an orphaned training process. Do not substitute shell-unsafe request text into these commands; quote values or write the exact remote command to a temporary script first.

## Launch And Queue Workflow

1. Resolve the requested server and repository path from the profile. Resolve the requested branch and commit locally, push it when the server needs it, then fetch and fast-forward that repository's server main checkout while preserving its user-owned files.
2. Select an idle configured GPU slot. If none is available, append a pending entry to `<primary>/.local_untracked/remote-training-queue.md`; do not start a competing process.
3. Start the exact command inside detached tmux. For a requested container, use `docker exec <container>` inside tmux.
4. Capture the launch pane after startup. Confirm device, environment count, checkpoint load, requested overrides, and first metrics.
5. Poll active runs at the requested cadence. Check tmux, process PID, `nvidia-smi`, the last 40 log lines, and the latest event/checkpoint timestamps.
6. On completion or crash, copy the complete remote run directory into the primary checkout using the destination convention below. Update the queue record with state, final checkpoint, remote and local result paths, and concise failure evidence when applicable. Use a checksum when transfer integrity matters.
7. Launch the next compatible pending entry only after its assigned resources are idle.

## Retrieved Run Destination

Preserve the normal RL-library and task log hierarchy, and insert `00_SERVER_LOGS` immediately before
the run directory. If the remote run is:

```text
<remote-repository>/logs/<rl-library>/<task>/<run-folder>/
```

copy the complete folder to:

```text
<primary>/logs/<rl-library>/<task>/00_SERVER_LOGS/<run-folder>/
```

For example, an RSL-RL G1-Wuji run belongs at:

```text
<primary>/logs/rsl_rl/g1_wuji_table_direct/00_SERVER_LOGS/<run-folder>/
```

Derive `<rl-library>`, `<task>`, and `<run-folder>` from the actual remote run path rather than
reconstructing or renaming them. Copy the entire run folder, including checkpoints, TensorBoard events,
resolved configuration, and any evaluation files stored there. Create the local parent directory when
needed. If the destination already exists, first confirm that it is the same server/run before resuming
an interrupted transfer; never merge unrelated runs or silently overwrite an existing local run. Mark
the queue entry `copied` only after the complete transfer succeeds, and record the resulting local path.

## Queue Record

Use `<primary>/.local_untracked/remote-training-queue.md`. One Markdown row per run:

| state | server/repository | GPUs | session | branch@commit | checkpoint | command | remote run directory | local copy | last check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

States: `pending`, `starting`, `running`, `complete`, `crashed`, `cancelled`, `copied`.

## Status Evidence

For each update report only: state, server/session, current iteration or completion, latest checkpoint, result path, and an error line if crashed. Read summaries or log tails; never dump full logs.
