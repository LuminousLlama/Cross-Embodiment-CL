#!/usr/bin/env bash
# One G1-Wuji training job per machine, in tmux, pinned to a commit.
#
#   scripts/exp.sh launch <local|server> <name> <commit> [isaaclab train args...]
#   scripts/exp.sh status <local|server> <name>
#   scripts/exp.sh stop   <local|server> <name>
#
# Server jobs check out <commit> in the yga100 repo (fetched from origin, so push first).
# Local jobs run from a detached worktree at ../CL-run-local, so edits to this checkout
# cannot leak into a running job; it borrows this checkout's .venv with its own src first
# on PYTHONPATH.  A commit that changes dependencies needs a manual `uv sync`.
# Output goes to logs/exp/<name>.log in the job's checkout and ends with EXP_EXIT=<code>.
set -euo pipefail

usage="usage: exp.sh launch|status|stop <local|server> <name> [commit] [train args...]"
cmd=${1:?$usage} lane=${2:?$usage} name=${3:?$usage}
shift 3
repo=$(cd "$(dirname "$0")/.." && pwd)

case $lane in
  local) dir=$(dirname "$repo")/CL-run-local venv=$repo/.venv ;;
  server) dir='~/cross-embodiment/Cross-Embodiment-CL' venv='~/cross-embodiment/Cross-Embodiment-CL/.venv' ;;
  *) echo "$usage" >&2; exit 2 ;;
esac

# Run the script on stdin with the given arguments on this lane's machine.
run() {
  if [ "$lane" = server ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 yga100 "bash -s -- $(printf '%q ' "$@")"
  else
    bash -s -- "$@"
  fi
}

case $cmd in
launch)
  commit=${1:?$usage}
  shift
  if [ "$lane" = local ] && [ ! -d "$dir" ]; then
    git -C "$repo" worktree add -q --detach "$dir" "$commit"
  fi
  run "$dir" "$venv" "$name" "$commit" "$@" <<'EOF'
dir=${1/#\~/$HOME} venv=${2/#\~/$HOME} name=$3 commit=$4
shift 4
cd "$dir"
if tmux ls -F '#S' 2>/dev/null | grep -q '^exp-'; then
  echo "REFUSED: a job is already running here: $(tmux ls -F '#S' | grep '^exp-' | tr '\n' ' ')"; exit 1
fi
if [ -e "logs/exp/$name.log" ]; then echo "REFUSED: logs/exp/$name.log exists, pick a new name"; exit 1; fi
git fetch -q origin || echo "WARN: git fetch failed, using local refs"
git checkout -q --detach "$commit" || { echo "REFUSED: checkout of $commit failed"; exit 1; }
mkdir -p logs/exp
args=$(printf '%q ' "$@")
cat > "logs/exp/$name.sh" <<RUNNER
cd "$dir"
export VIRTUAL_ENV="$venv" PATH="$venv/bin:\$HOME/.local/bin:\$PATH" PYTHONPATH="$dir/src"
{
  echo "EXP name=$name commit=\$(git rev-parse --short HEAD) host=\$(hostname) start=\$(date -Is) args=$args"
  python -c 'import Cross_Embodiment_CL as m; print("EXP package", m.__file__)'
  isaaclab train --rl_library rsl_rl --task CrossEmbodimentCl-G1-Wuji-Table-Direct --run_name $name presets=train $args
  echo "EXP_EXIT=\$?"
} 2>&1 | tee -a "logs/exp/$name.log"
RUNNER
tmux new-session -d -s "exp-$name" -c "$dir" "bash $dir/logs/exp/$name.sh"
echo "LAUNCHED exp-$name at $(git rev-parse --short HEAD) in $dir"
EOF
  ;;
status)
  run "$dir" "$name" <<'EOF'
dir=${1/#\~/$HOME} name=$2
log=$dir/logs/exp/$name.log
[ -f "$log" ] || { echo "$name NO_LOG"; exit 0; }
it=$(grep -a -o 'Learning iteration [0-9]*/[0-9]*' "$log" | tail -n 1 | cut -d' ' -f3)
rew=$(grep -a -o 'Mean reward: *[-0-9.naif]*' "$log" | tail -n 1 | awk '{print $NF}')
code=$(grep -a -o 'EXP_EXIT=[0-9]*' "$log" | tail -n 1 | cut -d= -f2)
age=$(( $(date +%s) - $(stat -c %Y "$log") ))
if tmux has-session -t "exp-$name" 2>/dev/null; then
  state=RUNNING; [ "$age" -lt 900 ] || state=STALLED
elif [ "$code" = 0 ]; then
  state=FINISHED
else
  state=CRASHED
fi
case $rew in *nan*|*inf*) state=NAN_$state ;; esac
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -n 1)
run=$(ls -td "$dir"/logs/rsl_rl/*/*_"$name" 2>/dev/null | head -n 1)
echo "$name $state it=${it:-0} reward=${rew:--} log_age=${age}s tracebacks=$(grep -ac Traceback "$log") gpu=[$gpu] run=${run:--}"
case $state in
  RUNNING|FINISHED) ;;
  *) echo "--- log tail"; tail -n 60 "$log" | sed 's/\x1b\[[0-9;]*m//g' | grep -v '^\s*$' | tail -n 25 ;;
esac
EOF
  ;;
stop)
  run "$name" <<'EOF'
name=$1
tmux send-keys -t "exp-$name" C-c 2>/dev/null || true
for _ in $(seq 30); do tmux has-session -t "exp-$name" 2>/dev/null || break; sleep 1; done
tmux kill-session -t "exp-$name" 2>/dev/null || true
pattern="--run_name $name presets="
pkill -f -- "$pattern" || true
sleep 10
pkill -9 -f -- "$pattern" || true
echo "STOPPED exp-$name"
EOF
  ;;
*) echo "$usage" >&2; exit 2 ;;
esac
