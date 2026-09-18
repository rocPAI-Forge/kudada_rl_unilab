#!/usr/bin/env bash
# One FastSAC job per gait, each pinned to a different MI210 via HIP_VISIBLE_DEVICES.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export UNILAB_EXTRA_REGISTRY_PACKAGES="${UNILAB_EXTRA_REGISTRY_PACKAGES:-roscon_rl_unilab.tasks}"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/runs"

start_job() {
  local gpu="$1" task="$2" seed="$3"
  local log="${ROOT}/runs/train_${task}_gpu${gpu}.log"
  echo "GPU ${gpu}: ${task} seed=${seed} log=${log}"
  nohup env HIP_VISIBLE_DEVICES="${gpu}" \
    UNILAB_EXTRA_REGISTRY_PACKAGES="${UNILAB_EXTRA_REGISTRY_PACKAGES}" \
    "${PY}" -c "from roscon_rl_unilab.cli import train_main; import sys; raise SystemExit(train_main(sys.argv[1:]))" \
    --task "${task}" --sim mujoco \
    "algo.seed=${seed}" \
    "training.no_play=true" \
    "training.log_root=runs" \
    >"${log}" 2>&1 &
  echo $! | tee "${ROOT}/runs/train_${task}_gpu${gpu}.pid"
}

start_job 0 crouch-walk 1
start_job 1 lateral-walk 2
start_job 2 march-recover 3
echo "started crouch/lateral/march on GPUs 0/1/2 (GPU 3 unused)"
