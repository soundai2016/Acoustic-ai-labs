#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-}"
if [[ -z "$CANN_ENV_SCRIPT" ]]; then
  for candidate in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/cann-8.5.1/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
    if [[ -r "$candidate" ]]; then CANN_ENV_SCRIPT="$candidate"; break; fi
  done
fi
if [[ -n "$CANN_ENV_SCRIPT" && ! -r "$CANN_ENV_SCRIPT" ]]; then
  echo "CANN environment script not readable: $CANN_ENV_SCRIPT" >&2
  exit 2
fi
if [[ -n "$CANN_ENV_SCRIPT" ]]; then
  set +u
  source "$CANN_ENV_SCRIPT"
  set -u
fi
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec "${PYTHON_BIN:-python}" "$ROOT/npu/run.py" "$@"
