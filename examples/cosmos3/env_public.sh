#!/usr/bin/env bash
# Source a runtime built from public sources; no image-site or offline tar use.
COSMOS3_PUBLIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -f "$COSMOS3_PUBLIC_ROOT/.runtime/public-runtime.env" ]]; then
  echo "Missing .runtime/public-runtime.env; follow COSMOS3_EDGE_PUBLIC_INSTALL.md" >&2
  return 2
fi
source "$COSMOS3_PUBLIC_ROOT/.runtime/public-runtime.env"
if [[ ! -f "$COSMOS3_CANN_ENV" ]]; then
  echo "Missing official CANN environment script: $COSMOS3_CANN_ENV" >&2
  return 2
fi
# Official CANN scripts may read unset shell variables.
COSMOS3_PUBLIC_RESTORE_NOUNSET=0
[[ $- != *u* ]] || COSMOS3_PUBLIC_RESTORE_NOUNSET=1
set +u
if source "$COSMOS3_CANN_ENV"; then
  COSMOS3_PUBLIC_CANN_STATUS=0
else
  COSMOS3_PUBLIC_CANN_STATUS=$?
fi
if [[ "$COSMOS3_PUBLIC_RESTORE_NOUNSET" == 1 ]]; then set -u; fi
[[ "$COSMOS3_PUBLIC_CANN_STATUS" == 0 ]] || return "$COSMOS3_PUBLIC_CANN_STATUS"
export VLLM_PYTHON="$COSMOS3_PUBLIC_ROOT/.venv-public/bin/python"
export COSMOS3_CLIENT_PYTHON="$VLLM_PYTHON"
export COSMOS3_MESA_ROOT="$COSMOS3_PUBLIC_ROOT/.runtime/mesa-23.2.1"
export LD_LIBRARY_PATH="$COSMOS3_PUBLIC_ROOT/.runtime/llvm-15.0.7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$COSMOS3_PUBLIC_ROOT/examples/cosmos3/env_native.sh" || return $?
export PATH="$COSMOS3_PUBLIC_ROOT/.venv-public/bin:$PATH"
export COSMOS3_INSTALL_PROFILE=public-source
