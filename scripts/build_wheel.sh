#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="vllm-omni"
RUN_QUALITY=false
SKIP_CLEAN=false

log() {
  local level="$1"
  shift
  printf '[%s] %s\n' "${level}" "$*"
}

abort() {
  log "ERROR" "$*"
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --run-quality    Run pre-commit, install dev deps, and pytest before building
  --skip-clean     Skip removing previous build artifacts
  -h, --help       Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-quality)
      RUN_QUALITY=true
      ;;
    --skip-clean)
      SKIP_CLEAN=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      abort "Unknown option: $1"
      ;;
  esac
  shift
done

log "INFO" "Switching to repository root: ${REPO_ROOT}"
cd "${REPO_ROOT}" || abort "Cannot enter repository root"

[[ -f pyproject.toml ]] || abort "pyproject.toml not found, please ensure correct script location"

log "INFO" "Checking build module"
if ! python -m build --version >/dev/null 2>&1; then
  abort "python -m build is not available, run 'pip install build' first"
fi

run_quality_steps() {
  log "INFO" "Running quality checks"
  pre-commit run --all-files
  pip install -e ".[dev]"
  pytest tests/ -v -m "not slow"
}

if [[ "${RUN_QUALITY}" == "true" ]]; then
  run_quality_steps
else
  log "INFO" "Quality steps available via --run-quality"
  log "INFO" "  - pre-commit run --all-files"
  log "INFO" "  - pip install -e '.[dev]'"
  log "INFO" "  - pytest tests/ -v -m \"not slow\""
fi

cleanup_artifacts() {
  log "INFO" "Cleaning previous build artifacts"
  rm -rf build dist "${PROJECT_NAME}.egg-info" "${PROJECT_NAME//-/_}.egg-info"
}

if [[ "${SKIP_CLEAN}" == "true" ]]; then
  log "INFO" "Skipping cleanup as requested"
else
  cleanup_artifacts
fi

log "INFO" "Building source and wheel distributions"
python -m build

log "INFO" "Build finished, artifacts:"
ls -lh dist
