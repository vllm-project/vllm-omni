#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="vllm-omni"
RUN_QUALITY=false
SKIP_CLEAN=false
CREATE_VENV=false
VENV_DIR=".venv-build"
PYTHON_BIN="python"
UV_BIN="uv"

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
  --create-venv    Build inside a fresh virtual environment (default path: .venv-build)
  --venv-dir PATH  Custom directory for the virtual environment (implies --create-venv)
  --python PATH    Python executable to use (default: python)
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
    --create-venv)
      CREATE_VENV=true
      ;;
    --venv-dir)
      CREATE_VENV=true
      shift
      [[ $# -gt 0 ]] || abort "--venv-dir requires a path"
      VENV_DIR="$1"
      ;;
    --python)
      shift
      [[ $# -gt 0 ]] || abort "--python requires a path"
      PYTHON_BIN="$1"
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

HOST_PYTHON="${PYTHON_BIN}"

log "INFO" "Switching to repository root: ${REPO_ROOT}"
cd "${REPO_ROOT}" || abort "Cannot enter repository root"

[[ -f pyproject.toml ]] || abort "pyproject.toml not found, please ensure correct script location"

ensure_uv() {
  if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    log "INFO" "uv not found, installing via ${HOST_PYTHON}"
    "${HOST_PYTHON}" -m pip install --upgrade pip
    "${HOST_PYTHON}" -m pip install uv
  fi
}

ensure_uv

if [[ "${CREATE_VENV}" == "true" ]]; then
  log "INFO" "Creating fresh virtual environment at ${VENV_DIR} via uv"
  "${UV_BIN}" venv --python "${HOST_PYTHON}" --seed "${VENV_DIR}"
  PYTHON_BIN="${VENV_DIR}/bin/python"
  [[ -x "${PYTHON_BIN}" ]] || abort "Failed to locate python inside ${VENV_DIR}"
  log "INFO" "Installing build module inside virtual environment"
  "${UV_BIN}" pip install --python "${PYTHON_BIN}" build
else
  log "INFO" "Ensuring build module is available via uv pip"
  "${UV_BIN}" pip install --python "${PYTHON_BIN}" build
fi

log "INFO" "Checking build module"
if ! "${PYTHON_BIN}" -m build --version >/dev/null 2>&1; then
  abort "${PYTHON_BIN} -m build is not available, install build first"
fi

run_quality_steps() {
  log "INFO" "Running quality checks"
  "${UV_BIN}" pip install --python "${PYTHON_BIN}" -e ".[dev]"
  "${PYTHON_BIN}" -m pre_commit run --all-files
  "${PYTHON_BIN}" -m pytest tests/ -v -m "not slow"
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
"${PYTHON_BIN}" -m build

log "INFO" "Build finished, artifacts:"
ls -lh dist
