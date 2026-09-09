#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

# Omni's CUDA workers use the V1 model runner. vLLM 0.28 may independently
# select V2 in the scheduler, which omits token history needed for V1 async
# batch readmission. Resolve the mode before any Python config is constructed.
if [[ "${VLLM_USE_V2_MODEL_RUNNER:-0}" != "0" ]]; then
    printf '%s\n' "This full-duplex runtime requires Model Runner V1; unset VLLM_USE_V2_MODEL_RUNNER or set it to 0." >&2
    exit 2
fi
export VLLM_USE_V2_MODEL_RUNNER=0

duplex_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
duplex_python="${duplex_repo_dir}/.venv/bin/python"
if [[ ! -x "$duplex_python" ]]; then
    printf '%s\n' "Missing project .venv. Install requirements/fullduplex-test-cu130.txt in a Linux CUDA environment." >&2
    exit 2
fi
export VIRTUAL_ENV="${duplex_repo_dir}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
cd "$duplex_repo_dir"

"$duplex_python" - "$duplex_repo_dir" <<'PY'
import importlib.metadata as metadata
import importlib.util
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1]).resolve()
prefix = pathlib.Path(sys.prefix).resolve()
assert prefix == root / ".venv", f"Wrong Python environment: {prefix}"
assert metadata.version("vllm") == "0.28.0", "This runner requires vLLM 0.28.0 exactly"
assert metadata.version("transformers") == "5.14.1", "Use the validated transformers 5.14.1"
for package, owner in (("vllm", prefix), ("vllm_omni", root / "vllm_omni")):
    spec = importlib.util.find_spec(package)
    assert spec and spec.origin and pathlib.Path(spec.origin).resolve().is_relative_to(owner), (
        f"{package} is shadowed by another checkout/environment: {spec}"
    )
ninja = shutil.which("ninja")
assert ninja and pathlib.Path(ninja).resolve().is_relative_to(prefix), "Project ninja is missing from PATH"
import cv2
gui = next(line.strip() for line in cv2.getBuildInformation().splitlines() if line.strip().startswith("GUI:"))
assert gui.split(":", 1)[1].strip() == "NONE", "Reinstall opencv-python-headless last; GUI cv2 replaced it"
PY

exec "$duplex_python" "$@"
