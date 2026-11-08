#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Source from the repository root in the installed serving environment.
# Use a subshell so sourcing does not change the caller's options. Explicitly
# propagate failures: Bash ignores errexit inside a conditional subshell.
if ! (
    set -euo pipefail
    reference_env="${QWEN_IMAGE_21_REFERENCE_ENV:-$PWD/tests/e2e/accuracy/qwen_image_21/.venv}"
    # Keep local reference packages first. Inherit only the serving interpreter's
    # enabled dependency roots below, rather than implicitly adding base packages.
    python -m venv "$reference_env" || exit 1
    python - "$reference_env" <<'PY' || exit 1
import subprocess
import sys
from pathlib import Path

reference_python = Path(sys.argv[1]).resolve() / "bin/python"
target = subprocess.check_output(
    [str(reference_python), "-c", "import site; print(site.getsitepackages()[0])"], text=True
).strip()
# sys.path carries the effective order, including an enabled user-site before
# system dist-packages. site.getsitepackages() alone loses that precedence.
dependency_roots = list(dict.fromkeys(
    str(Path(entry).resolve()) for entry in sys.path
    if entry and Path(entry).name in {"site-packages", "dist-packages"}
))
assert dependency_roots, "No serving dependency roots found"
Path(target, "serving_dependencies.pth").write_text(
    "\n".join(dependency_roots) + "\n", encoding="utf-8"
)
PY
    # Inherit torch/CUDA and serving dependencies. Pin reference-only packages
    # locally: this Diffusers revision needs torchao's FqnToConfig API on import.
    "$reference_env/bin/python" -m pip install --no-deps \
        'diffusers @ git+https://github.com/huggingface/diffusers.git@6256aa7666cedd47443adc8f82da9a10e110b09c' \
        'torchao==0.16.0'
); then
    return 1
fi
QWEN_IMAGE_21_REFERENCE_PYTHON=$(python - "${QWEN_IMAGE_21_REFERENCE_ENV:-$PWD/tests/e2e/accuracy/qwen_image_21/.venv}" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve() / "bin/python")
PY
) || return 1
export QWEN_IMAGE_21_REFERENCE_PYTHON
if [[ -z "${QWEN_IMAGE_21_MODEL:-}" ]]; then
    QWEN_IMAGE_21_MODEL=$(python - <<'PY'
from huggingface_hub import constants, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tests.e2e.accuracy.qwen_image_21.contract import MODEL_ID
try:
    # Resolve main once (cached refs/main when offline) for both runtimes.
    model = snapshot_download(
        MODEL_ID, ignore_patterns=["assets/*"],
        local_files_only=constants.HF_HUB_OFFLINE,
    )
except LocalEntryNotFoundError as exc:
    raise RuntimeError(
        f"No cached main snapshot for {MODEL_ID}. Pre-cache main "
        "or set QWEN_IMAGE_21_MODEL to a frozen complete local snapshot."
    ) from exc
print(model)
PY
    ) || return 1
    export QWEN_IMAGE_21_MODEL
fi
