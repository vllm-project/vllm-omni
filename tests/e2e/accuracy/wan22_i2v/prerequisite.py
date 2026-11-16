# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fail closed when a Wan2.2 accuracy comparison cannot execute."""

from __future__ import annotations

from pathlib import Path


def require_similarity_artifacts(paths: list[Path]) -> None:
    """Raise when the online video or offline baseline is absent.

    A skipped comparison is not accuracy coverage. The MI300 2-GPU nightly
    job produces these files in the same pod, in order, before this check.
    """
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return
    joined = ", ".join(str(path) for path in missing)
    raise AssertionError(f"Wan2.2 I2V accuracy comparison was not executed; missing prerequisite artifacts: {joined}")
