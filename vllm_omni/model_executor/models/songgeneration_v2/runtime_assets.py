# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime asset resolution for SongGeneration v2.

Resolves the upstream source tree path via SONGGENERATION_REPO env var,
explicit model path, or common local paths.
"""

from __future__ import annotations

import os
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

RUNTIME_REPO_ENV = "SONGGENERATION_REPO"

_COMMON_LOCAL_PATHS = (
    "/root/SongGeneration",
    "/workspace/SongGeneration",
    os.path.expanduser("~/SongGeneration"),
)


def has_songgeneration_runtime_assets(path: str | os.PathLike[str] | None) -> bool:
    """Return True when ``path`` contains the full upstream runtime layout."""
    if path is None:
        return False
    root = Path(path).expanduser()
    return (
        root.is_dir() and (root / "ckpt").is_dir() and (root / "third_party").is_dir() and (root / "codeclm").is_dir()
    )


def resolve_songgeneration_runtime_assets(
    explicit: str | None = None,
) -> str:
    """Resolve a local directory containing SongGeneration runtime assets.

    Priority:
    1. ``SONGGENERATION_REPO`` env var (local path).
    2. The explicit model path if it is a combined upstream checkout.
    3. Common local development paths.

    Raises FileNotFoundError with actionable setup instructions on failure.
    """
    env = os.environ.get(RUNTIME_REPO_ENV)
    for candidate in (env, explicit):
        if not candidate:
            continue
        if has_songgeneration_runtime_assets(candidate):
            return str(Path(candidate).expanduser())

    for candidate in _COMMON_LOCAL_PATHS:
        if has_songgeneration_runtime_assets(candidate):
            logger.info("Auto-detected SongGeneration source at %s", candidate)
            return candidate

    raise FileNotFoundError(
        "Could not locate SongGeneration runtime assets.\n"
        "\n"
        "SongGeneration Stage 1 requires the upstream source tree containing:\n"
        "  codeclm/      (Flow1dVAE decoder source code)\n"
        "  ckpt/         (Stage 1 model checkpoints)\n"
        "  third_party/  (VAE and auxiliary models)\n"
        "\n"
        "Setup:\n"
        "  git clone https://github.com/tencent-ailab/SongGeneration\n"
        f"  export {RUNTIME_REPO_ENV}=/path/to/SongGeneration\n"
        "\n"
        "Or pass the path directly via --model or model config."
    )
