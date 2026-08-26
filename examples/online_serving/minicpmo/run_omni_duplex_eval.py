#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate, evaluate, or summarize Omni-DuplexEval artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_omni.benchmarks.duplex.omni_duplex_eval_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
