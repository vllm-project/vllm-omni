#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility import for the packaged LingBot-World benchmark."""

import sys
from pathlib import Path

# Keep the historical direct-script command usable from a source checkout.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm_omni.benchmarks.lingbot_world import realtime as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

# Old imports and patches share the canonical module and its global state.
sys.modules[__name__] = _implementation
