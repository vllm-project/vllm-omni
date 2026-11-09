#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Legacy flags over ``vllm bench serve --omni --backend realtime-video``."""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm_omni.benchmarks.realtime_video.compat import main

if __name__ == "__main__":
    raise SystemExit(main())
