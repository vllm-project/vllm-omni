#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve the legacy stability CLI through the shared serving benchmark."""

from vllm_omni.benchmarks.realtime_video.compat import main

if __name__ == "__main__":
    raise SystemExit(main())
