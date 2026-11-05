# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility import for the packaged LingBot-World benchmark."""

import sys

from vllm_omni.benchmarks.lingbot_world import workload as _implementation

# Old imports and patches share the canonical module and its global state.
sys.modules[__name__] = _implementation
