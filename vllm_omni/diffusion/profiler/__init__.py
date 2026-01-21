# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .torch_profiler import TorchProfiler

# Default profiler – can be changed later via config
CurrentProfiler = TorchProfiler

__all__ = ["CurrentProfiler", "TorchProfiler"]
