# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Import preload for the CPU-only collective forkserver.

This module is imported in the forkserver process, before it starts workers.
Its CUDA visibility does not alter the pytest process or CUDA parity workers.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

# Retain the expensive worker-module imports across CPU process groups.
from tests.diffusion.distributed import test_comm, test_pipeline_parallel  # noqa: E402, F401

if torch.cuda.is_initialized():
    raise RuntimeError("CPU collective forkserver must not initialize CUDA")
