# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared runtime fixtures for diffusion offloader tests."""

import gc

import pytest
import torch.distributed as dist

from tests.helpers.runtime import get_distributed_init_method
from vllm_omni.platforms import current_omni_platform


@pytest.fixture(scope="module")
def dist_group():
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=get_distributed_init_method())
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
            current_omni_platform.synchronize()
