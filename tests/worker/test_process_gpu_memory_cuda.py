# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU allocation coverage for process-scoped memory accounting."""

import pytest

from tests.helpers.mark import hardware_test

pytestmark = pytest.mark.core_model


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_returns_memory_for_current_process():
    import torch

    from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

    if not torch.accelerator.is_available():
        pytest.skip("CUDA not available")

    tensor = torch.zeros(1000, 1000, device="cuda:0")
    memory = get_process_gpu_memory(0)
    assert memory >= 0
    del tensor
    torch.accelerator.empty_cache()
