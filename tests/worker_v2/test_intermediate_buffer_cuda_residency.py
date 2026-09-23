# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA-lane checks for GPU-resident intermediate buffer values."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.parametrize("key,dtype", [(("embed", "prefill"), torch.float32), (("codes", "ref"), torch.long)])
def test_gpu_resident_values_stay_on_device_after_update(key, dtype):
    # Declared gpu-resident keys must keep tensors on device (CLA snapshot real).
    buffer = OmniIntermediateBuffer(1)
    value = torch.ones(2, 3, device="cuda", dtype=dtype)
    buffer.add_request(0, SimpleNamespace(req_id="r1", mm_features=[]))

    buffer.update(0, {key[0]: {key[1]: value}}, {key})

    stored = buffer.buffers[0][key[0]][key[1]]
    assert stored.device.type == "cuda"
    assert stored.dtype == dtype
    assert torch.equal(stored.cpu(), value.cpu())
