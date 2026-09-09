# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.worker_v2.model_states.omni_model_state import _stage_cpu_indices


@pytest.mark.core_model
@pytest.mark.cpu
def test_cpu_indices_keep_values_and_dtype(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_ASYNC_METADATA", "1")
    assert _stage_cpu_indices([4, 1, 9], device=torch.device("cpu"), dtype=torch.long).tolist() == [4, 1, 9]


@pytest.mark.core_model
@pytest.mark.cuda
@pytest.mark.gpu
def test_inflight_index_transfers_are_isolated(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_ASYNC_METADATA", "1")
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    outputs = []
    with torch.cuda.stream(stream):
        for i in range(100):
            values = [i, 99 - i, 7]
            outputs.append(_stage_cpu_indices(values, device=device, dtype=torch.long))
            values[:] = [-1, -1, -1]
    stream.synchronize()
    assert torch.stack(outputs).cpu().tolist() == [[i, 99 - i, 7] for i in range(100)]
