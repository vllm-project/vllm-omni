# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_foreach_state_snapshots_keep_independent_storage(monkeypatch):
    buffer = OmniIntermediateBuffer(2)
    source = torch.arange(8, dtype=torch.float32)
    calls = []
    original = torch._foreach_copy_

    def copy(destinations, sources):
        calls.append(len(sources))
        return original(destinations, sources)

    monkeypatch.setattr(torch, "_foreach_copy_", copy)
    buffer.update_batch([(0, {"state": source}), (1, {"state": source})], {"state"})
    assert calls == [2]
    first, second = (record["state"] for record in buffer.buffers)
    source.zero_()
    assert torch.equal(second, torch.arange(8, dtype=torch.float32))
    first.resize_(16).fill_(99)
    assert torch.equal(second, torch.arange(8, dtype=torch.float32))


def test_batch_state_snapshots_match_scalar_updates_for_nested_and_tuple_keys():
    scalar, batch = OmniIntermediateBuffer(2), OmniIntermediateBuffer(2)
    entries = [(0, {"state": {"a": torch.arange(4), "metadata": [1, 2]}}), (1, {("state", "a"): torch.arange(6)})]
    keys = {("state", "a")}
    for index, update in entries:
        scalar.update(index, update, keys)
    batch.update_batch(entries, keys)
    for i in range(2):
        assert torch.equal(scalar.buffers[i]["state"]["a"], batch.buffers[i]["state"]["a"])
    assert batch.buffers[0]["state"]["metadata"] == [1, 2]
