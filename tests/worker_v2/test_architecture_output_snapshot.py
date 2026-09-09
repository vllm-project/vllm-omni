# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner
from vllm_omni.worker_v2.output_snapshot import pack_output_snapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_packed_snapshot_preserves_nested_values_and_consumer_ownership():
    source = torch.arange(12, dtype=torch.int64).view(3, 4).t()
    payload = {"noncontiguous": source, "nested": [torch.tensor(7), (torch.tensor([1.5]),)], "meta": "ok"}
    slot: dict[tuple[Any, ...], torch.Tensor] = {}
    snapshot = pack_output_snapshot(payload, slot, max_buckets=4)
    source.fill_(99)
    copies = []

    def copy(tensor):
        copies.append(tensor.numel())
        return tensor.clone()

    host = snapshot.copy_to_cpu(copy)
    assert len(copies) == 2  # int64 and float32, independent of tensor count
    assert host["noncontiguous"].tolist() == torch.arange(12).view(3, 4).t().tolist()
    assert host["nested"][0].shape == torch.Size([])
    assert host["nested"][0].item() == 7
    assert isinstance(host["nested"][1], tuple)
    assert host["meta"] == "ok"
    pack_output_snapshot(payload, slot, max_buckets=4)
    assert host["noncontiguous"][0, 0].item() == 0


def test_packed_snapshot_distinct_slots_and_bounded_shape_cache():
    slot: dict[tuple[Any, ...], torch.Tensor] = {}
    first = pack_output_snapshot(
        {"x": torch.tensor([1]), "empty": torch.empty(0, dtype=torch.int64)}, slot, max_buckets=1
    )
    second = pack_output_snapshot({"x": torch.tensor([2, 3]), "y": torch.tensor(4)}, slot, max_buckets=1)
    assert len(slot) == 1
    assert first.copy_to_cpu(torch.clone)["x"].tolist() == [1]
    assert second.copy_to_cpu(torch.clone)["x"].tolist() == [2, 3]
    assert first.copy_to_cpu(torch.clone)["empty"].shape == (0,)


def test_empty_admission_reuses_standard_sampler_state_only():
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace()
    runner.sampler = Sampler.__new__(Sampler)
    output = SchedulerOutput.make_empty()
    with patch.object(GPUModelRunner, "add_requests") as parent:
        runner.add_requests(output)
        parent.assert_not_called()
        # Unknown sampler implementations retain their own update semantics.
        runner.sampler = SimpleNamespace()
        runner.add_requests(output)
        parent.assert_called_once_with(output)


def test_admission_with_new_request_still_submits_sampler_state():
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace()
    runner.sampler = Sampler.__new__(Sampler)
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [SimpleNamespace()]
    with patch.object(GPUModelRunner, "add_requests") as parent:
        runner.add_requests(output)
        parent.assert_called_once_with(output)


def test_request_index_tracks_slot_reuse_and_duplicate_cleanup():
    buffer = OmniIntermediateBuffer(2)

    def add(slot, req_id):
        buffer.add_request(slot, SimpleNamespace(req_id=req_id, mm_features=[]))

    add(0, "old")
    add(0, "new")
    assert buffer.req_id_to_index == {"new": 0}
    buffer.remove_request(0)
    buffer.remove_request(0)
    assert buffer.req_id_to_index == {}
    add(1, "new")
    assert buffer.req_id_to_index == {"new": 1}
