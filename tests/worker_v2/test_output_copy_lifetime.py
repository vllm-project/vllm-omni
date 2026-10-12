# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device slab leases survive partial copies and unsafe device failures."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.worker_v2.test_omni_ar_model_runner import _async_output, _FakeEvent, _FakeStream
from vllm_omni.model_executor.output_snapshot import OutputCopyLifetimeError, pack_output_snapshot
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _snapshot():
    result = pack_output_snapshot({"tick": torch.tensor([7]), "value": torch.tensor([1.5])}, {}, max_buckets=2)
    assert result is not None
    return result


def _batch():
    return SimpleNamespace(
        query_start_loc_np=np.array([0, 1]),
        num_scheduled_tokens=[1],
        num_reqs=1,
        num_tokens_after_padding=1,
    )


def test_owned_snapshot_registers_once_and_copy_callback_is_one_time():
    snapshot = _snapshot()
    released: list[object] = []
    snapshot.set_copy_completion_callback(released.append)
    with pytest.raises(RuntimeError, match="already registered"):
        snapshot.set_copy_completion_callback(released.append)
    snapshot.mark_copy_started()
    assert snapshot.copy_started
    event = _FakeEvent()
    snapshot.bind_copy_event(event)
    assert released == [event]
    with pytest.raises(RuntimeError, match="already bound"):
        snapshot.bind_copy_event(event)


def test_snapshot_without_model_lease_keeps_existing_copy_behavior():
    snapshot = _snapshot()
    snapshot.mark_copy_started()
    snapshot.bind_copy_event(_FakeEvent())
    snapshot.bind_copy_event(_FakeEvent())
    assert not snapshot.copy_started
    assert snapshot.copy_to_cpu(lambda value: value.clone())["tick"].tolist() == [7]


def test_snapshot_rejects_noncallable_completion_hook():
    with pytest.raises(TypeError, match="callable"):
        _snapshot().set_copy_completion_callback(None)


def test_constructor_publishes_event_after_copy_and_cpu_data_is_independent(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    snapshot = _snapshot()
    released: list[object] = []
    snapshot.set_copy_completion_callback(released.append)
    output = _async_output(
        multimodal_outputs=snapshot,
        input_batch=_batch(),
        finalize_multimodal=lambda value, _count: value,
    )
    assert snapshot.copy_started and released == [output.copy_event]
    # The producer may overwrite the slab after its copy completion. CPU
    # materialization continues to own the original, separate storage.
    snapshot["tick"].fill_(99)
    result = output.get_output()
    assert result.inter_stage_outputs[0]["tick"].tolist() == [7]


def test_partial_d2h_constructor_failure_still_fences_queued_reads(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    snapshot = _snapshot()
    released: list[object] = []
    copied = []
    snapshot.set_copy_completion_callback(released.append)

    def partial_copy(copy_tensor):
        copied.append(copy_tensor(snapshot._groups[0][1]))
        raise MemoryError("second host slab allocation")

    monkeypatch.setattr(snapshot, "copy_to_cpu", partial_copy)
    event = _FakeEvent()
    with pytest.raises(MemoryError, match="second host slab"):
        _async_output(
            multimodal_outputs=snapshot,
            input_batch=_batch(),
            finalize_multimodal=lambda value, _count: value,
            copy_event=event,
        )
    assert copied and released == [event] and snapshot.copy_started


@pytest.mark.parametrize("failure", ["dependency", "event"])
def test_device_failure_leaves_lease_unavailable(monkeypatch, failure):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    snapshot = _snapshot()
    released: list[object] = []
    snapshot.set_copy_completion_callback(released.append)

    class FaultStream(_FakeStream):
        def wait_stream(self, stream):
            if failure == "dependency":
                raise RuntimeError("device fault")

    class FaultEvent(_FakeEvent):
        def record(self, stream):
            if failure == "event":
                raise RuntimeError("device fault")

    with pytest.raises(OutputCopyLifetimeError):
        _async_output(
            multimodal_outputs=snapshot,
            input_batch=_batch(),
            finalize_multimodal=lambda value, _count: value,
            copy_stream=FaultStream(),
            copy_event=FaultEvent(),
        )
    assert snapshot.copy_started and released == []


def test_extra_producer_dependency_is_observed_before_any_copy_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    snapshot = _snapshot()
    released: list[object] = []
    waited = []
    snapshot.set_copy_completion_callback(released.append)
    producer_event = object()
    snapshot.producer_event = producer_event
    extra_ready = object()

    class ObservedStream(_FakeStream):
        def wait_event(self, event):
            waited.append(event)

    class FailedSamplerMasks:
        def to_cpu_nonblocking(self):
            assert waited == [producer_event, extra_ready]
            raise MemoryError("host masks allocation")

    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    sampler = SamplerOutput(torch.tensor([[1]]), None, None, torch.tensor([1]), torch.tensor([0]))
    sampler.sampling_mask_tensors = FailedSamplerMasks()
    with pytest.raises(MemoryError, match="host masks"):
        _async_output(
            sampler_output=sampler,
            extra_multimodal_outputs=(snapshot, extra_ready),
            copy_stream=ObservedStream(),
        )
    assert len(released) == 1


def test_unsafe_copy_failure_is_an_engine_fault_not_request_recovery():
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.execute_model_state = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["owner"]))
    runner.eplb = MagicMock()
    runner._sample_tokens = MagicMock(side_effect=OutputCopyLifetimeError("no completion fence"))
    runner._abort_failed_transaction = MagicMock()
    with pytest.raises(OutputCopyLifetimeError, match="completion fence"):
        runner.sample_tokens(None)
    runner._abort_failed_transaction.assert_not_called()
