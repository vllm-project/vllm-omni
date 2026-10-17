# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.worker import async_omni_output
from vllm_omni.worker.async_omni_output import AsyncOmniOutputRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(async_omni_output, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    runner = AsyncOmniOutputRunnerMixin()
    runner._prefix_cache_materialize = Mock(return_value=(None, None))
    return runner


@pytest.mark.parametrize("needs_hidden", [False, True])
def test_prefix_cache_reuses_staged_hidden_and_snapshot_req_ids(runner, monkeypatch, needs_hidden):
    merged_hidden = {"r1": torch.ones(1, 4)}
    merged_mm: dict[str, object] = {"codes": {}}
    runner._prefix_cache_materialize.return_value = (merged_hidden, merged_mm)
    runner.input_batch = SimpleNamespace(req_ids=["next-step-request"])
    copy = Mock(side_effect=AssertionError("unexpected hidden-state copy"))
    monkeypatch.setattr(async_omni_output, "_to_cpu_contiguous", copy)
    staged = torch.ones(2, 4) if needs_hidden else None
    req_ids = ["r1"]

    cpu, combined, mm = runner._prepare_prefix_cache_pooler_payload_sources(
        staged_hidden_states_cpu=staged,
        needs_scheduled_hidden_payload=needs_hidden,
        req_ids=req_ids,
        step_id=7,
    )

    assert cpu is staged
    assert combined is merged_hidden
    assert mm is merged_mm
    runner._prefix_cache_materialize.assert_called_once_with(7, ["r1"])
    req_ids.append("later")
    assert runner._prefix_cache_materialize.call_args.args[1] == ["r1"]
    copy.assert_not_called()


@pytest.mark.parametrize("needs_hidden", [False, True])
def test_prefix_cache_without_step_serves_staged_slice(runner, needs_hidden):
    staged = torch.ones(2, 4)
    cpu, hidden, mm = runner._prepare_prefix_cache_pooler_payload_sources(
        staged_hidden_states_cpu=staged,
        needs_scheduled_hidden_payload=needs_hidden,
        req_ids=["r1"],
        step_id=None,
    )
    assert cpu is staged
    assert hidden is None and mm is None
    runner._prefix_cache_materialize.assert_not_called()


def test_prefix_cache_requires_staged_hidden_before_materializing(runner):
    with pytest.raises(RuntimeError, match="requires staged CPU hidden states"):
        runner._prepare_prefix_cache_pooler_payload_sources(
            staged_hidden_states_cpu=None,
            needs_scheduled_hidden_payload=True,
            req_ids=["r1"],
            step_id=7,
        )
    runner._prefix_cache_materialize.assert_not_called()


def _staging_runner(device: str) -> AsyncOmniOutputRunnerMixin:
    runner = AsyncOmniOutputRunnerMixin()
    runner._pooler_payload_include_hidden_flag = True  # type: ignore[attr-defined]
    runner.device = torch.device(device)
    return runner


def test_sync_output_moves_staged_device_slice_to_host():
    hidden = torch.arange(8.0).reshape(4, 2)
    snapshot = _staging_runner("cpu")._snapshot_omni_output_tensors_for_async_output(
        use_async_omni_output=False,
        hidden_states=hidden,
        staged_hidden_states=hidden[:3],
        multimodal_outputs={},
    )
    assert snapshot.staged_hidden_states_cpu is not None
    torch.testing.assert_close(snapshot.staged_hidden_states_cpu, hidden[:3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator stream")
def test_async_output_slices_staged_rows_from_the_hidden_host_copy(monkeypatch):
    hidden = torch.arange(8.0, device="cuda").reshape(4, 2)
    copy = Mock(side_effect=AssertionError("staged rows must reuse the hidden-state host copy"))
    monkeypatch.setattr(async_omni_output, "_to_cpu_contiguous", copy)
    snapshot = _staging_runner("cuda")._snapshot_omni_output_tensors_for_async_output(
        use_async_omni_output=True,
        hidden_states=hidden,
        staged_hidden_states=hidden[:3],
        multimodal_outputs={},
    )
    hidden.fill_(-1)
    assert snapshot.async_payload is not None
    snapshot.async_payload.wait()
    staged = snapshot.staged_hidden_states_cpu
    assert staged is not None and staged.device.type == "cpu" and staged.is_contiguous()
    assert staged.data_ptr() == snapshot.hidden_states.data_ptr()
    torch.testing.assert_close(staged, torch.arange(6.0).reshape(3, 2))
    copy.assert_not_called()


def test_accel_payload_nbytes_ignores_host_tensors():
    payload = {"hidden_states": torch.zeros(4, 2), "multimodal_outputs": {"codes": [torch.zeros(3)], "n": 1}}
    assert async_omni_output._accel_payload_nbytes(payload) == (None, 0)


def test_async_output_skips_snapshot_without_payload_consumers(monkeypatch):
    snapshot_async = Mock(side_effect=AssertionError("a step without payload consumers must not copy"))
    monkeypatch.setattr(async_omni_output, "_snapshot_tensor_payload_to_cpu_async", snapshot_async)
    hidden = torch.arange(8.0).reshape(4, 2)
    snapshot = _staging_runner("cpu")._snapshot_omni_output_tensors_for_async_output(
        use_async_omni_output=True,
        hidden_states=hidden,
        staged_hidden_states=hidden[:3],
        multimodal_outputs={"codes": torch.zeros(4)},
        skip_payload=True,
    )
    assert snapshot.async_payload is None
    assert snapshot.hidden_states.shape[0] == 0
    assert snapshot.staged_hidden_states_cpu is None and snapshot.multimodal_outputs is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator stream")
def test_async_payload_snapshot_survives_source_overwrite():
    hidden = torch.arange(8.0, device="cuda").reshape(4, 2)
    host_codes = torch.tensor([1, 2])
    snapshot = async_omni_output._snapshot_tensor_payload_to_cpu_async(
        {"hidden_states": hidden, "multimodal_outputs": {"codes": [host_codes]}},
        copy_stream=torch.cuda.Stream(),
        pin_memory=True,
    )
    hidden.fill_(-1)
    host_codes.fill_(0)
    snapshot.wait()
    torch.testing.assert_close(snapshot.payload["hidden_states"], torch.arange(8.0).reshape(4, 2))
    assert snapshot.payload["multimodal_outputs"]["codes"][0].tolist() == [1, 2]


@pytest.mark.parametrize("include_hidden", [False, True])
def test_async_staged_cpu_rows_survive_buffer_reuse(monkeypatch, include_hidden):
    runner = _staging_runner("cpu")
    runner._pooler_payload_include_hidden_flag = include_hidden
    runner._get_or_create_omni_payload_copy_stream = lambda: None
    monkeypatch.setattr(async_omni_output, "is_pin_memory_available", lambda: False)
    hidden = torch.arange(8.0).reshape(4, 2)
    expected = hidden[:3].clone()
    snapshot = runner._snapshot_omni_output_tensors_for_async_output(
        use_async_omni_output=True,
        hidden_states=hidden,
        staged_hidden_states=hidden[:3],
        multimodal_outputs={},
    )
    hidden.fill_(-1)
    snapshot.async_payload.wait()
    torch.testing.assert_close(snapshot.staged_hidden_states_cpu, expected)
    if include_hidden:
        assert snapshot.staged_hidden_states_cpu.data_ptr() == snapshot.hidden_states.data_ptr()


@pytest.mark.parametrize("device_type,nbytes,inline", [("npu", 4, True), ("npu", 8 << 20, False), ("cuda", 4, False)])
def test_inline_payload_copy_is_only_for_small_npu_outputs(monkeypatch, device_type, nbytes, inline):
    accel = Mock()
    accel.stream.side_effect = lambda stream: nullcontext()
    stream = Mock()
    monkeypatch.setattr(async_omni_output, "_accel_payload_nbytes", lambda value: (device_type, nbytes))
    monkeypatch.setattr(async_omni_output, "_accel_module", lambda device: accel)
    source = Mock(device=SimpleNamespace(type=device_type))
    payload = {"codes": torch.tensor([1, 2])}

    def clone(value, sources):
        sources.append(source)
        return value

    clone_payload = Mock(side_effect=clone)
    monkeypatch.setattr(async_omni_output, "_clone_accel_tensor_payload", clone_payload)
    snapshot = async_omni_output._snapshot_tensor_payload_to_cpu_async(payload, copy_stream=stream, pin_memory=False)
    assert clone_payload.call_count == (0 if inline else 1)
    if inline:
        payload["codes"].fill_(0)
        assert snapshot.payload["codes"].tolist() == [1, 2]
        stream.wait_stream.assert_not_called()
    else:
        stream.wait_stream.assert_called_once_with(accel.current_stream.return_value)
        source.record_stream.assert_called_once_with(stream)
    snapshot.wait()
    accel.Event.return_value.synchronize.assert_called_once()
