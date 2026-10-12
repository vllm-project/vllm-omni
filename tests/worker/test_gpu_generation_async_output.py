# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.worker.test_gpu_generation_model_runner import _make_runner
from vllm_omni.worker.gpu_generation_model_runner import _AsyncGenerationOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.parametrize("kind", ["tensor", "list", "mapping"])
def test_async_generation_owns_output_and_request_mapping(kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.arange(64, device="cuda", dtype=torch.float32).reshape(2, 32)
    expected = source.cpu()
    raw = source if kind == "tensor" else list(source.unbind())
    if kind == "mapping":
        raw = {"audio": list(source.unbind())}
    runner = _make_runner(raw, num_reqs=2)
    runner.device = torch.device("cuda")
    runner.use_async_scheduling = True
    runner.async_output_copy_stream = torch.cuda.Stream()
    # Publication must not wait for GPU work on the calling thread.
    with (
        patch.object(torch.cuda.Event, "synchronize", side_effect=AssertionError("host wait")),
        patch.object(torch.cuda.Stream, "synchronize", side_effect=AssertionError("host wait")),
    ):
        output = runner.sample_tokens()
    assert isinstance(output, _AsyncGenerationOutput)
    source.fill_(-1)
    runner.input_batch.req_ids.clear()
    runner.input_batch.req_id_to_index.clear()
    resolved = output.get_output()
    assert resolved.req_ids == ["req-1", "req-2"]
    assert resolved.req_id_to_index == {"req-1": 0, "req-2": 1}
    key = "audio" if kind == "mapping" else "model_outputs"
    for i, payload in enumerate(resolved.multimodal_outputs):
        torch.testing.assert_close(payload[key], expected[i])
    assert output.get_output() is resolved


def test_async_generation_preserves_none_rows():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = _make_runner([None, torch.ones(8, device="cuda")], num_reqs=2)
    runner.device = torch.device("cuda")
    runner.use_async_scheduling = True
    runner.async_output_copy_stream = torch.cuda.Stream()
    output = runner.sample_tokens().get_output()
    assert output.multimodal_outputs[0]["model_outputs"] is None
    torch.testing.assert_close(output.multimodal_outputs[1]["model_outputs"], torch.ones(8))


@pytest.mark.parametrize("owns_storage", [False, True])
def test_full_payload_accumulation_keeps_synchronous_host_copy(owns_storage):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = _make_runner(torch.ones((1, 8), device="cuda"))
    runner.device = torch.device("cuda")
    runner.use_async_scheduling = True
    runner.async_output_copy_stream = torch.cuda.Stream()
    runner.model = SimpleNamespace(owns_generation_output_storage=owns_storage)
    runner.requests = {"req-1": object()}
    seen = []
    runner._should_accumulate_full_payload_output = lambda: True
    runner.accumulate_full_payload_output = lambda rid, payload, state: seen.append(payload["model_outputs"].clone())
    with (
        patch("vllm_omni.worker.gpu_generation_model_runner.AsyncGPUModelRunnerOutput") as wrapper,
        patch(
            "vllm_omni.worker.gpu_generation_model_runner._snapshot_tensor_payload_to_cpu_async",
            side_effect=AssertionError("unexpected deferred copy"),
        ),
    ):
        runner.sample_tokens()
    wrapper.assert_called_once()
    torch.testing.assert_close(seen[0], torch.ones(8))


@pytest.mark.parametrize("kind", ["tensor", "list", "mapping"])
def test_owned_generation_output_skips_device_snapshot_and_retains_sources(kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.empty((2, 32), device="cuda")
    # Production must be ordered before copying, even if it is still pending.
    torch.cuda._sleep(5_000_000)
    source.fill_(7)
    raw = source if kind == "tensor" else list(source.unbind())
    if kind == "mapping":
        raw = {"audio": list(source.unbind()), "sr": [torch.tensor(24000), torch.tensor(24000)]}
    runner = _make_runner(raw, num_reqs=2)
    runner.model = SimpleNamespace(owns_generation_output_storage=True)
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.device = torch.device("cuda")
    runner.use_async_scheduling = True
    runner.async_output_copy_stream = torch.cuda.Stream()
    with (
        patch(
            "vllm_omni.worker.gpu_generation_model_runner._snapshot_tensor_payload_to_cpu_async",
            side_effect=AssertionError("unexpected device snapshot"),
        ),
        patch.object(torch.cuda.Event, "synchronize", side_effect=AssertionError("host wait")),
        patch.object(torch.cuda.Stream, "synchronize", side_effect=AssertionError("host wait")),
    ):
        output = runner.sample_tokens()
    assert isinstance(output, _AsyncGenerationOutput)
    expected_ptrs = [source.data_ptr()] if kind == "tensor" else [source[i].data_ptr() for i in range(2)]
    assert [t.data_ptr() for t in output._snapshot._cuda_sources] == expected_ptrs
    if kind == "mapping":
        for rate in raw["sr"]:
            rate.fill_(-1)
    del raw, source, runner
    resolved = output.get_output()
    assert not output._snapshot._cuda_sources
    key = "audio" if kind == "mapping" else "model_outputs"
    for payload in resolved.multimodal_outputs:
        torch.testing.assert_close(payload[key], torch.full((32,), 7.0))
        if kind == "mapping":
            assert payload["sr"].item() == 24000
    assert resolved.req_ids == ["req-1", "req-2"]
    assert output.get_output() is resolved


def test_owned_generation_output_preserves_none_rows():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = _make_runner([None, torch.ones(8, device="cuda")], num_reqs=2)
    runner.model = SimpleNamespace(owns_generation_output_storage=True)
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.device = torch.device("cuda")
    runner.use_async_scheduling = True
    runner.async_output_copy_stream = torch.cuda.Stream()
    output = runner.sample_tokens().get_output()
    assert output.multimodal_outputs[0]["model_outputs"] is None
    torch.testing.assert_close(output.multimodal_outputs[1]["model_outputs"], torch.ones(8))


def test_outer_graph_execution_keeps_snapshot_even_for_owned_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.ones((1, 8), device="cuda")
    runner = _make_runner(source)
    runner.model = SimpleNamespace(owns_generation_output_storage=True)
    runner.model_config = SimpleNamespace(enforce_eager=False)
    runner.device = torch.device("cuda")
    runner.use_async_scheduling = True
    runner.async_output_copy_stream = torch.cuda.Stream()
    with patch(
        "vllm_omni.worker.gpu_generation_model_runner._copy_owned_generation_payload_to_cpu_async",
        side_effect=AssertionError("unsafe borrowed graph output"),
    ):
        output = runner.sample_tokens()
    source.fill_(-1)
    resolved = output.get_output()
    torch.testing.assert_close(resolved.multimodal_outputs[0]["model_outputs"], torch.ones(8))


@pytest.mark.parametrize("owned", [False, True])
def test_discarded_output_records_copy_stream(owned):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.ones((2, 32), device="cuda")
    runner = _make_runner(source, num_reqs=2)
    runner.model = SimpleNamespace(owns_generation_output_storage=owned)
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.device = source.device
    runner.use_async_scheduling = True
    runner.async_output_copy_stream = torch.cuda.Stream()
    calls = []
    original = torch.Tensor.record_stream

    def record(tensor, stream):
        calls.append((tensor.data_ptr(), stream))
        return original(tensor, stream)

    with patch.object(torch.Tensor, "record_stream", record):
        output = runner.sample_tokens()
    assert calls == [(output._snapshot._cuda_sources[0].data_ptr(), runner.async_output_copy_stream)]
    # Drop the GPU snapshot while the copy can still be in flight. CPU storage
    # remains readable after the copy stream finishes, without calling get_output.
    host = output._output.multimodal_outputs
    del output
    runner.async_output_copy_stream.synchronize()
    for row in host:
        torch.testing.assert_close(row["model_outputs"], torch.ones(32))


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_dense_waveforms_share_one_host_allocation(owned, strided):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.arange(128, device="cuda", dtype=torch.float32).reshape(4, 32)
    if strided:
        source = source[:, ::2]
    expected = source.cpu()
    runner = _make_runner(source, num_reqs=4)
    runner.model = SimpleNamespace(owns_generation_output_storage=owned)
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.device = source.device
    runner.use_async_scheduling = True
    runner.async_output_copy_stream = torch.cuda.Stream()
    output = runner.sample_tokens().get_output()
    rows = [payload["model_outputs"] for payload in output.multimodal_outputs]
    expected_allocations = len(rows) if strided else 1
    assert len({row.untyped_storage().data_ptr() for row in rows}) == expected_allocations
    assert all(row.is_contiguous() for row in rows)
    for i, row in enumerate(rows):
        torch.testing.assert_close(row, expected[i])
