# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generation output copies must preserve producer ordering and CPU readiness."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm_ascend")

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms.npu.worker.npu_ar_model_runner import ExecuteModelState
from vllm_omni.platforms.npu.worker.npu_generation_model_runner import (
    NPUGenerationModelRunner,
    _AsyncNPUGenerationOutput,
    _NPUHostCopyBatch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


def make_runner(payload, num_reqs, *, asynchronous=True, accumulate=False):
    runner = object.__new__(NPUGenerationModelRunner)
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=False, need_timing=False))
    )
    runner.kv_connector_output = None
    runner.execute_model_state = ExecuteModelState(*([None] * 12), payload)
    ids = [f"req-{i}" for i in range(num_reqs)]
    runner.input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        vocab_size=10,
        set_async_sampled_token_ids=lambda *args: None,
    )
    runner.device = torch.device("npu:0")
    runner.async_output_copy_stream = torch.npu.Stream()
    runner.use_async_scheduling = asynchronous
    runner._async_chunk = True
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.dynamic_eplb = False
    runner.requests = dict.fromkeys(ids, object())
    runner._prepare_omni_aux_output = lambda: None
    runner._finish_omni_aux_output = lambda *a: None
    runner._should_accumulate_full_payload_output = lambda: accumulate
    runner.get_omni_connector_output = lambda: None
    runner._finalize_dump_data = lambda: None
    return runner


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["tensor", "list", "mapping"])
@torch.inference_mode()
def test_sample_tokens_preserves_outputs_across_next_forward(asynchronous, kind):
    source = torch.arange(4 * 24000, device="npu", dtype=torch.float32).reshape(4, 24000)
    expected = source.cpu()
    meta = torch.tensor(24000)
    if kind == "mapping":
        payload = {"audio": list(source.unbind()), "sr": meta}
        rows, key = 4, "audio"
    else:
        payload = source[:1] if kind == "tensor" else [source[0]]
        rows, key = 1, "model_outputs"
    runner = make_runner(payload, rows, asynchronous=asynchronous)
    output = runner.sample_tokens(None)
    if asynchronous:
        assert isinstance(output, _AsyncNPUGenerationOutput)
        # Simulate a subsequent model step reusing both device and CPU storage.
        source.fill_(-1)
        meta.fill_(0)
        runner.input_batch.req_ids.clear()
        output = output.get_output()
    for i in range(rows):
        value = output.multimodal_outputs[i][key]
        assert value.device.type == "cpu" and value.is_contiguous()
        torch.testing.assert_close(value, expected[i], rtol=0, atol=0)
        if kind == "mapping":
            assert output.multimodal_outputs[i]["sr"].item() == 24000
    assert len(output.req_ids) == rows


@pytest.mark.parametrize("pin_memory", [False, True])
@torch.inference_mode()
def test_strided_copy_empty_tensor_and_cpu_snapshot(pin_memory):
    x = torch.arange(64, device="npu").reshape(8, 8).t()
    with _NPUHostCopyBatch(x.device, pin_memory=pin_memory, snapshot_cpu=True) as copies:
        host = copies.copy(x)
        empty = copies.copy(x[:0])
        cpu = torch.tensor([7])
        snapshot = copies.copy(cpu)
    cpu.fill_(9)
    copies.wait()
    copies.wait()
    torch.testing.assert_close(host, x.cpu())
    assert host.is_contiguous() and empty.numel() == 0 and snapshot.item() == 7
    assert not copies.sources


@pytest.mark.parametrize("asynchronous", [False, True])
@torch.inference_mode()
def test_full_payload_accumulation_reads_completed_copies(asynchronous):
    source = torch.ones(24000, device="npu")
    runner = make_runner({"hidden": [source]}, 1, asynchronous=asynchronous, accumulate=True)
    seen = []
    runner.accumulate_full_payload_output = lambda rid, payload, state: seen.append(payload["hidden"].clone())
    output = runner.sample_tokens(None)
    if asynchronous:
        assert output._copies.event is None
        output.get_output()
    torch.testing.assert_close(seen[0], torch.ones(24000))


@torch.inference_mode()
def test_copy_exception_waits_for_already_submitted_tensors():
    x = torch.ones(24000, device="npu")
    with pytest.raises(ValueError, match="invalid later payload"):
        with _NPUHostCopyBatch(x.device, pin_memory=True) as copies:
            host = copies.copy(x)
            raise ValueError("invalid later payload")
    assert not copies.sources
    torch.testing.assert_close(host, torch.ones(24000))


@pytest.mark.parametrize("guard", ["pin_memory", "execution_timing"])
@torch.inference_mode()
def test_async_output_fallbacks(guard, monkeypatch):
    from vllm_omni.platforms.npu.worker import npu_generation_model_runner as module

    source = torch.full((1, 24000), 3.0, device="npu")
    runner = make_runner(source, 1)
    if guard == "pin_memory":
        monkeypatch.setattr(module, "is_pin_memory_available", lambda: False)
    else:
        config = runner.ascend_config.scheduler_config.profiling_chunk_config
        config.enabled = config.need_timing = True
    output = runner.sample_tokens(None)
    assert output._copies.event is None
    output = output.get_output()
    torch.testing.assert_close(output.multimodal_outputs[0]["model_outputs"], torch.full((24000,), 3.0))


@torch.inference_mode()
def test_mapping_payload_copies_asynchronously(monkeypatch):
    """A mapping payload must use the batch's async D2H, not a blocking ``.to("cpu")``.

    PR #9021 replaced the mapping branch's ``to_host.copy`` with
    ``tensor.detach().to("cpu").contiguous()`` while factoring out
    ``split_mapping_outputs``. That made every mapping tensor a synchronous
    device-to-host copy on the caller's thread: the tensor and list branches
    keep the pinned, event-tracked batch copy, so only mapping payloads
    regressed. On a stage-2 (audio) output this showed up as 37 blocking
    ``aten::copy_`` calls costing 1.1 s of a c8 run.
    """
    from vllm_omni.platforms.npu.worker import npu_generation_model_runner as module

    blocking: list[int] = []
    real_copy = module._NPUHostCopyBatch.copy

    def spy(self, value):
        blocking.append(1)
        return real_copy(self, value)

    monkeypatch.setattr(module._NPUHostCopyBatch, "copy", spy)

    source = torch.arange(4 * 24000, device="npu", dtype=torch.float32).reshape(4, 24000)
    runner = make_runner({"audio": list(source.unbind()), "sr": torch.tensor(24000)}, 4)
    output = runner.sample_tokens(None)

    # Every per-request tensor went through the batch copy, so an event was
    # recorded and the sources stay alive until the consumer waits.
    assert blocking, "mapping payload bypassed _NPUHostCopyBatch.copy"
    assert output._copies.event is not None
    assert output._copies.sources
    resolved = output.get_output()
    assert resolved.multimodal_outputs[0]["audio"].device.type == "cpu"
