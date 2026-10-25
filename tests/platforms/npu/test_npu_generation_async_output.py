# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generation publication on real NPU streams, including reused output buffers."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from tests.helpers.mark import hardware_marks  # noqa: E402
from vllm_omni.platforms.npu.worker import npu_generation_model_runner as module  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


def _runner(payload, *, asynchronous=True, accumulate=False):
    runner = object.__new__(module.NPUGenerationModelRunner)
    runner.device = torch.device("npu:0")
    runner.use_async_scheduling = asynchronous
    runner.async_output_copy_stream = torch.npu.Stream()
    runner.execute_model_state = SimpleNamespace(
        scheduler_output=None, ec_connector_output=None, cudagraph_stats=None, multimodal_outputs=payload
    )
    runner.kv_connector_output = None
    runner._prepare_omni_aux_output = lambda: None
    runner._finish_omni_aux_output = lambda *args: None
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=False))
    )
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["a", "b"],
        req_id_to_index={"a": 0, "b": 1},
        vocab_size=1,
        set_async_sampled_token_ids=lambda *args: None,
    )
    runner._async_chunk = False
    runner._should_accumulate_full_payload_output = lambda: accumulate
    runner.supports_mm_inputs = False
    runner.get_omni_connector_output = lambda: None
    runner.speculative_config = None
    runner.dynamic_eplb = False
    runner._finalize_dump_data = lambda: None
    runner.requests = {"a": object(), "b": object()}
    runner.accumulate_full_payload_output = lambda *args: None
    return runner


@pytest.mark.parametrize("kind", ["dense", "strided", "list", "mapping", "none"])
@pytest.mark.parametrize("width", [128, 600_000])
def test_deferred_generation_preserves_payload_and_request_snapshot(kind, width):
    source = torch.arange(2 * width, dtype=torch.float32, device="npu").reshape(2, width)
    if kind == "strided":
        source = source[:, ::2]
    expected = source.cpu().clone()
    payload = source
    if kind in ("list", "none"):
        payload = [source[0], None if kind == "none" else source[1]]
    elif kind == "mapping":
        payload = {"audio": list(source.unbind(0)), "sample_rate": torch.tensor(24000)}
    runner = _runner(payload)
    with patch.object(torch.npu.Event, "synchronize", side_effect=AssertionError("publication waited")):
        pending = runner.sample_tokens(None)
    assert isinstance(pending, module._AsyncGenerationOutput)
    # Simulate the next graph replay writing its reusable output allocation.
    source.fill_(-1)
    runner.input_batch.req_ids[:] = ["next"]
    runner.input_batch.req_id_to_index.clear()
    output = pending.get_output()
    assert pending.get_output() is output
    assert output.req_ids == ["a", "b"]
    assert output.req_id_to_index == {"a": 0, "b": 1}
    key = "audio" if kind == "mapping" else "model_outputs"
    for i, row in enumerate(output.multimodal_outputs):
        if kind == "none" and i == 1:
            assert row[key] is None
        else:
            assert row[key].device.type == "cpu"
            torch.testing.assert_close(row[key], expected[i], rtol=0, atol=0)


@pytest.mark.parametrize(
    "asynchronous,accumulate,pinned", [(False, False, True), (True, True, True), (True, False, False)]
)
def test_sync_fallback(asynchronous, accumulate, pinned):
    source = torch.ones(2, 128, device="npu")
    runner = _runner(source, asynchronous=asynchronous, accumulate=accumulate)
    with (
        patch.object(module, "is_pin_memory_available", return_value=pinned),
        patch.object(module, "_snapshot_tensor_payload_to_cpu_async", side_effect=AssertionError("async copy")),
    ):
        output = runner.sample_tokens(None)
        if asynchronous:
            output = output.get_output()
    torch.testing.assert_close(output.multimodal_outputs[0]["model_outputs"], torch.ones(128))
