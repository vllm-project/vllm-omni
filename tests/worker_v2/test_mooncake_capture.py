# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the MRv2 Mooncake capture/publication boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.mooncake_capture import CaptureRequestContext, MooncakeOutputRewrite, snapshot_mooncake_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CaptureConnector:
    hidden_layer = 24

    def __init__(self):
        self.attempts = {"a": "attempt-a", "b": "attempt-b"}
        self.calls = []
        self.prefix: dict[str, torch.Tensor] | None = {
            "embedding": torch.full((4, 2), 10.0),
            "hidden": torch.full((4, 2), 20.0),
        }

    def current_attempt(self, req_id):
        return self.attempts.get(req_id)

    def capture_step(self, req_id, attempt_id, start_token, prompt_len, tensors):
        self.calls.append((req_id, attempt_id, start_token, prompt_len))
        if self.current_attempt(req_id) != attempt_id:
            return None
        if req_id == "a" and self.prefix is not None:
            prefix, self.prefix = self.prefix, None
            return {key: torch.cat((prefix[key][:start_token], value)) for key, value in tensors.items()}
        return tensors


def _payload(value):
    return {
        "hidden_states.layer_0": torch.full((2, 2), value),
        "hidden_states.layer_24": torch.full((2, 2), value + 1),
        "embed.sampled": torch.full((1, 2), value + 2),
        "speaker": "current-request-speaker",
    }


def _output(*, aliased=True):
    payloads = [_payload(1.0), _payload(3.0)]
    return OmniModelRunnerOutput(
        req_ids=["a", "b"],
        req_id_to_index={"a": 0, "b": 1},
        inter_stage_outputs=payloads,
        pooler_output=payloads if aliased else [dict(p) for p in payloads],
        multimodal_outputs=[dict(p) for p in payloads],
        sampled_token_ids=[[7], [8]],
        request_errors=None,
    )


@pytest.mark.parametrize("aliased", [True, False])
def test_rewrite_once_before_publication_preserves_other_request_data(aliased):
    connector = CaptureConnector()
    rewrite = MooncakeOutputRewrite(connector, (CaptureRequestContext(0, "a", "attempt-a", 4, 6),))
    output = _output(aliased=aliased)
    untouched = output.inter_stage_outputs[1]["hidden_states.layer_0"]
    sampled = output.inter_stage_outputs[0]["embed.sampled"]
    published = []

    def publish(out):
        published.append(out.inter_stage_outputs[0]["hidden_states.layer_0"].shape[0])
        return out

    assert rewrite.finalize(publish)(output) is output
    assert published == [6]
    assert connector.calls == [("a", "attempt-a", 4, 6)]
    for channel in (output.inter_stage_outputs, output.pooler_output, output.multimodal_outputs):
        assert channel[0]["hidden_states.layer_0"].shape == (6, 2)
        assert channel[0]["hidden_states.layer_24"].shape == (6, 2)
        assert channel[0]["embed.sampled"] is sampled
        assert channel[0]["speaker"] == "current-request-speaker"
        assert channel[1]["hidden_states.layer_0"] is untouched
    with pytest.raises(RuntimeError, match="already consumed"):
        rewrite.apply(output)


def test_next_decode_step_does_not_reinsert_prompt_rows():
    connector = CaptureConnector()
    first = _output()
    MooncakeOutputRewrite(connector, (CaptureRequestContext(0, "a", "attempt-a", 4, 6),)).apply(first)
    second = _output()
    MooncakeOutputRewrite(connector, (CaptureRequestContext(0, "a", "attempt-a", 6, 6),)).apply(second)
    assert first.inter_stage_outputs[0]["hidden_states.layer_0"].shape[0] == 6
    assert second.inter_stage_outputs[0]["hidden_states.layer_0"].shape[0] == 2


@pytest.mark.parametrize("replacement", [None, "new-attempt"])
def test_cancelled_or_reused_request_id_cannot_publish_old_step(replacement):
    connector = CaptureConnector()
    rewrite = MooncakeOutputRewrite(connector, (CaptureRequestContext(0, "a", "attempt-a", 4, 6),))
    if replacement is None:
        del connector.attempts["a"]
    else:
        connector.attempts["a"] = replacement
    output = _output()
    rewrite.apply(output)
    assert output.sampled_token_ids == [[], [8]]
    for channel in (output.inter_stage_outputs, output.pooler_output, output.multimodal_outputs):
        assert not channel[0]
        assert channel[1]


def test_missing_capture_fails_only_affected_request_before_publication():
    output = _output()
    for channel in (output.inter_stage_outputs, output.multimodal_outputs):
        channel[0].pop("hidden_states.layer_24")
    connector = CaptureConnector()
    rewrite = MooncakeOutputRewrite(connector, (CaptureRequestContext(0, "a", "attempt-a", 4, 6),))
    rewrite.apply(output)
    assert output.request_errors == {"a": "Missing Thinker captures for Mooncake"}
    assert output.sampled_token_ids == [[], [8]]
    for channel in (output.inter_stage_outputs, output.pooler_output, output.multimodal_outputs):
        assert not channel[0]
        assert channel[1]
    assert not connector.calls


@pytest.mark.parametrize("failure", ["type", "layout"])
def test_malformed_capture_does_not_stop_healthy_rows(failure):
    class ValidatingConnector(CaptureConnector):
        def capture_step(self, req_id, attempt_id, start_token, prompt_len, tensors):
            if req_id == "a":
                raise ValueError("HS captures must have matching shapes and dtypes")
            return super().capture_step(req_id, attempt_id, start_token, prompt_len, tensors)

    connector = ValidatingConnector()
    output = _output()
    if failure == "type":
        output.inter_stage_outputs[0]["hidden_states.layer_0"] = "not a tensor"
    rewrite = MooncakeOutputRewrite(
        connector, (CaptureRequestContext(0, "a", "attempt-a", 4, 6), CaptureRequestContext(1, "b", "attempt-b", 0, 10))
    )
    rewrite.apply(output)
    assert set(output.request_errors) == {"a"}
    assert output.sampled_token_ids == [[], [8]]
    assert connector.calls == [("b", "attempt-b", 0, 10)]
    assert not output.inter_stage_outputs[0]
    assert output.inter_stage_outputs[1]["hidden_states.layer_0"].shape == (2, 2)


def test_model_error_does_not_try_to_save_missing_captures():
    output = _output()
    output.request_errors = {"a": "failed model output"}
    output.inter_stage_outputs[0] = None
    connector = CaptureConnector()
    MooncakeOutputRewrite(connector, (CaptureRequestContext(0, "a", "attempt-a", 4, 6),)).apply(output)
    assert not connector.calls


@dataclass
class PromptLengthMirror:
    """CPU mirror used instead of allocating RequestState's CUDA UVA tensors."""

    np: NDArray[np.int32]


@dataclass
class SnapshotRequestState:
    prompt_len: PromptLengthMirror


@dataclass
class SnapshotConnector:
    kv_connector: CaptureConnector


@dataclass
class SnapshotRunner:
    """Only runner-owned fields read by snapshot_mooncake_output."""

    vllm_config: VllmConfig
    kv_connector: SnapshotConnector
    req_states: SnapshotRequestState


def _runner_batch(connector: CaptureConnector) -> tuple[SnapshotRunner, InputBatch]:
    runner = SnapshotRunner(
        vllm_config=VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            kv_transfer_config=KVTransferConfig(
                kv_connector="OmniMooncakeStoreConnector",
                kv_connector_module_path="vllm_omni.distributed.kv_transfer.mooncake.connector",
                kv_role="kv_both",
                kv_load_failure_policy="recompute",
            ),
        ),
        kv_connector=SnapshotConnector(connector),
        req_states=SnapshotRequestState(PromptLengthMirror(np.array([10, 6], dtype=np.int32))),
    )
    # The real upstream factory establishes the tensor/CPU-mirror shapes.
    batch = InputBatch.make_dummy(2, 4, InputBuffers(2, 4, torch.device("cpu")), is_padding=False)
    batch.req_ids[:] = ["a", "b"]
    batch.idx_mapping_np[:] = [1, 0]
    batch.idx_mapping.copy_(torch.from_numpy(batch.idx_mapping_np))
    batch.num_computed_tokens_np[:] = [4, 0]
    return runner, batch


def test_snapshot_preserves_positions_after_batch_reuse():
    connector = CaptureConnector()
    runner, batch = _runner_batch(connector)
    rewrite = snapshot_mooncake_output(runner, batch)
    batch.num_computed_tokens_np[:] = 100
    runner.req_states.prompt_len.np[:] = 200
    batch.req_ids[:] = ["different"]
    assert rewrite.rows == (
        CaptureRequestContext(0, "a", "attempt-a", 4, 6),
        CaptureRequestContext(1, "b", "attempt-b", 0, 10),
    )


def test_no_connector_or_warmup_is_a_noop():
    runner, batch = _runner_batch(CaptureConnector())
    runner.vllm_config.kv_transfer_config = None
    assert snapshot_mooncake_output(runner, batch) is None
    runner, batch = _runner_batch(CaptureConnector())
    batch.req_ids = ["_warmup_0", "_v2_mixed_warmup_1"]
    assert snapshot_mooncake_output(runner, batch) is None


def test_missing_attempt_is_rejected():
    connector = CaptureConnector()
    runner, batch = _runner_batch(connector)
    del connector.attempts["a"]
    with pytest.raises(RuntimeError, match="Missing Mooncake load attempt"):
        snapshot_mooncake_output(runner, batch)


def test_real_request_assembly_across_chunked_prefill_and_decode():
    from vllm_omni.distributed.kv_transfer.mooncake.thinker_captures import RequestHS

    state = RequestHS(prompt_tokens=8, block_size=4, max_bytes=1024)
    prefix = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    state.restore({"embedding": prefix, "hidden": prefix + 100})

    class AssemblingConnector(CaptureConnector):
        def capture_step(self, req_id, attempt_id, start_token, prompt_len, tensors):
            return state.append(start_token, tensors)

    connector = AssemblingConnector()
    delivered = []
    for start in (4, 6, 8):
        output = _output()
        rows = torch.arange(start * 2, (start + 2) * 2, dtype=torch.float32).reshape(2, 2)
        output.inter_stage_outputs[0]["hidden_states.layer_0"] = rows
        output.inter_stage_outputs[0]["hidden_states.layer_24"] = rows + 100
        MooncakeOutputRewrite(connector, (CaptureRequestContext(0, "a", "attempt-a", start, 8),)).apply(output)
        delivered.append(output.inter_stage_outputs[0]["hidden_states.layer_0"])
    assert torch.equal(torch.cat(delivered), torch.arange(20, dtype=torch.float32).reshape(10, 2))
    blocks = state.pop_blocks()
    assert [(start, end) for start, end, _ in blocks] == [(4, 8)]
    assert torch.equal(blocks[0][2]["embedding"], torch.arange(8, 16, dtype=torch.float32).reshape(4, 2))
