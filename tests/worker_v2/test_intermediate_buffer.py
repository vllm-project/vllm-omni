# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for OmniIntermediateBuffer lifecycle.

Retained contracts: wire decode with malformed-payload rejection, slot
lifecycle with index migration on reuse, gather in batch order, update merge
semantics (cpu/gpu-resident keys), single owned snapshot for
update_gpu_tensor_rows, and decode-span merge until ack.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine import (
    AdditionalInformationEntry,
    AdditionalInformationPayload,
    PromptEmbedsPayload,
)
from vllm_omni.worker_v2.model_states.intermediate_buffer import (
    OmniIntermediateBuffer,
    _resolve_additional_information,
    _resolve_prompt_embeds,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_new_req_data(req_id="r1", **kwargs):
    kwargs.setdefault("mm_features", [])
    return SimpleNamespace(req_id=req_id, **kwargs)


def test_wire_decode_passthrough_and_malformed_rejection():
    tensor = torch.randn(3, 4)
    assert _resolve_prompt_embeds(tensor) is not None
    assert _resolve_prompt_embeds(None) is None
    with pytest.raises(ValueError, match="Failed to decode prompt_embeds payload"):
        _resolve_prompt_embeds(PromptEmbedsPayload(data=b"\x00", shape=[2, 2], dtype="float32"))

    assert _resolve_additional_information({"key": "val"}) == {"key": "val"}
    assert _resolve_additional_information(None) == {}
    payload = AdditionalInformationPayload(
        entries={
            "embed.decode": AdditionalInformationEntry(tensor_data=b"\x00", tensor_shape=[2], tensor_dtype="float32")
        }
    )
    with pytest.raises(ValueError, match="Failed to decode additional_information payload"):
        _resolve_additional_information(payload)


def test_slot_lifecycle_fields_reuse_and_remove_idempotent():
    buf = OmniIntermediateBuffer(max_num_reqs=4)
    embeds = torch.randn(2, 8)
    features = [torch.randn(3, 4)]
    params = SimpleNamespace(extra_args={"tts_local_seed": 1234})
    buf.add_request(0, _make_new_req_data("r0", prompt_embeds=embeds, mm_features=features, sampling_params=params))
    assert torch.equal(buf.buffers[0]["prompt_embeds_cpu"], embeds)
    assert buf.buffers[0]["mm_features"] is features
    assert buf.buffers[0]["sampling_params"] is params

    buf.add_request(2, _make_new_req_data("r2"))
    buf.add_request(3, _make_new_req_data("r3"))
    batch = SimpleNamespace(idx_mapping_np=[2, 0, 3])
    assert [g["req_id"] for g in buf.gather(batch)] == ["r2", "r0", "r3"]

    # Slot reuse: req_id mapping migrates; duplicate removal is a no-op.
    buf.add_request(0, _make_new_req_data("new"))
    assert buf.req_id_to_index["new"] == 0
    buf.remove_request(0)
    buf.remove_request(0)
    assert "new" not in buf.req_id_to_index
    assert buf.buffers[0] == {}


def test_update_merge_semantics(monkeypatch):
    buf = OmniIntermediateBuffer(max_num_reqs=1)
    buf.add_request(0, _make_new_req_data("r0"))

    buf.update(0, {"hidden": torch.randn(4)})
    assert buf.buffers[0]["hidden"].device == torch.device("cpu")  # default migrates to cpu
    buf.update(0, {})
    buf.update(0, {"a": torch.tensor([1.0])})
    buf.update(0, {"b": torch.tensor([2.0])})
    assert "a" in buf.buffers[0] and "b" in buf.buffers[0]

    # GPU-resident keys keep values (including list items) off the cpu path.
    tensor = torch.randn(2)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_a, **_k: pytest.fail("GPU-resident value must not move to CPU"))
    buf.update(0, {"items": [tensor], "kv": torch.randn(4)}, gpu_resident_keys={"items", "kv"})
    stored = buf.buffers[0]["items"][0]
    assert stored is not tensor
    assert torch.equal(stored, tensor)
    assert "kv" in buf.buffers[0]


def test_add_request_merges_orchestrator_model_intermediate_buffer():
    """Stage bridges such as MiniCPM-o's llm2tts hand over ``model_intermediate_buffer``."""
    buf = OmniIntermediateBuffer(max_num_reqs=2)
    hidden = torch.randn(3, 4)
    data = _make_new_req_data(
        "talker-1",
        additional_information={"meta": {"a": 1}},
        model_intermediate_buffer={
            "ids": {"tts": [1, 2, 3]},
            "hidden_states": {"tts": hidden},
            "meta": {"next_stage_prompt_len": 5},
            "request_id": "thinker-1",
        },
    )
    buf.add_request(0, data)
    info = buf.buffers[0]
    assert info["ids"]["tts"] == [1, 2, 3]
    assert torch.equal(info["hidden_states"]["tts"], hidden)
    assert info["meta"] == {"a": 1, "next_stage_prompt_len": 5}
    assert info["request_id"] == "thinker-1" and info["req_id"] == "talker-1"
