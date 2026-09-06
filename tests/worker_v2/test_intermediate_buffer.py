"""Unit tests for OmniIntermediateBuffer lifecycle."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine import (
    AdditionalInformationEntry,
    AdditionalInformationPayload,
    PromptEmbedsPayload,
)
from vllm_omni.worker.payload_span import merge_tensor_spans
from vllm_omni.worker_v2.model_states.intermediate_buffer import (
    OmniIntermediateBuffer,
    _resolve_additional_information,
    _resolve_prompt_embeds,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_merge_tensor_spans_reuses_incoming_after_cached_span_is_consumed() -> None:
    cached = torch.empty((0, 4))
    incoming = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    merged = merge_tensor_spans((cached, 7, 7), (incoming, 7, 9))

    assert merged is not None
    merged_tensor, start, end = merged
    assert merged_tensor is incoming
    assert (start, end) == (7, 9)


class _DummyInputBatch:
    def __init__(self, idx_mapping):
        self.idx_mapping_np = idx_mapping


def _make_new_req_data(req_id="r1", prompt_embeds=None, mm_features=None, additional_information=None):
    ns = SimpleNamespace(
        req_id=req_id,
        mm_features=mm_features or [],
    )
    if prompt_embeds is not None:
        ns.prompt_embeds = prompt_embeds
    if additional_information is not None:
        ns.additional_information = additional_information
    return ns


# ---------------------------------------------------------------
# _resolve_prompt_embeds
# ---------------------------------------------------------------


def test_resolve_prompt_embeds_tensor():
    t = torch.randn(3, 4, device="cpu")
    result = _resolve_prompt_embeds(t)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 4)
    assert result.is_contiguous()


def test_resolve_prompt_embeds_none():
    assert _resolve_prompt_embeds(None) is None


def test_resolve_prompt_embeds_rejects_malformed_wire_payload():
    payload = PromptEmbedsPayload(
        data=b"\x00",
        shape=[2, 2],
        dtype="float32",
    )

    with pytest.raises(ValueError, match="Failed to decode prompt_embeds payload"):
        _resolve_prompt_embeds(payload)


# ---------------------------------------------------------------
# _resolve_additional_information
# ---------------------------------------------------------------


def test_resolve_additional_information_dict_passthrough():
    d = {"key": "val"}
    assert _resolve_additional_information(d) == d


def test_resolve_additional_information_none():
    assert _resolve_additional_information(None) == {}


def test_resolve_additional_information_rejects_malformed_wire_payload():
    payload = AdditionalInformationPayload(
        entries={
            "embed.decode": AdditionalInformationEntry(
                tensor_data=b"\x00",
                tensor_shape=[2],
                tensor_dtype="float32",
            )
        }
    )

    with pytest.raises(ValueError, match="Failed to decode additional_information payload"):
        _resolve_additional_information(payload)


# ---------------------------------------------------------------
# OmniIntermediateBuffer core lifecycle
# ---------------------------------------------------------------


def test_add_and_remove():
    buf = OmniIntermediateBuffer(max_num_reqs=4)
    req = _make_new_req_data(req_id="r1")
    buf.add_request(0, req)
    assert buf.buffers[0]["req_id"] == "r1"

    buf.remove_request(0)
    assert buf.buffers[0] == {}


def test_add_with_prompt_embeds():
    pe = torch.randn(2, 8)
    req = _make_new_req_data(req_id="r1", prompt_embeds=pe)
    buf = OmniIntermediateBuffer(max_num_reqs=4)
    buf.add_request(0, req)
    assert "prompt_embeds_cpu" in buf.buffers[0]
    assert buf.buffers[0]["prompt_embeds_cpu"].shape == (2, 8)


def test_add_with_mm_features():
    features = [torch.randn(3, 4)]
    req = _make_new_req_data(req_id="r1", mm_features=features)
    buf = OmniIntermediateBuffer(max_num_reqs=4)
    buf.add_request(1, req)
    assert buf.buffers[1]["mm_features"] is features


def test_add_with_additional_information():
    info = {"audio_embeds": torch.randn(5)}
    req = _make_new_req_data(req_id="r1", additional_information=info)
    buf = OmniIntermediateBuffer(max_num_reqs=4)
    buf.add_request(0, req)
    assert torch.allclose(buf.buffers[0]["audio_embeds"], info["audio_embeds"])


def test_add_preserves_sampling_params():
    sampling_params = SimpleNamespace(extra_args={"tts_local_seed": 1234}, seed=5678)
    req = _make_new_req_data(req_id="r1")
    req.sampling_params = sampling_params
    buf = OmniIntermediateBuffer(max_num_reqs=4)

    buf.add_request(0, req)

    assert buf.buffers[0]["sampling_params"] is sampling_params


# ---------------------------------------------------------------
# gather
# ---------------------------------------------------------------


def test_gather_returns_batch_order():
    buf = OmniIntermediateBuffer(max_num_reqs=4)
    buf.add_request(0, _make_new_req_data(req_id="r0"))
    buf.add_request(2, _make_new_req_data(req_id="r2"))
    buf.add_request(3, _make_new_req_data(req_id="r3"))

    batch = _DummyInputBatch(idx_mapping=[2, 0, 3])
    gathered = buf.gather(batch)
    assert [g["req_id"] for g in gathered] == ["r2", "r0", "r3"]


# ---------------------------------------------------------------
# update
# ---------------------------------------------------------------


def test_update_merges_tensors_to_cpu():
    buf = OmniIntermediateBuffer(max_num_reqs=2)
    buf.add_request(0, _make_new_req_data(req_id="r0"))

    buf.update(0, {"hidden": torch.randn(4)})
    assert "hidden" in buf.buffers[0]
    assert buf.buffers[0]["hidden"].device == torch.device("cpu")


def test_update_gpu_resident_keys():
    buf = OmniIntermediateBuffer(max_num_reqs=2)
    buf.add_request(0, _make_new_req_data(req_id="r0"))

    t = torch.randn(4)
    buf.update(0, {"kv": t}, gpu_resident_keys={"kv"})
    assert "kv" in buf.buffers[0]


def test_update_gpu_tensor_rows_takes_one_owned_batch_snapshot():
    buf = OmniIntermediateBuffer(max_num_reqs=2)
    buf.add_request(0, _make_new_req_data(req_id="r0"))
    buf.add_request(1, _make_new_req_data(req_id="r1"))
    source = torch.tensor([[1, 2, 3], [4, 5, 6]])

    buf.update_gpu_tensor_rows(
        [1, 0],
        ("codes", "audio"),
        source,
    )
    source.fill_(99)

    first = buf.buffers[0]["codes"]["audio"]
    second = buf.buffers[1]["codes"]["audio"]
    assert first.shape == (1, 3)
    assert second.shape == (1, 3)
    assert torch.equal(first, torch.tensor([[4, 5, 6]]))
    assert torch.equal(second, torch.tensor([[1, 2, 3]]))
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert first.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_update_list_values():
    buf = OmniIntermediateBuffer(max_num_reqs=2)
    buf.add_request(0, _make_new_req_data(req_id="r0"))

    items = [torch.randn(2), "text"]
    buf.update(0, {"items": items})
    stored = buf.buffers[0]["items"]
    assert isinstance(stored[0], torch.Tensor)
    assert stored[1] == "text"


def test_gpu_resident_list_values_do_not_call_cpu(monkeypatch):
    buf = OmniIntermediateBuffer(max_num_reqs=1)
    buf.add_request(0, _make_new_req_data(req_id="r0"))
    tensor = torch.randn(2)

    def fail_cpu(_self, *_args, **_kwargs):
        raise AssertionError("GPU-resident list item must not move to CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)
    buf.update(0, {"items": [tensor]}, gpu_resident_keys={"items"})

    stored = buf.buffers[0]["items"][0]
    assert stored.data_ptr() != tensor.data_ptr()
    assert torch.equal(stored, tensor)


def test_update_empty_is_noop():
    buf = OmniIntermediateBuffer(max_num_reqs=2)
    buf.add_request(0, _make_new_req_data(req_id="r0"))
    before = dict(buf.buffers[0])
    buf.update(0, {})
    assert buf.buffers[0] == before


def test_update_accumulates():
    buf = OmniIntermediateBuffer(max_num_reqs=2)
    buf.add_request(0, _make_new_req_data(req_id="r0"))
    buf.update(0, {"a": torch.tensor([1.0])})
    buf.update(0, {"b": torch.tensor([2.0])})
    assert "a" in buf.buffers[0] and "b" in buf.buffers[0]


def test_update_merges_pending_absolute_decode_spans_until_ack() -> None:
    buf = OmniIntermediateBuffer(max_num_reqs=1)
    buf.add_request(0, _make_new_req_data(req_id="r0"))
    gpu_keys = {("embed", "decode")}

    buf.update(
        0,
        {
            "embed": {
                "decode": torch.tensor([[3.0]]),
                "decode_token_start": 3,
                "decode_token_end": 4,
            }
        },
        gpu_keys,
    )
    buf.update(
        0,
        {
            "embed": {
                "decode": torch.tensor([[4.0], [5.0], [6.0]]),
                "decode_token_start": 4,
                "decode_token_end": 7,
            }
        },
        gpu_keys,
    )

    embed = buf.buffers[0]["embed"]
    assert embed["decode_token_start"] == 3
    assert embed["decode_token_end"] == 7
    assert torch.equal(embed["decode"], torch.tensor([[3.0], [4.0], [5.0], [6.0]]))

    buf.update(
        0,
        {
            "embed": {
                "decode": None,
                "decode_token_start": None,
                "decode_token_end": None,
            }
        },
        gpu_keys,
    )

    assert buf.buffers[0]["embed"]["decode"] is None


def test_update_gpu_resident_list_keeps_tensor_items_on_device():
    buf = OmniIntermediateBuffer(max_num_reqs=1)
    source = torch.tensor([1.0])

    buf.update(0, {"items": [source]}, gpu_resident_keys={"items"})

    stored = buf.buffers[0]["items"][0]
    assert stored.device == source.device
    assert stored is not source
