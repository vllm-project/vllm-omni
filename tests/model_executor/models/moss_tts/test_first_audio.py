# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY
from vllm_omni.model_executor.models.moss_tts.first_audio_state import MossEarlyFirstAudioState
from vllm_omni.model_executor.models.moss_tts.first_frame_decoder import first_audio_enabled
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
    MossTTSCodecDecoder,
    _MossCodecStreamSession,
)
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration
from vllm_omni.model_executor.stage_input_processors.moss_tts import talker2codec_raw_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def owner():
    model = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
    nn.Module.__init__(model)
    model.n_vq = 2
    model.audio_pad_token_id = 16
    model.audio_assistant_slot_token_id = 7
    model.first_frame_decoder = object()
    model.embed_input_ids = lambda ids: ids[:, None].float().expand(-1, 4)
    return SimpleNamespace(
        model=model,
        _mtp_input_ids=torch.zeros(4, dtype=torch.long),
        _mtp_input_embeds=torch.zeros(4, 4),
        _mtp_hidden=torch.zeros(4, 4),
        _mtp_text_step=torch.zeros(4, 4),
        intermediate_buffer=SimpleNamespace(buffers=[{"req_id": "a"}, {"req_id": "b"}]),
    )


def entry(rid="a", idx=0, computed=2, length=2, prompt=4, cap=5):
    return (
        idx,
        idx,
        0,
        length,
        {
            "req_id": rid,
            "_omni_prompt_len": prompt,
            "_omni_num_computed_tokens": computed,
            "sampling_params": SimpleNamespace(max_tokens=cap),
        },
        True,
    )


def test_adapter_preserves_first_audio_promise_through_terminal_sentinel():
    manager = SimpleNamespace(
        connector=SimpleNamespace(config={"initial_codec_chunk_frames": 1, "codec_chunk_frames": 15})
    )
    request = SimpleNamespace(request_id="a")
    payload = talker2codec_raw_async_chunk(
        manager, {"codes": {"audio": torch.ones(1, 2)}, "meta": {"first_audio": torch.tensor(True)}}, request, False
    )
    assert bool(payload.meta.first_audio)
    manager.put_req_chunk["a"] = 1
    end = talker2codec_raw_async_chunk(manager, None, request, True)
    assert bool(end.meta.first_audio) and bool(end.meta.finished)
    assert not manager.request_payload and not manager.code_prompt_token_ids


class CausalCodec(nn.Module):
    downsample_rate = 1

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(codebook_size=16)

    def initialize_decoder_state_pool(self, capacity, scratch):
        self.state = torch.zeros(capacity + scratch)

    def reset_decoder_state_slots(self, slots):
        self.state[slots] = 0

    def decode_streaming_batch(self, codes, lengths, slots, valid_rows):
        audio = codes.sum(0).float().cumsum(-1) + self.state[slots, None]
        self.state[slots] = audio[:, -1]
        return SimpleNamespace(audio=audio[:, None].repeat(1, 2, 1), audio_lengths=lengths)


def decoder():
    d = MossTTSCodecDecoder.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(d)
    d._codec = CausalCodec()
    d._sr_tensor = torch.tensor(48000)
    d._n_channels = 2
    d._n_vq = 2
    d._async_chunk = True
    d._accept_first_audio = True
    d._stream_first_audio_requests = set()
    d._stream_req_slots = {}
    d._stream_state_capacity = 8
    d._stream_max_step_frames = 15
    s = _MossCodecStreamSession(d._codec, state_capacity=8, n_vq=2, vllm_config=None)
    d._stream_session = s
    d._ensure_stream_session = lambda: s
    return d


def call(d, req, codes, first=False, finished=False):
    return d.forward(
        input_ids=torch.tensor(codes, dtype=torch.long),
        seq_token_counts=[len(codes)],
        runtime_additional_information=[
            {"request_id": req, "meta": {"req_id": [req], "first_audio": first, "finished": finished}}
        ],
    ).multimodal_outputs


@pytest.mark.parametrize("first", [False, True])
def test_codec_primes_state_but_suppresses_only_already_delivered_frame(first):
    d = decoder()
    start = call(d, "a", [1, 2], first=first)
    expected = torch.tensor([[3.0], [3.0]])[:, 1:] if first else torch.tensor([[3.0], [3.0]])
    torch.testing.assert_close(start["model_outputs"][0], expected)
    # The second frame must include the causal state from the suppressed one.
    rest = call(d, "a", [2, 3], finished=True)
    torch.testing.assert_close(rest["model_outputs"][0], torch.tensor([[8.0], [8.0]]))
    if first:
        assert bool(rest[FIRST_AUDIO_REQUIRED_KEY][0])
    assert not d._stream_req_slots and not d._stream_first_audio_requests
    fresh = call(d, "new", [1, 2], finished=True)
    torch.testing.assert_close(fresh["model_outputs"][0], torch.tensor([[3.0], [3.0]]))


def test_codec_empty_terminal_carries_ordering_marker_until_cleanup():
    d = decoder()
    call(d, "a", [1, 2], first=True)
    end = call(d, "a", [], finished=True)
    assert bool(end[FIRST_AUDIO_REQUIRED_KEY][0])
    d.on_requests_finished({"a"})
    assert not d._stream_first_audio_requests


def test_feature_requires_supported_explicit_configuration():
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            stage_connector_config={"moss_talker_first_audio": True}, async_chunk=True, use_v2_model_runner=True
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, distributed_executor_backend="uni"
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        device_config=SimpleNamespace(device="cuda"),
        speculative_config=None,
    )
    assert first_audio_enabled(cfg)
    cfg.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError):
        first_audio_enabled(cfg)
    cfg.model_config.stage_connector_config["moss_talker_first_audio"] = False
    assert not first_audio_enabled(cfg)
    cfg.model_config.stage_connector_config["moss_talker_first_audio"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        first_audio_enabled(cfg)


@pytest.mark.parametrize("num_decode_rows", [1, 3])
def test_mixed_prefill_delivery_marker_survives_owned_output_copy(mocker, num_decode_rows):
    from vllm_omni.model_executor.models.output_templates import RequestBatchTensor
    from vllm_omni.worker_v2.omni_ar_model_runner import _async_copy_mm, _slice_pooler_value
    from vllm_omni.worker_v2.output_snapshot import pack_output_snapshot

    o = owner()
    state = MossEarlyFirstAudioState(o, None)
    o._first_audio_sender = object()
    req_ids = [f"decode-{i}" for i in range(num_decode_rows)] + ["new"]
    batch = SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids))
    state.record_prefills(batch, [entry(rid="new", idx=num_decode_rows, computed=3, length=1)])
    o._mtp_forward = mocker.Mock(return_value=(torch.ones(1, 4), torch.ones(1, 2, dtype=torch.long)))
    mocker.patch.object(state, "_publish", return_value=["new"])
    state.after_mtp(req_ids, torch.ones(len(req_ids), 2, dtype=torch.long), torch.full((len(req_ids),), 7))
    codes = torch.arange(len(req_ids) * 2).reshape(len(req_ids), 2)
    snapshot = pack_output_snapshot({"codes": {"audio": RequestBatchTensor(codes)}}, {}, max_buckets=8)
    outputs = state.after_sample(
        batch, torch.zeros(len(req_ids), 4), torch.full((len(req_ids), 1), 7), torch.ones(len(req_ids)), snapshot, None
    )
    codes.fill_(99)
    copied = _async_copy_mm(outputs, len(req_ids), pin_memory=False)
    for row in range(len(req_ids)):
        item = _slice_pooler_value(copied, req_index=row, start=row, end=row + 1, total_tokens=len(req_ids))
        assert bool(item["meta"]["first_audio"]) == (row == num_decode_rows)
        assert item["codes"]["audio"].tolist() == [[row * 2, row * 2 + 1]]
    # Steady decode preserves the original packed copy contract.
    assert state.after_sample(batch, None, None, None, snapshot, None) is snapshot


def test_metadata_only_prefill_promise_survives_steady_codec_and_terminal():
    manager = SimpleNamespace(
        connector=SimpleNamespace(config={"initial_codec_chunk_frames": 1, "codec_chunk_frames": 15})
    )
    request = SimpleNamespace(request_id="a")
    assert talker2codec_raw_async_chunk(manager, {"meta": {"first_audio": torch.tensor(True)}}, request, False) is None
    payload = talker2codec_raw_async_chunk(manager, {"codes": {"audio": torch.ones(1, 2)}}, request, False)
    assert bool(payload.meta.first_audio)
    manager.put_req_chunk["a"] = 1
    end = talker2codec_raw_async_chunk(manager, None, request, True)
    assert bool(end.meta.first_audio) and bool(end.meta.finished)
