# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_codec import (
    BreezeTTS2MimiCodec,
)
from vllm_omni.model_executor.stage_input_processors.breeze_tts_2 import (
    talker2codec_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TransferManager:
    def __init__(self, chunk_frames: int = 4):
        self.connector = SimpleNamespace(config={"breeze_codec_chunk_frames": chunk_frames})
        self.code_prompt_token_ids: defaultdict[str, list[torch.Tensor]] = defaultdict(list)


class _Request:
    def __init__(self, request_id: str, finished: bool = False):
        self.external_req_id = request_id
        self._finished = finished

    def is_finished(self):
        return self._finished


def test_async_processor_sends_only_unemitted_tail_and_flushes_finish():
    manager = _TransferManager(chunk_frames=4)
    request = _Request("req-1")

    for code0 in range(3):
        frame = torch.tensor([code0, code0 + 4, code0 + 8, code0 + 12], dtype=torch.long)
        assert talker2codec_async_chunk(manager, {"codes": {"audio": frame}}, request) is None

    payload = talker2codec_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([3, 7, 11, 15], dtype=torch.long)}},
        request,
    )
    assert payload is not None
    assert payload.codes.audio.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert bool(payload.meta.finished.item()) is False
    assert manager.code_prompt_token_ids["req-1"] == []

    assert talker2codec_async_chunk(manager, {"codes": {"audio": torch.arange(4)}}, request) is None
    payload = talker2codec_async_chunk(manager, None, _Request("req-1", finished=True))
    assert payload.codes.audio.tolist() == [0, 1, 2, 3]
    assert bool(payload.meta.finished.item()) is True
    assert bool(payload.meta.stream_finished.item()) is True
    assert payload.meta.codec_streaming is True


class _Decoder:
    def __init__(self):
        self.calls = []

    def batched_chunked_decode(self, codes, lengths, caches=None, **kwargs):
        self.calls.append((codes.shape, tuple(lengths), tuple(caches), kwargs))
        return [torch.ones(length * 1920, dtype=torch.float32) for length in lengths]


def test_stateful_codec_batches_request_chunks_and_releases_finished_state():
    decoder = _Decoder()
    codec = object.__new__(BreezeTTS2MimiCodec)
    codec._async_chunk = True
    codec._num_codebooks = 4
    codec._codebook_size = 8
    codec._sample_rate = 24_000
    codec._audio_tokenizer = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
    codec._decoder_state_cache = {}

    input_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    output = codec.forward(
        input_ids,
        runtime_additional_information=[
            # ``stream_finished`` is the model-visible flag: the connector
            # receiver strips ``meta.finished`` while merging runtime info.
            {"meta": {"request_id": "live", "stream_finished": False}},
            {"meta": {"request_id": "done", "stream_finished": True}},
        ],
        seq_token_counts=[4, 4],
        request_ids=["scheduler-live", "scheduler-done"],
    )

    assert decoder.calls[0][0] == (2, 4, 1)
    assert [len(item) for item in output.multimodal_outputs["model_outputs"]] == [1920, 1920]
    assert set(codec._decoder_state_cache) == {"scheduler-live"}

    codec.on_requests_finished(["scheduler-live"])
    assert codec._decoder_state_cache == {}


def test_stateful_codec_rejects_partial_frame_chunk():
    decoder = _Decoder()
    codec = object.__new__(BreezeTTS2MimiCodec)
    codec._async_chunk = True
    codec._num_codebooks = 4
    codec._codebook_size = 8
    codec._sample_rate = 24_000
    codec._audio_tokenizer = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
    codec._decoder_state_cache = {}

    # 5 codes cannot form a whole frame; a silent state reset would corrupt
    # the rest of the stream, so the codec must fail loudly.
    with pytest.raises(ValueError, match="not divisible"):
        codec.forward(
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
            runtime_additional_information=[{"meta": {"request_id": "bad", "finished": False}}],
            seq_token_counts=[5],
            request_ids=["scheduler-bad"],
        )
    assert decoder.calls == []


@pytest.mark.parametrize("token_counts", [(4, 0), (0, 4), (0, 4, 0, 8, 0)])
def test_stateful_codec_terminal_marker_pops_state_without_decoding(token_counts):
    decoder = _Decoder()
    codec = object.__new__(BreezeTTS2MimiCodec)
    codec._async_chunk = True
    codec._num_codebooks = 4
    codec._codebook_size = 8
    codec._sample_rate = 24_000
    codec._audio_tokenizer = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
    request_ids = [f"scheduler-{index}" for index in range(len(token_counts))]
    codec._decoder_state_cache = {request_id: {} for request_id in request_ids}

    output = codec.forward(
        torch.arange(sum(token_counts), dtype=torch.long) % codec._codebook_size,
        runtime_additional_information=[
            # Plain ``finished`` covers the fallback for direct in-process
            # callers that do not go through the connector receiver.
            {"meta": {"finished": count == 0}}
            for count in token_counts
        ],
        seq_token_counts=token_counts,
        request_ids=request_ids,
    )

    # Empty terminal markers release their state without consuming an output
    # slot belonging to another request, regardless of their batch position.
    assert len(decoder.calls) == 1
    assert decoder.calls[0][0][0] == sum(count > 0 for count in token_counts)
    assert [item.numel() for item in output.multimodal_outputs["model_outputs"]] == [
        count // codec._num_codebooks * 1920 for count in token_counts
    ]
    live_request_ids = [request_id for request_id, count in zip(request_ids, token_counts) if count]
    assert list(codec._decoder_state_cache) == live_request_ids
    assert all(codec._decoder_state_cache[request_id]["prefix_frames"] == 0 for request_id in live_request_ids)


@pytest.mark.parametrize("configured_chunk_frames", [None, 4])
def test_stateful_codec_advances_decoder_cache_after_sliding_window(monkeypatch, tmp_path, configured_chunk_frames):
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
    )

    torch.manual_seed(0)
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(2,),
        upsampling_ratios=(2,),
        sliding_window=72,
    )
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    # Codebooks initialize to zeros before checkpoint loading; populate them
    # so different codec IDs produce distinguishable cached frame histories.
    with torch.no_grad():
        for name, parameter in decoder.quantizer.named_parameters():
            if name.endswith("embedding_sum"):
                parameter.normal_()
    tokenizer = SimpleNamespace(model=SimpleNamespace(decoder=decoder, get_output_sample_rate=lambda: 24_000))
    monkeypatch.setattr(Qwen3TTSTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    connector_extra = {} if configured_chunk_frames is None else {"breeze_codec_chunk_frames": configured_chunk_frames}
    chunk_frames = configured_chunk_frames or 8
    codec = object.__new__(BreezeTTS2MimiCodec)
    codec.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(stage_connector_config={"extra": connector_extra}),
        device_config=SimpleNamespace(device="cpu"),
    )
    codec._tokenizer_path = tmp_path / "audio_tokenizer"
    codec._audio_tokenizer = None
    codec._async_chunk = True
    codec._num_codebooks = config.num_quantizers
    codec._codebook_size = config.codebook_size
    codec._decoder_state_cache = {}
    codec.load_weights(iter(()))

    # At least two chunks after the decoder's rolling window expose a stale
    # cache even when the first padded chunk still produces plausible audio.
    total_frames = config.sliding_window + 2 * chunk_frames
    tail_frames = 3
    codes = torch.randint(0, config.codebook_size, (1, config.num_quantizers, total_frames + tail_frames))
    for end in range(chunk_frames, total_frames + 1, chunk_frames):
        flat = codes[0, :, end - chunk_frames : end].reshape(-1)
        codec.forward(
            flat,
            runtime_additional_information=[{"meta": {"stream_finished": False}}],
            seq_token_counts=[flat.numel()],
            request_ids=["scheduler-live"],
        )
        state = codec._decoder_state_cache["scheduler-live"]
        assert state["suffix_frames"] == min(end - chunk_frames, config.sliding_window) + chunk_frames
        with torch.inference_mode():
            expected_quantized = decoder.quantizer.decode(codes[..., max(0, end - config.sliding_window) : end])
        torch.testing.assert_close(state["suffix_quantized"], expected_quantized)

    # A short terminal chunk must emit only its real frames, then release
    # state. Its waveform should agree with the unpadded single-request path.
    with torch.inference_mode():
        expected_tail = decoder.chunked_decode(codes[..., -tail_frames:], caches=copy.deepcopy(state))
    flat_tail = codes[0, :, -tail_frames:].reshape(-1)
    output = codec.forward(
        flat_tail,
        runtime_additional_information=[{"meta": {"stream_finished": True}}],
        seq_token_counts=[flat_tail.numel()],
        request_ids=["scheduler-live"],
    )
    waveform = output.multimodal_outputs["model_outputs"][0]
    assert waveform.numel() == tail_frames * decoder.total_upsample
    torch.testing.assert_close(waveform, expected_tail.reshape(-1), atol=1e-5, rtol=1e-4)
    assert codec._decoder_state_cache == {}
