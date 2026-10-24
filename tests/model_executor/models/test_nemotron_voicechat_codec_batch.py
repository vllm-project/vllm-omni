# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise decoder grouping and independent causal state on CPU."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.ear_tts_vae_codec import CausalConv1dCache
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_code2wav import (
    NemotronVoiceChatCode2Wav,
    _BatchedCausalConv1dCache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Codec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[int, int]] = []

    def decode(self, codes: torch.Tensor, lengths: torch.Tensor, *, cache: CausalConv1dCache | None):
        self.calls.append((codes.shape[0], codes.shape[1]))
        states = codes.float().sum(dim=-1).unsqueeze(1)
        padded = cache.update(states, "conv", 2) if cache is not None else torch.nn.functional.pad(states, (2, 0))
        wav = padded.unfold(2, 3, 1).sum(dim=-1).repeat_interleave(2, dim=2)
        return wav, lengths * 2


def _model() -> NemotronVoiceChatCode2Wav:
    model = NemotronVoiceChatCode2Wav.__new__(NemotronVoiceChatCode2Wav)
    nn.Module.__init__(model)
    model._sample_rate, model._num_quantizers, model._codebook_size, model._wav_per_frame = 22050, 31, 1024, 2
    model._use_codec_cache = False
    model._stream_caches = {}
    model.audio_codec = _Codec()
    return model


def _info(request_id: str | None, value: int, frames: int = 1, **meta: object) -> dict:
    if request_id is not None:
        meta.update(codec_streaming=True, request_id=request_id)
    return {"codes": {"audio": torch.full((frames, 31), value)}, "meta": meta}


def _decode(model: NemotronVoiceChatCode2Wav, infos: list[dict], request_ids: list[str] | None = None):
    return model.forward(runtime_additional_information=infos, request_ids=request_ids).multimodal_outputs[
        "model_outputs"
    ]


def _serial(model: NemotronVoiceChatCode2Wav, infos: list[dict]):
    return [_decode(model, [info])[0] for info in infos]


@pytest.mark.parametrize("streaming", [False, True])
def test_equal_length_requests_decode_together_in_output_order(streaming: bool) -> None:
    model, reference = _model(), _model()
    infos = [_info(f"r{i}" if streaming else None, value) for i, value in enumerate((3, 1, 7, 2))]
    actual, expected = _decode(model, infos), _serial(reference, infos)
    assert model.audio_codec.calls == [(4, 1)]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert all(audio.shape == (2,) for audio in actual)


def test_reordered_warm_and_cold_streams_keep_private_history() -> None:
    model, reference = _model(), _model()
    for infos in (
        [_info("a", 1), _info("b", 2)],
        [_info("b", 3), _info("c", 4)],  # a pauses; c starts without history
        [_info("c", 5), _info("a", 6), _info("b", 7)],
    ):
        actual, expected = _decode(model, infos), _serial(reference, infos)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        for request_id, cache in reference._stream_caches.items():
            assert torch.equal(model._stream_caches[request_id].cache["conv"], cache.cache["conv"])
    assert model.audio_codec.calls == [(2, 1), (2, 1), (3, 1)]
    model.on_requests_finished({"b"})
    reference.on_requests_finished({"b"})
    infos = [_info("a", 8), _info("b", 9)]
    assert all(torch.equal(a, b) for a, b in zip(_decode(model, infos), _serial(reference, infos), strict=True))


def test_mixed_lengths_modes_empty_and_trimmed_payloads() -> None:
    model, reference = _model(), _model()
    infos = [
        _info("a", 1),
        {"meta": {"nemotron_voicechat_dummy_profile": True}},
        _info(None, 2, frames=3, left_context_size=1),
        {"codes.audio": torch.full((1, 31), 3), "meta.codec_streaming": True, "meta.request_id": "b"},
        _info(None, 4, frames=2),
        _info("empty", 0, frames=0),
        _info(None, 5, frames=2, left_context_size=2),
    ]
    actual, expected = _decode(model, infos), _serial(reference, infos)
    assert model.audio_codec.calls == [(2, 1), (1, 3), (1, 2)]
    assert [len(audio) for audio in actual] == [2, 0, 4, 2, 4, 0, 0]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_shared_cache_is_advanced_serially_across_different_lengths() -> None:
    model, reference = _model(), _model()
    infos = [_info("a", 1), _info("a", 2, frames=2), _info("a", 3)]
    actual, expected = _decode(model, infos), _serial(reference, infos)
    assert model.audio_codec.calls == [(1, 1), (1, 2), (1, 1)]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_finished_stream_is_not_retained_in_another_stream_cache_storage() -> None:
    model = _model()
    _decode(model, [_info("a", 1, frames=256), _info("b", 2, frames=256)])
    model.on_requests_finished({"b"})
    tail = model._stream_caches["a"].cache["conv"]
    assert tail.shape == (1, 1, 2)
    assert tail.untyped_storage().nbytes() == tail.numel() * tail.element_size()


def test_turn_streaming_groups_only_the_new_suffix() -> None:
    model = _model()
    model._use_codec_cache = True
    _decode(model, [_info(None, 1, frames=2, left_context_size=0)], ["a"])
    infos = [_info(None, 2, frames=3, left_context_size=2), _info(None, 3, frames=1, left_context_size=0)]
    actual = _decode(model, infos, ["a", "b"])
    assert model.audio_codec.calls == [(1, 2), (2, 1)]
    assert [len(audio) for audio in actual] == [2, 2]
    assert set(model._stream_caches) == {"a", "b"}
    assert not torch.equal(actual[0], actual[1])


def test_batch_cache_preserves_padding_value_and_flush() -> None:
    caches = [CausalConv1dCache(), CausalConv1dCache()]
    batch = _BatchedCausalConv1dCache(caches)
    states = torch.tensor([[[1.0]], [[2.0]]])
    padded = batch.update(states, "conv", 2, padding_value=7)
    assert padded.tolist() == [[[7.0, 7.0, 1.0]], [[7.0, 7.0, 2.0]]]
    batch.update(states.flip(0), "conv", 2, flush=True)
    assert all(not cache.cache for cache in caches)
