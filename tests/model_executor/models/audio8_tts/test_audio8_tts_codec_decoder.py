# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_CODEBOOKS = 10
#: Samples the fake codec returns per frame, so trims are easy to assert.
SAMPLES_PER_FRAME = 100


@functools.lru_cache(maxsize=1)
def _decoder_cls():
    from vllm_omni.model_executor.models.audio8_tts.audio8_tts_codec_decoder import Audio8TTSCodecDecoder

    return Audio8TTSCodecDecoder


class _FakeCodec:
    """Returns ``SAMPLES_PER_FRAME`` ramp samples per frame."""

    def __init__(self):
        self.calls: list[tuple[int, ...]] = []

    def decode(self, codes_bqf: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(codes_bqf.shape))
        frames = int(codes_bqf.shape[-1])
        total = frames * SAMPLES_PER_FRAME
        return torch.arange(total, dtype=torch.float32).view(1, 1, total)


def _make_decoder(codec: _FakeCodec, num_codebooks: int = NUM_CODEBOOKS, splits=None):
    decoder = object.__new__(_decoder_cls())
    torch.nn.Module.__init__(decoder)
    decoder._codec = codec
    decoder._codec_device = torch.device("cpu")
    decoder._codec_dtype = torch.float32
    decoder._num_codebooks = num_codebooks
    decoder._sample_rate = 44100
    decoder._frame_size = 2048
    decoder._logged_stats = True
    decoder._ensure_codec_loaded = lambda: None
    if splits is not None:
        decoder._split_request_ids = lambda ids, seq_token_counts=None: splits
    return decoder


def test_left_context_frames_are_trimmed_from_the_emitted_chunk():
    """The decoder must emit *delta* audio: the left-context prefix exists only
    so the causal codec has history and must never reach the client twice."""
    codec = _FakeCodec()
    decoder = _make_decoder(codec)
    codes = torch.arange(NUM_CODEBOOKS * 4, dtype=torch.long).reshape(NUM_CODEBOOKS, 4)

    out = decoder.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[{"codes": {"audio": codes}, "meta": {"left_context_size": 3}}],
    )

    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 1
    # 4 frames decoded, 3 of context trimmed => 1 frame of new audio.
    assert audios[0].shape[0] == SAMPLES_PER_FRAME
    assert codec.calls == [(1, NUM_CODEBOOKS, 4)]


def test_tensor_codes_from_runtime_info_win_over_input_ids():
    codec = _FakeCodec()
    decoder = _make_decoder(codec, num_codebooks=2)
    out = decoder.forward(
        input_ids=torch.zeros(64, dtype=torch.long),
        runtime_additional_information=[{"codes": {"audio": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)}}],
    )
    assert codec.calls == [(1, 2, 2)]
    assert out.multimodal_outputs["model_outputs"][0].shape[0] == 2 * SAMPLES_PER_FRAME


def test_flat_input_ids_are_reshaped_codebook_major():
    codec = _FakeCodec()
    decoder = _make_decoder(codec, num_codebooks=2)
    out = decoder.forward(input_ids=torch.arange(6, dtype=torch.long), runtime_additional_information=[{}])
    assert codec.calls == [(1, 2, 3)]
    assert out.multimodal_outputs["model_outputs"][0].shape[0] == 3 * SAMPLES_PER_FRAME


def test_every_return_path_sets_model_outputs_and_sr():
    """A branch that omits ``model_outputs`` makes the serving layer drop the
    chunk silently, so assert the key exists on all the early-exit paths."""
    codec = _FakeCodec()

    empty = _make_decoder(codec).forward(input_ids=torch.zeros(0, dtype=torch.long))
    assert empty.multimodal_outputs["model_outputs"][0].numel() == 0
    assert empty.multimodal_outputs["sr"]

    ragged = _make_decoder(codec, num_codebooks=4).forward(
        input_ids=torch.arange(6, dtype=torch.long),  # 6 % 4 != 0
        runtime_additional_information=[{}],
    )
    assert ragged.multimodal_outputs["model_outputs"][0].numel() == 0

    codes = torch.zeros((NUM_CODEBOOKS, 2), dtype=torch.long)
    no_new = _make_decoder(codec).forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[{"codes": {"audio": codes}, "meta": {"left_context_size": 2}}],
    )
    assert no_new.multimodal_outputs["model_outputs"][0].numel() == 0
    assert codec.calls == [], "a context-only chunk must not be decoded"


def test_mixed_batch_keeps_per_request_alignment():
    """An empty request must not shift the following request's audio into its
    slot: the output list index is the request index."""
    codec = _FakeCodec()
    splits = [
        torch.empty((0,), dtype=torch.long),
        torch.arange(NUM_CODEBOOKS * 2, dtype=torch.long),
    ]
    decoder = _make_decoder(codec, splits=splits)
    out = decoder.forward(
        input_ids=torch.arange(NUM_CODEBOOKS * 2, dtype=torch.long),
        runtime_additional_information=[{}, {"meta": {"left_context_size": 1}}],
    )
    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert audios[0].numel() == 0
    assert audios[1].shape[0] == SAMPLES_PER_FRAME


def test_codes_with_wrong_codebook_count_fall_back_to_input_ids():
    codec = _FakeCodec()
    decoder = _make_decoder(codec, num_codebooks=2)
    out = decoder.forward(
        input_ids=torch.arange(4, dtype=torch.long),
        runtime_additional_information=[{"codes": {"audio": torch.zeros((5, 3), dtype=torch.long)}}],
    )
    assert codec.calls == [(1, 2, 2)]
    assert out.multimodal_outputs["model_outputs"][0].shape[0] == 2 * SAMPLES_PER_FRAME
