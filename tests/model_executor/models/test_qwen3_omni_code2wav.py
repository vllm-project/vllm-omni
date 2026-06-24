# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
    _reshape_code2wav_input_ids,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
    Qwen3OmniMoeCode2Wav,
)


class _Decoder:
    chunked_decode = Qwen3OmniMoeCode2Wav.chunked_decode
    chunked_decode_streaming = Qwen3OmniMoeCode2Wav.chunked_decode_streaming

    def __init__(self) -> None:
        self._cudagraph_enabled = False
        self._cudagraph_wrapper = None
        self.config = SimpleNamespace(num_quantizers=16)
        self.total_upsample = 2

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        batch_size = codes.shape[0]
        wav_len = codes.shape[-1] * self.total_upsample
        return torch.arange(
            batch_size * wav_len,
            dtype=torch.float32,
        ).reshape(batch_size, 1, wav_len)


def test_chunked_decode_ignores_stale_multi_request_metadata() -> None:
    decoder = _Decoder()
    codes = torch.zeros((1, 16, 3), dtype=torch.long)

    wavs = decoder.chunked_decode(
        codes,
        chunk_size=8,
        left_context_size=0,
        seq_token_counts=[16, 32],
    )

    expected = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    assert len(wavs) == 1
    torch.testing.assert_close(wavs[0], expected)


def test_chunked_decode_streaming_ignores_stale_multi_request_metadata() -> None:
    decoder = _Decoder()
    codes = torch.zeros((1, 16, 3), dtype=torch.long)

    wavs = decoder.chunked_decode_streaming(
        codes,
        left_context_size=[0, 1],
        seq_token_counts=[16, 32],
    )

    expected = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    assert len(wavs) == 1
    torch.testing.assert_close(wavs[0], expected)


def test_code2wav_forward_returns_request_aligned_empty_audio_for_stale_counts() -> None:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.model_stage = "code2wav"

    def raise_if_called(*args: object, **kwargs: object) -> list[torch.Tensor]:
        raise AssertionError("generate_audio should not run for stale scheduler counts")

    model.generate_audio = raise_if_called

    audio_tensors = Qwen3OmniMoeForConditionalGeneration.forward(
        model,
        input_ids=torch.arange(16, dtype=torch.long),
        positions=torch.empty(0, dtype=torch.long),
        seq_token_counts=[16, 32],
    )

    assert len(audio_tensors) == 2
    for audio_tensor in audio_tensors:
        assert audio_tensor.shape == (1, 0)
        assert audio_tensor.dtype == torch.float32


def test_code2wav_forward_pads_partial_scheduler_frames_per_request() -> None:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.model_stage = "code2wav"
    captured: dict[str, object] = {}

    def fake_generate_audio(
        codes: torch.Tensor,
        left_context_size: list[int],
        seq_token_counts: list[int],
    ) -> list[torch.Tensor]:
        captured["codes"] = codes
        captured["left_context_size"] = left_context_size
        captured["seq_token_counts"] = seq_token_counts
        return [torch.empty((1, seq_len // 16), dtype=torch.float32) for seq_len in seq_token_counts]

    model.generate_audio = fake_generate_audio

    audio_tensors = Qwen3OmniMoeForConditionalGeneration.forward(
        model,
        input_ids=torch.arange(65, dtype=torch.long),
        positions=torch.empty(0, dtype=torch.long),
        seq_token_counts=[16, 17, 32],
    )

    assert [audio_tensor.shape for audio_tensor in audio_tensors] == [(1, 1), (1, 1), (1, 2)]
    assert captured["seq_token_counts"] == [16, 17, 32]
    assert captured["left_context_size"] == [0, 0, 0]
    codes = captured["codes"]
    assert isinstance(codes, torch.Tensor)
    assert codes.shape == (3, 16, 2)
    padded_second_request = codes[1].reshape(-1)
    torch.testing.assert_close(
        padded_second_request[:17],
        torch.arange(16, 33, dtype=torch.long),
    )
    torch.testing.assert_close(padded_second_request[17:], torch.zeros(15, dtype=torch.long))


def test_code2wav_input_reshape_preserves_aligned_request_boundaries() -> None:
    input_ids = torch.arange(48, dtype=torch.long)

    codes = _reshape_code2wav_input_ids(input_ids, [16, 32])

    expected_first = torch.arange(16, dtype=torch.long).reshape(16, 1)
    expected_second = torch.arange(16, 48, dtype=torch.long).reshape(16, 2)

    assert codes.shape == (2, 16, 2)
    torch.testing.assert_close(codes[0, :, :1], expected_first)
    torch.testing.assert_close(codes[0, :, 1], torch.zeros(16, dtype=torch.long))
    torch.testing.assert_close(codes[1], expected_second)


def test_code2wav_forward_normalizes_left_context_per_request() -> None:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.model_stage = "code2wav"
    captured: dict[str, object] = {}

    def fake_generate_audio(
        codes: torch.Tensor,
        left_context_size: list[int],
        seq_token_counts: list[int],
    ) -> list[torch.Tensor]:
        captured["codes_shape"] = tuple(codes.shape)
        captured["left_context_size"] = left_context_size
        captured["seq_token_counts"] = seq_token_counts
        return [torch.empty((1, seq_len // 16), dtype=torch.float32) for seq_len in seq_token_counts]

    model.generate_audio = fake_generate_audio

    audio_tensors = Qwen3OmniMoeForConditionalGeneration.forward(
        model,
        input_ids=torch.arange(64, dtype=torch.long),
        positions=torch.empty(0, dtype=torch.long),
        runtime_additional_information=[
            {"meta": {"left_context_size": 2}},
            {"meta": {}},
            {"meta": {"left_context_size": 3}},
        ],
        seq_token_counts=[16, 16, 16, 16],
    )

    assert [audio_tensor.shape for audio_tensor in audio_tensors] == [(1, 1)] * 4
    assert captured["codes_shape"] == (4, 16, 1)
    assert captured["seq_token_counts"] == [16, 16, 16, 16]
    assert captured["left_context_size"] == [2, 0, 3, 0]


def test_code2wav_make_output_aligns_zero_token_requests() -> None:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.model_stage = "code2wav"
    model.code2wav_config = SimpleNamespace(sample_rate=24_000)
    audio = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    output = Qwen3OmniMoeForConditionalGeneration.make_omni_output(
        model,
        [audio],
        seq_token_counts=[0, 16],
    )

    assert len(output.multimodal_outputs["model_outputs"]) == 2
    assert output.multimodal_outputs["model_outputs"][0].shape == (1, 0)
    torch.testing.assert_close(output.multimodal_outputs["model_outputs"][1], audio)
    assert [sr.item() for sr in output.multimodal_outputs["sr"]] == [24_000, 24_000]

    output = Qwen3OmniMoeForConditionalGeneration.make_omni_output(
        model,
        [audio],
        seq_token_counts=[16, 0],
    )

    assert len(output.multimodal_outputs["model_outputs"]) == 2
    torch.testing.assert_close(output.multimodal_outputs["model_outputs"][0], audio)
