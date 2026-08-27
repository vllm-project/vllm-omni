# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.neutts_air.neutts_air_code2wav import (
    NEUTTS_HOP_LENGTH,
    NEUTTS_SAMPLE_RATE,
    NeuTTSAirCode2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeNeuCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.received_codes: list[torch.Tensor] = []
        self.returned_audio: list[torch.Tensor] = []

    def decode_code(self, codes: torch.Tensor) -> torch.Tensor:
        call_index = len(self.received_codes)
        self.received_codes.append(codes.clone())
        values = codes[0, 0].to(torch.float32) + call_index * 0.25
        waveform = torch.repeat_interleave(values, NEUTTS_HOP_LENGTH)
        audio = waveform.reshape(1, 1, -1)
        self.returned_audio.append(audio.clone())
        return audio


def _make_decoder() -> tuple[NeuTTSAirCode2Wav, FakeNeuCodec]:
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )
    decoder = NeuTTSAirCode2Wav(vllm_config=config)
    codec = FakeNeuCodec()
    decoder._codec = codec
    return decoder, codec


def _stream_info(
    request_id: str,
    *,
    left_context: int,
    right_holdback: int,
    processed_frames: int,
    finished: bool = False,
) -> list[dict]:
    return [
        {
            "meta": {
                "req_id": [request_id],
                "left_context_size": left_context,
                "right_holdback_size": right_holdback,
                "num_processed_tokens": processed_frames,
                "codec_streaming": True,
                "codec_chunk_frames": 25,
                "stream_finished": torch.tensor(finished),
            }
        }
    ]


def _audio(output) -> torch.Tensor:
    assert output.multimodal_outputs is not None
    return output.multimodal_outputs["model_outputs"][0]


def _reference_overlap_add(
    frames: list[torch.Tensor],
    stride: int,
) -> torch.Tensor:
    arrays = [frame.detach().cpu().numpy() for frame in frames]
    total_size = max(stride * index + frame.shape[-1] for index, frame in enumerate(arrays))
    output = np.zeros(total_size, dtype=np.float32)
    sum_weight = np.zeros(total_size, dtype=np.float32)
    for index, frame in enumerate(arrays):
        positions = np.linspace(
            0,
            1,
            frame.shape[-1] + 2,
            dtype=np.float32,
        )[1:-1]
        weight = 0.5 - np.abs(positions - 0.5)
        offset = index * stride
        output[offset : offset + frame.shape[-1]] += weight * frame
        sum_weight[offset : offset + frame.shape[-1]] += weight
    return torch.from_numpy(output / sum_weight)


def test_forward_splits_requests_before_neucodec_decode():
    decoder, codec = _make_decoder()

    output = decoder(
        input_ids=torch.tensor([0, 29, 8, 15, 20]),
        seq_token_counts=[2, 3],
    )

    assert len(codec.received_codes) == 2
    assert codec.received_codes[0].tolist() == [[[0, 29]]]
    assert codec.received_codes[1].tolist() == [[[8, 15, 20]]]

    assert output.multimodal_outputs is not None
    audios = output.multimodal_outputs["model_outputs"]
    sample_rates = output.multimodal_outputs["sr"]
    assert [audio.numel() for audio in audios] == [960, 1440]
    assert [int(sample_rate.item()) for sample_rate in sample_rates] == [
        NEUTTS_SAMPLE_RATE,
        NEUTTS_SAMPLE_RATE,
    ]
    assert all(audio.dtype == torch.float32 for audio in audios)
    assert all(audio.device.type == "cpu" for audio in audios)
    assert all(bool(torch.isfinite(audio).all()) for audio in audios)


def test_forward_empty_input_does_not_load_neucodec():
    decoder, codec = _make_decoder()

    output = decoder(input_ids=torch.empty((0,), dtype=torch.long))

    assert codec.received_codes == []
    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs["model_outputs"][0].numel() == 0


def test_streaming_30_55_63_chunks_match_official_overlap_add():
    decoder, codec = _make_decoder()
    stride = 25 * NEUTTS_HOP_LENGTH

    first = _audio(
        decoder(
            input_ids=torch.arange(81),
            runtime_additional_information=_stream_info(
                "rid-flow",
                left_context=51,
                right_holdback=3,
                processed_frames=25,
            ),
        )
    )
    second = _audio(
        decoder(
            input_ids=torch.arange(81) + 25,
            runtime_additional_information=_stream_info(
                "rid-flow",
                left_context=51,
                right_holdback=3,
                processed_frames=25,
            ),
        )
    )
    final = _audio(
        decoder(
            input_ids=torch.arange(64) + 50,
            runtime_additional_information=_stream_info(
                "rid-flow",
                left_context=50,
                right_holdback=0,
                processed_frames=13,
                finished=True,
            ),
        )
    )

    cropped_frames = [
        codec.returned_audio[0][0, 0][51 * NEUTTS_HOP_LENGTH : -3 * NEUTTS_HOP_LENGTH],
        codec.returned_audio[1][0, 0][51 * NEUTTS_HOP_LENGTH : -3 * NEUTTS_HOP_LENGTH],
        codec.returned_audio[2][0, 0][50 * NEUTTS_HOP_LENGTH :],
    ]
    expected = _reference_overlap_add(cropped_frames, stride)

    assert [first.numel(), second.numel(), final.numel()] == [
        stride,
        stride,
        expected.numel() - 2 * stride,
    ]
    assert torch.allclose(
        torch.cat([first, second, final]),
        expected,
        atol=1e-5,
        rtol=1e-5,
    )
    assert "rid-flow" not in decoder._stream_states


def test_streaming_context_only_terminal_flushes_tail_without_decoding():
    decoder, codec = _make_decoder()
    stride = 25 * NEUTTS_HOP_LENGTH

    first = _audio(
        decoder(
            input_ids=torch.arange(81),
            runtime_additional_information=_stream_info(
                "rid-eof",
                left_context=51,
                right_holdback=3,
                processed_frames=25,
            ),
        )
    )
    terminal = _audio(
        decoder(
            input_ids=torch.tensor([80]),
            runtime_additional_information=_stream_info(
                "rid-eof",
                left_context=1,
                right_holdback=0,
                processed_frames=0,
                finished=True,
            ),
        )
    )

    cropped = codec.returned_audio[0][0, 0][51 * NEUTTS_HOP_LENGTH : -3 * NEUTTS_HOP_LENGTH]
    assert len(codec.received_codes) == 1
    assert first.numel() == stride
    assert terminal.numel() == cropped.numel() - stride
    assert torch.allclose(torch.cat([first, terminal]), cropped)
    assert "rid-eof" not in decoder._stream_states


def test_streaming_state_is_isolated_by_request_id():
    decoder, _ = _make_decoder()

    for request_id in ["rid-a", "rid-b"]:
        decoder(
            input_ids=torch.arange(81),
            runtime_additional_information=_stream_info(
                request_id,
                left_context=51,
                right_holdback=3,
                processed_frames=25,
            ),
        )

    assert set(decoder._stream_states) == {"rid-a", "rid-b"}

    decoder(
        input_ids=torch.tensor([80]),
        runtime_additional_information=_stream_info(
            "rid-a",
            left_context=1,
            right_holdback=0,
            processed_frames=0,
            finished=True,
        ),
    )

    assert set(decoder._stream_states) == {"rid-b"}


def test_on_requests_finished_clears_only_target_stream_state():
    decoder, _ = _make_decoder()

    for request_id in ["rid-a", "rid-b"]:
        decoder(
            input_ids=torch.arange(81),
            runtime_additional_information=_stream_info(
                request_id,
                left_context=51,
                right_holdback=3,
                processed_frames=25,
            ),
        )

    assert set(decoder._stream_states) == {"rid-a", "rid-b"}

    decoder.on_requests_finished(["rid-a", "unknown"])
    assert set(decoder._stream_states) == {"rid-b"}

    decoder.on_requests_finished(["rid-b"])
    assert decoder._stream_states == {}
