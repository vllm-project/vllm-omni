# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only bridge contract tests for FunAudioChat's stage-1 decoder."""

from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.funaudiochat.funaudiochat_code2wav import (
    FunAudioChatCosyVoice3Code2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StubDecoder:
    def __init__(self) -> None:
        self.calls = []
        self.audio = [
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.4]),
            torch.tensor([0.5]),
        ]
        self.cache_states = [{"history": torch.tensor([1])}, {"history": torch.tensor([2])}, None]

    def forward_streaming(self, **kwargs):
        self.calls.append(kwargs)
        return self.audio[len(self.calls) - 1].reshape(1, 1, -1), self.cache_states[len(self.calls) - 1]


def _bridge() -> tuple[FunAudioChatCosyVoice3Code2Wav, _StubDecoder]:
    bridge = FunAudioChatCosyVoice3Code2Wav.__new__(FunAudioChatCosyVoice3Code2Wav)
    nn.Module.__init__(bridge)
    decoder = _StubDecoder()
    bridge.code2wav = decoder
    bridge.config = SimpleNamespace(sample_rate=24000, flow={"input_size": 80})
    bridge._default_speaker_embedding = torch.ones((1, 192))
    bridge._stream_cache_by_req = {}
    bridge._stream_cache_lock = Lock()
    return bridge, decoder


def _payload(codes: list[int], *, req_id: str, left_context: int, finished: bool) -> dict:
    return {
        "codes": {"audio": torch.tensor(codes, dtype=torch.long)},
        "meta": {
            "req_id": [req_id],
            "left_context_size": left_context,
            "stream_finished": torch.tensor(finished),
        },
    }


def test_forward_passes_cumulative_codec_prefix_and_emits_only_decoder_delta() -> None:
    bridge, decoder = _bridge()

    first = bridge.forward(
        torch.tensor([1, 2]),
        model_intermediate_buffer=[_payload([1, 2], req_id="req-a", left_context=0, finished=False)],
        seq_token_counts=[2],
    )
    second = bridge.forward(
        torch.tensor([1, 2, 3, 4]),
        model_intermediate_buffer=[_payload([1, 2, 3, 4], req_id="req-a", left_context=2, finished=False)],
        seq_token_counts=[4],
    )

    assert len(first.multimodal_outputs["audio"]) == 1
    torch.testing.assert_close(first.multimodal_outputs["audio"][0], torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(second.multimodal_outputs["audio"][0], torch.tensor([0.3, 0.4]))
    assert first.multimodal_outputs["sr"][0].item() == 24000
    assert decoder.calls[0]["token"].tolist() == [[1, 2]]
    assert decoder.calls[1]["token"].tolist() == [[1, 2, 3, 4]]
    assert decoder.calls[1]["token_offset_tokens"] == 2
    assert decoder.calls[1]["cache_state"] is decoder.cache_states[0]
    assert decoder.calls[0]["prompt_token"].shape == (1, 0)
    assert decoder.calls[0]["prompt_feat"].shape == (1, 0, 80)
    torch.testing.assert_close(decoder.calls[0]["embedding"], torch.ones((1, 192)))
    assert bridge._stream_cache_by_req["req-a"] is decoder.cache_states[1]

    terminal = bridge.forward(
        torch.tensor([1, 2, 3, 4, 5]),
        model_intermediate_buffer=[_payload([1, 2, 3, 4, 5], req_id="req-a", left_context=4, finished=True)],
        seq_token_counts=[5],
    )

    torch.testing.assert_close(terminal.multimodal_outputs["audio"][0], torch.tensor([0.5]))
    assert decoder.calls[2]["cache_state"] is decoder.cache_states[1]
    assert decoder.calls[2]["finalize"] is True
    assert "req-a" not in bridge._stream_cache_by_req


def test_streaming_cache_is_isolated_and_request_finish_cleans_it() -> None:
    bridge, _decoder = _bridge()
    bridge.forward(
        torch.tensor([1]),
        model_intermediate_buffer=[_payload([1], req_id="req-a", left_context=0, finished=False)],
        seq_token_counts=[1],
    )
    bridge.forward(
        torch.tensor([2]),
        model_intermediate_buffer=[_payload([2], req_id="req-b", left_context=0, finished=False)],
        seq_token_counts=[1],
    )

    assert set(bridge._stream_cache_by_req) == {"req-a", "req-b"}
    bridge.on_requests_finished(["req-a"])
    assert set(bridge._stream_cache_by_req) == {"req-b"}


def test_missing_official_default_speaker_asset_has_actionable_error() -> None:
    bridge, _decoder = _bridge()
    bridge.config = SimpleNamespace(flow={"spk_embed_dim": 192})
    bridge.model_dir = str(Path(__file__).resolve().parent / "missing_funaudiochat_model")

    with pytest.raises(FileNotFoundError, match="official CosyVoice3 default speaker embedding"):
        bridge._load_default_speaker_embedding()


def test_forward_rejects_mismatched_codec_token_lengths() -> None:
    bridge, _decoder = _bridge()

    with pytest.raises(ValueError, match="but input_ids contain 2"):
        bridge.forward(
            torch.tensor([1, 2]),
            model_intermediate_buffer=[_payload([1], req_id="req-a", left_context=0, finished=False)],
            seq_token_counts=[2],
        )


def test_forward_rejects_missing_codec_codes() -> None:
    bridge, _decoder = _bridge()

    with pytest.raises(ValueError, match="missing codes.audio"):
        bridge.forward(
            torch.tensor([1, 2]),
            model_intermediate_buffer=[{"meta": {"req_id": ["req-a"]}}],
            seq_token_counts=[2],
        )


@pytest.mark.parametrize(
    ("input_ids", "counts", "message"),
    [
        (torch.tensor([1, 2, 3]), [1, 1], "sum to 2"),
        (torch.tensor([1, 2]), [-1, 3], "non-negative"),
        (torch.tensor([1, 2]), [2], "Expected 2 codec-token lengths"),
    ],
)
def test_forward_rejects_invalid_sequence_lengths(input_ids, counts, message) -> None:
    bridge, _decoder = _bridge()

    with pytest.raises(ValueError, match=message):
        bridge.forward(
            input_ids,
            model_intermediate_buffer=[None, None],
            seq_token_counts=counts,
        )


def test_forward_rejects_missing_cosvoice_feature_dimension() -> None:
    bridge, _decoder = _bridge()
    bridge.config = SimpleNamespace(sample_rate=24000, flow={})

    with pytest.raises(ValueError, match="missing flow.input_size feature dimension"):
        bridge.forward(torch.tensor([1]), seq_token_counts=[1])


def test_forward_requires_decoder_waveform_tensor() -> None:
    bridge, decoder = _bridge()
    decoder.forward_streaming = lambda **kwargs: ("not-waveform", None)

    with pytest.raises(TypeError, match="must return waveform tensor"):
        bridge.forward(
            torch.tensor([1]),
            model_intermediate_buffer=[_payload([1], req_id="req-a", left_context=0, finished=False)],
            seq_token_counts=[1],
        )

    assert "req-a" not in bridge._stream_cache_by_req
