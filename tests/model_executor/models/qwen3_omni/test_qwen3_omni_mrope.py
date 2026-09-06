# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_non_thinker_mrope_positions_do_not_require_multimodal_kwargs():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.model_stage = "talker"

    positions, delta = model.get_mrope_input_positions([1, 2, 3])

    assert delta == 0
    assert torch.equal(
        positions,
        torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ]
        ),
    )


class _DummyTalker:
    def text_projection(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = input_ids.shape[0]
        return torch.ones((rows, 4), dtype=torch.bfloat16, device=input_ids.device)


def test_talker_assistant_parts_pad_short_assistant_segment_to_bootstrap_shape():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.talker = _DummyTalker()
    model.config = SimpleNamespace(
        tts_pad_token_id=99,
        talker_config=SimpleNamespace(
            codec_nothink_id=1,
            codec_think_bos_id=2,
            codec_think_eos_id=3,
            codec_pad_id=4,
            codec_bos_id=5,
            text_config=SimpleNamespace(hidden_size=4),
        ),
    )

    thinker_embed = torch.zeros((3, 4), dtype=torch.bfloat16)
    tts_pad_embed = torch.zeros((1, 4), dtype=torch.bfloat16)
    tts_bos_embed = torch.zeros((1, 4), dtype=torch.bfloat16)
    tts_eos_embed = torch.full((1, 4), 2, dtype=torch.bfloat16)

    input_embeds, input_ids, trailing_text_hidden = model._get_talker_assistant_parts(
        im_start_index=3,
        segment_end_index=3,
        speaker_id=6,
        thinker_embed=thinker_embed,
        tts_pad_embed=tts_pad_embed,
        tts_bos_embed=tts_bos_embed,
        tts_eos_embed=tts_eos_embed,
    )

    assert input_embeds.shape == (9, 4)
    assert input_ids.tolist() == [99] * 9
    assert torch.equal(trailing_text_hidden, tts_eos_embed)


def test_code2wav_uses_native_model_intermediate_buffer_codec_payload():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.model_stage = "code2wav"
    captured = {}

    def fake_generate_audio(codes, left_context_size=None, seq_token_counts=None):
        captured["codes"] = codes.detach().clone()
        captured["left_context_size"] = left_context_size
        captured["seq_token_counts"] = seq_token_counts
        return [torch.tensor([1.0])]

    model.generate_audio = fake_generate_audio

    payload_codes = torch.arange(32, dtype=torch.long)
    audio_tensors = model.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        positions=None,
        model_intermediate_buffer=[
            {
                "codes": {"audio": payload_codes},
                "meta": {"left_context_size": 2},
            }
        ],
        seq_token_counts=[1],
    )

    assert len(audio_tensors) == 1
    assert captured["codes"].shape == (1, 16, 2)
    assert torch.equal(captured["codes"][0], payload_codes.reshape(16, 2))
    assert captured["left_context_size"] == [2]
    assert captured["seq_token_counts"] == [32]
