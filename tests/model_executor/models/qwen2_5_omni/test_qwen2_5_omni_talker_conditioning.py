# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.skip(
        reason="Methods _build_thinker_to_talker_latent and _build_talker_decode_reply_cache not yet implemented in Qwen2_5OmniForConditionalGeneration"
    ),
]


def test_build_thinker_to_talker_latent_adds_token_embeddings():
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    embeds = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    latent = model._build_thinker_to_talker_latent(hidden, embeds, input_ids=None)

    assert torch.equal(latent, hidden + embeds)


def test_build_thinker_to_talker_latent_masks_multimodal_token_embeddings():
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.thinker_config = SimpleNamespace(
        audio_token_index=7,
        image_token_index=8,
        video_token_index=9,
    )

    class _FakeThinker:
        @staticmethod
        def embed_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
            values = input_ids.to(torch.float32)
            return torch.stack([values, values + 100.0], dim=-1)

    model.thinker = _FakeThinker()
    hidden = torch.zeros((4, 2), dtype=torch.float32)
    input_ids = torch.tensor([1, 7, 2, 8], dtype=torch.long)
    multimodal_embeds = torch.full((4, 2), 999.0, dtype=torch.float32)

    latent = model._build_thinker_to_talker_latent(hidden, multimodal_embeds, input_ids)

    assert torch.equal(
        latent,
        torch.tensor(
            [
                [1.0, 101.0],
                [0.0, 0.0],
                [2.0, 102.0],
                [0.0, 0.0],
            ],
        ),
    )


def test_build_talker_decode_reply_cache_appends_text_eos_pad():
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    model.embed_text_eos_token = torch.tensor([[100.0, 101.0]])
    model.embed_text_pad_token = torch.tensor([[200.0, 201.0]])
    thinker_result = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    reply = model._build_talker_decode_reply_cache(
        thinker_result,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        reply,
        torch.tensor(
            [
                [2.0, 2.0],
                [3.0, 3.0],
                [100.0, 101.0],
                [200.0, 201.0],
            ]
        ),
    )


def test_talker_decode_keeps_final_reply_pad_vector():
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model = nn.Linear(2, 2)

    class _FakeTalker:
        @staticmethod
        def embed_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros((input_ids.numel(), 2), dtype=torch.float32)

    model.talker = _FakeTalker()
    input_ids = torch.tensor([123], dtype=torch.long)
    input_embeds = torch.zeros((1, 2), dtype=torch.float32)
    final_pad = torch.tensor([[5.0, 6.0]], dtype=torch.float32)

    _, out_embeds, update = model.thinker_to_talker_decode_one_step(
        input_ids,
        input_embeds,
        {"embed": {"thinker_reply": final_pad}},
    )

    torch.testing.assert_close(out_embeds, final_pad)
    assert update == {}


class _FakeToken2Wav(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = []
        self.token2wav = SimpleNamespace(
            factor=SimpleNamespace(factor=2),
            code2wav_dit_model=SimpleNamespace(mel_dim=3),
        )

    def forward(
        self,
        code: torch.Tensor,
        *,
        conditioning: torch.Tensor,
        reference_mel: torch.Tensor,
        num_steps: int = 10,
        **_: object,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "code": code.clone(),
                "conditioning": conditioning.clone(),
                "reference_mel": reference_mel.clone(),
                "num_steps": num_steps,
            }
        )
        return torch.arange(6, dtype=torch.float32)

    def process_chunk(self, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("_codec_to_audio should not truncate codec through process_chunk")


def test_codec_to_audio_uses_full_token2wav_forward():
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    nn.Module.__init__(model)
    token2wav = _FakeToken2Wav()
    cond = torch.ones((1, 4), dtype=torch.float32)
    ref_mel = torch.full((1, 300, 3), 2.0, dtype=torch.float32)
    model.token2wav = token2wav
    model.token2wav_config = SimpleNamespace(
        dit_config=SimpleNamespace(enc_emb_dim=4, mel_dim=3),
    )
    model._token2wav_conds = {"Chelsie": cond}
    model._token2wav_ref_mels = {"Chelsie": ref_mel}

    out = model._codec_to_audio(torch.arange(80, dtype=torch.long), "Chelsie")

    assert len(token2wav.calls) == 1
    call = token2wav.calls[0]
    torch.testing.assert_close(call["code"], torch.arange(80, dtype=torch.long).unsqueeze(0))
    torch.testing.assert_close(call["conditioning"], cond)
    torch.testing.assert_close(call["reference_mel"], ref_mel)
    assert call["num_steps"] == 10
    torch.testing.assert_close(out, torch.arange(6, dtype=torch.float32))
