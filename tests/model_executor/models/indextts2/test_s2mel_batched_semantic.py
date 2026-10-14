# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: two ready Stage-1 requests batched into one IndexTTS2 S2Mel step.

ResidualVQ.vq2emb indexes its input as [num_quantizers, B, T]. Feeding
mel_codes.unsqueeze(1) ([B, 1, T]) only decoded the first request, so a
2-request batch crashed with semantic=(1, T, D) vs latent=(2, T, D).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder import (
    IndexTTS2S2MelDecoder,
)
from vllm_omni.model_executor.models.indextts2.utils.maskgct.quantize import ResidualVQ
from vllm_omni.model_executor.stage_input_processors.indextts2 import (
    _build_s2mel_additional_information,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 16
GPT_DIM = 12
CODEBOOK = 32


class _FakeS2Mel:
    def __init__(self):
        self.gpt_layer = torch.nn.Linear(GPT_DIM, HIDDEN)

    def forward_gpt(self, latent):
        return self.gpt_layer(latent)


def _make_decoder() -> IndexTTS2S2MelDecoder:
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.use_gpt_latent = True
    decoder.semantic_codec_type = "repcodec"
    decoder.config = SimpleNamespace(gpt={"stop_mel_token": 8193})
    decoder.s2mel = _FakeS2Mel()
    return decoder


def _make_codec() -> SimpleNamespace:
    # Same quantizer layout as IndexTTS2's RepCodec (num_quantizers=1).
    quantizer = ResidualVQ(
        input_dim=HIDDEN,
        num_quantizers=1,
        codebook_size=CODEBOOK,
        codebook_dim=8,
        quantizer_dropout=0.0,
    ).eval()
    return SimpleNamespace(quantizer=quantizer)


def _stage1_payload(length: int, seed: int) -> dict:
    """Build a Stage-1 payload the same way the full-payload connector does."""
    gen = torch.Generator().manual_seed(seed)
    mel = torch.randint(0, CODEBOOK, (length,), generator=gen)
    latent = torch.randn(length, GPT_DIM, generator=gen)
    meta = {
        "S_ref": torch.randn(1, 5, HIDDEN, generator=gen),
        "ref_mel": torch.randn(1, 80, 7, generator=gen),
        "style": torch.randn(1, 192, generator=gen),
    }
    return _build_s2mel_additional_information(mel, latent, meta, use_gpt_latent=True, context="test")


@torch.no_grad()
def test_two_ready_stage1_requests_build_per_request_semantic_embedding():
    torch.manual_seed(0)
    decoder = _make_decoder()
    codec = _make_codec()
    infos = [_stage1_payload(9, seed=1), _stage1_payload(6, seed=2)]

    # Both requests' payloads are ready in the same Stage-1 step.
    decoder._validate_payload_policy(infos)
    decoder._validate_payload_contract(infos)
    mel_codes, latent, _, code_lens, *_ = decoder._build_batched_inputs(
        infos, device=torch.device("cpu"), model_dtype=torch.float32
    )
    assert mel_codes.shape == (2, 9)
    assert latent.shape == (2, 9, GPT_DIM)

    batched = decoder._build_semantic_embedding(codec=codec, mel_codes=mel_codes, latent=latent, dtype=torch.float32)
    assert batched.shape == (2, 9, HIDDEN)

    # Each row must equal that request decoded alone (single-request path).
    for row, (info, length) in enumerate(zip(infos, code_lens)):
        single = decoder._build_semantic_embedding(
            codec=codec,
            mel_codes=info["mel_codes"],
            latent=info["latent"],
            dtype=torch.float32,
        )
        assert single.shape == (1, length, HIDDEN)
        torch.testing.assert_close(batched[row, :length], single[0])
