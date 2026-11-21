# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice packed attention when the selected backend has no packed varlen.

On sm_12x Blackwell the default diffusion backend (cuDNN) cannot run
multi-document packed varlen, so OmniVoice used to fall back to SDPA with a
dense block mask. When vLLM's FA2 varlen kernel is available, the layer runs
it over the same ``cu_seqlens`` instead. The test forces that branch on any
FA2-capable GPU and compares it with the SDPA block-mask fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
    OmniVoiceGenerator,
    _vllm_fa2_varlen,
)
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DEVICE = torch.device("cuda:0")
NUM_CB = 8


def _generator() -> OmniVoiceGenerator:
    torch.manual_seed(0)
    config = OmniVoiceConfig(
        audio_vocab_size=1025,
        num_audio_codebook=NUM_CB,
        llm_config={
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 512,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "head_dim": 128,
        },
        enable_cuda_graph=False,
    )
    return OmniVoiceGenerator(config, SimpleNamespace(max_num_seqs=4)).to(DEVICE, torch.bfloat16).eval()


def _force_attention(gen: OmniVoiceGenerator, fa2) -> None:
    """Route every layer through the FA2 branch (fa2 set) or the SDPA block-mask branch (None)."""
    for layer in gen.layers:
        layer.self_attn.use_packed_varlen = False
        layer.self_attn.fa2_varlen = fa2


def _step(gen: OmniVoiceGenerator, ids, mask, cu_seqs, *, needs_sdpa_mask: bool) -> torch.Tensor:
    seq_len = ids.shape[0]
    rope = gen._rope_table_for(seq_len, ids.device, gen.model_dtype)
    embeds = gen._prepare_embeddings(ids, mask)
    gen._needs_sdpa_mask = needs_sdpa_mask
    hidden = gen._transformer_forward(embeds, cu_seqs, max_seqlen=seq_len, rope_table=rope)
    return gen._get_logits(hidden)


@pytest.mark.skipif(_vllm_fa2_varlen() is None, reason="vLLM FA2 varlen unavailable on this device")
def test_fa2_branch_matches_sdpa_block_mask() -> None:
    """Two requests (cond/uncond each) plus a padding tail keep their boundaries under FA2."""
    gen = _generator()
    seq_len = 160
    ids = torch.randint(0, 100, (seq_len, NUM_CB), device=DEVICE)
    mask = torch.ones(seq_len, dtype=torch.bool, device=DEVICE)
    cu_seqs = torch.tensor([0, 40, 70, 115, 150, seq_len], dtype=torch.int32, device=DEVICE)

    with torch.no_grad():
        _force_attention(gen, None)
        reference = _step(gen, ids, mask, cu_seqs, needs_sdpa_mask=True).float()
        _force_attention(gen, _vllm_fa2_varlen())
        fa2 = _step(gen, ids, mask, cu_seqs, needs_sdpa_mask=False).float()

    # Both branches run attention in bf16 and differ only in kernel accumulation order.
    torch.testing.assert_close(fa2[:, :150], reference[:, :150], atol=2e-2, rtol=2e-2)

