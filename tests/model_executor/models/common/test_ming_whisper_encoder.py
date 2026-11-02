# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The packed Ming Whisper attention must bind FlashAttention's arguments by name.

vLLM's ``flash_attn_varlen_func`` takes ``max_seqlen_q`` before ``cu_seqlens_q``
(and the same for the K side), while the classic FlashAttention signature takes
the cumulative lengths first. The packed Ming path called it positionally in the
classic order, so ``cu_seqlens_k`` received the integer ``max_seqlen`` and the
kernel rejected the call:

    RuntimeError: _vllm_fa2_C::varlen_fwd() Expected a value of type 'Tensor'
    for argument 'cu_seqlens_k' but instead found type 'int'.

The other packed Ming paths (``ming3_tts/packed_qwen2.py`` and
``ming3_audio/audio_encoder_graphs.py``) pass the lengths in vLLM's order.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.common.ming import whisper_encoder as whisper_encoder_module
from vllm_omni.model_executor.models.common.ming.whisper_encoder import MultiHeadAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _packed_attention(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Run one packed attention call and return the captured FA arguments."""
    captured: dict[str, object] = {}

    def fake_flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        max_seqlen_q: object,
        cu_seqlens_q: object,
        max_seqlen_k: object,
        cu_seqlens_k: object,
        **kwargs: object,
    ) -> torch.Tensor:
        captured.update(
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            kwargs=kwargs,
        )
        return torch.zeros_like(q)

    monkeypatch.setattr(whisper_encoder_module, "flash_attn_varlen_func", fake_flash_attn_varlen_func)

    attn = MultiHeadAttention(n_state=32, n_head=4, use_flash_attn=False)
    # The packed path only calls FlashAttention for half precision inputs.
    attn.use_flash_attn = True
    tokens = 6
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32)
    attn(torch.randn(tokens, 32, dtype=torch.bfloat16), cu_seqlens=cu_seqlens)
    return captured


def test_packed_attention_binds_cu_seqlens_to_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cu_seqlens_q``/``cu_seqlens_k`` must reach FlashAttention as tensors."""
    captured = _packed_attention(monkeypatch)

    assert isinstance(captured["cu_seqlens_q"], torch.Tensor)
    assert isinstance(captured["cu_seqlens_k"], torch.Tensor)


def test_packed_attention_binds_max_seqlens_to_ints(monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_seqlen_q``/``max_seqlen_k`` must reach FlashAttention as integers."""
    captured = _packed_attention(monkeypatch)

    assert isinstance(captured["max_seqlen_q"], int)
    assert isinstance(captured["max_seqlen_k"], int)


def test_packed_attention_passes_the_packed_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bounds describe the packed sequences, not the padded length."""
    captured = _packed_attention(monkeypatch)

    assert captured["max_seqlen_q"] == 6
    assert captured["max_seqlen_k"] == 6
    assert isinstance(captured["cu_seqlens_q"], torch.Tensor)
    assert captured["cu_seqlens_q"].tolist() == [0, 6]
    assert captured["cu_seqlens_k"].tolist() == [0, 6]
