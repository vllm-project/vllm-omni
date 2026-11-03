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

The two signatures disagree on position but share the parameter names, so the
call has to bind by name to work with either of them. The other packed Ming
paths (``ming3_tts/packed_qwen2.py`` and ``ming3_audio/audio_encoder_graphs.py``)
pass the lengths in vLLM's order.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.common.ming import whisper_encoder as whisper_encoder_module
from vllm_omni.model_executor.models.common.ming.whisper_encoder import MultiHeadAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fake_flash_attn(order: str, captured: dict[str, Any]):
    """A recorder with the signature of the given FlashAttention variant.

    ``vllm`` is vLLM's dispatcher, ``classic`` is the FlashAttention/aiter
    signature. Both accept the same parameter names in a different order.
    """

    def record(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: object,
        cu_seqlens_k: object,
        max_seqlen_q: object,
        max_seqlen_k: object,
        **kwargs: object,
    ) -> torch.Tensor:
        captured.update(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            kwargs=kwargs,
        )
        return torch.zeros_like(q)

    if order == "vllm":

        def fake(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, **kwargs):
            return record(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)

    else:

        def fake(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
            return record(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)

    return fake


@pytest.mark.parametrize("order", ["vllm", "classic"])
def test_packed_attention_binds_arguments_by_name(monkeypatch: pytest.MonkeyPatch, order: str) -> None:
    """The lengths and the bounds must reach FlashAttention in the right slots.

    The same call has to work with either signature, because the arguments are
    bound by name rather than by position.
    """
    captured: dict[str, Any] = {}
    monkeypatch.setattr(whisper_encoder_module, "flash_attn_varlen_func", _fake_flash_attn(order, captured))

    attn = MultiHeadAttention(n_state=32, n_head=4, use_flash_attn=False)
    # The packed path only calls FlashAttention for half precision inputs.
    attn.use_flash_attn = True
    tokens = 6
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32)

    attn(torch.randn(tokens, 32, dtype=torch.bfloat16), cu_seqlens=cu_seqlens)

    assert isinstance(captured["cu_seqlens_q"], torch.Tensor)
    assert isinstance(captured["cu_seqlens_k"], torch.Tensor)
    assert isinstance(captured["max_seqlen_q"], int)
    assert isinstance(captured["max_seqlen_k"], int)
    assert captured["cu_seqlens_q"] is cu_seqlens
    assert captured["cu_seqlens_k"] is cu_seqlens
    # The bounds describe the packed sequences, not a padded length.
    assert captured["max_seqlen_q"] == tokens
    assert captured["max_seqlen_k"] == tokens


@pytest.mark.parametrize(
    ("cu_seqlens", "slices"),
    [
        # One sequence: the padding mask is all zeros and the per-sequence
        # scatter is trivial.
        (torch.tensor([0, 6], dtype=torch.int32), [(0, 6)]),
        # Two sequences of unequal length: the padding mask and the
        # cross-sequence boundaries have to be right as well.
        (torch.tensor([0, 2, 6], dtype=torch.int32), [(0, 2), (2, 6)]),
    ],
    ids=["one-sequence", "two-sequences"],
)
def test_packed_attention_falls_back_without_a_varlen_callable(
    monkeypatch: pytest.MonkeyPatch,
    cu_seqlens: torch.Tensor,
    slices: list[tuple[int, int]],
) -> None:
    """``HAS_FLASH_ATTN`` can be true while no varlen callable exists.

    The flag is also set when only the optional dense ``fa3-fwd`` binding was
    found, so the packed path has to fall back to the PyTorch implementation
    instead of calling ``None``.
    """
    monkeypatch.setattr(whisper_encoder_module, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(whisper_encoder_module, "flash_attn_varlen_func", None)

    attn = MultiHeadAttention(n_state=32, n_head=4, use_flash_attn=True)

    assert attn.use_flash_attn is False

    tokens = int(cu_seqlens[-1])
    x = torch.randn(tokens, 32, dtype=torch.bfloat16)

    output = attn(x, cu_seqlens=cu_seqlens)

    assert output.shape == (tokens, 32)

    # The fallback has to compute the right attention, not merely run. Each
    # packed sequence is a plain self-attention over its own frames, so the
    # dense implementation is an independent reference for it.
    for start, end in slices:
        reference = attn._forward_dense(x[start:end].float().unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(output[start:end].float(), reference, rtol=5e-2, atol=5e-2)
