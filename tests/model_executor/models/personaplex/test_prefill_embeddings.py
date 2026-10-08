# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import PersonaPlexConfig
from vllm_omni.model_executor.models.personaplex.personaplex_embeddings import PersonaPlexInputEmbeddings
from vllm_omni.model_executor.models.personaplex.personaplex_talker import PersonaPlexTalkerForConditionalGeneration

pytestmark = pytest.mark.core_model


def _make_talker(
    device: torch.device, dtype: torch.dtype, config: PersonaPlexConfig | None = None
) -> PersonaPlexTalkerForConditionalGeneration:
    talker = PersonaPlexTalkerForConditionalGeneration.__new__(PersonaPlexTalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.config = config or PersonaPlexConfig(
        temporal_config={"hidden_size": 32},
        text_vocab_size=31,
        text_embedding_rows=32,
        audio_vocab_size=16,
    )
    talker.input_embeddings = PersonaPlexInputEmbeddings(talker.config)
    generator = torch.Generator().manual_seed(7389)
    with torch.no_grad():
        for parameter in talker.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return talker.to(device=device, dtype=dtype).eval()


def _reference_prefill(
    talker: PersonaPlexTalkerForConditionalGeneration,
    prefill_text: torch.Tensor,
    offset: int,
    span: int,
    device: torch.device,
    silence: torch.Tensor | None = None,
    user_sine: torch.Tensor | None = None,
) -> torch.Tensor:
    """Original loop from main@13b85c56."""
    n_q = talker.config.num_audio_codebooks
    n_user = n_q // 2
    sil = silence.reshape(-1).to(device) if isinstance(silence, torch.Tensor) and silence.numel() >= n_user else None
    user = (
        user_sine.reshape(-1).to(device) if isinstance(user_sine, torch.Tensor) and user_sine.numel() >= n_user else sil
    )
    rows = []
    for i in range(span):
        pos = offset + i
        text_tok = int(prefill_text[pos].item()) if pos < prefill_text.numel() else 3
        stack = torch.zeros((1, 1 + n_q, 1), dtype=torch.long, device=device)
        stack[:, 0] = text_tok
        if sil is not None:
            stack[0, 1 : 1 + n_user, 0] = sil[:n_user]
        if user is not None:
            stack[0, 1 + n_user : 1 + 2 * n_user, 0] = user[:n_user]
        rows.append(talker.input_embeddings(stack).reshape(1, -1))
    return torch.cat(rows, dim=0)


@pytest.fixture(params=[pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch and a visible GPU are required")
    return torch.device(request.param)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("length", "offset", "span"),
    [(5, 0, 1), (5, 1, 3), (5, 3, 4), (5, 5, 3), (5, 8, 2), (0, 0, 3), (67, 3, 64)],
    ids=["single", "offset", "partial-pad", "at-end", "past-end", "empty-text", "long"],
)
@pytest.mark.parametrize(
    ("silence_size", "user_size"),
    [(None, None), (8, None), (8, 8), (None, 8), (7, 7), (8, 7), (10, 10)],
    ids=["absent", "silence-fallback", "both", "user-only", "short", "short-user", "long-codes"],
)
@torch.inference_mode()
def test_prefill_matches_loop_bitwise(device, dtype, length, offset, span, silence_size, user_size):
    talker = _make_talker(device, dtype)
    # Strided inputs include special token rows.
    text = (torch.arange(2 * length, device=device) % 33 - 1)[::2]
    silence = None if silence_size is None else ((torch.arange(2 * silence_size, device=device) - 1) % 18 - 1)[::2]
    user = None if user_size is None else (torch.arange(2 * user_size, device=device) % 17)[::2]
    originals = [value.clone() for value in (text, silence, user) if value is not None]
    expected = _reference_prefill(talker, text, offset, span, device, silence, user)
    actual = talker._build_prefill_embed(text, offset, span, device, silence, user)
    assert actual.shape == (span, 32)
    assert actual.dtype == dtype
    assert actual.device == expected.device
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for value, original in zip((v for v in (text, silence, user) if v is not None), originals):
        assert torch.equal(value, original)


@torch.inference_mode()
def test_prefill_manual_layout_and_padding(device):
    talker = _make_talker(device, torch.float32)
    # Encode table identity and token ID in the weights.
    for table_index, table in enumerate([talker.input_embeddings.text_emb, *talker.input_embeddings.audio_emb]):
        values = torch.arange(table.num_embeddings, device=device) + 100 * table_index
        table.weight.copy_(values[:, None].expand_as(table.weight))
    actual = talker._build_prefill_embed(
        torch.tensor([10, 20, 30], device=device),
        1,
        4,
        device,
        torch.arange(1, 9, device=device),
        torch.arange(9, 17, device=device),
    )
    # sum(100 * k + k for k in 1..16) = 13736.
    expected = torch.tensor([13756, 13766, 13739, 13739], dtype=torch.float32, device=device)
    assert torch.equal(actual, expected[:, None].expand(-1, 32))


@torch.inference_mode()
def test_prefill_joins_chunked_slices(device):
    talker = _make_talker(device, torch.bfloat16)
    text = torch.arange(23, device=device)
    silence = torch.arange(8, device=device)
    full = talker._build_prefill_embed(text, 0, 27, device, silence)
    pieces = [
        talker._build_prefill_embed(text, offset, span, device, silence) for offset, span in [(0, 7), (7, 13), (20, 7)]
    ]
    assert torch.equal(full.view(torch.uint8), torch.cat(pieces).view(torch.uint8))


@torch.inference_mode()
def test_prefill_uses_one_embedding_call_without_scalar_reads(device, monkeypatch):
    talker = _make_talker(device, torch.float32)
    text = torch.arange(12, device=device)
    shapes = []
    handle = talker.input_embeddings.register_forward_pre_hook(lambda _module, args: shapes.append(args[0].shape))

    def reject_scalar_read(_self):
        raise AssertionError("prefill must not read device scalars into Python")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", reject_scalar_read)
            talker._build_prefill_embed(text, 0, 12, device)
    finally:
        handle.remove()
    assert shapes == [(1, 17, 12)]
