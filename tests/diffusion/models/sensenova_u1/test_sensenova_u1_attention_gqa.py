# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for SenseNova-U1 attention head handling.

The model used to expand K/V from 8 heads to 32 with ``repeat_interleave``
before calling the attention backend, so the backend's own GQA dispatch never
saw a GQA shape, and it passed an all-zeros mask during single-token decode,
which only served to keep SDPA off its fused kernels. These pin both contracts:
K/V must reach the backend uncompressed, decode must carry no mask, and the
compressed path must compute the same thing the expanded one did.
"""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1Attention,
    SenseNovaU1Model,
    create_block_causal_mask,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

N_HEADS, N_KV_HEADS, HEAD_DIM, SEQ = 32, 8, 8, 6
GROUPS = N_HEADS // N_KV_HEADS


class _Recorder:
    """Stands in for the Attention module and records what it was handed."""

    def __init__(self):
        self.calls: list[tuple[torch.Size, torch.Size, AttentionMetadata | None]] = []

    def __call__(self, query, key, value, attn_metadata: AttentionMetadata | None):
        self.calls.append((query.shape, key.shape, attn_metadata))
        return query


class _Router:
    def __init__(self):
        self.native = _Recorder()
        self.fallback = _Recorder()
        self.sdpa_fallback = type("Fallback", (), {"forward": self.fallback})()
        self.attn_backend = type(
            "Backend",
            (),
            {
                "supports_piecewise_spans": True,
                "supports_attention_mask": classmethod(lambda cls, attention_spec=None: True),
            },
        )()
        self.attn_spec = None

    @property
    def calls(self):
        return self.native.calls

    def __call__(self, *args):
        return self.native(*args)


class _AttnHost:
    """Carries only the attributes ``_run_attn``/``_run_attn_bshd`` touch."""

    num_kv_groups = GROUPS
    _align_mask_dtype = staticmethod(SenseNovaU1Attention._align_mask_dtype)
    _plain_causal_spans = staticmethod(SenseNovaU1Attention._plain_causal_spans)
    _attn_metadata = SenseNovaU1Attention._attn_metadata
    _native_padding_mask_supported = SenseNovaU1Attention._native_padding_mask_supported
    _run_attn = SenseNovaU1Attention._run_attn
    _run_attn_bshd = SenseNovaU1Attention._run_attn_bshd

    def __init__(self):
        self.attn = _Router()


def test_bhsd_path_hands_the_backend_compressed_kv():
    host = _AttnHost()
    q = torch.randn(1, N_HEADS, SEQ, HEAD_DIM)
    k = torch.randn(1, N_KV_HEADS, SEQ, HEAD_DIM)
    host._run_attn(q, k, k.clone(), None)
    _, key_shape, _ = host.attn.calls[-1]
    # [B, S, H, D] after the transpose, so heads sit on dim 2.
    assert key_shape[2] == N_KV_HEADS, f"K reached the backend with {key_shape[2]} heads"


def test_bshd_path_hands_the_backend_compressed_kv():
    host = _AttnHost()
    q = torch.randn(1, SEQ, N_HEADS, HEAD_DIM)
    k = torch.randn(1, SEQ, N_KV_HEADS, HEAD_DIM)
    host._run_attn_bshd(q, k, k.clone(), None)
    _, key_shape, _ = host.attn.calls[-1]
    assert key_shape[2] == N_KV_HEADS, f"K reached the backend with {key_shape[2]} heads"


def test_exact_causal_mask_uses_native_piecewise_metadata():
    host = _AttnHost()
    q = torch.randn(1, N_HEADS, SEQ, HEAD_DIM)
    k = torch.randn(1, N_KV_HEADS, SEQ, HEAD_DIM)
    host._run_attn(q, k, k.clone(), create_block_causal_mask(torch.arange(SEQ)))
    assert len(host.attn.native.calls) == 1
    assert not host.attn.fallback.calls
    metadata = host.attn.native.calls[-1][2]
    assert metadata is not None
    assert metadata.attn_mask is None
    assert metadata.full_attn_spans == [[]]


def test_mixed_block_causal_mask_keeps_sdpa_fallback():
    host = _AttnHost()
    q = torch.randn(1, N_HEADS, SEQ, HEAD_DIM)
    k = torch.randn(1, N_KV_HEADS, SEQ, HEAD_DIM)
    mixed = create_block_causal_mask(torch.tensor([0, 1, 1, 2, 3, 4]))
    host._run_attn(q, k, k.clone(), mixed)
    assert not host.attn.native.calls
    assert len(host.attn.fallback.calls) == 1
    metadata = host.attn.fallback.calls[-1][2]
    assert metadata is not None
    assert metadata.attn_mask is mixed
    assert metadata.full_attn_spans is None


def test_bool_key_padding_mask_uses_native_backend():
    host = _AttnHost()
    q = torch.randn(2, 3, N_HEADS, HEAD_DIM)
    k = torch.randn(2, 7, N_KV_HEADS, HEAD_DIM)
    mask = torch.tensor(
        [
            [True, True, False, False, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )
    host._run_attn_bshd(q, k, k.clone(), mask)
    assert len(host.attn.native.calls) == 1
    assert not host.attn.fallback.calls
    metadata = host.attn.native.calls[-1][2]
    assert metadata is not None
    assert metadata.attn_mask is mask


def test_bool_key_padding_mask_reaches_flash_varlen(monkeypatch):
    from vllm_omni.diffusion.attention.backends.utils import fa

    calls = []

    def fake_varlen(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, kwargs))
        return torch.zeros_like(q)

    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", fake_varlen)

    host = _AttnHost()
    impl = FlashAttentionImpl(
        num_heads=N_HEADS,
        head_size=HEAD_DIM,
        softmax_scale=HEAD_DIM**-0.5,
        num_kv_heads=N_KV_HEADS,
    )
    host.attn.native = impl.forward_cuda
    host.attn.attn_backend = FlashAttentionBackend

    q = torch.randn(2, 3, N_HEADS, HEAD_DIM)
    k = torch.randn(2, 7, N_KV_HEADS, HEAD_DIM)
    mask = torch.tensor(
        [
            [True, True, False, False, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )
    output = host._run_attn_bshd(q, k, k.clone(), mask)

    assert output.shape == q.shape
    assert not host.attn.fallback.calls
    assert len(calls) == 1
    q_shape, k_shape, kwargs = calls[0]
    assert q_shape == (6, N_HEADS, HEAD_DIM)
    assert k_shape == (12, N_KV_HEADS, HEAD_DIM)
    assert kwargs["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert kwargs["cu_seqlens_k"].tolist() == [0, 5, 12]
    assert kwargs["max_seqlen_q"] == 3
    assert kwargs["max_seqlen_k"] == 7


def test_compressed_kv_computes_what_expansion_computed():
    """The backend must return the same tensor either way, or this is a bugfix
    dressed up as a perf change."""
    torch.manual_seed(0)
    q = torch.randn(1, SEQ, N_HEADS, HEAD_DIM)
    k = torch.randn(1, SEQ, N_KV_HEADS, HEAD_DIM)
    v = torch.randn(1, SEQ, N_KV_HEADS, HEAD_DIM)
    impl = SDPAImpl(
        num_heads=N_HEADS,
        head_size=HEAD_DIM,
        softmax_scale=HEAD_DIM**-0.5,
        num_kv_heads=N_KV_HEADS,
    )
    compressed = impl._forward_impl(q, k, v)
    expanded = impl._forward_impl(q, k.repeat_interleave(GROUPS, dim=2), v.repeat_interleave(GROUPS, dim=2))
    torch.testing.assert_close(compressed, expanded, atol=1e-6, rtol=1e-6)


class _MaskProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.masks: list[torch.Tensor | None] = []

    def forward(self, hidden_states, **kwargs):
        self.masks.append(kwargs["attention_mask"]["full_attention"])
        return hidden_states


def _fake_rope(hidden_states, position_ids):
    """The forward builds the three RoPE tables before the layer loop; the probe
    ignores them, so any 2-tuple will do."""
    return hidden_states, hidden_states


class _ModelHost:
    """Carries only the attributes ``SenseNovaU1Model.forward`` touches."""

    forward = SenseNovaU1Model.forward

    def __init__(self, probe):
        self.layers = [probe]
        self.norm = torch.nn.Identity()
        self.norm_mot_gen = torch.nn.Identity()
        self.rotary_emb = _fake_rope
        self.rotary_emb_hw = _fake_rope


def _run_model(seq_len: int):
    probe = _MaskProbe()
    _ModelHost(probe).forward(
        inputs_embeds=torch.zeros(1, seq_len, 4),
        indexes=torch.zeros(3, seq_len, dtype=torch.long),
    )
    return probe.masks[-1]


def test_single_token_decode_carries_no_mask():
    assert _run_model(1) is None, "decode still builds the all-zeros mask"


def test_prefill_still_gets_a_causal_mask():
    mask = _run_model(3)
    assert mask is not None and mask.shape == (1, 1, 3, 3)
    assert mask[0, 0, 0, 0] == 0.0
    assert torch.isinf(mask[0, 0, 0, 1]) and mask[0, 0, 0, 1] < 0
    assert mask[0, 0, 2, 1] == 0.0


@pytest.mark.parametrize("batched", [False, True])
def test_model_preserves_request_specific_rope_positions(batched):
    host = _ModelHost(_MaskProbe())
    positions = []

    def rope(hidden_states, ids):
        positions.append(ids)
        return hidden_states, hidden_states

    host.rotary_emb = host.rotary_emb_hw = rope
    indexes = torch.arange(3 * 2 * 4).reshape(3, 2, 4) if batched else torch.arange(12).reshape(3, 4)
    host.forward(
        inputs_embeds=torch.zeros(2, 4, 8),
        indexes=indexes,
        attention_mask={"full_attention": None},
        image_gen_indicators=torch.ones(2, 4, dtype=torch.bool),
    )
    for axis, actual in enumerate(positions):
        expected = indexes[axis] if batched else indexes[axis].unsqueeze(0)
        torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# The decode path changes which SDPA kernel runs, so it is not bit-identical to
# what it replaced. "Different" would be a regression if the new result were
# worse, so pin the direction against an independent float64 reference. This
# needs a fused GQA kernel, which only exists on CUDA.
# ---------------------------------------------------------------------------


def _float64_attention(q, k, v):
    q64 = q.double()
    k64 = k.double().repeat_interleave(GROUPS, dim=1)
    v64 = v.double().repeat_interleave(GROUPS, dim=1)
    scores = (q64 @ k64.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    return torch.softmax(scores, dim=-1) @ v64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused GQA")
@pytest.mark.cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("kv_len", [271, 512, 2048])
def test_decode_without_the_no_op_mask_is_closer_to_float64(kv_len):
    """The old decode path expanded K/V and passed an all-zeros mask, which kept
    SDPA off its fused kernels. Dropping the mask is only defensible if the
    kernel it unlocks is at least as accurate."""
    dev, dtype = "cuda", torch.bfloat16
    heads, dim = 32, 128
    new_errs: list[float] = []
    old_errs: list[float] = []
    trials = 8
    for t in range(trials):
        torch.manual_seed(2000 + t)
        q = torch.randn(1, heads, 1, dim, device=dev, dtype=dtype)
        k = torch.randn(1, heads // GROUPS, kv_len, dim, device=dev, dtype=dtype)
        v = torch.randn(1, heads // GROUPS, kv_len, dim, device=dev, dtype=dtype)
        ref = _float64_attention(q, k, v)

        zeros_mask = torch.zeros(1, 1, 1, kv_len, device=dev, dtype=dtype)
        old = torch.nn.functional.scaled_dot_product_attention(
            q,
            k.repeat_interleave(GROUPS, dim=1),
            v.repeat_interleave(GROUPS, dim=1),
            attn_mask=zeros_mask,
        )
        new = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        new_errs.append((new.double() - ref).abs().mean().item())
        old_errs.append((old.double() - ref).abs().mean().item())

    # A single trial can tie either way depending on the GPU and the CUDA build,
    # so gate on the aggregate: the maskless path must not be worse on average.
    new_mean = sum(new_errs) / trials
    old_mean = sum(old_errs) / trials
    wins = sum(n <= o for n, o in zip(new_errs, old_errs))
    assert new_mean <= old_mean, (
        f"kv_len={kv_len}: maskless mean error {new_mean:.4e} is worse than masked {old_mean:.4e} "
        f"({wins}/{trials} per-trial wins)"
    )
