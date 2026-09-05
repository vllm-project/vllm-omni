# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MammothModa2 DiT self-attention runs through the shared Omni attention layer.

``_reference_attention`` is the arithmetic the processor used before it was
wired to the shared layer: SDPA over K/V replicated to the query head count,
padded rows zeroed. Both sides read the same module weights, so any drift in
the projections, the QK RMSNorm, the interleaved real RoPE, the softmax scale
or the padding mask shows up as a mismatch here.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import (
    TransformerBlock,
    apply_real_rotary_emb,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Small stand-in for the real 2520 / 21 / 7 (head_dim 120) layout.
DIM, HEADS = 48, 6
SEQ, BATCH = 37, 2

# The shared layer resolves its backend from the diffusion config at construction;
# without one it takes the platform default, which on a CUDA host is FLASH_ATTN
# and cannot run CPU tensors. Pin SDPA the way a deployment would.
_SDPA_CONFIG = OmniDiffusionConfig(diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}})


def _block(kv_heads: int, modulation: bool = True) -> TransformerBlock:
    torch.manual_seed(0)
    with set_current_diffusion_config(_SDPA_CONFIG):
        block = TransformerBlock(
            DIM,
            HEADS,
            kv_heads,
            multiple_of=8,
            ffn_dim_multiplier=1.0,
            norm_eps=1e-5,
            modulation=modulation,
        )
    return block.eval()


def _inputs(head_dim: int):
    torch.manual_seed(1)
    hidden = torch.randn(BATCH, SEQ, DIM)
    mask = torch.ones(BATCH, SEQ, dtype=torch.bool)
    mask[0, SEQ - 9 :] = False
    mask[1, SEQ - 2 :] = False
    angles = torch.rand(1, SEQ, head_dim) * 6.283
    rotary = (angles.cos(), angles.sin())
    temb = torch.randn(BATCH, min(DIM, 1024))
    return hidden, mask, rotary, temb


def _reference_attention(attn, hidden, mask, rotary):
    """The pre-change processor, kept verbatim as the oracle."""
    batch, seq, _ = hidden.shape
    query, key, value = attn.to_q(hidden), attn.to_k(hidden), attn.to_v(hidden)
    head_dim = query.shape[-1] // attn.heads
    kv_heads = key.shape[-1] // head_dim
    query = attn.norm_q(query.view(batch, seq, attn.heads, head_dim))
    key = attn.norm_k(key.view(batch, seq, kv_heads, head_dim))
    value = value.view(batch, seq, kv_heads, head_dim)
    query = apply_real_rotary_emb(query, rotary[0], rotary[1])
    key = apply_real_rotary_emb(key, rotary[0], rotary[1])
    query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    if kv_heads < attn.heads:
        key = key.repeat_interleave(attn.heads // kv_heads, dim=1)
        value = value.repeat_interleave(attn.heads // kv_heads, dim=1)
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask.view(batch, 1, 1, seq), scale=attn.scale)
    out = (out * mask[:, None, :, None]).transpose(1, 2).reshape(batch, seq, attn.heads * head_dim)
    return attn.to_out[1](attn.to_out[0](out))


@pytest.mark.parametrize("kv_heads", [2, HEADS], ids=["gqa_3to1", "mha"])
def test_processor_matches_previous_arithmetic(kv_heads):
    block = _block(kv_heads)
    hidden, mask, rotary, _ = _inputs(block.head_dim)
    with torch.no_grad():
        got = block.attn(
            hidden_states=hidden, encoder_hidden_states=hidden, attention_mask=mask, image_rotary_emb=rotary
        )
        want = _reference_attention(block.attn, hidden, mask, rotary)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(got[~mask]) == 0, "padded rows must stay zero"


def test_no_mask_equals_all_valid_mask():
    block = _block(2)
    hidden, _, rotary, _ = _inputs(block.head_dim)
    full = torch.ones(BATCH, SEQ, dtype=torch.bool)
    with torch.no_grad():
        unmasked = block.attn(
            hidden_states=hidden, encoder_hidden_states=hidden, attention_mask=None, image_rotary_emb=rotary
        )
        masked = block.attn(
            hidden_states=hidden, encoder_hidden_states=hidden, attention_mask=full, image_rotary_emb=rotary
        )
    torch.testing.assert_close(unmasked, masked, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("modulation", [True, False], ids=["noise_block", "context_refiner"])
def test_block_forward_hands_native_kv_heads_to_shared_layer(monkeypatch, modulation):
    """Through the block's real forward, on both of its branches (the modulated
    noise/refiner blocks and the unmodulated context refiner): the shared layer
    is what runs, and it receives K/V with their native head count -- no
    replication before the call."""
    block = _block(2, modulation=modulation)
    hidden, mask, rotary, temb = _inputs(block.head_dim)
    if not modulation:
        temb = None
    seen = []
    original = block.attn.omni_attn.forward

    def spy(query, key, value, attn_metadata=None):
        seen.append((tuple(query.shape), tuple(key.shape), tuple(value.shape), attn_metadata))
        return original(query, key, value, attn_metadata)

    monkeypatch.setattr(block.attn.omni_attn, "forward", spy)
    with torch.no_grad():
        out = block(hidden, mask, rotary, temb)
    assert out.shape == hidden.shape and torch.isfinite(out).all()
    assert len(seen) == 1
    q_shape, k_shape, v_shape, metadata = seen[0]
    assert q_shape == (BATCH, SEQ, HEADS, block.head_dim)
    assert k_shape == v_shape == (BATCH, SEQ, 2, block.head_dim)
    assert metadata is not None and metadata.attn_mask.dtype == torch.bool
    assert torch.equal(metadata.attn_mask, mask)


def test_backend_selection_reaches_the_dit_layer():
    """The DiT's attention takes its backend from the diffusion config, not from
    a model-local switch: what the config names is what executes."""
    assert _block(2).attn.omni_attn.attn_backend is DiffusionAttentionBackendEnum.TORCH_SDPA.get_class()
    with set_current_diffusion_config(
        OmniDiffusionConfig(diffusion_attention_config={"default": {"backend": "FLASH_ATTN"}})
    ):
        block = TransformerBlock(DIM, HEADS, 2, multiple_of=8, ffn_dim_multiplier=1.0, norm_eps=1e-5)
    assert block.attn.omni_attn.attn_backend is DiffusionAttentionBackendEnum.FLASH_ATTN.get_class()


def test_shared_layer_owns_no_parameters_and_keeps_checkpoint_keys():
    block = _block(2)
    assert sum(p.numel() for p in block.attn.omni_attn.parameters()) == 0
    assert not list(block.attn.omni_attn.buffers())
    keys = set(block.state_dict())
    assert not any("omni_attn" in key for key in keys)
    # The projections and QK norms the checkpoint provides are still where they were.
    for expected in (
        "attn.to_q.weight",
        "attn.to_k.weight",
        "attn.to_v.weight",
        "attn.to_out.0.weight",
        "attn.norm_q.weight",
        "attn.norm_k.weight",
    ):
        assert expected in keys, expected


@pytest.mark.parametrize("with_mask", [True, False], ids=["empty_mask", "no_mask"])
def test_empty_text_stream_skips_the_kernel(monkeypatch, with_mask):
    """CFG's unconditional branch carries zero text tokens (the pipeline's default
    negative_prompt_embeds has no rows), so the context refiner attends over an
    empty sequence. The flash-attention varlen fallback cannot take that
    (arange step 0), so the processor must not reach the shared layer at all."""
    block = _block(kv_heads=2, modulation=False)
    attn = block.attn

    def _never(*args, **kwargs):
        raise AssertionError("shared layer called with an empty sequence")

    monkeypatch.setattr(attn.omni_attn, "forward", _never)
    hidden = torch.randn(BATCH, 0, DIM)
    mask = torch.ones(BATCH, 0, dtype=torch.bool) if with_mask else None
    angles = torch.rand(1, 0, block.head_dim)
    out = attn(
        hidden_states=hidden,
        encoder_hidden_states=hidden,
        attention_mask=mask,
        image_rotary_emb=(angles.cos(), angles.sin()),
    )
    assert out.shape == (BATCH, 0, DIM)
