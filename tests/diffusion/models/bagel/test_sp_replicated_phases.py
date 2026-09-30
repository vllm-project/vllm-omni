# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BAGEL must keep its replicated phases off the sequence-parallel strategy.

The text prefill, the ViT/VAE cache updates and the CFG cache-update forward
run on sequences every SP rank holds in full; only the sharded denoise path
may go through the strategy (Ulysses would otherwise all-to-all the identical
per-rank copies as if they were shards).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.bagel.bagel_transformer as bagel_mod
from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache, PackedAttentionMoT

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 4, 2, 8
HIDDEN = NUM_HEADS * HEAD_DIM
SEQ = 6  # [start_of_image, 4 vae tokens, end_of_image]


class _RecordingAttention:
    """Stands in for a DiffusionAttention layer: records calls, returns the query."""

    def __init__(self, joint: bool = False) -> None:
        self.joint = joint
        self.calls: list = []

    def __call__(self, q, k, v, attn_metadata=None):
        self.calls.append(attn_metadata)
        if self.joint and attn_metadata is not None and attn_metadata.joint_query is not None:
            return torch.cat([attn_metadata.joint_query, q], dim=1)
        return q


def test_constructor_builds_local_layers_for_replicated_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Attn(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.kwargs = kwargs

    monkeypatch.setattr(bagel_mod, "DiffusionAttention", _Attn)
    for name in ("MoTQKVParallelLinear", "MoTRowParallelLinear", "MoTRMSNorm", "RotaryEmbedding"):
        monkeypatch.setattr(bagel_mod, name, lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(bagel_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    cfg = SimpleNamespace(
        hidden_size=HIDDEN, num_attention_heads=NUM_HEADS, num_key_value_heads=NUM_KV_HEADS, rms_norm_eps=1e-6
    )

    attn = PackedAttentionMoT(cfg, layer_idx=0)

    assert attn.attn_causal.kwargs["causal"] is True
    assert attn.attn_causal.kwargs["skip_sequence_parallel"] is True
    assert attn.attn_noncausal.kwargs["causal"] is False
    assert attn.attn_noncausal.kwargs.get("skip_sequence_parallel", False) is False
    assert attn.attn_noncausal_local.kwargs["causal"] is False
    assert attn.attn_noncausal_local.kwargs["skip_sequence_parallel"] is True


def _stub(sp_active: bool) -> PackedAttentionMoT:
    attn = PackedAttentionMoT.__new__(PackedAttentionMoT)
    torch.nn.Module.__init__(attn)
    attn.layer_idx = 0
    attn.hidden_size = HIDDEN
    attn.num_heads, attn.num_kv_heads, attn.head_dim = NUM_HEADS, NUM_KV_HEADS, HEAD_DIM
    attn.q_size, attn.kv_size = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
    width = attn.q_size + 2 * attn.kv_size
    attn.qkv_proj = lambda x, *idx: (torch.zeros(x.shape[0], width), None)
    attn.q_norm = lambda x, *idx: x
    attn.k_norm = lambda x, *idx: x
    attn.rotary_op = lambda x, cos, sin: x
    attn.o_proj = lambda x, *idx: (x[:, :HIDDEN], None)
    attn.attn_causal = _RecordingAttention()
    attn.attn_noncausal = _RecordingAttention(joint=True)
    attn.attn_noncausal_local = _RecordingAttention()
    attn._is_sp_active = lambda: sp_active
    return attn


def _gen_inputs():
    packed = torch.zeros(SEQ, HIDDEN)
    pos = (torch.zeros(SEQ, HEAD_DIM), torch.zeros(SEQ, HEAD_DIM))
    return dict(
        packed_query_sequence=packed,
        query_lens=torch.tensor([SEQ]),
        packed_query_position_embeddings=pos,
        past_key_values=NaiveCache(1),
        packed_vae_token_indexes=torch.arange(1, SEQ - 1),
        packed_text_indexes=torch.tensor([0, SEQ - 1]),
    )


def test_gen_cache_update_under_sp_runs_locally() -> None:
    attn = _stub(sp_active=True)
    inputs = _gen_inputs()

    out, cache = attn._forward_gen(**inputs, update_past_key_values=True)

    assert attn.attn_noncausal.calls == []
    assert len(attn.attn_noncausal_local.calls) == 1
    assert out.shape == (SEQ, HIDDEN)
    assert cache.key_cache[0].shape == (SEQ, NUM_KV_HEADS, HEAD_DIM)


def test_gen_denoise_under_sp_still_uses_the_strategy() -> None:
    attn = _stub(sp_active=True)

    out, _ = attn._forward_gen(**_gen_inputs(), update_past_key_values=False)

    assert attn.attn_noncausal_local.calls == []
    (metadata,) = attn.attn_noncausal.calls
    assert metadata.joint_strategy == "front"
    assert metadata.joint_query.shape == (1, 2, NUM_HEADS, HEAD_DIM)
    assert metadata.joint_key.shape == (1, 2, NUM_KV_HEADS, HEAD_DIM)
    assert out.shape == (SEQ, HIDDEN)


def test_gen_without_sp_runs_locally() -> None:
    attn = _stub(sp_active=False)

    attn._forward_gen(**_gen_inputs(), update_past_key_values=False)

    assert attn.attn_noncausal.calls == []
    assert len(attn.attn_noncausal_local.calls) == 1


@pytest.mark.parametrize("is_causal", [True, False])
def test_und_prefill_never_reaches_the_strategy(is_causal: bool) -> None:
    attn = _stub(sp_active=True)
    pos = (torch.zeros(SEQ, HEAD_DIM), torch.zeros(SEQ, HEAD_DIM))

    out, cache = attn._forward_und(torch.zeros(SEQ, HIDDEN), pos, NaiveCache(1), is_causal=is_causal)

    assert attn.attn_noncausal.calls == []
    used = attn.attn_causal if is_causal else attn.attn_noncausal_local
    assert len(used.calls) == 1
    assert out.shape == (SEQ, HIDDEN)
    assert cache.key_cache[0].shape == (SEQ, NUM_KV_HEADS, HEAD_DIM)
