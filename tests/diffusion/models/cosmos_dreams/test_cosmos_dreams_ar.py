# SPDX-License-Identifier: Apache-2.0
"""CPU oracle tests for Cosmos-Dreams gathered-paged joint attention."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.cosmos_dreams.transformer_cosmos_dreams import (
    CosmosDreamsJointAttention,
)
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

BLOCK = 8
N_HEADS = 2
HEAD_DIM = 4
HIDDEN = N_HEADS * HEAD_DIM
BRANCH = "main"


class _DenseAttention(nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * (HEAD_DIM**-0.5)
        probability = torch.softmax(scores, dim=-1).to(value.dtype)
        return torch.einsum("bhqk,bkhd->bqhd", probability, value)


def _joint_attention() -> CosmosDreamsJointAttention:
    attention = CosmosDreamsJointAttention.__new__(CosmosDreamsJointAttention)
    nn.Module.__init__(attention)
    attention.num_heads_local = N_HEADS
    attention.num_kv_heads_local = N_HEADS
    attention.head_dim = HEAD_DIM
    attention.qk_norm = False
    attention.to_q = nn.Identity()
    attention.to_k = nn.Identity()
    attention.to_v = nn.Identity()
    attention.to_out = nn.Identity()
    attention.attn = _DenseAttention()
    return attention


def _state(*, window_frames: int = 2) -> tuple[ARDiffusionKVCache, ARDiffusionKVState]:
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(
            enable=True,
            chunk_size=BLOCK,
            window_chunks=window_frames,
        ),
        num_layers=1,
        num_kv_heads=N_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float32,
        block_size=BLOCK,
        max_model_len=1 << 20,
        available_bytes=1 << 24,
        kv_branches=(ARDiffusionKVBranchSpec(BRANCH, 0),),
        session_capacity=1,
        frames_per_block=1,
        device=torch.device("cpu"),
    )
    adapter = cache.begin_request("cosmos-dreams-main")
    return cache, ARDiffusionKVState(cache, "session", {BRANCH: adapter}, num_layers=1)


def _attention_forward(
    attention: CosmosDreamsJointAttention,
    hidden: torch.Tensor,
    text_k: torch.Tensor,
    text_v: torch.Tensor,
    *,
    real_text_kv_len: int,
    dense_history: tuple[torch.Tensor, torch.Tensor] | None = None,
    paged_context=None,
    null_action: bool = False,
    action_tokens_per_frame: int | None = 2,
):
    freqs_cos = torch.ones(1, BLOCK, 1, HEAD_DIM)
    freqs_sin = torch.zeros_like(freqs_cos)
    return attention(
        hidden,
        text_k=text_k,
        text_v=text_v,
        real_text_kv_len=real_text_kv_len,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        dense_history=dense_history,
        paged_context=paged_context,
        num_frames=1,
        tokens_per_frame=BLOCK,
        action_tokens_per_frame=action_tokens_per_frame,
        null_action_frame_indexes=(0,) if null_action else (),
    )


def test_joint_attention_skips_action_value_zeroing_without_conditioning_tokens() -> None:
    torch.manual_seed(3)
    attention = _joint_attention()
    hidden = torch.randn(1, BLOCK, HIDDEN)
    text_k = torch.randn(1, 3, N_HEADS, HEAD_DIM)
    text_v = torch.randn_like(text_k)

    action = _attention_forward(
        attention,
        hidden,
        text_k,
        text_v,
        real_text_kv_len=3,
    )
    no_conditioning = _attention_forward(
        attention,
        hidden,
        text_k,
        text_v,
        real_text_kv_len=3,
        action_tokens_per_frame=None,
    )

    for action_value, generic_value in zip(action, no_conditioning, strict=True):
        assert torch.equal(action_value, generic_value)
    with pytest.raises(ValueError, match="require action conditioning tokens"):
        _attention_forward(
            attention,
            hidden,
            text_k,
            text_v,
            real_text_kv_len=3,
            null_action=True,
            action_tokens_per_frame=None,
        )


def test_gathered_paged_joint_attention_is_exactly_the_dense_oracle() -> None:
    torch.manual_seed(0)
    attention = _joint_attention()
    _, state = _state()
    text_k = torch.randn(1, 5, N_HEADS, HEAD_DIM)
    text_v = torch.randn_like(text_k)
    history_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)
    history_v = torch.randn_like(history_k)

    commit = state.get_kv_caches(BRANCH, seq_len=BLOCK, commit_current=True)[0]
    commit.write_only(history_k, history_v)
    state.commit_paged_context(BRANCH)

    hidden = torch.randn(1, BLOCK, HIDDEN)
    dense = _attention_forward(
        attention,
        hidden,
        text_k,
        text_v,
        real_text_kv_len=3,
        dense_history=(history_k, history_v),
    )
    read_context = state.get_kv_caches(BRANCH, seq_len=BLOCK, commit_current=False)[0]
    paged = _attention_forward(
        attention,
        hidden,
        text_k,
        text_v,
        real_text_kv_len=3,
        paged_context=read_context,
    )
    state.commit_paged_context(BRANCH)

    for dense_value, paged_value in zip(dense, paged, strict=True):
        assert torch.equal(dense_value, paged_value)


def test_text_padding_is_invisible_in_the_real_joint_attention_path() -> None:
    torch.manual_seed(1)
    attention = _joint_attention()
    hidden = torch.randn(1, BLOCK, HIDDEN)
    text_k = torch.randn(1, 5, N_HEADS, HEAD_DIM)
    text_v = torch.randn_like(text_k)

    baseline = _attention_forward(
        attention,
        hidden,
        text_k,
        text_v,
        real_text_kv_len=3,
    )[0]
    text_k[:, 3:] = 1_000
    text_v[:, 3:] = -1_000
    changed_padding = _attention_forward(
        attention,
        hidden,
        text_k,
        text_v,
        real_text_kv_len=3,
    )[0]

    assert torch.equal(baseline, changed_padding)


def test_sequential_clean_commits_match_dense_history_and_roll_the_window() -> None:
    torch.manual_seed(2)
    attention = _joint_attention()
    _, state = _state(window_frames=2)
    text_k = torch.randn(1, 3, N_HEADS, HEAD_DIM)
    text_v = torch.randn_like(text_k)
    dense_history: tuple[torch.Tensor, torch.Tensor] | None = None

    for frame_idx in range(3):
        hidden = torch.randn(1, BLOCK, HIDDEN)
        dense = _attention_forward(
            attention,
            hidden,
            text_k,
            text_v,
            real_text_kv_len=3,
            dense_history=dense_history,
            null_action=frame_idx == 0,
        )
        commit = state.get_kv_caches(BRANCH, seq_len=BLOCK, commit_current=True)[0]
        paged = _attention_forward(
            attention,
            hidden,
            text_k,
            text_v,
            real_text_kv_len=3,
            paged_context=commit,
            null_action=frame_idx == 0,
        )
        state.commit_paged_context(BRANCH)

        for dense_value, paged_value in zip(dense, paged, strict=True):
            assert torch.equal(dense_value, paged_value)

        current_k, current_v = dense[1:]
        if dense_history is None:
            next_k, next_v = current_k, current_v
        else:
            next_k = torch.cat([dense_history[0], current_k], dim=1)
            next_v = torch.cat([dense_history[1], current_v], dim=1)
        dense_history = (next_k[:, -2 * BLOCK :], next_v[:, -2 * BLOCK :])

    read_context = state.get_kv_caches(BRANCH, seq_len=BLOCK, commit_current=False)[0]
    gathered = read_context.gather_history()
    state.commit_paged_context(BRANCH)
    assert dense_history is not None
    assert torch.equal(gathered[0], dense_history[0])
    assert torch.equal(gathered[1], dense_history[1])
