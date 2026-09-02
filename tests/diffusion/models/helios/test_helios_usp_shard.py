# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Helios per-component sequence splitting.

These tests verify the manual sharding logic that replaces the broken
rope/blocks.0 hooks in the Helios ``_sp_plan``.  The split helper is
replicated here (rather than imported from the model) so that the tests
run on a plain CPU without the full vllm_omni / vllm dependency stack.

Coverage:
  1. USP disabled (ws == 1) returns the tensor unchanged.
  2. Divisible split gives each rank the expected slice.
  3. Non-divisible split pads with replicated tokens (no crash).
  4. Per-component split: history + current tokens are distributed so
     that *both* ranks receive some current tokens (no mosaic regression).
  5. Edge cases: 1-D tensor, empty seq, 4-D tensor.
  6. Multi-component non-divisible history (RuixiangMa crash scenario).
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# ---------------------------------------------------------------------------
# Copy of HeliosTransformer3DModel._sp_split_seq.
# Keep in sync with
# vllm_omni/diffusion/models/helios/helios_transformer.py
# ---------------------------------------------------------------------------


def _sp_split_seq(x: torch.Tensor, ws: int, rank: int) -> torch.Tensor:
    """Split tensor along sequence dim (dim=1) for Ulysses SP."""
    if ws > 1 and x.dim() >= 2 and x.shape[1] > 0:
        seq_len = x.shape[1]
        remainder = seq_len % ws
        if remainder != 0:
            pad = ws - remainder
            last = x[:, -1:, ...].expand(-1, pad, *([-1] * (x.dim() - 2)))
            x = torch.cat([x, last], dim=1)
            seq_len = x.shape[1]
        n = seq_len // ws
        x = x[:, rank * n : (rank + 1) * n, ...].contiguous()
    return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpSplitSeqDisabled:
    """When USP is off (ws == 1) the helper must be a no-op."""

    def test_ws1_returns_unchanged(self):
        x = torch.randn(2, 100, 64)
        out = _sp_split_seq(x, ws=1, rank=0)
        assert torch.equal(out, x)
        assert out is x


class TestSpSplitSeqDivisible:
    """Divisible sequences: each rank gets exactly seq_len / ws tokens."""

    def test_each_rank_gets_correct_slice(self):
        for ws in [2, 4, 8]:
            seq_len = ws * 10
            x = torch.arange(seq_len * 4).float().reshape(1, seq_len, 4)
            for rank in range(ws):
                out = _sp_split_seq(x, ws=ws, rank=rank)
                n = seq_len // ws
                expected = x[:, rank * n : (rank + 1) * n, :]
                assert out.shape == (1, n, 4)
                assert torch.equal(out, expected)

    def test_output_is_contiguous(self):
        x = torch.randn(1, 20, 8)
        out = _sp_split_seq(x, ws=2, rank=0)
        assert out.is_contiguous()


class TestSpSplitSeqNonDivisible:
    """Non-divisible sequences are padded with replicated tokens."""

    def test_non_divisible_pads_instead_of_crash(self):
        """400x272 → 9*25*17 = 3825 tokens, USP=2 → not divisible.
        Should pad to 3826 and split, not crash."""
        x = torch.randn(1, 3825, 8)
        for rank in range(2):
            out = _sp_split_seq(x, ws=2, rank=rank)
            # 3825 → pad 1 → 3826 → /2 = 1913
            assert out.shape[1] == 1913

    def test_non_divisible_pads_with_replicated_last_token(self):
        """Verify the padding is the replicated last token, not zeros."""
        x = torch.randn(1, 7, 4)  # 7 % 2 = 1, pad 1
        last_token = x[:, -1:, :].clone()
        out = _sp_split_seq(x, ws=2, rank=0)
        # rank 0 gets tokens [0:4] (first 4 of the padded 8)
        assert out.shape == (1, 4, 4)
        # rank 1 gets tokens [4:8] — token 7 (the padded one) = last token
        out1 = _sp_split_seq(x, ws=2, rank=1)
        assert out1.shape == (1, 4, 4)
        assert torch.equal(out1[:, -1:, :], last_token)

    def test_non_divisible_7_4(self):
        """7 tokens, ws=4 → pad 1 → 8 → /4 = 2 per rank."""
        x = torch.randn(1, 7, 8)
        for rank in range(4):
            out = _sp_split_seq(x, ws=4, rank=rank)
            assert out.shape[1] == 2

    def test_non_divisible_15_4(self):
        """15 tokens, ws=4 → pad 1 → 16 → /4 = 4 per rank."""
        x = torch.randn(1, 15, 8)
        for rank in range(4):
            out = _sp_split_seq(x, ws=4, rank=rank)
            assert out.shape[1] == 4


class TestSpSplitSeqPerComponent:
    """Simulate Helios forward: split current + history components so that
    *both* ranks receive some current tokens (no mosaic regression)."""

    def test_both_ranks_get_current_tokens(self):
        """history=2400, current=540, ws=2.
        After per-component split, each rank must have > 0 current tokens."""
        current = torch.randn(1, 540, 64)
        history = torch.randn(1, 2400, 64)

        for rank in range(2):
            cur_local = _sp_split_seq(current, ws=2, rank=rank)
            hist_local = _sp_split_seq(history, ws=2, rank=rank)

            assert cur_local.shape[1] > 0, f"Rank {rank} received 0 current tokens"
            assert cur_local.shape[1] == 270  # 540 / 2
            assert hist_local.shape[1] == 1200  # 2400 / 2

            original_context_length = cur_local.shape[1]
            assert original_context_length > 0  # no mosaic regression

    def test_gather_restores_full_current(self):
        """After splitting current into 2 ranks, simulating proj_out
        gather must restore the full current sequence length so that
        the original unpatchify reshape succeeds."""
        current = torch.randn(1, 540, 64)
        local_chunks = [_sp_split_seq(current, ws=2, rank=rank) for rank in range(2)]
        gathered = torch.cat(local_chunks, dim=1)
        assert gathered.shape[1] == 540  # full current restored

    def test_gather_with_padding_truncates_correctly(self):
        """Non-divisible current (3825) → pad to 3826 → split →
        gather → 3826. Slice to 3825 for unpatchify."""
        current = torch.randn(1, 3825, 64)
        local_chunks = [_sp_split_seq(current, ws=2, rank=rank) for rank in range(2)]
        gathered = torch.cat(local_chunks, dim=1)
        assert gathered.shape[1] == 3826  # padded size
        # Simulate the slice before unpatchify
        expected_seq = 3825
        if gathered.shape[1] > expected_seq:
            gathered = gathered[:, :expected_seq, :]
        assert gathered.shape[1] == 3825  # correct after slice

    def test_rotary_emb_aligned_with_hidden_states(self):
        """rotary_emb and hidden_states must be split at the same
        positions so that tokens align inside the attention layer."""
        hidden = torch.randn(1, 540, 5120)
        rotary = torch.randn(1, 540, 256)
        for rank in range(2):
            h_local = _sp_split_seq(hidden, ws=2, rank=rank)
            r_local = _sp_split_seq(rotary, ws=2, rank=rank)
            assert h_local.shape[1] == r_local.shape[1]

    def test_whole_split_causes_mosaic_but_per_component_does_not(self):
        """Demonstrates why per-component split is needed:
        with whole split (simulated), rank 0 gets 0 current tokens;
        with per-component split, both ranks get current tokens."""
        history = torch.randn(1, 2400, 64)
        current = torch.randn(1, 540, 64)
        full = torch.cat([history, current], dim=1)  # 2940

        # Whole split (the OLD broken behavior)
        whole_n = full.shape[1] // 2  # 1470
        rank0_whole = full[:, :whole_n, :]
        rank0_current = rank0_whole.shape[1] - 2400  # negative → 0 current
        assert rank0_current <= 0, "Whole split should give rank 0 no current"

        # Per-component split (the NEW correct behavior)
        for rank in range(2):
            cur_local = _sp_split_seq(current, ws=2, rank=rank)
            assert cur_local.shape[1] == 270  # both ranks have current


class TestSpSplitSeqEdgeCases:
    """Edge cases that should not crash or silently misbehave."""

    def test_1d_tensor_skipped(self):
        """1-D tensors must be returned unchanged (dim < 2 guard)."""
        x = torch.randn(100)
        out = _sp_split_seq(x, ws=2, rank=0)
        assert torch.equal(out, x)

    def test_empty_seq_skipped(self):
        """Empty sequence dimension must be returned unchanged."""
        x = torch.randn(1, 0, 64)
        out = _sp_split_seq(x, ws=2, rank=0)
        assert out.shape == x.shape

    def test_4d_tensor_split_on_dim1(self):
        """4-D tensor [B, S, H, D] should split along dim=1."""
        x = torch.randn(1, 20, 8, 32)
        out = _sp_split_seq(x, ws=2, rank=1)
        assert out.shape == (1, 10, 8, 32)


class TestSpSplitSeqMultiComponentNonDivisible:
    """History components use different patch sizes -> different seq lengths,
    some non-divisible. Reproduces RuixiangMa's crash scenario: a non-divisible
    history component must not crash and both ranks must still receive current.
    """

    def test_non_divisible_history_does_not_crash(self):
        current = torch.randn(1, 540, 64)    # 540 % 2 == 0
        history = torch.randn(1, 3825, 64)  # 3825 % 2 == 1 -> pad to 3826
        for rank in range(2):
            cur_local = _sp_split_seq(current, ws=2, rank=rank)
            hist_local = _sp_split_seq(history, ws=2, rank=rank)
            assert cur_local.shape[1] == 270           # current unaffected
            assert hist_local.shape[1] == 1913          # (3825 + 1) / 2
            assert cur_local.shape[1] > 0               # no mosaic regression

    def test_different_history_sizes_all_split_without_crash(self):
        # short / mid / long history use different patch sizes -> different
        # sequence lengths, several non-divisible by ws=4.
        histories = [
            torch.randn(1, 800, 64),    # 800  % 4 == 0 -> 200 per rank
            torch.randn(1, 3825, 64),   # 3825 % 4 == 1 -> pad 3 -> 957 per rank
            torch.randn(1, 1001, 64),   # 1001 % 4 == 1 -> pad 3 -> 251 per rank
        ]
        for rank in range(4):
            for hist in histories:
                out = _sp_split_seq(hist, ws=4, rank=rank)
                assert out.shape[1] > 0
        assert _sp_split_seq(histories[0], ws=4, rank=0).shape[1] == 200
        assert _sp_split_seq(histories[1], ws=4, rank=0).shape[1] == 957
        assert _sp_split_seq(histories[2], ws=4, rank=0).shape[1] == 251
