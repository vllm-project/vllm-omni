# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Numerical equivalence tests for rotary embedding implementations (#2436).

Verifies that the optimized stack+flatten RoPE produces bit-identical results
to the original strided-slice implementation across various tensor shapes and
dtypes, ensuring the refactor is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import torch

from vllm_omni.diffusion.layers import rope as rope_module
from vllm_omni.diffusion.models.helios import helios_transformer as helios_module
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosRotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass(frozen=True)
class _DeviceMetadata:
    type: str


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: _DeviceMetadata

    def dim(self) -> int:
        return len(self.shape)


def _tensor_metadata(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device_type: str = "cpu",
) -> torch.Tensor:
    """Build typed tensor metadata for the NPU eligibility predicate."""
    metadata = _TensorMetadata(shape, dtype, _DeviceMetadata(device_type))
    return cast(torch.Tensor, metadata)


def _apply_rotary_emb_helios_original(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Original Helios RoPE using strided slice assignment (pre-#2436)."""
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
    out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
    return out.type_as(hidden_states)


def _apply_rotary_emb_helios_adapter(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Helios adapter with its portable fallback on non-NPU platforms."""
    return HeliosRotaryEmbedding()(hidden_states, freqs_cis)


def _make_inputs(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random hidden_states and freqs_cis for testing."""
    torch.manual_seed(42)
    hidden_states = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype)
    # freqs_cis: [B, seq, head_dim*2] — cos and sin concatenated along last dim
    freqs_cis = torch.randn(batch, seq_len, head_dim * 2, dtype=dtype)
    return hidden_states, freqs_cis


class TestHeliosRoPEEquivalence:
    """Verify optimized Helios RoPE is numerically identical to original."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_equivalence_across_dtypes(self, dtype: torch.dtype) -> None:
        """Optimized output must be bit-identical to original across dtypes."""
        hidden, freqs = _make_inputs(2, 16, 8, 64, dtype=dtype)
        original = _apply_rotary_emb_helios_original(hidden, freqs)
        optimized = _apply_rotary_emb_helios_adapter(hidden, freqs)
        torch.testing.assert_close(optimized, original, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "batch,seq_len,num_heads,head_dim",
        [
            (1, 8, 1, 32),  # minimal: single batch, single head
            (2, 16, 8, 64),  # typical transformer config
            (1, 8192, 4, 64),  # video-scale patch tokens (720p DiT)
            (4, 32, 16, 128),  # large head_dim
        ],
    )
    def test_equivalence_across_shapes(self, batch: int, seq_len: int, num_heads: int, head_dim: int) -> None:
        """Equivalence must hold across different tensor shapes."""
        hidden, freqs = _make_inputs(batch, seq_len, num_heads, head_dim)
        original = _apply_rotary_emb_helios_original(hidden, freqs)
        optimized = _apply_rotary_emb_helios_adapter(hidden, freqs)
        torch.testing.assert_close(optimized, original, atol=0, rtol=0)

    def test_output_contiguous(self) -> None:
        """Optimized output should be contiguous in memory."""
        hidden, freqs = _make_inputs(2, 16, 8, 64)
        optimized = _apply_rotary_emb_helios_adapter(hidden, freqs)
        assert optimized.is_contiguous()

    def test_output_shape_preserved(self) -> None:
        """Output shape must match input shape."""
        hidden, freqs = _make_inputs(2, 16, 8, 64)
        optimized = _apply_rotary_emb_helios_adapter(hidden, freqs)
        assert optimized.shape == hidden.shape

    def test_output_dtype_preserved(self) -> None:
        """Output dtype must match input dtype."""
        hidden, freqs = _make_inputs(2, 16, 8, 64, dtype=torch.float16)
        optimized = _apply_rotary_emb_helios_adapter(hidden, freqs)
        assert optimized.dtype == hidden.dtype

    def test_shared_rope_receives_half_width_interleaved_frequencies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The adapter must preserve Helios's adjacent-pair frequency layout."""
        hidden, freqs = _make_inputs(2, 16, 8, 64)
        rope = HeliosRotaryEmbedding()

        def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            assert x is hidden
            torch.testing.assert_close(cos, freqs[..., :64:2].unsqueeze(-2))
            torch.testing.assert_close(sin, freqs[..., 65::2].unsqueeze(-2))
            return x

        monkeypatch.setattr(rope, "_can_use_npu_impl", lambda *_: True)
        monkeypatch.setattr(rope.impl, "_forward_method", rotary_embedding)
        assert rope(hidden, freqs) is hidden

    def test_mixed_dtype_equivalence(self) -> None:
        """FP32 frequencies with BF16 activations match the prior arithmetic."""
        hidden, _ = _make_inputs(2, 16, 8, 64, dtype=torch.bfloat16)
        _, freqs = _make_inputs(2, 16, 8, 64, dtype=torch.float32)
        original = _apply_rotary_emb_helios_original(hidden, freqs)
        optimized = _apply_rotary_emb_helios_adapter(hidden, freqs)
        torch.testing.assert_close(optimized, original, atol=0, rtol=0)

    def test_shared_rope_is_stateless(self) -> None:
        """Adding the adapter must not add checkpoint parameters or buffers."""
        rope = HeliosRotaryEmbedding()
        assert not list(rope.parameters())
        assert not list(rope.buffers())
        assert not rope.state_dict()

    def test_npu_impl_requires_single_aligned_activation_and_frequency_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Batched tables stay native because fused backends flatten them."""
        monkeypatch.setattr(helios_module, "_is_validated_ascend_a3_device", lambda *_: True)
        npu_b1 = _tensor_metadata((1, 16, 8, 128), torch.bfloat16, "npu")
        npu_b2 = _tensor_metadata((2, 16, 8, 128), torch.bfloat16, "npu")
        cpu_b1 = _tensor_metadata((1, 16, 8, 128), torch.bfloat16)
        npu_fp16 = _tensor_metadata((1, 16, 8, 128), torch.float16, "npu")
        freqs_b1 = _tensor_metadata((1, 16, 256), torch.float32, "npu")
        freqs_b2 = _tensor_metadata((2, 16, 256), torch.float32, "npu")
        freqs_wrong_seq = _tensor_metadata((1, 15, 256), torch.float32, "npu")
        freqs_bf16 = _tensor_metadata((1, 16, 256), torch.bfloat16, "npu")
        freqs_cpu = _tensor_metadata((1, 16, 256), torch.float32)

        assert HeliosRotaryEmbedding._can_use_npu_impl(npu_b1, freqs_b1)
        assert not HeliosRotaryEmbedding._can_use_npu_impl(npu_b2, freqs_b1)
        assert not HeliosRotaryEmbedding._can_use_npu_impl(npu_b1, freqs_b2)
        assert not HeliosRotaryEmbedding._can_use_npu_impl(npu_b1, freqs_wrong_seq)
        assert not HeliosRotaryEmbedding._can_use_npu_impl(cpu_b1, freqs_b1)
        assert not HeliosRotaryEmbedding._can_use_npu_impl(npu_fp16, freqs_b1)
        assert not HeliosRotaryEmbedding._can_use_npu_impl(npu_b1, freqs_bf16)
        assert not HeliosRotaryEmbedding._can_use_npu_impl(npu_b1, freqs_cpu)

    def test_npu_impl_requires_validated_a3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hidden = _tensor_metadata((1, 16, 8, 128), torch.bfloat16, "npu")
        freqs = _tensor_metadata((1, 16, 256), torch.float32, "npu")
        monkeypatch.setattr(helios_module, "_is_validated_ascend_a3_device", lambda *_: False)
        assert not HeliosRotaryEmbedding._can_use_npu_impl(hidden, freqs)

    def test_npu_impl_adapts_to_mindie_sequence_layout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The eligible path presents one half-width row per token to MindIE."""
        hidden, freqs = _make_inputs(1, 16, 8, 64, dtype=torch.bfloat16)
        rope = HeliosRotaryEmbedding()
        rope.impl.has_mindie = True

        def apply_mindie(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            interleaved: bool,
            half_head_dim: bool,
        ) -> torch.Tensor:
            assert x is hidden
            assert cos.shape == sin.shape == (16, 32)
            assert interleaved and half_head_dim
            return x

        monkeypatch.setattr(rope, "_can_use_npu_impl", lambda *_: True)
        monkeypatch.setattr(rope_module, "apply_rotary_emb_mindiesd", apply_mindie)
        monkeypatch.setattr(rope.impl, "_forward_method", rope.impl.forward_npu)
        assert rope(hidden, freqs) is hidden

    def test_odd_head_dim_raises(self) -> None:
        """Odd head_dim should be rejected as an invalid RoPE config."""
        hidden = torch.randn(1, 4, 2, 63)
        freqs = torch.randn(1, 4, 126)
        with pytest.raises(ValueError, match="even head dimension"):
            _apply_rotary_emb_helios_adapter(hidden, freqs)
