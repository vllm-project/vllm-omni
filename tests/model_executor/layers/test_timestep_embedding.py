# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared timestep embedding primitives."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TestTimestepEmbedding:
    """Verify the timestep_embedding standalone function."""

    def test_output_shape_even(self) -> None:
        from vllm_omni.model_executor.layers.timestep_embedding import timestep_embedding

        t = torch.arange(4, dtype=torch.float32)
        out = timestep_embedding(t, 64)
        assert out.shape == (4, 64)

    def test_output_shape_odd(self) -> None:
        """Odd dim gets zero-padded to full size."""
        from vllm_omni.model_executor.layers.timestep_embedding import timestep_embedding

        t = torch.arange(4, dtype=torch.float32)
        out = timestep_embedding(t, 65)
        assert out.shape == (4, 65)

    def test_odd_dim_last_column_zero(self) -> None:
        """Odd dim pads the last column with zeros."""
        from vllm_omni.model_executor.layers.timestep_embedding import timestep_embedding

        t = torch.arange(4, dtype=torch.float32)
        out = timestep_embedding(t, 65)
        torch.testing.assert_close(out[:, -1], torch.zeros(4))

    def test_deterministic(self) -> None:
        """Same inputs produce identical outputs."""
        from vllm_omni.model_executor.layers.timestep_embedding import timestep_embedding

        t = torch.tensor([0.0, 0.5, 1.0])
        a = timestep_embedding(t, 32)
        b = timestep_embedding(t, 32)
        torch.testing.assert_close(a, b)

    def test_different_max_period(self) -> None:
        """Different max_period produces different embeddings."""
        from vllm_omni.model_executor.layers.timestep_embedding import timestep_embedding

        t = torch.tensor([1.0, 2.0])
        a = timestep_embedding(t, 32, max_period=10000.0)
        b = timestep_embedding(t, 32, max_period=1000.0)
        assert not torch.allclose(a, b)


class TestSinusPositionEmbedding:
    """Verify the SinusPositionEmbedding module."""

    def test_output_shape(self) -> None:
        from vllm_omni.model_executor.layers.timestep_embedding import SinusPositionEmbedding

        emb = SinusPositionEmbedding(128)
        x = torch.randn(8)
        assert emb(x).shape == (8, 128)

    def test_dtype_preserved(self) -> None:
        from vllm_omni.model_executor.layers.timestep_embedding import SinusPositionEmbedding

        emb = SinusPositionEmbedding(64)
        x = torch.randn(4, dtype=torch.float16)
        assert emb(x).dtype == torch.float16

    def test_different_scale(self) -> None:
        """Different scale values produce different embeddings."""
        from vllm_omni.model_executor.layers.timestep_embedding import SinusPositionEmbedding

        emb = SinusPositionEmbedding(64)
        x = torch.tensor([1.0, 2.0])
        a = emb(x, scale=1000.0)
        b = emb(x, scale=500.0)
        assert not torch.allclose(a, b)


class TestDiTTimestepEmbedding:
    """Verify the DiTTimestepEmbedding module."""

    def test_output_shape(self) -> None:
        from vllm_omni.model_executor.layers.timestep_embedding import DiTTimestepEmbedding

        emb = DiTTimestepEmbedding(dim=256, freq_embed_dim=128)
        t = torch.randn(4)
        assert emb(t).shape == (4, 256)

    def test_output_shape_default_freq(self) -> None:
        from vllm_omni.model_executor.layers.timestep_embedding import DiTTimestepEmbedding

        emb = DiTTimestepEmbedding(dim=512)
        t = torch.randn(2)
        assert emb(t).shape == (2, 512)

    def test_has_submodules(self) -> None:
        """Should contain time_embed and time_mlp."""
        from vllm_omni.model_executor.layers.timestep_embedding import DiTTimestepEmbedding

        emb = DiTTimestepEmbedding(dim=256)
        assert hasattr(emb, "time_embed")
        assert hasattr(emb, "time_mlp")
