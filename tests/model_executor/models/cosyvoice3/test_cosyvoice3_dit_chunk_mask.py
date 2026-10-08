# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU L1 coverage for CosyVoice3 DiT streaming chunk attention.

These tests exist because FLASH/fused SDPA can ignore a full query-key mask.
They assert pre-softmax chunk constraints and that the MATH fallback is pinned.
"""

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _identity_dit_attention(dim: int = 8):
    from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTAttention

    attn = DiTAttention(dim=dim, heads=1, dim_head=dim, dropout=0.0)
    with torch.no_grad():
        eye = torch.eye(dim)
        attn.to_q.weight.copy_(eye)
        attn.to_q.bias.zero_()
        attn.to_k.weight.copy_(eye)
        attn.to_k.bias.zero_()
        attn.to_v.weight.copy_(eye)
        attn.to_v.bias.zero_()
        attn.to_out[0].weight.copy_(eye)
        attn.to_out[0].bias.zero_()
    attn.eval()
    return attn


class TestSubsequentChunkMask:
    def test_example_from_docstring(self):
        from vllm_omni.model_executor.models.cosyvoice3.utils import subsequent_chunk_mask

        mask = subsequent_chunk_mask(size=4, chunk_size=2)
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ]
        )
        assert torch.equal(mask, expected)

    def test_chunk_size_one_allows_past_and_self(self):
        from vllm_omni.model_executor.models.cosyvoice3.utils import subsequent_chunk_mask

        mask = subsequent_chunk_mask(size=3, chunk_size=1)
        # Subsequent-chunk: attend to earlier chunks and the current chunk.
        expected = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ]
        )
        assert torch.equal(mask, expected)


class TestAddOptionalChunkMask:
    def test_static_chunk_and_pad_mask(self):
        from vllm_omni.model_executor.models.cosyvoice3.utils import add_optional_chunk_mask

        xs = torch.zeros(1, 4, 2)
        pad = torch.tensor([[[True, True, True, False]]])
        mask = add_optional_chunk_mask(xs, pad, False, False, 0, 2, -1)
        # Last key is padding, so no query may attend to it.
        assert mask.shape[-1] == 4
        assert not mask[..., 3].any()
        # First chunk (positions 0-1) cannot attend into the second chunk.
        assert not mask[0, 0, 2]
        assert not mask[0, 1, 2]

    def test_non_streaming_keeps_padding_only(self):
        from vllm_omni.model_executor.models.cosyvoice3.utils import add_optional_chunk_mask

        xs = torch.zeros(1, 4, 2)
        pad = torch.tensor([[[True, True, True, False]]])
        mask = add_optional_chunk_mask(xs, pad, False, False, 0, 0, -1)
        assert torch.equal(mask, pad)


class TestDiTAttentionMaskSemantics:
    def test_padding_mask_is_forwarded_to_diffusion_attention(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTAttention

        class CaptureAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_metadata = None

            def forward(self, q, k, v, attn_metadata=None):
                self.last_metadata = attn_metadata
                return torch.zeros_like(q)

        attention = DiTAttention(dim=16, heads=2, dim_head=8, dropout=0.0)
        capture = CaptureAttention()
        attention.attn = capture

        x = torch.randn(2, 5, 16)
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
            ]
        )
        out = attention(x, mask=mask)

        assert out.shape == x.shape
        assert capture.last_metadata is not None
        assert torch.equal(capture.last_metadata.attn_mask, mask)
        assert torch.allclose(out[~mask], torch.zeros_like(out[~mask]))

    def test_chunk_mask_blocks_future_keys_inside_softmax(self):
        attn = _identity_dit_attention(dim=8)
        x = torch.ones(1, 2, 8)
        x[0, 1] = 50.0
        # Query 0 may attend only to key 0.
        chunk_mask = torch.tensor([[[[True, False], [True, True]]]])

        out = attn(x, mask=chunk_mask)

        # If the mask were only applied after softmax, query 0 would mix in 50.
        assert out[0, 0].mean().item() < 5.0
        assert torch.allclose(out[0, 0], torch.ones(8), atol=1e-5)

    def test_full_qk_mask_pins_math_sdpa_backend(self, monkeypatch):
        from torch.nn.attention import SDPBackend

        from vllm_omni.diffusion.models.cosyvoice3_audio import cosyvoice3_dit as dit_mod

        if dit_mod.sdpa_kernel is None:
            pytest.skip("torch.nn.attention.sdpa_kernel is unavailable")

        seen: list[object] = []
        real = dit_mod.sdpa_kernel

        def wrapped(backend):
            seen.append(backend)
            return real(backend)

        monkeypatch.setattr(dit_mod, "sdpa_kernel", wrapped)

        attn = _identity_dit_attention(dim=8)
        x = torch.randn(1, 2, 8)
        chunk_mask = torch.tensor([[[[True, False], [True, True]]]])
        attn(x, mask=chunk_mask)

        assert seen == [SDPBackend.MATH]


class TestCFMStreamingPassthrough:
    def test_causal_cfm_forwards_streaming_flag(self):
        from omegaconf import DictConfig

        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

        class DummyEstimator(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_streaming = None

            def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
                self.last_streaming = streaming
                return torch.zeros_like(x)

        estimator = DummyEstimator()
        cfm = CausalConditionalCFM(
            in_channels=80,
            cfm_params=DictConfig(
                {
                    "sigma_min": 1e-6,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                }
            ),
            n_spks=1,
            spk_emb_dim=80,
            estimator=estimator,
        )
        batch, mel_dim, seq_len = 1, 80, 8
        mu = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        cfm(mu, mask, n_timesteps=1, spks=spks, cond=cond, streaming=True)

        assert estimator.last_streaming is True
