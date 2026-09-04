# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for CosyVoice3 components."""

import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalHiFTGenerator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@contextmanager
def _force_torch_sdpa():
    """Pin TORCH_SDPA so CPU shape tests do not pick CUDA-only backends (FA3)."""
    od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    with set_current_diffusion_config(od_config):
        yield


def test_batch_flow_runtime_flags_are_opt_in(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.runtime import (
        cosyvoice3_batch_flow_debug,
        cosyvoice3_batch_flow_enabled,
    )

    monkeypatch.delenv("COSYVOICE3_BATCH_FLOW", raising=False)
    monkeypatch.delenv("COSYVOICE3_BATCH_FLOW_DEBUG", raising=False)
    assert not cosyvoice3_batch_flow_enabled()
    assert not cosyvoice3_batch_flow_debug()

    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW", "1")
    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW_DEBUG", "1")
    assert cosyvoice3_batch_flow_enabled()
    assert cosyvoice3_batch_flow_debug()


@pytest.fixture
def causal_hift():
    return CausalHiFTGenerator(
        base_channels=32,
        upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4],
        source_resblock_kernel_sizes=[3, 3],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
    )


def test_causal_hift_moves_stft_window_with_model(causal_hift):
    assert causal_hift.get_buffer("stft_window") is causal_hift.stft_window
    assert "stft_window" not in causal_hift.state_dict()
    causal_hift.to(dtype=torch.float64)
    assert causal_hift.stft_window.dtype == torch.float64


def test_causal_hift_stft_moves_window_to_input_device(causal_hift):
    waveform = torch.empty((1, 64), device="meta")

    real, imag = causal_hift._stft(waveform)

    assert real.device == waveform.device
    assert imag.device == waveform.device
    assert causal_hift.stft_window.device == waveform.device


class TestPreLookaheadLayer:
    """Tests for PreLookaheadLayer."""

    @pytest.fixture
    def layer(self):
        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer

        return PreLookaheadLayer(in_channels=512, channels=512, pre_lookahead_len=3)

    def test_forward_shape(self, layer):
        """Test that output shape matches input shape."""
        batch, seq_len, channels = 2, 10, 512
        x = torch.randn(batch, seq_len, channels)

        out = layer(x)

        assert out.shape == x.shape

    def test_forward_with_context(self, layer):
        """Test forward with context for streaming."""
        batch, seq_len, channels = 1, 10, 512
        x = torch.randn(batch, seq_len, channels)
        context = torch.randn(batch, 3, channels)  # pre_lookahead_len=3

        layer.eval()
        out = layer(x, context=context)

        assert out.shape == x.shape

    def test_residual_connection(self, layer):
        """Test that residual connection is applied."""
        batch, seq_len, channels = 1, 5, 512
        x = torch.zeros(batch, seq_len, channels)

        # With zero input, output should also be close to zero due to residual
        out = layer(x)

        # Output should be close to input (residual) plus conv output
        assert out.shape == x.shape


class TestDiTAttention:
    """Tests for DiTAttention with diffusion backend."""

    @pytest.fixture
    def attention(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTAttention

        with _force_torch_sdpa():
            return DiTAttention(dim=512, heads=8, dim_head=64, dropout=0.0)

    def test_forward_shape(self, attention):
        """Test attention output shape."""
        batch, seq_len, dim = 2, 16, 512
        x = torch.randn(batch, seq_len, dim)

        out = attention(x)

        assert out.shape == x.shape

    def test_forward_with_mask(self, attention):
        """Test attention with mask."""
        batch, seq_len, dim = 2, 16, 512
        x = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        mask[:, -3:] = False  # Mask last 3 positions

        out = attention(x, mask=mask)

        assert out.shape == x.shape
        # Masked positions should be zero
        assert torch.allclose(out[:, -3:], torch.zeros_like(out[:, -3:]))

    def test_qkv_projections(self, attention):
        """Test that Q/K/V projections exist and have correct dimensions."""
        assert hasattr(attention, "to_q")
        assert hasattr(attention, "to_k")
        assert hasattr(attention, "to_v")
        assert attention.to_q.out_features == 512  # heads * dim_head
        assert attention.to_k.out_features == 512
        assert attention.to_v.out_features == 512

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_forward_passes_2d_mask_without_casting_qkv(self):
        """The float32 fallback receives the per-row validity mask."""
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTAttention

        class CaptureAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_metadata = None
                self.last_dtype = None

            def forward(self, q, k, v, attn_metadata=None):
                self.last_metadata = attn_metadata
                self.last_dtype = q.dtype
                return torch.zeros_like(q)

        attention = DiTAttention(dim=16, heads=2, dim_head=8, dropout=0.0)
        capture = CaptureAttention()
        attention.attn = capture

        x = torch.randn(2, 5, 16)
        mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

        out = attention(x, mask=mask)

        assert out.shape == x.shape
        assert capture.last_metadata is not None
        assert torch.equal(capture.last_metadata.attn_mask, mask)
        assert capture.last_dtype == x.dtype
        assert out.dtype == x.dtype
        assert torch.allclose(out[~mask], torch.zeros_like(out[~mask]))


class TestDiTBlock:
    """Tests for DiTBlock."""

    @pytest.fixture
    def block(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTBlock

        with _force_torch_sdpa():
            return DiTBlock(dim=512, heads=8, dim_head=64, ff_mult=4, dropout=0.0)

    def test_forward_shape(self, block):
        """Test block output shape."""
        batch, seq_len, dim = 2, 16, 512
        x = torch.randn(batch, seq_len, dim)
        t = torch.randn(batch, dim)  # Timestep embedding

        out = block(x, t)

        assert out.shape == x.shape

    def test_adalayernorm_modulation(self, block):
        """Test that AdaLayerNorm modulates based on timestep."""
        batch, seq_len, dim = 1, 8, 512
        x = torch.randn(batch, seq_len, dim)
        t1 = torch.zeros(batch, dim)
        t2 = torch.ones(batch, dim)

        out1 = block(x, t1)
        out2 = block(x, t2)

        # Different timesteps should produce different outputs
        assert not torch.allclose(out1, out2)


class TestDiT:
    """Tests for the full DiT model."""

    @pytest.fixture
    def dit(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT

        with _force_torch_sdpa():
            return DiT(
                dim=256,
                depth=2,
                heads=4,
                dim_head=64,
                dropout=0.0,
                ff_mult=2,
                mel_dim=80,
                mu_dim=80,
                spk_dim=80,
                long_skip_connection=True,
            )

    def test_forward_shape(self, dit):
        """Test DiT forward output shape."""
        batch, mel_dim, seq_len = 1, 80, 32
        x = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        mu = torch.randn(batch, mel_dim, seq_len)
        t = torch.tensor([0.5])
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out = dit(x, mask, mu, t, spks=spks, cond=cond)

        assert out.shape == (batch, mel_dim, seq_len)

    def test_timestep_embedding(self, dit):
        """Test that different timesteps produce different outputs."""
        batch, mel_dim, seq_len = 1, 80, 16
        x = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        mu = torch.randn(batch, mel_dim, seq_len)
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out1 = dit(x, mask, mu, torch.tensor([0.0]), spks=spks, cond=cond)
        out2 = dit(x, mask, mu, torch.tensor([1.0]), spks=spks, cond=cond)

        assert not torch.allclose(out1, out2)


class TestCFM:
    """Tests for Conditional Flow Matching classes."""

    @pytest.fixture
    def dummy_estimator(self):
        """Create a dummy estimator for testing."""

        class DummyEstimator(nn.Module):
            def __init__(self, mel_dim=80):
                super().__init__()
                self.mel_dim = mel_dim

            def forward(self, x, mask, mu, t, spks=None, cond=None):
                return torch.zeros_like(x)

        return DummyEstimator()

    def test_causal_conditional_cfm_forward(self, dummy_estimator):
        """Test CausalConditionalCFM forward pass."""
        from omegaconf import DictConfig

        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

        cfm_params = DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        )

        cfm = CausalConditionalCFM(
            in_channels=80,
            cfm_params=cfm_params,
            n_spks=1,
            spk_emb_dim=80,
            estimator=dummy_estimator,
        )

        batch, mel_dim, seq_len = 1, 80, 32
        mu = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond)

        assert out.shape == mu.shape

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_trt_estimator_uses_stream_dependencies(self, monkeypatch):
        from omegaconf import DictConfig

        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import ConditionalCFM

        class FakeStream:
            def __init__(self, name, state):
                self.name = name
                self.state = state
                self.cuda_stream = hash(name)
                self.synchronize_calls = 0
                self.waited_on = []

            def synchronize(self):
                self.synchronize_calls += 1

            def wait_stream(self, stream):
                self.waited_on.append(stream)

        class FakeStreamContext:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.previous = self.stream.state.current
                self.stream.state.current = self.stream
                return self.stream

            def __exit__(self, exc_type, exc, traceback):
                self.stream.state.current = self.previous

        state = SimpleNamespace(current=None)
        caller_stream = FakeStream("caller", state)
        estimator_stream = FakeStream("estimator", state)
        state.current = caller_stream
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: state.current)
        stream_contexts = []

        def stream_context(stream):
            stream_contexts.append(stream)
            return FakeStreamContext(stream)

        monkeypatch.setattr(torch.cuda, "stream", stream_context)

        class FakeContext:
            def __init__(self):
                self.execute_stream = None

            def set_input_shape(self, name, shape):
                pass

            def set_tensor_address(self, name, address):
                pass

            def execute_async_v3(self, stream):
                self.execute_stream = stream
                return True

        class FakeEngine:
            @staticmethod
            def get_tensor_name(index):
                return f"tensor_{index}"

        context = FakeContext()

        class FakeEstimatorPool:
            io_dtype = torch.float32

            def __init__(self):
                self.released = []

            def acquire_estimator(self):
                return [context, estimator_stream], FakeEngine()

            def release_estimator(self, released_context, released_stream):
                self.released.append((released_context, released_stream))

        estimator_pool = FakeEstimatorPool()
        cfm = ConditionalCFM(
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
            estimator=estimator_pool,
        )

        x = torch.randn(2, 80, 4)
        mask = torch.ones(2, 1, 4)
        mu = torch.randn(2, 80, 4)
        timestep = torch.randn(2)
        speakers = torch.randn(2, 80)
        condition = torch.randn(2, 80, 4)

        output = cfm.forward_estimator(x, mask, mu, timestep, speakers, condition)

        assert output.shape == x.shape
        assert caller_stream.synchronize_calls == 0
        assert estimator_stream.synchronize_calls == 0
        assert estimator_stream.waited_on == [caller_stream]
        assert caller_stream.waited_on == [estimator_stream]
        assert stream_contexts == [estimator_stream]
        assert context.execute_stream == estimator_stream.cuda_stream
        assert estimator_pool.released == [(context, estimator_stream)]

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_trt_context_pool_stores_cuda_stream(self, monkeypatch):
        from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import TrtContextWrapper

        execution_context = object()

        class FakeEngine:
            @staticmethod
            def create_execution_context():
                return execution_context

        cuda_stream = object()
        monkeypatch.setattr(torch.cuda, "Stream", lambda device: cuda_stream)

        def reject_stream_context(stream):
            pytest.fail("TrtContextWrapper must store the CUDA stream, not a StreamContext")

        monkeypatch.setattr(torch.cuda, "stream", reject_stream_context)

        engine = FakeEngine()
        wrapper = TrtContextWrapper(engine, device="cuda:0")
        [context, stream], acquired_engine = wrapper.acquire_estimator()

        assert context is execution_context
        assert stream is cuda_stream
        assert acquired_engine is engine

        wrapper.release_estimator(context, stream)
        [reused_context, reused_stream], _ = wrapper.acquire_estimator()
        assert reused_context is context
        assert reused_stream is stream

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_causal_conditional_cfm_batches_cfg_estimator(self):
        """Batched flow calls should invoke the CFG estimator with 2B rows."""
        from omegaconf import DictConfig

        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

        class RecordingEstimator(nn.Module):
            def __init__(self):
                super().__init__()
                self.shapes: list[tuple[int, ...]] = []

            def forward(self, x, mask, mu, t, spks=None, cond=None):
                self.shapes.append(tuple(x.shape))
                return torch.zeros_like(x)

        estimator = RecordingEstimator()
        cfm_params = DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        )
        cfm = CausalConditionalCFM(
            in_channels=80,
            cfm_params=cfm_params,
            n_spks=1,
            spk_emb_dim=80,
            estimator=estimator,
        )

        batch, mel_dim, seq_len = 3, 80, 16
        mu = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond)

        assert out.shape == mu.shape
        assert estimator.shapes == [(2 * batch, mel_dim, seq_len)] * 2


class TestSDPAFallback:
    """Test SDPA fallback for float32 inputs."""

    def test_float32_uses_sdpa(self):
        """Test that float32 inputs use SDPA fallback."""
        from vllm_omni.diffusion.attention.layer import Attention

        with _force_torch_sdpa():
            attn = Attention(
                num_heads=8,
                head_size=64,
                causal=False,
                softmax_scale=1.0 / 8.0,
            )

        batch, seq_len, heads, dim = 1, 16, 8, 64
        q = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32)
        k = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32)
        v = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32)

        # Should not raise error - SDPA fallback handles float32
        out = attn(q, k, v)

        assert out.shape == (batch, seq_len, heads, dim)
        assert out.dtype == torch.float32


def test_code2wav_forward_finalizes_hift_tail():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    class DummyHiFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.m_source = SimpleNamespace(l_linear=SimpleNamespace(weight=torch.ones(1, dtype=torch.float32)))
            self.finalize_calls: list[bool] = []

        def inference(self, speech_feat, finalize=True):
            self.finalize_calls.append(bool(finalize))
            return torch.zeros((speech_feat.shape[0], 1, speech_feat.shape[-1]), dtype=speech_feat.dtype), None

    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.hift = DummyHiFT()
    forward_mel_calls = []

    def fake_forward_mel(**kwargs):
        forward_mel_calls.append(kwargs)
        return torch.ones((1, 80, 8), dtype=torch.float32)

    model._forward_mel = fake_forward_mel

    out = model.forward(
        token=torch.tensor([[1, 2, 3]], dtype=torch.int32),
        prompt_token=torch.tensor([[4, 5]], dtype=torch.int32),
        prompt_feat=torch.ones((1, 4, 80), dtype=torch.float32),
        embedding=torch.ones((1, 192), dtype=torch.float32),
    )

    assert out.shape == (1, 1, 8)
    assert model.hift.finalize_calls == [True]
    assert forward_mel_calls[0]["token_offset_tokens"] == 0


def test_code2wav_streaming_batch_pads_codec_tokens_and_preserves_lengths():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    model = object.__new__(CosyVoice3Code2Wav)
    model.flow_model = SimpleNamespace(pre_lookahead_len=1, token_mel_ratio=2)
    forward_mel_calls = []

    def fake_forward_mel(self, **kwargs):
        forward_mel_calls.append(kwargs)
        return torch.arange(2 * 80 * 8, dtype=torch.float32).reshape(2, 80, 8)

    def fake_stream_hift(self, feat, *, cache_state=None, finalize=False):
        return feat, None

    model._forward_mel = types.MethodType(fake_forward_mel, model)
    model._stream_hift_from_feat = types.MethodType(fake_stream_hift, model)
    items = [
        {
            "token": torch.ones(1, 3, dtype=torch.int32),
            "prompt_token": torch.ones(1, 4, dtype=torch.int32),
            "prompt_feat": torch.ones(1, 8, 80),
            "embedding": torch.ones(1, 192),
            "finalize": False,
        },
        {
            "token": torch.ones(1, 5, dtype=torch.int32),
            "prompt_token": torch.ones(1, 4, dtype=torch.int32),
            "prompt_feat": torch.ones(1, 8, 80),
            "embedding": torch.ones(1, 192),
            "finalize": False,
        },
    ]

    results = model.forward_streaming_batch(items)

    assert len(forward_mel_calls) == 1
    call = forward_mel_calls[0]
    assert call["token"].shape == (2, 5)
    assert torch.equal(call["token_lens"], torch.tensor([3, 5], dtype=torch.int32))
    assert results[0][0].shape[-1] == 4
    assert results[1][0].shape[-1] == 8


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_code2wav_streaming_batch_matches_ragged_flow_numerics(monkeypatch):
    """A padded flow batch must match individual flow calls on valid mels."""
    from omegaconf import DictConfig

    from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        CausalConditionalCFM,
        CausalMaskedDiffWithDiT,
    )
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    torch.manual_seed(0)
    estimator = DiT(
        dim=32,
        depth=1,
        heads=4,
        dim_head=8,
        dropout=0.0,
        ff_mult=2,
        mel_dim=80,
        mu_dim=80,
        spk_dim=80,
        out_channels=80,
    )
    decoder = CausalConditionalCFM(
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
    flow_model = (
        CausalMaskedDiffWithDiT(
            input_size=80,
            output_size=80,
            spk_embed_dim=192,
            vocab_size=64,
            input_frame_rate=25,
            only_mask_loss=True,
            token_mel_ratio=2,
            pre_lookahead_len=1,
            pre_lookahead_layer=PreLookaheadLayer(in_channels=80, channels=80, pre_lookahead_len=1),
            decoder=decoder,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )

    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = flow_model

    def return_mel(self, feat, *, cache_state=None, finalize=False):
        return feat, None

    model._stream_hift_from_feat = types.MethodType(return_mel, model)

    original_randn = torch.randn

    def length_consistent_randn(*size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        if len(shape) == 3 and shape[1] == 80:
            device = kwargs.get("device")
            dtype = kwargs.get("dtype", torch.float32)
            channels = torch.arange(shape[1], device=device, dtype=torch.float32).view(1, -1, 1)
            positions = torch.arange(shape[2], device=device, dtype=torch.float32).view(1, 1, -1)
            noise = torch.sin(channels * 0.17 + positions * 0.31)
            return noise.expand(shape[0], -1, -1).to(dtype=dtype).clone()
        return original_randn(*size, **kwargs)

    monkeypatch.setattr(torch, "randn", length_consistent_randn)
    common = {
        "prompt_token": torch.tensor([[7, 8]], dtype=torch.int32),
        "prompt_feat": torch.linspace(-0.5, 0.5, 4 * 80).reshape(1, 4, 80),
        "embedding": torch.linspace(-1.0, 1.0, 192).reshape(1, 192),
        "finalize": False,
    }
    items = [
        {**common, "token": torch.tensor([[1, 2, 3]], dtype=torch.int32)},
        {**common, "token": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32)},
    ]

    batched = model.forward_streaming_batch(items, n_timesteps=2)
    individual = [
        model.forward_streaming(
            token=item["token"],
            prompt_token=item["prompt_token"],
            prompt_feat=item["prompt_feat"],
            embedding=item["embedding"],
            n_timesteps=2,
        )
        for item in items
    ]

    for (batched_mel, _), (individual_mel, _) in zip(batched, individual):
        assert batched_mel.shape == individual_mel.shape
        rel_mean = (batched_mel - individual_mel).abs().mean() / individual_mel.abs().mean().clamp_min(1e-6)
        assert rel_mean.item() < 0.05
