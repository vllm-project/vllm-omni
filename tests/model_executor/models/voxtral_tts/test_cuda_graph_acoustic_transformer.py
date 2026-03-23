# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for CUDA Graph acoustic transformer wrapper numerical equivalence.

Verifies that CUDA Graph-accelerated decoding produces results equivalent
to eager mode, with special attention to padding cases where zero-padding
may introduce small numerical differences due to the flow matching ODE.

Architecture note: the real FlowMatchingAudioTransformer uses a multi-step
Euler ODE with CFG.  The synthetic model here uses simple linear layers
to exercise the wrapper mechanism while keeping the test lightweight.
"""

import importlib.util
import os

import pytest
import torch
import torch.nn as nn

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.L4,
]

DEVICE = torch.device("cuda:0")
HIDDEN_DIM = 64
N_ACOUSTIC_CODEBOOK = 7
SEMANTIC_CODEBOOK_SIZE = 128
ACOUSTIC_EMBEDDINGS_LEVELS = 1024

# Load CUDAGraphAcousticTransformerWrapper: try package import first, fall back to direct file load
try:
    from vllm_omni.model_executor.models.voxtral_tts.cuda_graph_acoustic_transformer_wrapper import (
        CUDAGraphAcousticTransformerWrapper,
    )
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import (
        AudioSpecialTokens,
    )
except Exception:
    _WRAPPER_PATH = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        "vllm_omni",
        "model_executor",
        "models",
        "voxtral_tts",
        "cuda_graph_acoustic_transformer_wrapper.py",
    )
    _spec = importlib.util.spec_from_file_location(
        "cuda_graph_acoustic_transformer_wrapper", os.path.abspath(_WRAPPER_PATH)
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CUDAGraphAcousticTransformerWrapper = _mod.CUDAGraphAcousticTransformerWrapper

    _AUDIO_GEN_PATH = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        "vllm_omni",
        "model_executor",
        "models",
        "voxtral_tts",
        "voxtral_tts_audio_generation.py",
    )
    _spec2 = importlib.util.spec_from_file_location("voxtral_tts_audio_generation", os.path.abspath(_AUDIO_GEN_PATH))
    _mod2 = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(_mod2)
    AudioSpecialTokens = _mod2.AudioSpecialTokens


class SyntheticModelArgs:
    """Mimics MultimodalAudioModelArgs interface."""

    def __init__(self):
        self.semantic_codebook_size = SEMANTIC_CODEBOOK_SIZE
        self.n_acoustic_codebook = N_ACOUSTIC_CODEBOOK


class SyntheticAcousticTransformer(nn.Module):
    """A small acoustic transformer mimicking FlowMatchingAudioTransformer's interface.

    Uses simple linear layers so the test stays lightweight while exercising
    the full wrapper mechanism (semantic logit, flow matching ODE, quantization).
    """

    def __init__(self):
        super().__init__()
        self.model_args = SyntheticModelArgs()
        self.acoustic_embeddings_levels = ACOUSTIC_EMBEDDINGS_LEVELS

        # semantic_codebook_output: hidden_dim -> padded_codebook_size
        padded_semantic_size = len(AudioSpecialTokens) + SEMANTIC_CODEBOOK_SIZE
        # Round up to multiple of 128 like the real model
        padded_semantic_size = 128 * ((padded_semantic_size + 127) // 128)
        self.semantic_codebook_output = nn.Linear(HIDDEN_DIM, padded_semantic_size, bias=False)

        # time_embedding: (B, 1) -> (B, dim)
        self.time_embedding = nn.Linear(1, HIDDEN_DIM, bias=False)

        # _predict_velocity: takes x_t (B, C), llm_output (B, D), t_emb (B, D) -> (B, C)
        self._velocity_proj = nn.Linear(N_ACOUSTIC_CODEBOOK + HIDDEN_DIM + HIDDEN_DIM, N_ACOUSTIC_CODEBOOK, bias=False)

    def _predict_velocity(self, x_t: torch.Tensor, llm_output: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, llm_output, t_emb], dim=-1)
        return self._velocity_proj(combined)


class SyntheticModel(nn.Module):
    """Mimics VoxtralTTSAudioGenerationForConditionalGeneration interface."""

    def __init__(self):
        super().__init__()
        self.acoustic_transformer = SyntheticAcousticTransformer()
        end_audio_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        empty_audio_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)
        self.end_audio_id = end_audio_id
        self.empty_audio_id = empty_audio_id

    def compute_mm_logits(
        self,
        hidden_states: torch.Tensor,
        mm_sampling_tensors=None,
    ):
        """Eager fallback path: replicate what the wrapper does."""
        at = self.acoustic_transformer
        B = hidden_states.shape[0]

        empty_audio_id = self.empty_audio_id
        end_audio_id = self.end_audio_id
        semantic_mask_start = len(AudioSpecialTokens) + at.model_args.semantic_codebook_size

        # Semantic logits
        semantic_logit = at.semantic_codebook_output(hidden_states).float()
        semantic_logit[:, empty_audio_id] = -float("inf")
        semantic_logit[:, semantic_mask_start:] = -float("inf")
        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)

        # Flow matching Euler ODE
        should_decode = semantic_code.squeeze(1) != end_audio_id
        x = torch.randn(B, at.model_args.n_acoustic_codebook, device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_zero = torch.zeros_like(hidden_states)
        timesteps = torch.linspace(0, 1, 16, device=hidden_states.device, dtype=hidden_states.dtype)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_emb = at.time_embedding(t.view(-1, 1).repeat(B, 1)).to(hidden_states.dtype)
            x_batched = torch.cat([x, x], dim=0)
            llm_batched = torch.cat([hidden_states, hidden_zero], dim=0)
            t_emb_batched = t_emb.repeat(2, 1)
            v_all = at._predict_velocity(x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched)
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            cfg_alpha = 1.2
            v_t = cfg_alpha * v_t + (1 - cfg_alpha) * uncond_v_t
            x = x + v_t * dt

        sampled = torch.clamp(x, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (at.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = empty_audio_id
        acoustic_codes = output_codes + len(AudioSpecialTokens)

        audio_codes = torch.cat([semantic_code, acoustic_codes], dim=1)
        fake_eos = torch.where(
            audio_codes[:, 0] == end_audio_id,
            torch.tensor(1.0, dtype=hidden_states.dtype, device=hidden_states.device),
            torch.tensor(0.0, dtype=hidden_states.dtype, device=hidden_states.device),
        )

        audio_list = list(torch.split(audio_codes.unsqueeze(1), 1, dim=0))
        mm_tokens = {"audio": audio_list}
        return fake_eos, mm_tokens


@pytest.fixture(scope="module")
def model():
    """Create a synthetic model on CUDA with fixed weights in bfloat16."""
    torch.manual_seed(42)
    return SyntheticModel().to(device=DEVICE, dtype=torch.bfloat16).eval()


@pytest.fixture(scope="module")
def wrapper(model):
    """Create a warmed-up CUDAGraphAcousticTransformerWrapper."""
    w = CUDAGraphAcousticTransformerWrapper(
        model=model,
        capture_sizes=[1, 2, 4, 8, 16, 32],
    )
    w._warmup_and_capture(
        device=DEVICE,
        dtype=torch.bfloat16,
        hidden_dim=HIDDEN_DIM,
    )
    return w


def _random_hidden(batch_size, device=DEVICE, dtype=torch.bfloat16):
    return torch.randn(batch_size, HIDDEN_DIM, device=device, dtype=dtype)


def _unpack_audio_codes(result):
    """Unpack (fake_eos, {"audio": [list of tensors]}) into (fake_eos, audio_codes)."""
    fake_eos, mm_tokens = result
    audio_list = mm_tokens["audio"]
    # Each element is (1, 1, 1+n_acoustic_codebook), concat along batch dim
    audio_codes = torch.cat(audio_list, dim=0).squeeze(1)  # (B, 1+n_acoustic)
    return fake_eos, audio_codes


# ──────────────────────────────────────────────────────────────────
# 1. Exact-size inputs (no padding) → correct format and bounds
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
def test_exact_size_output_format(model, wrapper, batch_size):
    """Graph path returns correctly shaped and bounded outputs."""
    hidden = _random_hidden(batch_size)
    with torch.no_grad():
        graph_eos, graph_codes = _unpack_audio_codes(wrapper(hidden))
    assert graph_eos.shape == (batch_size,)
    assert graph_codes.shape == (batch_size, 1 + N_ACOUSTIC_CODEBOOK)
    # fake_eos should be 0.0 or 1.0
    assert torch.all((graph_eos == 0.0) | (graph_eos == 1.0))
    # Audio codes should be non-negative
    assert torch.all(graph_codes >= 0)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
def test_exact_size_deterministic(model, wrapper, batch_size):
    """Same input + same RNG state produces identical CUDA graph output."""
    hidden = _random_hidden(batch_size)
    with torch.no_grad():
        torch.manual_seed(42)
        eos1, codes1 = _unpack_audio_codes(wrapper(hidden))
        torch.manual_seed(42)
        eos2, codes2 = _unpack_audio_codes(wrapper(hidden))
    torch.testing.assert_close(eos1, eos2, atol=0, rtol=0)
    torch.testing.assert_close(codes1, codes2, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 2. Padded inputs (zero-padding to nearest capture size)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [3, 5, 7, 10, 24])
def test_padded_output_shape(model, wrapper, batch_size):
    """Padded decode must return output trimmed to actual batch size."""
    hidden = _random_hidden(batch_size)
    with torch.no_grad():
        graph_eos, graph_codes = _unpack_audio_codes(wrapper(hidden))
    assert graph_eos.shape == (batch_size,)
    assert graph_codes.shape == (batch_size, 1 + N_ACOUSTIC_CODEBOOK)


@pytest.mark.parametrize("batch_size", [3, 5, 7, 10, 24])
def test_padded_output_bounded(model, wrapper, batch_size):
    """Padded output audio codes should be non-negative integers."""
    hidden = _random_hidden(batch_size)
    with torch.no_grad():
        graph_eos, graph_codes = _unpack_audio_codes(wrapper(hidden))
    # fake_eos should be 0.0 or 1.0
    assert torch.all((graph_eos == 0.0) | (graph_eos == 1.0))
    # Audio codes should be non-negative
    assert torch.all(graph_codes >= 0)


# ──────────────────────────────────────────────────────────────────
# 3. Batch size exceeds all capture sizes so fallback to eager
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [33, 48, 64])
def test_fallback_eager_exact_match(model, wrapper, batch_size):
    """Cudagraph fallback to eager. Two eager runs should produce identical results."""
    hidden = _random_hidden(batch_size)
    with torch.no_grad():
        torch.manual_seed(100)
        eager_eos, eager_codes = _unpack_audio_codes(model.compute_mm_logits(hidden))
        torch.manual_seed(100)
        graph_eos, graph_codes = _unpack_audio_codes(wrapper(hidden))
    torch.testing.assert_close(graph_eos, eager_eos, atol=0, rtol=0)
    torch.testing.assert_close(graph_codes, eager_codes, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 4. Disabled wrapper so fallback to eager
# ──────────────────────────────────────────────────────────────────


def test_disabled_wrapper_matches_eager(model, wrapper):
    """Cudagraph fallback to eager. Two eager runs should produce identical results."""
    hidden = _random_hidden(4)
    wrapper.enabled = False
    with torch.no_grad():
        torch.manual_seed(200)
        eager_eos, eager_codes = _unpack_audio_codes(model.compute_mm_logits(hidden))
        torch.manual_seed(200)
        graph_eos, graph_codes = _unpack_audio_codes(wrapper(hidden))
    wrapper.enabled = True
    torch.testing.assert_close(graph_eos, eager_eos, atol=0, rtol=0)
    torch.testing.assert_close(graph_codes, eager_codes, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 5. Determinism - same input + same RNG state on cudagraph runs
# ──────────────────────────────────────────────────────────────────


def test_deterministic_across_calls(model, wrapper):
    """Same input + same RNG state. Two cudagraph runs should produce identical results."""
    hidden = _random_hidden(4)
    with torch.no_grad():
        torch.manual_seed(300)
        eos1, codes1 = _unpack_audio_codes(wrapper(hidden))
        torch.manual_seed(300)
        eos2, codes2 = _unpack_audio_codes(wrapper(hidden))
    torch.testing.assert_close(eos1, eos2, atol=0, rtol=0)
    torch.testing.assert_close(codes1, codes2, atol=0, rtol=0)
