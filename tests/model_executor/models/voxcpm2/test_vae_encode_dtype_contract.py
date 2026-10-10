# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the bf16 AudioVAE encode dtype contract.

The bf16 VAE cast wraps ``audio_vae.encode`` so it accepts fp32 waveforms
and returns float32 features. ``_build_prefill_inputs`` concatenates the
cached features with fp32 zero padding for voice-clone / continuation /
ICL prefill; a bf16 feature there would be silently type-promoted by
``torch.cat`` (no error on torch 2.x), drifting the cache to bf16
precision. These tests pin the explicit fp32 contract so dropping the
output ``.float()`` cast (or letting a native cache recast to the bf16
parameter dtype) is caught by the dtype assertion rather than silently.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


torch = pytest.importorskip("torch")


@functools.lru_cache(maxsize=1)
def _voxcpm2_talker_mod():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker as mod

    return mod


class _FakeBf16VAE(torch.nn.Module):
    """Minimal AudioVAE stand-in with bf16 parameters and bf16 encode output."""

    latent_dim = 4
    patch_size = 2

    def __init__(self) -> None:
        super().__init__()
        self._probe = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def encode(self, audio_data: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # bf16 latents, as the real bf16 AudioVAE would produce.
        return torch.zeros(1, self.latent_dim, 8, dtype=torch.bfloat16)


def _make_fake_tts():
    """A fake native ``tts`` exposing the attributes _encode_raw_audio needs."""
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        _install_dtype_aware_encode,
    )

    vae = _FakeBf16VAE()
    tts = SimpleNamespace(
        audio_vae=vae,
        _encode_sample_rate=16000,
        patch_size=2,
        chunk_size=4,
    )
    # Install the production wrapper exactly as __init__ does.
    _install_dtype_aware_encode(tts, next(vae.parameters()).dtype)
    return tts


def test_wrapped_encode_returns_float32_features() -> None:
    mod = _voxcpm2_talker_mod()
    tts = _make_fake_tts()

    samples = [0.01 * (i % 7) for i in range(64)]
    feat = mod._encode_raw_audio(tts, samples, sr=16000)

    assert feat.dtype == torch.float32, (
        "encode wrapper must restore the fp32 feature contract; a bf16 "
        "feature would break _build_prefill_inputs torch.cat with fp32 padding"
    )
    assert torch.isfinite(feat).all()


def test_prefill_inputs_accepts_wrapped_encode_features() -> None:
    VoxCPM2TalkerForConditionalGeneration = _voxcpm2_talker_mod().VoxCPM2TalkerForConditionalGeneration
    tts = _make_fake_tts()

    samples = [0.01 * (i % 7) for i in range(64)]
    audio_feat = _voxcpm2_talker_mod()._encode_raw_audio(tts, samples, sr=16000)
    assert audio_feat.dtype == torch.float32

    side_dtype = torch.float32
    fake_self = SimpleNamespace(
        tts=SimpleNamespace(
            text_tokenizer=lambda text: [1, 2, 3],
            audio_start_token=999,
            audio_vae=tts.audio_vae,
        ),
        _patch_size=2,
        _side_dtype=side_dtype,
        _active_states={
            "req": SimpleNamespace(
                prompt_cache={
                    "mode": "continuation",
                    "prompt_text": "",
                    "audio_feat": audio_feat,
                }
            )
        },
    )

    # Pins the end-to-end contract: cache features produced through the
    # wrapped encode keep the fp32 dtype _build_prefill_inputs expects.
    result = VoxCPM2TalkerForConditionalGeneration._build_prefill_inputs(
        fake_self, [10, 11], torch.device("cpu"), "req"
    )

    assert result.audio_feat.dtype == side_dtype
    assert torch.isfinite(result.audio_feat).all()


def test_unwrapped_vae_leaks_bf16_into_cache() -> None:
    """Documents why the wrapper exists: the bare bf16 VAE emits bf16
    latents, which would silently change the cached prompt-feature dtype.
    torch.cat type-promotes fp32+bf16 on torch 2.x (verified on CPU and
    Ascend NPU), so the failure mode is silent dtype drift, not a crash;
    the wrapper restores the historical fp32 cache contract explicitly."""
    vae = _FakeBf16VAE()
    out = vae.encode(torch.zeros(1, 1, 64), 16000)
    assert out.dtype == torch.bfloat16

    bf16_feat = out.reshape(1, vae.patch_size, -1)[..., : vae.latent_dim]
    promoted = torch.cat([torch.zeros(1, vae.patch_size, vae.latent_dim), bf16_feat])
    assert promoted.dtype == torch.float32
