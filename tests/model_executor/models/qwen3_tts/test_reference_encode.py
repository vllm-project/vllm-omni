# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference-audio encode helpers keep FP32 state independent of the caller's torch defaults."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_mel_caches_stay_fp32_when_first_built_under_bf16_default():
    from vllm_omni.model_executor.models.qwen3_tts import prompt_embeds_builder as builder

    builder._cached_mel_filter_bank.cache_clear()
    builder._cached_hann_window.cache_clear()
    default_dtype = torch.get_default_dtype()
    try:
        # Model loading runs under the model dtype; a first call there must not
        # cache BF16 constants that later FP32 requests multiply against.
        torch.set_default_dtype(torch.bfloat16)
        assert builder._cached_mel_filter_bank(24000, 1024, 128, 0, 12000).dtype == torch.float32
        assert builder._cached_hann_window(1024).dtype == torch.float32
    finally:
        torch.set_default_dtype(default_dtype)
    wav = torch.randn(1, 24000)
    mels = builder.mel_spectrogram(
        wav, n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000
    )
    assert mels.dtype == torch.float32 and mels.shape[1] == 128


def test_exact_fp32_reference_encode_restores_global_flags():
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import exact_fp32_reference_encode

    tf32, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.enabled
    default_dtype = torch.get_default_dtype()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_default_dtype(torch.bfloat16)
    try:
        with exact_fp32_reference_encode():
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.enabled
            assert torch.get_default_dtype() == torch.float32
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.enabled == cudnn
        assert torch.get_default_dtype() == torch.bfloat16
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.set_default_dtype(default_dtype)
