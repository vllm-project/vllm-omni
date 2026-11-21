# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The audio tokenizer stage emits its sampling rate under ``SAMPLE_RATE_KEY``."""

import functools
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# An unusual rate proves the emission reads the audio tokenizer (whose rate
# comes from codec_args config) instead of a literal.
UNUSUAL_RATE = 22050


@functools.lru_cache(maxsize=1)
def _voxtral_tts_model_cls():
    # ``vllm_omni...voxtral_tts`` does ``from vllm.model_executor.models.utils import ...``.
    # If that module was first pulled in under a different import context in the
    # same process, ``direct_register_custom_op`` for ``sequence_parallel_chunk_impl``
    # can run twice.  Match ``tests.model_executor.helpers`` + pin ``utils`` first.
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    import vllm.model_executor.models.utils  # noqa: F401

    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts import (
        VoxtralTTSForConditionalGeneration,
    )
    from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

    return VoxtralTTSForConditionalGeneration, SAMPLE_RATE_KEY, get_sampling_rate


def _bare_audio_tokenizer_model():
    model_cls, _, _ = _voxtral_tts_model_cls()
    model = object.__new__(model_cls)
    nn.Module.__init__(model)
    model.model_stage = "audio_tokenizer"
    model.audio_tokenizer = SimpleNamespace(
        num_codebooks=4,
        decode_helper_batch_async=lambda tokens: [torch.zeros(8) for _ in tokens],
        downsample_factor=1,
        sampling_rate=UNUSUAL_RATE,
    )
    return model


def test_audio_tokenizer_dummy_path_emits_config_derived_sample_rate() -> None:
    _, sample_rate_key, get_sampling_rate = _voxtral_tts_model_cls()
    model = _bare_audio_tokenizer_model()

    out = model.forward(input_ids=torch.zeros(10, dtype=torch.long), positions=torch.arange(10))

    payload = out.multimodal_outputs
    assert sample_rate_key in payload
    assert get_sampling_rate(payload) == UNUSUAL_RATE
