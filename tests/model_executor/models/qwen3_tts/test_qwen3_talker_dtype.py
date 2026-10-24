# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression test: Qwen3-TTS embedding dtype must follow model_config.dtype.

``_embedding_dtype`` (talker and prompt builder) used to be hardcoded to
``torch.bfloat16``, so serving with ``--dtype=half`` could mix incompatible
tensor dtypes.
"""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MOD = "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker"

# Heavy collaborators of Qwen3TTSTalkerForConditionalGeneration.__init__.
# Keep Qwen3TTSPromptEmbedsBuilder real so its hardcoded bf16 default is
# actually exercised by this regression test.
_PATCHED_NAMES = [
    "Qwen3Model",
    "ParallelLMHead",
    "LogitsProcessor",
    "Qwen3TTSTalkerResizeMLP",
    "Qwen3TTSSpeakerEncoder",
    "Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM",
    "Qwen3TTSTokenizerV2Config",
    "Qwen3TTSTokenizerV2Encoder",
    "AutoFeatureExtractor",
    "get_speaker_cache",
    "maybe_prefix",
    "talker_first_audio_enabled",
    "_qwen3_tts_gpu_resident_buffer_keys",
]


def _make_vllm_config(dtype: torch.dtype):
    talker_config = SimpleNamespace(
        code_predictor_config=SimpleNamespace(vocab_size=16),
        codec_eos_token_id=3,
        hidden_size=8,
        num_code_groups=4,
        vocab_size=32,
        text_vocab_size=10,
        text_hidden_size=8,
        hidden_act="silu",
    )
    model_config = SimpleNamespace(
        model="dummy/path",
        hf_config=SimpleNamespace(
            talker_config=talker_config,
            speaker_encoder_config=None,
        ),
        dtype=dtype,
        silence_ban_frames=0,
        use_v2_model_runner=False,
        subtalker_sampling_params=None,
    )
    return SimpleNamespace(
        model_config=model_config,
        compilation_config=SimpleNamespace(static_forward_context=None),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        quant_config=None,
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_talker_and_prompt_builder_dtype_follow_model_dtype(
    mocker: MockerFixture,
    dtype: torch.dtype,
) -> None:
    talker_mod = importlib.import_module(_MOD)

    for name in _PATCHED_NAMES:
        mocker.patch(f"{_MOD}.{name}")

    mocker.patch(
        f"{_MOD}.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    )
    mocker.patch(
        "vllm.config.vllm.set_current_vllm_config",
        return_value=nullcontext(),
    )
    mocker.patch.object(
        talker_mod.Qwen3TTSTalkerForConditionalGeneration,
        "_load_custom_voice_profiles",
    )

    talker = talker_mod.Qwen3TTSTalkerForConditionalGeneration(
        vllm_config=_make_vllm_config(dtype),
    )

    # Regression: talker used to be pinned to torch.bfloat16.
    assert talker._embedding_dtype == dtype

    # Regression: the real prompt builder still initializes its dtype to
    # torch.bfloat16, so the talker must explicitly propagate model_dtype
    # after constructing it.
    assert talker._prompt_builder._embedding_dtype == dtype

    # The constant pad embedding must agree with the selected model dtype.
    assert talker._tts_pad_embed.dtype == dtype
