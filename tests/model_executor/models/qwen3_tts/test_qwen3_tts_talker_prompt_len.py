"""Regression tests for `estimate_prompt_len_from_additional_information`.

Pins the 2D `voice_clone_prompt.ref_code` shape behaviour. Applying the
singleton-batch unwrapper `_first(...)` to that value strips its outer
dimension and reports `len(ref_code) == num_codebooks` instead of
`num_frames`, which silently truncates `inputs_embeds` downstream in
`_build_prompt_embeds`.
"""

import pytest

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fake_tokenize(text: str, **_kwargs):
    return [0] * (8 + max(1, len(text.split())))


def test_estimate_prompt_len_uses_full_ref_code_length() -> None:
    num_frames = 318
    num_codebooks = 8
    info = {
        "task_type": ["Base"],
        "text": ["hello"],
        "ref_text": ["world"],
        "voice_clone_prompt": [
            {
                "ref_spk_embedding": [0.0] * 512,
                "ref_code": [[0] * num_codebooks for _ in range(num_frames)],
                "icl_mode": True,
            }
        ],
        "non_streaming_mode": [True],
        "language": ["English"],
    }

    est = Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
        additional_information=info,
        task_type="Base",
        tokenize_prompt=_fake_tokenize,
        codec_language_id={"english": 0},
        spk_is_dialect=None,
    )

    # codec_lens = 1 + num_frames = 319; plus text-side and codec-prefix
    # terms ~20. Would be ~30 if `_first` collapses ref_code to its first row.
    assert est > 100, f"got {est}; expected ~339. Did `_first(ref_code)` collapse the 2D list again?"


def test_estimate_prompt_len_handles_1d_ref_code() -> None:
    num_frames = 50
    info = {
        "task_type": ["Base"],
        "text": ["hello"],
        "ref_text": ["world"],
        "voice_clone_prompt": [
            {
                "ref_spk_embedding": [0.0] * 512,
                "ref_code": list(range(num_frames)),
                "icl_mode": True,
            }
        ],
        "non_streaming_mode": [True],
        "language": ["English"],
    }

    est = Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
        additional_information=info,
        task_type="Base",
        tokenize_prompt=_fake_tokenize,
        codec_language_id={"english": 0},
        spk_is_dialect=None,
    )

    assert est > 50, f"got {est}; 1D ref_code must contribute its own length"
