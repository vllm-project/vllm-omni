"""Regression tests for `estimate_prompt_len_from_additional_information`.

Pins the 2D `voice_clone_prompt.ref_code` shape behaviour. Applying the
singleton-batch unwrapper `_first(...)` to that value strips its outer
dimension and reports `len(ref_code) == num_codebooks` instead of
`num_frames`, which silently truncates `inputs_embeds` downstream in
`_build_prompt_embeds`.
"""

import pytest

from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
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

    est = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
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

    est = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=info,
        task_type="Base",
        tokenize_prompt=_fake_tokenize,
        codec_language_id={"english": 0},
        spk_is_dialect=None,
    )

    assert est > 50, f"got {est}; 1D ref_code must contribute its own length"


def test_estimate_prompt_len_uses_ref_code_length_without_ref_audio() -> None:
    est = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information={
            "task_type": ["Base"],
            "text": ["hello"],
            "ref_text": ["reference transcript"],
            "x_vector_only_mode": [False],
            "ref_code_length": [6],
            "_qwen3_tts_ref_audio_cache_key": ["same-ref"],
            "language": ["English"],
        },
        task_type="Base",
        tokenize_prompt=lambda _text: list(range(10)),
        codec_language_id=None,
        spk_is_dialect=None,
        estimate_ref_code_len=lambda _ref_audio: None,
    )

    assert est == 15


def test_estimate_prompt_len_uses_list_of_int_ref_ids_from_voice_clone_prompt() -> None:
    """`voice_clone_prompt.ref_ids` as a list-of-int (the natural pre-tokenized shape)
    must be read as a sequence, not unwrapped to its first element. Without the fix
    `_first` collapsed it to an int, the `isinstance(list)` / `hasattr("shape")`
    branches fell through, and `ref_ids_len` defaulted to 0 — silently dropping the
    reference-token contribution to the prefix length budget.
    """

    def make_info(num_ref_ids: int) -> dict:
        return {
            "task_type": ["Base"],
            "text": ["hello"],
            "ref_text": ["world"],
            "voice_clone_prompt": [
                {
                    "ref_spk_embedding": [0.0] * 512,
                    "ref_code": [[0] * 8 for _ in range(200)],
                    "ref_ids": list(range(num_ref_ids)),
                    "icl_mode": True,
                }
            ],
            "non_streaming_mode": [True],
            "language": ["English"],
        }

    est_30 = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=make_info(30),
        task_type="Base",
        tokenize_prompt=_fake_tokenize,
        codec_language_id={"english": 0},
        spk_is_dialect=None,
    )
    est_50 = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=make_info(50),
        task_type="Base",
        tokenize_prompt=_fake_tokenize,
        codec_language_id={"english": 0},
        spk_is_dialect=None,
    )

    # Each additional ref_id contributes exactly one token to the prefix budget
    # (the estimator strips a fixed 5-token tail from ref_ids regardless of length).
    # Under the previous bug `_first` collapsed both lists to a single int, so both
    # estimates would have been identical (delta = 0).
    assert est_50 - est_30 == 20, (
        f"got delta={est_50 - est_30}; expected 20 (= 50 - 30 extra ref_ids). "
        f"Did `_first(ref_ids)` collapse the list to its first int?"
    )
