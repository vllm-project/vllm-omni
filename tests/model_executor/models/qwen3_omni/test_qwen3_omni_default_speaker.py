# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the Qwen3-Omni default talker voice.

The released checkpoint lists ``talker_config.speaker_id`` as
``{"chelsie": 2301, "ethan": 2302, "aiden": 2303}``. The default voice used to
be the first key, so a request that named no speaker was spoken by Chelsie,
while the reference ``generate(speaker="Ethan")`` (and other servers) use Ethan.
Benchmarks that omit ``speaker`` then compared two different voices.
"""

import pytest

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    DEFAULT_SPEAKER,
    Qwen3OmniMoeForConditionalGeneration,
    resolve_default_speaker,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Order as shipped in Qwen/Qwen3-Omni-30B-A3B-Instruct config.json.
CHECKPOINT_SPEAKER_IDS = {"chelsie": 2301, "ethan": 2302, "aiden": 2303}


def test_default_speaker_is_ethan_even_when_listed_second() -> None:
    assert DEFAULT_SPEAKER == "ethan"
    assert resolve_default_speaker(CHECKPOINT_SPEAKER_IDS) == "ethan"


def test_default_speaker_falls_back_to_first_key_without_ethan() -> None:
    assert resolve_default_speaker({"chelsie": 1, "aiden": 2}) == "chelsie"


def test_unknown_or_missing_speaker_uses_ethan_token() -> None:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.tts_text_spk_token_ids = dict(CHECKPOINT_SPEAKER_IDS)
    model.default_tts_text_spk_type = resolve_default_speaker(model.tts_text_spk_token_ids)

    assert model._get_text_spk_token_id("not-a-speaker") == CHECKPOINT_SPEAKER_IDS["ethan"]
    assert model._get_text_spk_token_id(model.default_tts_text_spk_type) == CHECKPOINT_SPEAKER_IDS["ethan"]
    assert model._get_text_spk_token_id("chelsie") == CHECKPOINT_SPEAKER_IDS["chelsie"]
