# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import (
    MiniCPMO45DuplexPolicy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_seed_tts_initial_text_is_part_of_native_duplex_context():
    prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
        "Speak exactly.",
        True,
        "The quick brown fox.",
    )

    assert prefix == "<|im_start|>system\nSpeak exactly.\n<|audio_start|>"
    assert suffix == (
        "<|audio_end|><|im_end|>\n<|im_start|>user\nThe quick brown fox.<|im_end|>\n<|im_start|>assistant\n"
    )
