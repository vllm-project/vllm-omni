# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for JoyAI action-to-TTS conversion."""

from types import SimpleNamespace

import pytest

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.model_executor.stage_input_processors import (
    joyai_vl_interaction as joyai_bridge,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_joyai_action_to_tts_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert each JoyAI action through the real stage-client entry point."""
    target_model_config = object()
    prompt_length_calls: list[tuple[dict[str, list[str]], object]] = []
    prompt_lengths = {
        "Colored squares are visible.": 12,
        "I am checking this request now.": 18,
    }

    def compute_prompt_length(metadata, model_config):
        prompt_length_calls.append((metadata, model_config))
        return prompt_lengths[metadata["text"][0]]

    monkeypatch.setattr(
        joyai_bridge,
        "_compute_talker_prompt_length",
        compute_prompt_length,
    )

    client = object.__new__(StageEngineCoreClient)
    client.custom_process_input_func = joyai_bridge.joyai_action_to_tts
    client.requires_multimodal_data = False
    client.vllm_config = SimpleNamespace(model_config=target_model_config)

    cases = [
        (
            "</response> Colored squares are visible.",
            None,
            {"language": ["Chinese"], "speaker": ["Ryan"]},
            ("Colored squares are visible.", "Chinese", "Ryan"),
        ),
        ("</silence>", None, {}, None),
        (
            "",
            (
                "</response> I am checking this request now. "
                "</delegation> Why is the purple elephant dancing beside a volcano?"
            ),
            {
                "language": "Chinese",
                "speaker": "Ryan",
                "tts_language": "English",
                "tts_speaker": "Vivian",
            },
            ("I am checking this request now.", "English", "Vivian"),
        ),
    ]

    for text, cumulative_text, additional_information, expected in cases:
        completion = SimpleNamespace(text=text)
        if cumulative_text is not None:
            completion.cumulative_text = cumulative_text

        call_count = len(prompt_length_calls)
        talker_inputs = client.process_engine_inputs(
            [SimpleNamespace(outputs=[completion])],
            prompt={"additional_information": additional_information},
        )

        if expected is None:
            assert talker_inputs == []
            assert len(prompt_length_calls) == call_count
            continue

        spoken_text, language, speaker = expected
        [talker_input] = talker_inputs
        metadata = talker_input["additional_information"]
        assert talker_input["prompt_token_ids"] == [0] * prompt_lengths[spoken_text]
        assert metadata == {
            "task_type": ["CustomVoice"],
            "language": [language],
            "speaker": [speaker],
            "instruct": [""],
            "text": [spoken_text],
        }
        assert prompt_length_calls[-1] == (metadata, target_model_config)
