# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.omnivoice.prompt_utils import prepare_instruct, validate_instruction

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestOmniVoiceValidateInstructions:
    """Test OmniVoice instructions validation."""

    @pytest.mark.parametrize(
        "instructions",
        ["female, young adult, australian accent", "男， 河南话"],
        ids=["english", "chinese"],
    )
    def test_valid_instruction(self, instructions) -> None:
        assert validate_instruction(instructions) is None

    @pytest.mark.parametrize(
        ("instructions", "warning_prefix"),
        [
            ("happy little scottish boy", "Unsupported instruct items found"),
            ("male, teenager, middle-aged", "Conflicting instruct items within the same category"),
            ("male, english accent", "Unsupported instruct items found"),
            ("男生", "Unsupported instruct items found"),
            ("male, american accent, 河南话", "Cannot mix Chinese dialect and English accent"),
        ],
        ids=["free-form", "conflicting", "unknown-english", "unknown-chinese", "mixed-dialect-accent"],
    )
    def test_invalid_instruction(self, instructions, warning_prefix) -> None:
        assert validate_instruction(instructions).startswith(warning_prefix)


class TestOmniVoicePrepareInstruct:
    """Test OmniVoice instruction preparation."""

    @pytest.mark.parametrize(
        ("instructions", "expected"),
        [
            ("male, elderly", "male, elderly"),
            ("女,儿童", "女，儿童"),
            ("女，儿童", "女，儿童"),
            ("少年, american accent", "teenager, american accent"),
            ("elderly, 河南话", "老年，河南话"),
        ],
        ids=["english", "convert-comma", "preserve-comma", "accent-forces-english", "dialect-forces-chinese"],
    )
    def test_prepare_instruct(self, instructions, expected) -> None:
        assert prepare_instruct(instructions) == expected
