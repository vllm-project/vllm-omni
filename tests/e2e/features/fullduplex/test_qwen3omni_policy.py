# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn-policy prompt tests for the Qwen3-Omni duplex adapter."""

import pytest

from vllm_omni.experimental.fullduplex.qwen3omni.policy import (
    INTERRUPTION_NOTE,
    SYSTEM_PROMPT,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_system_prompt_sets_duplex_rules():
    assert "duplex" in SYSTEM_PROMPT
    assert "short spoken turns" in SYSTEM_PROMPT
    assert "interrupted" in SYSTEM_PROMPT
    assert "markdown" in SYSTEM_PROMPT


def test_system_prompt_is_single_string_without_placeholders():
    assert isinstance(SYSTEM_PROMPT, str)
    assert "{user}" not in SYSTEM_PROMPT
    assert "{audio}" not in SYSTEM_PROMPT


def test_interruption_note_discards_previous_reply():
    assert "interrupted" in INTERRUPTION_NOTE
    assert "Discard it" in INTERRUPTION_NOTE
    assert "latest input" in INTERRUPTION_NOTE
