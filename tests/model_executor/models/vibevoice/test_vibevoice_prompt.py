# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speaker script parsing and native HF prompt layout (no checkpoint needed)."""

import pytest

from vllm_omni.model_executor.models.vibevoice.prompt import (
    AUDIO_BOS,
    AUDIO_EOS,
    AUDIO_TOKEN,
    check_reference_prompt,
    parse_vibevoice_script,
    render_script_prompt,
    render_vibevoice_prompt,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
SYSTEM = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)
REF_SEGMENT = f"{AUDIO_BOS}{AUDIO_TOKEN}{AUDIO_EOS}"


def test_parse_script():
    text = "First line.\nSecond line."
    assert parse_vibevoice_script(text) == ([(0, text)], [0])
    assert parse_vibevoice_script("Speaker 1: Hi.\nspeaker 3 : Hello.\nSpeaker 1: Goodbye.") == (
        [(1, "Hi."), (3, "Hello."), (1, "Goodbye.")],
        [1, 3],
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  ",
        AUDIO_TOKEN,
        "Speaker 0:",
        "Speaker -1: Hello.",
        "Speaker A: Hello.",
        "Speaker 0: Hi.\nplain line",
        "\n".join(f"Speaker {i}: Hi." for i in range(5)),
    ],
)
def test_reject_invalid_script(text):
    with pytest.raises(ValueError):
        parse_vibevoice_script(text)


@pytest.mark.parametrize("reference", [True, False])
def test_single_speaker_compatibility(reference):
    voice = f" Voice input:\n Speaker 0:{REF_SEGMENT}\n" if reference else ""
    expected = f"{SYSTEM}{voice} Text input:\n Speaker 0: Hello.\n Speech output:\n{AUDIO_BOS}"
    assert render_vibevoice_prompt("Hello.", reference=reference) == expected
    assert render_script_prompt([(0, "Hello.")], [0] if reference else []) == expected


@pytest.mark.parametrize("refs", [[], [3], [1, 3]])
def test_multispeaker_reference_subset(refs):
    lines = [(1, "A"), (3, "B"), (1, "C")]
    voice = " Voice input:\n" + "".join(f" Speaker {s}:{REF_SEGMENT}\n" for s in refs) if refs else ""
    expected = f"{SYSTEM}{voice} Text input:\n Speaker 1: A\n Speaker 3: B\n Speaker 1: C\n Speech output:\n{AUDIO_BOS}"
    assert render_script_prompt(lines, refs) == expected
    if refs:
        with pytest.raises(ValueError, match="no audio"):
            check_reference_prompt(expected, [])
    else:
        assert check_reference_prompt(expected, []) == expected


def test_reference_order_is_canonicalized_to_script_order():
    lines = [(1, "A"), (3, "B")]
    expected = render_script_prompt(lines, [1, 3])
    # Out-of-order input renders the HF message-order layout regardless.
    assert render_script_prompt(lines, [3, 1]) == expected
    assert expected.index("Speaker 1:") < expected.index("Speaker 3:")


@pytest.mark.parametrize("refs", [[0, 0], [2]])
def test_reject_invalid_reference_speakers(refs):
    with pytest.raises(ValueError, match="Reference speakers"):
        render_script_prompt([(0, "Hello.")], refs)
