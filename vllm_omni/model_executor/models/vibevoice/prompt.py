# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline/serving prompt construction, matching the native HF chat template.

Three guards are intentionally stricter than the official processor (which
warns-and-skips or renders verbatim): special-token literals, mixed plain /
`Speaker N:` input, and empty speaker lines are all rejected outright.
"""

from collections.abc import Sequence

import regex as re

AUDIO_BOS = "<|vision_start|>"
AUDIO_EOS = "<|vision_end|>"
AUDIO_TOKEN = "<|vision_pad|>"
AUDIO_TOKEN_ID = 151654

#: Official VibeVoice-1.5B speaker limit.
MAX_SPEAKERS = 4

_SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)
_SPEAKER_LINE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


def render_script_prompt(
    lines: Sequence[tuple[int, str]],
    speakers_with_reference: Sequence[int] = (),
) -> str:
    """Render a parsed script, mirroring the native chat template exactly.

    ``lines`` are ``(speaker_id, text)`` in script order. Only speakers in
    ``speakers_with_reference`` appear in the Voice input section, each with
    one placeholder segment, matching the HF template's ``speakers_with_audio``
    behavior (speakers without a reference get a noise-sampled voice).
    ``speakers_with_reference`` may be given in any order; it is canonicalized
    to the script's first-appearance order. This matches the HF template's
    message-order collection whenever each speaker's reference attaches at that
    speaker's first turn — the only form the vLLM-Omni APIs produce. (HF would
    differ only for audio attached at a later turn in a cross order, which this
    API cannot express.)
    """
    if not lines or any(not text.strip() or "<|" in text for _, text in lines):
        raise ValueError("VibeVoice requires non-empty text without special-token literals.")
    speakers = {speaker for speaker, _ in lines}
    if len(speakers) > MAX_SPEAKERS:
        raise ValueError(f"VibeVoice-1.5B supports at most {MAX_SPEAKERS} speakers per request.")
    if (
        len(set(speakers_with_reference)) != len(speakers_with_reference)
        or not set(speakers_with_reference) <= speakers
    ):
        raise ValueError("Reference speakers must be distinct speakers present in the script.")
    speaker_order = list(dict.fromkeys(speaker for speaker, _ in lines))
    referenced = sorted(speakers_with_reference, key=speaker_order.index)
    voice = ""
    if referenced:
        voice = " Voice input:\n" + "".join(
            f" Speaker {speaker}:{AUDIO_BOS}{AUDIO_TOKEN}{AUDIO_EOS}\n" for speaker in referenced
        )
    text = "".join(f" Speaker {speaker}: {line}\n" for speaker, line in lines)
    return f"{_SYSTEM_PROMPT}{voice} Text input:\n{text} Speech output:\n{AUDIO_BOS}"


def parse_vibevoice_script(text: str) -> tuple[list[tuple[int, str]], list[int]]:
    """Parse plain text or a ``Speaker N:`` script into lines and speaker order.

    Returns ``(lines, speaker_ids)`` where ``speaker_ids`` is deduplicated in
    first-appearance order. Plain text becomes a single speaker-0 line,
    preserved verbatim (embedded newlines kept, as in the HF template).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("VibeVoice requires non-empty text.")
    if "<|" in text:
        raise ValueError("VibeVoice text must not contain special-token literals.")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    matches = [_SPEAKER_LINE.fullmatch(line) for line in lines]
    if not any(matches) and not any(re.match(r"^Speaker\b[^:]*:", line, re.IGNORECASE) for line in lines):
        return [(0, text)], [0]
    if not all(matches):
        raise ValueError(
            "VibeVoice input must be either plain text or contain only "
            "`Speaker N: text` lines; mixed formats are not supported."
        )
    speakers: list[int] = []
    parsed: list[tuple[int, str]] = []
    for match in matches:
        assert match is not None
        speaker = int(match.group(1))
        content = match.group(2).strip()
        if not content:
            raise ValueError("VibeVoice speaker lines require non-empty text.")
        if speaker not in speakers:
            speakers.append(speaker)
        parsed.append((speaker, content))
    if len(speakers) > MAX_SPEAKERS:
        raise ValueError(f"VibeVoice-1.5B supports at most {MAX_SPEAKERS} speakers per request.")
    return parsed, speakers


def render_vibevoice_prompt(text: str, *, reference: bool = True) -> str:
    """Render plain text for speaker 0, with or without a reference voice.

    With ``reference=True``, pass the result as ``prompt`` and the reference
    waveform/sample-rate pair as ``multi_modal_data={"audio": (waveform,
    sample_rate)}`` to offline Omni; the multimodal processor expands the single
    placeholder. With ``reference=False``, the native no-voice prompt is
    rendered and no audio may be passed. As in HF, there is no default voice;
    the voice is sampled per request from the diffusion noise.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("VibeVoice requires non-empty text.")
    if "<|" in text:
        raise ValueError("VibeVoice text must not contain special-token literals.")
    return render_script_prompt([(0, text)], [0] if reference else [])


def check_reference_prompt(prompt: object, sampling_params_list: Sequence[object]) -> object:
    """Reject a reference placeholder without audio (stage-0 prompt transform).

    vLLM skips the multimodal processor when no audio is passed, so its
    placeholder check never runs and the placeholder would be silently embedded
    as text. Audio without a matching placeholder count is rejected by the
    processor.
    """
    fields = {"prompt": prompt} if isinstance(prompt, str) else prompt
    if isinstance(fields, dict) and not fields.get("multi_modal_data"):
        text = fields.get("prompt")
        token_ids = fields.get("prompt_token_ids") or ()
        if (isinstance(text, str) and AUDIO_TOKEN in text) or AUDIO_TOKEN_ID in token_ids:
            raise ValueError(
                "VibeVoice prompt has a reference-audio placeholder but no audio; pass "
                "multi_modal_data={'audio': ...} or use render_vibevoice_prompt(text, reference=False)."
            )
    return prompt
