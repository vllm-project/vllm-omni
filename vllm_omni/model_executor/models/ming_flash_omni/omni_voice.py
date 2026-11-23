# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference and IP voice selection shared by completed and chunked speech."""

import json
from pathlib import Path
from typing import Any

from .prompt_utils import create_instruction
from .reference_audio import REFERENCE_AUDIO_FIELDS, has_reference_audio

DEFAULT_IP_VOICE = "灵小柔"
_DATA_DIR = Path(__file__).parent / "data"
_IP_VOICES = {name.lower(): name for name in json.loads((_DATA_DIR / "ip_voices.json").read_text(encoding="utf-8"))}


def ip_voice_names() -> tuple[str, ...]:
    """Expose built-in captions, with the default first for serving metadata."""
    return (DEFAULT_IP_VOICE, *(name for name in _IP_VOICES.values() if name != DEFAULT_IP_VOICE))


def ip_voice_name(value: str) -> str | None:
    """Return the model's canonical IP caption, preserving its case."""
    return _IP_VOICES.get(value.strip().lower())


def validate_voice_controls(controls: dict[str, Any]) -> None:
    """Legacy IP controls are no longer a public voice selection interface."""
    if controls.get("voice_mode") not in (None, "reference") or controls.get("ip_name") is not None:
        raise ValueError('Use audio.output.voice: {"id": "voice name"}; voice_mode only supports reference')


def resolve_omni_voice(text: str, controls: dict[str, Any]) -> dict[str, Any]:
    validate_voice_controls(controls)
    selected = controls.get("voice_name")
    name = ip_voice_name(selected) if isinstance(selected, str) else None
    reference = {
        key: controls.get(key)
        for key in ("prompt_wav", "reference_audio", "prompt_text", "voice_name", *REFERENCE_AUDIO_FIELDS)
    }
    has_reference = (
        bool(reference["prompt_wav"])
        or has_reference_audio(reference)
        or bool(reference["voice_name"])
        or controls.get("spk_emb") is not None
    )
    if name is None and not has_reference and controls.get("voice_mode") != "reference":
        name = DEFAULT_IP_VOICE
    if name is not None:
        # An IP supplies voice identity through the caption. References and
        # cached features from an earlier request must not affect that voice.
        return {
            "voice_mode": "ip",
            "ip_name": name,
            "instruction": create_instruction({"IP": name}),
            "speaker_conditioning": "mute",
            "prompt_audio_mode": "none",
            "use_zero_spk_emb": False,
            **dict.fromkeys(
                (
                    "prompt_wav",
                    "reference_audio",
                    "prompt_text",
                    "spk_emb",
                    "prompt_wav_lat",
                    "prompt_wav_emb",
                    "voice_name",
                    *REFERENCE_AUDIO_FIELDS,
                )
            ),
        }

    if not has_reference:
        raise ValueError("reference voice_mode requires reference audio, a stored voice, or a speaker embedding")
    audio_mode = controls.get("prompt_audio_mode") or (
        "continuation" if reference["prompt_text"] or reference["voice_name"] else "speaker_only"
    )
    if audio_mode not in {"continuation", "speaker_only"}:
        raise ValueError("reference voice_mode requires continuation or speaker_only prompt_audio_mode")
    return {
        **reference,
        "voice_mode": "reference",
        "ip_name": None,
        "instruction": None,
        "speaker_conditioning": "prompt",
        "prompt_audio_mode": audio_mode,
        "use_zero_spk_emb": False,
    }
