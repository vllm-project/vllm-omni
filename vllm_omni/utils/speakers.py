# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Speaker-name helpers shared by serving and model-side voice selection."""

from __future__ import annotations

from typing import Any


def get_supported_speakers_from_hf_config(hf_config: Any) -> set[str]:
    """Extract supported speaker names from a model hf_config."""
    config = (
        hf_config.get("talker_config") if isinstance(hf_config, dict) else getattr(hf_config, "talker_config", None)
    )
    if config is None:
        return set()

    for spk_attr in ("speaker_id", "spk_id"):
        speakers_dict = config.get(spk_attr) if isinstance(config, dict) else getattr(config, spk_attr, None)
        if speakers_dict and isinstance(speakers_dict, dict):
            return {speaker.lower() for speaker in speakers_dict}
    return set()


def validate_requested_speaker(speaker: str | None, supported_speakers: set[str]) -> str | None:
    """Normalize and validate an optional speaker value.

    Returns the normalized speaker string when provided, otherwise ``None``.
    Raises ``ValueError`` when the speaker is not in the supported list.
    """
    if not isinstance(speaker, str) or not speaker.strip():
        return None

    normalized = speaker.lower().strip()
    if supported_speakers and normalized not in supported_speakers:
        raise ValueError(f"Invalid speaker '{speaker}'. Supported: {', '.join(sorted(supported_speakers))}")
    return normalized
