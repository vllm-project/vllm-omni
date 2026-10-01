# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bundled reference voices for VibeVoice requests without explicit audio."""

from pathlib import Path

DEFAULT_REFERENCE_AUDIO_FILENAMES = tuple(f"default_{index}.wav" for index in range(4))


def get_default_reference_audio_path(index: int) -> Path:
    """Return one packaged default reference, failing clearly if it is absent."""
    if not 0 <= index < len(DEFAULT_REFERENCE_AUDIO_FILENAMES):
        raise ValueError(
            f"VibeVoice default reference index must be between 0 and "
            f"{len(DEFAULT_REFERENCE_AUDIO_FILENAMES) - 1}, got {index}."
        )
    path = Path(__file__).resolve().parent / "assets" / DEFAULT_REFERENCE_AUDIO_FILENAMES[index]
    if not path.is_file():
        raise FileNotFoundError(f"Bundled VibeVoice reference audio is missing: {path}")
    return path


__all__ = [
    "DEFAULT_REFERENCE_AUDIO_FILENAMES",
    "get_default_reference_audio_path",
]
