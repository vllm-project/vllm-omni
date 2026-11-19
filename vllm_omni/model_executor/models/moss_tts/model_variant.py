# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared MOSS serving variant selection, without model loading."""


def detect_moss_variant(model: str) -> str:
    """Preserve the speech adapter's repository/path-based dispatch contract."""
    name = (model or "").lower().replace("-", "").replace("_", "")
    if "realtime" in name:
        return "realtime"
    if "local" in name:
        return "local"
    if "ttsd" in name:
        return "ttsd"
    if "soundeffect" in name:
        return "sound_effect"
    if "voicegenerator" in name:
        return "voice_generator"
    return "tts"
