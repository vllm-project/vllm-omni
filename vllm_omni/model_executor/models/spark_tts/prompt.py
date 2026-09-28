# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Spark-TTS prompt builders following the official SparkAudio token grammar."""

LEVELS = {"very_low": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 4}
GENDERS = {"female": 0, "male": 1}


def control_prompt(text: str, gender: str = "female", pitch: str = "moderate", speed: str = "moderate") -> str:
    if not text.strip():
        raise ValueError("Spark-TTS text must not be empty")
    if gender not in GENDERS or pitch not in LEVELS or speed not in LEVELS:
        raise ValueError("Invalid Spark-TTS gender, pitch, or speed level")
    return (
        "<|task_controllable_tts|><|start_content|>"
        + text.strip()
        + "<|end_content|><|start_style_label|>"
        + f"<|gender_{GENDERS[gender]}|><|pitch_label_{LEVELS[pitch]}|>"
        + f"<|speed_label_{LEVELS[speed]}|><|end_style_label|>"
    )


def clone_prompt(text: str, global_codes: list[int], semantic_codes: list[int], ref_text: str | None = None) -> str:
    if not text.strip() or not global_codes:
        raise ValueError("Spark-TTS clone requires text and reference global codes")
    globals_part = "".join(f"<|bicodec_global_{code}|>" for code in global_codes)
    if ref_text is None:
        return (
            "<|task_tts|><|start_content|>"
            + text.strip()
            + "<|end_content|><|start_global_token|>"
            + globals_part
            + "<|end_global_token|>"
        )
    semantics_part = "".join(f"<|bicodec_semantic_{code}|>" for code in semantic_codes)
    return (
        "<|task_tts|><|start_content|>"
        + ref_text
        + text.strip()
        + "<|end_content|><|start_global_token|>"
        + globals_part
        + "<|end_global_token|><|start_semantic_token|>"
        + semantics_part
    )
