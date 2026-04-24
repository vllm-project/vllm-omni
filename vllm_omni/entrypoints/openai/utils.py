# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id


def get_stage_type(stage_cfg: Any) -> str:
    """Best-effort stage type resolver across dict/omegaconf/object configs."""
    if isinstance(stage_cfg, dict):
        return stage_cfg.get("stage_type", "llm")
    if hasattr(stage_cfg, "get"):
        try:
            return stage_cfg.get("stage_type", "llm")
        except Exception:
            pass
    return getattr(stage_cfg, "stage_type", "llm")


def parse_lora_request(lora_body: Any) -> tuple[LoRARequest | None, float | None]:
    """Parse a request-level LoRA object into a LoRARequest and optional scale.

    Raises:
        ValueError: If the object shape is invalid or required fields are missing.
    """
    if lora_body is None:
        return None, None

    if not isinstance(lora_body, dict):
        raise ValueError("Invalid lora field: expected an object.")

    lora_name = lora_body.get("name") or lora_body.get("lora_name") or lora_body.get("adapter")
    lora_path = (
        lora_body.get("local_path")
        or lora_body.get("path")
        or lora_body.get("lora_path")
        or lora_body.get("lora_local_path")
    )
    lora_scale = lora_body.get("scale")
    if lora_scale is None:
        lora_scale = lora_body.get("lora_scale")
    lora_int_id = lora_body.get("int_id")
    if lora_int_id is None:
        lora_int_id = lora_body.get("lora_int_id")
    if lora_int_id is None and lora_path:
        lora_int_id = stable_lora_int_id(str(lora_path))

    if not lora_name or not lora_path:
        raise ValueError("Invalid lora object: both name and path are required.")

    scale = float(lora_scale) if lora_scale is not None else None
    return LoRARequest(str(lora_name), int(lora_int_id), str(lora_path)), scale


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
