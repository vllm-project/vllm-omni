# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

from vllm_omni.diffusion.lora.types import (
    LoRARequestInput,
    LoRAScaleInput,
    normalize_lora_composition,
    registered_lora_request,
    split_lora_composition,
)
from vllm_omni.lora.request import LoRARequest


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


def _parse_single_lora_request(lora_body: Any) -> tuple[LoRARequest, float | None]:
    if not isinstance(lora_body, dict):
        raise ValueError("Invalid lora field: expected an object or an array of objects.")

    path_fields = ("local_path", "path", "lora_path", "lora_local_path")
    int_id_fields = ("int_id", "lora_int_id")
    allowed_fields = {"name", "scale", "lora_scale", *path_fields, *int_id_fields}
    unknown_fields = set(lora_body) - allowed_fields
    if unknown_fields:
        fields = ", ".join(sorted(map(repr, unknown_fields)))
        raise ValueError(f"Invalid lora object: unknown field(s): {fields}.")
    if any(lora_body.get(field) is not None for field in path_fields):
        raise ValueError(
            "Request-level LoRA paths are not accepted. Register the adapter with "
            "--dynamic-lora at server startup, then select it by name."
        )

    scale_fields = [field for field in ("scale", "lora_scale") if lora_body.get(field) is not None]
    if len(scale_fields) > 1:
        raise ValueError(f"Invalid lora object: multiple scale fields were provided: {scale_fields}.")
    lora_scale = lora_body[scale_fields[0]] if scale_fields else None
    if any(lora_body.get(field) is not None for field in int_id_fields):
        raise ValueError("Invalid lora object: int_id is internal; select a registered adapter by name.")
    lora_name = lora_body.get("name")
    if not isinstance(lora_name, str) or not lora_name.strip():
        raise ValueError("Invalid lora object: name must be a non-empty string.")
    request = registered_lora_request(lora_name)

    try:
        scale = float(lora_scale) if lora_scale is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid lora object: scale must be a finite number.") from exc
    if scale is not None and not math.isfinite(scale):
        raise ValueError("Invalid lora object: scale must be a finite number.")
    return request, scale


def parse_lora_request(lora_body: Any) -> tuple[LoRARequestInput, LoRAScaleInput | None]:
    """Parse one or more request-level LoRAs and their mixing coefficients.

    Raises:
        ValueError: If the object shape is invalid or required fields are missing.
    """
    if lora_body is None:
        return None, None
    bodies = lora_body if isinstance(lora_body, list) else [lora_body]
    if not bodies:
        return (), ()
    parsed = tuple(_parse_single_lora_request(body) for body in bodies)
    if len(parsed) == 1:
        return parsed[0]
    composition = normalize_lora_composition(
        tuple(request for request, _ in parsed),
        tuple(1.0 if scale is None else scale for _, scale in parsed),
    )
    if not composition:
        # A non-empty request can normalize to an empty composition when
        # duplicate adapter scales cancel. Preserve that explicit disable
        # instead of projecting it to the omitted-request sentinel.
        return (), ()
    return split_lora_composition(composition)


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


def resolve_diffusion_od_config(engine_client: Any, diffusion_engine: Any = None) -> Any:
    """Resolve the OmniDiffusionConfig from the engine or diffusion engine."""
    od_config = None
    if hasattr(engine_client, "get_diffusion_od_config"):
        od_config = engine_client.get_diffusion_od_config()
    if od_config is None and diffusion_engine is not None:
        if hasattr(diffusion_engine, "get_diffusion_od_config"):
            od_config = diffusion_engine.get_diffusion_od_config()
        else:
            od_config = getattr(diffusion_engine, "od_config", None)
    return od_config


def is_single_stage_diffusion(engine_client: Any) -> bool:
    """Return True if the engine is a single-stage diffusion pipeline."""
    stage_configs = getattr(engine_client, "stage_configs", None) or []
    if len(stage_configs) != 1:
        return False
    return getattr(stage_configs[0], "stage_type", None) in ("diffusion", "DIFFUSION")


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
