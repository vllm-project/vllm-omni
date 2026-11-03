# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request controls shared by standalone TTS and thinker speech output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompt_utils import DEFAULT_PROMPT, create_instruction


@dataclass(slots=True)
class GenerationParams:
    prompt: str
    instruction: str | None
    cfg: float
    sigma: float
    temperature: float
    max_steps: int
    min_steps: int
    stop_threshold: float
    use_zero_spk_emb: bool
    max_text_length: int
    use_static_cache: bool
    stream_decode: bool
    normalize_text: bool
    tn_mode: str
    duration_limit: bool
    trim_trailing_silence: bool
    seed: int | None
    scrub_markup: bool = False


def resolve_generation_params(info: dict[str, Any], config: Any) -> GenerationParams:
    scrub_markup = info.get("scrub_markup", config.scrub_markup)
    if not isinstance(scrub_markup, bool):
        raise ValueError("scrub_markup must be a boolean")
    instruction = create_instruction(info.get("instruction"))
    if instruction is not None:
        instruction = instruction.strip() or None
    stream_decode = bool(info.get("stream_decode", config.stream_decode))
    trim_silence = info.get("trim_trailing_silence", config.trim_trailing_silence)
    if trim_silence is None:
        # Native Ming trims only the non-streaming tts_job output.
        trim_silence = not stream_decode
    params = GenerationParams(
        prompt=info.get("prompt") or DEFAULT_PROMPT,
        instruction=instruction,
        cfg=float(info.get("cfg", info.get("cfg_strength", config.cfg_strength))),
        sigma=float(info.get("sigma", 0.25)),
        temperature=float(info.get("temperature", 0.0)),
        max_steps=int(info.get("max_steps", info.get("max_decode_steps", config.max_decode_steps))),
        min_steps=int(info.get("min_decode_steps", 3)),
        stop_threshold=float(info.get("stop_threshold", 0.5)),
        use_zero_spk_emb=bool(
            info.get("use_zero_spk_emb", info.get("ming_task") == "omni" and info.get("spk_emb") is None)
        ),
        max_text_length=int(info.get("max_text_length", 50)),
        use_static_cache=bool(info.get("use_static_cache", True)),
        stream_decode=stream_decode,
        normalize_text=bool(info.get("normalize_text", config.normalize_text)),
        tn_mode=str(info.get("tn_mode", config.tn_mode)),
        duration_limit=bool(info.get("duration_limit", config.duration_limit)),
        trim_trailing_silence=bool(trim_silence),
        seed=int(info["seed"]) if info.get("seed") is not None else None,
        scrub_markup=scrub_markup,
    )
    if params.max_steps <= 0 or params.min_steps < 0 or params.max_text_length <= 0:
        raise ValueError("max_decode_steps/max_text_length must be positive and min_decode_steps nonnegative")
    if not 0.0 <= params.stop_threshold <= 1.0:
        raise ValueError("stop_threshold must be between 0 and 1")
    if params.cfg < 0 or params.sigma < 0 or params.temperature < 0:
        raise ValueError("cfg, sigma and temperature must be nonnegative")
    if params.tn_mode not in {"legacy", "upstream"}:
        raise ValueError("tn_mode must be 'legacy' or 'upstream'")
    return params


def resolve_conditioning_modes(info: dict[str, Any], *, has_reference: bool) -> tuple[str, str]:
    """Resolve Ming speaker and reference-audio conditioning modes."""
    mode = str(info.get("speaker_conditioning") or "auto").strip().lower()
    audio_mode = str(info.get("prompt_audio_mode") or "continuation").strip().lower()
    if mode not in {"auto", "prompt", "zero", "off", "mute", "instruct"}:
        raise ValueError(f"Invalid speaker_conditioning: {mode}")
    if audio_mode not in {"continuation", "speaker_only", "none"}:
        raise ValueError(f"Invalid prompt_audio_mode: {audio_mode}")
    if audio_mode == "speaker_only" and not has_reference:
        raise ValueError("speaker_only requires reference audio or speaker embeddings")
    if mode == "auto":
        # Explicit speaker vectors remain usable without a codec prefix;
        # a SIM-only reference WAV in audio_mode=none must stay unused.
        mode = "prompt" if info.get("spk_emb") is not None or has_reference and audio_mode != "none" else "zero"
    if mode == "prompt" and not has_reference:
        raise ValueError("speaker_conditioning=prompt requires reference audio or speaker embeddings")
    if mode in {"mute", "instruct"}:
        audio_mode = "none"
    return mode, audio_mode
