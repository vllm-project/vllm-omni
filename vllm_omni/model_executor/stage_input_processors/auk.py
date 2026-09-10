# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor for AuK: thinker encoder -> diffusion transition."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

# The encoder emits the fused condition as
# ``multimodal_outputs["hidden_states"]["output"]`` (``HiddenStatesStruct.output``,
# the same stage-wire slot MiniMax H3 uses). The payload flattener may deliver
# it under the dotted spelling instead, and a plain ``text_cond`` key is
# accepted so an encoder that emits the condition directly still works.
HIDDEN_STATES_KEY = "hidden_states"
TEXT_COND_KEY = "output"
_FALLBACK_KEYS = ("hidden_states.output", "text_cond")

# Per-request knobs the prompt carries for stage 1. nfe/cfg/seed are not here:
# they ride the stage-1 OmniDiffusionSamplingParams instead.
_AUK_KNOBS = ("gen_seconds", "sway", "t_grid", "vae_sample")

# Set to a directory to save each received condition for the stage-0 fusion
# parity test. Off unless the variable is set.
DUMP_DIR_ENV = "AUK_DEBUG_DUMP_DIR"


def _as_dict(prompt: Any) -> dict[str, Any]:
    """Normalize the many prompt spellings the orchestrator may pass."""
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    if prompt is None:
        return {}
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _has_audio(mm_data: Any) -> bool:
    return isinstance(mm_data, Mapping) and mm_data.get("audio") is not None


def _from_hidden_states(hidden: Any) -> Any:
    """Read the fused condition out of the payload's hidden-states slot."""
    if isinstance(hidden, Mapping):
        return hidden.get(TEXT_COND_KEY)
    # A deserialized payload arrives as a msgspec struct rather than a dict.
    return getattr(hidden, TEXT_COND_KEY, None)


def _extract_text_cond(ar_output: Any) -> Any:
    """Read the fused text condition from the encoder output.

    ``multimodal_output`` is attached to the RequestOutput; the CompletionOutput
    is checked as a fallback for vLLM versions that attach it there.
    """
    for holder in (ar_output, *(getattr(ar_output, "outputs", None) or ())):
        mm_output = getattr(holder, "multimodal_output", None)
        if not isinstance(mm_output, Mapping):
            continue
        text_cond = _from_hidden_states(mm_output.get(HIDDEN_STATES_KEY))
        if text_cond is not None:
            return text_cond
        for key in _FALLBACK_KEYS:
            if mm_output.get(key) is not None:
                return mm_output[key]
    return None


def _to_cpu_tensor(text_cond: Any) -> torch.Tensor:
    """Coerce the emitted condition to a single 2-D CPU tensor."""
    if isinstance(text_cond, list):
        if not text_cond:
            raise ValueError("AuK encoder emitted an empty text condition list")
        parts = [_to_cpu_tensor(part) for part in text_cond]
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    if isinstance(text_cond, torch.Tensor):
        tensor = text_cond
    else:
        # numpy array or any array-like the connector round-tripped.
        tensor = torch.as_tensor(text_cond)
    tensor = tensor.detach().cpu()
    if tensor.ndim != 2:
        raise ValueError(f"AuK text condition must be [tokens, hidden]; got shape {tuple(tensor.shape)}")
    return tensor


def encoder2dit(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> dict[str, Any] | None:
    """Turn the AuK encoder stage's fused hidden states into a diffusion prompt.

    The source clip is not decoded here: the original ``multi_modal_data`` is
    forwarded so stage 1 resamples and VAE-encodes it itself. It is forwarded
    unconditionally because the editing and enhancement tasks cannot run
    without it; ``requires_multimodal_data`` only records the stage's
    declaration.
    """
    del streaming_context, requires_multimodal_data
    if not source_outputs:
        return None

    ar_output = source_outputs[0]
    text_cond = _extract_text_cond(ar_output)
    if text_cond is None:
        raise ValueError(
            "AuK encoder stage produced no hidden_states.output multimodal payload; "
            "stage 1 cannot run without the fused thinker hidden states"
        )
    prompt_embeds = _to_cpu_tensor(text_cond)

    original = _as_dict(prompt)
    original_knobs = original.get("additional_information") or {}
    if isinstance(original_knobs, Mapping):
        original_knobs = original_knobs.get("auk") or {}
    if not isinstance(original_knobs, Mapping):
        original_knobs = {}

    mm_data = original.get("multi_modal_data")
    knobs: dict[str, Any] = {key: original_knobs.get(key) for key in _AUK_KNOBS}
    declared_audio = original_knobs.get("has_audio")
    knobs["has_audio"] = bool(declared_audio) if declared_audio is not None else _has_audio(mm_data)

    logger.debug(
        "[encoder2dit] text_cond=%s dtype=%s has_audio=%s gen_seconds=%s",
        tuple(prompt_embeds.shape),
        prompt_embeds.dtype,
        knobs["has_audio"],
        knobs["gen_seconds"],
    )
    _maybe_dump(prompt_embeds, getattr(ar_output, "request_id", None), original.get("prompt"))

    return {
        "prompt": "",
        "prompt_embeds": prompt_embeds,
        "multi_modal_data": mm_data,
        "additional_information": {"auk": knobs},
    }


def _maybe_dump(text_cond: torch.Tensor, request_id: Any, prompt: Any) -> None:
    """Save the received condition when AUK_DEBUG_DUMP_DIR is set."""
    dump_dir = os.environ.get(DUMP_DIR_ENV)
    if not dump_dir:
        return
    name = str(request_id) if request_id else "unknown"
    path = Path(dump_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"text_cond": text_cond, "request_id": request_id, "prompt": prompt}, path / f"{name}.pt")
    except OSError:
        logger.warning("[encoder2dit] could not write the %s dump for request %s", DUMP_DIR_ENV, name)
