# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage input processor for MammothModa2 (AR -> diffusion)."""

from collections.abc import Mapping
from typing import Any


def _as_dict(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _coerce_dim(value: Any, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def ar2diffusion(
    source_outputs: list[Any],
    prompt: Any | None = None,
    requires_multimodal_data: bool = False,
) -> dict[str, Any]:
    del requires_multimodal_data
    if len(source_outputs) != 1:
        raise ValueError(
            f"MammothModa2 request-mode diffusion expects exactly one AR output, got {len(source_outputs)}"
        )

    ar_output = source_outputs[0]
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    prompt_dict = _as_dict(prompt)
    additional = prompt_dict.get("additional_information") or {}
    mm_kwargs = prompt_dict.get("mm_processor_kwargs") or {}
    height = _coerce_dim(
        mm_kwargs.get("target_h"),
        _coerce_dim((additional.get("image_height") or [None])[0], 1024),
    )
    width = _coerce_dim(
        mm_kwargs.get("target_w"),
        _coerce_dim((additional.get("image_width") or [None])[0], 1024),
    )

    completion = ar_output.outputs[0]
    generated_token_ids = list(completion.cumulative_token_ids[:-1])
    prompt_token_ids = list(ar_output.prompt_token_ids)
    full_token_ids = prompt_token_ids + generated_token_ids
    multimodal_output = getattr(completion, "multimodal_output", None)
    if not isinstance(multimodal_output, Mapping) or "latent" not in multimodal_output:
        raise ValueError(
            "MammothModa2 AR stage output is missing latent multimodal output; "
            f"request_id={getattr(ar_output, 'request_id', None)}"
        )

    full_hidden_states = multimodal_output["latent"]
    hidden_total = int(full_hidden_states.shape[0])
    if hidden_total != len(full_token_ids):
        raise ValueError(
            "Hidden states length mismatch: "
            f"expected {len(full_token_ids)}, got {hidden_total}; "
            f"request_id={getattr(ar_output, 'request_id', None)}"
        )

    return {
        "prompt": "",
        "height": height,
        "width": width,
        "additional_information": {
            "full_hidden_states": full_hidden_states.float().contiguous(),
            "full_token_ids": full_token_ids,
            "answer_start_index": len(prompt_token_ids),
        },
    }
