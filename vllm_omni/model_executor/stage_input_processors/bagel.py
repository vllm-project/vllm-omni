"""Stage input processor for Bagel: CFG prompt expansion and KV cache collection.

Bagel's 3-branch CFG requires multiple prompts through the AR stage:
  - gen (conditional): user prompt
  - cfg_text (text unconditional): negative/empty prompt
  - cfg_img (image unconditional): user prompt without image (same as gen for text2img)

This module provides model-specific functions referenced by bagel.yaml:
  - prompt_expand_func  -> expand_cfg_prompts
  - cfg_kv_collect_func -> collect_cfg_kv_caches
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

CFG_TEXT_SUFFIX = "__cfg_text"


@dataclass
class ExpandedPrompt:
    """A single expanded prompt produced by the prompt expansion function."""

    prompt: dict[str, Any] | str
    role: str
    request_id_suffix: str


def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[ExpandedPrompt]:
    """Expand a user prompt into additional prompts needed for Bagel CFG.

    For text2img (modalities contains "image", no multi_modal_data with image):
      - One extra prompt for cfg_text (negative/empty prompt)
      - cfg_img reuses gen KV for text2img, so no extra prompt needed

    For text2text or img2text: returns empty list (no expansion needed).

    Args:
        prompt: The original user prompt (dict or string).
        sampling_params: The Stage-0 sampling params (may carry extra_args
            from the diffusion sampling params for negative_prompt).

    Returns:
        List of ExpandedPrompt. Empty if no expansion needed.
    """
    if not isinstance(prompt, dict):
        return []

    modalities = prompt.get("modalities", [])
    if "image" not in modalities and "img2img" not in modalities:
        return []

    neg_prompt = _get_negative_prompt(prompt, sampling_params)

    if "image" in modalities:
        neg_prompt_dict = {
            "prompt": neg_prompt,
            "modalities": prompt.get("modalities", []),
        }
        return [
            ExpandedPrompt(
                prompt=neg_prompt_dict,
                role="cfg_text",
                request_id_suffix=CFG_TEXT_SUFFIX,
            ),
        ]

    # img2img: more complex (3 distinct caches). Reserve for future.
    return []


def collect_cfg_kv_caches(
    request_id: str,
    cfg_request_ids: dict[str, str],
    kv_transfer_manager: Any,
    target_device: Any | None = None,
) -> dict[str, Any]:
    """Collect KV caches for all CFG roles from the KV transfer manager.

    Called by the diffusion model runner after receiving the primary KV cache.
    Uses the kv_transfer_manager to fetch companion KV caches by their
    request IDs.

    Args:
        request_id: The original (parent) request ID.
        cfg_request_ids: Mapping of role -> companion request ID,
            e.g. {"cfg_text": "req_0__cfg_text"}.
        kv_transfer_manager: The OmniKVTransferManager instance.
        target_device: Device to move tensors to.

    Returns:
        Dict with keys like "cfg_text_past_key_values",
        "cfg_text_kv_metadata", etc.
    """
    result: dict[str, Any] = {}

    for role, companion_rid in cfg_request_ids.items():
        try:
            data, size = kv_transfer_manager.receive_kv_cache_for_request(companion_rid, target_device)
            if data and "layer_blocks" in data:
                layer_blocks = data["layer_blocks"]
                kv_obj = SimpleNamespace(**layer_blocks)
                result[f"{role}_past_key_values"] = kv_obj
                if "metadata" in data:
                    result[f"{role}_kv_metadata"] = data["metadata"]
                logger.info(
                    "Collected CFG KV cache for role=%s, rid=%s, size=%d bytes",
                    role,
                    companion_rid,
                    size,
                )
            else:
                logger.warning(
                    "Failed to collect CFG KV cache for role=%s, rid=%s",
                    role,
                    companion_rid,
                )
        except Exception as e:
            logger.exception(
                "Error collecting CFG KV cache for role=%s, rid=%s: %s",
                role,
                companion_rid,
                e,
            )

    return result


def _get_negative_prompt(
    prompt: dict[str, Any],
    sampling_params: Any,
) -> str:
    """Resolve the negative prompt for CFG from prompt or sampling params.

    An empty string is treated the same as absent (falls through to
    the Bagel default token pair), because an empty negative prompt is
    not meaningful for CFG guidance.
    """
    neg = prompt.get("negative_prompt")
    if neg:
        return neg

    if hasattr(sampling_params, "extra_args") and sampling_params.extra_args:
        neg = sampling_params.extra_args.get("negative_prompt")
        if neg:
            return neg

    return "<|im_start|><|im_end|>"
