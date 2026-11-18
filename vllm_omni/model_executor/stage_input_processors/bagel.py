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
CFG_IMG_SUFFIX = "__cfg_img"


@dataclass
class ExpandedPrompt:
    """A single expanded prompt produced by the prompt expansion function."""

    prompt: dict[str, Any] | str
    role: str
    request_id_suffix: str
    sampling_params_override: dict[str, Any] | None = None

    def apply_overrides(
        self,
        base_params: Any,
        base_spl: list[Any],
    ) -> tuple[Any, list[Any]]:
        """Return ``(params, sampling_params_list)`` with overrides applied.

        If this prompt has no overrides the originals are returned as-is.
        """
        if not self.sampling_params_override:
            return base_params, base_spl
        patched = base_params.clone()
        for k, v in self.sampling_params_override.items():
            setattr(patched, k, v)
        spl = list(base_spl)
        if spl:
            spl[0] = patched
        return patched, spl


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
        if not neg_prompt:
            return []
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

    if "img2img" in modalities:
        IMG2IMG_PLACEHOLDER = "<|fim_middle|>"

        cfg_text_dict: dict[str, Any] = {
            "prompt": f"{IMG2IMG_PLACEHOLDER}{neg_prompt}",
            "modalities": ["img2img"],
        }
        mm_data = prompt.get("multi_modal_data")
        if mm_data:
            cfg_text_dict["multi_modal_data"] = mm_data

        original_text = prompt.get("prompt", "")
        cfg_img_text = original_text.replace(IMG2IMG_PLACEHOLDER, "")
        cfg_img_dict: dict[str, Any] = {
            "prompt": cfg_img_text,
            "modalities": ["img2img"],
        }

        return [
            ExpandedPrompt(
                prompt=cfg_text_dict,
                role="cfg_text",
                request_id_suffix=CFG_TEXT_SUFFIX,
            ),
            ExpandedPrompt(
                prompt=cfg_img_dict,
                role="cfg_img",
                request_id_suffix=CFG_IMG_SUFFIX,
            ),
        ]

    return []


GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind "
    "and then generate the image. \n"
    "The planning process is enclosed within <think> </think> tags, "
    "i.e. <think> planning process here </think> image here"
)

VLM_THINK_SYSTEM_PROMPT = (
    "You should first think about the reasoning process in the mind "
    "and then provide the user with the answer. \n"
    "The reasoning process is enclosed within <think> </think> tags, "
    "i.e. <think> reasoning process here </think> answer here"
)


def expand_cfg_prompts_think(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[ExpandedPrompt]:
    """Expand prompts for Bagel CFG in thinking mode.

    Same as expand_cfg_prompts but companion requests get max_tokens=1
    so they stop immediately after prefill (no thinking decode).

    In thinking mode the gen (main) request decodes thinking tokens until
    EOS; companions should only contribute their prefill KV cache.
    """
    if not isinstance(prompt, dict):
        return []

    modalities = prompt.get("modalities", [])
    if "image" not in modalities and "img2img" not in modalities:
        return []

    neg_prompt = _get_negative_prompt(prompt, sampling_params)
    companion_params = {"max_tokens": 1}

    if "image" in modalities:
        if not neg_prompt:
            return []
        neg_prompt_dict = {
            "prompt": neg_prompt,
            "modalities": prompt.get("modalities", []),
        }
        return [
            ExpandedPrompt(
                prompt=neg_prompt_dict,
                role="cfg_text",
                request_id_suffix=CFG_TEXT_SUFFIX,
                sampling_params_override=companion_params,
            ),
        ]

    if "img2img" in modalities:
        IMG2IMG_PLACEHOLDER = "<|fim_middle|>"

        original_text = prompt.get("prompt", "")
        # Extract system prompt prefix (everything before <|fim_middle|>)
        # so cfg_text gets system_prompt + image (no user text), matching
        # the original BAGEL code where cfg_text = deepcopy(gen after image).
        parts = original_text.split(IMG2IMG_PLACEHOLDER, 1)
        system_prefix = parts[0] if len(parts) > 1 else ""

        cfg_text_prompt = f"{system_prefix}{IMG2IMG_PLACEHOLDER}{neg_prompt}"
        cfg_text_dict: dict[str, Any] = {
            "prompt": cfg_text_prompt,
            "modalities": ["img2img"],
        }
        mm_data = prompt.get("multi_modal_data")
        if mm_data:
            cfg_text_dict["multi_modal_data"] = mm_data

        cfg_img_text = original_text.replace(IMG2IMG_PLACEHOLDER, "")
        cfg_img_dict: dict[str, Any] = {
            "prompt": cfg_img_text,
            "modalities": ["img2img"],
        }
        if mm_data:
            cfg_img_dict["multi_modal_data"] = mm_data

        return [
            ExpandedPrompt(
                prompt=cfg_text_dict,
                role="cfg_text",
                request_id_suffix=CFG_TEXT_SUFFIX,
                sampling_params_override=companion_params,
            ),
            ExpandedPrompt(
                prompt=cfg_img_dict,
                role="cfg_img",
                request_id_suffix=CFG_IMG_SUFFIX,
                sampling_params_override=companion_params,
            ),
        ]

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

    Returns the negative prompt string when one is supplied, otherwise an
    empty string. Callers decide how to treat the empty case: text2img
    skips the cfg_text companion entirely, while img2img substitutes it
    into the cfg_text prompt template.
    """
    neg = prompt.get("negative_prompt")
    if neg:
        return neg

    if hasattr(sampling_params, "extra_args") and sampling_params.extra_args:
        neg = sampling_params.extra_args.get("negative_prompt")
        if neg:
            return neg

    return ""
