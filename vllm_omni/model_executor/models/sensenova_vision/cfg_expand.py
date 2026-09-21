# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova-Vision CFG prompt expansion (multi-view-safe).

The shared BAGEL expanders (``vllm_omni.model_executor.stage_input_processors.bagel``)
assume img2img prompts carry exactly ONE ``<|fim_middle|>`` marker.  SenseNova-Vision's
recon3d / (think) img2img formatters feed N conditioning images under
``multi_modal_data['img2img']`` and prefix the prompt with N markers.  vLLM's
prompt-update matcher then requires every companion (cfg_text / cfg_img) to carry the
same number of markers as img2img items, or it raises

    Failed to apply prompt replacement for mm_items['img2img'][k]

These wrappers therefore emit N markers whenever ``multi_modal_data['img2img']`` holds
N items, and defer to the BAGEL behavior otherwise.
"""

from __future__ import annotations

from typing import Any

from vllm_omni.model_executor.stage_input_processors.bagel import (
    ExpandedPrompt,
    _get_negative_prompt,
    expand_cfg_prompts,
    expand_cfg_prompts_think,
)

IMG2IMG_PLACEHOLDER = "<|fim_middle|>"

__all__ = [
    "expand_sensenova_cfg_prompts",
    "expand_sensenova_cfg_prompts_think",
]

CFG_TEXT_SUFFIX = "__cfg_text"
CFG_IMG_SUFFIX = "__cfg_img"


def _img2img_item_count(prompt: dict[str, Any]) -> int:
    """Return the number of img2img conditioning items, or 0 when absent."""
    mm_data = prompt.get("multi_modal_data")
    if not isinstance(mm_data, dict):
        return 0
    items = mm_data.get("img2img")
    if isinstance(items, (list, tuple)):
        return len(items)
    return 1 if items is not None else 0


def _build_companion(
    prompt: dict[str, Any],
    *,
    fim_count: int,
    text: str,
) -> dict[str, Any]:
    """Companion prompt dict: N markers + ``text``, carrying every img2img item.

    The marker block must match ``multi_modal_data['img2img']`` item-for-item so
    vLLM's prompt-update matcher binds each placeholder range.
    """
    companion: dict[str, Any] = {
        "prompt": IMG2IMG_PLACEHOLDER * fim_count + text,
        "modalities": ["img2img"],
    }
    mm_data = prompt.get("multi_modal_data")
    if mm_data:
        companion["multi_modal_data"] = mm_data
    return companion


def _collapsed_text(prompt: dict[str, Any]) -> str:
    """Original user text with the marker block stripped (markers live on the companion)."""
    text = prompt.get("prompt", "")
    while text.startswith(IMG2IMG_PLACEHOLDER):
        text = text[len(IMG2IMG_PLACEHOLDER) :]
    return text


def _expand_multi_view(
    prompt: dict[str, Any],
    sampling_params: Any,
    *,
    think: bool,
) -> list[ExpandedPrompt]:
    fim_count = _img2img_item_count(prompt)
    neg_prompt = _get_negative_prompt(prompt, sampling_params)
    override = {"max_tokens": 1} if think else None

    cfg_text = _build_companion(prompt, fim_count=fim_count, text=neg_prompt)
    cfg_img = _build_companion(prompt, fim_count=fim_count, text=_collapsed_text(prompt))

    return [
        ExpandedPrompt(
            prompt=cfg_text,
            role="cfg_text",
            request_id_suffix=CFG_TEXT_SUFFIX,
            sampling_params_override=override,
        ),
        ExpandedPrompt(
            prompt=cfg_img,
            role="cfg_img",
            request_id_suffix=CFG_IMG_SUFFIX,
            sampling_params_override=override,
        ),
    ]


def expand_sensenova_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[ExpandedPrompt]:
    """Like BAGEL ``expand_cfg_prompts`` but N-marker-safe for img2img.

    Non-img2img prompts and single-image img2img delegate to the shared BAGEL
    expander unchanged.
    """
    if not isinstance(prompt, dict):
        return expand_cfg_prompts(prompt, sampling_params)

    modalities = prompt.get("modalities", [])
    if "img2img" not in modalities:
        return expand_cfg_prompts(prompt, sampling_params)

    if _img2img_item_count(prompt) <= 1:
        return expand_cfg_prompts(prompt, sampling_params)

    return _expand_multi_view(prompt, sampling_params, think=False)


def expand_sensenova_cfg_prompts_think(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[ExpandedPrompt]:
    """Think variant: N-marker-safe img2img expansion with max_tokens=1 companions."""
    if not isinstance(prompt, dict):
        return expand_cfg_prompts_think(prompt, sampling_params)

    modalities = prompt.get("modalities", [])
    if "img2img" not in modalities:
        return expand_cfg_prompts_think(prompt, sampling_params)

    if _img2img_item_count(prompt) <= 1:
        return expand_cfg_prompts_think(prompt, sampling_params)

    return _expand_multi_view(prompt, sampling_params, think=True)
