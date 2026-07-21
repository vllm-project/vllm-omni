# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input helpers for SenseNova-U1 AR -> DiT separation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    IMG_START_TOKEN,
    SYSTEM_MESSAGE_FOR_GEN,
    _build_t2i_query,
)

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
        """Return ``(params, sampling_params_list)`` with overrides applied."""
        if not self.sampling_params_override:
            return base_params, base_spl
        patched = base_params.clone()
        for key, value in self.sampling_params_override.items():
            setattr(patched, key, value)
        spl = list(base_spl)
        if spl:
            spl[0] = patched
        return patched, spl


def _extra_args(sampling_params: Any) -> dict[str, Any]:
    extra = getattr(sampling_params, "extra_args", None)
    return extra if isinstance(extra, dict) else {}


def _float_extra(sampling_params: Any, key: str, default: float) -> float:
    try:
        return float(_extra_args(sampling_params).get(key, default))
    except (TypeError, ValueError):
        return default


def _get_negative_prompt(
    prompt: dict[str, Any],
    sampling_params: Any,
) -> str:
    neg = prompt.get("negative_prompt")
    if neg:
        return neg

    extra_args = _extra_args(sampling_params)
    neg = extra_args.get("negative_prompt")
    if neg:
        return neg

    return ""


def _cfg_branch_plan(modalities: list[str], sampling_params: Any) -> tuple[bool, bool]:
    """Return whether ``cfg_text`` and ``cfg_img`` companions are needed."""
    cfg_scale = _float_extra(sampling_params, "cfg_scale", 4.0)
    img_cfg_scale = _float_extra(sampling_params, "img_cfg_scale", 1.0)

    if "image" in modalities:
        return cfg_scale > 1, False

    if "img2img" not in modalities:
        return False, False

    needs_cfg = not (cfg_scale == 1 and img_cfg_scale == 1)
    if not needs_cfg:
        return False, False

    return img_cfg_scale != 1, img_cfg_scale == 1 or cfg_scale != img_cfg_scale


def _image_count(prompt: dict[str, Any]) -> int:
    mm_data = prompt.get("multi_modal_data") if isinstance(prompt, dict) else None
    if not isinstance(mm_data, dict):
        return 0
    raw = mm_data.get("image")
    if raw is None:
        raw = mm_data.get("img2img")
    if raw is None:
        return 0
    return len(raw) if isinstance(raw, list) else 1


def _prompt_with_mm(prompt: dict[str, Any], text: str, modalities: list[str]) -> dict[str, Any]:
    patched: dict[str, Any] = {
        "prompt": text,
        "modalities": modalities,
    }
    mm_data = prompt.get("multi_modal_data")
    if mm_data:
        patched["multi_modal_data"] = mm_data
    return patched


def build_t2i_stage0_prompt(prompt_text: str, *, think: bool = False) -> str:
    """Build the exact text prefix consumed by SenseNova-U1's AR stage for T2I."""
    think_content = "<think>\n" if think else "<think>\n\n</think>\n\n" + IMG_START_TOKEN
    return _build_t2i_query(prompt_text, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)


def build_t2i_cfg_text_prompt(negative_prompt: str = "") -> str:
    """Build the text-unconditional T2I prefix used for CFG companion KV."""
    return _build_t2i_query(negative_prompt, append_text=IMG_START_TOKEN)


def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[ExpandedPrompt]:
    """Expand SenseNova-U1 image requests into CFG companion AR requests.

    For text2img (modalities contains "image"):
      - One extra prompt for ``cfg_text`` when ``cfg_scale`` requires CFG.
      - ``cfg_img`` reuses the conditional KV, so no extra prompt is needed.

    For img2img (modalities contains "img2img"):
      - ``cfg_text`` is added when text CFG is needed.
      - ``cfg_img`` is added when image CFG is needed.

    For text2text or img2text: returns empty list (no expansion needed).

    Args:
        prompt: The original user prompt (dict or string).
        sampling_params: The Stage-0 sampling params.

    Returns:
        List of ExpandedPrompt. Empty if no expansion is needed.
    """
    if not isinstance(prompt, dict):
        return []

    modalities = prompt.get("modalities", [])
    if not isinstance(modalities, list):
        return []

    needs_cfg_text, needs_cfg_img = _cfg_branch_plan(modalities, sampling_params)
    if not needs_cfg_text and not needs_cfg_img:
        return []

    negative_prompt = _get_negative_prompt(prompt, sampling_params)
    cfg_text_prompt = build_t2i_cfg_text_prompt(negative_prompt)
    expanded: list[ExpandedPrompt] = []

    if "image" in modalities:
        expanded.append(
            ExpandedPrompt(
                prompt={"prompt": cfg_text_prompt, "modalities": ["image"]},
                role="cfg_text",
                request_id_suffix=CFG_TEXT_SUFFIX,
            )
        )
        return expanded

    if needs_cfg_text:
        expanded.append(
            ExpandedPrompt(
                prompt={"prompt": cfg_text_prompt, "modalities": ["image"]},
                role="cfg_text",
                request_id_suffix=CFG_TEXT_SUFFIX,
            )
        )

    if needs_cfg_img:
        count = max(_image_count(prompt), 1)
        expanded.append(
            ExpandedPrompt(
                prompt=_prompt_with_mm(prompt, "<image>" * count, ["img2img"]),
                role="cfg_img",
                request_id_suffix=CFG_IMG_SUFFIX,
            )
        )

    return expanded


def collect_cfg_kv_caches(
    request_id: str,
    cfg_request_ids: dict[str, str],
    kv_transfer_manager: Any,
    target_device: Any | None = None,
) -> dict[str, Any]:
    """Collect SenseNova-U1 CFG companion KV caches from the transfer manager."""
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
                    "Collected SenseNova-U1 CFG KV cache for role=%s rid=%s size=%d bytes",
                    role,
                    companion_rid,
                    size,
                )
            else:
                logger.warning(
                    "Failed to collect SenseNova-U1 CFG KV cache for role=%s rid=%s",
                    role,
                    companion_rid,
                )
        except Exception as e:
            logger.exception(
                "Error collecting SenseNova-U1 CFG KV cache for role=%s rid=%s: %s",
                role,
                companion_rid,
                e,
            )

    return result
