# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for HunyuanImage3: AR to Diffusion transition.

In IT2I (image editing) mode:
  - Stage 0 (AR) receives (image + edit instruction), generates CoT/latent tokens
  - Stage 1 (DiT) receives the AR output + original image, denoises to edited image

The ar2diffusion function bridges these two stages, following the same
signature pattern as glm_image.ar2diffusion.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    Resolution,
    ResolutionGroup,
    get_cached_resolution_group,
)
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS,
)
from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def _truncate_at_cot_end(generated_text: str) -> str:
    """Truncate AR output at first `</recaption>` (or `</think>` fallback).

    Mirrors upstream `HunyuanImage3ForCausalMM.generate_image` which feeds
    DiT only the cot text up to the closing tag; the trailing
    `<answer><boi><img_size_*><img_ratio_*>` is consumed via height/width
    extraction and must not leak into DiT's prompt builder.
    """
    for marker in ("</recaption>", "</think>"):
        idx = generated_text.find(marker)
        if idx != -1:
            return generated_text[: idx + len(marker)]
    return generated_text


@lru_cache(maxsize=4)
def _build_ratio_id_lookup() -> dict[int, int]:
    """Return `{token_id: ratio_index}` for HunyuanImage3 ratio tokens.

    The ids are fixed in tokenizer.json and already pinned in prompt_utils.
    Avoid loading AutoTokenizer here: this bridge runs on the hot AR->DiT
    transition path and must keep working in offline deployments where the
    tokenizer object is not exposed to the stage-input processor.
    """
    ratio_0 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"]
    ratio_32 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_32>"]
    ratio_33 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_33>"]
    ratio_36 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_36>"]

    table: dict[int, int] = {}
    for i in range(ratio_32 - ratio_0 + 1):
        table[ratio_0 + i] = i
    base_idx = ratio_32 - ratio_0 + 1
    for j in range(ratio_36 - ratio_33 + 1):
        table[ratio_33 + j] = base_idx + j
    return table


def _extract_ratio_index(generated_token_ids) -> int | None:
    """Resolve the AR-predicted ratio_index from this stage's output.

    `HunyuanImage3ForCausalMM`'s `_stage_transitions` forces the AR to emit
    exactly one `<img_ratio_*>` token after `</recaption><answer><boi>
    <img_size_*>`, so we scan the token stream from the tail for the first
    id that maps to a ratio. Token-ids are the source of truth; text-side
    regex is unreliable because most deploy yamls run AR with
    `skip_special_tokens: True` (special tokens are stripped from text but
    still present in `cumulative_token_ids`).
    """
    if generated_token_ids is None:
        return None
    table = _build_ratio_id_lookup()
    for tid in reversed(list(generated_token_ids)):
        idx = table.get(int(tid))
        if idx is not None:
            return idx
    return None


def ar2diffusion(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> dict[str, Any] | None:
    """Process AR stage outputs to create Diffusion stage inputs.

    HunyuanImage3 produces one downstream diffusion request per parent AR
    request. ``source_outputs`` may include companion outputs, but the parent
    output is expected first: the orchestrator bundles diffusion inputs as
    ``[parent_output, *cfg_companion_outputs]`` before invoking this bridge.

    Args:
        prompt: Original user prompt (may contain multimodal data).
        requires_multimodal_data: Whether to forward multimodal data.

    Returns:
        One prompt dict consumable by the HunyuanImage3 diffusion pipeline, or
        ``None`` when no parent output is available.
    """
    if not source_outputs:
        return None

    # The orchestrator constructs this list as [parent, *cfg_companions].
    ar_output = source_outputs[0]
    output = ar_output.outputs[0]
    generated_token_ids = output.cumulative_token_ids
    # Prefer cumulative_text, fallback to text if aggregation dropped it
    generated_text = getattr(output, "cumulative_text", None) or getattr(output, "text", "") or ""

    if isinstance(prompt, list):
        original_prompt = prompt[0] if prompt else {}
    elif prompt is not None:
        original_prompt = prompt
    else:
        original_prompt = {}
    if isinstance(original_prompt, dict):
        pass
    elif hasattr(original_prompt, "_asdict"):
        original_prompt = original_prompt._asdict()
    elif hasattr(original_prompt, "__dict__"):
        original_prompt = vars(original_prompt)
    else:
        original_prompt = {}

    height = original_prompt.get("height", 1024)
    width = original_prompt.get("width", 1024)
    text_prompt = original_prompt.get("prompt", "")
    use_system_prompt = original_prompt.get("use_system_prompt")
    custom_system_prompt = original_prompt.get("system_prompt")

    # Prefer the AR's predicted output aspect (`<img_size_*><img_ratio_*>`
    # tail emitted by `HunyuanImage3ForCausalMM.sample` under the
    # ratio-restriction logits processor) over the carried-through
    # height/width, which the serving layer fills with the first
    # reference image's bucket and so collapses non-square targets to
    # square in the multi-image / mismatched-aspect case. Mirrors the
    # official upstream where `reso_group[ratio_index]` is the
    # canonical source of the diffusion target shape.
    ratio_idx = _extract_ratio_index(generated_token_ids)
    ar_predicted = False
    if ratio_idx is not None:
        base_size = int(original_prompt.get("image_base_size", 1024))
        reso_group: ResolutionGroup = get_cached_resolution_group(base_size=base_size)
        try:
            reso: Resolution = reso_group[ratio_idx]
            height = reso.height
            width = reso.width
            ar_predicted = True
        except IndexError:
            logger.warning(
                "[ar2diffusion] Request 0: ratio_index=%d out of range [0,%d), keeping prompt size %dx%d",
                ratio_idx,
                len(reso_group),
                height,
                width,
            )

    cot_text_for_dit = _truncate_at_cot_end(generated_text)

    logger.info(
        "[ar2diffusion] Request 0: AR generated %d tokens, text length=%d, cot_text length=%d, target size=%dx%d (%s)",
        len(generated_token_ids),
        len(generated_text),
        len(cot_text_for_dit),
        height,
        width,
        f"AR ratio_idx={ratio_idx}" if ar_predicted else "from prompt (no AR ratio token)",
    )

    diffusion_input: dict[str, Any] = {
        "prompt": text_prompt,
        "height": height,
        "width": width,
        "extra": {
            "ar_generated_text": cot_text_for_dit,
        },
    }

    # Forward use_system_prompt so the DiT can build the same system prefix.
    # Also forward the custom system prompt body when sys_type=custom so
    # DiT's `get_system_prompt(use, "image", body)` doesn't fall back to
    # an empty prefix and silently diverge from AR.
    if use_system_prompt is not None:
        diffusion_input["use_system_prompt"] = use_system_prompt
    if custom_system_prompt is not None:
        diffusion_input["system_prompt"] = custom_system_prompt

    # Forward multimodal data (original image for IT2I conditioning).
    # The diffusion pre_process_func reads multi_modal_data["image"], which
    # matches vLLM's standard prompt schema, so we only need to pass it once.
    mm_data = original_prompt.get("multi_modal_data")
    if mm_data:
        prompt_images = mm_data.get("image")
        if prompt_images is None:
            prompt_images = mm_data.get("images")
        if prompt_images is not None:
            diffusion_input["multi_modal_data"] = {"image": prompt_images}

    # Forward multimodal output from AR (if any)
    if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
        mm_output = ar_output.multimodal_output
        if isinstance(mm_output, Mapping):
            diffusion_input["extra"]["ar_multimodal_output"] = mm_output

    # Forward sampling params
    for key in ["seed", "num_inference_steps", "guidance_scale", "negative_prompt"]:
        if key in original_prompt:
            diffusion_input[key] = original_prompt[key]

    return diffusion_input
