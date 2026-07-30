# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


def build_x_to_text_prompt(
    model: str,
    prompt: str,
    has_image: bool,
) -> tuple[dict[str, Any], list[int] | None]:
    """Build HunyuanImage-3 prompt tokens and stopping rules for T2T/I2T."""
    from transformers import AutoTokenizer

    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        build_prompt_tokens,
        resolve_stop_token_ids,
        resolve_sys_type,
    )

    task = "i2t" if has_image else "t2t"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    build_kwargs: dict[str, Any] = {"task": task, "bot_task": None}
    if has_image:
        build_kwargs["num_images"] = 1
    built = build_prompt_tokens(prompt, tokenizer, **build_kwargs)
    return (
        {
            "prompt": prompt,
            "prompt_token_ids": built.token_ids,
            "modalities": ["text"],
            "use_system_prompt": resolve_sys_type(None),
        },
        resolve_stop_token_ids(task=task, bot_task=None, tokenizer=tokenizer, image_size="auto"),
    )


def build_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    """Build a HunyuanImage-3.0 text-to-image engine prompt.

    Returns a lightweight payload only. The AR-stage prompt token-ids and
    stop-token-ids are sourced inside the model's AR input path (driven by the
    ``bot_task`` / ``use_system_prompt`` knobs that flow via ``extra_body`` ->
    ``extra_args``), mirroring the MammothModa2 pattern -- the example no longer
    tokenizes. ``height`` / ``width`` travel through OmniDiffusionSamplingParams.
    """
    del height, width
    out: dict[str, Any] = {"prompt": prompt, "modalities": ["image"]}
    if negative_prompt is not None:
        out["negative_prompt"] = negative_prompt
    return out


def build_image_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    input_image: Image.Image | list[Image.Image],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    """Build a HunyuanImage-3.0 image-editing (IT2I) engine prompt.

    A single reference image is forwarded as-is; a list (up to
    ``MAX_IMAGES_PER_REQUEST``) is forwarded for multi-image editing. As with
    text-to-image, AR token-ids / stop-token-ids are sourced inside the AR
    input path from the ``bot_task`` / ``use_system_prompt`` knobs.
    """
    del height, width
    out: dict[str, Any] = {
        "prompt": prompt,
        "modalities": ["image"],
        "multi_modal_data": {"image": input_image},
    }
    if negative_prompt is not None:
        out["negative_prompt"] = negative_prompt
    return out


@dataclass
class ARStageInputs:
    """AR-stage inputs the shared example applies to a HunyuanImage3 request.

    ``prompt_token_ids`` (byte-for-byte HF parity) is preferred; ``prompt``
    is the string fallback used when no tokenizer is available. ``modalities``
    picks the AR output type (``["image"]`` for generation, ``["text"]`` for
    comprehension). ``use_system_prompt`` feeds the DiT system-prompt prefix
    and ``stop_token_ids`` terminate the AR decode.

    ``stage_indices`` names which stage(s) in the request's
    ``sampling_params_list`` these ``stop_token_ids`` belong to -- the shared
    scripts apply them by explicit index, not by scanning for "whichever
    stage isn't a diffusion stage." HunyuanImage-3.0's topology always puts
    its single AR stage at index 0 (see
    ``vllm_omni.model_executor.models.hunyuan_image3.pipeline``); a future
    model with multiple AR/understanding stages would declare its own
    indices here instead of relying on type-based inference.
    """

    prompt: str | None
    prompt_token_ids: list[int] | None
    modalities: list[str]
    use_system_prompt: str
    stop_token_ids: list[int]
    stage_indices: list[int] = field(default_factory=lambda: [0])


def build_ar_stage_inputs(
    prompt: str,
    tokenizer: Any | None,
    extra_body: Mapping[str, Any] | None,
    *,
    num_images: int = 0,
    height: int | None = None,
    width: int | None = None,
    text_output: bool = False,
) -> ARStageInputs:
    """Resolve HunyuanImage-3.0 AR-stage inputs declaratively.

    Invoked generically by the shared task examples (``text_to_image`` /
    ``image_to_image`` / understanding / chat) whenever the model declares an
    ``ar_input_builder``. All model-specific knobs (``bot_task`` /
    ``use_system_prompt`` / ``system_prompt``) arrive via ``extra_body``, so
    the example itself stays model-agnostic. The heavy lifting is delegated to
    :func:`build_ar_prompt_inputs`, the same seam the OpenAI server uses.
    """
    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        build_ar_prompt_inputs,
    )

    extra_body = extra_body or {}
    has_image = num_images > 0
    if text_output:
        task = "i2t" if has_image else "t2t"
    else:
        task = "it2i" if has_image else "t2i"

    # ar_image_size: None -> AR predicts aspect ratio; explicit "{w}x{h}" ->
    # AR stops at the terminator. Comprehension (text) tasks always auto-size.
    ar_image_size: str | None = None
    if not text_output and height is not None and width is not None:
        ar_image_size = f"{width}x{height}"

    kwargs: dict[str, Any] = {
        "task": task,
        "sys_type": extra_body.get("use_system_prompt"),
        "custom_system_prompt": extra_body.get("system_prompt"),
        "num_images": max(num_images, 1) if has_image else 1,
        "tokenizer": tokenizer,
        "image_size": ar_image_size,
    }
    # Only forward bot_task when the caller set it: present (incl. explicit
    # ``None`` -> plain mode) overrides; absent lets each task pick its default
    # trigger via build_ar_prompt_inputs' own sentinel default.
    if "bot_task" in extra_body:
        kwargs["bot_task"] = extra_body["bot_task"]

    resolved = build_ar_prompt_inputs(prompt, **kwargs)
    return ARStageInputs(
        prompt=resolved.prompt,
        prompt_token_ids=resolved.prompt_token_ids,
        modalities=["text"] if text_output else ["image"],
        use_system_prompt=resolved.system_prompt_type,
        stop_token_ids=resolved.stop_token_ids,
        # HunyuanImage-3.0's AR stage is always stage 0 (see pipeline.py's
        # HUNYUAN_IMAGE3_PIPELINE / _AR_PIPELINE topologies).
        stage_indices=[0],
    )


def validate_ar_tokenizer(tokenizer: Any) -> None:
    """Registry-declared ``ar_tokenizer_validator`` hook for HunyuanImage3.

    Called by the shared task examples right after they load a *real*
    tokenizer for the AR stage, so a model/tokenizer revision bump that
    silently shifts special-token ids fails loudly instead of producing a
    request that completes with the wrong stop tokens.
    """
    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        validate_special_token_ids,
    )

    validate_special_token_ids(tokenizer)


# Per-request, model-specific knobs (non-standard sampling-param fields).
# Standard knobs (guidance_scale / num_inference_steps / seed / height / width)
# stay on OmniDiffusionSamplingParams; engine-level knobs
# (diffusion_kv_cache_* / vae_use_tiling) stay as Omni() init args.
HUNYUAN_IMAGE3_EXTRA_BODY_PARAMS = frozenset(
    {
        "bot_task",
        "use_system_prompt",
        "system_prompt",
        "negative_prompt",
    }
)
# Text outputs for i2t/t2t are surfaced through the standard AR text path;
# no diffusion custom-output params are declared.
HUNYUAN_IMAGE3_EXTRA_OUTPUT_PARAMS = frozenset()
# HunyuanImage3 runs an AR stage before DiT; its extra_args must be initialised
# so bot_task / use_system_prompt reach the AR input path.
HUNYUAN_IMAGE3_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES = True
