# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for GLM-Image: AR → Diffusion transition."""

import math
import time
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def _has_source_image(mm_data: Any) -> bool:
    """Return whether prompt multi_modal_data contains a source image.

    Normalizes legacy/new keys used across omni pipelines:
    - `image`: single PIL image or list
    - `img2img`: legacy single-image key
    - `images`: list or single image
    """
    if not isinstance(mm_data, dict):
        return False
    if mm_data.get("image") is not None:
        return True
    if mm_data.get("img2img") is not None:
        return True
    images = mm_data.get("images")
    return bool(images)


def _first_source_image(mm_data: Any) -> Any:
    """Get first source image from normalized multimodal keys."""
    if not isinstance(mm_data, dict):
        return None

    image = mm_data.get("image")
    if image is not None:
        if isinstance(image, list):
            return image[0] if image else None
        return image

    image = mm_data.get("img2img")
    if image is not None:
        if isinstance(image, list):
            return image[0] if image else None
        return image

    images = mm_data.get("images")
    if isinstance(images, list):
        return images[0] if images else None
    return images


def compute_max_tokens(height: int, width: int, factor: int = 32, is_i2i: bool = False) -> int:
    """
    Compute max_new_tokens for GLM-Image AR generation.

    GLM-Image generation differs by mode:

    - text-to-image (t2i): small preview + large target + EOS
    - image-to-image (i2i): large target + EOS

    Args:
        height: Target image height in pixels
        width: Target image width in pixels
        factor: Downsampling factor (32 for GLM-Image AR output)
        is_i2i: Whether the request is image-to-image mode

    Returns:
        Total number of tokens to generate for the specified mode
    """
    # Large image tokens (target resolution)
    token_h = height // factor
    token_w = width // factor
    large_tokens = token_h * token_w

    # Small preview tokens (half resolution in each dimension)
    import math

    ratio = token_h / token_w if token_w > 0 else 1.0
    small_token_h = max(1, int(math.sqrt(ratio) * (factor // 2)))
    small_token_w = max(1, int(math.sqrt(1 / ratio) * (factor // 2)))
    small_tokens = small_token_h * small_token_w

    # Mode-dependent totals:
    # - t2i: small + large + EOS
    # - i2i: large + EOS
    if is_i2i:
        return large_tokens + 1
    return small_tokens + large_tokens + 1


def _upsample_token_ids(token_ids: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
    """Upsample token IDs by 2x using nearest neighbor interpolation.

    GLM-Image AR model generates tokens at 32x downsampling, but DiT expects
    16x downsampling, so we need to upsample by 2x.

    Args:
        token_ids: Prior token IDs of shape [num_tokens]
        token_h: Height in token space (at 32x downsampling)
        token_w: Width in token space (at 32x downsampling)

    Returns:
        Upsampled token IDs of shape [num_tokens * 4]
    """
    token_ids = token_ids.view(1, 1, token_h, token_w)
    token_ids = torch.nn.functional.interpolate(token_ids.float(), scale_factor=2, mode="nearest").to(dtype=torch.long)
    token_ids = token_ids.view(-1)
    return token_ids


def _parse_generated_tokens(
    token_ids: list[int],
    height: int,
    width: int,
    factor: int = 32,
    is_i2i: bool = False,
) -> tuple[torch.Tensor, int, int]:
    """Parse AR-generated tokens to extract prior_token_ids.

    Args:
        token_ids: Generated token IDs from AR model
        height: Target image height
        width: Target image width
        factor: Downsampling factor (default 32)
        is_i2i: Whether this is image-to-image mode. In i2i mode, the AR model
                generates only large image tokens (no small preview tokens).
    """
    # Calculate token dimensions for target image
    token_h = height // factor
    token_w = width // factor
    large_image_tokens = token_h * token_w

    # Calculate small preview image dimensions (used in text-to-image)
    ratio = token_h / token_w if token_w > 0 else 1.0
    small_token_h = max(1, int(math.sqrt(ratio) * (factor // 2)))
    small_token_w = max(1, int(math.sqrt(1 / ratio) * (factor // 2)))
    small_image_tokens = small_token_h * small_token_w

    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    # Remove EOS token (16385) from the end if present
    eos_token_id = 16385
    has_terminal_eos = len(token_ids) > 0 and token_ids[-1] == eos_token_id
    if has_terminal_eos:
        token_tensor = token_tensor[:-1]

    actual_tokens = len(token_tensor)

    if is_i2i:
        if actual_tokens >= small_image_tokens + large_image_tokens:
            large_start = small_image_tokens
            large_end = large_start + large_image_tokens
            prior_token_ids_d32 = token_tensor[large_start:large_end]
            actual_h, actual_w = token_h, token_w
            logger.warning(
                "[_parse_generated_tokens] i2i detected t2i-style token layout; "
                "using small-offset extraction: large_start=%s large_end=%s",
                large_start,
                large_end,
            )
        elif actual_tokens >= large_image_tokens:
            prior_token_ids_d32 = token_tensor[:large_image_tokens]
            actual_h, actual_w = token_h, token_w
            logger.info(
                "[_parse_generated_tokens] i2i using offset-0 extraction: large_tokens=%s",
                large_image_tokens,
            )
        else:
            logger.warning(
                "[_parse_generated_tokens] i2i token parse failed: actual_tokens=%s < expected_large_tokens=%s",
                actual_tokens,
                large_image_tokens,
            )
            raise ValueError(
                f"i2i token parse failed: actual_tokens={actual_tokens} < expected_large_tokens={large_image_tokens}"
            )
    elif actual_tokens >= small_image_tokens + large_image_tokens:
        # Text-to-image: extract large image tokens after small image tokens
        large_start = small_image_tokens
        large_end = large_start + large_image_tokens
        prior_token_ids_d32 = token_tensor[large_start:large_end]
        actual_h, actual_w = token_h, token_w
    elif actual_tokens >= large_image_tokens:
        logger.warning(
            "[_parse_generated_tokens] t2i token parse failed: got only large tokens without small preview "
            "(actual_tokens=%s, expected_small_plus_large=%s)",
            actual_tokens,
            small_image_tokens + large_image_tokens,
        )
        raise ValueError("t2i token parse failed: missing small-preview tokens; refusing low-quality fallback")
    else:
        logger.warning(
            "[_parse_generated_tokens] token parse failed: insufficient tokens "
            "(actual_tokens=%s, expected=%s, mode=%s)",
            actual_tokens,
            large_image_tokens if is_i2i else (small_image_tokens + large_image_tokens),
            "i2i" if is_i2i else "t2i",
        )
        raise ValueError(f"token parse failed: actual_tokens={actual_tokens}, mode={'i2i' if is_i2i else 't2i'}")

    # Upsample from 32x to 16x
    prior_token_ids = _upsample_token_ids(prior_token_ids_d32, actual_h, actual_w)

    return prior_token_ids, height, width


def ar2diffusion(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[dict[str, Any]]:
    """Process AR stage outputs to create Diffusion stage inputs.

    This processor accepts the stage-pool transition interface:
    ``ar2diffusion(source_outputs, prompt, requires_multimodal_data)``.
    """
    del streaming_context

    _t_total = time.perf_counter()
    ar_outputs = source_outputs
    diffusion_inputs = []

    # Normalize prompt to list
    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_output in enumerate(ar_outputs):
        _t_req = time.perf_counter()
        output = ar_output.outputs[0]
        generated_token_ids = output.cumulative_token_ids

        # Get original prompt info
        original_prompt = prompt[i] if i < len(prompt) else {}
        if isinstance(original_prompt, dict):
            pass
        elif hasattr(original_prompt, "_asdict"):
            original_prompt = original_prompt._asdict()
        elif hasattr(original_prompt, "__dict__"):
            original_prompt = vars(original_prompt)
        else:
            original_prompt = {}

        mm_processor_kwargs = original_prompt.get("mm_processor_kwargs")

        def _coerce_dim(v: Any, default: int) -> int:
            try:
                iv = int(v)
                return iv if iv > 0 else default
            except (TypeError, ValueError):
                return default

        # Prefer GLM-Image target size from mm_processor_kwargs (set by serving layer),
        # then fall back to top-level fields for backward compatibility.
        height = _coerce_dim(
            mm_processor_kwargs.get("target_h") if isinstance(mm_processor_kwargs, dict) else None,
            _coerce_dim(original_prompt.get("height"), 1024),
        )
        width = _coerce_dim(
            mm_processor_kwargs.get("target_w") if isinstance(mm_processor_kwargs, dict) else None,
            _coerce_dim(original_prompt.get("width"), 1024),
        )
        text_prompt = original_prompt.get("prompt", "")

        # Detect i2i mode.
        # Prefer normalized prompt multi_modal_data source-image presence, with
        # multimodal output as secondary signal.
        _t_mode = time.perf_counter()
        is_i2i = False

        prompt_modalities = original_prompt.get("modalities")
        if isinstance(prompt_modalities, list) and "img2img" in prompt_modalities:
            is_i2i = True

        prompt_mm_data = original_prompt.get("multi_modal_data")
        if _has_source_image(prompt_mm_data):
            is_i2i = True

        if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
            mm_output = ar_output.multimodal_output
            if isinstance(mm_output, dict) and mm_output.get("ids", {}).get("prior_image") is not None:
                is_i2i = True
        _dt_mode = (time.perf_counter() - _t_mode) * 1000

        # Parse and upsample prior tokens
        _t_parse = time.perf_counter()
        try:
            prior_token_ids, pixel_h, pixel_w = _parse_generated_tokens(
                generated_token_ids,
                height,
                width,
                is_i2i=is_i2i,
            )
        except ValueError as e:
            logger.warning(
                "[ar2diffusion] Request %s: skip due to token parse failure: %s "
                "(target=%sx%s, mode=%s, raw_tokens=%s, tail=%s)",
                i,
                e,
                height,
                width,
                "i2i" if is_i2i else "t2i",
                len(generated_token_ids),
                generated_token_ids[-8:] if len(generated_token_ids) >= 8 else generated_token_ids,
            )
            continue
        _dt_parse = (time.perf_counter() - _t_parse) * 1000

        # Get prior_token_image_ids from AR model output (for i2i mode)
        # This contains VQ-VAE tokens from input image, used for KV cache conditioning
        # NOTE: multimodal_output is attached to ar_output (RequestOutput), NOT output (CompletionOutput)
        _t_prior_img = time.perf_counter()
        prior_token_image_ids = None

        # Check ar_output (RequestOutput) for multimodal_output - this is the correct location
        if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
            mm_output = ar_output.multimodal_output
            if isinstance(mm_output, dict):
                raw_prior_image_ids = mm_output.get("ids", {}).get("prior_image")
                if raw_prior_image_ids is not None:
                    # Handle different formats:
                    # 1. Single tensor -> wrap in list
                    # 2. List of tensors -> use as-is
                    # 3. List of Python lists (from serialization) -> convert to tensors
                    if isinstance(raw_prior_image_ids, torch.Tensor):
                        prior_token_image_ids = [raw_prior_image_ids]
                    elif isinstance(raw_prior_image_ids, list):
                        # Check if elements are tensors or Python lists
                        if raw_prior_image_ids and isinstance(raw_prior_image_ids[0], torch.Tensor):
                            prior_token_image_ids = raw_prior_image_ids
                        elif raw_prior_image_ids and isinstance(raw_prior_image_ids[0], list):
                            # Convert Python lists back to tensors
                            prior_token_image_ids = [torch.tensor(ids, dtype=torch.long) for ids in raw_prior_image_ids]
                        else:
                            logger.warning(
                                f"[ar2diffusion] Request {i}: unexpected prior_token_image_ids format: "
                                f"{type(raw_prior_image_ids[0]) if raw_prior_image_ids else 'empty'}"
                            )
        else:
            # Fallback: also check output (CompletionOutput) in case of different vLLM versions
            if hasattr(output, "multimodal_output") and output.multimodal_output:
                mm_output = output.multimodal_output
                logger.debug(f"[ar2diffusion] Request {i}: found multimodal_output on CompletionOutput (fallback)")
                if isinstance(mm_output, dict):
                    raw_prior_image_ids = mm_output.get("ids", {}).get("prior_image")
                    if raw_prior_image_ids is not None:
                        if isinstance(raw_prior_image_ids, torch.Tensor):
                            prior_token_image_ids = [raw_prior_image_ids]
                        elif isinstance(raw_prior_image_ids, list):
                            prior_token_image_ids = raw_prior_image_ids
        _dt_prior_img = (time.perf_counter() - _t_prior_img) * 1000

        diffusion_input = {
            "prompt": text_prompt,
            "height": pixel_h,
            "width": pixel_w,
            "extra": {
                "prior_token_ids": prior_token_ids,
                "prior_token_image_ids": prior_token_image_ids,
            },
        }

        if requires_multimodal_data:
            mm_data = original_prompt.get("multi_modal_data")
            if mm_data:
                pil_image = _first_source_image(mm_data)
                diffusion_input["pil_image"] = pil_image

        for key in ["seed", "num_inference_steps", "guidance_scale", "negative_prompt"]:
            if key in original_prompt:
                diffusion_input[key] = original_prompt[key]

        _dt_req = (time.perf_counter() - _t_req) * 1000
        logger.info(
            "[ar2diffusion] req=%d mode=%s target=%dx%d "
            "raw_tokens=%d prior_tokens=%d prior_image_ids=%s "
            "timing: mode_detect=%.3fms parse+upsample=%.3fms "
            "prior_image_ids_extract=%.3fms req_total=%.3fms",
            i,
            "i2i" if is_i2i else "t2i",
            pixel_h,
            pixel_w,
            len(generated_token_ids),
            len(prior_token_ids),
            "yes" if prior_token_image_ids is not None else "no",
            _dt_mode,
            _dt_parse,
            _dt_prior_img,
            _dt_req,
        )
        diffusion_inputs.append(diffusion_input)

    _dt_total = (time.perf_counter() - _t_total) * 1000
    logger.info(
        "[ar2diffusion] batch done: %d reqs, total=%.3fms",
        len(diffusion_inputs),
        _dt_total,
    )

    return diffusion_inputs
