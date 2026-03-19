# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for GLM-Image: AR → Diffusion transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


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
    small_token_h = token_h // 2
    small_token_w = token_w // 2
    small_image_tokens = small_token_h * small_token_w

    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    # Remove EOS token (16385) from the end if present
    eos_token_id = 16385
    if len(token_ids) > 0 and token_ids[-1] == eos_token_id:
        token_tensor = token_tensor[:-1]

    actual_tokens = len(token_tensor)

    logger.debug(
        f"[_parse_generated_tokens] height={height}, width={width}, "
        f"token_h={token_h}, token_w={token_w}, "
        f"large_image_tokens={large_image_tokens}, small_image_tokens={small_image_tokens}, "
        f"actual_tokens={actual_tokens}"
    )

    if is_i2i:
        # Image-to-image mode: check if AR generated small+large tokens (like t2i) or just large tokens
        # Some AR models output small+large even in i2i mode
        if actual_tokens >= small_image_tokens + large_image_tokens:
            # AR generated full t2i-style output, extract large tokens after small
            large_start = small_image_tokens
            large_end = large_start + large_image_tokens
            prior_token_ids_d32 = token_tensor[large_start:large_end]
            actual_h, actual_w = token_h, token_w
        else:
            # AR generated only large tokens (pure i2i output)
            prior_token_ids_d32 = token_tensor[:large_image_tokens]
            actual_h, actual_w = token_h, token_w
    elif actual_tokens >= small_image_tokens + large_image_tokens:
        # Text-to-image: extract large image tokens after small image tokens
        large_start = small_image_tokens
        large_end = large_start + large_image_tokens
        prior_token_ids_d32 = token_tensor[large_start:large_end]
        actual_h, actual_w = token_h, token_w
    elif actual_tokens >= large_image_tokens:
        # Image-to-image: large image tokens are at the beginning
        prior_token_ids_d32 = token_tensor[:large_image_tokens]
        actual_h, actual_w = token_h, token_w
    else:
        # Insufficient tokens - try to infer the actual grid size
        import math

        for scale in [1, 2, 4]:
            test_h = token_h // scale
            test_w = token_w // scale
            test_small_h = test_h // 2
            test_small_w = test_w // 2
            test_large = test_h * test_w
            test_small = test_small_h * test_small_w

            if actual_tokens >= test_small + test_large:
                prior_token_ids_d32 = token_tensor[test_small : test_small + test_large]
                actual_h, actual_w = test_h, test_w
                height = test_h * factor
                width = test_w * factor
                logger.warning(f"Adjusted grid to {test_h}x{test_w}, output will be {height}x{width}")
                break
            elif actual_tokens >= test_large:
                prior_token_ids_d32 = token_tensor[:test_large]
                actual_h, actual_w = test_h, test_w
                height = test_h * factor
                width = test_w * factor
                logger.warning(f"Adjusted grid to {test_h}x{test_w}, output will be {height}x{width}")
                break
        else:
            sqrt_tokens = int(math.sqrt(actual_tokens))
            actual_h = actual_w = sqrt_tokens
            usable_tokens = sqrt_tokens * sqrt_tokens
            prior_token_ids_d32 = token_tensor[:usable_tokens]
            height = sqrt_tokens * factor
            width = sqrt_tokens * factor
            logger.error(f"Grid pattern mismatch. Using {sqrt_tokens}x{sqrt_tokens}, output: {height}x{width}")

    # Upsample from 32x to 16x
    prior_token_ids = _upsample_token_ids(prior_token_ids_d32, actual_h, actual_w)

    return prior_token_ids, height, width


def ar2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Process AR stage outputs to create Diffusion stage inputs."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    ar_outputs = stage_list[source_stage_id].engine_outputs
    diffusion_inputs = []

    # Normalize prompt to list
    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_output in enumerate(ar_outputs):
        output = ar_output.outputs[0]
        generated_token_ids = output.token_ids

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

        height = original_prompt.get("height", 1024)
        width = original_prompt.get("width", 1024)
        text_prompt = original_prompt.get("prompt", "")

        # Detect i2i mode first by checking if multimodal_output contains prior_token_image_ids
        is_i2i = False
        if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
            mm_output = ar_output.multimodal_output
            if isinstance(mm_output, dict) and mm_output.get("prior_token_image_ids") is not None:
                is_i2i = True

        # Parse and upsample prior tokens
        prior_token_ids, pixel_h, pixel_w = _parse_generated_tokens(generated_token_ids, height, width, is_i2i=is_i2i)

        # Get prior_token_image_ids from AR model output (for i2i mode)
        # This contains VQ-VAE tokens from input image, used for KV cache conditioning
        # NOTE: multimodal_output is attached to ar_output (RequestOutput), NOT output (CompletionOutput)
        prior_token_image_ids = None

        # Check ar_output (RequestOutput) for multimodal_output - this is the correct location
        if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
            mm_output = ar_output.multimodal_output
            if isinstance(mm_output, dict):
                raw_prior_image_ids = mm_output.get("prior_token_image_ids")
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
                    raw_prior_image_ids = mm_output.get("prior_token_image_ids")
                    if raw_prior_image_ids is not None:
                        if isinstance(raw_prior_image_ids, torch.Tensor):
                            prior_token_image_ids = [raw_prior_image_ids]
                        elif isinstance(raw_prior_image_ids, list):
                            prior_token_image_ids = raw_prior_image_ids

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
                pil_image = mm_data.get("image")
                if pil_image is None:
                    # Try "images" (plural) as fallback
                    images = mm_data.get("images")
                    if images:
                        pil_image = images[0] if isinstance(images, list) else images
                diffusion_input["pil_image"] = pil_image

        for key in ["seed", "num_inference_steps", "guidance_scale", "negative_prompt"]:
            if key in original_prompt:
                diffusion_input[key] = original_prompt[key]

        diffusion_inputs.append(diffusion_input)

    return diffusion_inputs
