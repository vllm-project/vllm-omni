# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor for GLM-Image: AR → Diffusion transition."""

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTokensPrompt

logger = init_logger(__name__)

# Fallback target edge when neither the request nor the diffusion stage's
# sampling params name a size. Matches the shipped `deploy/glm_image.yaml`.
DEFAULT_TARGET_SIZE = 1024

# Edge length of one AR prior token in pixels. The AR stage emits a
# `(h // 32) x (w // 32)` grid, so only multiples of 32 are representable;
# `GlmImageProcessor._build_prompt_with_target_shape` floors to this factor too.
AR_GRID_FACTOR = 32


def _has_source_image(mm_data: Any) -> bool:
    """Return whether prompt multi_modal_data contains a source image.

    Normalizes legacy/new keys used across omni pipelines:
    - `image`: single PIL image or list
    - `img2img`: legacy single-image key
    - `images`: list or single image
    """
    if not isinstance(mm_data, Mapping):
        return False
    if mm_data.get("image") is not None:
        return True
    if mm_data.get("img2img") is not None:
        return True
    images = mm_data.get("images")
    return bool(images)


def _first_source_image(mm_data: Any) -> Any:
    """Get first source image from normalized multimodal keys."""
    if not isinstance(mm_data, Mapping):
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

    # A healthy completion emits exactly the grid the scaffold asked for and then
    # stops on EOS, so the stream should never be longer than the widest layout
    # the branches below accept (i2i also tolerates a t2i-style small+large
    # stream). An overrun means the offsets we are about to slice at do not
    # delimit the image, and slicing anyway hands the diffusion stage tokens that
    # render as a confidently wrong picture rather than an obvious failure.
    #
    # Two ways to get here, both fatal but worth telling apart:
    #   * no terminal EOS -- generation was cut off by the `max_tokens` ceiling,
    #     so the grid never closed;
    #   * EOS present but too many tokens -- the model emitted a different grid
    #     than the one we scaffolded, so our small/large split is wrong.
    max_usable_tokens = small_image_tokens + large_image_tokens
    if actual_tokens > max_usable_tokens:
        reason = (
            f"never emitted EOS ({eos_token_id}), so generation hit the max_tokens ceiling"
            if not has_terminal_eos
            else "emitted a different grid than the one it was given"
        )
        raise ValueError(
            f"AR stage {reason}: generated {actual_tokens} tokens for a {height}x{width} target "
            f"that needs at most {max_usable_tokens}. The prior tokens are unusable. This target "
            f"size is likely outside the range the AR model supports -- 1024x1024 is the "
            f"reference size."
        )

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

    # Report the size the grid actually describes, not the one that was asked
    # for: the caller feeds this straight to the diffusion stage, which derives
    # its own expected sequence length from it. `token_h * factor` is what the
    # AR scaffold committed to, so a request that was not a multiple of `factor`
    # stays consistent end to end instead of tripping the DiT's length check.
    return prior_token_ids, actual_h * factor, actual_w * factor


def _coerce_dim(value: Any, default: int) -> int:
    """Coerce a requested pixel dimension, falling back on anything unusable."""
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _as_prompt_mapping(prompt: Any) -> Mapping[str, Any]:
    """Normalize an Omni prompt into a mapping of prompt fields.

    A bare ``str`` prompt — what offline ``Omni.generate("<text>")`` hands
    down — becomes ``{"prompt": <text>}``. Without that the caption would be
    dropped on the floor and the diffusion stage conditioned on empty text.
    """
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    if isinstance(prompt, str):
        return {"prompt": prompt}
    if isinstance(prompt, Mapping):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _snap_to_ar_grid(value: int) -> int:
    """Floor a pixel dimension onto the AR prior-token grid.

    The AR stage can only describe whole 32px cells, but the diffusion stage
    floors to `vae_scale_factor * patch_size` (16) instead. Left alone the two
    disagree for anything that is a multiple of 16 but not of 32 -- a 1328px
    edge becomes a 41-cell AR grid (1312px, 82 latent patches) while the DiT
    still expects 83, and the run dies in the pipeline's `prior_token_ids
    seq_len` check. Snapping here keeps every consumer on one size.
    """
    return max(AR_GRID_FACTOR, (value // AR_GRID_FACTOR) * AR_GRID_FACTOR)


def _diffusion_sampling_params(sampling_params_list: Sequence[Any]) -> Any | None:
    """Pick the diffusion stage's params out of the per-stage params list.

    ``None`` when the list does not hold exactly one — the size then falls back
    to :data:`DEFAULT_TARGET_SIZE`, which is what a pipeline without a single
    diffusion stage would have used anyway.
    """
    diffusion_params = [
        sampling_params
        for sampling_params in sampling_params_list
        if isinstance(sampling_params, OmniDiffusionSamplingParams)
    ]
    if len(diffusion_params) != 1:
        return None
    return diffusion_params[0]


def _resolve_target_size(prompt: Mapping[str, Any], sampling_params: Any | None = None) -> tuple[int, int]:
    """Resolve GLM-Image's AR target size in pixels.

    Both size-sensitive sites share this: :func:`prepare_ar_prompt` picks the
    grid the AR stage generates, and :func:`ar2diffusion` re-derives it to slice
    the prior tokens back out. A disagreement between the two would offset that
    slice without raising anything, so they must resolve identically — which
    holds because both are handed the same untransformed prompt and the same
    diffusion sampling params.

    Per dimension, in order: ``mm_processor_kwargs`` (what the OpenAI serving
    layer attaches, so an explicit request size always wins), the top-level
    field (kept for backward compatibility), the diffusion stage's sampling
    params, then :data:`DEFAULT_TARGET_SIZE`.

    The result is snapped onto :data:`AR_GRID_FACTOR` so the AR grid, the prior
    token slice and the diffusion stage all describe the same image.
    """
    mm_processor_kwargs = prompt.get("mm_processor_kwargs")
    if not isinstance(mm_processor_kwargs, Mapping):
        mm_processor_kwargs = {}

    def _resolve(mm_key: str, prompt_key: str) -> int:
        requested = _coerce_dim(
            mm_processor_kwargs.get(mm_key),
            _coerce_dim(
                prompt.get(prompt_key),
                _coerce_dim(getattr(sampling_params, prompt_key, None), DEFAULT_TARGET_SIZE),
            ),
        )
        snapped = _snap_to_ar_grid(requested)
        if snapped != requested:
            logger.warning_once(
                "[glm_image] requested %s=%d is not a multiple of %d; generating %d instead.",
                prompt_key,
                requested,
                AR_GRID_FACTOR,
                snapped,
            )
        return snapped

    return _resolve("target_h", "height"), _resolve("target_w", "width")


def prepare_ar_prompt(prompt: Any, sampling_params_list: Sequence[Any]) -> Any:
    """Stage-0 prompt transform: stamp the AR target size into the prompt.

    The AR stage reads ``target_h``/``target_w`` out of ``mm_processor_kwargs``
    to append GLM-Image's ``<sop>H W<eop>`` grid scaffold and build the M-RoPE
    generation grids. Only a prompt carrying those kwargs reaches the
    multimodal processor at all (``OmniRenderer._routes_no_media_kwargs``), and
    the OpenAI serving layer is the only caller that attaches them — so an
    offline ``Omni.generate("<text>")`` used to reach the AR stage as plain
    text, decode past EOS to the ``max_tokens`` ceiling, and leave the
    diffusion stage conditioned on garbage prior tokens.

    Filling the size in here closes that gap for every entry point. The size
    comes from the diffusion stage's sampling params, so the scaffold matches
    what the caller actually asked for; kwargs already on the prompt take
    precedence, so the served path is unchanged.

    Returns the prompt unchanged when there is nothing to stamp onto, so a
    caller that owns its own AR input keeps it.
    """
    if isinstance(prompt, str):
        fields: dict[str, Any] = {"prompt": prompt}
    elif isinstance(prompt, Mapping):
        # Covers TextPrompt and TokensPrompt alike -- both are TypedDicts.
        # Stamping a token prompt is harmless: the processor guards the scaffold
        # append with a suffix check, so a caller that built its own keeps it.
        fields = dict(prompt)
    else:
        # Anything else (e.g. a list of prompts) is not ours to rewrite.
        return prompt

    # A caller supplying embeddings has already decided the AR stage's input.
    if "prompt_embeds" in fields:
        return prompt

    height, width = _resolve_target_size(fields, _diffusion_sampling_params(sampling_params_list))
    mm_processor_kwargs = dict(fields.get("mm_processor_kwargs") or {})
    # Write the resolved values rather than preserving whatever was there:
    # _resolve_target_size already honors valid kwargs, and this keeps the
    # stamped size identical to what ar2diffusion will re-derive.
    mm_processor_kwargs["target_h"] = height
    mm_processor_kwargs["target_w"] = width
    fields["mm_processor_kwargs"] = mm_processor_kwargs
    return fields


def ar2diffusion(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any] | None:
    """Process AR stage outputs to create Diffusion stage inputs.

    GLM-Image only produces one downstream diffusion request per AR request.
    ``source_outputs`` may still include CFG companion outputs, but only the
    first AR output is used to build the diffusion payload.

    ``sampling_params`` is the diffusion stage's own params, supplied by the
    orchestrator via a signature probe (so the parameter name is load-bearing).
    It is the size fallback for offline requests: ``prompt`` here is the
    *untransformed* prompt, so :func:`prepare_ar_prompt`'s stamped kwargs are
    not visible and the size has to be resolved from the same source again.
    """
    del streaming_context

    _t_total = time.perf_counter()
    if not source_outputs:
        return None

    ar_output = source_outputs[0]
    _t_req = time.perf_counter()
    output = ar_output.outputs[0]
    generated_token_ids = output.cumulative_token_ids

    original_prompt = _as_prompt_mapping(prompt)

    height, width = _resolve_target_size(original_prompt, sampling_params)
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
        if isinstance(mm_output, Mapping) and mm_output.get("ids", {}).get("prior_image") is not None:
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
            0,
            e,
            height,
            width,
            "i2i" if is_i2i else "t2i",
            len(generated_token_ids),
            generated_token_ids[-8:] if len(generated_token_ids) >= 8 else generated_token_ids,
        )
        return None
    _dt_parse = (time.perf_counter() - _t_parse) * 1000

    # Get prior_token_image_ids from AR model output (for i2i mode)
    # This contains VQ-VAE tokens from input image, used for KV cache conditioning
    # NOTE: multimodal_output is attached to ar_output (RequestOutput), NOT output (CompletionOutput)
    _t_prior_img = time.perf_counter()
    prior_token_image_ids = None

    # Check ar_output (RequestOutput) for multimodal_output - this is the correct location
    if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
        mm_output = ar_output.multimodal_output
        if isinstance(mm_output, Mapping):
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
                            f"[ar2diffusion] Request 0: unexpected prior_token_image_ids format: "
                            f"{type(raw_prior_image_ids[0]) if raw_prior_image_ids else 'empty'}"
                        )
    else:
        # Fallback: also check output (CompletionOutput) in case of different vLLM versions
        if hasattr(output, "multimodal_output") and output.multimodal_output:
            mm_output = output.multimodal_output
            logger.debug("[ar2diffusion] Request 0: found multimodal_output on CompletionOutput (fallback)")
            if isinstance(mm_output, Mapping):
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
        0,
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

    _dt_total = (time.perf_counter() - _t_total) * 1000
    logger.info(
        "[ar2diffusion] request done: 1 req, total=%.3fms",
        _dt_total,
    )

    return diffusion_input
