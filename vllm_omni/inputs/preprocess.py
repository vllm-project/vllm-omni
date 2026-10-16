# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni renderer: the upstream renderer class with Omni engine-input routing.

``OmniRenderer`` is a mixin. ``omni_renderer_cls`` puts it in front of a
concrete ``BaseRenderer`` subclass, so the Omni instance *is* an upstream
renderer (tokenizer, executors, multimodal cache and every public entry point
come from upstream unchanged) and only ``_process_singleton`` /
``_process_singleton_async`` are extended. ``build_omni_renderer`` resolves
the concrete class the same way upstream ``renderer_from_config`` does.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from vllm.exceptions import VLLMValidationError
from vllm.renderers import BaseRenderer
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config, tokenizer_args_from_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.tokenizers import TokenizerLike


class OmniRenderer:
    """Mixin that routes Omni prompts through the upstream renderer.

    It has no base class of its own; ``omni_renderer_cls`` prepends it to a
    concrete ``BaseRenderer`` subclass, so ``self`` and ``super()`` are that
    renderer at runtime (hence the casts below).

    Two extensions over upstream ``_process_singleton``:

    * a token prompt that carries ``mm_processor_kwargs`` but no
      ``multi_modal_data`` still goes through the multimodal processor (AR
      image-generation models such as GLM-Image read their target size from
      those kwargs);
    * Omni-only prompt keys (``prompt``, ``cache_salt``,
      ``additional_information``, ``model_intermediate_buffer``) are copied
      onto the engine input.
    """

    @staticmethod
    def _with_omni_extras(inputs, prompt):
        for key in ("prompt", "cache_salt", "additional_information", "model_intermediate_buffer"):
            if key in prompt:
                inputs[key] = prompt[key]
        return inputs

    @staticmethod
    def _routes_no_media_kwargs(prompt) -> bool:
        return "prompt_embeds" not in prompt and "mm_processor_kwargs" in prompt and not prompt.get("multi_modal_data")

    def _validate_video_processor_kwargs(self, prompt) -> None:
        renderer = cast(BaseRenderer, self)
        if "video" not in (prompt.get("multi_modal_data") or {}):
            return
        if getattr(renderer.model_config.hf_config, "model_type", None) not in ("qwen3_vl", "qwen3_vl_moe"):
            return
        processor = renderer.get_mm_processor()
        kwargs = processor.info.ctx.get_merged_mm_kwargs(prompt.get("mm_processor_kwargs") or {}, modality="video")
        # Qwen3-VL merges this dict before entering its HF error wrapper.
        # Reject malformed request options here; unrelated TypeErrors still
        # propagate as internal failures rather than being hidden as bad input.
        size = kwargs.get("size")
        if size is not None and not isinstance(size, dict):
            raise VLLMValidationError(
                "Video processor size must be a dictionary.", parameter="mm_processor_kwargs", value=size
            )
        for name in ("fps", "num_frames"):
            value = kwargs.get(name)
            if value is None:
                continue
            per_video = isinstance(value, list)
            values = value if per_video else [value]
            # vLLM indexes per-video lists before entering its HF error wrapper.
            # Its fps list dispatch specifically requires floats.
            if name == "fps":
                types = (float,) if per_video else (int, float)
                expected = "finite floats" if per_video else "finite numbers"
            else:
                types = (int,)
                expected = "integers"
            if not values or any(
                isinstance(item, bool)
                or not isinstance(item, types)
                or item <= 0
                or (isinstance(item, float) and not math.isfinite(item))
                for item in values
            ):
                raise VLLMValidationError(
                    f"Video processor {name} must contain positive {expected}.",
                    parameter="mm_processor_kwargs",
                    value=value,
                )
            if per_video:
                # Use the upstream parser: a frame list or (frames, metadata)
                # tuple represents one video, not one video per list element.
                video_items = processor.info.parse_mm_data(
                    {"video": prompt["multi_modal_data"]["video"]}, validate=False
                )
                num_videos = video_items.get_count("video", strict=False)
                if len(value) < num_videos:
                    raise VLLMValidationError(
                        f"Video processor {name} needs at least {num_videos} per-video values, got {len(value)}.",
                        parameter="mm_processor_kwargs",
                        value=value,
                    )

    def _process_singleton(self, prompt, *, skip_mm_cache: bool = False, tok_params=None):
        renderer = cast(BaseRenderer, self)
        self._validate_video_processor_kwargs(prompt)
        if self._routes_no_media_kwargs(prompt):
            inputs = renderer._process_multimodal(
                prompt["prompt_token_ids"],
                {},
                mm_processor_kwargs=prompt["mm_processor_kwargs"],
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
            renderer._truncate_expanded_prompt(inputs, tok_params)
        else:
            inputs = cast(BaseRenderer, super())._process_singleton(
                prompt, skip_mm_cache=skip_mm_cache, tok_params=tok_params
            )
        return self._with_omni_extras(inputs, prompt)

    async def _process_singleton_async(self, prompt, *, skip_mm_cache: bool = False, tok_params=None):
        renderer = cast(BaseRenderer, self)
        self._validate_video_processor_kwargs(prompt)
        if self._routes_no_media_kwargs(prompt):
            inputs = await renderer._process_multimodal_async(
                prompt["prompt_token_ids"],
                {},
                mm_processor_kwargs=prompt["mm_processor_kwargs"],
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
            renderer._truncate_expanded_prompt(inputs, tok_params)
        else:
            inputs = await cast(BaseRenderer, super())._process_singleton_async(
                prompt, skip_mm_cache=skip_mm_cache, tok_params=tok_params
            )
        return self._with_omni_extras(inputs, prompt)


_OMNI_RENDERER_CLASSES: dict[type[BaseRenderer], type[BaseRenderer]] = {}


def omni_renderer_cls(base_cls: type[BaseRenderer]) -> type[BaseRenderer]:
    """Return the Omni subclass of ``base_cls`` (created once per base class)."""
    if issubclass(base_cls, OmniRenderer):
        return base_cls
    cls = _OMNI_RENDERER_CLASSES.get(base_cls)
    if cls is None:
        name = f"Omni{base_cls.__name__}"
        cls = type(name, (OmniRenderer, base_cls), {"__module__": __name__, "__qualname__": name})
        _OMNI_RENDERER_CLASSES[base_cls] = cls
    return cls


def build_omni_renderer(
    vllm_config: VllmConfig,
    *,
    renderer_cls: type[BaseRenderer] | None = None,
    tokenizer: TokenizerLike | None = None,
) -> BaseRenderer:
    """Build the Omni renderer for ``vllm_config``.

    With ``renderer_cls`` unset this mirrors upstream ``renderer_from_config``:
    the tokenizer and renderer mode come from the model config and the class
    from ``RENDERER_REGISTRY``. Pass ``renderer_cls`` (and optionally
    ``tokenizer``) to skip that resolution, e.g. for tokenizer-less stages.
    """
    model_config = vllm_config.model_config
    if renderer_cls is None:
        if tokenizer is None:
            tokenizer = cached_tokenizer_from_config(model_config)
        renderer_mode, *_ = tokenizer_args_from_config(model_config)
        renderer_cls = RENDERER_REGISTRY.load_renderer_cls(renderer_mode)
    return omni_renderer_cls(renderer_cls)(vllm_config, tokenizer)
