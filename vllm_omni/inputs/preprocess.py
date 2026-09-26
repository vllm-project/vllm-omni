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

from typing import TYPE_CHECKING, cast

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

    def _process_singleton(self, prompt, *, skip_mm_cache: bool = False):
        renderer = cast(BaseRenderer, self)
        if self._routes_no_media_kwargs(prompt):
            inputs = renderer._process_multimodal(
                prompt["prompt_token_ids"],
                {},
                mm_processor_kwargs=prompt["mm_processor_kwargs"],
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
        else:
            inputs = cast(BaseRenderer, super())._process_singleton(prompt, skip_mm_cache=skip_mm_cache)
        return self._with_omni_extras(inputs, prompt)

    async def _process_singleton_async(self, prompt, *, skip_mm_cache: bool = False):
        renderer = cast(BaseRenderer, self)
        if self._routes_no_media_kwargs(prompt):
            inputs = await renderer._process_multimodal_async(
                prompt["prompt_token_ids"],
                {},
                mm_processor_kwargs=prompt["mm_processor_kwargs"],
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
        else:
            inputs = await cast(BaseRenderer, super())._process_singleton_async(prompt, skip_mm_cache=skip_mm_cache)
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
