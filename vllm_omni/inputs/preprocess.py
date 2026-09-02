# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

from typing_extensions import assert_never
from vllm.inputs import (
    EmbedsInput,
    MultiModalDataDict,
    MultiModalInput,
    MultiModalUUIDDict,
    SingletonInput,
    TextPrompt,
    TokensInput,
    TokensPrompt,
    tokens_input,
)
from vllm.logger import init_logger
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.renderers.inputs import SingletonDictPrompt

from vllm_omni.inputs.data import (
    OmniEmbedsPrompt,
    OmniTextPrompt,
    OmniTokenInputs,
    OmniTokensPrompt,
    token_inputs_omni,
)

logger = init_logger(__name__)


class _InputPreprocessor:
    """Compatibility shim for the upstream-removed ``InputPreprocessor``.

    Upstream vLLM moved raw-prompt preprocessing into
    :class:`vllm.renderers.BaseRenderer` and removed
    ``vllm.inputs.preprocess.InputPreprocessor``. vLLM-Omni still exposes an
    omni-specific preprocessor object for stage-0 input processing, so keep the
    same thin renderer-delegating surface locally and adapt the few calls whose
    renderer signature changed upstream.
    """

    def __init__(
        self,
        vllm_config: Any,
        renderer: BaseRenderer | None = None,
    ) -> None:
        self.model_config = vllm_config.model_config
        self.renderer = renderer or renderer_from_config(vllm_config)

    @property
    def tokenizer(self) -> Any | None:
        return self.renderer.tokenizer

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Apply the model's tokenizer to a text prompt."""
        tok_params = self.renderer.default_cmpl_tok_params.with_kwargs(**(tokenization_kwargs or {}))

        tok_prompt = self.renderer._tokenize_singleton_prompt(
            TextPrompt(prompt=prompt),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    def _process_multimodal(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInput:
        """Apply the model's multi-modal processor to a multi-modal prompt.

        The upstream renderer no longer accepts raw ``str`` prompts or
        ``tokenization_kwargs`` in ``_process_multimodal``; tokenize text
        prompts here first to preserve the legacy ``InputPreprocessor`` API.
        """
        if isinstance(prompt, str):
            prompt = self._tokenize_prompt(prompt, tokenization_kwargs)

        return self.renderer._process_multimodal(
            prompt,
            mm_data,
            mm_uuids=mm_uuids,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def _process_embeds(
        self,
        parsed_content: OmniEmbedsPrompt,
    ) -> EmbedsInput:
        return self.renderer._process_embeds(parsed_content)

    def _truncate_inputs(
        self,
        inputs: list[int],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        tok_params = self.renderer.default_cmpl_tok_params.with_kwargs(**(tokenization_kwargs or {}))

        tok_prompt = self.renderer._tokenize_singleton_prompt(
            TokensPrompt(prompt_token_ids=inputs),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    def _process_tokens(
        self,
        parsed_content: OmniTokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> TokensInput | MultiModalInput:
        prompt_token_ids = self._truncate_inputs(parsed_content["prompt_token_ids"], tokenization_kwargs)

        inputs: TokensInput | MultiModalInput
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs"),
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=parsed_content.get("multi_modal_uuids"),
            )
        else:
            inputs = tokens_input(prompt_token_ids)

        if prompt_text := parsed_content.get("prompt"):
            inputs["prompt"] = prompt_text
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_text(
        self,
        parsed_content: OmniTextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> TokensInput | MultiModalInput:
        prompt_text = parsed_content["prompt"]

        inputs: TokensInput | MultiModalInput
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
        else:
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            inputs = tokens_input(prompt_token_ids)

        inputs["prompt"] = prompt_text

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> SingletonInput:
        """Extract the singleton inputs from a prompt."""
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        if "prompt_token_ids" in prompt:
            return self._process_tokens(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
            )

        if "prompt" in prompt:
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        assert_never(prompt)  # type: ignore[arg-type]


class OmniInputPreprocessor(_InputPreprocessor):
    """Input preprocessor for omni models.

    Extends the base InputPreprocessor to handle omni-specific input
    types including prompt embeddings and additional information payloads.
    Supports processing tokens, embeddings, text, and multimodal inputs.
    """

    def _process_text(
        self,
        parsed_content: OmniTextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: Any | None = None,
    ) -> OmniTokenInputs | MultiModalInput:
        """Process text prompts with support for mm_processor_kwargs.

        Extends base class to support mm_processor_kwargs without multi_modal_data.
        This is needed for models like GLM-Image where text-to-image generation
        requires processor kwargs (target_h, target_w) to format the prompt.
        """
        prompt_text = parsed_content["prompt"]
        mm_processor_kwargs = parsed_content.get("mm_processor_kwargs") or {}
        # When the deprecated raw-prompt path is used, process_inputs does
        # not pass mm_uuids to preprocess().  Fall back to reading it from
        # the prompt dict so the Renderer's _validate_mm_uuids can see it.
        effective_mm_uuids = mm_uuids or parsed_content.get("multi_modal_uuids")

        inputs: OmniTokenInputs | MultiModalInput
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                mm_processor_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=effective_mm_uuids,
            )
            prompt_embeds = parsed_content.get("prompt_embeds")
            if prompt_embeds is not None:
                inputs["prompt_embeds"] = prompt_embeds
            additional_information = parsed_content.get("additional_information")
            if additional_information is not None:
                inputs["additional_information"] = additional_information
            model_intermediate_buffer = parsed_content.get("model_intermediate_buffer")
            if model_intermediate_buffer is not None:
                inputs["model_intermediate_buffer"] = model_intermediate_buffer
        elif "mm_processor_kwargs" in parsed_content:
            # Presence — not truthiness. An explicitly-set empty dict still
            # signals "route through the multimodal processor" (needed for
            # AR-based image-gen where the HF processor supplies its own
            # defaults and scaffold).
            inputs = self._process_multimodal(
                prompt_text,
                {},
                mm_processor_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=effective_mm_uuids,
            )
        else:
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            inputs = token_inputs_omni(
                prompt_token_ids,
                prompt_embeds=parsed_content.get("prompt_embeds"),
                additional_information=parsed_content.get("additional_information"),
                model_intermediate_buffer=parsed_content.get("model_intermediate_buffer"),
            )
        prompt_embeds = parsed_content.get("prompt_embeds")
        if prompt_embeds is not None:
            inputs["prompt_embeds"] = prompt_embeds
        additional_information = parsed_content.get("additional_information")
        if additional_information is not None:
            inputs["additional_information"] = additional_information
        model_intermediate_buffer = parsed_content.get("model_intermediate_buffer")
        if model_intermediate_buffer is not None:
            inputs["model_intermediate_buffer"] = model_intermediate_buffer
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_tokens(
        self,
        parsed_content: OmniTokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> OmniTokenInputs | MultiModalInput:
        prompt_token_ids = self._truncate_inputs(parsed_content["prompt_token_ids"], tokenization_kwargs)
        prompt_embeds = parsed_content.get("prompt_embeds")
        additional_information = parsed_content.get("additional_information")
        model_intermediate_buffer = parsed_content.get("model_intermediate_buffer")

        multi_modal_data = parsed_content.get("multi_modal_data")

        inputs: OmniTokenInputs | MultiModalInput
        if multi_modal_data:
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs"),
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=parsed_content.get("multi_modal_uuids"),
            )

        else:
            inputs = token_inputs_omni(
                prompt_token_ids=prompt_token_ids,
                prompt_embeds=prompt_embeds,
                additional_information=additional_information,
                model_intermediate_buffer=model_intermediate_buffer,
            )
        if prompt_embeds is not None:
            inputs["prompt_embeds"] = prompt_embeds
        if additional_information is not None:
            inputs["additional_information"] = additional_information
        if model_intermediate_buffer is not None:
            inputs["model_intermediate_buffer"] = model_intermediate_buffer
        if prompt_text := parsed_content.get("prompt"):
            inputs["prompt"] = prompt_text
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_embeds(
        self,
        parsed_content: OmniEmbedsPrompt,
    ) -> EmbedsInput:
        """Process embeddings prompt with omni-specific extensions.

        Extends base _process_embeds to handle additional_information payload
        for direct transfer between pipeline stages.
        """
        # Call parent implementation for base embeds processing
        inputs = super()._process_embeds(parsed_content)

        # Add omni-specific additional_information if present
        additional_information = parsed_content.get("additional_information")
        if additional_information is not None:
            inputs["additional_information"] = additional_information  # type: ignore[typeddict-unknown-key]
        model_intermediate_buffer = parsed_content.get("model_intermediate_buffer")
        if model_intermediate_buffer is not None:
            inputs["model_intermediate_buffer"] = model_intermediate_buffer  # type: ignore[typeddict-unknown-key]

        return inputs

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: Any | None = None,
    ) -> SingletonInput:
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * prompt: single encoder or decoder input prompt

        Returns:

        * [`SingletonInput`][vllm.inputs.engine.SingletonInput] instance
        """
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        if "prompt_token_ids" in prompt:
            return self._process_tokens(
                prompt,  # type: ignore[arg-type]
            )

        if "prompt" in prompt:
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        assert_never(prompt)  # type: ignore[arg-type]
