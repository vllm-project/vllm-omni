from typing import Any

from typing_extensions import assert_never
from vllm.inputs.data import EmbedsInputs, SingletonInputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalInputs, MultiModalUUIDDict
from vllm.renderers.inputs import SingletonDictPrompt

from vllm_omni.inputs.data import (
    OmniEmbedsPrompt,
    OmniTextPrompt,
    OmniTokenInputs,
    OmniTokensPrompt,
    token_inputs_omni,
)

logger = init_logger(__name__)


class OmniInputPreprocessor(InputPreprocessor):
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
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> OmniTokenInputs | MultiModalInputs:
        prompt_text = parsed_content["prompt"]

        inputs: OmniTokenInputs | MultiModalInputs
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
            prompt_embeds = parsed_content.get("prompt_embeds")
            if prompt_embeds is not None:
                inputs["prompt_embeds"] = prompt_embeds
            additional_information = parsed_content.get("additional_information")
            if additional_information is not None:
                inputs["additional_information"] = additional_information
        else:
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            inputs = token_inputs_omni(
                prompt_token_ids,
                prompt_embeds=parsed_content.get("prompt_embeds"),
                additional_information=parsed_content.get("additional_information"),
            )

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_tokens(
        self,
        parsed_content: OmniTokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> OmniTokenInputs | MultiModalInputs:
        prompt_token_ids = self._truncate_inputs(parsed_content["prompt_token_ids"], tokenization_kwargs)
        prompt_embeds = parsed_content.get("prompt_embeds")
        additional_information = parsed_content.get("additional_information")

        inputs: OmniTokenInputs | MultiModalInputs
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
            if prompt_embeds is not None:
                inputs["prompt_embeds"] = prompt_embeds
            if additional_information is not None:
                inputs["additional_information"] = additional_information
        else:
            inputs = token_inputs_omni(
                prompt_token_ids=prompt_token_ids,
                prompt_embeds=prompt_embeds,
                additional_information=additional_information,
            )

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_embeds(
        self,
        parsed_content: OmniEmbedsPrompt,
    ) -> EmbedsInputs:
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

        return inputs

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> SingletonInputs:
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * prompt: single encoder or decoder input prompt

        Returns:

        * [`SingletonInputs`][vllm.inputs.data.SingletonInputs] instance
        """
        if "prompt_token_ids" in prompt:
            return self._process_tokens(
                prompt,  # type: ignore[arg-type]
                mm_uuids=mm_uuids,
            )

        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        if "prompt" in prompt:
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        assert_never(prompt)  # type: ignore[arg-type]
