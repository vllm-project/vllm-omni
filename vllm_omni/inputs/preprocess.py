from typing import Optional, Any, Union
from typing_extensions import assert_never

from vllm.lora.request import LoRARequest
from vllm.inputs.preprocess import InputPreprocessor
from vllm.inputs.data import TokensPrompt, SingletonPrompt, SingletonInputs, TextPrompt
from vllm.multimodal.inputs import MultiModalInputs
from vllm_omni.inputs.data import OmniTokenInputs, token_inputs_omni
from vllm_omni.inputs.parse import parse_singleton_prompt_omni


class OmniInputPreprocessor(InputPreprocessor):
    def _process_tokens(
        self,
        parsed_content: TokensPrompt,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        lora_request: Optional[LoRARequest] = None,
        return_mm_hashes: bool = False,
    ) -> Union[OmniTokenInputs, MultiModalInputs]:
        prompt_token_ids = parsed_content["prompt_token_ids"]
        token_type_ids = parsed_content.get("token_type_ids")
        prompt_embeds = parsed_content.get("prompt_embeds")
        additional_information = parsed_content.get("additional_information")

        inputs: Union[OmniTokenInputs, MultiModalInputs]
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs"),
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request,
                return_mm_hashes=return_mm_hashes,
            )
        else:
            inputs = token_inputs_omni(
                prompt_token_ids=prompt_token_ids,
                token_type_ids=token_type_ids,
                prompt_embeds=prompt_embeds,
                additional_information=additional_information,
            )

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs
    
    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonPrompt,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        lora_request: Optional[LoRARequest] = None,
        return_mm_hashes: bool = False,
    ) -> SingletonInputs:
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * prompt: single encoder or decoder input prompt
        * lora_request: this is only valid for decoder prompts
        * return_mm_hashes: whether to return multimodal hashes

        Returns:

        * [`SingletonInputs`][vllm.inputs.data.SingletonInputs] instance
        """
        parsed = parse_singleton_prompt_omni(prompt)

        if parsed["type"] == "embeds":
            return self._process_embeds(parsed["content"])
        if parsed["type"] == "tokens":
            return self._process_tokens(
                parsed["content"],
                lora_request=lora_request,
                return_mm_hashes=return_mm_hashes,
            )
        if parsed["type"] == "text":
            return self._process_text(
                parsed["content"],
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request,
                return_mm_hashes=return_mm_hashes,
            )
        if parsed["type"] == "str":
            return self._process_text(
                TextPrompt(prompt=parsed["content"]),
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request,
                return_mm_hashes=return_mm_hashes,
            )

        assert_never(parsed)
    
    async def _process_tokens_async(
        self,
        parsed_content: TokensPrompt,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        lora_request: Optional[LoRARequest] = None,
        return_mm_hashes: bool = False,
    ) -> Union[OmniTokenInputs, MultiModalInputs]:
        prompt_token_ids = parsed_content["prompt_token_ids"]
        token_type_ids = parsed_content.get("token_type_ids")
        prompt_embeds = parsed_content.get("prompt_embeds")
        additional_information = parsed_content.get("additional_information")

        inputs: Union[OmniTokenInputs, MultiModalInputs]
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = await self._process_multimodal_async(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs"),
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request,
                return_mm_hashes=return_mm_hashes,
            )
        else:
            inputs = token_inputs_omni(
                prompt_token_ids=prompt_token_ids,
                token_type_ids=token_type_ids,
                prompt_embeds=prompt_embeds,
                additional_information=additional_information,
            )

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs
    
    async def _prompt_to_llm_inputs_async(
        self,
        prompt: SingletonPrompt,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        lora_request: Optional[LoRARequest] = None,
        return_mm_hashes: bool = False,
    ) -> SingletonInputs:
        """
        Async version of
        [`_prompt_to_llm_inputs`][vllm.inputs.preprocess.InputPreprocessor._prompt_to_llm_inputs].
        """
        parsed = parse_singleton_prompt_omni(prompt)

        if parsed["type"] == "embeds":
            return await self._process_embeds_async(parsed["content"])
        if parsed["type"] == "tokens":
            return await self._process_tokens_async(
                parsed["content"],
                lora_request=lora_request,
                return_mm_hashes=return_mm_hashes,
            )
        if parsed["type"] == "text":
            return await self._process_text_async(
                parsed["content"],
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request,
                return_mm_hashes=return_mm_hashes,
            )
        if parsed["type"] == "str":
            return await self._process_text_async(
                TextPrompt(prompt=parsed["content"]),
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request,
                return_mm_hashes=return_mm_hashes,
            )

        assert_never(parsed)