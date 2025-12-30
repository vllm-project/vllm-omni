import asyncio
import base64
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import TYPE_CHECKING, Any, Optional

import jinja2
from fastapi import Request
from PIL import Image
from pydantic import TypeAdapter

try:
    import soundfile
except ImportError:
    soundfile = None

from openai.types.chat.chat_completion_audio import ChatCompletionAudio as OpenAIChatCompletionAudio
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
    apply_hf_chat_template,
    apply_mistral_chat_template,
    get_history_tool_calls_cnt,
    make_tool_call_id,
    resolve_chat_template_content_format,
)
from vllm.entrypoints.harmony_utils import parse_chat_output
from vllm.entrypoints.openai.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    FunctionCall,
    FunctionDefinition,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import (
    ChatLikeRequest,
    EngineTokensPrompt,
    RequestPrompt,
    ResponsesRequest,
    TextTokensPrompt,
    clamp_prompt_logprobs,
    is_list_of,
)
from vllm.entrypoints.openai.tool_parsers import ToolParser
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import MistralToolCall
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.mistral import (
    MistralTokenizer,
    maybe_serialize_tool_calls,
    truncate_tool_call_ids,
    validate_request_params,
)
from vllm.utils.collection_utils import as_list

from vllm_omni.entrypoints.chat_utils import parse_chat_messages_futures
from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import AudioResponse, CreateAudio
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion

logger = init_logger(__name__)


class OmniOpenAIServingChat(OpenAIServingChat, AudioMixin):
    """OpenAI-compatible chat serving for both LLM and Diffusion models.

    This class extends OpenAIServingChat to support:
    - Standard LLM chat completions
    - Diffusion model image generation via chat interface

    For diffusion mode, use the `for_diffusion` class method to create an instance.
    """

    # Diffusion mode attributes
    _diffusion_mode: bool = False
    _diffusion_engine: Optional["AsyncOmniDiffusion"] = None
    _diffusion_model_name: str = ""

    @classmethod
    def for_diffusion(
        cls,
        diffusion_engine: "AsyncOmniDiffusion",
        model_name: str,
    ) -> "OmniOpenAIServingChat":
        """Create a chat serving instance for diffusion models.

        Args:
            diffusion_engine: The async diffusion engine
            model_name: Name of the model being served

        Returns:
            OmniOpenAIServingChat instance configured for diffusion mode

        Note:
            Request-level parameters (num_inference_steps, guidance_scale, seed,
            height, width, num_frames, fps, etc.) are passed per-request via the API.
        """
        instance = cls.__new__(cls)
        instance._diffusion_mode = True
        instance._diffusion_engine = diffusion_engine
        instance._diffusion_model_name = model_name
        return instance

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.

        For diffusion models, this generates images and returns them
        in a chat completion response format.
        """
        # Handle diffusion mode
        if self._diffusion_mode:
            return await self._create_diffusion_chat_completion(request, raw_request)

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

            model_name = self.models.model_name(lora_request)

            tokenizer = await self.engine_client.get_tokenizer()

            tool_parser = self.tool_parser

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)
                truncate_tool_call_ids(request)
                validate_request_params(request)

            if (
                request.tool_choice == "auto"
                and not (self.enable_auto_tools and tool_parser is not None)
                and not isinstance(tokenizer, MistralTokenizer)
                and not self.use_harmony
            ):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
                )

            if request.tools is None or (request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none):
                tool_dicts = None
            else:
                tool_dicts = [tool.model_dump() for tool in request.tools]

            # Common case.
            request_chat_template = request.chat_template
            chat_template_kwargs = request.chat_template_kwargs
            if not self.trust_request_chat_template and (
                request_chat_template is not None
                or (chat_template_kwargs and chat_template_kwargs.get("chat_template") is not None)
            ):
                return self.create_error_response(
                    "Chat template is passed with request, but --trust-request-chat-template is not set. "
                    "Refused request with untrusted chat template."
                )
            (
                conversation,
                request_prompts,
                engine_prompts,
            ) = await self._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=request_chat_template or self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
                tool_dicts=tool_dicts,
                documents=request.documents,
                chat_template_kwargs=request.chat_template_kwargs,
                tool_parser=tool_parser,
                add_special_tokens=request.add_special_tokens,
            )

        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        output_modalities = getattr(request, "modalities", self.engine_client.output_modalities)

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                if hasattr(request, "sampling_params_list"):
                    sampling_params_list = self._to_sampling_params_list(request.sampling_params_list)
                else:
                    # Use standard OpenAI API parameters for comprehension stage
                    sampling_params_list = self._build_sampling_params_list_from_request(request)

                self._log_inputs(
                    request_id,
                    request_prompts[i],
                    params_list=sampling_params_list,
                    lora_request=lora_request,
                )

                trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)

                generator = self.engine_client.generate(
                    prompt=engine_prompt,
                    request_id=request_id,
                    sampling_params_list=sampling_params_list,
                    output_modalities=output_modalities,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        (result_generator,) = generators

        # Streaming response
        if request.stream:
            raise RuntimeError("Not support streaming output now.")

        try:
            return await self.chat_completion_full_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def _preprocess_chat(
        self,
        request: ChatLikeRequest | ResponsesRequest,
        tokenizer: TokenizerLike,
        messages: list[ChatCompletionMessageParam],
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
        add_special_tokens: bool = False,
    ) -> tuple[
        list[ConversationMessage],
        Sequence[RequestPrompt],
        list[EngineTokensPrompt],
    ]:
        model_config = self.model_config

        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tool_dicts,
            chat_template_content_format,
            tokenizer,
            model_config=model_config,
        )
        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages,
            model_config,
            tokenizer,
            content_format=resolved_content_format,
            mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
        )

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tool_dicts,
            documents=documents,
        )
        _chat_template_kwargs.update(chat_template_kwargs or {})

        request_prompt: str | list[int]

        if tokenizer is None:
            request_prompt = "placeholder"
        elif isinstance(tokenizer, MistralTokenizer):
            request_prompt = apply_mistral_chat_template(
                tokenizer,
                messages=messages,
                **_chat_template_kwargs,
            )
        else:
            request_prompt = apply_hf_chat_template(
                tokenizer=tokenizer,
                conversation=conversation,
                model_config=model_config,
                **_chat_template_kwargs,
            )

        mm_data = await mm_data_future

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        should_parse_tools = tool_parser is not None and (
            hasattr(request, "tool_choice") and request.tool_choice != "none"
        )

        if should_parse_tools:
            if not isinstance(request, ChatCompletionRequest):
                msg = "Tool usage is only supported for Chat Completions API"
                raise NotImplementedError(msg)

            request = tool_parser(tokenizer).adjust_request(  # type: ignore
                request=request
            )

        if tokenizer is None:
            assert isinstance(request_prompt, str), (
                "Prompt has to be a string",
                "when the tokenizer is not initialised",
            )
            prompt_inputs = TextTokensPrompt(prompt=request_prompt, prompt_token_ids=[1])
        elif isinstance(request_prompt, str):
            prompt_inputs = await self._tokenize_prompt_input_async(
                request,
                tokenizer,
                request_prompt,
                add_special_tokens=add_special_tokens,
            )
        else:
            # For MistralTokenizer
            assert is_list_of(request_prompt, int), "Prompt has to be either a string or a list of token ids"
            prompt_inputs = TextTokensPrompt(
                prompt=tokenizer.decode(request_prompt),
                prompt_token_ids=request_prompt,
            )

        engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_inputs["prompt_token_ids"])
        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data

        if mm_uuids is not None:
            engine_prompt["multi_modal_uuids"] = mm_uuids

        if request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        if hasattr(request, "cache_salt") and request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return conversation, [request_prompt], [engine_prompt]

    def _to_sampling_params_list(self, sampling_params_list: list[dict]) -> list[SamplingParams]:
        final_sampling_params_list = []
        for sampling_params in sampling_params_list:
            if isinstance(sampling_params, dict):
                final_sampling_params_list.append(SamplingParams(**sampling_params))
            elif isinstance(sampling_params, SamplingParams):
                final_sampling_params_list.append(sampling_params)
            else:
                raise ValueError(f"Invalid sampling params: {sampling_params}")
        return final_sampling_params_list

    def _get_comprehension_stage_index(self) -> int:
        for idx, stage in enumerate(self.engine_client.stage_list):
            if stage.is_comprehension:
                return idx
        raise ValueError("No comprehension stage (is_comprehension=True) found in stage_list")

    # OpenAI API standard sampling parameters that can be safely overridden.
    # These are the most commonly used parameters with compatible types
    # between ChatCompletionRequest and SamplingParams.
    # Users who need more control can use sampling_params_list in extra_body.
    _OPENAI_SAMPLING_FIELDS: set[str] = {
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "stop",
        "frequency_penalty",
        "presence_penalty",
    }

    def _apply_request_overrides(
        self,
        default_params: SamplingParams,
        request: ChatCompletionRequest,
    ) -> SamplingParams:
        """Clone default params and override with user-provided request values.

        Starts with YAML defaults and only overrides fields that the user
        explicitly provided (non-None values) in the request.

        Args:
            default_params: Default SamplingParams from stage config YAML.
            request: The chat completion request containing user-provided values.

        Returns:
            New SamplingParams with YAML defaults overridden by request values.
        """
        params = default_params.clone()

        for field_name in self._OPENAI_SAMPLING_FIELDS:
            value = getattr(request, field_name, None)
            if value is not None:
                setattr(params, field_name, value)

        return params

    def _build_sampling_params_list_from_request(
        self,
        request: ChatCompletionRequest,
    ) -> list[SamplingParams]:
        """Build sampling_params_list using standard OpenAI API parameters.

        For the comprehension stage, starts with YAML defaults and overrides with
        user-provided request values. For other stages, uses cloned YAML defaults.

        This approach ensures all YAML defaults (including seed, detokenize, etc.)
        are preserved while allowing users to override specific parameters.

        Args:
            request: The chat completion request containing OpenAI API parameters.

        Returns:
            List of SamplingParams, one for each stage.
        """
        default_params_list = self.engine_client.default_sampling_params_list
        comprehension_idx = self._get_comprehension_stage_index()

        sampling_params_list = []
        for idx, default_params in enumerate(default_params_list):
            if isinstance(default_params, dict):
                default_params = SamplingParams(**default_params)
            if idx == comprehension_idx:
                params = self._apply_request_overrides(default_params, request)
                sampling_params_list.append(params)
            else:
                # For other stages, clone default params
                sampling_params_list.append(default_params.clone())

        return sampling_params_list

    def _log_inputs(
        self,
        request_id: str,
        inputs: RequestPrompt | PromptType,
        params_list: list[SamplingParams] | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return
        prompt, prompt_token_ids, prompt_embeds = None, None, None
        if isinstance(inputs, str):
            prompt = inputs
        elif isinstance(inputs, list):
            prompt_token_ids = inputs
        else:
            prompt = getattr(inputs, "prompt", None)
            prompt_token_ids = getattr(inputs, "prompt_token_ids", None)

        logger.info(
            "Received request %s: prompt: %r, params_list: %s, prompt_token_ids: %s, prompt_embeds shape: %s, lora_request: %s.",  # noqa: E501
            request_id,
            prompt,
            params_list,
            prompt_token_ids,
            prompt_embeds.shape if prompt_embeds is not None else None,
            lora_request,
        )

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        final_outputs: list[OmniRequestOutput] = []
        try:
            async for res in result_generator:
                final_outputs.append(res)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_outputs is not None

        choices: list[ChatCompletionResponseChoice] = []

        usage = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        role = self.get_chat_request_role(request)
        prompt_logprobs = None
        prompt_token_ids = None
        kv_transfer_params = None

        for omni_outputs in final_outputs:
            choices_data = []
            if omni_outputs.final_output_type == "text":
                (
                    choices_data,
                    usage,
                    prompt_logprobs,
                    prompt_token_ids,
                    kv_transfer_params,
                ) = self._create_text_choice(request, omni_outputs, tokenizer, conversation, role)
            elif omni_outputs.final_output_type == "audio":
                choices_data = self._create_audio_choice(omni_outputs, role)
            elif omni_outputs.final_output_type == "image":
                choices_data = self._create_image_choice(omni_outputs, role)
            else:
                logger.warning(f"Unsupported final output type: {omni_outputs.final_output_type}")
                continue
            choices.extend(choices_data)

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=prompt_logprobs,
            prompt_token_ids=prompt_token_ids,
            kv_transfer_params=kv_transfer_params,
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:
                        if hasattr(tc.function, "name") and hasattr(tc.function, "arguments"):
                            tool_call_descriptions.append(f"{tc.function.name}({tc.function.arguments})")
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _create_text_choice(
        self,
        request: ChatCompletionRequest,
        omni_outputs: OmniRequestOutput,
        tokenizer: TokenizerLike,
        conversation: list[ConversationMessage],
        role: str,
    ):
        final_res = omni_outputs.request_output
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        choices: list[ChatCompletionResponseChoice] = []

        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs
            tool_call_info = None

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if self.use_harmony:
                reasoning_content, content, _ = parse_chat_output(token_ids)
                if not request.include_reasoning:
                    reasoning_content = None

                if self.tool_parser is not None:
                    tool_parser = self.tool_parser(tokenizer)
                    # NOTE: We use token_ids for openai tool parser
                    tool_call_info = tool_parser.extract_tool_calls(
                        "",
                        request=request,
                        token_ids=token_ids,  # type: ignore
                    )
                    content = tool_call_info.content
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=content,
                        tool_calls=tool_call_info.tool_calls,
                    )
                else:
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=content,
                    )

                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    logprobs=logprobs,
                    finish_reason=(
                        "tool_calls"
                        if (tool_call_info is not None and tool_call_info.tools_called)
                        else (output.finish_reason if output.finish_reason else "stop")
                    ),
                    stop_reason=output.stop_reason,
                )
                choices.append(choice_data)
                continue

            if self.reasoning_parser:
                try:
                    reasoning_parser = self.reasoning_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning_content, content = reasoning_parser.extract_reasoning_content(output.text, request=request)
                if not request.include_reasoning:
                    reasoning_content = None
            else:
                reasoning_content = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if (not self.enable_auto_tools or not self.tool_parser) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(role=role, reasoning_content=reasoning_content, content=content)

            # if the request uses tools and specified a tool choice
            elif request.tool_choice and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam:
                tool_call_class = MistralToolCall if isinstance(tokenizer, MistralTokenizer) else ToolCall
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content="",
                    tool_calls=[
                        tool_call_class(
                            function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=content,
                            )
                        )
                    ],
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class = MistralToolCall if isinstance(tokenizer, MistralTokenizer) else ToolCall

                # the fields of FunctionDefinition are a superset of the
                # tool call outputs and can be used for parsing
                assert content is not None
                tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(content)
                tool_call_ids = []
                for tool_call in tool_calls:
                    tool_call_ids.append(
                        make_tool_call_id(
                            id_type=self.tool_call_id_type,
                            func_name=tool_call.name,
                            idx=history_tool_call_cnt,
                        )
                    )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        tool_call_class(
                            id=tool_call_ids[i],
                            function=FunctionCall(
                                name=tool_call.name,
                                arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                            ),
                        )
                        for i, tool_call in enumerate(tool_calls)
                    ],
                    reasoning_content=reasoning_content,
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, reasoning_content=reasoning_content, content=content)

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and self.tool_parser
            ):
                try:
                    tool_parser = self.tool_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in tool parser creation.")
                    return self.create_error_response(str(e))

                tool_call_info = tool_parser.extract_tool_calls(content if content is not None else "", request=request)
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=tool_call_info.content,
                        tool_calls=tool_call_info.tool_calls,
                    )

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    ret_content = content

                    # try to use content return from tool parser first,
                    # tool parser may do some modify for the content.
                    if tool_call_info.content and len(tool_call_info.content) > 0:
                        ret_content = tool_call_info.content
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=ret_content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine if tools should be extracted. "
                    "Returning a standard chat completion."
                )
                message = ChatMessage(role=role, reasoning_content=reasoning_content, content=content)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason=(
                    "tool_calls" if auto_tools_called else output.finish_reason if output.finish_reason else "stop"
                ),
                stop_reason=output.stop_reason,
                token_ids=(as_list(output.token_ids) if request.return_token_ids else None),
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if conversation and "content" in conversation[-1] and conversation[-1].get("role") == role:
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)

        prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
        prompt_token_ids = final_res.prompt_token_ids if request.return_token_ids else None
        kv_transfer_params = final_res.kv_transfer_params

        return choices, usage, prompt_logprobs, prompt_token_ids, kv_transfer_params

    def _create_audio_choice(self, omni_outputs: OmniRequestOutput, role: str):
        choices: list[ChatCompletionResponseChoice] = []
        final_res = omni_outputs.request_output
        audio_tensor = final_res.multimodal_output["audio"].float().detach().cpu().numpy()

        # Ensure audio is 1D (flatten if needed)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.flatten()

        audio_obj = CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=24000,
            response_format="wav",
            speed=1.0,
            stream_format="audio",
            base64_encode=True,
        )

        audio_response: AudioResponse = self.create_audio(audio_obj)
        audio_base64 = audio_response.audio_data

        # Generate unique ID for the audio
        audio_id = f"audio-{uuid.uuid4().hex[:16]}"

        # Set expiration time (e.g., 24 hours from now) as Unix timestamp
        expires_at = int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp())

        # Create OpenAIChatCompletionAudio object with all required fields
        audio_obj = OpenAIChatCompletionAudio(
            id=audio_id,
            data=audio_base64,
            expires_at=expires_at,
            transcript="",  # Empty transcript if not available
        )

        for output in final_res.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, audio=audio_obj),
                logprobs=None,
                finish_reason="stop",
                stop_reason=None,
            )
            choices.append(choice_data)
        return choices

    def _create_image_choice(self, omni_outputs: OmniRequestOutput, role: str):
        """Create chat completion response choices for image output.

        Converts image tensor or PIL Image output from diffusion models
        into base64-encoded image data for API response.

        Args:
            omni_outputs: Output containing image data from diffusion stage
            role: The role for the response message (e.g., "assistant")

        Returns:
            List of ChatCompletionResponseChoice with image content
        """
        from PIL import Image

        choices: list[ChatCompletionResponseChoice] = []
        final_res = omni_outputs.request_output

        # Handle different image output formats
        images = []

        # First check omni_outputs.images directly (for diffusion mode via from_diffusion)
        if omni_outputs.images:
            images = omni_outputs.images
        # Fall back to request_output for pipeline mode
        elif final_res is not None:
            if hasattr(final_res, "multimodal_output") and final_res.multimodal_output:
                image_data = final_res.multimodal_output.get("image")
                if image_data is not None:
                    if isinstance(image_data, Image.Image):
                        images.append(image_data)
                    elif hasattr(image_data, "cpu"):  # Tensor
                        import numpy as np

                        # Convert tensor to PIL Image
                        img_array = image_data.float().detach().cpu().numpy()
                        # Handle different tensor formats (CHW -> HWC)
                        if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]:
                            img_array = np.transpose(img_array, (1, 2, 0))
                        # Normalize to 0-255
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)
                        # Handle grayscale
                        if img_array.ndim == 2:
                            images.append(Image.fromarray(img_array, mode="L"))
                        elif img_array.shape[-1] == 1:
                            images.append(Image.fromarray(img_array.squeeze(-1), mode="L"))
                        elif img_array.shape[-1] == 3:
                            images.append(Image.fromarray(img_array, mode="RGB"))
                        elif img_array.shape[-1] == 4:
                            images.append(Image.fromarray(img_array, mode="RGBA"))
            elif hasattr(final_res, "images") and final_res.images:
                images = final_res.images

        # Convert images to base64
        image_contents = []
        for img in images:
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                    },
                }
            )

        # Create message content
        if len(image_contents) == 1:
            content = image_contents
        elif len(image_contents) > 1:
            content = image_contents
        else:
            content = [{"type": "text", "text": "Image generation completed but no images were produced."}]

        # Create response choice
        # Use model_construct to bypass validation for multimodal content
        # (ChatMessage.content only accepts str, but we need list for images)
        # Then use object.__setattr__ to directly set the field, bypassing Pydantic's type checking
        import warnings as warnings_module

        with warnings_module.catch_warnings():
            warnings_module.filterwarnings("ignore", category=UserWarning, module="pydantic")
            message = ChatMessage.model_construct(role=role)
            object.__setattr__(message, "content", content)
            # Mark content as set in fields_set to ensure proper serialization
            if hasattr(message, "__pydantic_fields_set__"):
                message.__pydantic_fields_set__.add("content")
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )
        choices.append(choice_data)

        return choices

    # ==================== Diffusion Mode Methods ====================

    async def _create_diffusion_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Generate images via chat completion interface for diffusion models.

        Args:
            request: Chat completion request
            raw_request: Raw FastAPI request object

        Returns:
            ChatCompletionResponse with generated images or ErrorResponse
        """
        try:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
            created_time = int(time.time())

            # Convert messages to dict format
            messages = []
            for msg in request.messages:
                if hasattr(msg, "model_dump"):
                    messages.append(msg.model_dump())
                elif isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append({"role": getattr(msg, "role", "user"), "content": getattr(msg, "content", "")})

            # Extract prompt and images from messages
            prompt, reference_images = self._extract_diffusion_prompt_and_images(messages)

            if not prompt:
                return self._create_error_response("No text prompt found in messages")

            # Extract generation parameters from extra_body (preferred)
            # Reference: text_to_image.py and text_to_video.py for supported parameters
            extra_body = getattr(request, "extra_body", None) or {}

            # Parse size if provided (supports "1024x1024" format)
            height = extra_body.get("height")
            width = extra_body.get("width")
            if "size" in extra_body:
                try:
                    size_str = extra_body["size"]
                    if isinstance(size_str, str) and "x" in size_str.lower():
                        w, h = size_str.lower().split("x")
                        width, height = int(w), int(h)
                except ValueError:
                    logger.warning("Invalid size format: %s", extra_body.get("size"))

            # Get request parameters from extra_body
            # Text-to-image parameters (ref: text_to_image.py)
            num_inference_steps = extra_body.get("num_inference_steps", 50)
            guidance_scale = extra_body.get("guidance_scale")
            true_cfg_scale = extra_body.get("true_cfg_scale")  # Qwen-Image specific
            seed = extra_body.get("seed")
            negative_prompt = extra_body.get("negative_prompt")
            num_outputs_per_prompt = extra_body.get("num_outputs_per_prompt", 1)

            # Text-to-video parameters (ref: text_to_video.py)
            num_frames = extra_body.get("num_frames")
            guidance_scale_2 = extra_body.get("guidance_scale_2")  # For video high-noise CFG

            logger.info(
                "Diffusion chat request %s: prompt=%r, ref_images=%d, params=%s",
                request_id,
                prompt[:50] + "..." if len(prompt) > 50 else prompt,
                len(reference_images),
                {k: v for k, v in extra_body.items() if v is not None},
            )

            # Decode reference images if provided
            pil_images: list[Image.Image] = []
            for img_b64 in reference_images:
                try:
                    img_bytes = base64.b64decode(img_b64)
                    pil_images.append(Image.open(BytesIO(img_bytes)))
                except Exception as e:
                    logger.warning("Failed to decode reference image: %s", e)

            # Build generation kwargs
            gen_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "request_id": request_id,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "negative_prompt": negative_prompt,
                "num_outputs_per_prompt": num_outputs_per_prompt,
                "seed": seed,
            }

            if guidance_scale is not None:
                gen_kwargs["guidance_scale"] = guidance_scale

            # Add Qwen-Image specific parameter
            if true_cfg_scale is not None:
                gen_kwargs["true_cfg_scale"] = true_cfg_scale

            # Add video generation parameters if set
            if num_frames is not None:
                gen_kwargs["num_frames"] = num_frames
            if guidance_scale_2 is not None:
                gen_kwargs["guidance_scale_2"] = guidance_scale_2

            # Add reference image if provided
            if pil_images:
                if len(pil_images) == 1:
                    gen_kwargs["pil_image"] = pil_images[0]
                else:
                    od_config = getattr(self._diffusion_engine, "od_config", None)
                    supports_multimodal_inputs = getattr(od_config, "supports_multimodal_inputs", False)
                    if od_config is None:
                        # TODO: entry is asyncOmni. We hack the od config here.
                        supports_multimodal_inputs = True
                    if supports_multimodal_inputs:
                        gen_kwargs["pil_image"] = pil_images
                    else:
                        return self._create_error_response(
                            "Multiple input images are not supported by the current diffusion model. "
                            "For multi-image editing, start the server with Qwen-Image-Edit-2509 "
                            "and send multiple images in the user message content.",
                            status_code=400,
                        )

            # Generate image
            # Handle both AsyncOmniDiffusion (returns OmniRequestOutput) and AsyncOmni (returns AsyncGenerator)
            if hasattr(self._diffusion_engine, "stage_list"):
                # AsyncOmni: iterate through async generator to get final output
                result = None
                async for output in self._diffusion_engine.generate(
                    prompt=gen_kwargs["prompt"],
                    request_id=gen_kwargs.get("request_id"),
                    sampling_params_list=[gen_kwargs],  # Pass as single-stage params
                ):
                    result = output
                if result is None:
                    return self._create_error_response("No output generated from AsyncOmni")
            else:
                # AsyncOmniDiffusion: direct call
                result = await self._diffusion_engine.generate(**gen_kwargs)
            # Extract images from result
            # Handle nested OmniRequestOutput structure where images might be in request_output
            images: list[Image.Image] = []
            if result.request_output["images"]:
                images = result.request_output["images"]

            # Convert images to base64 content
            image_contents: list[dict[str, Any]] = []
            for img in images:
                with BytesIO() as buffer:
                    img.save(buffer, format="PNG")
                    img_bytes = buffer.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                image_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}",
                        },
                    }
                )

            # Build response
            if not image_contents:
                content = "Image generation completed but no images were produced."
            else:
                content = image_contents

            # Use model_construct to bypass validation for multimodal content
            # (ChatMessage.content only accepts str, but we need list for images)
            # Then use object.__setattr__ to directly set the field, bypassing Pydantic's type checking
            import warnings as warnings_module

            with warnings_module.catch_warnings():
                warnings_module.filterwarnings("ignore", category=UserWarning, module="pydantic")
                message = ChatMessage.model_construct(role="assistant")
                object.__setattr__(message, "content", content)
                # Mark content as set in fields_set to ensure proper serialization
                if hasattr(message, "__pydantic_fields_set__"):
                    message.__pydantic_fields_set__.add("content")
            choice = ChatCompletionResponseChoice.model_construct(
                index=0,
                message=message,
                finish_reason="stop",
                logprobs=None,
                stop_reason=None,
            )

            response = ChatCompletionResponse(
                id=request_id,
                created=created_time,
                model=self._diffusion_model_name,
                choices=[choice],
                usage=UsageInfo(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=1,
                    total_tokens=len(prompt.split()) + 1,
                ),
            )

            logger.info(
                "Diffusion chat completed for request %s: %d images",
                request_id,
                len(images),
            )

            return response

        except Exception as e:
            logger.exception("Diffusion chat completion failed: %s", e)
            return self._create_error_response(
                f"Image generation failed: {str(e)}",
                status_code=500,
            )

    def _extract_diffusion_prompt_and_images(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[str]]:
        """Extract text prompt and base64 images from chat messages.

        Args:
            messages: List of chat messages

        Returns:
            Tuple of (prompt_text, list_of_base64_images)
        """
        prompt_parts: list[str] = []
        images: list[str] = []

        for message in messages:
            role = message.get("role", "")
            if role != "user":
                continue

            content = message.get("content", "")

            # String content
            if isinstance(content, str):
                prompt_parts.append(content)
                continue

            # List of content items
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        prompt_parts.append(item)
                    elif isinstance(item, dict):
                        # Handle {"type": "text", "text": "..."} format
                        if item.get("type") == "text":
                            prompt_parts.append(item.get("text", ""))
                        # Handle {"text": "..."} format
                        elif "text" in item and "type" not in item:
                            prompt_parts.append(item["text"])
                        # Handle {"type": "image_url", "image_url": {"url": "..."}}
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:image"):
                                try:
                                    _, b64_data = url.split(",", 1)
                                    images.append(b64_data)
                                except ValueError:
                                    logger.warning("Invalid data URL format")
                        # Handle {"image": "base64..."} format
                        elif "image" in item:
                            images.append(item["image"])

        prompt = " ".join(prompt_parts).strip()
        return prompt, images

    def _create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> ErrorResponse:
        """Create an error response following OpenAI error format."""
        return ErrorResponse(
            message=message,
            type=err_type,
            code=status_code,
        )
