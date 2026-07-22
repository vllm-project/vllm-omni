import asyncio
import base64
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Final, cast

import jinja2
import torch
from fastapi import Request
from openai.types.chat.chat_completion_audio import ChatCompletionAudio as OpenAIChatCompletionAudio
from PIL import Image
from pydantic import TypeAdapter
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
    get_history_tool_calls_cnt,
    make_tool_call_id,
)

from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.protocol.chat_completion import OmniChatCompletionResponse
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.metrics import definitions as _metric_defs
from vllm_omni.metrics.modality import (
    observe_audio_first_packet,
    observe_audio_streaming_finalize,
)
from vllm_omni.model_extras import get_extra_body_params, get_extra_output_params

try:
    import soundfile
except ImportError:
    soundfile = None


from vllm.entrypoints.generate.base.serving import clamp_prompt_logprobs
from vllm.entrypoints.launcher import terminate_if_errored
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorInfo,
    ErrorResponse,
    FunctionCall,
    FunctionDefinition,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_streamable_parser_for_assistant,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.typing import ChatLikeRequest
from vllm.entrypoints.serve.utils.api_utils import should_include_usage
from vllm.entrypoints.serve.utils.tool_calls_utils import maybe_filter_parallel_tool_calls
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.multimodal.media.connector import MediaConnector
from vllm.outputs import RequestOutput
from vllm.reasoning import ReasoningParser
from vllm.renderers import BaseRenderer, merge_kwargs
from vllm.renderers.inputs import TokPrompt
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers import TokenizerLike as AnyTokenizer
from vllm.tokenizers.mistral import (
    MistralTokenizer,
    maybe_serialize_tool_calls,
    truncate_tool_call_ids,
    validate_request_params,
)
from vllm.tool_parsers import ToolParser
from vllm.tool_parsers.mistral_tool_parser import MistralToolCall
from vllm.tool_parsers.streaming import extract_required_tool_call_streaming
from vllm.utils.collection_utils import as_list
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.image_api_utils import encode_image_base64_with_compression, validate_layered_layers
from vllm_omni.entrypoints.openai.protocol import OmniChatCompletionStreamResponse
from vllm_omni.entrypoints.openai.protocol.audio import (
    DEFAULT_AUDIO_FORMAT,
    SUPPORTED_CHAT_AUDIO_FORMATS,
    AudioResponse,
    CreateAudio,
)
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageEditARDeltaChunk,
    ImageEditImageChunk,
    ImageEditStreamError,
)
from vllm_omni.entrypoints.openai.stage_params import (
    build_stage_sampling_params_list,
    clone_sampling_params,
    get_default_sampling_params_list,
)
from vllm_omni.entrypoints.openai.utils import (
    get_stage_type,
    get_supported_speakers_from_hf_config,
    is_single_stage_diffusion,
    parse_lora_request,
    resolve_diffusion_od_config,
    validate_requested_speaker,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.output_metadata import DiffusionMetadataMapping, DiffusionMetadataValue
from vllm_omni.utils.audio import audio_chunk_pcm_bytes, audio_chunk_sample_rate

logger = init_logger(__name__)


async def _identity_async(value: Any) -> Any:
    return value


class OmniOpenAIServingChat(OpenAIServingChat, AudioMixin):
    """OpenAI-compatible chat serving for both LLM and Diffusion models.

    This class extends OpenAIServingChat to support:
    - Standard LLM chat completions
    - Diffusion model image generation via chat interface

    For diffusion mode, use the `for_diffusion` class method to create an instance.
    """

    # Diffusion mode attributes
    _diffusion_mode: bool = False
    _diffusion_engine: AsyncOmni | None = None
    _diffusion_model_name: str = ""
    _supported_speakers: set[str] | None = None
    _diffusion_extra_body_params: frozenset[str] | None = None
    _diffusion_extra_output_params: frozenset[str] | None = None

    # Harmony flag (always False for vllm-omni models)
    use_harmony: bool = False

    @property
    def tool_call_id_type(self) -> str:
        """Return the tool call ID type, delegating to model config.

        Upstream vLLM removed the stored ``tool_call_id_type`` attribute
        from ``OpenAIServingChat`` after the ParserManager refactor; the
        field is now resolved on demand via ``get_tool_call_id_type``.
        """
        try:
            from vllm.entrypoints.chat_utils import get_tool_call_id_type

            return get_tool_call_id_type(self.model_config)
        except Exception:
            return "random"

    def _should_stream_with_auto_tool_parsing(self, request: ChatCompletionRequest) -> bool:
        """Check if streamed tokens should go through the tool-call parser.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.

        This method existed in upstream vLLM commit 91df0fad4 (OpenAIServingChat)
        but was removed in the Harmony refactoring (PR #45171, #45104).  Omni's
        independently-maintained ``chat_completion_stream_generator`` still calls
        it, so we keep a local copy.
        """
        # parser_cls may be None (no tool parser configured); guard accordingly
        tp_cls = self.parser_cls.tool_parser_cls if self.parser_cls is not None else None
        return request.tools and tp_cls is not None and self.enable_auto_tools and request.tool_choice in ["auto", None]

    @classmethod
    def for_diffusion(
        cls,
        diffusion_engine: AsyncOmni,
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
        instance._diffusion_extra_body_params = None
        instance._diffusion_extra_output_params = None
        instance.engine_client = None
        instance.has_kv_connector = False
        # Extra body/output params are resolved lazily on first use; see
        # _get_diffusion_extra_body_params / _get_diffusion_extra_output_params.
        return instance

    def _get_diffusion_extra_body_params(self) -> frozenset[str]:
        """Return model-specific extra_body params from the extra registry."""
        if self._diffusion_extra_body_params is not None:
            return self._diffusion_extra_body_params

        params: frozenset[str] = frozenset()
        try:
            od_config = resolve_diffusion_od_config(self.engine_client, self._diffusion_engine)
            if od_config is not None and getattr(od_config, "model_class_name", None):
                params = get_extra_body_params(od_config.model_class_name)
        except Exception as e:
            logger.warning("Failed to read model extra_body params: %s", e)

        self._diffusion_extra_body_params = params
        return params

    def _get_diffusion_extra_output_params(
        self,
        output: object,
    ) -> dict[str, DiffusionMetadataValue] | None:
        """Pick model-specific extra output keys from diffusion metadata."""
        metadata: DiffusionMetadataMapping = {}
        mm_output = getattr(output, "multimodal_output", None)
        if isinstance(mm_output, dict):
            raw_metadata = mm_output.get("metadata")
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata

        if self._diffusion_extra_output_params is None:
            params: frozenset[str] = frozenset()
            try:
                od_config = resolve_diffusion_od_config(self.engine_client, self._diffusion_engine)
                if od_config is not None and getattr(od_config, "model_class_name", None):
                    params = get_extra_output_params(od_config.model_class_name)
            except Exception as e:
                logger.warning("Failed to read model extra output params: %s", e)
            self._diffusion_extra_output_params = params

        if not self._diffusion_extra_output_params:
            return None
        flat_metadata: dict[str, DiffusionMetadataValue] = {}
        for section in metadata.values():
            if isinstance(section, dict):
                flat_metadata.update(section)
        out = {k: flat_metadata[k] for k in self._diffusion_extra_output_params if k in flat_metadata}
        return out or None

    @staticmethod
    def _get_diffusion_text_output(output: object) -> str:
        mm_output = getattr(output, "multimodal_output", None)
        if isinstance(mm_output, dict) and mm_output.get("text") is not None:
            return str(mm_output["text"])
        return ""

    def _get_supported_speakers(self) -> set[str]:
        """Load supported speakers from model config (cached)."""
        if self._supported_speakers is not None:
            return self._supported_speakers
        try:
            self._supported_speakers = get_supported_speakers_from_hf_config(self.model_config.hf_config)
            return self._supported_speakers
        except Exception as e:
            logger.warning("Could not load speakers from model config: %s", e)
        self._supported_speakers = set()
        return self._supported_speakers

    @staticmethod
    def _truthy_extra_body_flag(request: Any, key: str) -> bool:
        if isinstance(request, dict):
            extra_body = request
            model_extra = {}
        else:
            extra_body = getattr(request, "extra_body", None) or {}
            model_extra = getattr(request, "model_extra", None) or {}
        value = extra_body.get(key, model_extra.get(key))
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @classmethod
    def _filter_stage_metrics_detail(cls, metrics: dict[str, Any] | None, request: Any) -> dict[str, Any] | None:
        if not metrics:
            return metrics
        if cls._truthy_extra_body_flag(request, "return_stage_metrics"):
            return metrics
        return None

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
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_chat_completion(request, raw_request), request, raw_request
        )

    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # Handle diffusion mode
        if self._diffusion_mode:
            return await self._create_diffusion_chat_completion(request, raw_request)

        request_timestamp = time.time()
        if raw_request is not None:
            request_timestamp = float(getattr(raw_request.state, "request_timestamp", request_timestamp))

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

            renderer = self.renderer
            tokenizer = renderer.get_tokenizer()
            if tokenizer is None:
                tokenizer = await self.engine_client.get_tokenizer()

            reasoning_parser: ReasoningParser | None = None
            if self.parser_cls is not None and self.parser_cls.reasoning_parser_cls is not None:
                chat_template_kwargs = self._effective_chat_template_kwargs(request)
                reasoning_parser = self.parser_cls.reasoning_parser_cls(
                    tokenizer,
                    chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
                )

            tool_parser = self.parser_cls.tool_parser_cls if self.parser_cls is not None else None

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)
                truncate_tool_call_ids(request)
                validate_request_params(request)

            # Check if tool parsing is unavailable (common condition)
            tool_parsing_unavailable = (
                tool_parser is None and not isinstance(tokenizer, MistralTokenizer) and not self.use_harmony
            )

            # Validate tool_choice when tool parsing is required but unavailable
            if tool_parsing_unavailable and request.tool_choice not in (
                None,
                "none",
            ):
                if request.tool_choice == "auto" and not self.enable_auto_tools:
                    # for hf tokenizers, "auto" tools requires
                    # --enable-auto-tool-choice and --tool-call-parser
                    return self.create_error_response(
                        '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
                    )
                elif request.tool_choice != "auto":
                    # "required" or named tool requires tool parser
                    return self.create_error_response(
                        f'tool_choice="{request.tool_choice}" requires --tool-call-parser to be set'
                    )

            if request.tools is None or (request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none):
                tool_dicts = None
            else:
                tool_dicts = [tool.model_dump() for tool in request.tools]

            if not self.use_harmony:
                error_check_ret = self.online_renderer.validate_chat_template(
                    request_chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs,
                    trust_request_chat_template=self.trust_request_chat_template,
                )
                if error_check_ret is not None:
                    return error_check_ret

                # Effective kwargs fold request.chat_template_kwargs, reasoning_effort,
                # and server defaults — mirrors OpenAIServingChat._effective_chat_template_kwargs.
                merged_template_kwargs = self._effective_chat_template_kwargs(request)
                conversation, engine_prompts = await self._preprocess_chat(
                    request,
                    request.messages,
                    default_template=request.chat_template or self.chat_template,
                    default_template_content_format=self.chat_template_content_format,
                    default_template_kwargs=merged_template_kwargs,
                    tool_dicts=tool_dicts,
                    tool_parser=tool_parser,
                    # OMNI: Additional parameters
                    renderer=renderer,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    documents=getattr(request, "documents", None),
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                should_include_tools = tool_dicts is not None
                conversation, engine_prompts = self.online_renderer._make_request_with_harmony(
                    request, should_include_tools
                )

        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            message = str(e)
            if e.__cause__ is not None:
                message = f"{message} {e.__cause__}"
            return self.create_error_response(message)

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Some models will return a list like ["text", None, "audio"], better
        # to strip None in the list
        engine_output_modalities = [x for x in self.engine_client.output_modalities if x is not None]
        output_modalities = getattr(request, "modalities", engine_output_modalities)
        request.modalities = output_modalities if output_modalities is not None else engine_output_modalities

        if request.modalities and "audio" in request.modalities:
            audio_format_check = self._resolve_audio_format(request)
            if isinstance(audio_format_check, ErrorResponse):
                return audio_format_check

        num_inference_steps = None
        extra_body: dict[str, Any] = {}
        # Omni multistage image generation: Stage-0 (AR) should receive a clean
        # text prompt (and optional conditioning image/size) so the model's own
        # processor can construct the correct inputs.
        # If we pass pre-tokenized chat-template ids, GLM-Image can become
        # effectively unconditioned and produce nonsense images.
        if request.modalities and ("image" in request.modalities):
            try:
                extracted_prompt, reference_images = self._extract_diffusion_prompt_and_images_from_messages(
                    request.messages
                )
                if not extracted_prompt:
                    return self.create_error_response("No text prompt found in messages")

                # [NOTE] When sending request via openai client Python library,
                #   `extra_body` is flattented and merged into the payload's root.
                #   These extra fields are accessible via `model_extra` property (from Pydantic base class).
                #   When sending raw request with curl, no flattening happens. Directly read the `extra_body` dict.
                extra_body = getattr(request, "extra_body", None) or request.model_extra or {}

                height, width = self._resolve_height_width_from_extra_body(extra_body)

                num_inference_steps = extra_body.get("num_inference_steps")
                if num_inference_steps is not None:
                    try:
                        num_inference_steps = int(num_inference_steps)
                    except Exception:
                        num_inference_steps = None

                negative_prompt = extra_body.get("negative_prompt")
                engine_prompt_image: dict[str, Any] | None = None
                if reference_images:
                    # Best-effort decode first reference image for i2i.
                    try:
                        img_bytes = base64.b64decode(reference_images[0])
                        img = Image.open(BytesIO(img_bytes))
                        engine_prompt_image = {"img2img": img}
                    except Exception:
                        engine_prompt_image = None

                # Override the prompts produced by chat-template preprocessing.
                is_img2img = engine_prompt_image is not None
                tprompt: OmniTextPrompt = {"prompt": extracted_prompt}
                if is_img2img:
                    tprompt["modalities"] = ["img2img"]
                else:
                    tprompt["modalities"] = ["image"]
                if negative_prompt is not None:
                    tprompt["negative_prompt"] = negative_prompt
                # Always attach mm_processor_kwargs (possibly empty) so
                # OmniInputPreprocessor._process_text routes through the
                # multimodal processor path. Without it, the preprocessor
                # falls back to plain _tokenize_prompt and AR-based image-gen
                # models like GLM-Image never see their image-generation
                # scaffold.
                mm_processor_kwargs: dict[str, Any] = {}
                if height is not None:
                    mm_processor_kwargs["target_h"] = height
                if width is not None:
                    mm_processor_kwargs["target_w"] = width
                # Pass output modalities so model-specific MM processors can
                # detect image-generation requests and apply their own prompt
                # rewrites (e.g. query-token expansion, placeholder injection).
                mm_processor_kwargs["modalities"] = ["img2img"] if is_img2img else ["image"]
                tprompt["mm_processor_kwargs"] = mm_processor_kwargs
                if engine_prompt_image is not None:
                    tprompt["multi_modal_data"] = engine_prompt_image
                    # Provide multi_modal_uuids so that newer vLLM versions
                    # can validate multi_modal_data / multi_modal_uuids
                    # consistency.  After the multimodal processor consumes
                    # the image data, the uuids remain as a stable reference.
                    tprompt["multi_modal_uuids"] = {
                        k: [f"{request_id}-{k}-{i}" for i in range(len(v))]
                        if isinstance(v, list)
                        else [f"{request_id}-{k}-0"]
                        for k, v in engine_prompt_image.items()
                    }

                engine_prompts = [tprompt]
                # Store height/width for applying to diffusion stage sampling params later
                _image_gen_height = height
                _image_gen_width = width
            except Exception as e:
                logger.warning("Failed to build image-generation prompt for omni multistage: %s", e)
                _image_gen_height = None
                _image_gen_width = None
        elif request.modalities and ("text" in request.modalities) and is_single_stage_diffusion(self.engine_client):
            # Single-stage diffusion text output (img2text / text2text).
            # Build a diffusion-style prompt with modalities=["text"] so the
            # pipeline routes to its text generation path.
            try:
                extracted_prompt, reference_images = self._extract_diffusion_prompt_and_images_from_messages(
                    request.messages
                )
                if not extracted_prompt:
                    return self.create_error_response("No text prompt found in messages")

                tprompt: OmniTextPrompt = {"prompt": extracted_prompt}
                tprompt["modalities"] = ["text"]

                if reference_images:
                    try:
                        img_bytes = base64.b64decode(reference_images[0])
                        img = Image.open(BytesIO(img_bytes))
                        tprompt["multi_modal_data"] = {"image": img}
                        tprompt["multi_modal_uuids"] = {"image": [f"{request_id}-image-0"]}
                    except Exception:
                        pass

                engine_prompts = [tprompt]
            except Exception as e:
                logger.warning("Failed to build text-output prompt for single-stage diffusion: %s", e)
            _image_gen_height = None
            _image_gen_width = None
        else:
            _image_gen_height = None
            _image_gen_width = None

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                if hasattr(request, "sampling_params_list"):
                    sampling_params_list = self._to_sampling_params_list(request.sampling_params_list)
                else:
                    # Use standard OpenAI API parameters for comprehension stage
                    sampling_params_list = self._build_sampling_params_list_from_request(request)

                # If this is a streaming (output) request, coerce cumulative outputs
                # to delta to ensure emitted outputs are correctly drained. Otherwise
                # convert cumulative to Final Only to ensure the output is correct.
                sampling_params_list = coerce_param_message_types(sampling_params_list, request.stream)

                # Apply user-specified overrides to diffusion stage(s) for image generation
                for idx, sp in enumerate(sampling_params_list):
                    if hasattr(sp, "height") and _image_gen_height is not None:
                        sp.height = _image_gen_height
                    if hasattr(sp, "width") and _image_gen_width is not None:
                        sp.width = _image_gen_width
                    if hasattr(sp, "num_inference_steps") and num_inference_steps is not None:
                        sp.num_inference_steps = num_inference_steps
                    apply_declared_extra_args(sp, self._get_diffusion_extra_body_params(), extra_body)

                self._log_inputs(
                    request_id,
                    engine_prompt,
                    params_list=sampling_params_list,
                    lora_request=lora_request,
                )

                generator = self.engine_client.generate(
                    prompt=engine_prompt,
                    request_id=request_id,
                    sampling_params_list=sampling_params_list,
                    output_modalities=output_modalities,
                    arrival_time=request_timestamp,
                )

                generators.append(generator)
        except ValueError as e:
            return self.create_error_response(e)

        assert len(generators) == 1
        (result_generator,) = generators

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
                raw_request=raw_request,
            )

        try:
            return await self.chat_completion_full_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )
        except ValueError as e:
            return self.create_error_response(e)

    async def _preprocess_chat(
        self,
        request: ChatLikeRequest | ResponsesRequest,
        messages: list[ChatCompletionMessageParam],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None = None,
        tool_dicts: list[dict[str, Any]] | None = None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
        # OMNI: Additional parameters for backward compatibility
        renderer: BaseRenderer | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        documents: list[dict[str, str]] | None = None,
        add_special_tokens: bool = False,
    ) -> tuple[list[ConversationMessage], list[TokPrompt]]:
        if renderer is None:
            renderer = self.renderer

        # Keep OMNI compatibility args wired while delegating rendering
        # to the upstream async renderer pipeline.
        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                add_special_tokens=add_special_tokens,
                tokenize=isinstance(renderer.tokenizer, MistralTokenizer),
            ),
        )

        tok_params = request.build_tok_params(self.model_config)
        mm_config = self.model_config.multimodal_config
        chat_params = request.build_chat_params(
            default_template,
            default_template_content_format,
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
        )

        deferred_multi_modal_data: dict[str, Any] | None = None
        if self._needs_multistage_multimodal_split():
            messages, deferred_multi_modal_data = await self._prepare_multistage_multimodal_inputs(
                messages,
                request,
            )

        (conversation,), (engine_prompt,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v for k in ("mm_processor_kwargs", "cache_salt") if (v := getattr(request, k, None)) is not None
            },
        )

        tokenizer = renderer.get_tokenizer()

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

        # Preserve a clean text prompt for downstream stages (e.g., GLM-Image diffusion).
        # For image generation, we want the raw user caption instead of a rendered template.
        # But for multimodal comprehension (img2text), we MUST keep the rendered prompt
        # containing image tokens.
        req_modalities = getattr(request, "modalities", [])
        if req_modalities and ("image" in req_modalities):
            extracted_prompt, _ = self._extract_diffusion_prompt_and_images_from_messages(messages)
            if extracted_prompt:
                engine_prompt["prompt"] = extracted_prompt

        mm_processor_kwargs = getattr(request, "mm_processor_kwargs", None)
        if mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = mm_processor_kwargs

        if hasattr(request, "cache_salt") and request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        additional_information = getattr(request, "additional_information", None)
        if isinstance(additional_information, dict):
            prompt_additional_information = self._ensure_prompt_additional_information(engine_prompt)
            prompt_additional_information.update(additional_information)

        if deferred_multi_modal_data:
            prompt_additional_information = self._ensure_prompt_additional_information(engine_prompt)
            prompt_additional_information["deferred_multi_modal_data"] = deferred_multi_modal_data

        speaker = getattr(request, "voice", None) or getattr(request, "speaker", None)
        normalized = validate_requested_speaker(speaker, self._get_supported_speakers())
        if normalized is not None:
            prompt_additional_information = self._ensure_prompt_additional_information(engine_prompt)
            prompt_additional_information["speaker"] = [normalized]

        language = getattr(request, "language", None)
        if language is not None and isinstance(language, str) and language.strip():
            prompt_additional_information = self._ensure_prompt_additional_information(engine_prompt)
            prompt_additional_information["language"] = [language.strip()]

        # Style instruction — used by Ming-flash-omni instruct TTS path
        # (ming_task="instruct").  For the omni speech path the thinker2talker
        # bridge drops this field to match upstream omni_audio_generation
        # which hardcodes instruction=None.
        instructions = getattr(request, "instructions", None)
        if instructions is not None and isinstance(instructions, str) and instructions.strip():
            prompt_additional_information = self._ensure_prompt_additional_information(engine_prompt)
            prompt_additional_information["instruction"] = instructions.strip()

        return conversation, [engine_prompt]

    @staticmethod
    def _ensure_prompt_additional_information(engine_prompt: dict[str, Any]) -> dict[str, Any]:
        additional_information = engine_prompt.get("additional_information")
        if not isinstance(additional_information, dict):
            additional_information = {}
            engine_prompt["additional_information"] = additional_information
        return additional_information

    def _needs_multistage_multimodal_split(self) -> bool:
        return bool(self._deferred_multimodal_modalities())

    def _deferred_multimodal_modalities(self) -> set[str]:
        stage_configs = list(getattr(self.engine_client, "stage_configs", []) or [])
        if len(stage_configs) < 2:
            return set()

        first_stage_modalities = self._stage_input_modalities(stage_configs[0])
        if not first_stage_modalities:
            return set()

        downstream_modalities: set[str] = set()
        for stage in stage_configs[1:]:
            downstream_modalities.update(self._stage_input_modalities(stage))
        return downstream_modalities - first_stage_modalities

    @staticmethod
    def _stage_input_modalities(stage: Any) -> set[str]:
        engine_args = getattr(stage, "engine_args", None)
        explicit = (
            getattr(stage, "input_modalities", None)
            or getattr(stage, "modalities", None)
            or getattr(engine_args, "input_modalities", None)
            or getattr(engine_args, "modalities", None)
        )
        if explicit:
            return {str(modality) for modality in as_list(explicit)}

        model_stage = str(getattr(engine_args, "model_stage", None) or getattr(stage, "model_stage", "")).lower()
        if model_stage in {"asr", "stt"} or model_stage.endswith("_asr"):
            return {"audio"}
        if any(name in model_stage for name in ("vision", "vl", "aura")):
            return {"image", "video"}
        if any(name in model_stage for name in ("tts", "talker", "code2wav")):
            return set()

        if getattr(stage, "requires_multimodal_data", False):
            return {"audio", "image", "video"}
        return set()

    async def _prepare_multistage_multimodal_inputs(
        self,
        messages: list[ChatCompletionMessageParam],
        request: ChatLikeRequest | ResponsesRequest,
    ) -> tuple[list[ChatCompletionMessageParam], dict[str, Any] | None]:
        """Hide modalities unsupported by stage 0 and stash them for downstream stages."""
        deferred_modalities = self._deferred_multimodal_modalities()
        deferred_parts: dict[str, list[Any]] = {modality: [] for modality in deferred_modalities}
        stripped_messages: list[ChatCompletionMessageParam] = []

        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                stripped_messages.append(message)
                continue

            stripped_content: list[Any] = []
            changed = False
            for part in content:
                modality, payload = self._deferred_multimodal_part(part, deferred_modalities)
                if modality is not None:
                    if payload is not None:
                        deferred_parts.setdefault(modality, []).append(payload)
                    changed = True
                    continue
                stripped_content.append(part)

            if changed:
                stripped_message = dict(message)
                stripped_message["content"] = stripped_content
                stripped_messages.append(cast(ChatCompletionMessageParam, stripped_message))
            else:
                stripped_messages.append(message)

        deferred_parts = {modality: parts for modality, parts in deferred_parts.items() if parts}
        if not deferred_parts:
            return messages, None

        media_connector = MediaConnector(
            media_io_kwargs=getattr(request, "media_io_kwargs", None),
            allowed_local_media_path=getattr(self.model_config, "allowed_local_media_path", "") or "",
            allowed_media_domains=getattr(self.model_config, "allowed_media_domains", None),
        )
        multi_modal_data: dict[str, Any] = {}
        for modality, parts in deferred_parts.items():
            multi_modal_data[modality] = await self._materialize_deferred_multimodal_parts(
                media_connector,
                modality,
                parts,
            )
        return stripped_messages, multi_modal_data

    @staticmethod
    def _deferred_multimodal_part(part: Any, deferred_modalities: set[str]) -> tuple[str | None, Any | None]:
        if not isinstance(part, dict):
            return None, None
        part_type = part.get("type")
        if part_type in {"video_url", "video"} and "video" in deferred_modalities:
            video = part.get("video_url", part.get("video"))
            if isinstance(video, dict):
                video = video.get("url")
            return "video", video
        if part_type in {"image_url", "image_pil", "image"} and "image" in deferred_modalities:
            image = part.get("image_pil", part.get("image_url", part.get("image")))
            if isinstance(image, dict):
                image = image.get("url")
            return "image", image
        if part_type in {"audio_url", "input_audio", "audio"} and "audio" in deferred_modalities:
            audio = part.get("audio_url", part.get("input_audio", part.get("audio")))
            if isinstance(audio, dict):
                audio = audio.get("url") or audio.get("data")
            return "audio", audio
        return None, None

    @staticmethod
    async def _materialize_deferred_multimodal_parts(
        media_connector: MediaConnector,
        modality: str,
        parts: list[Any],
    ) -> list[Any]:
        if modality == "video":
            return list(await asyncio.gather(*(media_connector.fetch_video_async(part) for part in parts)))
        if modality == "image":
            fetch_image = getattr(media_connector, "fetch_image_async", None)
            if fetch_image is None:
                return parts
            return list(
                await asyncio.gather(
                    *(fetch_image(part) if isinstance(part, str) else _identity_async(part) for part in parts)
                )
            )
        if modality == "audio":
            fetch_audio = getattr(media_connector, "fetch_audio_async", None)
            if fetch_audio is None:
                return parts
            return list(
                await asyncio.gather(
                    *(fetch_audio(part) if isinstance(part, str) else _identity_async(part) for part in parts)
                )
            )
        return parts

    async def _inject_audio_from_video_urls(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        """Pre-extract audio from video URLs and inject as audio_url content items.

        When use_audio_in_video=True, the qwen2_5_omni_thinker multimodal
        processor requires that the number of audio items equals the number of
        video items (it subtracts mm_counts["video"] from mm_counts["audio"]).
        The client only sends video_url items; this method adds the matching
        audio_url items on the server side before the renderer processes them.
        """
        import io

        from vllm_omni.entrypoints.chat_utils import extract_audio_from_video_async

        new_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if not isinstance(content, list):
                new_messages.append(msg)
                continue

            video_urls = [
                part.get("video_url", {}).get("url")
                for part in content
                if isinstance(part, dict) and part.get("type") == "video_url" and part.get("video_url", {}).get("url")
            ]

            if not video_urls:
                new_messages.append(msg)
                continue

            audios = await asyncio.gather(*(extract_audio_from_video_async(u) for u in video_urls))

            audio_items: list[dict] = []
            for audio_array, sample_rate in audios:
                buf = io.BytesIO()
                if soundfile is not None:
                    soundfile.write(buf, audio_array, samplerate=int(sample_rate), format="WAV")
                else:
                    import struct

                    import numpy as np

                    audio_np = np.asarray(audio_array, dtype=np.float32)
                    sr = int(sample_rate)
                    num_channels = 1
                    bits_per_sample = 32
                    num_frames = len(audio_np)
                    data_size = num_frames * num_channels * (bits_per_sample // 8)
                    # Write minimal RIFF/WAV header
                    buf.write(b"RIFF")
                    buf.write(struct.pack("<I", 36 + data_size))
                    buf.write(b"WAVE")
                    buf.write(b"fmt ")
                    buf.write(
                        struct.pack(
                            "<IHHIIHH",
                            16,
                            3,
                            num_channels,
                            sr,
                            sr * num_channels * (bits_per_sample // 8),
                            num_channels * (bits_per_sample // 8),
                            bits_per_sample,
                        )
                    )
                    buf.write(b"data")
                    buf.write(struct.pack("<I", data_size))
                    buf.write(audio_np.tobytes())

                audio_b64 = base64.b64encode(buf.getvalue()).decode()
                audio_items.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                    }
                )

            new_content = list(content) + audio_items
            if isinstance(msg, dict):
                new_msg = {**msg, "content": new_content}
            else:
                new_msg = msg.model_copy(update={"content": new_content})
            new_messages.append(new_msg)

        return new_messages

    def _to_sampling_params_list(self, sampling_params_list: list[dict]) -> list[Any]:
        """Convert request dicts to stage-typed sampling params objects.

        For diffusion stages, build ``OmniDiffusionSamplingParams`` so
        downstream ``StageDiffusionClient._sampling_params_to_dict`` (which
        requires a dataclass) works. For LLM stages build ``SamplingParams``.
        If callers provide params for fewer stages than the native pipeline has
        (for example AURA has three semantic models but four engine stages),
        append cloned deploy defaults for the omitted tail stages.
        """
        stage_configs = list(getattr(self.engine_client, "stage_configs", []) or [])
        default_params_list = list(getattr(self.engine_client, "default_sampling_params_list", []) or [])
        final_sampling_params_list: list[Any] = []
        for idx, sampling_params in enumerate(sampling_params_list):
            stage_type = get_stage_type(stage_configs[idx]) if idx < len(stage_configs) else "llm"
            target_cls = OmniDiffusionSamplingParams if stage_type == "diffusion" else SamplingParams
            if isinstance(sampling_params, dict):
                final_sampling_params_list.append(target_cls(**sampling_params))
            elif isinstance(sampling_params, target_cls):
                final_sampling_params_list.append(sampling_params)
            elif isinstance(sampling_params, SamplingParams | OmniDiffusionSamplingParams):
                # Cross-typed (e.g. user passed SamplingParams but this is a
                # diffusion stage) — rebuild via a dict round-trip so we end
                # up with the correct target class.
                as_dict = {
                    f.name: getattr(sampling_params, f.name)
                    for f in (fields(sampling_params) if is_dataclass(sampling_params) else [])
                } or sampling_params.__dict__
                final_sampling_params_list.append(target_cls(**as_dict))
            else:
                raise ValueError(f"Invalid sampling params: {sampling_params}")
        for idx in range(len(final_sampling_params_list), len(stage_configs)):
            if idx < len(default_params_list):
                final_sampling_params_list.append(clone_sampling_params(default_params_list[idx]))
            else:
                final_sampling_params_list.append(SamplingParams())
        return final_sampling_params_list

    def _get_comprehension_stage_index(self) -> int:
        for idx, stage in enumerate(self.engine_client.stage_configs):
            if stage.is_comprehension:
                return idx
        raise ValueError("No comprehension stage (is_comprehension=True) found in stage configs")

    # OpenAI API standard sampling parameters that can be safely overridden.
    # These are the most commonly used parameters with compatible types
    # between ChatCompletionRequest and SamplingParams.
    # Users who need more control can use sampling_params_list in extra_body.
    _OPENAI_SAMPLING_FIELDS: set[str] = {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "min_tokens",
        "seed",
        "ignore_eos",
        "stop",
        "stop_token_ids",
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

        For models needing spatial metadata (e.g. GLM-Image), target_h/w is
        injected into extra_args so the runner can build M-RoPE position grids.
        max_tokens is NOT computed dynamically — it uses the deploy YAML default.

        Args:
            default_params: Default SamplingParams from stage config YAML.
            request: The chat completion request containing user-provided values.

        Returns:
            New SamplingParams with YAML defaults overridden by request values.
        """
        params = default_params.clone()

        # Only apply fields explicitly provided by user, not protocol defaults.
        # Pydantic v2 uses `model_fields_set`; keep v1 fallback for compatibility.
        explicit_fields = getattr(request, "model_fields_set", None)
        if explicit_fields is None:
            explicit_fields = getattr(request, "__fields_set__", set())

        for field_name in self._OPENAI_SAMPLING_FIELDS:
            if field_name not in explicit_fields:
                continue

            value = getattr(request, field_name, None)
            if (value is not None and not isinstance(value, list)) or (isinstance(value, list) and len(value) > 0):
                setattr(params, field_name, value)

        # For GLM-Image: compute max_tokens from height/width with mode-aware
        # budgeting (t2i vs i2i).
        extra_body = getattr(request, "extra_body", {}) or {}
        height, width = self._resolve_height_width_from_extra_body(extra_body)

        if height is not None and width is not None:
            # Keep target size in stage-0 sampling params so runner/model can
            # build deterministic M-RoPE grids for t2i (no MM features).
            extra_args = dict(getattr(params, "extra_args", {}) or {})
            extra_args["target_h"] = int(height)
            extra_args["target_w"] = int(width)
            params.extra_args = extra_args

        return params

    @staticmethod
    def _set_if_supported(obj: Any, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is not None and hasattr(obj, key):
                setattr(obj, key, value)

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: Any,
        output: Any,
    ) -> bool:
        """Check whether the streaming generator should flush unstreamed
        tool-arg tokens at finish-time.

        This method was moved from OpenAIServingChat to the tool-parser layer
        by upstream commit 9affc17a05.  Omni's independently-maintained
        ``chat_completion_stream_generator`` still calls it, so we keep a
        local copy with the pre-9affc17a05 semantics.
        """
        return (
            output.finish_reason is not None
            and self.enable_auto_tools
            and self.parser_cls is not None
            and self.parser_cls.tool_parser_cls is not None
            and delta_message is not None
            and delta_message.tool_calls
        )

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
        inputs: PromptType | TokPrompt,
        params_list: list[SamplingParams] | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return
        components = self._extract_prompt_components(inputs)
        self.request_logger.log_inputs(
            request_id,
            components.text,
            components.token_ids,
            components.embeds,
            params=params_list,
            lora_request=lora_request,
        )

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
        raw_request: Request | None = None,
    ):
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration_dict = {}
        assert hasattr(request, "modalities") and request.modalities is not None, (
            "Streaming request must specify output modalities"
        )
        for modality in request.modalities:
            first_iteration_dict[modality] = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        modality_finished: list[set[str]] = [set() for _ in range(num_choices)]
        modality_seen: list[set[str]] = [set() for _ in range(num_choices)]
        stop_reason_emitted: list[bool] = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        if self.use_harmony:
            harmony_parsers = [get_streamable_parser_for_assistant() for _ in range(num_choices)]
            harmony_tools_streamed = [False] * num_choices
        tools_streamed = [False] * num_choices

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = not tool_choice_function_name and self._should_stream_with_auto_tool_parsing(request)

        all_previous_token_ids: list[list[int]] | None
        function_name_returned = [False] * num_choices
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto or reasoning_parser:
            # These are only required in "auto" tool choice case
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        else:
            all_previous_token_ids = None
        # Prepare the tool parser if it's needed
        try:
            tool_parser_cls = self.parser_cls.tool_parser_cls if self.parser_cls is not None else None
            if tool_choice_auto and tool_parser_cls is not None:
                tool_parsers: list[ToolParser | None] = [
                    tool_parser_cls(tokenizer, request.tools) for _ in range(num_choices)
                ]
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(stream_options, self.enable_force_include_usage)

        last_metrics: dict[str, Any] | None = None
        # Hold a strong reference to the audio request state so the
        # streaming-finalize hook below survives the inner generator's
        # _log_summary_and_cleanup, which pops request_states[request_id]
        # before this outer block runs.
        req_state_audio_ref: Any = None
        try:
            async for omni_res in result_generator:
                final_output_type = omni_res.final_output_type
                res = omni_res.request_output
                if final_output_type not in first_iteration_dict:
                    logger.warning(f"final output type: {final_output_type} is not needed by the request")
                    continue

                # Track which modalities have actually appeared in the stream.
                # This is used to determine when all *produced* modalities have
                # finished, which may be a subset of request.modalities when the
                # engine does not produce every requested modality.
                for output in res.outputs:
                    modality_seen[output.index].add(final_output_type)

                if omni_res.metrics:
                    last_metrics = omni_res.metrics

                # Initialize role before conditional blocks to avoid UnboundLocalError
                # when handling audio/image responses
                role = self.get_chat_request_role(request)

                # Compute prompt_text once at first iteration (upstream #42052)
                prompt_text = getattr(res, "prompt", None) if getattr(request, "return_prompt_text", None) else None

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration_dict[final_output_type] and final_output_type == "text":
                    # NOTE: prompt token IDs / cached tokens are based on the first iteration
                    # of the stage that producing text output for consistency for current
                    # non-stream behaviors.
                    if res.prompt_token_ids is not None:
                        num_prompt_tokens = len(res.prompt_token_ids)
                        if res.encoder_prompt_token_ids is not None:
                            num_prompt_tokens += len(res.encoder_prompt_token_ids)

                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each choice with role
                    # NOTE: num_choices defaults to 1 so this usually executes once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = OmniChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(res.prompt_token_ids if request.return_token_ids else None),
                            prompt_text=prompt_text,
                            modality=final_output_type,
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if conversation and "content" in conversation[-1] and conversation[-1].get("role") == role:
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = OmniChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                    modality=final_output_type,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration_dict[final_output_type] = False

                if final_output_type == "text":
                    for output in res.outputs:
                        i = output.index
                        tool_parser = tool_parsers[i]

                        if finish_reason_sent[i]:
                            continue

                        if request.logprobs and request.top_logprobs is not None:
                            assert output.logprobs is not None, "Did not output logprobs"
                            logprobs = self._create_chat_logprobs(
                                token_ids=output.token_ids,
                                top_logprobs=output.logprobs,
                                tokenizer=tokenizer,
                                num_output_top_logprobs=request.top_logprobs,
                                return_as_token_id=request.return_tokens_as_token_ids,
                            )
                        else:
                            logprobs = None

                        if self.use_harmony:
                            harmony_parser = harmony_parsers[i]
                            prev_recipient = harmony_parser.current_recipient
                            delta_text = ""
                            for token_id in output.token_ids:
                                harmony_parser.process(token_id)
                                delta_text += harmony_parser.last_content_delta or ""
                            cur_channel = harmony_parser.current_channel
                            cur_recipient = harmony_parser.current_recipient
                        else:
                            delta_text = output.text or ""

                        if not delta_text and not output.token_ids and not previous_num_tokens[i]:
                            # Chunked prefill case, don't return empty chunks
                            continue

                        delta_message: DeltaMessage | None

                        # just update previous_texts and previous_token_ids
                        if tool_choice_auto or reasoning_parser:
                            assert previous_texts is not None
                            assert all_previous_token_ids is not None
                            previous_text = previous_texts[i]
                            previous_token_ids = all_previous_token_ids[i]
                            current_text = previous_text + delta_text
                            # avoid the None + list error.
                            if previous_token_ids:
                                current_token_ids = previous_token_ids + as_list(output.token_ids)
                            else:
                                current_token_ids = as_list(output.token_ids)

                        if self.use_harmony:
                            if cur_channel == "final":
                                delta_message = DeltaMessage(content=delta_text)
                            elif cur_channel == "analysis":
                                if request.include_reasoning:
                                    delta_message = DeltaMessage(reasoning=delta_text)
                                else:
                                    delta_message = None
                            elif (
                                cur_channel == "commentary" and cur_recipient and cur_recipient.startswith("functions.")
                            ):
                                # Count completed tool calls to determine index
                                base_index = 0
                                for msg in harmony_parser.messages:
                                    if (
                                        msg.channel == "commentary"
                                        and msg.recipient
                                        and msg.recipient.startswith("functions.")
                                    ):
                                        base_index += 1

                                if prev_recipient != cur_recipient:
                                    tool_name = cur_recipient.split("functions.", 1)[1]
                                    delta_message = DeltaMessage(
                                        tool_calls=[
                                            DeltaToolCall(
                                                id=make_tool_call_id(),
                                                type="function",
                                                function=DeltaFunctionCall(
                                                    name=tool_name,
                                                    arguments="",
                                                ),
                                                index=base_index,
                                            )
                                        ]
                                    )
                                elif delta_text:
                                    delta_message = DeltaMessage(
                                        tool_calls=[
                                            DeltaToolCall(
                                                index=base_index,
                                                function=DeltaFunctionCall(arguments=delta_text),
                                            )
                                        ]
                                    )
                                else:
                                    delta_message = None

                                if delta_message is not None:
                                    harmony_tools_streamed[i] = True
                            else:
                                delta_message = None
                        # handle streaming deltas for tools with named tool_choice
                        elif tool_choice_function_name:
                            if (
                                reasoning_parser
                                and not reasoning_end_arr[i]
                                and not reasoning_parser.is_reasoning_end(previous_token_ids)
                            ):
                                assert reasoning_parser is not None
                                delta_message = reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                )
                                # When encountering think end id in delta_token_ids
                                # or think end id in prompt_token_ids
                                # i.e {"enable_thinking": False},
                                # set reasoning status to end.
                                # Only keep 'content', remove 'reasoning'.
                                if reasoning_parser.is_reasoning_end(as_list(output.token_ids)) or (
                                    res.prompt_token_ids and reasoning_parser.is_reasoning_end(res.prompt_token_ids)
                                ):
                                    reasoning_end_arr[i] = True
                                    if delta_message and delta_message.content:
                                        # This need to be added to next `delta_text`
                                        current_text = delta_message.content
                                        delta_message.content = None
                                    else:
                                        current_text = ""
                            else:
                                # Just to add remaining `content`
                                if reasoning_parser:
                                    delta_text = previous_text + delta_text
                                    current_text = ""

                                if function_name_returned[i]:
                                    delta_tool_call = DeltaToolCall(
                                        function=DeltaFunctionCall(arguments=delta_text),
                                        index=i,
                                    )
                                else:
                                    delta_tool_call = DeltaToolCall(
                                        id=make_tool_call_id(),
                                        type="function",
                                        function=DeltaFunctionCall(
                                            name=tool_choice_function_name,
                                            arguments=delta_text,
                                        ),
                                        index=i,
                                    )
                                    function_name_returned[i] = True

                                delta_message = DeltaMessage(
                                    tool_calls=[
                                        delta_tool_call,
                                    ]
                                )
                                tools_streamed[i] = True

                        elif request.tool_choice == "required":
                            assert previous_texts is not None
                            previous_text = previous_texts[i]
                            current_text = previous_text + delta_text
                            fn_name_returned = function_name_returned[i]
                            output_token_ids = as_list(output.token_ids)

                            if (
                                reasoning_parser is not None
                                and not reasoning_end_arr[i]
                                and res.prompt_token_ids
                                and reasoning_parser.is_reasoning_end(res.prompt_token_ids)
                            ):
                                reasoning_end_arr[i] = True

                            if reasoning_parser and not reasoning_end_arr[i]:
                                delta_message = reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output_token_ids,
                                )
                                if reasoning_parser.is_reasoning_end(output_token_ids):
                                    reasoning_end_arr[i] = True
                                    if delta_message and delta_message.content:
                                        current_text = delta_message.content
                                        delta_message.content = None
                                    else:
                                        # reasoning ended
                                        current_text = ""

                            else:
                                # either finished reasoning or no reasoning at all
                                content = current_text

                                delta_message, function_name_returned[i] = extract_required_tool_call_streaming(
                                    previous_text=previous_text,
                                    current_text=content,
                                    delta_text=delta_text,
                                    function_name_returned=fn_name_returned,
                                    tool_call_idx=history_tool_call_cnt,
                                    tool_call_id_type=self.tool_call_id_type,
                                )
                                if (
                                    delta_message
                                    and delta_message.tool_calls
                                    and delta_message.tool_calls[0].id is not None
                                ):
                                    history_tool_call_cnt += 1
                                    tools_streamed[i] = True

                        # handle streaming deltas for tools with "auto" tool choice
                        # and reasoning parser
                        elif tool_choice_auto and reasoning_parser:
                            assert tool_parser is not None
                            assert reasoning_parser is not None
                            assert added_content_delta_arr is not None
                            assert reasoning_end_arr is not None
                            output_token_ids = as_list(output.token_ids)
                            if not reasoning_end_arr[i]:
                                delta_message = reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output_token_ids,
                                )
                                # When encountering think end id in prompt_token_ids
                                # i.e {"enable_thinking": False},
                                # set reasoning status to end.
                                # Remove the text and token ids related
                                # to 'reasoning'.
                                if res.prompt_token_ids and reasoning_parser.is_reasoning_end(res.prompt_token_ids):
                                    reasoning_end_arr[i] = True
                                    current_token_ids = output_token_ids
                                    if delta_message and delta_message.content:
                                        current_text = delta_message.content
                                        delta_message.content = None
                                    else:
                                        current_text = ""
                                # When encountering think end id in delta_token_ids,
                                # set reasoning status to end.
                                # Remove the text and token ids related
                                # to 'reasoning'.
                                if reasoning_parser.is_reasoning_end(output_token_ids):
                                    reasoning_end_arr[i] = True
                                    current_token_ids = reasoning_parser.extract_content_ids(output_token_ids)
                                    if delta_message and delta_message.content:
                                        current_text = delta_message.content
                                        delta_message.content = None
                                    else:
                                        current_text = ""

                            # handle tool calls only after reasoning is done,
                            else:
                                delta_token_ids = output_token_ids
                                # First time to tool call,
                                # add the remaining text and token ids
                                # to delta from previous
                                if not added_content_delta_arr[i]:
                                    added_content_delta_arr[i] = True
                                    previous_text = ""
                                    previous_token_ids = []
                                    delta_text = current_text
                                    delta_token_ids = current_token_ids

                                delta_message = tool_parser.extract_tool_calls_streaming(
                                    previous_text=previous_text,
                                    current_text=current_text,
                                    delta_text=delta_text,
                                    previous_token_ids=previous_token_ids,
                                    current_token_ids=current_token_ids,
                                    delta_token_ids=delta_token_ids,
                                    request=request,
                                )
                                if delta_message and delta_message.tool_calls:
                                    tools_streamed[i] = True
                        # when only tool calls
                        elif tool_choice_auto:
                            assert tool_parser is not None
                            delta_message = tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=output.token_ids,
                                request=request,
                            )
                            if delta_message and delta_message.tool_calls:
                                tools_streamed[i] = True

                        # when only reasoning
                        elif reasoning_parser:
                            delta_message = reasoning_parser.extract_reasoning_streaming(
                                previous_text,
                                current_text,
                                delta_text,
                                previous_token_ids,
                                current_token_ids,
                                output.token_ids,
                            )
                        # handle streaming just a content delta
                        else:
                            delta_message = DeltaMessage(content=delta_text)

                        # update the previous values for the next iteration
                        if (tool_choice_auto or reasoning_parser) and not self.use_harmony:
                            assert previous_texts is not None
                            assert all_previous_token_ids is not None
                            previous_texts[i] = current_text
                            all_previous_token_ids[i] = current_token_ids
                        else:
                            # Update for comprehensive logging even in simple case
                            assert previous_texts is not None
                            previous_texts[i] += delta_text

                        # set the previous values for the next iteration
                        previous_num_tokens[i] += len(output.token_ids)

                        # if the message delta is None (e.g. because it was a
                        # "control token" for tool calls or the parser otherwise
                        # wasn't ready to send a token, then
                        #   get the next token without streaming a chunk
                        if delta_message is None:
                            if output.finish_reason is None and not request.return_token_ids:
                                continue
                            delta_message = DeltaMessage()

                        # Log streaming delta if output logging is enabled
                        if self.enable_log_outputs and self.request_logger:
                            delta_content = ""
                            if delta_message.content:
                                delta_content = delta_message.content
                            elif delta_message.tool_calls:
                                delta_content = "".join(
                                    tc.function.arguments
                                    for tc in delta_message.tool_calls
                                    if tc.function and tc.function.arguments
                                )

                            if delta_content:
                                self.request_logger.log_outputs(
                                    request_id=request_id,
                                    outputs=delta_content,
                                    output_token_ids=as_list(output.token_ids),
                                    finish_reason=output.finish_reason,
                                    is_streaming=True,
                                    delta=True,
                                )

                        if output.finish_reason is None:
                            # Send token-by-token response for each request.n
                            choice_data = ChatCompletionResponseStreamChoice(
                                index=i,
                                delta=delta_message,
                                logprobs=logprobs,
                                finish_reason=None,
                                token_ids=(as_list(output.token_ids) if request.return_token_ids else None),
                            )

                        # if the model is finished generating
                        else:
                            # check to make sure we haven't "forgotten" to stream
                            #   any tokens that were generated but previously
                            #   matched by partial json parsing
                            # only happens if we are NOT using structured outputs
                            auto_tools_called = False
                            if tool_parser:
                                auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                                index = len(tool_parser.prev_tool_call_arr) - 1 if auto_tools_called else 0
                            else:
                                index = 0

                            if self._should_check_for_unstreamed_tool_arg_tokens(delta_message, output) and tool_parser:
                                latest_delta_len = 0
                                if (
                                    isinstance(
                                        delta_message.tool_calls[0].function,
                                        DeltaFunctionCall,
                                    )
                                ) and isinstance(delta_message.tool_calls[0].function.arguments, str):
                                    latest_delta_len = len(delta_message.tool_calls[0].function.arguments)

                                # get the expected call based on partial JSON
                                # parsing which "autocompletes" the JSON.
                                # Tool parsers (e.g. Qwen3Coder) store
                                # arguments as a JSON string in
                                # prev_tool_call_arr. Calling json.dumps()
                                # on an already-serialized string would
                                # double-serialize it (e.g. '{"k":1}' becomes
                                # '"{\\"k\\":1}"'), which then causes the
                                # replace() below to fail and append the
                                # entire double-serialized string as a
                                # spurious final delta.
                                args = tool_parser.prev_tool_call_arr[index].get("arguments", {})
                                if isinstance(args, str):
                                    expected_call = args
                                else:
                                    expected_call = json.dumps(args, ensure_ascii=False)

                                # get what we've streamed so far for arguments
                                # for the current tool
                                actual_call = tool_parser.streamed_args_for_tool[index]
                                if latest_delta_len > 0:
                                    actual_call = actual_call[:-latest_delta_len]

                                # check to see if there's anything left to stream
                                remaining_call = expected_call.replace(actual_call, "", 1)
                                # set that as a delta message
                                delta_message = DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            index=index,
                                            function=DeltaFunctionCall(arguments=remaining_call).model_dump(
                                                exclude_none=True
                                            ),
                                        )
                                    ]
                                )

                            # Send the finish response for each request.n only once
                            # In OpenAI's API, when a tool is called, the
                            # finish_reason is:
                            # "tool_calls" for "auto" or "required" tool calls,
                            # and "stop" for named tool calls.
                            if (
                                auto_tools_called
                                or (tools_streamed[i] and not tool_choice_function_name)
                                or (self.use_harmony and harmony_tools_streamed[i])
                            ):
                                finish_reason_ = "tool_calls"
                            else:
                                finish_reason_ = output.finish_reason if output.finish_reason else "stop"
                            # Only emit finish_reason on the last modality to
                            # comply with OpenAI streaming spec: exactly one
                            # chunk per choice carries finish_reason="stop".
                            modality_finished[i].add("text")
                            if modality_seen[i] < set(request.modalities) or not all(
                                m in modality_finished[i] for m in modality_seen[i]
                            ):
                                finish_reason_ = None
                            else:
                                stop_reason_emitted[i] = True
                            choice_data = ChatCompletionResponseStreamChoice(
                                index=i,
                                delta=delta_message,
                                logprobs=logprobs,
                                finish_reason=finish_reason_,
                                stop_reason=output.stop_reason,
                                token_ids=(as_list(output.token_ids) if request.return_token_ids else None),
                            )

                            finish_reason_sent[i] = True

                        choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                        chunk = OmniChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            modality=final_output_type,
                            metrics=self._filter_stage_metrics_detail(omni_res.metrics, request),
                        )

                        # handle usage stats if requested & if continuous
                        if include_continuous_usage:
                            completion_tokens = previous_num_tokens[i]
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=num_prompt_tokens + completion_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                elif final_output_type == "audio":
                    # Observe audio_ttfp_s on first audio packet for this request_id
                    # (once-per-request guard via first_audio_ts). The same hook
                    # also captures (stage, replica) for the streaming-continuity
                    # emit at request finalize. self.engine_client.request_states
                    # is keyed by the internal UUID-suffixed id (set by
                    # AsyncOmni.generate via _get_unique_request_id), but the
                    # `request_id` we have here is the external (user-visible) id,
                    # so resolve via the external_request_id field.
                    req_state = next(
                        (s for s in self.engine_client.request_states.values() if s.external_request_id == request_id),
                        None,
                    )
                    if req_state is not None and req_state_audio_ref is None:
                        req_state_audio_ref = req_state
                    now_ts = time.time()
                    if req_state is not None and req_state.first_audio_ts is None:
                        req_state.first_audio_ts = now_ts
                        stage_pools = getattr(self.engine_client.engine, "stage_pools", None)
                        # The orchestrator binds requests by their internal id,
                        # not the user-visible external id, so look up the
                        # replica with req_state.request_id (internal).
                        replica_id = (
                            stage_pools[omni_res.stage_id].get_bound_replica_id(req_state.request_id)
                            if stage_pools is not None and 0 <= omni_res.stage_id < len(stage_pools)
                            else None
                        )
                        req_state.audio_emit_stage_id = omni_res.stage_id
                        req_state.audio_emit_replica_id = replica_id
                        observe_audio_first_packet(
                            self.engine_client.mod_metrics,
                            stage_id=omni_res.stage_id,
                            replica_id=replica_id,
                            arrival_ts=req_state.request_arrival_ts,
                            now_ts=now_ts,
                        )

                    role = self.get_chat_request_role(request)
                    choices_data = self._create_audio_choice(omni_res, role, request, stream=True)
                    # Only emit finish_reason on the last modality to
                    # comply with OpenAI streaming spec.
                    for choice in choices_data:
                        if choice.finish_reason is not None:
                            modality_finished[choice.index].add("audio")
                        if modality_seen[choice.index] < set(request.modalities) or not all(
                            m in modality_finished[choice.index] for m in modality_seen[choice.index]
                        ):
                            choice.finish_reason = None
                        else:
                            stop_reason_emitted[choice.index] = True
                    # Record per-chunk PCM byte count + arrival timestamp for
                    # audio_underrun_s / audio_continuity_ok_total at finalize.
                    if req_state is not None and req_state.request_arrival_ts > 0:
                        chunk_bytes = audio_chunk_pcm_bytes(omni_res)
                        if chunk_bytes > 0:
                            req_state.audio_chunk_arrivals_s.append(max(now_ts - req_state.request_arrival_ts, 0.0))
                            req_state.audio_chunk_bytes.append(chunk_bytes)
                            if req_state.audio_sample_rate is None:
                                req_state.audio_sample_rate = audio_chunk_sample_rate(omni_res)
                    chunk = OmniChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=choices_data,
                        model=model_name,
                        modality=final_output_type,
                        metrics=self._filter_stage_metrics_detail(omni_res.metrics, request),
                    )
                    chunk.usage = UsageInfo(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=0,
                        total_tokens=num_prompt_tokens,
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

                elif final_output_type == "image":
                    role = self.get_chat_request_role(request)
                    choices_data = []
                    for choice in self._create_image_choice(omni_res, role, request, stream=True):
                        delta = DeltaMessage.model_construct(role=role)
                        object.__setattr__(delta, "content", choice.message.content)
                        if hasattr(delta, "__pydantic_fields_set__"):
                            delta.__pydantic_fields_set__.add("content")
                        stream_choice = ChatCompletionResponseStreamChoice(
                            index=choice.index,
                            delta=delta,
                            logprobs=None,
                            finish_reason=choice.finish_reason,
                            stop_reason=choice.stop_reason,
                        )
                        if stream_choice.finish_reason is not None:
                            modality_finished[stream_choice.index].add("image")
                        if modality_seen[stream_choice.index] < set(request.modalities) or not all(
                            m in modality_finished[stream_choice.index] for m in modality_seen[stream_choice.index]
                        ):
                            stream_choice.finish_reason = None
                        else:
                            stop_reason_emitted[stream_choice.index] = True
                        choices_data.append(stream_choice)
                    chunk = OmniChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=choices_data,
                        model=model_name,
                        modality=final_output_type,
                        metrics=self._filter_stage_metrics_detail(omni_res.metrics, request),
                    )
                    # NOTE: Currently usage is only set the text stages to align with the behavior
                    # of the full generator. TODO (Alex): Add support for usage on all stages for
                    # both streaming and non-streaming.
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

                else:
                    logger.warning(f"Unsupported streaming final output type: {final_output_type}")
                    continue

            # Fallback: if a choice had modalities finish but no finish_reason="stop"
            # was emitted (e.g. request.modalities included a modality the engine
            # never produced), emit a final stop chunk for that choice.
            for i in range(num_choices):
                if modality_finished[i] and not stop_reason_emitted[i]:
                    stop_choice = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )
                    stop_chunk = OmniChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[stop_choice],
                        model=model_name,
                    )
                    data = stop_chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
            # Emit audio_underrun_s + audio_continuity_ok_total once per
            # request after the audio chunk stream is exhausted. The
            # captured reference (req_state_audio_ref) outlives the inner
            # _log_summary_and_cleanup that pops request_states[request_id].
            if (
                req_state_audio_ref is not None
                and req_state_audio_ref.audio_chunk_arrivals_s
                and req_state_audio_ref.audio_emit_replica_id is not None
            ):
                observe_audio_streaming_finalize(
                    self.engine_client.mod_metrics,
                    stage_id=req_state_audio_ref.audio_emit_stage_id or 0,
                    replica_id=req_state_audio_ref.audio_emit_replica_id,
                    chunk_arrival_times_s=req_state_audio_ref.audio_chunk_arrivals_s,
                    chunk_bytes=req_state_audio_ref.audio_chunk_bytes,
                    sample_rate=(req_state_audio_ref.audio_sample_rate or _metric_defs.DEFAULT_AUDIO_SAMPLE_RATE),
                )

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(cached_tokens=num_cached_tokens)

                final_usage_chunk = OmniChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    metrics=self._filter_stage_metrics_detail(last_metrics, request),
                )
                final_usage_data = final_usage_chunk.model_dump_json(exclude_unset=True, exclude_none=True)
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except EngineDeadError as e:
            logger.error(
                "EngineDeadError during streaming for request %s: %s",
                request_id,
                e,
            )
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            # Actively signal shutdown instead of waiting for the watchdog
            # (5s polling interval).
            if raw_request is not None:
                terminate_if_errored(
                    server=raw_request.app.state.server,
                    engine=self.engine_client,
                )
        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> ErrorResponse | OmniChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        final_outputs: list[OmniRequestOutput] = []
        try:
            async for res in result_generator:
                final_outputs.append(res)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)

        assert final_outputs is not None

        choices: list[ChatCompletionResponseChoice] = []

        usage = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        role = self.get_chat_request_role(request)
        prompt_logprobs = None
        prompt_token_ids = None
        kv_transfer_params = None
        response_metrics: dict[str, Any] | None = None

        # Build requested modalities set for filtering
        requested_modalities = (
            set(request.modalities) if hasattr(request, "modalities") and request.modalities else None
        )

        for omni_outputs in final_outputs:
            choices_data = []
            if omni_outputs.request_output is not None and not getattr(omni_outputs.request_output, "finished", False):
                continue

            # Filter outputs based on requested modalites
            if requested_modalities is not None and omni_outputs.final_output_type not in requested_modalities:
                logger.warning(f"final output type: {omni_outputs.final_output_type} is not needed by the request")
                continue

            if omni_outputs.final_output_type == "text":
                if omni_outputs.request_output is not None:
                    (
                        choices_data,
                        usage,
                        prompt_logprobs,
                        prompt_token_ids,
                        kv_transfer_params,
                    ) = self._create_text_choice(
                        request,
                        omni_outputs,
                        tokenizer,
                        conversation,
                        role,
                        reasoning_parser,
                    )
                    final_res = omni_outputs.request_output
                else:
                    # Diffusion pipeline text output (e.g. single-stage
                    # img2text / text2text) — no AR request_output, so build
                    # a simple text choice from diffusion multimodal output.
                    text_body = self._get_diffusion_text_output(omni_outputs)
                    message = ChatMessage(role=role, content=text_body)
                    choices_data = [
                        ChatCompletionResponseChoice(
                            index=0,
                            message=message,
                            logprobs=None,
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ]
            elif omni_outputs.final_output_type == "audio":
                choices_data = self._create_audio_choice(omni_outputs, role, request, stream=False)
            elif omni_outputs.final_output_type == "image":
                choices_data = self._create_image_choice(omni_outputs, role, request, stream=False)
            else:
                logger.warning(f"Unsupported final output type: {omni_outputs.final_output_type}")
                continue
            if omni_outputs.metrics:
                response_metrics = dict(omni_outputs.metrics)
            if omni_outputs.final_output_type == "image":
                # Expose diffusion profiler metrics on the top-level response for benchmarks / clients.
                if response_metrics is None:
                    response_metrics = {}
                response_metrics.setdefault("stage_durations", omni_outputs.stage_durations or {})
                response_metrics.setdefault("peak_memory_mb", float(omni_outputs.peak_memory_mb or 0.0))
                extra = self._get_diffusion_extra_output_params(omni_outputs)
                if extra:
                    response_metrics.update(extra)
            choices.extend(choices_data)

        response_metrics = self._filter_stage_metrics_detail(response_metrics, request)

        # Compute prompt_text for non-streaming response (upstream #42052)
        prompt_text = (
            getattr(final_res, "prompt", None)
            if final_res is not None and getattr(request, "return_prompt_text", None)
            else None
        )

        response = OmniChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=prompt_logprobs,
            prompt_token_ids=prompt_token_ids,
            prompt_text=prompt_text,
            kv_transfer_params=kv_transfer_params,
            metrics=response_metrics,
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
        reasoning_parser: ReasoningParser | None = None,
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
                # Use upstream's HarmonyParser (inherited as self.parser_cls for
                # gpt_oss/harmony models) instead of vendoring a copy of the
                # pre-refactor parse_chat_output. We take only reasoning+content;
                # tool calls are extracted by omni's own path below.
                if self.parser_cls is not None:
                    parser = self.parser_cls(tokenizer, request.tools)
                    reasoning, content, _ = parser.parse(
                        "",
                        request,
                        model_output_token_ids=token_ids,
                    )
                else:
                    reasoning, content = None, None
                if not request.include_reasoning:
                    reasoning = None

                if self.parser_cls is not None and self.parser_cls.tool_parser_cls is not None:
                    tool_parser = self.parser_cls.tool_parser_cls(tokenizer, request.tools)
                    # NOTE: We use token_ids for openai tool parser
                    tool_call_info = tool_parser.extract_tool_calls(
                        "",
                        request=request,
                        token_ids=token_ids,  # type: ignore
                    )
                    content = tool_call_info.content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_info.tool_calls,
                    )
                else:
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
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

            if reasoning_parser:
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning, content = reasoning_parser.extract_reasoning(output.text, request=request)
                if not request.include_reasoning:
                    reasoning = None
            else:
                reasoning = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if (not self.enable_auto_tools or self.parser_cls is None or self.parser_cls.tool_parser_cls is None) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # if the request uses tools and specified a tool choice
            elif request.tool_choice and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam:
                tool_call_class = MistralToolCall if isinstance(tokenizer, MistralTokenizer) else ToolCall
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
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
                    reasoning=reasoning,
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and self.parser_cls is not None
                and self.parser_cls.tool_parser_cls is not None
            ):
                try:
                    tool_parser = self.parser_cls.tool_parser_cls(tokenizer, request.tools)
                except RuntimeError as e:
                    logger.exception("Error in tool parser creation.")
                    return self.create_error_response(e)

                tool_call_info = tool_parser.extract_tool_calls(content if content is not None else "", request=request)
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
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
                        reasoning=reasoning,
                        content=ret_content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine if tools should be extracted. "
                    "Returning a standard chat completion."
                )
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

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

    def _create_audio_choice(
        self, omni_outputs: OmniRequestOutput, role: str, request: ChatCompletionRequest, stream: bool = False
    ):
        choices: list[ChatCompletionResponseChoice] = []
        final_res = omni_outputs.request_output
        # OMNI: Access multimodal_output from CompletionOutput (outputs[0]), not from RequestOutput
        # Reference: examples/offline_inference/qwen3_omni/end2end.py line 421
        # The attribute is attached dynamically when stage audio arrives; fall
        # back to the no-audio error response instead of an AttributeError 500
        # when the pipeline produced no audio for this request.
        mm_output = getattr(final_res.outputs[0], "multimodal_output", None) or {}
        audio_data = mm_output.get("audio")
        if isinstance(audio_data, list):
            if not audio_data:
                audio_tensor = None
            elif stream:
                audio_tensor = audio_data[-1]
            else:
                audio_tensor = torch.cat(audio_data, dim=-1)
        else:
            audio_tensor = audio_data
        if audio_tensor is None:
            return self._create_error_response("Audio generation completed but no audio was produced.")
        audio_tensor = audio_tensor.detach().cpu().float().numpy()

        # Ensure audio is 1D (flatten if needed)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.flatten()

        # Prefer the talker-reported sample rate when present. Qwen3-Omni
        # omits "sr" and runs at 24kHz; Ming-flash-omni surfaces a 44.1kHz
        # AudioVAE rate via multimodal_output["sr"].
        sr_raw = mm_output.get("sr")
        if isinstance(sr_raw, (list, tuple)):
            sr_raw = next((item for item in sr_raw if item is not None), None)
        if sr_raw is None:
            sample_rate = 24000
        elif hasattr(sr_raw, "item"):
            sample_rate = int(sr_raw.item())
        else:
            sample_rate = int(sr_raw)

        audio_format = self._resolve_audio_format(request)
        if isinstance(audio_format, ErrorResponse):
            return audio_format

        audio_obj = CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=sample_rate,
            response_format=audio_format,
            speed=1.0,
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
            if stream:
                choice_data = ChatCompletionResponseStreamChoice(
                    index=output.index,
                    delta=DeltaMessage(role=role, content=audio_base64),
                    logprobs=None,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                    token_ids=(as_list(output.token_ids) if request.return_token_ids else None),
                )
            else:
                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=ChatMessage(role=role, audio=audio_obj),
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=output.stop_reason,
                )
            choices.append(choice_data)
        return choices

    def _create_image_choice(
        self, omni_outputs: OmniRequestOutput, role: str, request: ChatCompletionRequest, stream: bool = False
    ):
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

        # Handle profiling data
        stage_durations = omni_outputs.stage_durations
        peak_memory_mb = omni_outputs.peak_memory_mb

        # Handle different image output formats
        images = []

        # First check omni_outputs.images directly (for diffusion mode via from_diffusion)
        if omni_outputs.images:
            images = omni_outputs.images
        # Fall back to request_output for pipeline mode
        # OMNI: Access multimodal_output from CompletionOutput (outputs[0]), not from RequestOutput
        elif final_res is not None and final_res.outputs:
            completion_output = final_res.outputs[0]
            if hasattr(completion_output, "multimodal_output") and completion_output.multimodal_output:
                image_data = completion_output.multimodal_output.get("image")
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
                    "stage_durations": stage_durations,
                    "peak_memory_mb": peak_memory_mb,
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
    def _build_multistage_generation_inputs(
        self,
        *,
        engine: AsyncOmni,
        prompt: str,
        extra_body: dict[str, Any],
        reference_images: list[Image.Image],
        gen_params: OmniDiffusionSamplingParams,
        tokenizer: Any = None,
    ) -> tuple[OmniTextPrompt, list[Any]]:
        """Build the shared multistage generation prompt and stage params."""
        stage_configs = getattr(engine, "stage_configs", None) or []
        default_params_list = get_default_sampling_params_list(engine)

        height = gen_params.height
        width = gen_params.width
        seed = gen_params.seed
        generator_device = gen_params.generator_device
        num_outputs_per_prompt = gen_params.num_outputs_per_prompt
        num_inference_steps = extra_body.get("num_inference_steps")
        guidance_scale = extra_body.get("guidance_scale")
        true_cfg_scale = extra_body.get("true_cfg_scale") or extra_body.get("cfg_scale")
        negative_prompt = extra_body.get("negative_prompt")
        num_frames = extra_body.get("num_frames")
        guidance_scale_2 = extra_body.get("guidance_scale_2")
        lora_body = extra_body.get("lora")
        layers = extra_body.get("layers")
        resolution = extra_body.get("resolution")
        bot_task = extra_body.get("bot_task")
        use_system_prompt = extra_body.get("use_system_prompt") or extra_body.get("sys_type")
        custom_system_prompt = extra_body.get("system_prompt")

        engine_prompt_data: dict[str, Any] | None = None
        modalities = ["image"]
        if reference_images:
            if len(reference_images) == 1:
                engine_prompt_data = {"img2img": reference_images[0]}
                modalities = ["img2img"]
            else:
                engine_prompt_data = {"image": reference_images}

        prompt_token_ids: list[int] | None = None
        system_prompt_type: str | None = None
        build_kwargs: dict[str, Any] = {}
        ar_stop_token_ids: list[int] | None = None

        if bot_task is not None or use_system_prompt is not None or custom_system_prompt is not None:
            from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
                build_prompt,
                build_prompt_tokens,
                resolve_stop_token_ids,
            )

            build_kwargs: dict[str, Any] = {
                "task": "it2i" if reference_images else "t2i",
                "sys_type": use_system_prompt,
                "custom_system_prompt": custom_system_prompt,
                "num_images": len(reference_images) if reference_images else 1,
            }

            if bot_task is not None:
                build_kwargs["bot_task"] = bot_task
            elif "bot_task" in extra_body:
                # Explicit None from the caller is plain-mode; omitted lets
                # each task fall back to its default trigger.
                build_kwargs["bot_task"] = extra_body["bot_task"]
            if tokenizer is not None:
                # Feed segment-tokenized prompt_token_ids so AR matches HF
                # apply_chat_template byte-for-byte (engine BPE would merge
                # across template boundaries, e.g. "。\n\n" -> single id).
                result = build_prompt_tokens(prompt, tokenizer, **build_kwargs)
                prompt_token_ids = result.token_ids
                system_prompt_type = result.system_prompt_type
            else:
                prompt = build_prompt(prompt, **build_kwargs)
            if reference_images and len(reference_images) == 1:
                engine_prompt_data = {"image": reference_images[0]}
                modalities = ["image"]

            ar_task = "it2i" if reference_images else "t2i"
            # ar_image_size: None -> need_ratio=True (AR predicts ratio);
            # explicit size -> need_ratio=False (AR stops at terminator).
            ar_image_size: str | None = None
            if height is not None and width is not None:
                ar_image_size = f"{width}x{height}"
            ar_stop_token_ids = resolve_stop_token_ids(
                task=ar_task,
                bot_task=bot_task,
                tokenizer=tokenizer,
                image_size=ar_image_size,
            )

        engine_prompt: OmniTextPrompt = {"prompt": prompt}
        if prompt_token_ids is not None:
            engine_prompt["prompt_token_ids"] = prompt_token_ids
        if system_prompt_type is not None:
            engine_prompt["use_system_prompt"] = system_prompt_type
        # DiT's get_system_prompt(use_system_prompt, "image", system_prompt) reads
        # this; omitting it makes sys_type=custom yield an empty DiT prefix.
        if custom_system_prompt is not None:
            engine_prompt["system_prompt"] = custom_system_prompt
        engine_prompt["modalities"] = modalities
        if negative_prompt is not None:
            engine_prompt["negative_prompt"] = negative_prompt

        mm_processor_kwargs: dict[str, Any] = {}
        if height is not None:
            mm_processor_kwargs["target_h"] = height
        if width is not None:
            mm_processor_kwargs["target_w"] = width
        if seed is not None and engine_prompt_data is not None:
            mm_processor_kwargs["vae_generator_seed"] = int(seed)
        if mm_processor_kwargs:
            engine_prompt["mm_processor_kwargs"] = mm_processor_kwargs
        if engine_prompt_data is not None:
            engine_prompt["multi_modal_data"] = engine_prompt_data
            # Provide multi_modal_uuids so that newer vLLM versions can
            # validate multi_modal_data / multi_modal_uuids consistency.
            # Generate one uuid per image when the value is a list (multi-image inputs).
            engine_prompt["multi_modal_uuids"] = {
                k: [f"img-{k}-{i}" for i in range(len(v))] if isinstance(v, list) else [f"img-{k}-0"]
                for k, v in engine_prompt_data.items()
            }

        comprehension_idx = None
        for idx, stage in enumerate(stage_configs):
            if getattr(stage, "is_comprehension", False):
                comprehension_idx = idx
                break

        sampling_params_list = build_stage_sampling_params_list(
            stage_configs,
            default_params_list,
            diffusion_params=gen_params,
        )
        for idx, stage_cfg in enumerate(stage_configs):
            stage_type = get_stage_type(stage_cfg)
            default_stage_params = sampling_params_list[idx]

            # AR stop tokens: use stage_type=="llm" instead of comprehension_idx
            # (None for DictConfig where is_comprehension is nested in engine_args).
            if stage_type == "llm" and ar_stop_token_ids is not None:
                default_stage_params.stop_token_ids = ar_stop_token_ids

            if (
                comprehension_idx is not None
                and idx == comprehension_idx
                and seed is not None
                and hasattr(default_stage_params, "seed")
            ):
                default_stage_params.seed = seed

            # Inject target_h/w into AR stage for M-RoPE position pre-computation
            # (e.g. GLM-Image). max_tokens comes from deploy YAML.
            if comprehension_idx is not None and idx == comprehension_idx and height is not None and width is not None:
                extra_args = getattr(default_stage_params, "extra_args", None)
                if extra_args is None:
                    extra_args = {}
                    default_stage_params.extra_args = extra_args
                extra_args["target_h"] = int(height)
                extra_args["target_w"] = int(width)

            if stage_type == "diffusion":
                self._set_if_supported(
                    default_stage_params,
                    height=height,
                    width=width,
                    seed=seed,
                    generator_device=generator_device,
                    num_outputs_per_prompt=num_outputs_per_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    num_frames=num_frames,
                    guidance_scale_2=guidance_scale_2,
                    layers=layers,
                    resolution=resolution,
                )
                apply_declared_extra_args(
                    default_stage_params,
                    self._get_diffusion_extra_body_params(),
                    extra_body,
                )
                if lora_body and isinstance(lora_body, dict):
                    try:
                        lora_req, lora_scale = parse_lora_request(lora_body)
                        if lora_req is not None:
                            default_stage_params.lora_request = lora_req
                            if lora_scale is not None:
                                default_stage_params.lora_scale = lora_scale
                    except Exception as e:  # pragma: no cover - safeguard
                        logger.warning("Failed to parse LoRA request: %s", e)

        return engine_prompt, sampling_params_list

    def _prepare_diffusion_image_request(
        self,
        *,
        prompt: str,
        extra_body: dict[str, Any] | None = None,
        reference_images: list[str] | None = None,
    ) -> tuple[Any, OmniTextPrompt, OmniDiffusionSamplingParams, list[Image.Image]] | ErrorResponse:
        if extra_body is None:
            extra_body = {}
        if reference_images is None:
            reference_images = []

        engine = self._diffusion_engine if self._diffusion_engine is not None else self.engine_client

        height, width = self._resolve_height_width_from_extra_body(extra_body)

        seed = extra_body.get("seed")
        generator_device = extra_body.get("generator_device")
        negative_prompt = extra_body.get("negative_prompt")
        num_outputs_per_prompt = extra_body.get("num_outputs_per_prompt", 1)
        lora_body = extra_body.get("lora")

        pil_images: list[Image.Image] = []
        for img_b64 in reference_images:
            try:
                img_bytes = base64.b64decode(img_b64)
                pil_images.append(Image.open(BytesIO(img_bytes)))
            except Exception as e:
                logger.warning("Failed to decode reference image: %s", e)

        gen_params = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_outputs_per_prompt=num_outputs_per_prompt,
            seed=seed,
        )
        self._set_if_supported(
            gen_params,
            generator_device=generator_device,
            num_inference_steps=extra_body.get("num_inference_steps"),
            guidance_scale=extra_body.get("guidance_scale"),
            true_cfg_scale=extra_body.get("true_cfg_scale") or extra_body.get("cfg_scale"),
            num_frames=extra_body.get("num_frames"),
            guidance_scale_2=extra_body.get("guidance_scale_2"),
            layers=extra_body.get("layers"),
            resolution=extra_body.get("resolution"),
            strength=extra_body.get("strength"),
        )

        if lora_body and isinstance(lora_body, dict):
            try:
                lora_req, lora_scale = parse_lora_request(lora_body)
                if lora_req is not None:
                    gen_params.lora_request = lora_req
                    if lora_scale is not None:
                        gen_params.lora_scale = lora_scale
            except Exception as e:  # pragma: no cover - safeguard
                logger.warning("Failed to parse LoRA request: %s", e)

        gen_prompt: OmniTextPrompt = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "modalities": ["image"],
        }
        if pil_images:
            if len(pil_images) == 1:
                gen_prompt["multi_modal_data"] = {"image": pil_images[0]}
            else:
                od_config = getattr(engine, "od_config", None)
                supports_multimodal_inputs = getattr(od_config, "supports_multimodal_inputs", False)
                if od_config is None:
                    supports_multimodal_inputs = True
                if supports_multimodal_inputs:
                    gen_prompt["multi_modal_data"] = {"image": pil_images}
                else:
                    return self._create_error_response(
                        "Multiple input images are not supported by the current diffusion model. "
                        "For multi-image editing, start the server with Qwen-Image-Edit-2509 "
                        "and send multiple images in the user message content.",
                        status_code=400,
                    )

        return engine, gen_prompt, gen_params, pil_images

    async def generate_diffusion_images(
        self,
        *,
        prompt: str,
        extra_body: dict[str, Any] | None = None,
        reference_images: list[str] | None = None,
        request_id: str | None = None,
        arrival_time: float | None = None,
        stream: bool = False,
        model: str | None = None,
        output_format: str = "png",
        output_compression: int = 100,
        size: str = "auto",
        raw_request: Request | None = None,
    ) -> tuple[list[Image.Image], dict[str, Any], float, str | None] | ErrorResponse | AsyncIterator[str]:
        """Generate diffusion images and return raw images plus generation stats."""
        if request_id is None:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"

        prepared = self._prepare_diffusion_image_request(
            prompt=prompt,
            extra_body=extra_body,
            reference_images=reference_images,
        )
        if isinstance(prepared, ErrorResponse):
            return prepared
        engine, gen_prompt, gen_params, pil_images = prepared
        if extra_body is None:
            extra_body = {}
        return_stage_metrics = self._truthy_extra_body_flag(extra_body, "return_stage_metrics")

        if isinstance(engine, AsyncOmni):
            diffusion_engine = cast(AsyncOmni, engine)
            stage_configs = getattr(diffusion_engine, "stage_configs", None) or []
            if stream and len(stage_configs) <= 1:
                return self._create_error_response(
                    "stream=true is only supported for multi-stage image editing pipelines.",
                    status_code=400,
                )
            if len(stage_configs) > 1:
                # Pull tokenizer from the comprehension (AR) stage so we can
                # build HF byte-for-byte prompt_token_ids in the helper. If
                # the engine doesn"t expose one, fall back to the legacy
                # string-prompt path (engine re-tokenizes).
                tokenizer = None
                get_tok = getattr(diffusion_engine, "get_tokenizer", None)
                if get_tok is not None:
                    try:
                        tokenizer = await get_tok()
                    except Exception as exc:
                        logger.warning("get_tokenizer failed; falling back to string prompt path: %s", exc)
                engine_prompt, sampling_params_list = self._build_multistage_generation_inputs(
                    engine=diffusion_engine,
                    prompt=prompt,
                    extra_body=extra_body,
                    reference_images=pil_images,
                    gen_params=gen_params,
                    tokenizer=tokenizer,
                )
            else:
                engine_prompt = gen_prompt
                sampling_params_list = [gen_params]

            sampling_params_list = coerce_param_message_types(sampling_params_list, stream)
            result_generator = diffusion_engine.generate(
                prompt=engine_prompt,
                sampling_params_list=sampling_params_list,
                request_id=request_id,
                arrival_time=arrival_time,
            )
            if stream:
                return self._stream_diffusion_image_chunks(
                    result_generator,
                    model=model or self._diffusion_model_name,
                    output_format=output_format,
                    output_compression=output_compression,
                    size=size,
                    return_stage_metrics=return_stage_metrics,
                    raw_request=raw_request,
                )

            result = None
            async for output in result_generator:
                result = output
            if result is None:
                return self._create_error_response("No output generated from AsyncOmni", status_code=500)
        elif stream:
            return self._create_error_response(
                "Streaming image edits require a multi-stage AsyncOmni engine.",
                status_code=400,
            )
        else:
            result = await engine.generate(
                prompt=gen_prompt,
                sampling_params=gen_params,
                request_id=request_id,
            )

        images = getattr(result.request_output, "images", [])
        stage_durations = result.stage_durations
        peak_memory_mb = result.peak_memory_mb
        cot_output = None

        req_out = getattr(result, "request_output", None)
        if req_out:
            prompt_obj = getattr(req_out, "prompt", None)
            if isinstance(prompt_obj, dict):
                extra = prompt_obj.get("extra", {})
                if isinstance(extra, dict):
                    ar_text = extra.get("ar_generated_text")
                    if isinstance(ar_text, str) and ar_text.strip():
                        cot_output = ar_text

        req_out = getattr(result, "request_output", None)
        if req_out:
            prompt_obj = getattr(req_out, "prompt", None)
            if isinstance(prompt_obj, dict):
                extra = prompt_obj.get("extra", {})
                if isinstance(extra, dict):
                    ar_text = extra.get("ar_generated_text")
                    if isinstance(ar_text, str) and ar_text.strip():
                        cot_output = ar_text

        req_out = getattr(result, "request_output", None)
        if req_out:
            prompt_obj = getattr(req_out, "prompt", None)
            if isinstance(prompt_obj, dict):
                extra = prompt_obj.get("extra", {})
                if isinstance(extra, dict):
                    ar_text = extra.get("ar_generated_text")
                    if isinstance(ar_text, str) and ar_text.strip():
                        cot_output = ar_text

        return self._flatten_diffusion_images(images), stage_durations, peak_memory_mb, cot_output

    async def _stream_diffusion_image_chunks(
        self,
        result_generator: AsyncIterator[OmniRequestOutput],
        *,
        model: str,
        output_format: str,
        output_compression: int,
        size: str,
        return_stage_metrics: bool,
        raw_request: Request | None = None,
    ) -> AsyncIterator[str]:
        """Yield image edit SSE chunks from multi-stage diffusion outputs."""
        created = int(time.time())
        emitted_image = False
        try:
            async for output in result_generator:
                final_output_type = getattr(output, "final_output_type", None)
                stage_id = getattr(output, "stage_id", None)
                metrics = getattr(output, "metrics", None) if return_stage_metrics else None
                if final_output_type == "text" and stage_id == 0:
                    request_output = output.request_output
                    for completion in request_output.outputs:
                        text = completion.text or ""
                        if not text:
                            continue
                        chunk = ImageEditARDeltaChunk(
                            delta=text,
                            index=completion.index,
                            created=created,
                            model=model,
                            metrics=metrics,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                elif final_output_type == "image":
                    images = self._flatten_diffusion_images(getattr(output.request_output, "images", []))
                    if not images:
                        raise RuntimeError("Streaming image edit produced an empty final image output.")
                    image_data = [
                        ImageData(
                            b64_json=encode_image_base64_with_compression(
                                img,
                                format=output_format,
                                output_compression=output_compression,
                            ),
                            revised_prompt=None,
                        )
                        for img in images
                    ]
                    chunk = ImageEditImageChunk(
                        data=image_data,
                        output_format=output_format,
                        size=size,
                        created=created,
                        model=model,
                        metrics=metrics,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    emitted_image = True
            if not emitted_image:
                raise RuntimeError("Streaming image edit completed without a final image output.")
        except EngineDeadError as exc:
            logger.error("EngineDeadError during streaming image edit: %s", exc)
            data = self.create_streaming_error_response(exc)
            yield f"data: {data}\n\n"
            if raw_request is not None:
                terminate_if_errored(
                    server=raw_request.app.state.server,
                    engine=self.engine_client,
                )
            else:
                logger.warning(
                    "[OmniOpenAIServingChat] Engine dead during streaming image edit, "
                    "but cannot terminate server (no raw_request context)."
                )
        except OmniClientError as exc:
            logger.info("Client error during streaming image edit: %s", exc)
            chunk = ImageEditStreamError(
                created=created,
                model=model,
                error={
                    "message": exc.message,
                    "type": exc.error_type,
                    "code": exc.status_code,
                },
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as exc:
            logger.exception("Streaming image edit failed: %s", exc)
            chunk = ImageEditStreamError(
                created=created,
                model=model,
                error={
                    "message": str(exc),
                    "type": "server_error",
                    "code": 500,
                },
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    @staticmethod
    def _flatten_diffusion_images(images: Any) -> list[Image.Image]:
        flat_images: list[Image.Image] = []
        for item in images or []:
            if isinstance(item, list):
                flat_images.extend(item)
            else:
                flat_images.append(item)
        return flat_images

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

            # Extract prompt and multimodal inputs from messages
            prompt, reference_images, reference_videos, reference_audios = self._extract_diffusion_prompt_and_media(
                messages
            )

            # Extract generation parameters from extra_body (preferred)
            # Reference: text_to_image.py and text_to_video.py for supported parameters
            # [NOTE] When sending request via openai client Python library,
            #   `extra_body` is flattented and merged into the payload's root.
            #   These extra fields are accessible via `model_extra` property (from Pydantic base class).
            #   When sending raw request with curl, no flattening happens. Directly read the `extra_body` dict.
            extra_body = getattr(request, "extra_body", None)
            if not extra_body:
                extra_body = request.model_extra or {}

            # Parse size if provided (supports "1024x1024" format)
            height, width = self._resolve_height_width_from_extra_body(extra_body)

            # Get request parameters from extra_body.
            # Avoid hardcoded defaults here — let each pipeline's forward()
            # method apply its own model-specific default when the user does
            # not provide a value.
            num_inference_steps = extra_body.get("num_inference_steps")
            guidance_scale = extra_body.get("guidance_scale")
            true_cfg_scale = extra_body.get("true_cfg_scale") or extra_body.get("cfg_scale")
            seed = extra_body.get("seed")
            if seed is None:
                seed = getattr(request, "seed", None)
            negative_prompt = extra_body.get("negative_prompt")
            num_outputs_per_prompt = extra_body.get("num_outputs_per_prompt", 1)

            # Text-to-video parameters (ref: text_to_video.py)
            num_frames = extra_body.get("num_frames")
            guidance_scale_2 = extra_body.get("guidance_scale_2")
            lora_body = extra_body.get("lora")

            # Qwen-Image-Layered parameters
            layers = extra_body.get("layers")
            resolution = extra_body.get("resolution")

            try:
                layers = validate_layered_layers(layers)
            except ValueError as e:
                return self._create_error_response(str(e), status_code=400)

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
            gen_prompt: OmniTextPrompt = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "modalities": ["image"],
            }
            gen_params = OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_outputs_per_prompt=num_outputs_per_prompt,
                seed=seed,
            )

            # Only override defaults when the user explicitly provides values
            if num_inference_steps is not None:
                gen_params.num_inference_steps = num_inference_steps
            if guidance_scale is not None:
                gen_params.guidance_scale = guidance_scale
            if true_cfg_scale is not None:
                gen_params.true_cfg_scale = true_cfg_scale
            apply_declared_extra_args(gen_params, self._get_diffusion_extra_body_params(), extra_body)
            if num_frames is not None:
                gen_params.num_frames = num_frames
            if guidance_scale_2 is not None:
                gen_params.guidance_scale_2 = guidance_scale_2
            if layers is not None:
                gen_params.layers = layers
            if resolution is not None:
                gen_params.resolution = resolution

            # Pipeline-agnostic escape hatch (mirrors ``extra_params`` on the /v1/videos
            # endpoint in ``serving_video.py``): a single reserved ``extra_args`` dict in
            # ``extra_body`` flows straight into ``gen_params.extra_args``, with no keys
            # hardcoded here.
            extra_args_body = extra_body.get("extra_args")
            if isinstance(extra_args_body, dict):
                gen_params.extra_args.update(extra_args_body)

            # Parse per-request LoRA.
            if lora_body and isinstance(lora_body, dict):
                try:
                    lora_req, lora_scale = parse_lora_request(lora_body)
                    if lora_req is not None:
                        gen_params.lora_request = lora_req
                        if lora_scale is not None:
                            gen_params.lora_scale = lora_scale
                except Exception as e:  # pragma: no cover - safeguard
                    logger.warning("Failed to parse LoRA request: %s", e)

            # Route text modality for single-stage diffusion (img2text / text2text)
            requested_modalities = extra_body.get("modalities") or []
            is_text_request = "text" in requested_modalities

            if is_text_request:
                gen_prompt["modalities"] = ["text"]

            # Add reference image if provided (from messages content)
            if pil_images:
                if len(pil_images) == 1:
                    gen_prompt["multi_modal_data"] = {}
                    gen_prompt["multi_modal_data"]["image"] = pil_images[0]
                else:
                    od_config = getattr(self._diffusion_engine, "od_config", None)
                    supports_multimodal_inputs = getattr(od_config, "supports_multimodal_inputs", False)
                    if od_config is None:
                        # TODO: entry is asyncOmni. We hack the od config here.
                        supports_multimodal_inputs = True
                    if supports_multimodal_inputs:
                        gen_prompt["multi_modal_data"] = {}
                        gen_prompt["multi_modal_data"]["image"] = pil_images
                    else:
                        return self._create_error_response(
                            "Multiple input images are not supported by the current diffusion model. "
                            "For multi-image editing, start the server with Qwen-Image-Edit-2509 "
                            "and send multiple images in the user message content.",
                            status_code=400,
                        )

            if reference_videos:
                gen_params.extra_args["video_path"] = reference_videos[0]
            if reference_audios:
                gen_params.extra_args["audio_path"] = reference_audios[0]

            # Generate image or audio (e.g. AudioX) via AsyncOmni
            diffusion_engine = cast(AsyncOmni, self._diffusion_engine)
            stage_configs = list(getattr(diffusion_engine, "stage_configs", []) or [])
            sampling_params_list = build_stage_sampling_params_list(
                stage_configs,
                get_default_sampling_params_list(diffusion_engine),
                diffusion_params=gen_params,
                replace_diffusion_params=True,
            )

            if not sampling_params_list:
                sampling_params_list = [gen_params]

            result = None
            async for output in diffusion_engine.generate(
                prompt=gen_prompt,
                sampling_params_list=sampling_params_list,
                request_id=request_id,
            ):
                result = output
            if result is None:
                return self._create_error_response("No output generated from AsyncOmni")

            # Text output path (img2text / text2text)
            if is_text_request and result.final_output_type == "text":
                text_body = self._get_diffusion_text_output(result)
                message = ChatMessage(role="assistant", content=text_body)
                choice = ChatCompletionResponseChoice(
                    index=0,
                    message=message,
                    finish_reason="stop",
                    logprobs=None,
                    stop_reason=None,
                )
                response = OmniChatCompletionResponse(
                    id=request_id,
                    created=created_time,
                    model=self._diffusion_model_name,
                    choices=[choice],
                    usage=UsageInfo(
                        prompt_tokens=len(prompt.split()),
                        completion_tokens=len(text_body.split()),
                        total_tokens=len(prompt.split()) + len(text_body.split()),
                    ),
                    metrics=self._get_diffusion_extra_output_params(result),
                )
                logger.info(
                    "Diffusion chat completed for request %s: text output (%d chars)",
                    request_id,
                    len(text_body),
                )
                return response

            # Image output path (text2img / img2img)
            final_output_type = getattr(result, "final_output_type", "image")
            # Handle nested OmniRequestOutput structure where images might be in request_output
            images = getattr(result.request_output, "images", [])
            multimodal_output = getattr(result, "multimodal_output", {}) or {}
            stage_durations = result.stage_durations
            peak_memory_mb = result.peak_memory_mb

            if final_output_type == "audio":
                sample_rate = 48000
                for key in ("audio_sample_rate", "sample_rate", "sampling_rate", "sr"):
                    raw_rate = multimodal_output.get(key)
                    try:
                        if raw_rate is not None:
                            sample_rate = int(raw_rate)
                            break
                    except (TypeError, ValueError):
                        pass

                audio_payload = multimodal_output.get("audio")
                if isinstance(audio_payload, list):
                    if len(audio_payload) == 0:
                        audio_payload = None
                    elif len(audio_payload) == 1:
                        audio_payload = audio_payload[0]
                if audio_payload is None:
                    return self._create_error_response("Audio generation completed but no audio was produced.")

                if isinstance(audio_payload, torch.Tensor):
                    audio_tensor = audio_payload.detach().cpu().float()
                else:
                    audio_tensor = torch.as_tensor(audio_payload).detach().cpu().float()
                # Pipelines deliver audio as (C, T), (T,), or (B, C, T) in channels-first
                # convention (torch default). Drop a leading batch dim, then transpose to
                # (T, C) for soundfile / CreateAudio. Flattening here would corrupt stereo.
                if audio_tensor.ndim == 3:
                    audio_tensor = audio_tensor[0]
                if audio_tensor.ndim == 2:
                    audio_tensor = audio_tensor.transpose(0, 1).contiguous()
                elif audio_tensor.ndim > 3:
                    raise ValueError(f"Unexpected audio tensor rank {audio_tensor.ndim}; expected 1-3 dims.")
                audio_array = audio_tensor.numpy()

                audio_format = self._resolve_audio_format(request)
                if isinstance(audio_format, ErrorResponse):
                    return audio_format

                audio_obj = CreateAudio(
                    audio_tensor=audio_array,
                    sample_rate=sample_rate,
                    response_format=audio_format,
                    speed=1.0,
                    base64_encode=True,
                )
                audio_response: AudioResponse = self.create_audio(audio_obj)
                audio_base64 = audio_response.audio_data
                audio_id = f"audio-{uuid.uuid4().hex[:16]}"
                expires_at = int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp())
                message = ChatMessage(
                    role="assistant",
                    audio=OpenAIChatCompletionAudio(
                        id=audio_id,
                        data=audio_base64,
                        expires_at=expires_at,
                        transcript="",
                    ),
                )
            else:
                # Convert images to base64 content
                image_contents: list[dict[str, Any]] = []
                flat_images = []
                for item in images:
                    if isinstance(item, list):
                        flat_images.extend(item)
                    else:
                        flat_images.append(item)

                for img in flat_images:
                    with BytesIO() as buffer:
                        img.save(buffer, format="PNG")
                        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    image_contents.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                            },
                            "stage_durations": stage_durations,
                            "peak_memory_mb": peak_memory_mb,
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

            response = OmniChatCompletionResponse(
                id=request_id,
                created=created_time,
                model=self._diffusion_model_name,
                choices=[choice],
                usage=UsageInfo(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=1,
                    total_tokens=len(prompt.split()) + 1,
                ),
                metrics=self._get_diffusion_extra_output_params(result),
            )

            logger.info(
                "Diffusion chat completed for request %s: output_type=%s, image_count=%d",
                request_id,
                final_output_type,
                len(images),
            )

            return response

        except OmniClientError as e:
            logger.info("Client error during diffusion chat completion: %s", e)
            return self._create_error_response(
                e.message,
                err_type=e.error_type,
                status_code=e.status_code,
            )
        except Exception as e:
            logger.exception("Diffusion chat completion failed: %s", e)
            return self._create_error_response(
                f"Image generation failed: {str(e)}",
                status_code=500,
            )

    def _extract_diffusion_prompt_and_media(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Extract text prompt and multimodal inputs from chat messages.

        Args:
            messages: List of chat messages

        Returns:
            Tuple of (prompt_text, list_of_base64_images, list_of_video_urls, list_of_audio_urls)
        """
        prompt_parts: list[str] = []
        images: list[str] = []
        videos: list[str] = []
        audios: list[str] = []

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
                        elif item.get("type") == "video_url":
                            url = item.get("video_url", {}).get("url", "")
                            if isinstance(url, str) and url:
                                videos.append(url)
                        elif item.get("type") == "audio_url":
                            url = item.get("audio_url", {}).get("url", "")
                            if isinstance(url, str) and url:
                                audios.append(url)
                        # Handle {"image": "base64..."} format
                        elif "image" in item:
                            images.append(item["image"])
                        elif "video" in item and isinstance(item["video"], str):
                            videos.append(item["video"])
                        elif "audio" in item and isinstance(item["audio"], str):
                            audios.append(item["audio"])

        prompt = " ".join(prompt_parts).strip()
        return prompt, images, videos, audios

    def _extract_diffusion_prompt_and_images_from_messages(
        self,
        messages: list[Any],
    ) -> tuple[str, list[str]]:
        """Normalize mixed message types and extract prompt + reference images once."""
        prompt, images, _videos, _audios = self._extract_diffusion_prompt_and_media(self._messages_to_dicts(messages))
        return prompt, images

    @staticmethod
    def _messages_to_dicts(messages: list[Any]) -> list[dict[str, Any]]:
        """Normalize request messages to plain dicts."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                out.append(msg.model_dump())
            elif isinstance(msg, dict):
                out.append(msg)
            else:
                out.append(
                    {
                        "role": getattr(msg, "role", "user"),
                        "content": getattr(msg, "content", ""),
                    }
                )
        return out

    @staticmethod
    def _resolve_height_width_from_extra_body(extra_body: dict[str, Any]) -> tuple[Any, Any]:
        """Extract generation height/width with optional size string fallback."""
        height = extra_body.get("height")
        width = extra_body.get("width")

        if "size" in extra_body and (height is None or width is None):
            try:
                size_str = extra_body["size"]
                if isinstance(size_str, str) and "x" in size_str.lower():
                    w, h = size_str.lower().split("x")
                    width, height = int(w), int(h)
            except ValueError:
                logger.warning("Invalid size format: %s", extra_body.get("size"))

        return height, width

    def _resolve_audio_format(self, request: ChatCompletionRequest) -> str | ErrorResponse:
        """Extract and validate the audio output format from a chat completion request."""
        audio_params = getattr(request, "audio", None)
        if isinstance(audio_params, dict):
            audio_format = audio_params.get("format", DEFAULT_AUDIO_FORMAT)
        else:
            audio_format = DEFAULT_AUDIO_FORMAT
        if audio_format not in SUPPORTED_CHAT_AUDIO_FORMATS:
            return self._create_error_response(
                f"Invalid audio format '{audio_format}'. Supported formats: {sorted(SUPPORTED_CHAT_AUDIO_FORMATS)}",
            )
        if audio_format == "pcm16":
            audio_format = "pcm"
        return audio_format

    def _create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> ErrorResponse:
        """Create an error response following OpenAI error format."""
        return ErrorResponse(
            error=ErrorInfo(
                message=message,
                type=err_type,
                code=status_code,
            )
        )
