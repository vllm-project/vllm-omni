import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.inputs import ProcessorInputs, PromptType
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import argsort_mm_positions
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.jsontree import json_iter_leaves
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.engine import (
    AdditionalInformationEntry,
    AdditionalInformationPayload,
    OmniEngineCoreRequest,
    PromptEmbedsPayload,
)
from vllm_omni.inputs.preprocess import OmniInputPreprocessor
from vllm_omni.lora.request import LoRARequest

logger = init_logger(__name__)

_OMNI_EXTRA_KEYS = (
    "additional_information",
    "prompt_embeds",
    "negative_prompt",
    "negative_prompt_embeds",
)


def reinject_omni_fields(
    results: list[ProcessorInputs],
    original_prompts: list[dict],
) -> None:
    """Re-inject omni-specific fields that the upstream renderer discards.

    The upstream renderer's ``process_for_engine`` creates new dicts that only
    copy standard vLLM fields (prompt_token_ids, multi_modal_data, …).
    Omni-specific fields such as ``additional_information`` and
    ``prompt_embeds`` are silently dropped.  This helper copies them back from
    the *original* parsed prompts into the renderer outputs so they survive
    into ``OmniInputProcessor.process_inputs()``.
    """
    for result, orig in zip(results, original_prompts):
        if not isinstance(orig, dict):
            continue
        for key in _OMNI_EXTRA_KEYS:
            val = orig.get(key)
            if val is not None and key not in result:
                result[key] = val


class OmniInputProcessor(InputProcessor):
    """Processor for omni models, handling multimodal inputs and embeddings.

    Extends the base vLLM Processor with support for processing prompt
    embeddings and additional information payloads, enabling direct transfer
    of pre-computed embeddings between pipeline stages.

    Args:
        vllm_config: Global vLLM configuration
        mm_registry: Multi-modal registry for processing multimodal inputs
    """

    @staticmethod
    def _dtype_to_name(dtype: torch.dtype) -> str:
        """Convert torch dtype to string representation.

        Args:
            dtype: PyTorch dtype to convert

        Returns:
            String representation of the dtype (e.g., "float32", "int64")
        """
        mapping = {
            torch.float32: "float32",
            torch.float: "float32",
            torch.float16: "float16",
            torch.half: "float16",
            torch.bfloat16: "bfloat16",
            torch.float64: "float64",
            torch.double: "float64",
            torch.int64: "int64",
            torch.long: "int64",
            torch.int32: "int32",
            torch.int: "int32",
            torch.int16: "int16",
            torch.short: "int16",
            torch.int8: "int8",
            torch.uint8: "uint8",
            torch.bool: "bool",
        }
        return mapping.get(dtype, str(dtype).replace("torch.", ""))

    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        super().__init__(vllm_config, renderer=renderer, mm_registry=mm_registry)
        self.input_preprocessor = OmniInputPreprocessor(
            vllm_config=vllm_config,
            renderer=self.renderer,
            mm_registry=mm_registry,
        )

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | ProcessorInputs,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...] = ("generate",),
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
    ) -> OmniEngineCoreRequest:
        """Process input prompt into an engine core request.

        Converts a prompt (text, tokens, or multimodal) into an
        OmniEngineCoreRequest that can be processed by the engine.
        Handles prompt embeddings and additional information payloads
        for direct transfer between stages.

        Args:
            request_id: Unique identifier for this request
            prompt: Input prompt (text, token IDs, embeddings, or multimodal)
            params: Sampling or pooling parameters for generation
            supported_tasks: Tuple of supported tasks for validation
            arrival_time: Optional arrival timestamp (defaults to current time)
            lora_request: Optional LoRA adapter request
            tokenization_kwargs: Optional additional tokenization arguments
            trace_headers: Optional tracing headers for observability
            priority: Request priority (higher values processed first)
            data_parallel_rank: Optional data parallel rank for distributed
                inference
            resumable: Whether the request supports streaming input

        Returns:
            OmniEngineCoreRequest ready for the engine

        Raises:
            ValueError: If data_parallel_rank is out of range or prompt_embeds
                has incorrect shape
        """
        self._validate_params(params, supported_tasks)
        self._validate_lora(lora_request)

        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise ValueError(f"data_parallel_rank {data_parallel_rank} is out of range [0, {num_ranks}).")

        # Short-circuit for prompts already processed by the renderer
        # (they carry a "type" key).  Raw prompts must still go through the
        # omni preprocessor which preserves additional_information, etc.
        if isinstance(prompt, dict) and "type" in prompt:
            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())
            processed_inputs: ProcessorInputs = prompt  # type: ignore[assignment]
        else:
            if arrival_time is None:
                arrival_time = time.time()

            processed_inputs = self.input_preprocessor.preprocess(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
            )

        self._platform_validate_request(processed_inputs, params)

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)
        self._validate_model_inputs(encoder_inputs, decoder_inputs)

        # Normalize decoder prompt access across TypedDict variants.
        if decoder_inputs["type"] == "embeds":
            prompt_token_ids = None
            prompt_embeds = decoder_inputs["prompt_embeds"]
        else:
            prompt_token_ids = decoder_inputs["prompt_token_ids"]
            prompt_embeds = decoder_inputs.get("prompt_embeds")

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds)
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len
            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id(),
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            decoder_mm_hashes = decoder_inputs["mm_hashes"]

            if not all(isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)):
                raise ValueError(
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "
                    "This is likely due to an incorrect custom implementation of "
                    "MultiModalProcessor.apply method."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(base_mm_hash, lora_request),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        # Compatibility: decode serialized prompt embeds if provided.
        if isinstance(prompt_embeds, PromptEmbedsPayload):
            prompt_embeds = self._decode_prompt_embeds(prompt_embeds)

        additional_information_payload: AdditionalInformationPayload | None = None
        raw_info: dict[str, Any] | AdditionalInformationPayload | None = decoder_inputs.get("additional_information")
        if isinstance(raw_info, AdditionalInformationPayload):
            additional_information_payload = raw_info
        elif raw_info is not None:
            entries: dict[str, AdditionalInformationEntry] = {}
            for key, value in raw_info.items():
                if isinstance(value, torch.Tensor):
                    v_cpu = value.detach().to("cpu").contiguous()
                    dtype_str = self._dtype_to_name(v_cpu.dtype)
                    data_bytes = v_cpu.numpy().tobytes()
                    entry = AdditionalInformationEntry(
                        tensor_data=data_bytes,
                        tensor_shape=[int(x) for x in list(v_cpu.shape)],
                        tensor_dtype=dtype_str,
                    )
                elif isinstance(value, list):
                    entry = AdditionalInformationEntry(list_data=value)
                else:
                    raise ValueError("additional_information values must be Tensor or list")
                entries[key] = entry
            additional_information_payload = AdditionalInformationPayload(entries=entries)

        return OmniEngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            prompt_embeds=prompt_embeds,
            additional_information=additional_information_payload,
            resumable=resumable,
        )

    @staticmethod
    def _decode_prompt_embeds(payload: PromptEmbedsPayload) -> torch.Tensor:
        dtype = getattr(np, payload.dtype)
        arr = np.frombuffer(payload.data, dtype=dtype)
        arr = arr.reshape(payload.shape)
        return torch.from_numpy(arr)
