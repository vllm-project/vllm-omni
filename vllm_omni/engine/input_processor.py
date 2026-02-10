import os
import time
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.inputs import ProcessorInputs, PromptType
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalUUIDDict

try:
    from vllm.multimodal.processing import set_request_id
except ImportError:  # vllm without set_request_id (older releases)
    from contextlib import contextmanager

    @contextmanager
    def set_request_id(_request_id: str):
        yield


from vllm.multimodal.utils import argsort_mm_positions
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.torch_utils import set_default_torch_num_threads
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
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        super().__init__(vllm_config, mm_registry)
        self.input_preprocessor = OmniInputPreprocessor(
            self.model_config,
            vllm_config.observability_config,
            mm_registry,
            mm_processor_cache=self.mm_processor_cache,
        )

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: SamplingParams | PoolingParams,
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
            arrival_time: Optional arrival timestamp (defaults to current time)
            lora_request: Optional LoRA adapter request
            tokenization_kwargs: Optional additional tokenization arguments
            trace_headers: Optional tracing headers for observability
            priority: Request priority (higher values processed first)
            data_parallel_rank: Optional data parallel rank for distributed
                inference

        Returns:
            Tuple of (prompt_string, OmniEngineCoreRequest) where:
                - prompt_string: The original prompt as a string, or None if
                  using embeddings
                - OmniEngineCoreRequest: Processed request ready for the engine

        Raises:
            ValueError: If data_parallel_rank is out of range or prompt_embeds
                has incorrect shape
        """
        self._validate_lora(lora_request)
        self._validate_params(params)

        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise ValueError(f"data_parallel_rank {data_parallel_rank} is out of range [0, {num_ranks}).")

        if arrival_time is None:
            arrival_time = time.time()

        # Optionally generate multimodal hash overrides to avoid hashing
        # multimodal data items by their content as their identifiers.

        # NOTE: when users explicitly turn off BOTH prefix caching and input
        # processing caching, no multimodal features or embeddings will be
        # reused across requests, therefore identifying multimodal data items
        # by their content is no longer necessary, and we create uuids with
        # request id-modality-index as multimodal hash overrides.
        if (
            self.model_config.multimodal_config
            and self.model_config.multimodal_config.mm_processor_cache_gb == 0
            and not self.cache_config.enable_prefix_caching
        ):
            mm_uuids = self._maybe_build_mm_uuids(request_id, prompt)
        else:
            # Otherwise, use user-provided uuids as multimodal hash overrides
            # if provided.
            self._validate_mm_uuids(prompt)
            if isinstance(prompt, dict):
                mm_uuids = cast(MultiModalUUIDDict | None, prompt.get("multi_modal_uuids"))
            else:
                mm_uuids = None

        # Process inputs, which includes:
        # 1. Tokenize text prompt, with LoRA request if one exists.
        # 2. For multimodal models with a merged preprocessor, preprocess
        #   multimodal data and expand prompt token ids accordingly.
        num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        if "OMP_NUM_THREADS" not in os.environ:
            logger.debug_once(
                "OMP_NUM_THREADS is not set; defaulting Torch threads to %d for input preprocessing.",
                num_threads,
            )

        with set_request_id(request_id), set_default_torch_num_threads(num_threads):
            processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        current_platform.validate_request(
            prompt=prompt,
            params=params,
            processed_inputs=processed_inputs,
        )

        eos_token_id = self.input_preprocessor.get_eos_token_id()

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
            sampling_params.update_from_generation_config(self.generation_config_fields, eos_token_id)
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
            eos_token_id=eos_token_id,
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
