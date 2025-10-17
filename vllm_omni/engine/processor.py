import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

from vllm.inputs import ProcessorInputs, PromptType
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.utils import merge_and_sort_multimodal_metadata
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.config import VllmConfig
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm_omni.inputs.preprocess import OmniInputPreprocessor

from vllm.v1.engine.processor import Processor
from vllm_omni.engine import PromptEmbedsPayload, AdditionalInformationPayload, AdditionalInformationEntry, OmniEngineCoreRequest


class OmniProcessor(Processor):
    def __init__(self, 
                 vllm_config: VllmConfig,
                 tokenizer: TokenizerGroup,
                 mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
                 ):
        super().__init__(vllm_config, tokenizer, mm_registry)
        self.input_preprocessor = OmniInputPreprocessor(self.model_config,
                                                    self.tokenizer,
                                                    mm_registry)

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> tuple[Optional[str], OmniEngineCoreRequest]:

        # TODO(woosuk): Support pooling models.
        # TODO(woosuk): Support encoder-decoder models.
        self._validate_lora(lora_request)
        self._validate_params(params, lora_request)
        if trace_headers is not None:
            raise ValueError("V1 does not support tracing yet.")

        data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank <
                                                   data_parallel_size):
            raise ValueError(f"data_parallel_rank {data_parallel_rank} "
                             f"is out of range [0, {data_parallel_size}).")

        if arrival_time is None:
            arrival_time = time.time()

        # Process inputs, which includes:
        # 1. Tokenize text prompt, with LoRA request if one exists.
        # 2. For multimodal models with a merged preprocessor, preprocess
        #   multimodal data and expand prompt token ids accordingly.
        return_mm_hashes = (self.model_config.processor_return_mm_hashes
                            or bool(self.cache_config.enable_prefix_caching))
        processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            lora_request=lora_request,
            return_mm_hashes=return_mm_hashes,
        )
        from vllm.platforms import current_platform
        current_platform.validate_request(
            prompt=prompt,
            params=params,
            processed_inputs=processed_inputs,
        )
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        self._validate_model_inputs(processed_inputs, lora_request)

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)

        # TODO: Impl encoder-decoder
        if encoder_inputs is not None:
            raise NotImplementedError

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                sampling_params.max_tokens = (
                    self.model_config.max_model_len -
                    len(decoder_inputs["prompt_token_ids"]))
            sampling_params.update_from_generation_config(
                self.generation_config_fields, eos_token_id)
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(
                    self.tokenizer.get_lora_tokenizer(lora_request))
        else:
            pooling_params = params.clone()

        # Multimodal related.
        sorted_mm_inputs: Optional[Sequence[Optional[MultiModalKwargs]]] = None
        sorted_mm_positions: Optional[list[PlaceholderRange]] = None
        sorted_mm_hashes: Optional[list[str]] = None
        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            (
                sorted_item_modalities,
                sorted_mm_positions,
                sorted_mm_hashes,
            ) = merge_and_sort_multimodal_metadata(
                decoder_inputs["mm_placeholders"],
                decoder_inputs["mm_hashes"] if return_mm_hashes else None,
            )

            # The output of merged multi-modal processor (`decoder_mm_inputs`)
            # is a single MultiModalKwargs for all items from all modalities.
            # This code flattens kwargs for individual items in a list and
            # sorts them by each item's position in the input sequence if there
            # are multiple modalities.
            unique_modalities = set(sorted_item_modalities)
            if len(unique_modalities) > 1:
                orig_sorted_mm_inputs = []
                used_indices = {modality: 0 for modality in unique_modalities}

                for modality in sorted_item_modalities:
                    items = decoder_mm_inputs.get_items(modality)
                    item = items[used_indices[modality]]

                    orig_sorted_mm_inputs.append(
                        MultiModalKwargs.from_items([item]))
                    used_indices[modality] += 1
            else:
                orig_sorted_mm_inputs = [
                    MultiModalKwargs.from_items([item]) for item in
                    decoder_mm_inputs.get_items(sorted_item_modalities[0])
                ]

            if sorted_mm_hashes is not None:
                sorted_mm_inputs = self.mm_input_cache_client.get_and_update(
                    orig_sorted_mm_inputs, sorted_mm_hashes)
            else:
                sorted_mm_inputs = orig_sorted_mm_inputs
        
        # Serialize prompt_embeds and additional_information if provided (direct-transfer path)
        prompt_embeds_payload: Optional[PromptEmbedsPayload] = None
        additional_information_payload: Optional[AdditionalInformationPayload] = None
        if "prompt_embeds" in decoder_inputs:  # type: ignore[operator]
            import numpy as np
            import torch
            pe: torch.Tensor = decoder_inputs["prompt_embeds"]  # type: ignore[index]
            if pe.ndim != 2:
                raise ValueError(
                    "prompt_embeds must be of shape (seq_len, hidden_size)")
            # Move to CPU and ensure contiguous memory for stable serialization
            pe_cpu = pe.detach().to("cpu").contiguous()
            seq_len, hidden_size = pe_cpu.shape
            dtype_str = str(pe_cpu.dtype).replace("torch.", "")
            data_bytes = pe_cpu.numpy().tobytes()
            prompt_embeds_payload = PromptEmbedsPayload(
                data=data_bytes,
                shape=[int(seq_len), int(hidden_size)],
                dtype=dtype_str,
            )
        if "additional_information" in decoder_inputs:  # type: ignore[operator]
            import numpy as np
            import torch
            raw_info: dict[str, Any] = decoder_inputs["additional_information"]  # type: ignore[index]
            entries: dict[str, AdditionalInformationEntry] = {}
            for key, value in raw_info.items():
                if isinstance(value, torch.Tensor):
                    v_cpu = value.detach().to("cpu").contiguous()
                    dtype_str = str(v_cpu.dtype).replace("torch.", "")
                    data_bytes = v_cpu.numpy().tobytes()
                    entry = AdditionalInformationEntry(
                        tensor_data=data_bytes,
                        tensor_shape=[int(x) for x in list(v_cpu.shape)],
                        tensor_dtype=dtype_str,
                    )
                elif isinstance(value, list):
                    entry = AdditionalInformationEntry(list_data=value)
                else:
                    raise ValueError(
                        "additional_information values must be Tensor or list")
                entries[key] = entry
            additional_information_payload = AdditionalInformationPayload(
                entries=entries)

        return decoder_inputs.get("prompt"), OmniEngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=decoder_inputs["prompt_token_ids"],
            mm_inputs=sorted_mm_inputs,
            mm_hashes=sorted_mm_hashes,
            mm_placeholders=sorted_mm_positions,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            prompt_embeds=prompt_embeds_payload,
            additional_information=additional_information_payload,
        )