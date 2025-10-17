from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import itertools

import torch

from vllm.worker.model_runner import ModelInputForGPU, ModelInputForGPUBuilder

from vllm.utils import flatten_2d_lists, async_tensor_h2d
from vllm.lora.layers import LoRAMapping
from vllm.sequence import MultiModalKwargs


@dataclass(frozen=True)
class OmniModelInputForGPU(ModelInputForGPU):
    # New: optional additional information dict for current scheduled tokens
    additional_information: Optional[Dict[str, Any]] = None


class OmniModelInputForGPUBuilder(ModelInputForGPUBuilder):
    def build(self) -> OmniModelInputForGPU:
        """Finalize the builder intermediate data and
        create on-device tensors.
        """
        # Combine and flatten intermediate data.
        input_tokens = list[int]()
        inputs_embeds_list = list[torch.Tensor]()
        token_types = list[int]()
        for inter_data in self.inter_data_list:
            for cur_input_tokens in inter_data.input_tokens:
                input_tokens.extend(cur_input_tokens)
            for cur_token_types in inter_data.token_types:
                token_types.extend(cur_token_types)
            if inter_data.inputs_embeds is not None:
                inputs_embeds_list.append(
                    inter_data.inputs_embeds.to(
                        dtype=self.runner.model_config.dtype,
                        device=self.runner.device))
            # Support v1 direct-transfer prompt embeds attached on request state
            if inter_data.inputs_embeds is None:
                try:
                    # Locate req_id -> request state to fetch decoded CPU embeds
                    req_id = inter_data.request_id  # type: ignore[attr-defined]
                    req_state = getattr(self.runner, "requests", {}).get(req_id)
                    pe_cpu = getattr(req_state, "prompt_embeds_cpu", None)
                    if pe_cpu is not None:
                        inter_data.inputs_embeds = pe_cpu.to(
                            dtype=self.runner.model_config.dtype,
                            device=self.runner.device)
                        inputs_embeds_list.append(inter_data.inputs_embeds)
                except Exception:
                    pass
        inputs_embeds: Optional[torch.Tensor]
        if len(inputs_embeds_list) == 0:
            inputs_embeds = None
        else:
            inputs_embeds = torch.cat(inputs_embeds_list, dim=0).to(
                dtype=self.runner.model_config.dtype,
                device=self.runner.device)
            assert len(inputs_embeds) == len(input_tokens)

        if not input_tokens and inputs_embeds is None:
            # This may happen when all prefill requests hit
            # prefix caching and there is no decode request.
            return self.model_input_cls()

        mrope_input_positions: Optional[List[List[int]]] = None
        if any(inter_data.mrope_input_positions is not None
               for inter_data in self.inter_data_list):
            mrope_input_positions = [[] for _ in range(3)]
            for idx in range(3):
                for inter_data in self.inter_data_list:
                    msections = inter_data.mrope_input_positions
                    if msections is None:
                        for _seq_input_positions in inter_data.input_positions:
                            mrope_input_positions[idx].extend(
                                _seq_input_positions)
                    else:
                        for _seq_mrope_input_positions in msections:
                            mrope_input_positions[idx].extend(
                                _seq_mrope_input_positions[idx])
            input_positions = None
        else:
            input_positions = []
            for inter_data in self.inter_data_list:
                for cur_input_positions in inter_data.input_positions:
                    input_positions.extend(cur_input_positions)

        seq_lens = []
        query_lens = []
        max_decode_seq_len = 0
        max_encoder_seq_len = 0
        for inter_data in self.inter_data_list:
            seq_lens.extend(inter_data.seq_lens)
            query_lens.extend(inter_data.query_lens)
            if not inter_data.is_prompt:
                max_decode_seq_len = max(max_decode_seq_len,
                                         max(inter_data.seq_lens))
                if self.runner.model_config.is_encoder_decoder:
                    max_encoder_seq_len = max(max_encoder_seq_len,
                                              inter_data.encoder_seq_len)

        # Mapping from request IDs to sequence IDs. Used for Jamba models
        # that manages the cache by itself.
        request_ids_to_seq_ids = {
            data.request_id: data.seq_ids
            for data in self.inter_data_list
        }

        cuda_graph_pad_size = self._get_cuda_graph_pad_size(
            num_seqs=len(seq_lens),
            max_decode_seq_len=max_decode_seq_len,
            max_encoder_seq_len=max_encoder_seq_len)

        batch_size = len(input_tokens)
        if cuda_graph_pad_size != -1:
            # If cuda graph can be used, pad tensors accordingly.
            # See `capture_model` API for more details.
            # vLLM uses cuda graph only for decoding requests.
            batch_size += cuda_graph_pad_size

        # Tokens and positions.
        if cuda_graph_pad_size:
            input_tokens.extend(itertools.repeat(0, cuda_graph_pad_size))
        assert self.runner.device is not None
        input_tokens_tensor = async_tensor_h2d(input_tokens, torch.long,
                                               self.runner.device,
                                               self.runner.pin_memory)

        token_types_tensor = async_tensor_h2d(token_types, torch.long,
                                               self.runner.device,
                                               self.runner.pin_memory) \
                                                if token_types else None

        if mrope_input_positions is not None:
            for idx in range(3):
                mrope_input_positions[idx].extend(
                    itertools.repeat(0, cuda_graph_pad_size))
            input_positions_tensor = async_tensor_h2d(mrope_input_positions,
                                                      torch.long,
                                                      self.runner.device,
                                                      self.runner.pin_memory)
        else:
            input_positions.extend(itertools.repeat(0, cuda_graph_pad_size))
            input_positions_tensor = async_tensor_h2d(input_positions,
                                                      torch.long,
                                                      self.runner.device,
                                                      self.runner.pin_memory)
        # Sequence and query lengths.
        if cuda_graph_pad_size:
            seq_lens.extend(itertools.repeat(1, cuda_graph_pad_size))

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(
            seq_lens, query_lens, cuda_graph_pad_size, batch_size)

        # LoRA data.
        lora_requests = set()
        lora_mapping = None
        if self.enable_lora:
            lora_requests = set(r for data in self.inter_data_list
                                for r in data.lora_requests)
            lora_index_mapping = flatten_2d_lists([
                flatten_2d_lists(inter_data.lora_index_mapping)
                for inter_data in self.inter_data_list
            ])
            if cuda_graph_pad_size:
                lora_index_mapping.extend(
                    itertools.repeat(0, cuda_graph_pad_size))
            lora_prompt_mapping = flatten_2d_lists([
                flatten_2d_lists(inter_data.lora_prompt_mapping)
                for inter_data in self.inter_data_list
            ])

            lora_mapping = LoRAMapping(
                **dict(index_mapping=lora_index_mapping,
                       prompt_mapping=lora_prompt_mapping,
                       is_prefill=not self.decode_only))

        # Multi-modal data.
        multi_modal_kwargs_list = [
            data.multi_modal_kwargs for data in self.inter_data_list
            if data.multi_modal_kwargs is not None
        ]
        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return self.model_input_cls(
            input_tokens=input_tokens_tensor,
            inputs_embeds=inputs_embeds,
            input_positions=input_positions_tensor,
            token_types=token_types_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_kwargs=multi_modal_kwargs,
            request_ids_to_seq_ids=request_ids_to_seq_ids,
            finished_requests_ids=self.finished_requests_ids)