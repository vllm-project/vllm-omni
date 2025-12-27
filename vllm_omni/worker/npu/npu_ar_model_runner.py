# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pcp_group, get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (
    update_attn_dcp_pcp_params,
    update_attn_params,
    update_mla_attn_dcp_pcp_params,
    update_mla_attn_params,
)
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.utils import ProfileExecuteDuration, enable_sp, lmhead_tp_enable

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.npu.npu_model_runner import OmniNPUModelRunner


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    kv_connector_output: KVConnectorOutput | None
    attn_metadata: dict[str, Any]
    positions: torch.Tensor
    multimodal_outputs: Any

class NPUARModelRunner(OmniNPUModelRunner):
    """Autoregressive NPU model runner that returns hidden states per request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)

    def _make_buffer(self, *size, dtype, numpy=True):
        # Prevent ray from pinning the buffer due to large size
        from vllm_omni.distributed.ray_utils.utils import (
            calculate_total_bytes,
            maybe_disable_pin_memory_for_ray,
        )

        total_bytes = calculate_total_bytes(size, dtype)

        # Use the context manager to temporarily disable pinning if needed
        with maybe_disable_pin_memory_for_ray(self, total_bytes):
            return super()._make_buffer(*size, dtype=dtype, numpy=numpy)

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        dict[str, Any],
        torch.Tensor,
        np.ndarray,
        int,
        torch.Tensor,
        int,
        torch.Tensor,
        SpecDecodeMetadata,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        int
    ]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)

        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        _, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        positions_np = np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
        )

        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)
        if self.pcp_size > 1:
            if not self.vllm_config.model_config.use_mla:
                self.generate_kv_idx(scheduler_output)
            tokens, position_pcp, pcp_unpad_mask = self._update_tokens_for_pcp(
                tokens)
            num_scheduled_tokens = np.array(tokens, dtype=np.int32)
            total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
        else:
            position_pcp, pcp_unpad_mask = None, None
            self.num_pcp_pads = self.num_pcp_pads[:num_reqs]

        total_num_pcp_pads = sum(self.num_pcp_pads)
        max_num_scheduled_tokens = max(tokens)
        num_valid_tokens = np.array([
            num_tokens -
            len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
            for num_tokens, i in zip(tokens, req_ids)
        ],
                                    dtype=np.int32)

        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.cudagraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        elif self.use_aclgraph and enable_sp(self.vllm_config):
            # When using aclgraph, if total_num_scheduled_tokens exceeds the maximum graph size,
            # the model will fall back to running its FX graph in eager mode.
            # In this case, when sequence parallelism is enabled, we need to pad tokens to align
            # with tp_size because pad_size cannot be captured by the FX graph
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_input_tokens = math.ceil(
                total_num_scheduled_tokens / tp_size) * tp_size
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp,
         with_prefill) = self._sync_metadata_across_dp(num_input_tokens,
                                                       with_prefill)

        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        num_input_tokens = maybe_padded_num_tokens

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        if self.pcp_size > 1:
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
                   position_pcp[:total_num_scheduled_tokens],
                   out=positions_np)
        else:
            self.positions.np[:total_num_scheduled_tokens] = positions_np

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           token_indices_tensor,
                           out=self.input_ids.cpu[:total_num_scheduled_tokens])
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens])

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or
                                                   self.enable_prompt_embeds):
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[output_idx:output_idx +
                                           actual_num_sched].copy_(
                                               req_embeds[start_pos:actual_end]
                                           )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc.copy_to_gpu()

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        self.seq_lens.copy_to_gpu()

        # Fill unused with -1. Needed for reshape_and_cache
        self.query_start_loc.gpu[num_reqs + 1:].fill_(-1)
        self.seq_lens.gpu[num_reqs:].fill_(0)

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens,
                                cu_num_tokens)
        self.positions.cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions.copy_to_gpu()

        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_mask = self._make_attention_mask(attn_state)
        self.attn_state = attn_state  # type: ignore

        self.with_prefill = with_prefill
        self.num_tokens_across_dp = num_tokens_across_dp
        attn_metadata: dict[str, Any] = {}

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [
            self.requests[r].num_tokens for r in self.input_batch.req_ids
        ]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        num_reqs = self.input_batch.num_reqs
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [
                scheduler_output.num_scheduled_tokens[i] for i in req_ids
            ]
            original_seq_lens_np = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                np.array(tokens_original, dtype=np.int32))
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[:self.num_discarded_requests] = (
            discard_request_indices)
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
            ):
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)

                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids.gpu[:total_num_scheduled_tokens]
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(
                    scheduler_output)

            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            #  -------------------------------------- Omni-new -------------------------------------------------
            input_ids = self.input_ids.gpu[:num_input_tokens]
            #  -------------------------------------- Omni-new -------------------------------------------------
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the acl graph all the time. The v0
            # engine avoids this by "double compiling" the acl graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the acl graph will be more performant (like in the else case
            # below).
            token_ids_idx = self.is_token_ids.gpu[:total_num_scheduled_tokens] \
                .nonzero(as_tuple=False) \
                .squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(
                    input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
        positions = self.positions.gpu[:num_input_tokens]
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]

        #  -------------------------------------- Omni-new -------------------------------------------------
        self._omni_num_scheduled_tokens_np = num_scheduled_tokens

        # Note: only prefill need collect additional_information for now.
        # Decode don't need per_req_additional_information anymore.
        if inputs_embeds is not None:
            # Prefill: overlay prompt_embeds and collect additional_information
            self._collect_additional_information_for_prefill(num_scheduled_tokens)

        if hasattr(self.model, "has_preprocess") and self.model.has_preprocess:
            # Overlay custom prompt_embeds per request for the prompt portion;
            # collect additional_information (tensor/list) for prefill portion only
            for req_index, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests.get(req_id)
                req_infos = getattr(req_state, "additional_information_cpu", None) if req_state is not None else None

                start_offset = int(self.query_start_loc.cpu[req_index])
                sched_tokens = int(num_scheduled_tokens[req_index])
                s, e = start_offset, start_offset + sched_tokens
                span_len = int(e) - int(s)

                # call the custom process function
                req_input_ids, req_embeds, update_dict = self.model.preprocess(
                    input_ids=input_ids[s:e], input_embeds=inputs_embeds[s:e], **req_infos
                )
                # TODO(Peiqi): the merge stage could move out from the critical path
                self._merge_additional_information_update(req_id, update_dict)

                # update the inputs_embeds and input_ids
                seg_len = min(span_len, req_embeds.shape[0])
                inputs_embeds[s : s + seg_len] = req_embeds[:seg_len]
                if isinstance(req_input_ids, torch.Tensor) and req_input_ids.numel() == seg_len:
                    input_ids[s : s + seg_len] = req_input_ids

        #  -------------------------------------- Omni-new -------------------------------------------------


        # type: ignore
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            # If both flashcomm1 and pp are used simultaneously,
            # the shape of the received data and the shape of the space to be copied to will not match,
            # requiring a recalculation of the incoming data's shape.
            tp_size = get_tensor_model_parallel_world_size()
            num_input_tokens_with_flashcomm1 = num_input_tokens
            if enable_sp():
                num_input_tokens_with_flashcomm1 = (num_input_tokens +
                                                    tp_size - 1) // tp_size
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[
                    k][:num_input_tokens_with_flashcomm1].copy_(
                        v[:num_input_tokens_with_flashcomm1],
                        non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k:
                v[:num_input_tokens_with_flashcomm1]
                for k, v in self.intermediate_tensors.items()
            })

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            if self.pcp_size * self.dcp_size > 1:
                logits_indices = torch.from_numpy(
                    cu_num_tokens
                ) * self.pcp_size - self.num_pcp_pads[:num_reqs] - 1
                logits_indices = logits_indices.pin_memory().to(
                    self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1:num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (len(draft_token_ids) if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]) else -1)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens, self.num_pcp_pads[:num_reqs])
            logits_indices = spec_decode_metadata.logits_indices

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:
                                            num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Used in the below loop.
        # query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        num_computed_tokens_cpu = (
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
        self.spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        if self.speculative_config and self.pcp_size > 1:
            self._generate_pcp_mtp_input(
                num_reqs, scheduler_output.total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens)

        long_seq_metadata = self._generate_pcp_metadata(
            total_num_scheduled_tokens)
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            # NOTE: This is strange, why did we use total_num_scheduled_tokens before?
            slot_mapping_size = (total_num_scheduled_tokens
                                 if self.pcp_size == 1 else
                                 total_num_scheduled_tokens * self.pcp_size -
                                 total_num_pcp_pads)
            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                blk_table.slot_mapping.gpu[slot_mapping_size:].fill_(0)
                if self.pcp_size > 1:
                    slot_mapping_for_pcp = blk_table.slot_mapping.gpu[:
                                                                      long_seq_metadata
                                                                      .
                                                                      num_actual_tokens_pcp_padded]
                    slot_mapping_for_pcp[slot_mapping_size:].fill_(-1)
                    assert pcp_unpad_mask is not None
                    pcp_padded_slot_mapping = self.pcp_padded_slot_mapping[:
                                                                           pcp_unpad_mask
                                                                           .
                                                                           shape[
                                                                               0]]
                    pcp_padded_slot_mapping.fill_(-1)
                    pcp_padded_slot_mapping[
                        pcp_unpad_mask] = slot_mapping_for_pcp[:
                                                               slot_mapping_size]
                    slot_mapping_for_pcp[:long_seq_metadata.
                                         num_actual_tokens_pcp_padded] = pcp_padded_slot_mapping
                    blk_table.slot_mapping.gpu[:long_seq_metadata.num_actual_tokens_pcp_padded] = \
                        slot_mapping_for_pcp
                slot_mapping = blk_table.slot_mapping.gpu

            # NOTE: This is a temporary hack, now in GPUModelRunner, this prepare_inputs
            # has been split to multiple parts, and there are 3 parts that is related to this
            # `num_reqs`, we'll take `query_start_loc` as an example:
            # 1. self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            # 2. get `num_reqs_padded`, this depends on dispatcher and which is why we have the
            #    following simplified `dispatch` logic here, we try to minimize the impact
            # 3. query_start_loc = self.query_start_loc.gpu[: num_reqs_padded + 1]
            uniform_decode = (max_num_scheduled_tokens == self.uniform_decode_query_len) \
                and (total_num_scheduled_tokens == max_num_scheduled_tokens * num_reqs)

            # TODO: We should make this official ASAP. Also note that if we pad here,
            # the builders won’t need to add any extra padding.
            if self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                uniform_decode:
                num_reqs_padded = num_input_tokens // self.uniform_decode_query_len
                pad_size = num_reqs_padded - num_reqs
                if pad_size > 0:
                    last_query_loc = self.query_start_loc.gpu[num_reqs]

                    steps = torch.arange(1,
                                         pad_size + 1,
                                         device=self.device,
                                         dtype=self.query_start_loc.gpu.dtype)
                    fill_values = last_query_loc + (
                        steps * self.uniform_decode_query_len)

                    self.query_start_loc.gpu[num_reqs + 1:num_reqs_padded +
                                             1] = fill_values
                # So we are trying to simulate the behavior of GPUModelRunner's
                # prepare_inputs for uniform decode mode by padding query_start_loc
                num_reqs = num_reqs_padded

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs + 1],
                seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                seq_lens=self.seq_lens.gpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=slot_mapping_size,
                num_input_tokens=num_input_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                # TODO: change this to the right block table for linear attn
                block_table_tensor=blk_table_tensor[:num_reqs],
                slot_mapping=slot_mapping,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                positions=self.positions.gpu,
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                attn_state=self.attn_state,
                is_only_prefill=bool(np.all(num_valid_tokens != 1)),
                max_query_len=max_num_scheduled_tokens,
                decode_token_per_req=self.decode_token_per_req,
                prefill_context_parallel_metadata=long_seq_metadata,
            )

            if self.speculative_config and self.pcp_size > 1:
                # For pcp + spec decode, we flatten block_table
                # to avoid irregular spec_attn_mask shape, e.g.,
                # num_decode_req=2, num_prefill_req=3, num_speculative_tokens=1,
                # ori block_table: # [d0, d1, p0, p1, p2]
                # (num_reqs_d + num_reqs_p, max_num_blocks),
                # flattened block_table: [d0, d0, d1, d1, p0, p1, p2]
                # (num_reqs_d * decode_threshold + num_reqs_p, max_num_blocks),
                ori_query_lens = self.query_start_loc_pcp_full.cpu[1:num_reqs+1] - \
                    self.query_start_loc_pcp_full.cpu[:num_reqs]
                num_prefill_reqs = (ori_query_lens
                                    > self.decode_threshold).sum().item()
                num_decode_reqs = num_reqs - num_prefill_reqs
                num_decode_reqs_flatten = num_decode_reqs * self.decode_threshold
                blk_table_tensor[
                    num_decode_reqs_flatten:num_decode_reqs_flatten +
                    num_prefill_reqs].copy_(
                        blk_table_tensor[num_decode_reqs:num_decode_reqs +
                                         num_prefill_reqs].clone())
                blk_table_tensor[:num_decode_reqs_flatten].copy_(
                    blk_table_tensor[:num_decode_reqs].repeat_interleave(
                        self.decode_threshold, dim=0))
                common_attn_metadata.block_table_tensor = \
                    blk_table_tensor[:num_decode_reqs_flatten + num_prefill_reqs]

            if self.speculative_config and \
                self.spec_decode_common_attn_metadata is None:
                self.spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    if use_spec_decode:
                        patch_torch_npu_argsort()
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.
                            gpu[:num_reqs],
                            num_decode_draft_tokens_cpu=self.
                            num_decode_draft_tokens.cpu[:num_reqs],
                        )
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)
                elif self.model_config.runner_type == "pooling":
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)
                else:
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        model=self.get_model(),
                        **extra_attn_metadata_args)

                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(
                logits_indices,
                (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (attn_metadata, positions, num_scheduled_tokens,
                num_input_tokens, num_tokens_across_dp,
                maybe_padded_num_tokens, logits_indices, spec_decode_metadata,
                input_ids, inputs_embeds, intermediate_tensors,
                max_num_scheduled_tokens)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")

        with ProfileExecuteDuration().capture_async("prepare input"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            self._update_states(scheduler_output)
            self._decode_and_store_request_payloads(scheduler_output)
            #  ------------------------------------------------------------------------------------------------

            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                ):
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(
                        scheduler_output)

            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOutput if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)

            if self.dynamic_eplb:
                self.eplb_updator.forward_before()

            (attn_metadata, positions, num_scheduled_tokens_np,
             num_input_tokens, num_tokens_across_dp, maybe_padded_num_tokens,
             logits_indices, spec_decode_metadata, input_ids, inputs_embeds,
             intermediate_tensors,
             max_query_len) = (self._prepare_inputs(scheduler_output,
                                                    intermediate_tensors))

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        moe_comm_type = self._select_moe_comm_method(num_input_tokens)
        # prevent debugger is None
        need_dump = self.dump_enable and self.debugger is not None
        if need_dump:
            assert self.debugger is not None
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(
                getattr(dbg_cfg, "level",
                        "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            scheduler_output.total_num_scheduled_tokens
            == self.input_batch.num_reqs * max_query_len)
        has_lora = len(self.input_batch.lora_id_to_lora_request) > 0
        aclgraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(
                num_tokens=num_input_tokens, uniform_decode=uniform_decode, has_lora=has_lora)

        # Run forward pass
        with ProfileExecuteDuration().capture_async("forward"):
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=self.with_prefill,
                    moe_comm_type=moe_comm_type,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    num_actual_tokens=scheduler_output.
                    total_num_scheduled_tokens,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model,
                    weight_prefetch_method=self.weight_prefetch_method):
                self.maybe_setup_kv_connector(scheduler_output)

                hidden_states = self._generate_process_reqs_hidden_states(
                    maybe_padded_num_tokens, input_ids, positions,
                    intermediate_tensors, inputs_embeds)

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output)

            aux_hidden_states = None
            if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                hidden_states, aux_hidden_states = hidden_states

        kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving)
        finished_sending = None
        finished_recving = None
        with ProfileExecuteDuration().capture_async("post process"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            multimodal_outputs = hidden_states.multimodal_outputs
            hidden_states = hidden_states.text_hidden_states

            if multimodal_outputs is not None:
                keys_or_type = (
                    list(multimodal_outputs.keys())
                    if isinstance(multimodal_outputs, dict)
                    else type(multimodal_outputs)
                )
                logger.debug(f"[AR] execute_model: multimodal_outputs keys = {keys_or_type}")
            else:
                logger.debug("[AR] execute_model: multimodal_outputs is None")
            #  -------------------------------------- Omni-new -------------------------------------------------
            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = \
                self.parallel_config.distributed_executor_backend \
                == "external_launcher" and len(get_pp_group().ranks) > 0
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    hidden_states.kv_connector_output = kv_connector_output
                    if need_dump:
                        assert self.debugger is not None
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    pool_output = self._pool(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np)
                    if need_dump:
                        assert self.debugger is not None
                        self.debugger.stop()
                        self.debugger.step()
                    return pool_output
                # Sometimes, after the model is compiled through the AOT backend,
                # the model output may become a list containing only one Tensor object.
                if isinstance(hidden_states, list) and \
                        len(hidden_states) == 1 and \
                        isinstance(hidden_states[0], torch.Tensor):
                    hidden_states = hidden_states[0]
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            if broadcast_pp_output:
                model_output_broadcast_data = {
                    "logits": logits.contiguous(),
                } if logits is not None else {}
                model_output_broadcast_data = get_pp_group(
                ).broadcast_tensor_dict(model_output_broadcast_data,
                                        src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                kv_connector_output,
                attn_metadata,
                positions,
                multimodal_outputs # Omni-new
            )
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            return None  # noqa
        need_dump = self.dump_enable and self.debugger is not None
        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,  # noqa
            kv_connector_output,
            attn_metadata,
            positions,
            multimodal_outputs, # Omni-new
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            logits = self.apply_grammar_bitmask(scheduler_output,
                                                grammar_output, logits)

        with ProfileExecuteDuration().capture_async("Sample"):
            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if spec_decode_metadata is None:
                if lmhead_tp_enable() and logits is not None:
                    logits = logits[:self.input_batch.num_reqs]
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                if lmhead_tp_enable() and logits is not None:
                    logits = logits[:len(spec_decode_metadata.logits_indices)]
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                assert logits is not None
                bonus_logits = logits[
                    spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                target_logits = logits[
                    spec_decode_metadata.target_logits_indices]
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    None,  # draft_probs
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
                sampler_output.sampled_token_ids = output_token_ids
                if self.need_accepted_tokens:
                    self._update_states_after_model_execute(output_token_ids)
            discard_sampled_tokens_req_indices = \
                self.discard_request_indices.np[:self.num_discarded_requests]
            for i in discard_sampled_tokens_req_indices:
                generator = self.input_batch.generators.get(int(i))
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)

            # Copy some objects so they don't get modified after returning.
            # This is important when using async scheduling.
            req_ids_output_copy = self.input_batch.req_ids.copy()
            req_id_to_index_output_copy = \
                self.input_batch.req_id_to_index.copy()

            # NOTE: NPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = logprobs_tensors.tolists() \
                if logprobs_tensors is not None else None

            # Compute prompt logprobs if needed.
            prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                hidden_states[:scheduler_output.total_num_scheduled_tokens],
                scheduler_output,
            )

            num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
            sampled_token_ids = sampler_output.sampled_token_ids

            if not self.use_async_scheduling:
                # Get the valid generated tokens.
                max_gen_len = sampled_token_ids.shape[-1]
                if max_gen_len == 1:
                    # No spec decode tokens. It's a tensor.
                    valid_sampled_token_ids = sampled_token_ids.tolist()
                else:
                    # Includes spec decode tokens. It's a numpy array
                    valid_sampled_token_ids, _ = self.rejection_sampler.parse_output(
                        sampled_token_ids,
                        self.input_batch.vocab_size,
                    )
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
            else:
                valid_sampled_token_ids = []
                invalid_req_indices = discard_sampled_tokens_req_indices.tolist(
                )
                invalid_req_indices_set = set(invalid_req_indices)
                if self.num_spec_tokens <= 0:
                    assert sampled_token_ids.shape[-1] == 1
                    # Cache the sampled tokens on the NPU and avoid CPU sync.
                    # These will be copied into input_ids in the next step
                    # when preparing inputs.
                    self.input_batch.prev_sampled_token_ids = sampled_token_ids


                self.input_batch.prev_sampled_token_ids_invalid_indices = \
                    invalid_req_indices_set
                self.input_batch.prev_req_id_to_index = {
                    req_id: i
                    for i, req_id in enumerate(self.input_batch.req_ids)
                    if i not in invalid_req_indices_set
                }
            # Cache the sampled tokens in the model runner, so that the scheduler
            # doesn't need to send them back.
            # NOTE(woosuk): As an exception, when using PP, the scheduler sends
            # the sampled tokens back, because there's no direct communication
            # between the first-stage worker and the last-stage worker.
            for req_idx in range(num_sampled_tokens):
                if self.use_async_scheduling:
                    sampled_ids = [-1] * 1 if \
                        req_idx not in invalid_req_indices_set else None
                else:
                    sampled_ids = valid_sampled_token_ids[req_idx]
                if not sampled_ids:
                    continue

                start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                end_idx = start_idx + len(sampled_ids)
                assert end_idx <= self.model_config.max_model_len, (
                    "Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: "
                    f"{self.model_config.max_model_len}")

                self.input_batch.token_ids_cpu[req_idx,
                                               start_idx:end_idx] = sampled_ids
                self.input_batch.is_token_ids[req_idx,
                                              start_idx:end_idx] = True
                self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                self.input_batch.num_tokens[req_idx] = end_idx
                req_id = self.input_batch.req_ids[req_idx]
                req_state = self.requests[req_id]
                req_state.output_token_ids.extend(sampled_ids)

        def propose_draft_token_ids(sampled_token_ids):
            assert self.spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                attn_metadata,
                aux_hidden_states,
            )

        with ProfileExecuteDuration().capture_async("Draft"):
            if self.speculative_config:
                use_padded_batch_for_eagle = self.speculative_config and \
                    self.speculative_config.method == "mtp" and \
                    not self.speculative_config.disable_padded_drafter_batch
                if use_padded_batch_for_eagle:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch_for_eagle:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

        #  -------------------------------------- Omni-new -------------------------------------------------
        hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )

        self._process_additional_information_updates(hidden_states, multimodal_outputs, num_scheduled_tokens_np)

        pooler_output: list[dict[str, object]] = []
        for rid in req_ids_output_copy:
            idx = req_id_to_index_output_copy[rid]
            start = int(self.query_start_loc.cpu[idx])
            sched = int(num_scheduled_tokens_np[idx])
            end = start + sched
            hidden_slice = hidden_states_cpu[start:end]
            payload: dict[str, object] = {"hidden": hidden_slice}
            if isinstance(multimodal_outputs, dict) and multimodal_outputs:
                mm_payload: dict[str, object] = {}
                for k, v in multimodal_outputs.items():
                    try:
                        if isinstance(v, torch.Tensor) and v.shape[0] == hidden_states_cpu.shape[0]:
                            mm_payload[k] = v.detach().to("cpu")[start:end].contiguous()
                        elif isinstance(v, dict):
                            sub_dict: dict[str, torch.Tensor] = {}
                            for sk, sv in v.items():
                                if isinstance(sv, torch.Tensor) and sv.shape[0] == hidden_states_cpu.shape[0]:
                                    sub_dict[str(sk)] = sv.detach().to("cpu")[start:end].contiguous()
                            if sub_dict:
                                mm_payload[k] = sub_dict
                        elif isinstance(v, list):
                            element = v[0]
                            if isinstance(element, torch.Tensor):
                                element = element.detach().to("cpu").contiguous()
                            mm_payload[k] = element
                    except Exception as e:
                        logger.error(f"Error in merge multimodal outputs: {e}")
                if mm_payload:
                    payload.update(mm_payload)
            pooler_output.append(payload)

        model_runner_output = OmniModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=(pooler_output if self.vllm_config.model_config.engine_output_type != "text" else None),
            kv_connector_output=kv_connector_output,
        )
        #  -------------------------------------- Omni-new -------------------------------------------------

        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [
                f"[{tag}]:{duration:.2f}ms"
                for tag, duration in durations.items()
            ]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name,
                        " ".join(dr_str))
        if self.dynamic_eplb:
            self.eplb_updator.forward_end()
        if not self.use_async_scheduling:
            if need_dump:
                assert self.debugger is not None
                self.debugger.stop()
                self.debugger.step()
            return model_runner_output

        if need_dump:
            assert self.debugger is not None
            self.debugger.stop()
            self.debugger.step()
        return AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )

    def _merge_additional_information_update(self, req_id: str, upd: dict) -> None:
        req_state = self.requests.get(req_id)
        if req_state is None:
            return
        existing = getattr(req_state, "additional_information_cpu", {})
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        for k, v in upd.items():
            if isinstance(v, torch.Tensor):
                merged[k] = v.detach().to("cpu").contiguous()
            elif isinstance(v, list):
                merged[k] = [
                    (item.detach().to("cpu").contiguous() if isinstance(item, torch.Tensor) else item) for item in v
                ]
            else:
                merged[k] = v
        setattr(req_state, "additional_information_cpu", merged)

    def _generate_process_reqs_hidden_states(self, maybe_padded_num_tokens,
                                             input_ids, positions,
                                             intermediate_tensors,
                                             inputs_embeds):
        hidden_states = self._model_forward(input_ids=input_ids,
                                   positions=positions,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds,
                                   **self._init_model_kwargs())

        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL \
            and not self.use_sparse:
            # TODO: maybe_padded_num_tokens will be removed, use num_input_tokens instead
            if self.vllm_config.model_config.use_mla:
                if self.pcp_size * self.dcp_size > 1:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_dcp_pcp_params(self.update_stream,
                                                   forward_context,
                                                   maybe_padded_num_tokens)
                else:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_params(self.update_stream, forward_context,
                                           maybe_padded_num_tokens,
                                           self.speculative_config)
            else:
                if self.pcp_size * self.dcp_size > 1:
                    update_attn_dcp_pcp_params(self.update_stream,
                                               forward_context,
                                               maybe_padded_num_tokens)
                else:
                    update_attn_params(self.update_stream, forward_context,
                                       maybe_padded_num_tokens)

        if get_forward_context().sp_enabled and not isinstance(
                hidden_states, IntermediateTensors):
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            pad_size = get_forward_context().pad_size
            if pad_size > 0:
                hidden_states = hidden_states[:-pad_size, :]

        if self.pcp_size > 1:
            hidden_states = get_pcp_group().all_gather(
                hidden_states[:self.num_actual_tokens_pcp_padded //
                              self.pcp_size], 0)
            hidden_states = torch.index_select(
                hidden_states, 0,
                self.pcp_allgather_restore_idx[:hidden_states.shape[0]])
        return hidden_states

    def _process_additional_information_updates(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: object,
        num_scheduled_tokens_np: np.ndarray,
    ) -> None:
        """Process model-provided per-request additional_information updates and merge into request state."""
        try:
            # execute the custom postprocess function
            # TODO(Peiqi): do we have a more elegant way to do this?
            if hasattr(self.model, "has_postprocess") and self.model.has_postprocess:
                for req_index, req_id in enumerate(self.input_batch.req_ids):
                    req_state = self.requests.get(req_id)
                    req_infos = (
                        getattr(req_state, "additional_information_cpu", None) if req_state is not None else None
                    )
                    start_offset = int(self.query_start_loc.cpu[req_index])
                    sched_tokens = int(num_scheduled_tokens_np[req_index])
                    s, e = start_offset, start_offset + sched_tokens
                    # only consider to store data into update dict.
                    hidden_states_slice = hidden_states[s:e]
                    update_dict = self.model.postprocess(hidden_states_slice, **req_infos)
                    self._merge_additional_information_update(req_id, update_dict)
        except Exception as e:
            logger.error(
                f"Error merging for requests:{self.input_batch.req_ids} "
                f"additional information update: {e}, with the multimodal_outputs "
                f"as {multimodal_outputs}"
            )
            import traceback

            traceback.print_exc()
