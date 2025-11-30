# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.core.sched.output import SchedulerOutput

# yapf conflicts with isort for this block
# yapf: disable
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
)

# yapf: enable
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import update_attn_params, update_mla_attn_params
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.utils import (
    ProfileExecuteDuration,
    enable_sp,
    lmhead_tp_enable,
)
from vllm_ascend.worker.model_runner_v1 import AsyncNPUModelRunnerOutput

from vllm_omni.engine import AdditionalInformationPayload, PromptEmbedsPayload
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.npu.npu_model_runner import OmniNPUModelRunner


class NPUARModelRunner(OmniNPUModelRunner):
    """Autoregressive NPU model runner that returns hidden states per request."""

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
        int,
        dict[str, dict] | None,
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
        max_num_scheduled_tokens = num_scheduled_tokens.max()
        num_valid_tokens = np.array(
            [
                num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for num_tokens, i in zip(tokens, req_ids)
            ],
            dtype=np.int32,
        )

        if self.use_aclgraph and total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]:
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(total_num_scheduled_tokens)
        elif self.use_aclgraph and enable_sp(self.vllm_config):
            # When using aclgraph, if total_num_scheduled_tokens exceeds the maximum graph size,
            # the model will fall back to running its FX graph in eager mode.
            # In this case, when sequence parallelism is enabled, we need to pad tokens to align
            # with tp_size because pad_size cannot be captured by the FX graph
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_input_tokens = math.ceil(total_num_scheduled_tokens / tp_size) * tp_size
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]

        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(), attn_state, total_num_scheduled_tokens)

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill, enable_dbo) = self._sync_metadata_across_dp(
            num_input_tokens, with_prefill, enable_dbo
        )

        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        num_input_tokens = maybe_padded_num_tokens

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens], non_blocking=True
            )

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]

        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices),
            out=self.input_ids_cpu[:total_num_scheduled_tokens],
        )

        # Prepare some information for building Attention-Metadata
        # Compute and commit slot mapping
        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc[: num_reqs + 1].copy_(self.query_start_loc_cpu[: num_reqs + 1], non_blocking=True)

        self.seq_lens_np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs], non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.query_start_loc[num_reqs + 1 :].fill_(-1)
        self.seq_lens[num_reqs:].fill_(0)

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)
        self.positions_cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions[:num_input_tokens].copy_(self.positions_cpu[:num_input_tokens], non_blocking=True)

        # Make Attention metadata
        positions_cpu = self.positions_cpu[:num_input_tokens]
        positions = self.positions[:num_input_tokens]
        seq_lens_cpu = self.seq_lens_cpu[:num_reqs]
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)
        self.attn_mask = self._make_attention_mask(seq_lens=seq_lens_cpu, position=positions_cpu, attn_state=attn_state)
        self.attn_state = attn_state  # type: ignore

        self.with_prefill = with_prefill
        self.num_tokens_across_dp = num_tokens_across_dp
        self._update_graph_pad_size(with_prefill, maybe_padded_num_tokens)
        attn_metadata: dict[str, Any] = {}

        # Omni-new: per_req_additional_information
        per_req_additional_information: dict[str, dict] | None = None

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:total_num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:total_num_scheduled_tokens].copy_(inputs_embeds)

            #  -------------------------------------- Omni-new -------------------------------------------------
            # Omni-new: Reset per-step additional information collector (deprecated concat path)
            if hasattr(self, "_forward_additional_information"):
                self._forward_additional_information = None
            # Omni-new: per-request additional information for this step
            per_req_additional_information = {}

            # Omni-new: Overlay custom prompt_embeds per request for the prompt portion;
            # collect additional_information (tensor/list) for prefill portion only
            for req_index, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                pe_cpu = getattr(req_state, "prompt_embeds_cpu", None)
                addi_cpu = getattr(req_state, "additional_information_cpu", None)
                num_computed_tokens = int(self.input_batch.num_computed_tokens_cpu[req_index])
                prompt_len = len(req_state.prompt_token_ids)
                prompt_remaining = max(0, prompt_len - num_computed_tokens)
                sched_tokens = int(num_scheduled_tokens[req_index])
                overlay_len = min(sched_tokens, prompt_remaining)
                if overlay_len <= 0:
                    continue
                if pe_cpu is not None:
                    src = pe_cpu[num_computed_tokens : num_computed_tokens + overlay_len].to(
                        dtype=self.dtype, device=self.device, non_blocking=True
                    )
                    start_offset = int(self.query_start_loc_cpu[req_index])
                    self.inputs_embeds[start_offset : start_offset + overlay_len].copy_(src)
                # Build per-request additional information (no cross-request concat)
                if addi_cpu is not None and isinstance(addi_cpu, dict):
                    req_info: dict[str, object] = {}
                    for k, v in addi_cpu.items():
                        if isinstance(v, torch.Tensor):
                            # For prefill tokens, pass only the scheduled slice;
                            # for decode or no scheduled tokens, pass whole tensor
                            if overlay_len > 0:
                                try:
                                    seg = (
                                        v[num_computed_tokens : num_computed_tokens + overlay_len]
                                        .detach()
                                        .to("cpu")
                                        .contiguous()
                                    )
                                except Exception:
                                    seg = v.detach().to("cpu").contiguous()
                                req_info[k] = seg
                            else:
                                req_info[k] = v.detach().to("cpu").contiguous()
                        elif isinstance(v, list):
                            req_info[k] = v
                        else:
                            req_info[k] = v
                    per_req_additional_information[req_id] = req_info
            #  ------------------------------------------------------------------------------------------------

            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = self.input_ids[:num_input_tokens]
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        positions = self.positions[:num_input_tokens]
        input_ids, positions = self._update_input_ids_and_positions(
            input_ids, positions, num_input_tokens, with_prefill, maybe_padded_num_tokens
        )

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors(
                {k: v[:num_input_tokens] for k, v in self.intermediate_tensors.items()}
            )

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            logits_indices = torch.from_numpy(cu_num_tokens - 1).to(self.device, non_blocking=True)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices
            self.num_draft_tokens.np[:num_reqs] = num_draft_tokens
            self.num_draft_tokens.np[num_reqs:].fill(0)
            self.num_draft_tokens.copy_to_gpu()

        # Used in the below loop.
        # query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
        spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(kv_cache_group_spec.kv_cache_spec, EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                slot_mapping = blk_table.slot_mapping_cpu[:total_num_scheduled_tokens]
                self.slot_mapping[:total_num_scheduled_tokens].copy_(
                    slot_mapping[:total_num_scheduled_tokens],
                    non_blocking=True,
                )
                self.slot_mapping[total_num_scheduled_tokens:].fill_(0)

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc_cpu[: num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[: num_reqs + 1],
                seq_lens_cpu=self.seq_lens_cpu,
                seq_lens=self.seq_lens_cpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                num_input_tokens=num_input_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                # TODO: change this to the right block table for linear attn
                block_table_tensor=blk_table_tensor[:num_reqs],
                slot_mapping=self.slot_mapping,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                positions=self.positions,
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                attn_state=self.attn_state,
                enable_dbo_across_dp=enable_dbo,
                is_only_prefill=bool(np.all(num_valid_tokens != 1)),
                max_query_len=max_num_scheduled_tokens,
                graph_pad_size=self.graph_pad_size,
                decode_token_per_req=self.decode_token_per_req,
                cos=self.cos,
                sin=self.sin,
            )

            if self.speculative_config and spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder) or self.model_config.runner_type == "pooling":
                    if use_spec_decode:
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs],
                            num_draft_tokens=self.num_draft_tokens.gpu[:num_reqs],
                        )
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args,
                    )
                else:
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        model=self.get_model(),
                        **extra_attn_metadata_args,
                    )

                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        if lmhead_tp_enable():
            max_num_reqs_across_dp = maybe_padded_num_tokens if not with_prefill else self.max_num_reqs
            logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (
            attn_metadata,
            positions,
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_across_dp,
            maybe_padded_num_tokens,
            logits_indices,
            spec_decode_metadata,
            input_ids,
            inputs_embeds,
            intermediate_tensors,
            max_num_scheduled_tokens,
            per_req_additional_information,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        with ProfileExecuteDuration().capture_async("prepare input"):
            self._update_states(scheduler_output)

            #  -------------------------------------- Omni-new -------------------------------------------------
            # Omni-new: Decode per-request prompt_embeds / additional_hidden_states payloads
            # (if present) into CPU tensors
            try:
                new_reqs = getattr(scheduler_output, "scheduled_new_reqs", [])
                if new_reqs:
                    for nr in new_reqs:
                        req_id = getattr(nr, "req_id", None) or getattr(nr, "request_id", None)
                        if req_id is None:
                            continue
                        # prompt_embeds
                        payload_pe = getattr(nr, "prompt_embeds", None)
                        if payload_pe is not None:
                            if isinstance(payload_pe, torch.Tensor):
                                pe_cpu = payload_pe.detach().to("cpu").contiguous()
                            elif isinstance(payload_pe, PromptEmbedsPayload):
                                dt = np.dtype(getattr(payload_pe, "dtype", "float32"))
                                arr = np.frombuffer(payload_pe.data, dtype=dt)
                                arr = arr.reshape(payload_pe.shape)
                                pe_cpu = torch.from_numpy(arr)
                            else:
                                pe_cpu = None
                            if pe_cpu is not None and req_id in self.requests:
                                setattr(
                                    self.requests[req_id],
                                    "prompt_embeds_cpu",
                                    pe_cpu,
                                )
                        # additional_information
                        payload_info = getattr(nr, "additional_information", None)
                        if payload_info is not None:
                            info_dict = {}
                            if isinstance(payload_info, dict):
                                # Already decoded
                                info_dict = payload_info
                            elif isinstance(payload_info, AdditionalInformationPayload):
                                for k, entry in payload_info.entries.items():
                                    if entry.tensor_data is not None:
                                        dt = np.dtype(getattr(entry, "tensor_dtype", "float32"))
                                        arr = np.frombuffer(entry.tensor_data, dtype=dt)
                                        arr = arr.reshape(entry.tensor_shape)
                                        info_dict[k] = torch.from_numpy(arr)
                                    else:
                                        info_dict[k] = entry.list_data
                            if info_dict and req_id in self.requests:
                                setattr(
                                    self.requests[req_id],
                                    "additional_information_cpu",
                                    info_dict,
                                )
            except Exception as e:
                logger.error(f"Error decoding prompt_embeds / additional_information: {e}")
                pass
            #  ------------------------------------------------------------------------------------------------

            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug("skip this step for we receive the data from remote disaggregate prefill node")
                    # Return empty ModelRunnerOutput if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output)

            if self.dynamic_eplb:
                self.eplb_updator.forward_before()

            (
                attn_metadata,
                positions,
                num_scheduled_tokens_np,
                num_input_tokens,
                num_tokens_across_dp,
                maybe_padded_num_tokens,
                logits_indices,
                spec_decode_metadata,
                input_ids,
                inputs_embeds,
                intermediate_tensors,
                max_query_len,
                per_req_additional_information,
            ) = self._prepare_inputs(scheduler_output, intermediate_tensors)

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        moe_comm_type = self._select_moe_comm_method(num_input_tokens, self.with_prefill)

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            scheduler_output.total_num_scheduled_tokens == self.input_batch.num_reqs * max_query_len
        )
        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=uniform_decode)
        aclgraph_runtime_mode, batch_descriptor = self.aclgraph_dispatcher.dispatch(batch_descriptor)

        # Run forward pass
        with ProfileExecuteDuration().capture_async("forward"):
            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                with_prefill=self.with_prefill,
                reserved_mc2_mask=self.reserved_mc2_mask,
                moe_comm_type=moe_comm_type,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                prefetch_stream=self.prefetch_stream,
                model_instance=self.model,
                weight_prefetch_method=self.weight_prefetch_method,
            ):
                self.maybe_setup_kv_connector(scheduler_output)
                #  -------------------------------------- Omni-new -------------------------------------------------
                model_kwargs_extra = {}
                # Pass per-request additional information map for this step (no concat)
                if per_req_additional_information:
                    model_kwargs_extra["additional_information_by_req_id"] = per_req_additional_information
                # Always pass per-request runtime additional_information (persisted in request state)
                try:
                    per_req_runtime_info = []
                    for req_id in self.input_batch.req_ids:
                        req_state = self.requests.get(req_id)
                        info = getattr(req_state, "additional_information_cpu", None) if req_state is not None else None
                        per_req_runtime_info.append(info if isinstance(info, dict) else {})
                    model_kwargs_extra["runtime_additional_information"] = per_req_runtime_info
                    model_kwargs_extra["request_ids"] = self.input_batch.req_ids
                    # Pass each request's token span within the flattened sequence for this step,
                    # enabling the model to map decode/prefill by request
                    req_token_spans = []
                    for req_index in range(len(self.input_batch.req_ids)):
                        start_offset = int(self.query_start_loc_cpu[req_index])
                        sched_tokens = int(num_scheduled_tokens_np[req_index])
                        req_token_spans.append((start_offset, start_offset + sched_tokens))
                    model_kwargs_extra["request_token_spans"] = req_token_spans
                except Exception:
                    pass
                #  ------------------------------------------------------------------------------------------------

                hidden_states = self._generate_process_reqs_hidden_states(
                    attn_metadata,
                    self.with_prefill,
                    maybe_padded_num_tokens,
                    input_ids,
                    positions,
                    intermediate_tensors,
                    inputs_embeds,
                    model_kwargs_extra,
                )

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(scheduler_output)

            aux_hidden_states = None
            if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                hidden_states, aux_hidden_states = hidden_states

        kv_connector_output = KVConnectorOutput(finished_sending=finished_sending, finished_recving=finished_recving)
        finished_sending = None
        finished_recving = None
        with ProfileExecuteDuration().capture_async("post process"):
            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019

            #  -------------------------------------- Omni-new -------------------------------------------------
            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
            # The model side may return per-request additional_information updates (model-agnostic channel).
            # Convention: multimodal_outputs["additional_information_update"] is a list[dict] in batch order;
            # the runner merges it into the corresponding request's additional_information_cpu for subsequent decode.
            try:
                if isinstance(multimodal_outputs, dict) and (
                    "additional_information_update" in multimodal_outputs
                    or "additional_information_update_by_req_id" in multimodal_outputs
                ):
                    # Option A: list[dict] in batch order
                    updates_list = multimodal_outputs.get("additional_information_update")
                    if isinstance(updates_list, list):
                        for idx, upd in enumerate(updates_list):
                            if not isinstance(upd, dict) or idx >= len(self.input_batch.req_ids):
                                continue
                            req_id = self.input_batch.req_ids[idx]
                            self._merge_additional_information_update(req_id, upd)
                    # Option B: dict[str, dict] keyed by req_id
                    updates_map = multimodal_outputs.get("additional_information_update_by_req_id")
                    if isinstance(updates_map, dict):
                        for req_id, upd in updates_map.items():
                            if not isinstance(upd, dict):
                                continue
                            if req_id not in self.requests:
                                continue
                            self._merge_additional_information_update(req_id, upd)
            except Exception as e:
                logger.error(
                    f"Error merging for requests:{self.input_batch.req_ids} additional \
                        information update: {e}, with the multimodal_outputs as {multimodal_outputs}"
                )
            #  ------------------------------------------------------------------------------------------------
            broadcast_pp_output = (
                self.parallel_config.distributed_executor_backend == "external_launcher"
                and len(get_pp_group().ranks) > 0
            )
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    return self._pool(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        finished_sending,
                        finished_recving,
                        kv_connector_output,
                    )
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            if broadcast_pp_output:
                model_output_broadcast_data = (
                    {
                        "logits": logits.contiguous(),
                    }
                    if logits is not None
                    else {}
                )
                model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                logits = self.apply_grammar_bitmask(scheduler_output, logits)

            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if spec_decode_metadata is None:
                if lmhead_tp_enable() and logits is not None:
                    logits = logits[: self.input_batch.num_reqs]
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                if lmhead_tp_enable() and logits is not None:
                    logits = logits[: len(spec_decode_metadata.logits_indices)]
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                assert logits is not None
                bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                target_logits = logits[spec_decode_metadata.target_logits_indices]
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

            discard_sampled_tokens_req_indices: list[int] = []
            # TODO(woosuk): The following loop can be slow since it iterates over
            # the requests one by one. Optimize.
            discard_sampled_tokens_req_indices = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
                if seq_len < req_state.num_tokens:
                    # Ignore the sampled token.
                    # Rewind the generator state as if the token was not sampled.
                    generator = self.input_batch.generators.get(i)
                    if generator is not None:
                        generator.set_offset(generator.get_offset() - 4)
                    discard_sampled_tokens_req_indices.append(i)

            # Copy some objects so they don't get modified after returning.
            # This is important when using async scheduling.
            req_ids_output_copy = self.input_batch.req_ids.copy()
            req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

            # NOTE: NPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = logprobs_tensors.tolists() if logprobs_tensors is not None else None

            # Compute prompt logprobs if needed.
            prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                hidden_states[: scheduler_output.total_num_scheduled_tokens],
                scheduler_output,
            )

            num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
            sampled_token_ids = sampler_output.sampled_token_ids
            if not self.use_async_scheduling:
                # Get the valid generated tokens.
                max_gen_len = sampled_token_ids.shape[-1]
                if max_gen_len == 1:
                    # No spec decode tokens.
                    valid_sampled_token_ids = sampled_token_ids.tolist()
                else:
                    # Includes spec decode tokens.
                    valid_sampled_token_ids = self.rejection_sampler.parse_output(
                        sampled_token_ids,
                        self.input_batch.vocab_size,
                    )
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[i].clear()
            else:
                valid_sampled_token_ids = []
                invalid_req_indices = list(discard_sampled_tokens_req_indices)
                invalid_req_indices_set = set(invalid_req_indices)
                assert sampled_token_ids.shape[-1] == 1

                # Cache the sampled tokens on the NPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
                self.input_batch.prev_sampled_token_ids_invalid_indices = invalid_req_indices_set
                self.input_batch.prev_req_id_to_index = {
                    req_id: i for i, req_id in enumerate(self.input_batch.req_ids) if i not in invalid_req_indices_set
                }
            # Cache the sampled tokens in the model runner, so that the scheduler
            # doesn't need to send them back.
            # NOTE(woosuk): As an exception, when using PP, the scheduler sends
            # the sampled tokens back, because there's no direct communication
            # between the first-stage worker and the last-stage worker.
            for req_idx in range(num_sampled_tokens):
                if self.use_async_scheduling:
                    sampled_ids = [-1] * 1 if req_idx not in invalid_req_indices_set else None
                else:
                    sampled_ids = valid_sampled_token_ids[req_idx]
                if not sampled_ids:
                    continue

                start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                end_idx = start_idx + len(sampled_ids)
                assert end_idx <= self.model_config.max_model_len, (
                    "Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: "
                    f"{self.model_config.max_model_len}"
                )

                self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
                self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                self.input_batch.num_tokens[req_idx] = end_idx
                req_id = self.input_batch.req_ids[req_idx]
                req_state = self.requests[req_id]
                req_state.output_token_ids.extend(sampled_ids)

            if self.speculative_config:
                self._draft_token_ids = self.propose_draft_token_ids(
                    valid_sampled_token_ids,
                    sampling_metadata,
                    scheduler_output,
                    spec_decode_metadata,
                    positions,
                    scheduler_output.total_num_scheduled_tokens,
                    hidden_states,
                    attn_metadata,
                    aux_hidden_states,
                )

            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

        #  -------------------------------------- Omni-new -------------------------------------------------
        # Omni-new: Convert to per-request tensors on CPU
        hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
        pooler_output: list[torch.Tensor | None] = []
        prev_logits_index = 0
        for rid, logits_index in zip(req_ids_output_copy, logits_indices):
            # Base payload: hidden slice for this request in this iteration
            hidden_slice = hidden_states_cpu[prev_logits_index : logits_index + 1]
            payload: dict[str, object] = {"hidden": hidden_slice}
            # Merge multimodal_outputs if present
            if isinstance(multimodal_outputs, dict) and multimodal_outputs:
                mm_payload: dict[str, object] = {}
                for k, v in multimodal_outputs.items():
                    try:
                        # Case 1: tensor aligned on token dimension
                        if isinstance(v, torch.Tensor) and v.shape[0] == hidden_states_cpu.shape[0]:
                            mm_payload[k] = v.detach().to("cpu")[prev_logits_index : logits_index + 1].contiguous()
                        # Case 2: nested dict of tensors aligned on token dimension (e.g., selected_hidden_layers)
                        elif isinstance(v, dict):
                            sub_dict: dict[str, torch.Tensor] = {}
                            for sk, sv in v.items():
                                if isinstance(sv, torch.Tensor) and sv.shape[0] == hidden_states_cpu.shape[0]:
                                    sub_dict[str(sk)] = (
                                        sv.detach().to("cpu")[prev_logits_index : logits_index + 1].contiguous()
                                    )
                            if sub_dict:
                                mm_payload[k] = sub_dict
                        elif isinstance(v, list):
                            element: torch.Tensor = v[0]
                            multimodal_outputs[k] = v[1:] if len(v) > 1 else v
                            mm_payload[k] = element
                    except Exception as e:
                        # Best-effort; skip malformed entries
                        logger.error(f"Error in merge multimodal outputs: {e}")
                if mm_payload:
                    payload.update(mm_payload)
            pooler_output.append(payload)  # type: ignore[arg-type]
            prev_logits_index = logits_index + 1

        # Omni-new
        output = OmniModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=(pooler_output if self.vllm_config.model_config.engine_output_type != "text" else None),
            kv_connector_output=kv_connector_output,
        )
        #  ------------------------------------------------------------------------------------------------

        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [f"[{tag}]:{duration:.2f}ms" for tag, duration in durations.items()]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name, " ".join(dr_str))
        if self.dynamic_eplb:
            self.eplb_updator.forward_end()
        if not self.use_async_scheduling:
            return output

        return AsyncNPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
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
                new_list = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.detach().to("cpu").contiguous())
                    else:
                        new_list.append(item)
                merged[k] = new_list
            else:
                merged[k] = v
        setattr(req_state, "additional_information_cpu", merged)

    def _generate_process_reqs_hidden_states(
        self,
        attn_metadata,
        with_prefill,
        maybe_padded_num_tokens,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        model_kwargs_extra,
    ):
        assert self.model is not None
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs_extra,
        )

        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
            # TODO: maybe_padded_num_tokens will be removed, use num_input_tokens instead
            if self.vllm_config.model_config.use_mla:
                # FIXME: Try using `auto_dispatch_capture=True`
                update_mla_attn_params(
                    self.update_stream, forward_context, maybe_padded_num_tokens, self.speculative_config
                )
            else:
                update_attn_params(self.update_stream, forward_context, maybe_padded_num_tokens)

        if get_forward_context().sp_enabled:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            pad_size = get_forward_context().pad_size
            if pad_size > 0:
                hidden_states = hidden_states[:-pad_size, :]
        return hidden_states
