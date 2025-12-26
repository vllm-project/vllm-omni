# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm_ascend.ascend_forward_context import MoECommType, get_mc2_tokens_capacity, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.utils import ProfileExecuteDuration, enable_sp, lmhead_tp_enable

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.npu.npu_model_runner import OmniNPUModelRunner


class NPUGenerationModelRunner(OmniNPUModelRunner):
    """Generation model runner for vLLM-omni on NPU (non-autoregressive)."""

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

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        _, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        positions_np = np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
        )

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
        if self.pcp_size > 1:
            if not self.vllm_config.model_config.use_mla:
                self.generate_kv_idx(scheduler_output)
            tokens, position_pcp, pcp_unpad_mask = self._update_tokens_for_pcp(tokens)
            num_scheduled_tokens = np.array(tokens, dtype=np.int32)
            total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
        else:
            position_pcp, pcp_unpad_mask = None, None
            self.num_pcp_pads = self.num_pcp_pads[:num_reqs]

        total_num_pcp_pads = sum(self.num_pcp_pads)
        max_num_scheduled_tokens = max(tokens)
        num_valid_tokens = np.array(
            [
                num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for num_tokens, i in zip(tokens, req_ids)
            ],
            dtype=np.int32,
        )

        if self.use_aclgraph and total_num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]:
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

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill) = self._sync_metadata_across_dp(
            num_input_tokens, with_prefill
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

        if self.pcp_size > 1:
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                position_pcp[:total_num_scheduled_tokens],
                out=positions_np,
            )
        else:
            self.positions.np[:total_num_scheduled_tokens] = positions_np

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens], non_blocking=True
            )

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids, 0, token_indices_tensor, out=self.is_token_ids.cpu[:total_num_scheduled_tokens]
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or self.enable_prompt_embeds):
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
                    self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
                        req_embeds[start_pos:actual_end]
                    )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc.copy_to_gpu()

        self.seq_lens.np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        self.seq_lens.copy_to_gpu()

        # Fill unused with -1. Needed for reshape_and_cache
        self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)
        self.seq_lens.gpu[num_reqs:].fill_(0)

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens, cu_num_tokens)
        self.positions.cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions.copy_to_gpu()

        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)
        self.attn_mask = self._make_attention_mask(attn_state)
        self.attn_state = attn_state  # type: ignore

        self.with_prefill = with_prefill
        self.num_tokens_across_dp = num_tokens_across_dp
        attn_metadata: dict[str, Any] = {}

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        num_reqs = self.input_batch.num_reqs
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            original_seq_lens_np = self.input_batch.num_computed_tokens_cpu[:num_reqs] + np.array(
                tokens_original, dtype=np.int32
            )
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
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
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:total_num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            #  -------------------------------------- Omni-new -------------------------------------------------
            # NOTE(gcanlin): We don't set input_ids to None in vllm-omni.
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
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
            token_ids_idx = self.is_token_ids.gpu[:total_num_scheduled_tokens].nonzero(as_tuple=False).squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
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
                num_input_tokens_with_flashcomm1 = (num_input_tokens + tp_size - 1) // tp_size
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens_with_flashcomm1].copy_(
                    v[:num_input_tokens_with_flashcomm1], non_blocking=True
                )
            intermediate_tensors = IntermediateTensors(
                {k: v[:num_input_tokens_with_flashcomm1] for k, v in self.intermediate_tensors.items()}
            )

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            if self.pcp_size * self.dcp_size > 1:
                logits_indices = torch.from_numpy(cu_num_tokens) * self.pcp_size - self.num_pcp_pads[:num_reqs] - 1
                logits_indices = logits_indices.pin_memory().to(self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (
                    len(draft_token_ids)
                    if (
                        self.input_batch.num_computed_tokens_cpu[req_idx] >= self.input_batch.num_prompt_tokens[req_idx]
                    )
                    else -1
                )

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens, self.num_pcp_pads[:num_reqs]
            )
            logits_indices = spec_decode_metadata.logits_indices

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Used in the below loop.
        # query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
        self.spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        if self.speculative_config and self.pcp_size > 1:
            self._generate_pcp_mtp_input(
                num_reqs, scheduler_output.total_num_scheduled_tokens, scheduler_output.num_scheduled_tokens
            )

        long_seq_metadata = self._generate_pcp_metadata(total_num_scheduled_tokens)
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            # NOTE: This is strange, why did we use total_num_scheduled_tokens before?
            slot_mapping_size = (
                total_num_scheduled_tokens
                if self.pcp_size == 1
                else total_num_scheduled_tokens * self.pcp_size - total_num_pcp_pads
            )
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
                blk_table.slot_mapping.gpu[slot_mapping_size:].fill_(0)
                if self.pcp_size > 1:
                    slot_mapping_for_pcp = blk_table.slot_mapping.gpu[: long_seq_metadata.num_actual_tokens_pcp_padded]
                    slot_mapping_for_pcp[slot_mapping_size:].fill_(-1)
                    assert pcp_unpad_mask is not None
                    pcp_padded_slot_mapping = self.pcp_padded_slot_mapping[: pcp_unpad_mask.shape[0]]
                    pcp_padded_slot_mapping.fill_(-1)
                    pcp_padded_slot_mapping[pcp_unpad_mask] = slot_mapping_for_pcp[:slot_mapping_size]
                    slot_mapping_for_pcp[: long_seq_metadata.num_actual_tokens_pcp_padded] = pcp_padded_slot_mapping
                    blk_table.slot_mapping.gpu[: long_seq_metadata.num_actual_tokens_pcp_padded] = slot_mapping_for_pcp
                slot_mapping = blk_table.slot_mapping.gpu

            # NOTE: This is a temporary hack, now in GPUModelRunner, this prepare_inputs
            # has been split to multiple parts, and there are 3 parts that is related to this
            # `num_reqs`, we'll take `query_start_loc` as an example:
            # 1. self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            # 2. get `num_reqs_padded`, this depends on dispatcher and which is why we have the
            #    following simplified `dispatch` logic here, we try to minimize the impact
            # 3. query_start_loc = self.query_start_loc.gpu[: num_reqs_padded + 1]
            uniform_decode = (max_num_scheduled_tokens == self.uniform_decode_query_len) and (
                total_num_scheduled_tokens == max_num_scheduled_tokens * num_reqs
            )

            # TODO: We should make this official ASAP. Also note that if we pad here,
            # the builders won’t need to add any extra padding.
            if self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and uniform_decode:
                num_reqs_padded = num_input_tokens // self.uniform_decode_query_len
                pad_size = num_reqs_padded - num_reqs
                if pad_size > 0:
                    last_query_loc = self.query_start_loc.gpu[num_reqs]

                    steps = torch.arange(1, pad_size + 1, device=self.device, dtype=self.query_start_loc.gpu.dtype)
                    fill_values = last_query_loc + (steps * self.uniform_decode_query_len)

                    self.query_start_loc.gpu[num_reqs + 1 : num_reqs_padded + 1] = fill_values
                # So we are trying to simulate the behavior of GPUModelRunner's
                # prepare_inputs for uniform decode mode by padding query_start_loc
                num_reqs = num_reqs_padded

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
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
                ori_query_lens = (
                    self.query_start_loc_pcp_full.cpu[1 : num_reqs + 1] - self.query_start_loc_pcp_full.cpu[:num_reqs]
                )
                num_prefill_reqs = (ori_query_lens > self.decode_threshold).sum().item()
                num_decode_reqs = num_reqs - num_prefill_reqs
                num_decode_reqs_flatten = num_decode_reqs * self.decode_threshold
                blk_table_tensor[num_decode_reqs_flatten : num_decode_reqs_flatten + num_prefill_reqs].copy_(
                    blk_table_tensor[num_decode_reqs : num_decode_reqs + num_prefill_reqs].clone()
                )
                blk_table_tensor[:num_decode_reqs_flatten].copy_(
                    blk_table_tensor[:num_decode_reqs].repeat_interleave(self.decode_threshold, dim=0)
                )
                common_attn_metadata.block_table_tensor = blk_table_tensor[: num_decode_reqs_flatten + num_prefill_reqs]

            if self.speculative_config and self.spec_decode_common_attn_metadata is None:
                self.spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    if use_spec_decode:
                        patch_torch_npu_argsort()
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs],
                            num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs],
                        )
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args,
                    )
                elif self.model_config.runner_type == "pooling":
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
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
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
            model_kwargs,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | IntermediateTensors:
        with ProfileExecuteDuration().capture_async("prepare input"):
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                return EMPTY_MODEL_RUNNER_OUTPUT

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
                model_kwargs,
            ) = self._prepare_inputs(scheduler_output, intermediate_tensors)

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        moe_comm_type = self._select_moe_comm_method(num_input_tokens)
        # prevent debugger is None
        need_dump = self.dump_enable and self.debugger is not None
        if need_dump:
            assert self.debugger is not None
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(getattr(dbg_cfg, "level", "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            scheduler_output.total_num_scheduled_tokens == self.input_batch.num_reqs * max_query_len
        )
        has_lora = len(self.input_batch.lora_id_to_lora_request) > 0
        aclgraph_runtime_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_input_tokens, uniform_decode=uniform_decode, has_lora=has_lora
        )

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
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                prefetch_stream=self.prefetch_stream,
                model_instance=self.model,
                weight_prefetch_method=self.weight_prefetch_method,
            ):
                self.maybe_setup_kv_connector(scheduler_output)
                #  -------------------------------------- Omni-new -------------------------------------------------
                outputs = self._run_generation(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    multimodal_kwargs=model_kwargs,
                    logits_indices=logits_indices,
                )
                #  -------------------------------------- Omni-new -------------------------------------------------

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(scheduler_output)

            aux_hidden_states = None
            if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                hidden_states, aux_hidden_states = outputs

        kv_connector_output = KVConnectorOutput(finished_sending=finished_sending, finished_recving=finished_recving)
        finished_sending = None
        finished_recving = None
        #  -------------------------------------- Omni-new -------------------------------------------------
        # We don't need any post-process for generation model outputs
        _, multimodal_outputs = self.extract_multimodal_outputs(outputs)
        pooler_output: list[object] = []
        if isinstance(multimodal_outputs, torch.Tensor):
            assert multimodal_outputs.shape[0] == 1, (
                "model should return a single tensor, to return multiple tensors, use a dict"
            )
            assert multimodal_outputs.shape[0] == self.input_batch.num_reqs
            for i in range(self.input_batch.num_reqs):
                pooler_output.append({"model_outputs": multimodal_outputs[i].detach().to("cpu").contiguous()})
        elif isinstance(multimodal_outputs, list):
            assert len(multimodal_outputs) == 1, (
                "model should return a single list, to return multiple lists, use a dict"
            )
            for out in multimodal_outputs:
                pooler_output.append(
                    {"model_outputs": out.detach().to("cpu").contiguous() if out is not None else None}
                )
        elif isinstance(multimodal_outputs, dict):
            for key, out in multimodal_outputs.items():
                pooler_output.append({key: out.detach().to("cpu").contiguous() if out is not None else None})
        else:
            raise RuntimeError("Unsupported diffusion output type")
        output = OmniModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits={},
        )
        #  -------------------------------------- Omni-new -------------------------------------------------
        if not self.use_async_scheduling:
            return output

        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=[],
            invalid_req_indices=[],
            async_output_copy_stream=self.async_output_copy_stream,
        )

    def _run_generation(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        multimodal_kwargs: dict,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run generation from codec codes to waveforms.

        Args:
            scheduler_output: Contains codec codes in input_ids or additional info
            intermediate_tensors: PP intermediate tensors if applicable

        Returns:
            Audio waveforms: [batch, 1, waveform_len] or list of tensors
        """
        kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **MultiModalKwargs.as_kwargs(multimodal_kwargs, device=self.device),
            sampling_metadata=self.input_batch.sampling_metadata,
            logits_index=logits_indices,
            sampler=self.sampler,
        )

        if hasattr(self.model, "forward"):
            return self.model.forward(**kwargs)

        raise RuntimeError(
            "The loaded model does not expose generation interfaces 'sample', "
            "'forward', or 'diffuse'. Please implement one of them or adapt the runner."
        )

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        logger.warning("Dummy sampler run is not implemented for generation model")
        return None

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        aclgraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
    ) -> torch.Tensor:
        # only support eager mode and piecewise graph now
        assert aclgraph_runtime_mode is None or aclgraph_runtime_mode in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL,
        }
        # In multi-DP scenarios, there may be situations where all DP groups are executing dummy runs.
        # If sequence parallelism is enabled, it is essential to ensure that num_tokens is divisible by tp_size.
        if self.use_aclgraph and enable_sp(self.vllm_config):
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_tokens = math.ceil(num_tokens / tp_size) * tp_size

        # Force dummy run on prefill stage when this node is deemed as kv producer.
        if self.is_kv_producer and not self.is_kv_consumer:
            with_prefill = True

        # Padding for DP
        (num_tokens, num_tokens_across_dp, with_prefill) = self._sync_metadata_across_dp(num_tokens, with_prefill)

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.max_num_reqs
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            if with_prefill:
                num_reqs = num_tokens
            else:
                num_reqs = (num_tokens + self.decode_token_per_req - 1) // self.decode_token_per_req
            num_reqs = min(num_reqs, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        if not self.in_profile_run and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        has_lora = True if self.lora_config and self.compilation_config.cudagraph_specialize_lora else False
        _ag_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora
        )

        num_tokens_padded = batch_descriptor.num_tokens
        num_reqs_padded = batch_descriptor.num_reqs if batch_descriptor.num_reqs is not None else num_reqs
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        moe_comm_type = self._select_moe_comm_method(num_tokens_padded)

        # filter out the valid batch descriptor
        if aclgraph_runtime_mode is not None:
            # we allow forcing NONE when the dispatcher disagrees to support
            # warm ups for aclgraph capture
            if aclgraph_runtime_mode != CUDAGraphMode.NONE and aclgraph_runtime_mode != _ag_mode:
                raise ValueError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}."
                )
        else:
            aclgraph_runtime_mode = _ag_mode

        # TODO(Mengqing): Set create_mixed_batch to False since it's only used in FI warmup
        # and not supported in ASCEND now. We could remove it in the future.
        attn_metadata = self._build_dummy_attn_metadata(
            False,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            force_attention=force_attention,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens, num_sampled_tokens):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and flashcomm1 are enabled,
                # during dummy_run the estimated space should divide num_tokens by tp_size;
                # otherwise, on non-first PP ranks it would effectively perform an extra all-gather,
                # leading to incorrect memory estimation and potentially causing OOM.
                actual_tokens = num_tokens
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    actual_tokens = num_tokens // tp_size
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=actual_tokens, dtype=self.dtype, device=self.device
                    )
                intermediate_tensors = IntermediateTensors(
                    {k: v[:num_tokens_padded] for k, v in self.intermediate_tensors.items()}
                )

            need_dummy_logits = not self.in_profile_run and lmhead_tp_enable()
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return
                if hasattr(self.drafter, "model") and hasattr(self.drafter.model, "compute_logits"):
                    return self.drafter.model.compute_logits(hidden_states[dummy_indices])

            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                with_prefill=with_prefill,
                in_profile_run=self.in_profile_run,
                # reserved_mc2_mask=self.reserved_mc2_mask,
                moe_comm_type=moe_comm_type,
                num_actual_tokens=0,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                prefetch_stream=self.prefetch_stream,
                model_instance=self.model,
                weight_prefetch_method=self.weight_prefetch_method,
            ):
                hidden_states = self._generate_dummy_run_hidden_states(
                    input_ids, positions, num_tokens_padded, intermediate_tensors, inputs_embeds
                )
                dummy_compute_logits(hidden_states)

            if self.drafter:
                self.drafter.dummy_run(
                    num_tokens=num_tokens_padded,
                    with_prefill=with_prefill,
                    num_reqs=num_reqs_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    skip_attn=not force_attention,
                )
            if self.in_profile_run and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if not self.in_profile_run and self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()
                self.eplb_updator.forward_end()
            # -------------------------------------- Omni-new -------------------------------------------------
            hidden_states, _ = self.extract_multimodal_outputs(hidden_states)
            # -------------------------------------------------------------------------------------------------
            return hidden_states

    def profile_run(self) -> None:
        # Trigger compilation for general shape.
        with self.set_in_profile_run():
            hidden_states = self._dummy_run(
                self.max_num_tokens // self.pcp_size if self.pcp_size > 1 else self.max_num_tokens, with_prefill=True
            )
            # MC2 will consume additional NPU memory.
            # Therefore, we need to run the MC2 path once here to complete its initialization,
            # allowing vLLM to correctly estimate the maximum memory required.
            mc2_tokens_capacity = get_mc2_tokens_capacity()
            if (
                self.max_num_tokens > mc2_tokens_capacity
                and self._select_moe_comm_method(mc2_tokens_capacity) == MoECommType.MC2
            ):
                self._dummy_run(mc2_tokens_capacity, with_prefill=True)

        output = None

        NPUPlatform.synchronize()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()
