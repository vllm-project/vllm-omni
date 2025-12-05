# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sequence import IntermediateTensors
from vllm.utils import (
    cdiv,
)
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
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.utils import (
    ProfileExecuteDuration,
    enable_sp,
    lmhead_tp_enable,
)
from vllm_ascend.worker.model_runner_v1 import AsyncNPUModelRunnerOutput

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.npu.npu_model_runner import OmniNPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


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
        dict[str, Any],
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

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

            # -------------------------------------- Omni-new -------------------------------------------------
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_input_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_input_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]

            model_kwargs = {
                **self._init_model_kwargs(num_input_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
            # (NOTE) Omni-new: input_ids isn't set as None
            # -------------------------------------------------------------------------------------------------
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
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
                query_start_loc=self.query_start_loc[: num_reqs + 1],
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
            num_input_tokens,
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
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
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

        moe_comm_type = self._select_moe_comm_method(num_input_tokens, self.with_prefill)

        # -------------------------------------- Omni-new -------------------------------------------------
        # Omni-new: don't use cudagraph_dispatcher
        # and remove ubatch_slices
        aclgraph_runtime_mode = CUDAGraphMode.NONE
        # -------------------------------------------------------------------------------------------------

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
                batch_descriptor=None,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                prefetch_stream=self.prefetch_stream,
                model_instance=self.model,
                weight_prefetch_method=self.weight_prefetch_method,
            ):
                self.maybe_setup_kv_connector(scheduler_output)

                outputs = self._run_generation(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    multimodal_kwargs=model_kwargs,
                    logits_indices=logits_indices,
                )

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(scheduler_output)

            aux_hidden_states = None
            if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                hidden_states, aux_hidden_states = outputs

        kv_connector_output = KVConnectorOutput(finished_sending=finished_sending, finished_recving=finished_recving)
        finished_sending = None
        finished_recving = None

        # -------------------------------------- Omni-new -------------------------------------------------
        # Omni-new: extract_multimodal_outputs
        _, multimodal_outputs = self.extract_multimodal_outputs(outputs)
        # Ensure one tensor per request, map to CPU for output struct
        pooler_output: list[torch.Tensor | None] = []
        if isinstance(multimodal_outputs, torch.Tensor):
            # If model returned a single stacked tensor, split by requests
            assert multimodal_outputs.shape[0] == self.input_batch.num_reqs
            for i in range(self.input_batch.num_reqs):
                pooler_output.append(multimodal_outputs[i].detach().to("cpu").contiguous())
        elif isinstance(multimodal_outputs, list):
            for out in multimodal_outputs:
                pooler_output.append(out.detach().to("cpu").contiguous() if out is not None else None)
        elif isinstance(multimodal_outputs, dict):
            for out in multimodal_outputs.values():
                pooler_output.append(out.detach().to("cpu").contiguous() if out is not None else None)
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
        # -------------------------------------------------------------------------------------------------

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
        """Runs the generation process and returns per-request tensors.

        Tries model interfaces in the following order for maximal compatibility:
        1) model.sample(condition=..., **kwargs)
        2) model.forward(condition=..., **kwargs)
        3) model.diffuse(condition=..., **kwargs)
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
        # TODO: add the diffuse method for other models

        raise RuntimeError(
            "The loaded model does not expose generation interfaces 'sample', "
            "'forward', or 'diffuse'. Please implement one of them or adapt the runner."
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        is_torchair_compile: bool = False,
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
        (num_tokens, num_tokens_across_dp, with_prefill, _) = self._sync_metadata_across_dp(
            num_tokens, with_prefill, False
        )

        moe_comm_type = self._select_moe_comm_method(num_tokens, with_prefill)

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

        if not self.in_profile_run and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens):
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=num_tokens, dtype=self.dtype, device=self.device
                    )
                intermediate_tensors = IntermediateTensors(
                    {k: v[:num_tokens] for k, v in self.intermediate_tensors.items()}
                )

            # filter out the valid batch descriptor
            _ag_mode, batch_descriptor = self.aclgraph_dispatcher.dispatch(
                BatchDescriptor(num_tokens=num_tokens, uniform_decode=uniform_decode)
            )
            if aclgraph_runtime_mode is not None:
                # we allow forcing NONE when the dispatcher disagrees to support
                # warm ups for aclgraph capture
                assert aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode == _ag_mode, (
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}."
                )
            else:
                aclgraph_runtime_mode = _ag_mode

            # TODO(Mengqing): Set create_mixed_batch to False since it's only used in FI warmup
            # and not supported in ASCEND now. We could remove it in the future.
            attn_metadata = self._build_dummy_attn_metadata(
                False,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                max_query_len=max_query_len,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                force_attention=force_attention,
            )

            need_dummy_logits = not self.in_profile_run and lmhead_tp_enable()

            if need_dummy_logits:
                max_num_reqs_across_dp = num_tokens if not with_prefill else max_num_reqs
                dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

                def dummy_compute_logits(hidden_states):
                    return self.model.compute_logits(hidden_states[dummy_indices])

            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                with_prefill=with_prefill,
                in_profile_run=self.in_profile_run,
                reserved_mc2_mask=self.reserved_mc2_mask,
                moe_comm_type=moe_comm_type,
                num_actual_tokens=0,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                prefetch_stream=self.prefetch_stream,
                model_instance=self.model,
                weight_prefetch_method=self.weight_prefetch_method,
            ):
                hidden_states = self._generate_dummy_run_hidden_states(
                    with_prefill,
                    is_torchair_compile,
                    input_ids,
                    positions,
                    attn_metadata,
                    num_tokens,
                    intermediate_tensors,
                    inputs_embeds,
                )
                if need_dummy_logits:
                    dummy_compute_logits(hidden_states)

            if self.drafter:
                self.drafter.dummy_run(
                    num_tokens=num_tokens,
                    with_prefill=with_prefill,
                    skip_attn=True,
                    num_reqs=num_reqs,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                )
                if need_dummy_logits:
                    self.drafter.model.compute_logits(hidden_states[dummy_indices])
            if self.in_profile_run and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if not self.in_profile_run and self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()
                self.eplb_updator.forward_end()

            # -------------------------------------- Omni-new -------------------------------------------------
            hidden_states, _ = self.extract_multimodal_outputs(hidden_states)
            # -------------------------------------------------------------------------------------------------
            return hidden_states
