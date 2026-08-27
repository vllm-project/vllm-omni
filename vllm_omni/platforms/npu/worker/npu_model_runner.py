# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from typing import Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu_model_runner import PerLayerAttnMetadata
from vllm_ascend.ascend_forward_context import get_forward_context, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import using_paged_attention
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import enable_sp, lmhead_tp_enable
from vllm_ascend.worker.model_runner_v1 import SEQ_LEN_WITH_MAX_PA_WORKSPACE

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms.npu._310p import is_310p
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

logger = init_logger(__name__)


if is_310p():
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310 as NPUModelRunner
else:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class OmniNPUModelRunner(OmniGPUModelRunner, NPUModelRunner):
    def initialize_kv_cache(self, kv_cache_config) -> None:
        """Create the omni tensor prefix cache.

        The omni prefix cache is used to store the hidden states of the prefix tokens
        """
        NPUModelRunner.initialize_kv_cache(self, kv_cache_config)
        if self.omni_prefix_cache is None and self.cache_config.enable_prefix_caching:
            # Read num_blocks back off self.kv_cache_config: vllm-ascend
            # deepcopies the config it was handed, so the value it stored is the
            # authoritative one, not our caller's argument.
            num_blocks = self.kv_cache_config.num_blocks
            self.omni_prefix_cache = OmniTensorPrefixCache(
                num_blocks=num_blocks,
                block_size=self.cache_config.block_size,
                hidden_size=self.model_config.get_hidden_size(),
                hs_dtype=self.dtype,
            )
            logger.info(
                "Initialized omni prefix cache on NPU (num_blocks=%d, block_size=%d, hidden_size=%d). "
                "Hidden-state cache is pinned host memory of roughly %.1f GiB; each per-token "
                "multimodal output key allocates another tensor of the same block shape.",
                num_blocks,
                self.cache_config.block_size,
                self.model_config.get_hidden_size(),
                num_blocks
                * self.cache_config.block_size
                * self.model_config.get_hidden_size()
                * self.dtype.itemsize
                / (1024**3),
            )

    def load_model(self, *args, **kwargs) -> None:
        if is_310p():
            from vllm_omni.platforms.npu._310p.patch import apply_model_patches

            apply_model_patches(self.model_config)
        NPUModelRunner.load_model(self, *args, **kwargs)
        # Initialize enable_sp cache to avoid get_current_vllm_config() error
        # in _pad_for_sequence_parallelism during execute_model.
        # This is a workaround for vllm-ascend not passing vllm_config to enable_sp().
        enable_sp(self.vllm_config)
        # ---------------------------------------Omni-new----------------------------------------------
        model = getattr(self, "model", None)
        override_fn = None
        if bool(getattr(model, "supports_sampled_token_ids_cpu_override", False)):
            candidate = getattr(model, "consume_sampled_token_ids_cpu_override", None)
            if callable(candidate):
                override_fn = candidate
        self._sampled_token_ids_cpu_override = override_fn
        self._omni_query_start_loc_model_kwarg = bool(getattr(model, "supports_omni_query_start_loc", False))
        self._maybe_enable_output_token_ids_for_model_sampler()
        self._init_talker_mtp()
        self._prewarm_attention_capture_workspaces()
        # ---------------------------------------Omni-new----------------------------------------------

    def _update_states(self, scheduler_output) -> Any:
        """Mirror NPUModelRunner._update_states and
        call OmniGPUModelRunner._update_states to update Omni-specific states.
        """
        req_data = scheduler_output.scheduled_cached_reqs

        if self.use_async_scheduling:
            for i, req_id in enumerate(req_data.req_ids):
                req_state = self.requests.get(req_id)
                if req_state is None:
                    continue
                if req_data.num_computed_tokens[i] < req_state.num_computed_tokens:
                    req_state.prev_num_draft_len = 0

        self._apply_pp_sampled_tokens_from_scheduler_output(scheduler_output)

        return OmniGPUModelRunner._update_states(self, scheduler_output)

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        profile_cpp: bool = False,
        randomize_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.is_valid_runtime_mode()
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
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            raise NotImplementedError("create_mixed_batch is used for warmup deepgemm, vllm-ascend does not need it")
        elif uniform_decode:
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        elif profile_cpp:
            num_reqs = 1
            num_scheduled_tokens_list = [num_tokens] * num_reqs
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        _cudagraph_mode, batch_desc, _, num_tokens_across_dp, _ = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=allow_microbatching,
            force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE) or profile_cpp,
            # `force_uniform_decode` is used for cudagraph capture; because for
            # capturing mixed prefill-decode batches, we sometimes use
            # num_tokens == num_reqs which looks like a uniform decode batch to the
            # dispatcher; but we actually want to capture a piecewise cudagraph
            force_uniform_decode=uniform_decode,
            # `force_has_lora` is used for cudagraph capture; because LoRA is
            # activated later in the context manager, but we need to know the
            # LoRA state when determining the batch descriptor for capture
            force_has_lora=num_active_loras > 0,
            force_num_active_loras=num_active_loras,
        )
        if self.use_dcp:
            self.dcp_manager.init_batch_info(
                num_scheduled_tokens,
                num_reqs,
                self.input_batch.num_computed_tokens_cpu,
                self.input_batch.num_prompt_tokens,
            )
            if self.speculative_config:
                self.dcp_manager.query_lens_full.cpu[:num_reqs] = torch.from_numpy(num_scheduled_tokens)
                self.dcp_manager.query_lens_full.copy_to_gpu()
        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )
        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        if self.dynamic_eplb:
            self.update_eplb_heat_collection_status(num_tokens_padded)

        # vllm-ascend does not support ubatch now
        ubatch_slices, ubatch_slices_padded = None, None
        attn_metadata: PerLayerAttnMetadata | None = None
        with self.synchronize_input_prep():
            # Build attention metadata for dummy_run
            if self._should_build_dummy_attn_metadata(force_attention, is_profile, cudagraph_runtime_mode):
                if create_mixed_batch:
                    raise NotImplementedError(
                        "create_mixed_batch is used for warmup deepgemm, vllm-ascend does not need it"
                    )
                self.attn_state = AscendAttentionState.DecodeOnly
                if self.speculative_config and self.speculative_config.method == "mtp":
                    # `AscendAttentionState.SpecDecoding` is only designed for mla
                    if self.vllm_config.model_config.use_mla:
                        self.attn_state = AscendAttentionState.SpecDecoding
                    else:
                        self.attn_state = AscendAttentionState.ChunkedPrefill
                # The reason why we use a fixed seq_len rather than max_query_len is that
                # _npu_paged_attention_get_workspace only returns max workspace with specific
                # seq_lens. We use this seq_len only when capturing graph, and still use max_query_len
                # in inference. This will be removed once npu_fused_infer_attention_score
                # outperforms _npu_paged_attention on all cases.
                if profile_seq_lens is not None:
                    seq_lens = profile_seq_lens
                else:
                    seq_lens = (
                        SEQ_LEN_WITH_MAX_PA_WORKSPACE
                        if is_graph_capturing and using_paged_attention(num_tokens, self.vllm_config)
                        else max_query_len
                    )  # type: ignore[assignment]
                self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
                self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

                cum_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
                self.query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
                self.query_start_loc.copy_to_gpu()
                if self._has_gdn:
                    self.gdn_query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
                    self.gdn_query_start_loc.copy_to_gpu()

                if not profile_cpp:
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        self.query_start_loc,
                        num_tokens_padded,
                        num_reqs_padded,
                        num_reqs,
                        cudagraph_runtime_mode,
                        batch_desc.num_reqs,
                    )

                # Dummy graph runs do not go through _prepare_inputs(), but GDN/Mamba
                # metadata reads block_table[:num_reqs_padded] below. Sync padded
                # rows as well so device-side metadata does not see stale block ids.
                self.input_batch.block_table.commit_block_table(num_reqs_padded)

                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_query_len,
                    ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                    for_cudagraph_capture=is_graph_capturing,
                    num_scheduled_tokens_np=num_scheduled_tokens,
                )
                if not is_graph_capturing:
                    for kv_cache_gid in range(len(self.kv_cache_config.kv_cache_groups)):
                        blk_table = self.input_batch.block_table[kv_cache_gid]
                        blk_table.slot_mapping.gpu.fill_(-1)

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            # TODO: The next line is a temporary workaround
            # to fix the accuracy issue of test_llama32_lora.py,
            # which is introduced by vllm-project/vllm#32005
            num_active_loras=(self.lora_config.max_loras if self.lora_config is not None else num_active_loras),
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            # ---------------------------------------Omni-new----------------------------------------------
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            elif getattr(getattr(self, "model", None), "has_preprocess", False):
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None
            # ---------------------------------------Omni-new----------------------------------------------

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions[:num_tokens_padded]

            # update global cos, sin
            update_cos_sin(positions)

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and flashcomm1 are enabled, during dummy_run the estimated space should divide num_tokens by
                # tp_size; otherwise, on non-first PP ranks it would effectively perform an extra all-gather, leading
                # to incorrect memory estimation and potentially causing OOM.
                intermediate_tokens = num_tokens_padded
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    intermediate_tokens = (num_tokens_padded + tp_size - 1) // tp_size
                if self.intermediate_tensors is None:
                    max_actual_tokens = self.max_num_tokens
                    if enable_sp():
                        max_actual_tokens = (self.max_num_tokens + tp_size - 1) // tp_size
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=max_actual_tokens, dtype=self.dtype, device=self.device
                    )
                intermediate_tensors = IntermediateTensors(
                    {k: v[:intermediate_tokens] for k, v in self.intermediate_tensors.items()}
                )

            need_dummy_logits = not is_profile and lmhead_tp_enable()
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

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds, randomize_inputs=randomize_inputs),
                set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    in_profile_run=is_profile,
                    num_actual_tokens=num_tokens_padded,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    model_instance=self.model,
                    has_sinks=self._has_sinks,
                    eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
                ),
            ):
                # ---------------------------------------Omni-new----------------------------------------------
                if getattr(self.model, "talker", None) is not None and self.has_talker_mtp:
                    num_tokens_padded_talker_mtp = num_tokens_padded
                    if num_tokens_padded_talker_mtp == self.max_num_tokens:
                        num_tokens_padded_talker_mtp = self.talker_mtp_input_ids.gpu.shape[0]
                    outputs = self.talker_mtp(
                        self.talker_mtp_input_ids.gpu[:num_tokens_padded_talker_mtp],
                        self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded_talker_mtp],
                        self.last_talker_hidden.gpu[:num_tokens_padded_talker_mtp],
                        self.text_step.gpu[:num_tokens_padded_talker_mtp],
                    )
                    self.compilation_config.cache_dir = None
                # Call self.model() directly (like GPU) to avoid make_omni_output during dummy_run
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
                # ---------------------------------------Omni-new----------------------------------------------
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs
            # ---------------------------------------Omni-new----------------------------------------------
            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
            # ---------------------------------------Omni-new----------------------------------------------
            dummy_compute_logits(hidden_states)

            if self.drafter and not profile_cpp:
                self.drafter.dummy_run(
                    num_tokens=num_tokens_padded,
                    with_prefill=with_prefill,
                    num_reqs=num_reqs_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    in_graph_capturing=not force_attention,
                    is_profile=is_profile,
                )
            if is_profile and self.dynamic_eplb:
                self.eplb_updator.adaptor.clear_all_moe_loads()
            if not is_profile and self.dynamic_eplb:
                self.eplb_updator.forward_end(self.eplb_heat_collection_status)
            self._finalize_dump_data(dump=False)
            return hidden_states, hidden_states

    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        """Override to combine NPUModelRunner's signature with OmniGPUModelRunner's logic.

        This method:
        1. Accepts num_tokens_padded as required by NPUModelRunner
        2. Injects omni-specific kwargs (runtime_additional_information)
        3. Caches model output for multimodal results
        4. Handles NPU-specific post-forward logic (graph params update, SP all-gather)
        """
        # Omni-specific: build and inject extra model kwargs
        model_kwargs_extra = self._build_model_kwargs_extra()
        assert self.model is not None

        forward_context = get_forward_context()
        assert forward_context is not None

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
            **model_kwargs,
            **model_kwargs_extra,
        }
        run_model = partial(self.model, **model_inputs)

        if self.enable_enpu:
            # The soft segmentation scenario requires event.record first, then event.wait.
            self._update_full_graph_params_if_needed(forward_context, num_tokens_padded)
            model_output = run_model()
        else:
            model_output = run_model()
            self._update_full_graph_params_if_needed(forward_context, num_tokens_padded)

        # Omni-specific: wrap output if needed
        if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
            model_output = self.model.make_omni_output(model_output, **model_kwargs, **model_kwargs_extra)

        # Omni-specific: cache model output for later sample_tokens
        self._omni_last_model_output = model_output

        # NPU-specific: all-gather for sequence parallelism
        if forward_context.flash_comm_v1_enabled and not isinstance(model_output, IntermediateTensors):
            model_output = self._all_gather_hidden_states_and_aux(model_output)

        return model_output

    def _gather_runtime_additional_information(self):
        """NPU parity fix for MiniCPM-o-4.5 Code2Wav request_id injection.

        On GPU, ``OmniGPUModelRunner._preprocess`` unconditionally writes
        ``req_infos["request_id"] = req_id`` (gpu_model_runner.py:1756) so that
        every stage's ``model_intermediate_buffer`` entry carries the engine
        request id. The NPU runner inherits ``execute_model`` from vllm_ascend
        and never runs omni's ``_preprocess``, so this id is otherwise missing
        on NPU. This override mirrors that GPU step at the gather point (called
        from ``_build_model_kwargs_extra`` -> forward), using the engine
        authoritative ``input_batch.req_ids`` and writing it both top-level and
        into ``meta`` so Code2Wav._parse_item (which reads either key) resolves
        it. Only fills when absent, so async_chunk=True handoffs that already
        carry the id are left untouched.
        """
        infos = super()._gather_runtime_additional_information()
        for req_id, info in zip(self.input_batch.req_ids, infos):
            if not isinstance(info, dict):
                continue
            if info.get("request_id"):
                continue
            meta = info.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                info["meta"] = meta
            meta["request_id"] = req_id
            info["request_id"] = req_id
        return infos
