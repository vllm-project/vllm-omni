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
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import enable_sp, lmhead_tp_enable
from vllm_ascend.worker.model_runner_v1 import SEQ_LEN_WITH_MAX_PA_WORKSPACE, NPUModelRunner

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

logger = init_logger(__name__)


class OmniNPUModelRunner(OmniGPUModelRunner, NPUModelRunner):
    def load_model(self, *args, **kwargs) -> None:
        NPUModelRunner.load_model(self, *args, **kwargs)
        # Initialize enable_sp cache to avoid get_current_vllm_config() error
        # in _pad_for_sequence_parallelism during execute_model.
        # This is a workaround for vllm-ascend not passing vllm_config to enable_sp().
        enable_sp(self.vllm_config)
        # TODO move this model specific logic to a separate class
        # TTS model IS the talker (no .talker sub-attr); use getattr to support both Omni and TTS.
        self.has_talker_mtp = False
        talker_mtp = getattr(self.model, "talker_mtp", None)
        if talker_mtp is not None:
            self.talker_mtp = talker_mtp  # type: ignore[assignment]
            self.has_talker_mtp = True
            cudagraph_mode = self.compilation_config.cudagraph_mode
            assert cudagraph_mode is not None
            has_separate_talker = getattr(self.model, "talker", None) is not None
            talker_mtp_graph_safe = getattr(self.model, "talker_mtp_graph_safe", False)
            if cudagraph_mode.has_full_cudagraphs() and (has_separate_talker or talker_mtp_graph_safe):
                self.talker_mtp = ACLGraphWrapper(talker_mtp, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
            # TTS exposes mtp_hidden_size; Omni uses hf_text_config.hidden_size.
            hidden_size = int(
                getattr(self.model, "mtp_hidden_size", 0) or getattr(self.model_config.hf_text_config, "hidden_size")
            )
            max_batch_size = max(self.max_num_reqs, self.compilation_config.max_cudagraph_capture_size)
            self.talker_mtp_input_ids = self._make_buffer(max_batch_size, dtype=torch.int32)
            self.talker_mtp_inputs_embeds = self._make_buffer(
                max_batch_size, hidden_size, dtype=self.dtype, numpy=False
            )
            self.last_talker_hidden = self._make_buffer(max_batch_size, hidden_size, dtype=self.dtype, numpy=False)
            self.text_step = self._make_buffer(max_batch_size, hidden_size, dtype=self.dtype, numpy=False)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()
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
        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                num_reqs,
            )
            if self.speculative_config:
                self.pcp_manager.query_lens_pcp_full.cpu[:num_reqs] = torch.from_numpy(num_scheduled_tokens)
                self.pcp_manager.query_lens_pcp_full.copy_to_gpu()
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
        # vllm-ascend does not support ubatch now
        ubatch_slices, ubatch_slices_padded = None, None
        attn_metadata: PerLayerAttnMetadata | None = None
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
                    num_tokens_padded, num_reqs_padded, num_reqs, cudagraph_runtime_mode, batch_desc.num_reqs
                )

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
                num_scheduled_tokens_np=num_scheduled_tokens,
            )

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
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder or self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

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

            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                in_profile_run=is_profile,
                num_actual_tokens=num_tokens_padded,
                aclgraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                model_instance=self.model,
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

            if self.drafter:
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
                target = self.model.language_model if hasattr(self.model, "language_model") else self.model
                target.clear_all_moe_loads()
            if self.dynamic_eplb:
                self.eplb_updator.forward_end()
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
            self._update_full_graph_params_if_needed(forward_context, num_tokens_padded, positions)
            model_output = run_model()
        else:
            model_output = run_model()
            self._update_full_graph_params_if_needed(forward_context, num_tokens_padded, positions)

        # Omni-specific: wrap output if needed
        if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
            model_output = self.model.make_omni_output(model_output, **model_kwargs, **model_kwargs_extra)

        # Omni-specific: cache model output for later sample_tokens
        self._omni_last_model_output = model_output

        # NPU-specific: all-gather for sequence parallelism
        if forward_context.flash_comm_v1_enabled and not isinstance(model_output, IntermediateTensors):
            model_output = self._all_gather_hidden_states_and_aux(model_output)

        return model_output

    def _talker_mtp_forward(self, decode_req_ids: list[str], inputs_embeds: torch.Tensor) -> None:
        decode_batch_size = len(decode_req_ids)
        if decode_batch_size == 0:
            return
        _cudagraph_mode, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
            num_tokens=decode_batch_size,
            num_reqs=decode_batch_size,
            num_scheduled_tokens_np=np.ones(decode_batch_size, dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )
        # Force eager for unwrapped code predictors (AR loops / multinomial).
        if not isinstance(self.talker_mtp, ACLGraphWrapper):
            _cudagraph_mode = CUDAGraphMode.NONE
        num_tokens_padded = batch_desc.num_tokens
        req_input_ids = self.talker_mtp_input_ids.gpu[:num_tokens_padded]
        req_embeds = self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded]
        last_talker_hidden = self.last_talker_hidden.gpu[:num_tokens_padded]
        text_step = self.text_step.gpu[:num_tokens_padded]
        subtalker_params = getattr(self.vllm_config.model_config, "subtalker_sampling_params", None)
        if not isinstance(subtalker_params, dict):
            subtalker_params = {}
        # Extract seed from the first request's sampling params for Fast AR
        # determinism.  NOTE: when batch_size > 1, all requests share the first
        # request's seed.  Per-request Fast AR seeding requires row-by-row
        # torch.multinomial calls and is left as a follow-up optimisation.
        _seed = None
        if decode_req_ids:
            _first_sp = getattr(self.requests[decode_req_ids[0]], "sampling_params", None)
            if _first_sp is not None and getattr(_first_sp, "seed", None) is not None:
                _seed = _first_sp.seed
            # Warn when batched requests have different seeds.
            if len(decode_req_ids) > 1 and _seed is not None:
                _other_seeds = {
                    getattr(getattr(self.requests[rid], "sampling_params", None), "seed", None)
                    for rid in decode_req_ids[1:]
                }
                if _other_seeds != {_seed}:
                    logger.warning(
                        "Fast AR seed: batch has mixed seeds; using first request's seed=%d for all %d requests.",
                        _seed,
                        len(decode_req_ids),
                    )
        talker_kwargs = {
            "do_sample": subtalker_params.get("do_sample"),
            "temperature": subtalker_params.get("temperature"),
            "top_k": subtalker_params.get("top_k"),
            "top_p": subtalker_params.get("top_p"),
        }
        if _seed is not None:
            talker_kwargs["seed"] = _seed
        with set_ascend_forward_context(
            None, self.vllm_config, aclgraph_runtime_mode=_cudagraph_mode, batch_descriptor=batch_desc
        ):
            req_embeds, code_predictor_codes = self.talker_mtp(
                req_input_ids,
                req_embeds,
                last_talker_hidden,
                text_step,
                **talker_kwargs,
            )
        # update the inputs_embeds and code_predictor_codes
        code_predictor_codes_cpu = code_predictor_codes.detach().to("cpu").contiguous()
        out_key = getattr(self.model, "talker_mtp_output_key", ("codes", "audio"))
        if not isinstance(out_key, tuple) or len(out_key) != 2:
            raise TypeError(f"talker_mtp_output_key must be a 2-tuple, got {type(out_key).__name__}: {out_key!r}")
        for idx, req_id in enumerate(decode_req_ids):
            req_index = self.input_batch.req_ids.index(req_id)
            start_offset = int(self.query_start_loc.cpu[req_index])
            inputs_embeds[start_offset : start_offset + 1] = req_embeds[idx : idx + 1]
            update_dict = {out_key[0]: {out_key[1]: code_predictor_codes_cpu[idx : idx + 1]}}
            self._merge_additional_information_update(req_id, update_dict)
