# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import math
from copy import copy

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput, make_empty_encoder_model_runner_output
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm_ascend.ascend_forward_context import MoECommType, get_mc2_tokens_capacity, set_ascend_forward_context
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import ProfileExecuteDuration, enable_sp, lmhead_tp_enable

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.platforms.npu.worker.npu_ar_model_runner import ExecuteModelState
from vllm_omni.platforms.npu.worker.npu_model_runner import OmniNPUModelRunner


class NPUGenerationModelRunner(OmniNPUModelRunner):
    """Generation model runner for vLLM-omni on NPU (non-autoregressive)."""

    def _update_request_states(self, scheduler_output: SchedulerOutput):
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for _, req_id in enumerate(cached_reqs.req_ids):
            req_state = self.requests.get(req_id)
            assert req_state is not None
            req_state.prompt_token_ids = cached_reqs.prompt_token_ids.get(req_id)
            self.input_batch.remove_request(req_id)
            # update the request state in self.input_batch
            self.input_batch.add_request(req_state)
            self._init_mrope_positions(req_state)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        with ProfileExecuteDuration().capture_async("prepare input"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            if self.model_config.async_chunk:
                self._update_request_states(scheduler_output)
            #  -------------------------------------- Omni-new -------------------------------------------------
            self._update_states(scheduler_output)
            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug("skip this step for we receive the data from remote disaggregate prefill node")
                    # Return empty ModelRunnerOutput if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.dynamic_eplb:
                self.eplb_updator.forward_before()

            (
                attn_metadata,
                num_scheduled_tokens_np,
                num_input_tokens,
                num_tokens_across_dp,
                logits_indices,
                spec_decode_metadata,
                max_query_len,
            ) = self._prepare_inputs(scheduler_output)

            (input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output) = (
                self._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)
            )

            # update global cos, sin
            update_cos_sin(positions)

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        # prevent debugger is None
        if self.debugger is not None:
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

        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        # Run forward pass
        with ProfileExecuteDuration().capture_async("forward"):
            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                model_instance=self.model,
            ):
                self.maybe_setup_kv_connector(scheduler_output)
                #  -------------------------------------- Omni-new -------------------------------------------------
                outputs = self._run_generation_model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    model_kwargs=model_kwargs,
                    logits_indices=logits_indices,
                )
                #  -------------------------------------- Omni-new -------------------------------------------------

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(scheduler_output)

            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                outputs, aux_hidden_states = outputs

        kv_connector_output = KVConnectorOutput(finished_sending=finished_sending, finished_recving=finished_recving)
        finished_sending = None
        finished_recving = None

        _, multimodal_outputs = self.extract_multimodal_outputs(outputs)
        # Apply structured output bitmasks if present
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            None,
            spec_decode_metadata,
            outputs,
            None,
            aux_hidden_states,
            kv_connector_output,
            attn_metadata,
            positions,
            ec_connector_output,
            multimodal_outputs,
        )
        self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            kv_connector_output,
            attn_metadata,
            positions,
            ec_connector_output,
            multimodal_outputs,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        #  -------------------------------------- Omni-new -------------------------------------------------
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
            mm_payload = {}
            for key, out in multimodal_outputs.items():
                if out is not None and isinstance(out, torch.Tensor):
                    mm_payload[key] = out.detach().to("cpu").contiguous()
            pooler_output.append(mm_payload)
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
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
        )
        #  -------------------------------------- Omni-new -------------------------------------------------
        if not self.use_async_scheduling:
            return output
        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=torch.tensor([], device=self.device),
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )

    def _run_generation_model(
        self,
        *,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        model_kwargs: dict,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run generation from codec codes to waveforms.

        Args:
            scheduler_output: Contains codec codes in input_ids or additional info
            intermediate_tensors: PP intermediate tensors if applicable

        Returns:
            Audio waveforms: [batch, 1, waveform_len] or list of tensors
        """
        # Keep inputs identical to AR runner
        kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
            sampling_metadata=self.input_batch.sampling_metadata,
            logits_index=logits_indices,
            sampler=self.sampler,
        )

        if hasattr(self.model, "forward"):
            return self._model_forward(**kwargs)

        raise RuntimeError(
            "The loaded model does not expose diffusion interfaces 'sample', "
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
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
    ) -> torch.Tensor:
        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode in {
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

        has_lora = True if self.lora_config and self.compilation_config.cudagraph_specialize_lora else False
        _ag_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora
        )

        # Padding for DP
        (num_tokens, num_tokens_across_dp, with_prefill) = self._sync_metadata_across_dp(
            batch_descriptor.num_tokens, with_prefill
        )

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

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        if num_tokens != batch_descriptor.num_tokens:
            _ag_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora
            )

        num_tokens_padded = batch_descriptor.num_tokens
        num_reqs_padded = batch_descriptor.num_reqs if batch_descriptor.num_reqs is not None else num_reqs
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        # filter out the valid batch descriptor
        if cudagraph_runtime_mode is not None:
            # we allow forcing NONE when the dispatcher disagrees to support
            # warm ups for aclgraph capture
            if cudagraph_runtime_mode != CUDAGraphMode.NONE and cudagraph_runtime_mode != _ag_mode:
                raise ValueError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {cudagraph_runtime_mode}."
                )
        else:
            cudagraph_runtime_mode = _ag_mode

        # TODO(Mengqing): Set create_mixed_batch to False since it's only used in FI warmup
        # and not supported in ASCEND now. We could remove it in the future.
        attn_metadata = self._build_dummy_attn_metadata(
            False,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            aclgraph_runtime_mode=cudagraph_runtime_mode,
            force_attention=force_attention,
            is_graph_capturing=is_graph_capturing,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens, num_sampled_tokens):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.is_multimodal_model and not self.model_config.is_encoder_decoder:
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
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            # update global cos, sin
            update_cos_sin(positions)

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
                num_actual_tokens=0,
                aclgraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                model_instance=self.model,
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
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    in_graph_capturing=not force_attention,
                    is_profile=is_profile,
                )
            if is_profile and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if self.dynamic_eplb:
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
