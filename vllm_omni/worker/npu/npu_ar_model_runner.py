# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from copy import copy
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import update_full_graph_params
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import ProfileExecuteDuration

from vllm_omni.core.sched.omni_ar_scheduler import KVCacheTransferData
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
    ec_connector_output: ECConnectorOutput | None
    multimodal_outputs: Any

class NPUARModelRunner(OmniNPUModelRunner):
    """Autoregressive NPU model runner that returns hidden states per request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        self.omni_connector = None

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

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")

        #  -------------------------------------- Omni-new -------------------------------------------------
        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        self.kv_extracted_req_ids = self._handle_finished_requests_kv_transfer(scheduler_output)
        #  -------------------------------------- Omni-new -------------------------------------------------

        with ProfileExecuteDuration().capture_async("prepare input"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            self._update_states(scheduler_output)
            self._decode_and_store_request_payloads(scheduler_output)
            #  ------------------------------------------------------------------------------------------------

            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
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

            (attn_metadata, num_scheduled_tokens_np, num_input_tokens,
             num_tokens_across_dp, logits_indices, spec_decode_metadata,
             max_query_len) = self._prepare_inputs(scheduler_output)

            (input_ids, inputs_embeds, positions, intermediate_tensors,
             model_kwargs, ec_connector_output) = self._preprocess(scheduler_output,
                                              num_input_tokens,
                                              intermediate_tensors)

            # update global cos, sin
            update_cos_sin(positions)

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        # prevent debugger is None
        if self.debugger is not None:
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
                num_tokens=num_input_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora
            )

        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators)

        # Run forward pass
        with ProfileExecuteDuration().capture_async("forward"):
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    num_actual_tokens=scheduler_output.
                    total_num_scheduled_tokens,
                    model_instance=self.model):
                self.maybe_setup_kv_connector(scheduler_output)

                hidden_states = self._generate_process_reqs_hidden_states(
                    num_input_tokens, input_ids, positions,
                    intermediate_tensors, inputs_embeds, model_kwargs)

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output)

            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states

        kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving)
        finished_sending = None
        finished_recving = None
        with ProfileExecuteDuration().capture_async("post process"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)

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
                    self.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
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
                        num_scheduled_tokens_np, kv_connector_output)
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return pool_output
                sample_hidden_states = hidden_states[logits_indices]
                #  -------------------------------------- Omni-new -------------------------------------------------
                # Try with sampling_metadata first; fall back to without for models that don't support it
                try:
                    logits = self.model.compute_logits(
                        sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                    )
                except TypeError:
                    logits = self.model.compute_logits(sample_hidden_states)
                #  -------------------------------------- Omni-new -------------------------------------------------
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
                ec_connector_output,
                multimodal_outputs, # Omni-new
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

        #  -------------------------------------- Omni-new -------------------------------------------------
        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None
        #  -------------------------------------- Omni-new -------------------------------------------------

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

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output,
                                  self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        with ProfileExecuteDuration().capture_async("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert self.spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                attn_metadata,
                aux_hidden_states,
                sample_hidden_states
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        with ProfileExecuteDuration().capture_async("Draft"):
            if self.speculative_config:
                use_padded_batch_for_eagle = self.speculative_config and \
                    self.speculative_config.use_eagle() and \
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

        self._process_additional_information_updates(
            hidden_states, multimodal_outputs, num_scheduled_tokens_np, scheduler_output
        )

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
        model_runner_output.kv_extracted_req_ids = kv_extracted_req_ids
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
            if self.debugger is not None:
                assert self.debugger is not None
                self.debugger.stop()
                self.debugger.step()
            return model_runner_output

        if self.debugger is not None:
            assert self.debugger is not None
            self.debugger.stop()
            self.debugger.step()
        return AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )

    def _generate_process_reqs_hidden_states(self, num_input_tokens,
                                             input_ids, positions,
                                             intermediate_tensors,
                                             inputs_embeds, model_kwargs):
        assert self.model is not None
        hidden_states = self._model_forward(input_ids=input_ids,
                                   positions=positions,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds,
                                   **model_kwargs)

        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL \
            and not self.use_sparse:
            if self.vllm_config.model_config.use_mla:
                if self.pcp_size * self.dcp_size > 1:
                    update_full_graph_params(self.attn_backend, self.update_stream, forward_context,
                                     num_input_tokens, self.vllm_config,
                                      self.vllm_config.speculative_config)

        if get_forward_context().sp_enabled and not isinstance(
                hidden_states, IntermediateTensors):
            hidden_states = self._all_gather_hidden_states_and_aux(
                hidden_states)
        return hidden_states if self.pcp_size == 1 else self.pcp_manager.get_restore_hidden_states(
            hidden_states)

    def _handle_finished_requests_kv_transfer(self, scheduler_output: SchedulerOutput) -> list[str]:
        """Handle KV cache transfer for finished requests.

        Returns list of request IDs that were processed (for Scheduler to free blocks).
        """
        finished_reqs = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if not finished_reqs:
            return []

        logger.debug(f"Processing KV transfer for {len(finished_reqs)} requests")

        extracted_ids = []
        for req_id, data in finished_reqs.items():
            try:
                seq_len = data.get("seq_len", 0)
                block_ids = data.get("block_ids", [])
                if not block_ids:
                    logger.warning(f"Request {req_id} has no block IDs, skipping")
                    continue

                # Extract KV cache from GPU blocks -> CPU tensors
                kv_data = self._extract_kv_cache(req_id, block_ids, seq_len)
                if kv_data:
                    # Transfer to downstream stage via connector
                    self._transfer_kv_cache(kv_data)

            except Exception as e:
                logger.error(f"Failed KV transfer for {req_id}: {e}")
            finally:
                extracted_ids.append(req_id)

        return extracted_ids

    def _extract_kv_cache(self, req_id: str, block_ids: list[int], seq_len: int) -> KVCacheTransferData | None:
        """Extract KV cache from GPU blocks for a single request."""
        num_layers = len(self.kv_caches)
        key_cache = [None] * num_layers
        value_cache = [None] * num_layers

        for layer_idx, kv_tensor in enumerate(self.kv_caches):
            # Validate block IDs
            max_block = kv_tensor.shape[1] - 1
            valid_ids = [bid for bid in block_ids if 0 <= bid <= max_block]
            if not valid_ids:
                continue

            # Extract and reshape: [2, n_blocks, block_size, n_heads, head_dim]
            # -> [2, seq_len, n_heads, head_dim]
            selected = kv_tensor[:, valid_ids]  # [2, n_valid, block_size, n_heads, head_dim]
            n_kv, n_blks, blk_sz, n_heads, d_head = selected.shape
            flat = selected.reshape(n_kv, n_blks * blk_sz, n_heads, d_head)
            if seq_len < flat.shape[1]:
                flat = flat[:, :seq_len]

            # Move to CPU
            flat_cpu = flat.detach().cpu().contiguous()
            key_cache[layer_idx] = flat_cpu[0]
            value_cache[layer_idx] = flat_cpu[1]

        if not any(k is not None for k in key_cache):
            return None

        return KVCacheTransferData(
            request_id=req_id,
            layer_blocks={"key_cache": key_cache, "value_cache": value_cache},
            block_ids=block_ids,
            metadata={
                "block_size": self.cache_config.block_size,
                "num_layers": num_layers,
                "dtype": str(self.cache_config.cache_dtype),
                "seq_len": seq_len,
            },
        )

    def _transfer_kv_cache(self, kv_data: KVCacheTransferData) -> None:
        """Transfer KV cache data to downstream stage via OmniConnector."""
        connector = self._get_or_create_connector()
        if not connector:
            return

        # Resolve global request ID if available
        transfer_req_id = self._resolve_global_request_id(kv_data.request_id)
        from_stage, to_stage = self._detect_transfer_stages()

        # Prepare data and transfer with retry
        data_dict = kv_data.to_dict()
        data_dict["request_id"] = transfer_req_id

        success, size, _ = self._transfer_with_retry(
            connector, from_stage, to_stage, f"kv_cache_{transfer_req_id}", data_dict
        )

        if success:
            logger.info(f"KV transfer OK: {transfer_req_id}, {size} bytes")
        else:
            logger.error(f"KV transfer FAILED: {transfer_req_id}")

    def _get_or_create_connector(self) -> Any | None:
        """Get existing connector or create one from config."""
        if self.omni_connector:
            return self.omni_connector

        from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
        from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

        config = self._get_omni_connector_config()
        if not config or not isinstance(config, dict):
            logger.warning("No valid OmniConnector config found")
            return None

        c_type = config.get("type")
        if not c_type:
            logger.error("OmniConnector config missing 'type' field")
            return None

        c_extra = {k: v for k, v in config.items() if k != "type"}
        self.omni_connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=c_type, extra=c_extra))
        return self.omni_connector

    def _get_omni_connector_config(self) -> dict[str, Any] | None:
        """Get OmniConnector configuration from model config."""
        # Primary: omni_kv_config from YAML
        omni_kv = getattr(self.model_config, "omni_kv_config", None)
        if isinstance(omni_kv, dict):
            cfg = omni_kv.get("connector_config")
            if isinstance(cfg, dict) and cfg:
                return cfg

        # Fallback: kv_transfer_config
        kv_cfg = getattr(self.vllm_config, "kv_transfer_config", None)
        if kv_cfg:
            direct = getattr(kv_cfg, "omni_connector_config", None)
            if isinstance(direct, dict) and direct:
                return direct
            extra = getattr(kv_cfg, "kv_connector_extra_config", None)
            if isinstance(extra, dict):
                omni = extra.get("omni_connector_config")
                if isinstance(omni, dict) and omni:
                    return omni

        return None

    def _detect_transfer_stages(self) -> tuple[str, str]:
        """Detect source and target stages for KV transfer."""
        omni_kv = getattr(self.model_config, "omni_kv_config", None)
        if isinstance(omni_kv, dict):
            from_s = omni_kv.get("omni_from_stage")
            to_s = omni_kv.get("omni_to_stage")
            if from_s and to_s:
                return str(from_s), str(to_s)

        raise ValueError(
            "KV transfer stages not configured. Please set 'omni_from_stage' and 'omni_to_stage' in omni_kv_config."
        )

    def _resolve_global_request_id(self, req_id: str) -> str:
        """Resolve global request ID from request state."""
        req_state = self.requests.get(req_id)
        if not req_state:
            return req_id

        add_info = getattr(req_state, "additional_information_cpu", {}) or {}
        global_id = add_info.get("global_request_id")
        if global_id:
            if isinstance(global_id, list) and global_id:
                global_id = global_id[0]
            if isinstance(global_id, bytes):
                return global_id.decode("utf-8")
            return str(global_id)
        return req_id

    def _transfer_with_retry(
        self,
        connector: Any,
        from_stage: str,
        to_stage: str,
        request_id: str,
        data: dict[str, Any],
        max_retries: int = 3,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Transfer data with retry and exponential backoff."""
        import time

        for attempt in range(max_retries):
            try:
                put_key = f"omni_{from_stage}_to_{to_stage}_{request_id}"
                success, size, metadata = connector.put(
                    from_stage=from_stage, to_stage=to_stage, put_key=put_key, data=data
                )
                if success:
                    return success, size, metadata
                logger.warning(f"Transfer attempt {attempt + 1} failed for {request_id}")
            except Exception as e:
                logger.warning(f"Transfer attempt {attempt + 1} exception: {e}")

            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))

        return False, 0, None
