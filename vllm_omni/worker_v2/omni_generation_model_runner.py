"""OmniGenerationModelRunner — non-autoregressive stage runner on MR V2.

Used for stages like Code2Wav that convert codec codes to audio waveforms.
No token sampling or logits computation — model output goes directly into
``pooler_output``.  Inherits from ``OmniGPUModelRunner`` for intermediate
buffer and lifecycle hooks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    get_ep_all2all_manager,
)
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu.model_runner import (
    BatchDescriptor,
    ExecuteModelState,
    IntermediateTensors,
)

from vllm_omni.core.sched.output import OmniCachedRequestData, OmniNewRequestData
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.omni_ar_model_runner import (
    _async_copy_mm,
    _ensure_tensor_values,
)
from vllm_omni.worker_v2.omni_model_runner import (
    OmniGPUModelRunner,
)

logger = init_logger(__name__)


class OmniGenerationAsyncOutput(AsyncModelRunnerOutput):
    """Overlap generation-stage D2H copies with the next model step."""

    def __init__(
        self,
        *,
        model_runner_output: OmniModelRunnerOutput,
        multimodal_outputs: dict[str, Any] | None,
        num_reqs: int,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        finalize_output: Any | None = None,
        check_ep_fault: bool = False,
    ) -> None:
        self.model_runner_output = model_runner_output
        self.num_reqs = num_reqs
        self.finalize_output = finalize_output
        self.copy_event = torch.cuda.Event(blocking=True)
        self._has_fault: torch.Tensor | None = None

        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(main_stream)
            self.multimodal_outputs_cpu = _async_copy_mm(
                multimodal_outputs,
                total_tokens=0,
                copy_stream=copy_stream,
                pin_memory=PIN_MEMORY,
            )
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)

    def get_output(self) -> OmniModelRunnerOutput:
        self.copy_event.synchronize()
        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: one or more ranks "
                f"timed out during dispatch/combine. Mask: {mask.cpu().tolist()}"
            )
        payloads = OmniGenerationModelRunner._build_pooler_output_from_cpu(
            self.multimodal_outputs_cpu,
            self.num_reqs,
        )
        self.model_runner_output.multimodal_outputs = [
            _ensure_tensor_values(payload) if payload else {} for payload in payloads
        ]
        if self.finalize_output is not None:
            return self.finalize_output(self.model_runner_output)
        return self.model_runner_output


class OmniGenerationModelRunner(OmniGPUModelRunner):
    """Non-autoregressive generation runner (e.g. Code2Wav).

    Overrides ``execute_model`` to skip the tensor-only assertion and
    ``sample_tokens`` to construct ``pooler_output`` from multimodal
    model outputs without performing token sampling.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gen_model_output: Any = None
        self._gen_input_batch: Any = None
        # Placeholder for ExecuteModelState.hidden_states — allocated
        # once and reused every step to avoid per-forward allocation.
        self._dummy_hidden = torch.zeros(1, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------
    # Async-chunk support: replace prompt_token_ids for cached requests
    # ------------------------------------------------------------------

    def _handle_async_chunk_updates(self, scheduler_output: SchedulerOutput) -> None:
        """In-place update cached requests whose prompt_token_ids changed.

        In async_chunk mode, the ``ChunkTransferAdapter`` replaces
        ``Request.prompt_token_ids`` with new codec frames for each
        chunk and resets ``num_computed_tokens`` to 0.  The scheduler
        propagates the new ``prompt_token_ids`` via
        ``OmniCachedRequestData``.

        Instead of remove + re-add (which involves free_indices churn
        and redundant model_state init), we update the existing slot
        in-place.  This is safe for Code2Wav because:
        - No KV cache / rope state to reinitialize
        - staged writes are applied once at the end

        The old intermediate buffer for this slot is cleared here; the
        inherited ``OmniGPUModelRunner.update_requests`` (called right after
        this method in ``execute_model``) writes the current chunk state.
        """
        cached = scheduler_output.scheduled_cached_reqs
        if not cached.req_ids:
            return

        if not isinstance(cached, OmniCachedRequestData):
            return

        new_prompt_ids = cached.prompt_token_ids
        if not new_prompt_ids:
            return

        updated = False
        released_chunks: list[OmniNewRequestData] = []

        for i, req_id in enumerate(cached.req_ids):
            new_ids = new_prompt_ids.get(req_id)
            if new_ids is None:
                continue

            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is None:
                block_ids = cached.new_block_ids[i]
                released_chunks.append(
                    OmniNewRequestData(
                        req_id=req_id,
                        prompt_token_ids=new_ids,
                        mm_features=[],
                        sampling_params=None,
                        pooling_params=None,
                        block_ids=block_ids if block_ids is not None else tuple(),
                        num_computed_tokens=0,
                        lora_request=None,
                        prompt_embeds=None,
                        prefill_token_ids=new_ids,
                        additional_information=cached.additional_information.get(req_id),
                    )
                )
                continue

            self.model_state.remove_request(req_idx)

            # In-place update token state — same slot, no remove/re-add.
            # .np[] = direct write (no GPU buffer); stage_write = GPU-synced.
            n = len(new_ids)
            self.req_states.prompt_len.np[req_idx] = n
            self.req_states.prefill_len.np[req_idx] = n
            self.req_states.total_len.stage_write_elem(req_idx, n)
            self.req_states.all_token_ids.stage_write(req_idx, 0, new_ids)
            self.req_states.num_computed_tokens.stage_write_elem(req_idx, 0)
            self.req_states.num_computed_prefill_tokens[req_idx] = 0

            updated = True

        if released_chunks:
            self.add_requests(SimpleNamespace(scheduled_new_reqs=released_chunks))
        if updated:
            self.req_states.apply_staged_writes()

    def _release_generation_slots(self, input_batch: Any) -> None:
        if not getattr(self.model_config, "async_chunk", False):
            return
        model_state = getattr(self, "model_state", None)
        remove_request = getattr(self, "_remove_request", None)
        if model_state is None or remove_request is None:
            return
        for i in range(input_batch.num_reqs):
            req_id = input_batch.req_ids[i]
            req_idx = int(input_batch.idx_mapping_np[i])
            model_state.remove_request(req_idx)
            remove_request(req_id)

    # ------------------------------------------------------------------
    # profile / warmup — skip sampler since there are no logits
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Generation models have no KV cache — skip profiling.

        Code2Wav shares GPU memory with the Talker stage (same device);
        its memory footprint is managed via ``gpu_memory_utilization``
        config, not profiled dynamically.  Running the real model with
        random input_ids causes out-of-bounds indexing in codec lookup
        tables.
        """
        torch.accelerator.synchronize()

    # ------------------------------------------------------------------
    # execute_model — run the generation model, store raw output
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            self._prepare_native_data_plane(scheduler_output)
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            # Handle async_chunk prompt_token_ids replacement for cached
            # requests BEFORE add/update — update the existing slot
            # in-place with the new chunk's tokens.
            self._handle_async_chunk_updates(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self._sync_native_data_plane_payloads(scheduler_output)
            self._apply_block_table_staged_writes_if_available()
            if scheduler_output.total_num_scheduled_tokens == 0:
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return self._attach_native_data_plane_signals(
                    self._merge_ec_connector_no_forward(scheduler_output, empty_output)
                )

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        batch_req_state, uniform_tok_count = self.gather_batch_req_state(scheduler_output, dummy_run)
        if batch_req_state is not None:
            num_toks = batch_req_state.num_tokens
        batch_desc, _ = self._dispatch_batch_descriptor(
            num_reqs=num_reqs,
            num_toks=num_toks,
            uniform_tok_count=uniform_tok_count,
            num_active_loras=0,
            use_eager=is_profile,
            max_query_len=max_query_len,
        )

        if batch_desc.num_tokens == 0:
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return self._attach_native_data_plane_signals(
                self._merge_ec_connector_no_forward(scheduler_output, empty_output)
            )

        if not dummy_run:
            assert batch_req_state is not None
            input_batch = self.prepare_inputs(scheduler_output, batch_req_state, batch_desc)
        else:
            from vllm.v1.worker.gpu.input_batch import InputBatch

            input_batch = InputBatch.make_dummy(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
                max_query_len=batch_desc.max_query_len,
            )

        attn_metadata = None
        slot_mappings_by_layer = None

        input_ids, inputs_embeds, ec_connector_output = self._prepare_mm_inputs(
            scheduler_output,
            input_batch,
            dummy_run=dummy_run,
        )

        model_inputs = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": intermediate_tensors,
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        self._add_legacy_forward_inputs(model_inputs, input_batch)

        eplb = getattr(self, "eplb", None)
        if eplb is not None:
            eplb.prepare_forward(self.model_config, input_batch.num_tokens)

        batch_descriptor = BatchDescriptor(
            num_tokens=input_batch.num_tokens_after_padding,
            has_lora=self.lora_config is not None,
            num_active_loras=batch_desc.num_active_loras,
        )
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=input_batch.num_tokens_after_padding,
            cudagraph_runtime_mode=batch_desc.cg_mode,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mappings_by_layer,
            is_padding=input_batch.is_padding,
        ):
            self.kv_connector.pre_forward(scheduler_output)
            if batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                assert self.cudagraph_manager is not None
                model_output = self.cudagraph_manager.run_pw_graph(
                    self.model,
                    model_inputs,
                )
            else:
                model_output = self.model(**model_inputs)

        # Convert raw model output to OmniOutput.
        if not isinstance(model_output, OmniOutput):
            buffer_list = self.model_state.intermediate_buffer.gather(input_batch)
            make_output_kwargs = {"model_intermediate_buffer": buffer_list}
            if not getattr(self.model, "requires_native_model_intermediate_buffer", False):
                make_output_kwargs["runtime_additional_information"] = buffer_list
            model_output = self.model.make_omni_output(model_output, **make_output_kwargs)

        self._gen_model_output = model_output
        self._gen_input_batch = input_batch

        # ExecuteModelState is required by the upstream engine loop
        # (EngineCore checks execute_model_state is not None before
        # calling sample_tokens).
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=self._dummy_hidden,
            aux_hidden_states=None,
            finished_req_ids=scheduler_output.finished_req_ids,
            ec_connector_output=ec_connector_output,
            routed_experts=None,
        )
        return None

    # ------------------------------------------------------------------
    # sample_tokens — build pooler_output, no actual sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | None:
        model_output = self._gen_model_output
        input_batch = self._gen_input_batch
        execute_model_state = self.execute_model_state
        self._gen_model_output = None
        self._gen_input_batch = None
        self.execute_model_state = None

        if model_output is None or input_batch is None or execute_model_state is None:
            return None

        kv_connector_output = self.kv_connector.post_forward(execute_model_state.finished_req_ids)
        num_reqs = input_batch.num_reqs

        # Mark all scheduled tokens as computed so the scheduler does
        # not re-schedule them.  Unlike AR stages we do NOT call
        # self.postprocess() — that kernel advances num_computed_tokens
        # by 1 and emits sampled tokens, which would cause check_stop
        # to fire.  Instead, set num_computed_tokens = prompt_len
        # directly, matching V1's behavior.
        for i in range(num_reqs):
            req_idx = int(input_batch.idx_mapping_np[i])
            prompt_len = int(self.req_states.prompt_len.np[req_idx])
            self.req_states.num_computed_tokens.stage_write_elem(req_idx, prompt_len)
        self.req_states.num_computed_tokens.apply_write()

        # Async finalization can run after the input batch is reused for the
        # next scheduler step, so output metadata must not alias the batch.
        req_ids = list(input_batch.req_ids)

        # Generation models don't do token sampling.  Return one empty
        # list per request so the scheduler does NOT trigger check_stop
        # (which would prematurely finish the request).  The request
        # stays RUNNING until the orchestrator marks it done via
        # chunk_transfer_adapter.finished_requests.
        sampled_token_ids: list[list[int]] = [[] for _ in range(len(req_ids))]
        output = OmniModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            # Match the V1 generation runner contract: final generation stages
            # publish audio only through multimodal_outputs.  If the same
            # payload is also exposed as pooling_output, OutputProcessor routes
            # it through the pooling path and can duplicate accumulated audio.
            pooler_output=None,
            multimodal_outputs=None,
            kv_connector_output=kv_connector_output,
            ec_connector_output=execute_model_state.ec_connector_output,
        )

        raw_multimodal_outputs = model_output.multimodal_outputs
        can_copy_async = getattr(getattr(self, "device", None), "type", "cpu") == "cuda" and isinstance(
            raw_multimodal_outputs, (dict, type(None))
        )
        if can_copy_async:
            async_output = OmniGenerationAsyncOutput(
                model_runner_output=output,
                multimodal_outputs=raw_multimodal_outputs,
                num_reqs=num_reqs,
                main_stream=self.main_stream,
                copy_stream=self.output_copy_stream,
                finalize_output=self._finalize_native_data_plane_output,
                check_ep_fault=self.check_ep_fault,
            )
            self._reserve_native_data_plane_outputs(list(req_ids))
            self._release_generation_slots(input_batch)
            return async_output

        # CPU execution keeps the synchronous materialization path. Final
        # generation stages publish audio only through multimodal_outputs,
        # matching the V1 generation-runner contract.
        self._reserve_native_data_plane_outputs(list(req_ids))
        multimodal_outputs = self._build_pooler_output(model_output, num_reqs)
        output.multimodal_outputs = [
            _ensure_tensor_values(payload) if payload else {} for payload in multimodal_outputs
        ]
        self._release_generation_slots(input_batch)
        return self._finalize_native_data_plane_output(output)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_block_table_staged_writes_if_available(self) -> None:
        """Flush block-table writes only when this stage has a writer.

        Generation stages such as Code2Wav do not use KV cache or attention
        metadata. Newer vLLM BlockTable implementations assert that a fused
        writer exists before applying staged writes; no-KV generation stages
        legitimately do not have one.
        """
        block_tables = getattr(self, "block_tables", None)
        if block_tables is None:
            return
        if getattr(block_tables, "fused_writer", None) is None:
            return
        block_tables.apply_staged_writes()

    @staticmethod
    def _build_pooler_output(
        model_output: OmniOutput,
        num_reqs: int,
    ) -> list[dict[str, Any] | None]:
        """Extract per-request pooler payloads from model output.

        Code2Wav's ``make_omni_output`` returns
        ``{"model_outputs": [tensor_per_req, ...], "sr": [...]}``,
        so each value is a ``list`` with ``len == num_reqs``.
        """
        mm = model_output.multimodal_outputs
        if not isinstance(mm, dict):
            logger.warning(
                "Unexpected multimodal_outputs type: %s; returning empty pooler_output",
                type(mm).__name__ if mm is not None else "None",
            )
            return [None] * num_reqs

        pooler: list[dict[str, Any] | None] = []
        for i in range(num_reqs):
            payload: dict[str, Any] = {}
            for key, val in mm.items():
                # Primary path: val is list[Tensor] with len == num_reqs
                # (Code2Wav make_omni_output format).
                if isinstance(val, list) and len(val) == num_reqs:
                    out = val[i]
                    payload[key] = out.detach().cpu().contiguous() if isinstance(out, torch.Tensor) else out
                elif isinstance(val, torch.Tensor):
                    if val.dim() > 0 and val.shape[0] == num_reqs:
                        payload[key] = val[i].detach().cpu().contiguous()
                    else:
                        payload[key] = val.detach().cpu().contiguous()
                else:
                    payload[key] = val
            pooler.append(payload)
        return pooler

    @staticmethod
    def _build_pooler_output_from_cpu(
        multimodal_outputs: dict[str, Any] | None,
        num_reqs: int,
    ) -> list[dict[str, Any] | None]:
        """Build per-request payloads after an async whole-batch D2H copy."""
        if not isinstance(multimodal_outputs, dict):
            return [None] * num_reqs

        pooler: list[dict[str, Any] | None] = []
        for i in range(num_reqs):
            payload: dict[str, Any] = {}
            for key, value in multimodal_outputs.items():
                if isinstance(value, list) and len(value) == num_reqs:
                    out = value[i]
                    payload[key] = out.contiguous() if isinstance(out, torch.Tensor) else out
                elif isinstance(value, torch.Tensor):
                    if value.dim() > 0 and value.shape[0] == num_reqs:
                        payload[key] = value[i].contiguous()
                    else:
                        payload[key] = value.contiguous()
                else:
                    payload[key] = value
            pooler.append(payload)
        return pooler
