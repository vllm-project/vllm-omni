# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniGenerationModelRunner — non-autoregressive stage runner on MR V2.

Used for stages like Code2Wav that convert codec codes to audio waveforms.
No token sampling or logits computation — model output goes directly into
``pooler_output``.  Inherits from ``OmniGPUModelRunner`` for intermediate
buffer and lifecycle hooks.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
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
from vllm_omni.data_entry_keys import flatten_payload
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


def _contains_cuda_tensor(value: Any) -> bool:
    """Return whether a nested output payload still owns CUDA storage."""
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, dict):
        return any(_contains_cuda_tensor(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_cuda_tensor(v) for v in value)
    return False


def check_exact_input_shape(model: Any, input_ids: torch.Tensor | None, num_tokens: int) -> None:
    """Enforce ``requires_exact_input_shape`` (#6712).

    Such models split the flat ``input_ids`` by per-request
    ``seq_token_counts``, so padding rows would misalign every request. V1
    trims cudagraph padding for them (``maybe_unpad_input_ids``); this runner
    dispatches eagerly and must already hand over the exact batch.
    """
    if input_ids is None or not getattr(model, "requires_exact_input_shape", False):
        return
    if int(input_ids.shape[0]) != int(num_tokens):
        raise RuntimeError(
            f"{type(model).__name__} requires an exact input shape: got {int(input_ids.shape[0])} "
            f"input ids for {int(num_tokens)} scheduled tokens"
        )


def _materialize_generation_value(
    value: Any,
    req_index: int,
    num_reqs: int,
    *,
    top_level: bool = True,
    copy_cpu: bool = True,
) -> Any:
    """Recursively select and own one request's generation payload."""
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == num_reqs:
            value = value[req_index]
        return value.detach().to(device="cpu", copy=copy_cpu and value.device.type == "cpu").contiguous()
    if isinstance(value, dict):
        return {
            key: _materialize_generation_value(item, req_index, num_reqs, top_level=False, copy_cpu=copy_cpu)
            for key, item in value.items()
        }
    if isinstance(value, list):
        # Tensor lists (e.g. variable-length waveforms) carry one item per
        # request even inside a dict. Scalar lists remain payload metadata.
        per_request = top_level or all(item is None or isinstance(item, torch.Tensor) for item in value)
        if per_request and len(value) == num_reqs:
            # The selected item no longer has a batch axis.
            return _materialize_generation_value(value[req_index], 0, -1, top_level=False, copy_cpu=copy_cpu)
        return [
            _materialize_generation_value(item, req_index, num_reqs, top_level=False, copy_cpu=copy_cpu)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _materialize_generation_value(item, req_index, num_reqs, top_level=False, copy_cpu=copy_cpu)
            for item in value
        )
    return value


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
        pending_aux_output: Any | None = None,
    ) -> None:
        self.model_runner_output = model_runner_output
        self.num_reqs = num_reqs
        self.finalize_output = finalize_output
        self.pending_aux_output = pending_aux_output
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
            if pending_aux_output is not None:
                # Generation stages emit no sampled/rejected token IDs.
                counts = np.zeros(num_reqs, dtype=np.int32)
                pending_aux_output.enqueue_cpu_copy(num_sampled=counts, num_rejected=counts)
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)

    def get_output(self) -> OmniModelRunnerOutput:
        self.copy_event.synchronize()
        if self.pending_aux_output is not None:
            self.model_runner_output.aux_output_connector_output = self.pending_aux_output.process_output()
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
        self._prev_step_idle = False
        self._failed_optional_model_hooks: set[str] = set()
        self._gen_model_output: Any = None
        self._gen_input_batch: Any = None
        # Placeholder for ExecuteModelState.hidden_states — allocated
        # once and reused every step to avoid per-forward allocation.
        self._dummy_hidden = torch.zeros(1, dtype=self.dtype, device=self.device)
        self._retains_state_across_chunks = bool(getattr(self.model_config, "retains_state_across_chunks", False))

    def _call_optional_model_hook(self, name: str, *args: object) -> None:
        """Keep optional cache warmup failures from terminating the engine."""
        hook = getattr(self.model, name, None)
        if not callable(hook):
            return
        try:
            hook(*args)
        except Exception:
            # Idle polling is frequent; retain diagnostics without log flooding.
            log = logger.debug if name in self._failed_optional_model_hooks else logger.warning
            self._failed_optional_model_hooks.add(name)
            log("Optional model hook %s failed; continuing without it.", name, exc_info=True)

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

            block_ids = cached.new_block_ids[i]
            chunk_data = OmniNewRequestData(
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
            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is None:
                released_chunks.append(chunk_data)
                continue

            # Reset chunk-local model inputs using the same initialization as
            # re-adding a released chunk, without recycling the runner slot.
            self.model_state.remove_request(req_idx)
            self.model_state.add_request(req_idx, chunk_data)

            # In-place update token state — same slot, no remove/re-add.
            # .np[] = direct write (no GPU buffer); stage_write = GPU-synced.
            n = len(new_ids)
            # Match add_request's max_tokens=1 fallback for chunk_data above.
            self.req_states.max_seq_len[req_idx] = n + 1
            self.req_states.draft_tokens[req_idx].zero_()
            self.req_states.prompt_len.np[req_idx] = n
            self.req_states.prefill_len.np[req_idx] = n
            self.req_states.total_len.stage_write_elem(req_idx, n)
            self.req_states.all_token_ids.stage_write(req_idx, 0, new_ids)
            self.req_states.num_computed_tokens.stage_write_elem(req_idx, 0)
            self.req_states.num_computed_prefill_tokens[req_idx] = 0

            updated = True

        if released_chunks:
            self.add_requests(
                replace(
                    scheduler_output,
                    scheduled_new_reqs=released_chunks,
                )
            )
        if updated:
            # Match add_requests: collect request and model-state writes, then
            # flush them together before subsequent input preparation.
            with self._staged_writes.deferred():
                self.req_states.apply_staged_writes()
                self.model_state.apply_staged_writes()

    def _release_generation_slots(self, input_batch: Any, keep: set[str] | frozenset[str] = frozenset()) -> None:
        if not getattr(self.model_config, "async_chunk", False):
            return
        for req_id in input_batch.req_ids:
            if req_id not in keep:
                self._remove_request(req_id)

    def _pending_generation_req_ids(self, req_ids: list[str]) -> set[str]:
        """Requests whose model-owned generation continues past this step.

        As in the V1 generation runner, the scheduler re-runs their current
        input unit instead of finishing it.
        """
        get_pending = getattr(getattr(self, "model", None), "get_pending_generation_request_ids", None)
        if get_pending is None:
            return set()
        return set(req_ids).intersection(get_pending())

    # ------------------------------------------------------------------
    # profile / warmup — skip sampler since there are no logits
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def profile_run(self, randomize_inputs: bool = False) -> None:
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
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        prev_step_idle = self._prev_step_idle
        # Dummy forwards also break consecutive idle polls: their device work
        # may still be in flight when the next real idle step arrives.
        self._prev_step_idle = scheduler_output.total_num_scheduled_tokens == 0
        if not dummy_run:
            self._prepare_native_data_plane(scheduler_output)
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            # Finish old state before queuing a prewarm for a reused request id.
            if prewarms := getattr(scheduler_output, "pending_request_prewarms", None):
                self._call_optional_model_hook("on_requests_added", prewarms)
            if (
                scheduler_output.total_num_scheduled_tokens == 0
                and not scheduler_output.scheduled_new_reqs
                and not scheduler_output.scheduled_cached_reqs.req_ids
            ):
                # Registration/terminal/abort work has been applied above.
                # There is no new or cached request state to add or update.
                self._apply_block_table_staged_writes_if_available()
                if self.aux_output_connector is not None:
                    self.aux_output_connector.begin_step(scheduler_output.aux_output_connector_metadata)
                # The first idle step may still overlap the previous output.
                # Match V1: warm only on a subsequent idle poll, in the same
                # inference mode and stream as the model forward.
                if prev_step_idle:
                    self._call_optional_model_hook("run_idle_prefetch")
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return self._attach_native_data_plane_signals(
                    self._merge_ec_connector_no_forward(scheduler_output, empty_output)
                )
            # Handle async_chunk prompt_token_ids replacement for cached
            # requests BEFORE add/update — update the existing slot
            # in-place with the new chunk's tokens.
            self._handle_async_chunk_updates(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self._sync_native_data_plane_payloads(scheduler_output)
            self._apply_block_table_staged_writes_if_available()
            if self.aux_output_connector is not None:
                self.aux_output_connector.begin_step(scheduler_output.aux_output_connector_metadata)
            if scheduler_output.total_num_scheduled_tokens == 0:
                if prev_step_idle:
                    self._call_optional_model_hook("run_idle_prefetch")
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
        batch_desc, dp_sync = self._dispatch_batch_descriptor(
            num_reqs=num_reqs,
            num_toks=num_toks,
            uniform_tok_count=uniform_tok_count,
            num_active_loras=0,
            # Generation stages return structured OmniOutput payloads and do
            # not expose the tensor-only outer graph contract. Their model may
            # still use an internal decoder graph, but the vLLM outer runner
            # always uses eager dispatch here.
            use_eager=True,
            max_query_len=max_query_len,
        )

        if batch_desc.num_tokens == 0:
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return self._attach_native_data_plane_signals(
                self._merge_ec_connector_no_forward(scheduler_output, empty_output)
            )

        if not dummy_run:
            assert batch_req_state is not None
            input_batch = self.prepare_inputs(scheduler_output, batch_req_state, batch_desc, 0)
        else:
            from vllm.v1.worker.gpu.input_batch import InputBatch

            input_batch = InputBatch.make_dummy(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
                max_query_len=batch_desc.max_query_len,
                is_padding=not is_profile,
            )
            # Codec IDs have model-specific bounds; keep valid dummy tokens
            # even when upstream requests randomized language-model inputs.

        attn_metadata = None
        slot_mappings_by_layer = None

        input_ids, inputs_embeds, ec_connector_output = self._prepare_mm_inputs(
            scheduler_output,
            input_batch,
            dummy_run=dummy_run,
        )
        if not dummy_run:
            check_exact_input_shape(self.model, input_ids, input_batch.num_tokens)

        model_inputs = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": intermediate_tensors,
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        if getattr(self.model, "requires_request_ids", False):
            # Request-owned model state (e.g. MiniCPM-o's vocoder caches) is
            # keyed by the scheduler request id, in batch order.
            model_inputs["request_ids"] = list(input_batch.req_ids)
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
            dp_sync_state=dp_sync,
            finished_req_ids=scheduler_output.finished_req_ids,
            ec_connector_output=ec_connector_output,
            cudagraph_stats=None,
            num_spec_tokens_to_schedule=scheduler_output.num_spec_tokens_to_schedule,
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

        num_reqs = input_batch.num_reqs
        pending_aux_output = None
        if self.aux_output_connector is not None:
            pending_aux_output = self.aux_output_connector.prepare_output(input_batch)

        # Async finalization can run after the input batch is reused for the
        # next scheduler step, so output metadata must not alias the batch.
        req_ids = list(input_batch.req_ids)
        pending = self._pending_generation_req_ids(req_ids)

        # Mark all scheduled tokens as computed so the scheduler does
        # not re-schedule them.  Unlike AR stages we do NOT call
        # self.postprocess() — that kernel advances num_computed_tokens
        # by 1 and emits sampled tokens, which would cause check_stop
        # to fire.  Instead, set num_computed_tokens = prompt_len
        # directly, matching V1's behavior.  Pending generations keep
        # their count: the scheduler rolls it back to re-run the unit.
        for i in range(num_reqs):
            if req_ids[i] in pending:
                continue
            req_idx = int(input_batch.idx_mapping_np[i])
            prompt_len = int(self.req_states.prompt_len.np[req_idx])
            self.req_states.num_computed_tokens.stage_write_elem(req_idx, prompt_len)
        self.req_states.num_computed_tokens.apply_write()
        kv_connector_output = self.kv_connector.post_forward(execute_model_state.finished_req_ids)

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
            sampled_token_ids_materialized=True,
        )
        output.pending_generation_req_ids = pending

        raw_multimodal_outputs = model_output.multimodal_outputs
        can_copy_async = getattr(getattr(self, "device", None), "type", "cpu") == "cuda" and (
            self.check_ep_fault or pending_aux_output is not None or _contains_cuda_tensor(raw_multimodal_outputs)
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
                pending_aux_output=pending_aux_output,
            )
            self._reserve_native_data_plane_outputs(list(req_ids))
            if not self._retains_state_across_chunks:
                self._release_generation_slots(input_batch, keep=pending)
            return async_output

        # CPU execution keeps the synchronous materialization path. Final
        # generation stages publish audio only through multimodal_outputs,
        # matching the V1 generation-runner contract.
        self._reserve_native_data_plane_outputs(list(req_ids))
        multimodal_outputs = self._build_pooler_output(model_output, num_reqs)
        output.multimodal_outputs = [
            _ensure_tensor_values(payload) if payload else {} for payload in multimodal_outputs
        ]
        if not self._retains_state_across_chunks:
            self._release_generation_slots(input_batch, keep=pending)
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
            payload = {key: _materialize_generation_value(value, i, num_reqs) for key, value in mm.items()}
            pooler.append(flatten_payload(payload) if isinstance(payload, dict) else payload)
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
            payload = {
                key: _materialize_generation_value(value, i, num_reqs, copy_cpu=False)
                for key, value in multimodal_outputs.items()
            }
            pooler.append(flatten_payload(payload) if isinstance(payload, dict) else payload)
        return pooler
