"""Omni v2 GPU model runner hooks."""

from __future__ import annotations

import threading
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import (
    BatchDescriptor,
    BatchExecutionDescriptor,
    ExecuteModelState,
    GPUModelRunner,
    IntermediateTensors,
    build_slot_mappings_by_layer,
    get_uniform_token_count,
)

from vllm_omni.compat import make_filtered_namedtuple
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.forward_compat import add_forward_compat_kwargs
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.intermediate_buffer import (
    _resolve_additional_information,
)
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

logger = init_logger(__name__)

_model_state_patch_lock = threading.RLock()
_GPU_MODEL_RUNNER_HAS_SHUTDOWN = hasattr(GPUModelRunner, "shutdown")
# Fields the omni runners still pass to ExecuteModelState that current vLLM no
# longer declares on the namedtuple. They are filtered out by
# make_filtered_namedtuple and carried out-of-band instead:
#  - num_tokens_across_dp: only consumed inside execute_model's forward context.
#  - kv_connector_output: removed from ExecuteModelState in vLLM 0.23.0 (post_forward
#    now runs in sample_tokens). Carried via self._last_kv_connector_output (AR) /
#    self._gen_kv_connector_output (generation).
_KNOWN_EXECUTE_MODEL_STATE_COMPAT_FIELDS = {"num_tokens_across_dp", "kv_connector_output"}


def _make_execute_model_state(**kwargs):
    state, unknown = make_filtered_namedtuple(
        ExecuteModelState,
        known_extra_fields=_KNOWN_EXECUTE_MODEL_STATE_COMPAT_FIELDS,
        **kwargs,
    )
    if unknown:
        logger.warning("Unknown fields passed to ExecuteModelState: %s", sorted(unknown))
    return state


def _needs_capture_tensor_unwrap(model: Any) -> bool:
    return bool(getattr(model, "_returns_tuple", False) or getattr(model, "model_stage", None) == "thinker")


class OmniGPUModelRunner(GPUModelRunner):
    """Thin layer over v2 ``GPUModelRunner`` for Omni lifecycle hooks."""

    model_state: OmniModelState
    _last_aux_output: Any
    _last_multimodal_outputs: dict[str, Any] | None
    _model_returns_tuple: bool

    def shutdown(self) -> None:
        if _GPU_MODEL_RUNNER_HAS_SHUTDOWN:
            super().shutdown()

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        import vllm.v1.worker.gpu.model_runner as _mr_module

        with _model_state_patch_lock:
            _orig = _mr_module.init_model_state
            _mr_module.init_model_state = init_omni_model_state
            try:
                super().load_model(*args, **kwargs)
            finally:
                _mr_module.init_model_state = _orig
        self._last_aux_output = None
        self._last_multimodal_outputs = None
        self._last_kv_connector_output = None
        self._model_returns_tuple = _needs_capture_tensor_unwrap(self.model)
        self._exclude_full_graph = self._model_returns_tuple or hasattr(self.model, "_last_captured_layers")

        # Preprocess models own embedding buffers; encoder_runner sizing would mismatch.
        if getattr(self.model, "has_preprocess", False) and self.supports_mm_inputs:
            self.supports_mm_inputs = False
            self.encoder_cache = None

    # ------------------------------------------------------------------
    # CUDA Graph: conditionally exclude FULL mode
    # ------------------------------------------------------------------

    def capture_model(self) -> int:
        """Handle CUDA graph capture for Omni models.

        Exclude FULL mode for tuple-returning models because
        ``run_fullgraph`` bypasses Python-level tuple intercept.

        For PIECEWISE capture, the warmup pass runs with
        ``CUDAGraphMode.NONE`` which hits ``torch.empty_like(hidden_states)``
        in the cudagraph framework.  If the model returns a tuple, that call
        crashes.  We temporarily wrap the model's forward to extract only the
        tensor part during capture, then restore the original forward.
        """
        if self._exclude_full_graph:
            mgr = self.cudagraph_manager
            # NOTE: ``_capture_descs`` / ``_candidates`` are private CudaGraphManager
            # internals (present and same-typed as of vLLM 0.23.0). Guard with hasattr
            # so a future vLLM that renames/removes them degrades to a warning instead
            # of an AttributeError; the worst case is FULL graphs not being excluded.
            if hasattr(mgr, "_capture_descs") and hasattr(mgr, "_candidates"):
                if CUDAGraphMode.FULL in mgr._capture_descs:
                    del mgr._capture_descs[CUDAGraphMode.FULL]
                    for i, descs in enumerate(mgr._candidates):
                        mgr._candidates[i] = [d for d in descs if d.cg_mode != CUDAGraphMode.FULL]
                    logger.info("Excluded FULL CUDA graph capture for Omni model. PIECEWISE graphs will still be captured.")
            else:
                logger.warning(
                    "CudaGraphManager lacks _capture_descs/_candidates; cannot exclude "
                    "FULL graph capture for tuple-returning Omni model. vLLM internals "
                    "may have changed — verify omni_model_runner.capture_model()."
                )

        # Wrap model forward during capture so tuple returns don't crash
        # torch.empty_like() in the PIECEWISE warmup pass.
        if self._model_returns_tuple:
            original_forward = self.model.forward

            def _capture_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                output = original_forward(*args, **kwargs)
                if isinstance(output, OmniOutput):
                    return output.text_hidden_states
                if isinstance(output, tuple):
                    return output[0]
                return output

            self.model.forward = _capture_forward  # type: ignore[assignment]
            try:
                return super().capture_model()
            finally:
                self.model.forward = original_forward  # type: ignore[assignment]
        return super().capture_model()

    def _dispatch_batch_descriptor(
        self,
        *,
        num_reqs: int,
        num_toks: int,
        uniform_tok_count: int,
        use_eager: bool,
    ):
        if use_eager:
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_toks,
                num_reqs=num_reqs,
            )
        else:
            batch_desc = self.cudagraph_manager.dispatch(num_reqs, num_toks, uniform_tok_count)
        if self.dp_size > 1:
            from vllm.v1.worker.gpu.dp_utils import sync_cudagraph_and_dp_padding

            return sync_cudagraph_and_dp_padding(
                self.cudagraph_manager,
                batch_desc,
                num_toks,
                num_reqs,
                uniform_tok_count,
                self.dp_size,
                self.dp_rank,
            )
        return batch_desc, None

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> Any:
        if not dummy_run:
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                return self.kv_connector.no_forward(scheduler_output)

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)
        # Encoder-decoder models: disable compilation when encoder inputs
        # are scheduled (dynamic cross-attention cache updates).
        skip_compiled = self.is_encoder_decoder and bool(scheduler_output.scheduled_encoder_inputs)
        batch_desc, num_tokens_across_dp = self._dispatch_batch_descriptor(
            num_reqs=num_reqs,
            num_toks=num_toks,
            uniform_tok_count=uniform_tok_count,
            use_eager=is_profile or skip_compiled,
        )

        if batch_desc.num_tokens == 0:
            return self.kv_connector.no_forward(scheduler_output)

        if not dummy_run:
            input_batch = self.prepare_inputs(scheduler_output, batch_desc)
            block_tables, slot_mappings = self.prepare_attn(input_batch)

            if self.lora_config:
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            from vllm.v1.worker.gpu.input_batch import InputBatch

            input_batch = InputBatch.make_dummy(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
            )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
            else:
                block_tables = None
                slot_mappings = None

        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run):
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)
            assert block_tables is not None
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
            )

        inputs_embeds = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            # vLLM 0.23 dropped the req_states arg from get_mm_embeddings.
            inputs_embeds = self.model_state.get_mm_embeddings(
                scheduler_output.scheduled_encoder_inputs,
                input_batch,
            )

        model_inputs: dict[str, Any] = {
            "input_ids": input_batch.input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": intermediate_tensors,
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        add_forward_compat_kwargs(model_inputs, input_batch, self.sampler)
        if not self.is_first_pp_rank:
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None
            assert intermediate_tensors is not None

        # ★ PRE-FORWARD: per-request preprocess + batched MTP.
        # Runs for ALL graph modes (FULL, PIECEWISE, NONE).
        # For FULL graph: OmniModelState provides a static inputs_embeds
        # buffer that was captured by the graph.  Preprocess writes
        # in-place to this buffer, and FULL graph replay reads the
        # updated values from the same tensor address.
        if not dummy_run:
            self.model_state.run_preprocess(input_batch, model_inputs)

        # --- Model forward ---
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # FULL graph replay.  Preprocess already wrote to the static
            # inputs_embeds buffer above.
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
            hidden_states = model_output
            self._last_aux_output = None
            self._last_multimodal_outputs = None
        else:
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
            )
            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
            ):
                self.kv_connector.pre_forward(scheduler_output)
                model_output = self.model(**model_inputs)

            # Extract hidden_states from model output.
            self._last_aux_output = None
            self._last_multimodal_outputs = None
            if isinstance(model_output, OmniOutput):
                hidden_states = model_output.text_hidden_states
                if model_output.multimodal_outputs:
                    self._last_multimodal_outputs = model_output.multimodal_outputs
            elif isinstance(model_output, tuple) and len(model_output) == 2:
                hidden_states, self._last_aux_output = model_output
                if hasattr(self.model, "_last_captured_layers"):
                    self.model._last_captured_layers = self._last_aux_output
            else:
                hidden_states = model_output

        if not dummy_run and isinstance(hidden_states, torch.Tensor):
            self.model_state.run_postprocess(hidden_states, input_batch)

        kv_connector_output = self.kv_connector.post_forward(scheduler_output.finished_req_ids)
        self._last_kv_connector_output = kv_connector_output
        self.execute_model_state = _make_execute_model_state(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=None,
            finished_req_ids=scheduler_output.finished_req_ids,
            kv_connector_output=kv_connector_output,
            num_tokens_across_dp=num_tokens_across_dp,
        )

        if not self.is_last_pp_rank:
            assert isinstance(hidden_states, IntermediateTensors)
            hidden_states.kv_connector_output = kv_connector_output
            return hidden_states
        assert isinstance(hidden_states, torch.Tensor)
        return None

    # ------------------------------------------------------------------
    # Request lifecycle: update intermediate buffer from cached requests
    # ------------------------------------------------------------------

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        """Merge updated additional_information into intermediate_buffer.

        In async_chunk mode, chunk_transfer_adapter attaches updated
        additional_information (e.g. thinker_decode_embeddings) to
        OmniCachedRequestData for cached requests every schedule step.
        Upstream GPUModelRunner.update_requests does not handle this
        field, so we merge it into the intermediate buffer here.
        """
        super().update_requests(scheduler_output)

        cached = scheduler_output.scheduled_cached_reqs
        addl_info = getattr(cached, "additional_information", None)
        if not addl_info:
            return
        for req_id, info in addl_info.items():
            if info is None:
                continue
            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is None:
                continue
            resolved = _resolve_additional_information(info)
            if resolved:
                self.model_state.intermediate_buffer.update(req_idx, resolved)

    # ------------------------------------------------------------------
    # Request lifecycle: clean up intermediate buffer on finish
    # ------------------------------------------------------------------

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        # IMPORTANT: Must query req_id_to_index BEFORE super().finish_requests()
        # because super() calls req_states.remove_request(req_id) which pops the
        # mapping and returns the slot index to free_indices.
        finished = scheduler_output.finished_req_ids
        preempted = scheduler_output.preempted_req_ids
        all_done = finished | preempted if preempted else finished
        for req_id in all_done:
            idx = self.req_states.req_id_to_index.get(req_id)
            if idx is not None:
                self.model_state.remove_request(idx)
        super().finish_requests(scheduler_output)
