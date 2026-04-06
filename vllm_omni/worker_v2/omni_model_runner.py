"""OmniGPUModelRunner — thin inheritance layer over v2 GPUModelRunner.

Injects ``OmniModelState`` via ``load_model`` and adds:

* ``execute_model`` override with pre-forward (preprocess + MTP) and
  post-forward (postprocess) hooks, plus generic tuple-return interception
  support (not used by current Omni models)
* ``finish_requests`` hook to clean up the intermediate buffer
* ``capture_model`` override to conditionally exclude FULL CUDA graph mode
  (only when the model forward returns a tuple that requires Python-level
  interception)
"""

from __future__ import annotations

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

from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

logger = init_logger(__name__)


class OmniGPUModelRunner(GPUModelRunner):
    """Thin layer over v2 ``GPUModelRunner`` for Omni lifecycle hooks."""

    model_state: OmniModelState
    _last_aux_output: Any
    # Whether the model.forward returns (hidden_states, aux_dict) tuple.
    # If False, FULL CUDA graph mode is allowed since no Python-level
    # tuple interception is needed.
    # NOTE: No current Omni model sets _returns_tuple (thinker uses
    # _last_captured_layers instead).  Kept for future model support.
    _model_returns_tuple: bool

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        import vllm.v1.worker.gpu.model_runner as _mr_module

        _orig = _mr_module.init_model_state
        _mr_module.init_model_state = init_omni_model_state
        try:
            super().load_model(*args, **kwargs)
        finally:
            _mr_module.init_model_state = _orig
        self._last_aux_output = None
        # Detect whether model.forward returns a tuple.
        self._model_returns_tuple = getattr(self.model, "_returns_tuple", False)
        # Exclude FULL graph when any per-step Python-level post-forward
        # logic must run (tuple intercept, _last_captured_layers, etc.).
        # FULL graph replay bypasses Python entirely — only PIECEWISE
        # is safe for these models.
        self._exclude_full_graph = self._model_returns_tuple or hasattr(self.model, "_last_captured_layers")

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
            if CUDAGraphMode.FULL in mgr._capture_descs:
                del mgr._capture_descs[CUDAGraphMode.FULL]
                for i, descs in enumerate(mgr._candidates):
                    mgr._candidates[i] = [d for d in descs if d.cg_mode != CUDAGraphMode.FULL]
                logger.info("Excluded FULL CUDA graph capture for Omni model. PIECEWISE graphs will still be captured.")

        # Wrap model forward during capture so tuple returns don't crash
        # torch.empty_like() in the PIECEWISE warmup pass.
        if self._model_returns_tuple:
            original_forward = self.model.forward

            def _capture_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                output = original_forward(*args, **kwargs)
                if isinstance(output, tuple):
                    return output[0]
                return output

            self.model.forward = _capture_forward  # type: ignore[assignment]
            try:
                return super().capture_model()
            finally:
                self.model.forward = original_forward  # type: ignore[assignment]
        return super().capture_model()

    # ------------------------------------------------------------------
    # execute_model: preprocess → forward → postprocess + tuple intercept
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> Any:
        if not dummy_run:
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                return self.kv_connector.no_forward(scheduler_output)

        # --- Batch descriptor + dispatch ---
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)
        batch_desc = self.cudagraph_manager.dispatch(num_reqs, num_toks, uniform_tok_count)

        # Encoder-decoder models: disable compilation when encoder inputs
        # are scheduled (dynamic cross-attention cache updates).
        skip_compiled = False
        if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            skip_compiled = True
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_toks,
                num_reqs=num_reqs,
            )

        # DP sync: align batch sizes across data-parallel ranks.
        num_tokens_across_dp = None
        if self.dp_size > 1:
            from vllm.v1.worker.gpu.dp_utils import sync_cudagraph_and_dp_padding

            batch_desc, num_tokens_across_dp = sync_cudagraph_and_dp_padding(
                self.cudagraph_manager,
                batch_desc,
                num_toks,
                num_reqs,
                uniform_tok_count,
                self.dp_size,
                self.dp_rank,
            )

        if batch_desc.num_tokens == 0:
            return self.kv_connector.no_forward(scheduler_output)

        # --- Prepare inputs ---
        if not dummy_run:
            input_batch = self.prepare_inputs(scheduler_output, batch_desc)
            block_tables, slot_mappings = self.prepare_attn(input_batch)

            # LoRA activation
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

        # --- Attention metadata ---
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

        # --- MM embeddings ---
        inputs_embeds = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            inputs_embeds = self.model_state.get_mm_embeddings(
                scheduler_output.scheduled_encoder_inputs,
                input_batch,
                self.req_states,
            )

        # --- Build model_inputs ---
        model_inputs: dict[str, Any] = {
            "input_ids": input_batch.input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": intermediate_tensors,
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
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
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
            hidden_states = model_output
            self._last_aux_output = None
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

            # ★ TUPLE INTERCEPT: handle models that return (hidden, aux_dict).
            # torch.compile may prevent in-model side-effects like
            # self._last_captured_layers = ... from taking effect,
            # so the tuple may surface here even when the model tries to
            # store captured layers internally.
            if isinstance(model_output, tuple) and len(model_output) == 2:
                first, second = model_output
                if isinstance(first, torch.Tensor):
                    self._last_aux_output = second
                    # Store captured layers on the model so
                    # make_omni_output can retrieve them.
                    if hasattr(self.model, "_last_captured_layers"):
                        self.model._last_captured_layers = second
                    hidden_states = first
                else:
                    self._last_aux_output = None
                    hidden_states = model_output
            else:
                self._last_aux_output = None
                hidden_states = model_output

        # ★ POST-FORWARD: per-request postprocess
        if not dummy_run and isinstance(hidden_states, torch.Tensor):
            self.model_state.run_postprocess(hidden_states, input_batch)

        kv_connector_output = self.kv_connector.post_forward(scheduler_output)
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=None,
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
