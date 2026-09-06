"""Omni v2 GPU model runner hooks."""

from __future__ import annotations

import threading
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import set_dummy_context
from vllm.v1.worker.gpu.lora_utils import get_num_active_loras_for_dispatch
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras
from vllm.v1.worker.gpu.model_runner import (
    BatchDescriptor,
    BatchExecutionDescriptor,
    ExecuteModelState,
    GPUModelRunner,
    IntermediateTensors,
    build_slot_mappings_by_layer,
)

from vllm_omni.core.sched.omni_scheduling_coordinator import (
    uses_native_mrv2_data_plane,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.sampling_utils import sanitize_sampling_params_min_tokens_stop_ids
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.intermediate_buffer import (
    _resolve_additional_information,
)
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

logger = init_logger(__name__)

_model_state_patch_lock = threading.RLock()


def _needs_capture_tensor_unwrap(model: Any) -> bool:
    return bool(getattr(model, "_returns_tuple", False) or getattr(model, "model_stage", None) == "thinker")


class OmniGPUModelRunner(GPUModelRunner):
    """Thin layer over v2 ``GPUModelRunner`` for Omni lifecycle hooks."""

    model_state: OmniModelState
    _last_aux_output: Any
    _last_multimodal_outputs: dict[str, Any] | None
    _model_returns_tuple: bool
    _supports_full_graph_aux_outputs: bool
    _aux_output_tree_spec: Any

    def _configure_cudagraph_output_contract(self) -> None:
        """Select the CUDA graph output contract declared by the model."""
        self._model_returns_tuple = _needs_capture_tensor_unwrap(self.model)
        self._supports_full_graph_aux_outputs = bool(getattr(self.model, "supports_mrv2_full_graph_aux_outputs", False))
        if self._supports_full_graph_aux_outputs and not self._model_returns_tuple:
            raise ValueError("supports_mrv2_full_graph_aux_outputs requires a tuple-returning model")
        self._aux_output_tree_spec = None
        self._exclude_full_graph = (self._model_returns_tuple and not self._supports_full_graph_aux_outputs) or hasattr(
            self.model, "_last_captured_layers"
        )
        if self._supports_full_graph_aux_outputs:
            self.use_aux_hidden_state_outputs = True

    def _prepare_cudagraph_capture_output(
        self,
        model_output: Any,
        cg_mode: CUDAGraphMode,
    ) -> Any:
        """Adapt an Omni output to vLLM's native CUDA graph contract."""
        if isinstance(model_output, OmniOutput):
            return model_output.text_hidden_states
        if not (isinstance(model_output, tuple) and len(model_output) == 2):
            return model_output

        hidden_states, aux_output = model_output
        supports_aux = getattr(self, "_supports_full_graph_aux_outputs", False)
        if cg_mode == CUDAGraphMode.PIECEWISE or not supports_aux:
            return hidden_states

        flat_aux, tree_spec = tree_flatten(aux_output)
        if not flat_aux or not all(isinstance(value, torch.Tensor) for value in flat_aux):
            raise TypeError("MRv2 FULL CUDA graph auxiliary outputs require a non-empty tensor-only pytree")
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("MRv2 FULL CUDA graph primary output must be a tensor")
        for value in flat_aux:
            if value.ndim == 0 or value.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "MRv2 FULL CUDA graph auxiliary tensor leading dimensions must match the primary hidden states"
                )

        if self._aux_output_tree_spec is None:
            self._aux_output_tree_spec = tree_spec
        elif self._aux_output_tree_spec != tree_spec:
            raise RuntimeError("MRv2 FULL CUDA graph auxiliary output structure changed during capture")
        return hidden_states, flat_aux

    def _unpack_full_graph_output(self, model_output: Any) -> tuple[Any, Any]:
        """Restore the model-owned auxiliary pytree after FULL graph replay."""
        if not self._supports_full_graph_aux_outputs:
            return model_output, None
        if not (isinstance(model_output, tuple) and len(model_output) == 2):
            raise RuntimeError("MRv2 FULL CUDA graph replay did not return auxiliary outputs")
        if self._aux_output_tree_spec is None:
            raise RuntimeError("MRv2 FULL CUDA graph auxiliary output schema was not captured")
        hidden_states, flat_aux = model_output
        return hidden_states, tree_unflatten(list(flat_aux), self._aux_output_tree_spec)

    def _add_legacy_forward_inputs(self, model_inputs: dict[str, Any], input_batch: Any) -> None:
        """Supply forward-only metadata for pipelines not on the native plane."""
        if self._omni_data_plane is not None:
            return
        model_inputs.setdefault("sampling_metadata", getattr(input_batch, "sampling_metadata", None))
        logits_index = getattr(
            input_batch,
            "logits_indices",
            getattr(input_batch, "logits_index", None),
        )
        model_inputs.setdefault("logits_index", logits_index)
        model_inputs.setdefault("sampler", self.sampler)

    def _get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: Any,
    ) -> torch.Tensor:
        """Call the native vLLM multimodal embedding contract."""
        return self.model_state.get_mm_embeddings(
            scheduled_encoder_inputs,
            input_batch,
            self.req_states,
        )

    def _prepare_mm_inputs(
        self,
        scheduler_output: SchedulerOutput,
        input_batch: Any,
        *,
        dummy_run: bool,
    ) -> tuple[Any, torch.Tensor | None, Any]:
        """Prepare native vLLM multimodal model inputs."""
        input_ids = input_batch.input_ids
        inputs_embeds = None
        ec_connector_output = None
        if not self.supports_mm_inputs or not self.is_first_pp_rank:
            return input_ids, inputs_embeds, ec_connector_output

        if dummy_run:
            inputs_embeds = self.model_state.dummy_inputs_embeds(input_batch.num_tokens_after_padding)
        else:
            scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
            if self.lora_config is not None:
                set_active_mm_loras(
                    model=self.model,
                    lora_manager=self.lora_manager,
                    encoder_cache=self.encoder_cache,
                    req_id_to_index=self.req_states.req_id_to_index,
                    lora_state=self.lora_state,
                    scheduled_encoder_inputs=scheduled_encoder_inputs,
                )
            with self.ec_connector.maybe_get_output(scheduler_output) as ec_connector_output:
                inputs_embeds = self._get_mm_embeddings(
                    scheduled_encoder_inputs,
                    input_batch,
                )
        if inputs_embeds is not None and not self.model.requires_raw_input_tokens:
            input_ids = None
        return input_ids, inputs_embeds, ec_connector_output

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._validate_parallel_support()
        self._omni_data_plane = (
            OmniRunnerDataPlane(self.vllm_config, self.model_config)
            if uses_native_mrv2_data_plane(
                self.model_config,
                use_v2_model_runner=True,
            )
            else None
        )

    def _validate_parallel_support(self) -> None:
        parallel_config = self.vllm_config.parallel_config
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Omni Model Runner V2 does not support pipeline parallel; "
                "use model_runner: v1 when pipeline_parallel_size > 1"
            )
        if getattr(parallel_config, "prefill_context_parallel_size", 1) > 1:
            raise NotImplementedError(
                "Omni Model Runner V2 does not support prefill context parallelism; "
                "use model_runner: v1 when prefill_context_parallel_size > 1"
            )

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        logits_processor = getattr(self.model, "logits_processor", None)
        logits_vocab = getattr(logits_processor, "vocab_size", None)
        if isinstance(logits_vocab, int) and logits_vocab > 0:
            for request_data in scheduler_output.scheduled_new_reqs:
                sampling_params = request_data.sampling_params
                if sampling_params is not None:
                    sanitize_sampling_params_min_tokens_stop_ids(
                        sampling_params,
                        logits_vocab,
                    )
        super().add_requests(scheduler_output)

    def shutdown(self) -> None:
        if self._omni_data_plane is not None:
            self._omni_data_plane.close()
        super().shutdown()

    def _prepare_native_data_plane(self, scheduler_output: SchedulerOutput) -> None:
        plane = getattr(self, "_omni_data_plane", None)
        if plane is None:
            return
        for request_data in getattr(scheduler_output, "scheduled_new_reqs", []):
            if str(getattr(request_data, "req_id", "")).startswith("_warmup_"):
                continue
            plane.register_request(request_data)
        plane.register_receivers(list(getattr(scheduler_output, "pending_input_registrations", [])))
        natural_terminal_req_ids = set(getattr(scheduler_output, "data_plane_terminal_req_ids", set()))
        aborted_req_ids = set(getattr(scheduler_output, "finished_req_ids", set())).difference(natural_terminal_req_ids)
        if natural_terminal_req_ids:
            plane.request_terminal(natural_terminal_req_ids)
        if aborted_req_ids:
            plane.abort_requests(aborted_req_ids)

    def _sync_native_data_plane_payloads(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        plane = getattr(self, "_omni_data_plane", None)
        if plane is None:
            return
        gpu_keys = getattr(self.model, "gpu_resident_buffer_keys", set())
        for req_id in scheduler_output.num_scheduled_tokens:
            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is None:
                continue
            payload = plane.pop_local_stage_payload(req_id)
            if isinstance(payload, dict) and payload:
                self.model_state.intermediate_buffer.update(
                    req_idx,
                    payload,
                    gpu_keys,
                )

    def _finalize_native_data_plane_output(self, output: Any) -> Any:
        plane = getattr(self, "_omni_data_plane", None)
        if plane is None or output is None:
            return output
        inter_stage_outputs = getattr(output, "inter_stage_outputs", None)
        plane.enqueue_outputs(
            req_ids=list(getattr(output, "req_ids", [])),
            inter_stage_outputs=inter_stage_outputs,
            sampled_token_ids=getattr(output, "sampled_token_ids", None),
        )
        output.inter_stage_outputs = None
        output.omni_connector_output = plane.get_omni_connector_output()
        return output

    def _reserve_native_data_plane_outputs(self, req_ids: list[str]) -> None:
        plane = getattr(self, "_omni_data_plane", None)
        if plane is not None:
            plane.reserve_outputs(req_ids)

    def _attach_native_data_plane_signals(self, output: Any) -> Any:
        plane = getattr(self, "_omni_data_plane", None)
        if plane is None:
            return output
        return plane.attach_omni_connector_output(output)

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
        self._configure_cudagraph_output_contract()

        # Preprocess models own embedding buffers; encoder_runner sizing would mismatch.
        if getattr(self.model, "has_preprocess", False) and self.supports_mm_inputs:
            self.supports_mm_inputs = False
            self.encoder_cache = None

    # ------------------------------------------------------------------
    # CUDA Graph: conditionally exclude FULL mode
    # ------------------------------------------------------------------

    @staticmethod
    def _without_full_graph_candidates(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: OmniGPUModelRunner._without_full_graph_candidates(item)
                for key, item in value.items()
                if key != CUDAGraphMode.FULL
            }
        if isinstance(value, list):
            return [
                OmniGPUModelRunner._without_full_graph_candidates(item)
                for item in value
                if getattr(item, "cg_mode", None) != CUDAGraphMode.FULL
            ]
        if isinstance(value, tuple):
            return tuple(
                OmniGPUModelRunner._without_full_graph_candidates(item)
                for item in value
                if getattr(item, "cg_mode", None) != CUDAGraphMode.FULL
            )
        return value

    @staticmethod
    def _contains_full_graph_candidate(value: Any) -> bool:
        if isinstance(value, dict):
            return CUDAGraphMode.FULL in value or any(
                OmniGPUModelRunner._contains_full_graph_candidate(item) for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(OmniGPUModelRunner._contains_full_graph_candidate(item) for item in value)
        return getattr(value, "cg_mode", None) == CUDAGraphMode.FULL

    def _exclude_unsupported_full_graphs(self) -> None:
        manager = self.cudagraph_manager
        capture_descs = getattr(manager, "_capture_descs", None)
        candidates = getattr(manager, "_candidates", None)
        if not isinstance(capture_descs, dict) or not isinstance(candidates, (dict, list, tuple)):
            raise RuntimeError(
                "cannot safely exclude FULL CUDA graphs for this Omni model: "
                "vLLM CudaGraphManager internals have changed"
            )

        manager._capture_descs = self._without_full_graph_candidates(capture_descs)
        manager._candidates = self._without_full_graph_candidates(candidates)
        if self._contains_full_graph_candidate(manager._capture_descs) or self._contains_full_graph_candidate(
            manager._candidates
        ):
            raise RuntimeError(
                "cannot safely exclude FULL CUDA graphs for this Omni model: FULL candidates remain after filtering"
            )
        logger.info("Excluded FULL CUDA graph capture for Omni model. PIECEWISE graphs will still be captured.")

    def capture_model(self) -> int:
        """Handle CUDA graph capture for Omni models.

        Tuple-returning models must explicitly declare a stable tensor-only
        auxiliary output schema to use FULL mode. Other tuple outputs keep the
        conservative FULL exclusion.

        For PIECEWISE capture, the warmup pass runs with
        ``CUDAGraphMode.NONE`` which hits ``torch.empty_like(hidden_states)``
        in the cudagraph framework.  If the model returns a tuple, that call
        crashes.  We temporarily wrap the model's forward to extract only the
        tensor part during capture, then restore the original forward.
        """
        if self._exclude_full_graph:
            self._exclude_unsupported_full_graphs()

        # Wrap model forward during capture so tuple returns don't crash
        # torch.empty_like() in the PIECEWISE warmup pass.
        if self._model_returns_tuple:
            original_forward = self.model.forward

            def _capture_forward(*args: Any, **kwargs: Any) -> Any:
                output = original_forward(*args, **kwargs)
                cg_mode = CUDAGraphMode.NONE
                if is_forward_context_available():
                    cg_mode = get_forward_context().cudagraph_runtime_mode
                return self._prepare_cudagraph_capture_output(output, cg_mode)

            self.model.forward = _capture_forward  # type: ignore[assignment]
            try:
                result = super().capture_model()
            finally:
                self.model.forward = original_forward  # type: ignore[assignment]
        else:
            result = super().capture_model()

        capture_talker_mtp = getattr(getattr(self, "model_state", None), "capture_talker_mtp_graphs", None)
        if callable(capture_talker_mtp):
            capture_talker_mtp(self._dispatch_mtp_batch_descriptor)
        return result

    def _dispatch_mtp_batch_descriptor(self, num_mtp_reqs: int) -> Any:
        capture_sizes = self.model_state._get_talker_mtp_capture_sizes()
        captured_bucket = next(
            (size for size in sorted(capture_sizes) if num_mtp_reqs <= size <= self.scheduler_config.max_num_seqs),
            None,
        )
        if captured_bucket is None:
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_mtp_reqs,
                num_reqs=num_mtp_reqs,
            )
        return self.cudagraph_manager.dispatch(
            captured_bucket,
            captured_bucket,
            1,
            0,
        )

    def _dispatch_batch_descriptor(
        self,
        *,
        num_reqs: int,
        num_toks: int,
        uniform_tok_count: int | None,
        num_active_loras: int,
        use_eager: bool,
        max_query_len: int,
    ):
        return dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            max_query_len=max_query_len,
            need_eager=use_eager,
            num_active_loras=num_active_loras,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
    ) -> Any:
        if not dummy_run:
            self._prepare_native_data_plane(scheduler_output)
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self._sync_native_data_plane_payloads(scheduler_output)
            self.block_tables.apply_staged_writes()
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
        num_active_loras = 0
        if self.lora_config:
            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            num_active_loras = get_num_active_loras_for_dispatch(
                self.lora_config,
                self.lora_state,
                req_ids,
                dummy_run,
            )
        # Encoder-decoder models: disable compilation when encoder inputs
        # are scheduled (dynamic cross-attention cache updates).
        skip_compiled = self.is_encoder_decoder and bool(scheduler_output.scheduled_encoder_inputs)
        batch_desc, num_tokens_across_dp = self._dispatch_batch_descriptor(
            num_reqs=num_reqs,
            num_toks=num_toks,
            uniform_tok_count=uniform_tok_count,
            num_active_loras=num_active_loras,
            use_eager=is_profile or skip_compiled,
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
            block_tables, slot_mappings = self.prepare_attn(input_batch)
            self.model_state.preprocess_state(
                input_batch,
                block_tables,
                self.kv_cache_config,
                self.req_states.num_computed_tokens.gpu,
            )

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
                max_query_len=batch_desc.max_query_len,
            )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
                if context_len:
                    set_dummy_context(
                        input_batch,
                        self.block_tables,
                        context_len,
                        self.kv_cache_config.num_blocks,
                        self.max_model_len,
                    )
            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata is required for FULL CUDA graph dummy runs"
                )
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
                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
            )
        input_ids, inputs_embeds, ec_connector_output = self._prepare_mm_inputs(
            scheduler_output,
            input_batch,
            dummy_run=dummy_run,
        )

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": intermediate_tensors,
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        self._add_legacy_forward_inputs(model_inputs, input_batch)
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
            self.model_state.run_preprocess(
                input_batch,
                model_inputs,
                self.req_states,
                self._dispatch_mtp_batch_descriptor,
            )

        self.eplb.prepare_forward(self.model_config, input_batch.num_tokens)

        # --- Model forward ---
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # FULL graph replay.  Preprocess already wrote to the static
            # inputs_embeds buffer above.
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
            hidden_states, self._last_aux_output = self._unpack_full_graph_output(model_output)
            self._last_multimodal_outputs = None
            if hasattr(self.model, "_last_captured_layers"):
                self.model._last_captured_layers = self._last_aux_output
        else:
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
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
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

        routed_experts = None
        if not dummy_run and (capturer := self.routed_experts_capturer) is not None:
            assert slot_mappings is not None
            routed_experts = capturer.get_routed_experts(slot_mappings, num_toks)

        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=None,
            finished_req_ids=scheduler_output.finished_req_ids,
            ec_connector_output=ec_connector_output,
            routed_experts=routed_experts,
        )

        if not self.is_last_pp_rank:
            assert isinstance(hidden_states, IntermediateTensors)
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
                gpu_keys = getattr(self.model, "gpu_resident_buffer_keys", set())
                self.model_state.intermediate_buffer.update(req_idx, resolved, gpu_keys)

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
