# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni v2 GPU model runner hooks."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import (
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
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.core.sched.omni_scheduling_coordinator import (
    uses_native_mrv2_data_plane,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.runner_assisted_metadata import RunnerAssistedFullAttentionMetadataRequest
from vllm_omni.worker.sampling_utils import sanitize_sampling_params_min_tokens_stop_ids
from vllm_omni.worker_v2 import staged_writes
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.intermediate_buffer import (
    _resolve_additional_information,
)
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane
from vllm_omni.worker_v2.prefix_cache import MRv2PrefixCacheMixin
from vllm_omni.worker_v2.warmup import is_warmup_request

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

    from vllm_omni.distributed.weight_sharing.adapters import WeightSharingAdapter
    from vllm_omni.worker_v2.native_output_worker import NativeOutputWorker

logger = init_logger(__name__)

_model_state_patch_lock = threading.RLock()


def _runner_assisted_batch_descriptor(
    batch_desc: BatchExecutionDescriptor,
    request: RunnerAssistedFullAttentionMetadataRequest,
    num_reqs: int,
    max_num_seqs: int,
) -> BatchExecutionDescriptor | None:
    bucket = request.num_reqs_padded
    tokens = request.num_tokens_padded if request.num_tokens_padded is not None else bucket
    if not (num_reqs <= bucket <= max_num_seqs and batch_desc.num_tokens <= tokens and tokens >= bucket):
        return None
    return replace(batch_desc, num_reqs=bucket, num_tokens=tokens)


def _needs_capture_tensor_unwrap(model: Any) -> bool:
    return bool(getattr(model, "_returns_tuple", False))


def _supports_full_graph_aux_outputs(model: Any) -> bool:
    """Tuple-returning model whose auxiliary output FULL graphs may replay.

    The model declares ``supports_mrv2_full_graph_aux_outputs`` when its
    auxiliary output is a tensor-only pytree with the same structure for every
    batch shape and each leaf's leading dimension is the token axis.
    """
    return _needs_capture_tensor_unwrap(model) and bool(getattr(model, "supports_mrv2_full_graph_aux_outputs", False))


class OmniGPUModelRunner(MRv2PrefixCacheMixin, GPUModelRunner):
    """Thin layer over v2 ``GPUModelRunner`` for Omni lifecycle hooks."""

    supports_mm_inputs: bool
    encoder_cache: EncoderCache | None
    _native_output_materializer: NativeOutputWorker | None
    model_state: OmniModelState
    vocab_size: int
    # Speculative decoding state; the upstream __init__ sets the real values.
    use_aux_hidden_state_outputs: bool = False
    speculator: Any = None
    num_speculative_steps: int = 0
    _last_aux_output: Any
    _last_multimodal_outputs: dict[str, Any] | None
    _model_returns_tuple: bool
    # FULL-graph aux-output contract (see _configure_cudagraph_output_contract).
    _full_graph_aux_outputs = False
    _aux_output_spec: Any = None

    def _configure_cudagraph_output_contract(self) -> None:
        """Select the CUDA graph output contract declared by the model."""
        self._model_returns_tuple = _needs_capture_tensor_unwrap(self.model)
        # FULL replay returns what capture stored: the hidden states and, for a
        # declared tuple model, its auxiliary pytree flattened to leaves.
        self._full_graph_aux_outputs = _supports_full_graph_aux_outputs(self.model)
        self._aux_output_spec = None
        self._exclude_full_graph = (self._model_returns_tuple and not self._full_graph_aux_outputs) or hasattr(
            self.model, "_last_captured_layers"
        )

    def _split_fullgraph_output(self, replayed: Any) -> tuple[Any, Any]:
        """FULL replay output -> ``(hidden_states, aux pytree or None)``."""
        if self._full_graph_aux_outputs and isinstance(replayed, tuple):
            hidden, leaves = replayed
            return hidden, tree_unflatten(list(leaves), self._aux_output_spec)
        return replayed, None

    def _flatten_capture_aux_output(self, model_output: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``(hidden, aux pytree)`` -> ``(hidden, leaves)`` for the graph manager's aux buffers."""
        hidden, aux = model_output
        leaves, spec = tree_flatten(aux)
        num_tokens = hidden.shape[0]
        if not leaves or any(not isinstance(leaf, torch.Tensor) or leaf.shape[:1] != (num_tokens,) for leaf in leaves):
            raise RuntimeError(
                f"{type(self.model).__name__} declares supports_mrv2_full_graph_aux_outputs, but its auxiliary "
                "output is not a non-empty tensor pytree on the token axis"
            )
        if self._aux_output_spec is None:
            self._aux_output_spec = spec
        elif spec != self._aux_output_spec:
            raise RuntimeError(f"{type(self.model).__name__} auxiliary output structure changed between captures")
        return hidden, leaves

    @staticmethod
    def _prepare_cudagraph_capture_output(model_output: Any) -> Any:
        """Unwrap model outputs for the upstream PIECEWISE capture warmup."""
        if isinstance(model_output, OmniOutput):
            return model_output.text_hidden_states
        if isinstance(model_output, tuple) and len(model_output) == 2:
            return model_output[0]
        return model_output

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
        return self.model_state.prepare_inputs_embeds(
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
        # Before the runner builds its staged-write state tensors.
        staged_writes.install()
        self._staged_writes = staged_writes.StagedWrites()
        with self._staged_writes.activate():
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

    @staged_writes.runner_context
    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        # The upstream Sampler stages static parameters only in add_request.
        # With no new requests, its UVA snapshots remain valid and immutable.
        # Keep the upstream path for custom/speculative samplers, whose staged
        # writes may have a different lifecycle.
        sampler = getattr(self, "sampler", None)
        static_staged_writes = (type(sampler) is Sampler or getattr(sampler, "omni_static_staged_writes", False)) and [
            type(p) for p in getattr(sampler, "logits_processors", ())
        ] == [LogitBiasState, PenaltiesState, BadWordsState]
        if not scheduler_output.scheduled_new_reqs and static_staged_writes:
            return
        logits_processor = getattr(self.model, "logits_processor", None)
        logits_vocab = getattr(logits_processor, "vocab_size", None)
        if logits_vocab is None:
            # Narrow model heads without a vLLM LogitsProcessor (codec heads).
            logits_vocab = getattr(self.model, "logits_vocab_size", None)
        if isinstance(logits_vocab, int) and logits_vocab > 0:
            for request_data in scheduler_output.scheduled_new_reqs:
                sampling_params = request_data.sampling_params
                if sampling_params is not None:
                    sanitize_sampling_params_min_tokens_stop_ids(
                        sampling_params,
                        logits_vocab,
                    )
        # Custom processors may read request tokens in apply_staged_writes;
        # preserve upstream's request-state flush before their callback.
        with self._staged_writes.deferred() if static_staged_writes else nullcontext():
            super().add_requests(scheduler_output)

    def shutdown(self) -> None:
        # Every owner must release its resources even if an earlier drain fails.
        try:
            state = getattr(self, "model_state", None)
            close = getattr(state, "close_first_audio_sender", None)
            if close is not None:
                close()
        finally:
            try:
                worker = getattr(self, "_native_output_materializer", None)
                if worker is not None:
                    worker.close()
            finally:
                self._native_output_materializer = None
                try:
                    if self._omni_data_plane is not None:
                        self._omni_data_plane.close()
                finally:
                    super().shutdown()

    def _prepare_native_data_plane(self, scheduler_output: SchedulerOutput) -> None:
        plane = getattr(self, "_omni_data_plane", None)
        if plane is None:
            return
        for request_data in getattr(scheduler_output, "scheduled_new_reqs", []):
            if is_warmup_request(request_data.req_id):
                continue
            plane.register_request(request_data)
        plane.register_receivers(list(getattr(scheduler_output, "pending_input_registrations", [])))
        if not getattr(self.model_config, "async_chunk", False):
            plane.recv_full_payload_inputs(scheduler_output)
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
        gpu_keys: set[tuple[str, str]] = getattr(self.model, "gpu_resident_buffer_keys", set())
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

    def _uses_native_output_materializer(self) -> bool:
        parallel = getattr(getattr(self, "vllm_config", None), "parallel_config", None)
        # Multi-rank executors already consume outputs asynchronously and choose
        # the replying rank themselves. Preserve that ownership contract.
        return getattr(parallel, "tensor_parallel_size", 1) == 1 and getattr(self, "_omni_data_plane", None) is not None

    def _materialize_native_output(self, output: Any) -> Any:
        from vllm_omni.worker_v2.native_output_worker import NativeOutputWorker

        worker = getattr(self, "_native_output_materializer", None)
        if worker is None:
            worker = NativeOutputWorker(getattr(self.vllm_config, "max_concurrent_batches", 2), self.device)
            self._native_output_materializer = worker
        return worker.submit(output, self._omni_data_plane)

    def _finalize_native_data_plane_output(self, output: Any) -> Any:
        plane = getattr(self, "_omni_data_plane", None)
        if plane is None or output is None:
            return output
        # MRv2's data plane consumes the host-materialized token lists from
        # the runner output.  Do not let a future connector path fall back to
        # reading sampler tensors and synchronizing the device here.
        if hasattr(output, "sampled_token_ids_materialized") and getattr(output, "sampled_token_ids", None) is not None:
            if not output.sampled_token_ids_materialized:
                raise RuntimeError("MRv2 native data plane received non-materialized sampled_token_ids")
        plane.publish_output(output)
        output.omni_connector_output = plane.get_omni_connector_output()
        return output

    def release_omni_shm_resources(self, request_ids: list[str]) -> int:
        plane = getattr(self, "_omni_data_plane", None)
        return 0 if plane is None else plane.release_shm_resources(request_ids)

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
        # Shared weights allow only one load attempt per worker lifetime.
        # Reload requires full sharing-group teardown and worker recreation.
        if getattr(self, "_weight_share_backend", None) is not None:
            raise RuntimeError(
                "A weight-sharing load was already attempted; tear down the sharing group before retrying."
            )
        import vllm.v1.worker.gpu.model_runner as _mr_module

        def _init_model_state(vllm_config: Any, model: Any, encoder_cache: Any, device: Any) -> Any:
            # Runs after the model is built and before the sampler is: a model
            # whose logits head is narrower than its HF config reports (the
            # MiniCPM-o codec Talker's config has no vocab_size at all) sizes
            # the MRv2 sampler and request state by that head.
            self._adopt_model_logits_vocab(model)
            return init_omni_model_state(vllm_config, model, encoder_cache, device)

        with _model_state_patch_lock:
            _orig = _mr_module.init_model_state
            _mr_module.init_model_state = _init_model_state
            try:
                # Upstream's model-construction scope ends before custom_sampler.
                # Collect kernels selected by that hook in this runner's registry.
                with self.jit_warmup_registry.activate():
                    super().load_model(*args, **kwargs)
            finally:
                _mr_module.init_model_state = _orig
        self._last_aux_output = None
        self._last_multimodal_outputs = None
        self._configure_cudagraph_output_contract()
        if self.use_aux_hidden_state_outputs and (
            self._model_returns_tuple or hasattr(self.model, "_last_captured_layers")
        ):
            # Both would use the forward's tuple output slot.
            raise NotImplementedError(
                f"{type(self.model).__name__} returns its own auxiliary output; a drafter that "
                "reads target hidden layers is not supported with it on MRv2 yet"
            )

        # Preprocess models own embedding buffers; encoder_runner sizing would
        # mismatch unless the model builds its preprocess on the encoder output.
        if (
            getattr(self.model, "has_preprocess", False)
            and self.supports_mm_inputs
            and not getattr(self.model_state, "preprocess_keeps_mm_inputs", False)
        ):
            self.supports_mm_inputs = False
            self.encoder_cache = None

        self._publish_shared_backbone_if_configured()
        # After model ``__init__`` (which may set cudagraph_specialize_moe_routing).
        # V2 does not use the V1 MoE annotate/select hooks, so this is the
        # fail-fast for Mammoth FULL + use_v2_model_runner mismatches (#9911).
        from vllm_omni.cudagraph.moe_routing_compat import (
            ensure_moe_specialty_runner_compatible,
        )

        ensure_moe_specialty_runner_compatible(self)

    def reload_weights(self, *args: Any, **kwargs: Any) -> None:
        """Preserve upstream reload arguments, but reject updates to shared weights."""
        if getattr(self, "_weight_share_backend", None) is not None:
            raise RuntimeError(
                "A weight-sharing load was already attempted; "
                "weight updates require full sharing-group teardown and worker recreation."
            )
        super().reload_weights(*args, **kwargs)

    def _publish_shared_backbone_if_configured(self) -> None:
        """Publish finalized AR backbone weights when this rank is a provider.

        Read this TP rank's bootstrap from additional_config after the model
        and all post-load transforms are complete. Keep the backend and export
        session alive until sharing-group teardown.
        """
        from vllm_omni.distributed.weight_sharing.worker_session import resolve_worker_sharing

        sharing = resolve_worker_sharing(self.vllm_config, role="provider")
        if sharing is None:
            return
        group, _ = sharing
        if getattr(self, "_weight_share_session", None) is not None:
            return  # Already published; step-3 owns teardown, do not re-export.
        if getattr(self, "_weight_share_backend", None) is not None:
            raise RuntimeError(
                "A previous weight-sharing publish attempt did not complete; "
                "tear down the sharing group before retrying."
            )
        get_adapter = getattr(self.model, "get_weight_sharing_adapter", None)
        if not callable(get_adapter):
            raise ValueError(
                f"weight sharing requested for a provider rank, but {type(self.model).__name__} "
                "has no get_weight_sharing_adapter hook"
            )
        adapter: WeightSharingAdapter = get_adapter()
        from vllm_omni.distributed.weight_sharing import TorchIpcBackend
        from vllm_omni.platforms import current_omni_platform

        device_index = current_omni_platform.current_device()
        expected_device = current_omni_platform.get_torch_device(device_index)
        if self.device.type not in ("cpu", "meta") and (
            self.device.type != expected_device.type or self.device.index not in (None, device_index)
        ):
            raise ValueError(
                f"Weight-sharing target device {self.device} does not match current device {expected_device}"
            )
        backend = TorchIpcBackend()
        # Retain partial exports and their diagnostics if publish raises. The
        # session ID remains queryable once the backend registers the session.
        self._weight_share_backend = backend
        self._weight_share_session_id = group.session_id
        self._weight_share_session = adapter.publish_backbone(
            backend, self.model, group=group, device_index=device_index
        )
        logger.info("Published shared backbone weights for sharing group %s", getattr(group, "group_id", None))

    def _adopt_model_logits_vocab(self, model: Any) -> None:
        vocab = getattr(model, "logits_vocab_size", None)
        if not isinstance(vocab, int) or vocab <= 0 or vocab == self.vocab_size:
            return
        logger.info("Sizing the MRv2 sampler by the model's logits head: vocab %s -> %d", self.vocab_size, vocab)
        self.vocab_size = vocab
        self.req_states.vocab_size = vocab
        # Kernel warmup sizes its structured-output bitmask from the config.
        arch_config = getattr(self.model_config, "model_arch_config", None)
        if arch_config is not None and getattr(arch_config, "vocab_size", vocab) != vocab:
            arch_config.vocab_size = vocab

    # ------------------------------------------------------------------
    # CUDA Graph: conditionally exclude FULL mode
    # ------------------------------------------------------------------

    def _exclude_unsupported_full_graphs(self) -> None:
        # vLLM 0.29 keys dispatch candidates by (uniform token count, LoRA count).
        manager = self.cudagraph_manager
        assert manager is not None
        manager._capture_descs.pop(CUDAGraphMode.FULL, None)
        manager._candidates = {
            key: [desc for desc in candidates if desc.cg_mode != CUDAGraphMode.FULL]
            for key, candidates in manager._candidates.items()
        }
        logger.info("Excluded FULL CUDA graph capture for Omni model. PIECEWISE graphs will still be captured.")

    @staged_writes.runner_context
    def capture_model(self, *, profile_only: bool = False) -> int:
        """Handle CUDA graph capture for Omni models.

        Tuple-returning models use PIECEWISE graphs; FULL replay requires
        a tensor output.

        For PIECEWISE capture, the warmup pass runs with
        ``CUDAGraphMode.NONE`` which hits ``torch.empty_like(hidden_states)``
        in the cudagraph framework.  If the model returns a tuple, that call
        crashes.  We temporarily wrap the model's forward to extract only the
        tensor part during capture, then restore the original forward.
        """
        if self._exclude_full_graph:
            self._exclude_unsupported_full_graphs()

        # Wrap model forward during capture so tuple returns don't crash
        # torch.empty_like() in the PIECEWISE warmup pass. A model with the
        # FULL aux-output contract instead hands the graph manager its leaves,
        # which it keeps in persistent buffers and returns on replay.
        if self._model_returns_tuple:
            original_forward = self.model.forward
            aux_outputs = self._full_graph_aux_outputs

            def _capture_forward(*args: Any, **kwargs: Any) -> Any:
                output = original_forward(*args, **kwargs)
                if aux_outputs:
                    return self._flatten_capture_aux_output(output)
                return self._prepare_cudagraph_capture_output(output)

            self.model.forward = _capture_forward  # type: ignore[assignment]
            use_aux = self.use_aux_hidden_state_outputs
            self.use_aux_hidden_state_outputs = use_aux or aux_outputs
            try:
                result = super().capture_model(profile_only=profile_only)
            finally:
                self.model.forward = original_forward  # type: ignore[assignment]
                self.use_aux_hidden_state_outputs = use_aux
        else:
            result = super().capture_model(profile_only=profile_only)

        # The MTP runner is a CUDAGraphWrapper, so during profiling upstream
        # has swapped it to the throwaway pool and clears it afterwards. Report
        # its pool with the rest in both passes.
        result += self._capture_mtp_graphs(profile_only)
        if profile_only:
            # Upstream discards captures in its temporary profiling pool.
            # Model-owned graphs must be recorded during the real capture.
            return result

        capture_first_frame = getattr(self.model, "capture_first_frame_graphs", None)
        if callable(capture_first_frame):
            capture_first_frame()
        capture_stream = getattr(self.model, "capture_stream_decode_graphs", None)
        if callable(capture_stream):
            sizes = [int(s) for s in self.model_state._get_mtp_capture_sizes()]
            capture_stream(sorted(set(sizes + [1])))
        return result

    def _capture_mtp_graphs(self, profile_only: bool) -> int:
        """Capture the MTP graphs and return the device memory they took."""
        model_state = getattr(self, "model_state", None)
        is_graph_runner = getattr(model_state, "_is_mtp_graph_runner", None)
        if model_state is None or not callable(is_graph_runner) or not is_graph_runner():
            return 0
        torch.accelerator.synchronize()
        free_before = torch.accelerator.get_memory_info()[0]
        model_state.capture_mtp_graphs(self._dispatch_mtp_batch_descriptor, restore_capturing=profile_only)
        torch.accelerator.synchronize()
        return max(free_before - torch.accelerator.get_memory_info()[0], 0)

    def _dispatch_mtp_batch_descriptor(self, num_mtp_reqs: int) -> Any:
        capture_sizes = self.model_state._get_mtp_capture_sizes()
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

    def _make_dummy_input_batch(
        self,
        num_reqs: int,
        num_tokens: int,
        *,
        max_query_len: int | None,
        is_padding: bool,
    ) -> Any:
        from vllm.v1.worker.gpu.input_batch import InputBatch

        batch = InputBatch.make_dummy(
            num_reqs,
            num_tokens,
            self.input_buffers,
            max_query_len=max_query_len,
            is_padding=is_padding,
        )
        # Graph capture (and any other dummy pass) must never drive the K-frame
        # loop: its batch carries no real codec history, yet the simulated
        # schedule books a uniform multi-token span that the loop's engage
        # check would otherwise accept (observed: 8x9 capture batch engaged a
        # 9-frame fused graph, one frame too many, and the engine died).
        self._omni_dummy_batch = batch
        return batch

    def _forward_context(self, attn_metadata: Any, vllm_config: Any, **kwargs: Any):
        """Forward-context scope. NPU replaces this with the Ascend context."""
        kwargs.pop("in_profile_run", None)
        kwargs.pop("num_actual_tokens", None)
        return set_forward_context(attn_metadata, vllm_config, **kwargs)

    def _dispatch_batch_descriptor(
        self,
        *,
        num_reqs: int,
        num_toks: int,
        uniform_tok_count: int | None,
        num_active_loras: int,
        use_eager: bool,
        max_query_len: int | None,
        allow_skip_drafts: bool = False,
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
            parallel_config=self.parallel_config,
            # This override never calls ubatch_runner.run. Selecting a
            # multi-ubatch descriptor would dispatch a shape the forward
            # does not execute.
            allow_ubatching=False,
            uniform_decode=uniform_tok_count == self.decode_query_len,
            allow_skip_drafts=allow_skip_drafts,
        )

    def _adjust_dispatch_token_count(
        self,
        batch_req_state: Any,
        num_toks: int,
        max_query_len: int | None,
    ) -> tuple[int, int | None]:
        if batch_req_state is None:
            return num_toks, max_query_len
        num_toks = batch_req_state.num_tokens
        if batch_req_state.has_prefill:
            # Varlen decode graphs replay decodes only. Their query bound
            # alone would admit a short prefill.
            max_query_len = None
        if self.pcp_manager is not None:
            num_toks = self.pcp_manager.get_num_tokens_for_dispatch(
                batch_req_state.num_scheduled_tokens,
                batch_req_state.is_prefilling_np,
            )
        return num_toks, max_query_len

    def _apply_encoder_input_updates(self, scheduler_output: SchedulerOutput) -> None:
        # add_requests initializes RoPE positions from CPU metadata first.
        encoder_cache = self.encoder_cache
        if encoder_cache is None:
            return
        for req_id, updates in scheduler_output.encoder_input_updates.items():
            encoder_cache.update_encoder_inputs(req_id, updates)

    def _num_spec_tokens_to_schedule(self, scheduler_output: SchedulerOutput, dp_sync: Any) -> int:
        skip_drafts = dp_sync.skip_drafts if dp_sync is not None else scheduler_output.skip_drafts
        if skip_drafts:
            return 0
        return scheduler_output.num_spec_tokens_to_schedule

    def _talker_kstep_frames(self) -> int:
        if not current_omni_platform.is_cuda() or not getattr(self.model, "supports_multi_frame_decode", False):
            return 0
        from vllm_omni.model_executor.models.minicpmo_4_5.talker_kstep_contract import talker_kstep_frames

        return talker_kstep_frames(self.model_config.model_stage, self.vllm_config.speculative_config)

    def _dummy_sampler_run(self, hidden_states: Any) -> Any:
        """Skip the sampler warmup while the Talker multi-frame loop is armed.

        Under K-step ``compute_logits`` emits the two-wide continue/stop row
        (the codec stream is sampled in-model), but the sampler's profile pass
        asks for ``num_logprobs`` top-k scores, which is wider than 2 and trips
        ``torch.topk``. The first real step compiles the sampler kernels
        instead; that cost lands once, off the critical path.
        """
        if self._talker_kstep_frames() > 1:
            return None
        return super()._dummy_sampler_run(hidden_states)

    def _build_kstep_model_kwargs_extra(self, model_inputs: dict[str, Any], input_batch: Any) -> dict[str, Any]:
        """Per-step omni kwargs the Talker multi-frame loop reads.

        Mirrors the V1 runner's ``_build_model_kwargs_extra``: the per-request
        runtime buffers (the model state already gathered them into
        ``model_inputs``) plus the exact hidden-row spans this step scheduled.
        The loop only engages on a uniform decode (every request scheduled the
        same K positions); everything else declines to the single forward.
        """
        extra: dict[str, Any] = {
            "model_intermediate_buffer": model_inputs.get("model_intermediate_buffer"),
            "runtime_additional_information": model_inputs.get("runtime_additional_information"),
        }
        num_reqs = int(input_batch.num_reqs)
        nstp = input_batch.num_scheduled_tokens
        counts = [int(nstp[i]) for i in range(num_reqs)]
        spans: list[tuple[int, int]] = []
        start = 0
        for count in counts:
            spans.append((start, start + count))
            start += count
        extra["request_token_spans"] = spans
        # Tag prefilling rows: their spans are multi-token by construction, and
        # both the loop's engage check and the multi-token guard must see them
        # as prefill, not as a decode step whose frame loop declined.
        prefilling = getattr(input_batch, "is_prefilling_np", None)
        buffers = extra.get("model_intermediate_buffer")
        if prefilling is not None and isinstance(buffers, (list, tuple)):
            # Copy only the prefill rows: decode rows must stay the very dict
            # the preprocess hooks write through, so the state they seed is
            # visible to the loop on this same step.
            marked = []
            for index, info in enumerate(buffers):
                if isinstance(info, dict) and index < len(prefilling) and bool(prefilling[index]):
                    prefill_info = dict(info)
                    prefill_info["_omni_is_prefill"] = True
                    marked.append(prefill_info)
                else:
                    marked.append(info)
            extra["model_intermediate_buffer"] = marked
            extra["runtime_additional_information"] = marked
        model = getattr(self, "model", None)
        talker = getattr(model, "talker", model)
        states = getattr(talker, "_request_audio_states", {})
        for info in extra.get("model_intermediate_buffer") or []:
            if isinstance(info, dict) and info.get("req_id") in states:
                info["request_id"] = info["req_id"]
                info["audio_state"] = states[info["req_id"]]
        # The loop's prepare path declines ineligible rows; decode rows of a
        # uniform step are always eligible. Params stay None so the model's
        # stage defaults apply (V1 parity).
        # The first K-frame decode step can arrive before any prefill-time
        # advance of the request's Talker state. Seed only from a confirmed
        # non-placeholder codec id; a missing seed must fall back to the
        # single-frame path rather than silently conditioning on CONTINUE.
        extra["request_sample_eligible"] = [True] * num_reqs
        seed_ids = model_inputs.get("input_ids")
        for index, info in enumerate(buffers or []):
            if not isinstance(info, dict) or index >= len(extra["request_sample_eligible"]):
                continue
            from vllm_omni.worker_v2.warmup import is_warmup_request

            if is_warmup_request(str(info.get("req_id") or info.get("request_id", index))):
                extra["request_sample_eligible"][index] = False
                continue
            if prefilling is not None and index < len(prefilling) and bool(prefilling[index]):
                # Prefill rows: the model's codec chain samples the prefill
                # frame itself on the plain path (V1 parity). Seeding here
                # would gate that chain off via sample_eligible and write a
                # text-token last_code into a per-step buffer copy.
                continue
            state = info.get("audio_state")
            if not isinstance(state, dict):
                state = {}
                info["audio_state"] = state
            if state.get("finished") or int(state.get("step", 0)) > 0:
                continue
            row = int(spans[index][0]) if index < len(spans) else -1
            seeded = None
            handoff = info.get("audio_codes")
            if not isinstance(handoff, dict):
                handoff = info.get("codes")
            candidates = []
            if isinstance(handoff, dict):
                for key in ("accumulated", "current", "audio"):
                    handoff_value = handoff.get(key)
                    if handoff_value is not None:
                        candidates.append(handoff_value)
            elif handoff is not None:
                candidates.append(handoff)
            for handoff_value in candidates:
                if isinstance(handoff_value, torch.Tensor):
                    if handoff_value.numel():
                        candidate = int(handoff_value.reshape(-1)[-1].item())
                    else:
                        continue
                elif isinstance(handoff_value, (list, tuple)) and handoff_value:
                    candidate = int(handoff_value[-1])
                else:
                    continue
                if candidate > 0:
                    seeded = candidate
                    break
            if seeded is None and seed_ids is not None and row >= 0:
                try:
                    tok = seed_ids[row] if getattr(seed_ids, "ndim", 1) == 1 else seed_ids[row, 0]
                    candidate = int(tok.item())
                    if candidate > 0:
                        seeded = candidate
                except (IndexError, RuntimeError, TypeError, ValueError):
                    pass
            if seeded is None:
                # MRv2 prefill seeds frame 1 into the Talker's confirmed
                # history (make_omni_output_mrv2) and into its own audio state,
                # but the per-step runtime buffer handed to the K-frame loop
                # can be a fresh dict that carries only the sampling knobs. In
                # a concurrent tier that left ``audio_codes``/``last_code``
                # missing here, so the row was marked sample_eligible=False:
                # the frame loop then emits only ``continue``, never advances
                # state["step"], and ``state.step < min_tokens`` masks codec
                # EOS forever (C4-C32 audio pinned at the frame cap). Fall back
                # to the confirmed frame before disabling sampling.
                # The Talker stage's model IS MiniCPMO45OmniTTSForConditional-
                # Generation: make_omni_output_mrv2 and the confirmed-history
                # / audio-state dicts live on self.model directly, not on a
                # nested ``talker`` submodule (that name belongs to the Thinker
                # wrapper on another stage).
                _talker = getattr(self, "model", None)
                if _talker is not None:
                    # Prefill seeds under ``info["request_id"]``; the runner's
                    # runtime buffer may key the same request by ``req_id``.
                    # Try both before giving up on the seed.
                    _rids = [
                        str(info.get("request_id") or ""),
                        str(info.get("req_id") or ""),
                        str(index),
                    ]
                    for _rid in dict.fromkeys(r for r in _rids if r):
                        _hist = getattr(_talker, "_request_codec_history", {}).get(_rid) or []
                        if _hist:
                            try:
                                _candidate = int(_hist[-1])
                                if _candidate > 0:
                                    seeded = _candidate
                            except (TypeError, ValueError):
                                pass
                        if seeded is None:
                            _st = getattr(_talker, "_request_audio_states", {}).get(_rid) or {}
                            _lc = _st.get("last_code") if isinstance(_st, dict) else None
                            if isinstance(_lc, int) and _lc > 0:
                                seeded = _lc
                        if seeded is not None:
                            break
            if seeded is None:
                state["_kstep_seed_missing"] = True
                extra["request_sample_eligible"][index] = False
                continue
            state.pop("_kstep_seed_missing", None)
            state["last_code"] = seeded
            state["_kstep_last_code"] = (
                seed_ids.new_tensor([seeded])
                if isinstance(seed_ids, torch.Tensor)
                else torch.tensor([seeded], dtype=torch.long)
            )
            state["step"] = 1
        # MRv2 buffers carry the real request id under ``req_id`` and the
        # request's SamplingParams under ``sampling_params``. The V1 K-step
        # branch keys its host state by ``request_id`` and merges those params
        # (min/max/stop/codec-EOS contract) into it; index ids plus [None]
        # params made every frame see a fresh, contract-less state, so the
        # codec stream ran to the 2048-frame ceiling (80 s of audio for a
        # 51-char answer).
        active_buffers = extra.get("model_intermediate_buffer") or []
        for info in active_buffers:
            if isinstance(info, dict) and "request_id" not in info and info.get("req_id") is not None:
                info["request_id"] = str(info["req_id"])
        extra["request_sampling_params"] = [
            info.get("sampling_params") if isinstance(info, dict) else None for info in active_buffers
        ]
        extra["request_max_tokens_remaining"] = [None] * num_reqs
        # The Thinker ends an utterance by injecting the codec EOS id into the
        # Talker's input stream; the single-frame path reads it as ``last_ids``
        # and forces EOS. A speculative K-row step's inputs are constant
        # ``continue`` drafts, so the only place that signal can appear is the
        # bonus row (span start). Without detecting it here the K-frame loop
        # never sees the turn end and keeps emitting until ``step >= limit``
        # (787 frames / 31 s of audio for a 51-char answer). Marking the state
        # finished makes the loop skip the row and the draft supply stop, so
        # the engine finishes the request on the same step as the single-frame
        # path would.
        model = getattr(self, "model", None)
        talker = getattr(model, "talker", None) if model is not None else None
        codec_eos = getattr(talker if talker is not None else model, "_codec_eos_id", None)
        ib_ids = getattr(input_batch, "input_ids", None)
        logits_idx = getattr(input_batch, "logits_indices", None)
        if codec_eos is not None and int(codec_eos) > 0 and ib_ids is not None and logits_idx is not None:
            try:
                # logits_indices point at the last speculative draft row, which
                # is always the constant continue id; the real previous token
                # (and the upstream codec EOS) sits on each request's first
                # (bonus) row, i.e. the span start.
                last_ids = ib_ids.index_select(0, logits_idx[:num_reqs].long())
                bonus_rows = torch.tensor(
                    [int(span[0]) for span in spans[:num_reqs]],
                    device=ib_ids.device,
                    dtype=torch.long,
                )
                bonus_ids = ib_ids.index_select(0, bonus_rows)
                eos_rows = (
                    ((last_ids == int(codec_eos)) | (bonus_ids == int(codec_eos)))
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
            except (RuntimeError, IndexError, AttributeError, TypeError):
                eos_rows = []
            for index in eos_rows:
                if index >= len(active_buffers):
                    continue
                info = active_buffers[index]
                if not isinstance(info, dict):
                    continue
                state = info.get("audio_state")
                if isinstance(state, dict):
                    state["finished"] = True
        return extra

    def _rewrite_kstep_initial_embeds(
        self,
        inputs_embeds: Any,
        extra: dict[str, Any],
        spans: list[tuple[int, int]],
        input_ids: Any = None,
    ) -> None:
        """Replace each request's first-frame embed with its real last codec.

        A speculative decode step embeds the *drafts* (constant ``continue``
        ids), but the K-frame graph's first frame must be conditioned on the
        codec frame the request actually emitted last -- the in-graph feedback
        (``emb_code`` on the sampled frame) carries frames 2..K but nothing can
        fix frame 1 from inside. Without this rewrite the model conditions on
        the placeholder embed every step and its codec stream never converges
        to EOS (observed as 80 s of runaway audio for a 51-char answer).
        """
        infos = extra.get("model_intermediate_buffer") or []
        emb = getattr(self.model, "emb_code", None)
        if emb is None:
            talker_module = getattr(self.model, "talker", None)
            if talker_module is not None:
                emb = getattr(talker_module, "emb_code", None)
        if not infos or not spans or inputs_embeds is None or emb is None:
            return
        try:
            emb_rows = int(emb[0].weight.shape[0])
        except (AttributeError, IndexError, TypeError, RuntimeError):
            emb_rows = None
        rows: list[int] = []
        codes: list[int] = []
        row_infos: list[dict[str, Any]] = []
        for index, span in enumerate(spans):
            try:
                start, end = int(span[0]), int(span[1])
            except (TypeError, ValueError, IndexError):
                continue
            # A K-row span books K rows for this request; the fused graph
            # conditions frame 1 on the first of them and rebuilds the rest
            # from the in-graph feedback, so only that row is rewritten.
            if end <= start or index >= len(infos):
                continue
            info = infos[index]
            state = info.get("audio_state") if isinstance(info, dict) else None
            if not isinstance(state, dict):
                continue
            code = None
            # Use the last emitted codec, which can differ from the last packed frame.
            confirmed = None
            if isinstance(info, dict):
                host = getattr(self.model, "talker", None)
                if host is None or not hasattr(host, "_request_codec_history"):
                    host = self.model
                history = getattr(host, "_request_codec_history", {})
                for _rid_key in (info.get("request_id"), info.get("req_id")):
                    if _rid_key is None:
                        continue
                    confirmed = history.get(str(_rid_key))
                    if confirmed:
                        break
            if confirmed:
                code = int(confirmed[-1])
            # Only confirmed history or the explicit previous-frame shadow may
            # condition the first row. ``state["codes"]`` is a full-width
            # device mirror and its last element can be an un-emitted row.
            if code is None:
                device_last = state.get("_kstep_last_code")
                if isinstance(device_last, torch.Tensor) and device_last.numel():
                    candidate = int(device_last.reshape(-1)[-1].item())
                    if candidate > 0:
                        code = candidate
            if code is None:
                last_code = state.get("last_code")
                if isinstance(last_code, int) and last_code > 0:
                    code = last_code
            if code is None and int(state.get("step", 0)) <= 1:
                # V1 parity (preprocess_decode_batch embeds ``input_ids[-1:]``):
                # at the first decode step the row's committed token is the
                # PROMPT's last token, so the first K-row's codec slot must be
                # that same id. The K-step's ``input_ids`` rows here carry the
                # constant draft continues (0), not the prompt tail, so read
                # the prompt ids from the request info first and only fall
                # back to ``input_ids`` when the info carries none.
                # Later K-steps resolve through the confirmed history above
                # and never reach this fallback.
                _ptids = info.get("prompt_token_ids") if isinstance(info, dict) else None
                candidate = 0
                if _ptids:
                    try:
                        candidate = int(_ptids[-1])
                    except (TypeError, ValueError, IndexError):
                        candidate = 0
                if not candidate and input_ids is not None and int(input_ids.numel()) > start:
                    try:
                        candidate = int(input_ids[start].item())
                    except (RuntimeError, IndexError, ValueError):
                        candidate = 0
                if candidate > 0 and (emb_rows is None or candidate < emb_rows):
                    # The prompt tail normally lives in the text-vocab space
                    # (audio placeholder ids), not the codec space; embedding
                    # an out-of-range id trips a device-side index assert, so
                    # only accept candidates the codec table can actually
                    # embed.
                    code = candidate
            if code is None or code <= 0:
                continue
            rows.append(start)
            codes.append(code)
            row_infos.append(info)
        if not rows:
            return
        tokens = torch.tensor(codes, device=inputs_embeds.device, dtype=torch.long).unsqueeze(1)
        vectors = emb[0](tokens).to(inputs_embeds.dtype).reshape(len(rows), -1)
        inputs_embeds[rows] = vectors
        for code, info in zip(codes, row_infos, strict=True):
            state = info["audio_state"]
            state["_kstep_last_code"] = code
            state["last_code"] = code
        if input_ids is not None:
            input_ids[rows] = tokens.view(-1).to(input_ids.dtype)

    def _run_kstep_model_forward(
        self,
        model_inputs: dict[str, Any],
        frames: int,
        extra: dict[str, Any],
    ) -> Any:
        """Execute each codec frame against its own one-query attention view."""
        from vllm.forward_context import get_forward_context

        from vllm_omni.platforms.npu.worker import talker_multiframe

        spans = extra.get("request_token_spans") or []
        # Replace the first placeholder with the confirmed previous codec.
        self._rewrite_kstep_initial_embeds(
            model_inputs.get("inputs_embeds"), extra, spans, model_inputs.get("input_ids")
        )
        from vllm_omni.worker.talker_cuda_graph import begin_cuda_narrow_step

        # The eager fallback needs the same per-frame positions and KV slots
        # as graph replay. Reusing the wide context would overwrite KV entries.
        narrow = begin_cuda_narrow_step(
            forward_context=get_forward_context(),
            frames=frames,
            positions=model_inputs["positions"],
            inputs_embeds=model_inputs.get("inputs_embeds"),
            spans=spans,
            graph_owner=self,
        )
        if narrow is None:
            raise RuntimeError("MiniCPM-o MRv2 K-step requires a one-query CUDA attention view")

        def run_model() -> Any:
            active = narrow is not None and getattr(narrow, "context", None) is not None
            kwargs = dict(model_inputs)
            kwargs["input_ids"] = None if active else model_inputs.get("input_ids")
            kwargs["positions"] = narrow.positions if active else model_inputs["positions"]
            kwargs["inputs_embeds"] = narrow.inputs_embeds if active else model_inputs.get("inputs_embeds")
            kwargs.update(extra)
            return self.model(**kwargs)

        try:
            model_output = talker_multiframe.run(
                model=self.model,
                run_model=run_model,
                after_forward=lambda: None,
                inputs_embeds=model_inputs.get("inputs_embeds"),
                input_ids=model_inputs.get("input_ids"),
                frames=frames,
                # ``run`` calls ``make_omni_output(hidden, **model_kwargs,
                # **frame_kwargs)`` where frame_kwargs is built from
                # model_kwargs_extra. v2's model_inputs carries the same omni
                # extras (model_intermediate_buffer, ...); passing both would
                # raise "got multiple values for keyword argument".
                model_kwargs={k: v for k, v in model_inputs.items() if k not in extra},
                model_kwargs_extra=extra,
                narrow=narrow,
                keep_cuda_narrow_on_decline=True,
            )
        finally:
            if narrow is not None:
                narrow.restore()
        # The Talker's vLLM-level head is the two-wide continue/stop row; the
        # rejection sampler drops every accepted token without this gate.
        stop_rows = getattr(self.model, "batch_stop_logits", None)
        talker_multiframe.ensure_stop_token_vocab(
            self, stop_rows if stop_rows is not None else model_output.text_hidden_states
        )
        return model_output

    @torch.inference_mode()
    @staged_writes.runner_context
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
    ) -> Any:
        if not dummy_run:
            # Before finish_requests: finished writes escalate while their slots are live.
            self._prefix_cache_step_begin(scheduler_output)
            self._prepare_native_data_plane(scheduler_output)
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self._apply_encoder_input_updates(scheduler_output)
            self._sync_native_data_plane_payloads(scheduler_output)
            self.block_tables.apply_staged_writes()
            reanchor = getattr(self.model, "apply_duplex_kv_reanchor", None)
            if callable(reanchor):
                reanchor(self, scheduler_output=scheduler_output)
            if self.aux_output_connector is not None:
                self.aux_output_connector.begin_step(scheduler_output.aux_output_connector_metadata)
            if scheduler_output.total_num_scheduled_tokens == 0:
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return self._attach_native_data_plane_signals(
                    self._merge_ec_connector_no_forward(scheduler_output, empty_output)
                )

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        batch_req_state, uniform_tok_count = self.gather_batch_req_state(scheduler_output, dummy_run)
        num_toks, max_query_len = self._adjust_dispatch_token_count(batch_req_state, num_toks, max_query_len)
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
        batch_desc, dp_sync = self._dispatch_batch_descriptor(
            num_reqs=num_reqs,
            num_toks=num_toks,
            uniform_tok_count=uniform_tok_count,
            num_active_loras=num_active_loras,
            use_eager=is_profile or skip_compiled,
            max_query_len=max_query_len,
            allow_skip_drafts=scheduler_output.skip_drafts,
        )

        if batch_desc.num_tokens == 0:
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return self._attach_native_data_plane_signals(
                self._merge_ec_connector_no_forward(scheduler_output, empty_output)
            )

        runner_assisted_request = None
        if (
            not dummy_run
            and batch_desc.cg_mode == CUDAGraphMode.NONE
            and intermediate_tensors is None
            and batch_req_state is not None
            and num_reqs > 0
        ):
            request_hook = getattr(self.model, "get_runner_assisted_full_attention_metadata_request", None)
            context_hook = getattr(self.model, "set_runner_assisted_full_attention_metadata_context", None)
            if callable(request_hook) and callable(context_hook):
                runner_assisted_request = request_hook(
                    req_ids=batch_req_state.req_ids,
                    num_reqs=num_reqs,
                    num_scheduled_tokens=batch_req_state.num_scheduled_tokens,
                    num_computed_tokens=self.req_states.num_computed_tokens_np[batch_req_state.idx_mapping_np],
                    max_num_scheduled_tokens=max_query_len,
                    num_prompt_tokens=self.req_states.prompt_len.np[batch_req_state.idx_mapping_np],
                )
                if runner_assisted_request is not None:
                    if not isinstance(runner_assisted_request, RunnerAssistedFullAttentionMetadataRequest):
                        raise TypeError("runner-assisted full attention metadata hook returned an invalid request")
                    padded_desc = _runner_assisted_batch_descriptor(
                        batch_desc, runner_assisted_request, num_reqs, self.scheduler_config.max_num_seqs
                    )
                    if padded_desc is not None:
                        # Prepare the padded query locations, slot mappings,
                        # and block-table rows through the normal InputBatch
                        # path before requesting FULL attention metadata.
                        batch_desc = padded_desc
                    else:
                        runner_assisted_request = None

        if not dummy_run:
            assert batch_req_state is not None
            input_batch = self.prepare_inputs(scheduler_output, batch_req_state, batch_desc, num_active_loras)
            if runner_assisted_request is not None:
                # Preserve the uniform decode bound. Mixed graphs instead
                # need a stable varlen launch layout across packed layouts.
                query_bound = (
                    max_query_len
                    if runner_assisted_request.num_tokens_padded is None
                    else max(2, runner_assisted_request.num_tokens_padded)
                )
                input_batch = replace(input_batch, max_query_len=query_bound)
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
            # Encode vision and audio before language-model attention metadata.
            # A long video's ViT FlashAttention workspace and the prefill's FA3
            # scheduler workspace do not fit together; V1 runs the encoder first.
            input_ids, inputs_embeds, ec_connector_output = self._prepare_mm_inputs(
                scheduler_output,
                input_batch,
                dummy_run=False,
            )
        else:
            input_batch = self._make_dummy_input_batch(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                max_query_len=batch_desc.max_query_len,
                is_padding=not is_profile,
            )
            if randomize_inputs:
                input_batch.input_ids.random_(0, self.vocab_size)
            if self.pcp_manager is not None:
                input_batch = self.pcp_manager.prepare_inputs_for_capture_or_profile(input_batch)
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch, valid_dummy_state_slots)
                if context_len:
                    set_dummy_context(
                        input_batch,
                        self.block_tables,
                        context_len,
                        self.kv_cache_config.num_blocks,
                        self.max_model_len,
                        input_block_tables=block_tables,
                    )
            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata is required for FULL CUDA graph dummy runs"
                )
                block_tables = None
                slot_mappings = None

        attn_metadata = None
        slot_mappings_by_layer = None
        # Models with their own unified decode graph need FULL attention
        # metadata even when the v2 runner itself is executing eagerly. Only
        # pad request/token metadata to the model-owned unified graph bucket.
        runner_assisted_full_attn = runner_assisted_request is not None
        runner_assisted_capture = runner_assisted_request is not None and runner_assisted_request.for_cudagraph_capture
        if not (dummy_run and skip_attn_for_dummy_run):
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)
            assert block_tables is not None
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                CUDAGraphMode.FULL if runner_assisted_full_attn else batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
                for_capture=(dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL) or runner_assisted_capture,
            )
        if dummy_run:
            # Real steps encode above, before this metadata exists. Dummy steps
            # only fill the static embed buffer and do not run the ViT.
            input_ids, inputs_embeds, ec_connector_output = self._prepare_mm_inputs(
                scheduler_output,
                input_batch,
                dummy_run=True,
            )

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": intermediate_tensors,
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        self._add_legacy_forward_inputs(model_inputs, input_batch)
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

        # --- Omni: Talker multi-frame (K-step) decode ---
        # A decode step that schedules K positions per request runs K codec
        # frames through the platform-neutral ``talker_multiframe`` loop (the
        # same one the V1 runner calls): the loop re-embeds the real previous
        # codec frame per frame and owns its own graph. The v2 graph manager is
        # bypassed for these steps so the two graph layers cannot stack.
        kstep_frames = 0
        kstep_extra: dict[str, Any] = {}
        if not dummy_run and not is_profile:
            from vllm_omni.platforms.npu.worker import talker_multiframe

            expected_frames = self._talker_kstep_frames()
            if expected_frames and not any(is_warmup_request(req_id) for req_id in input_batch.req_ids):
                self.model.flush_cuda_kstep_feedback()
                kstep_extra = self._build_kstep_model_kwargs_extra(model_inputs, input_batch)
            if kstep_extra and scheduler_output.scheduled_spec_decode_tokens:
                kstep_frames = talker_multiframe.applies(self.model, kstep_extra)
                if kstep_frames != expected_frames:
                    raise RuntimeError(
                        f"MiniCPM-o scheduled {expected_frames}-frame Talker drafts "
                        "without an executable multi-frame decode batch"
                    )

        # Target layers a drafter reads (EAGLE-3, DFlash, DSpark). They ride in
        # the forward's tuple output, separate from the Omni auxiliary output.
        draft_aux_hidden_states = None
        # --- Model forward ---
        if batch_desc.cg_mode == CUDAGraphMode.FULL and kstep_frames <= 1:
            # FULL graph replay.  Preprocess already wrote to the static
            # inputs_embeds buffer above.
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(scheduler_output)
            self._last_multimodal_outputs = None
            stepped = self._full_graph_step_output(
                batch_desc,
                model_inputs=model_inputs,
                input_batch=input_batch,
            )
            if stepped is not None:
                # The platform loop already sampled this step.
                hidden_states = stepped.text_hidden_states
                self._last_aux_output = None
                if stepped.multimodal_outputs:
                    self._last_multimodal_outputs = cast(dict[str, Any], stepped.multimodal_outputs)
            else:
                replayed = self.cudagraph_manager.run_fullgraph(batch_desc)
                if self.use_aux_hidden_state_outputs:
                    replayed, draft_aux_hidden_states = replayed
                hidden_states, self._last_aux_output = self._split_fullgraph_output(replayed)
                if hasattr(self.model, "_last_captured_layers"):
                    self.model._last_captured_layers = self._last_aux_output
        else:
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
                num_active_loras=batch_desc.num_active_loras,
            )
            try:
                if runner_assisted_full_attn:
                    self.model.set_runner_assisted_full_attention_metadata_context(
                        enabled=True, num_reqs=input_batch.num_reqs
                    )
                with self._forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=input_batch.num_tokens_after_padding,
                    num_actual_tokens=input_batch.num_tokens,
                    in_profile_run=is_profile,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL if runner_assisted_full_attn else batch_desc.cg_mode,
                    num_tokens_across_dp=dp_sync.num_tokens_across_dp if dp_sync is not None else None,
                    batch_descriptor=batch_descriptor,
                    slot_mapping=slot_mappings_by_layer,
                    skip_compiled=skip_compiled,
                    is_padding=input_batch.is_padding,
                ):
                    self.kv_connector.pre_forward(scheduler_output)
                    if kstep_frames > 1:
                        model_output = self._run_kstep_model_forward(model_inputs, kstep_frames, kstep_extra)
                    elif batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                        assert self.cudagraph_manager is not None
                        model_output = self.cudagraph_manager.run_pw_graph(
                            self.model,
                            model_inputs,
                        )
                    else:
                        model_output = self._call_model(
                            model_inputs,
                            input_batch=input_batch,
                            batch_desc=batch_desc,
                        )
            finally:
                if runner_assisted_full_attn:
                    self.model.set_runner_assisted_full_attention_metadata_context(enabled=False)

            if (
                not dummy_run
                and isinstance(model_output, torch.Tensor)
                and getattr(self.model, "make_omni_output", None) is not None
                # MRv2 exception: when the model exposes the MRv2
                # device-side chain, that chain owns every step (prefill
                # included) via model_state.postprocess_model_output.
                # Running the V1 chain here first would return a
                # truthy-multimodal OmniOutput whose
                # _last_multimodal_outputs makes postprocess_model_output
                # skip the MRv2 chain, so the prefill frame never gets
                # sampled and the first K-row conditions on the placeholder
                # codec embedding (observed 2026-11-10: ~2x audio duration,
                # every first-frame rewrite missing the codec history).
                and getattr(self.model, "make_omni_output_mrv2", None) is None
                and not any(is_warmup_request(str(rid)) for rid in scheduler_output.num_scheduled_tokens)
            ):
                # V1 parity (gpu_model_runner.py:2467): every plain step, the
                # prefill step included, must run the talker's codec chain so
                # each request's audio state is created under its real request
                # id with the sampled prefill frame and the codec contract.
                # Without it the K-step loop engaged a step late on placeholder
                # conditions and the stream diverged from frame one. OmniOutput
                # results (the K-step frame loop) already carry their own
                # bookkeeping and pass through untouched.
                # Synthetic steps are excluded: the kernel warmup drives
                # execute_model with dummy_run=False but `_warmup_*` requests
                # and fabricated spec metadata; running the chain there
                # registers bogus audio states and desyncs the sampler's
                # codec-window warmup (observed 2026-11-10:
                # _apply_codec_window_penalty_gpu scatter shape crash).
                plain_kwargs = dict(model_inputs)
                plain_kwargs.update(kstep_extra or {})
                model_output = self.model.make_omni_output(model_output, **plain_kwargs)

            # Extract hidden_states from model output.
            self._last_aux_output = None
            self._last_multimodal_outputs = None
            if self.use_aux_hidden_state_outputs:
                model_output, draft_aux_hidden_states = model_output
            if isinstance(model_output, OmniOutput):
                hidden_states = model_output.text_hidden_states
                if model_output.multimodal_outputs:
                    self._last_multimodal_outputs = cast(dict[str, Any], model_output.multimodal_outputs)
            elif isinstance(model_output, tuple) and len(model_output) == 2:
                hidden_states, self._last_aux_output = model_output
                if hasattr(self.model, "_last_captured_layers"):
                    self.model._last_captured_layers = self._last_aux_output
            else:
                hidden_states = model_output
        if kstep_extra and kstep_frames == 0 and isinstance(hidden_states, torch.Tensor):
            kstep_extra["request_sample_eligible"] = [
                int(input_batch.num_computed_prefill_tokens_np[i]) + int(input_batch.num_scheduled_tokens[i])
                >= int(input_batch.prefill_len_np[i])
                for i in range(input_batch.num_reqs)
            ]
            output = self.model.make_omni_output(hidden_states, **kstep_extra)
            hidden_states = output.text_hidden_states
            self._last_multimodal_outputs = output.multimodal_outputs
        if (
            not dummy_run
            and isinstance(hidden_states, torch.Tensor)
            # These outputs are materialized by make_omni_output() in the AR
            # sampling phase, so their state update must wait for that phase.
            and not getattr(self.model, "postprocess_requires_multimodal_outputs", False)
        ):
            self.model_state.run_postprocess(hidden_states, input_batch)

        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=draft_aux_hidden_states,
            dp_sync_state=dp_sync,
            finished_req_ids=scheduler_output.finished_req_ids,
            ec_connector_output=ec_connector_output,
            cudagraph_stats=None,
            num_spec_tokens_to_schedule=self._num_spec_tokens_to_schedule(scheduler_output, dp_sync),
        )

        assert isinstance(hidden_states, torch.Tensor)
        return None

    def _full_graph_step_output(self, batch_desc, *, model_inputs, input_batch):
        """Platform hook for FULL-graph replay; None keeps the stock replay."""
        return None

    def _call_model(self, model_inputs: dict[str, Any], *, input_batch: Any, batch_desc: Any) -> Any:
        """Invoke the model once; platform runners may override for frame loops."""
        return self.model(**model_inputs)

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
                gpu_keys: set[tuple[str, str]] = getattr(self.model, "gpu_resident_buffer_keys", set())
                self.model_state.intermediate_buffer.update(req_idx, resolved, gpu_keys)

    # ------------------------------------------------------------------
    # Request lifecycle: clean up intermediate buffer on finish
    # ------------------------------------------------------------------

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        # IMPORTANT: Must query req_id_to_index BEFORE super().finish_requests()
        # because super() calls req_states.remove_request(req_id) which pops the
        # mapping and returns the slot index to free_indices.
        finished = scheduler_output.finished_req_ids
        on_finished = getattr(self.model, "on_requests_finished", None)
        if finished and callable(on_finished):
            # Request-owned state outlives generation slots and chunk boundaries.
            # Notify even if the last chunk already released its runner slot.
            on_finished(finished)
        finish_state = getattr(self.model_state, "on_requests_finished", None)
        if finished and callable(finish_state):
            finish_state(finished)
        preempted = scheduler_output.preempted_req_ids
        suspend_state = getattr(self.model_state, "on_request_preempted", None)
        if preempted and callable(suspend_state):
            for req_id in preempted - finished:
                idx = self.req_states.req_id_to_index.get(req_id)
                if idx is not None:
                    suspend_state(req_id, idx)
        all_done = finished | preempted if preempted else finished
        for req_id in all_done:
            idx = self.req_states.req_id_to_index.get(req_id)
            if idx is not None:
                self.model_state.remove_request(idx)
        super().finish_requests(scheduler_output)
