"""AR GPU Model Runner for vLLM-Omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, make_empty_encoder_model_runner_output
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncGPUModelRunnerOutput,
    IntermediateTensors,
)
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.utils.mm_outputs import build_mm_cpu, partition_payload_list, to_payload_element
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

logger = init_logger(__name__)


def _to_cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type == "cpu":
        return tensor.contiguous()
    return tensor.to("cpu").contiguous()


def _clone_cuda_tensor_payload(value: Any, sources: list[torch.Tensor]) -> Any:
    """Clone CUDA tensors on the current stream before async CPU copies.

    The clone protects async Omni output snapshots from CUDA graph output
    buffers that may be reused by subsequent decode steps. CPU tensors are
    cloned synchronously because they are already host-owned snapshots.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            cloned = value.detach().clone()
            sources.append(cloned)
            return cloned
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_cuda_tensor_payload(v, sources) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_cuda_tensor_payload(v, sources) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_cuda_tensor_payload(v, sources) for v in value)
    return value


def _copy_tensor_payload_to_cpu(value: Any, pin_memory: bool) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cuda":
            return value
        cpu = torch.empty_like(value, device="cpu", pin_memory=pin_memory)
        cpu.copy_(value, non_blocking=True)
        return cpu
    if isinstance(value, dict):
        return {k: _copy_tensor_payload_to_cpu(v, pin_memory) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_tensor_payload_to_cpu(v, pin_memory) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_tensor_payload_to_cpu(v, pin_memory) for v in value)
    return value


class _AsyncCPUPayloadSnapshot:
    def __init__(
        self,
        payload: Any,
        ready_event: torch.cuda.Event | None,
        cuda_sources: list[torch.Tensor],
    ) -> None:
        self.payload = payload
        self._ready_event = ready_event
        self._cuda_sources = cuda_sources
        self._waited = False

    def wait(self) -> None:
        if self._waited:
            return
        if self._ready_event is not None:
            self._ready_event.synchronize()
        self._cuda_sources.clear()
        self._waited = True


def _snapshot_tensor_payload_to_cpu_async(
    value: Any,
    *,
    copy_stream: torch.cuda.Stream,
    pin_memory: bool,
) -> _AsyncCPUPayloadSnapshot:
    cuda_sources: list[torch.Tensor] = []
    cloned = _clone_cuda_tensor_payload(value, cuda_sources)
    if not cuda_sources:
        return _AsyncCPUPayloadSnapshot(cloned, None, cuda_sources)

    source_stream = torch.cuda.current_stream()
    ready_event = torch.cuda.Event()
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(source_stream)
        cpu_payload = _copy_tensor_payload_to_cpu(cloned, pin_memory)
        ready_event.record(copy_stream)
    return _AsyncCPUPayloadSnapshot(cpu_payload, ready_event, cuda_sources)


class _OmniOutputTensorSnapshot(NamedTuple):
    hidden_states: torch.Tensor
    staged_hidden_states_cpu: torch.Tensor | None
    multimodal_outputs: Any
    async_payload: _AsyncCPUPayloadSnapshot | None = None


class OmniAsyncGPUModelRunnerOutput(AsyncGPUModelRunnerOutput):
    def __init__(
        self,
        *,
        model_runner_output_builder: Callable[[], OmniModelRunnerOutput],
        cuda_device: torch.device | int | str | None = None,
        **kwargs: Any,
    ) -> None:
        sampled_token_ids = kwargs.pop("sampled_token_ids")
        logprobs_tensors = kwargs.pop("logprobs_tensors")
        invalid_req_indices = kwargs.pop("invalid_req_indices")
        async_output_copy_stream = kwargs.pop("async_output_copy_stream")
        vocab_size = kwargs.pop("vocab_size")
        routed_experts = kwargs.pop("routed_experts", None)
        if kwargs:
            raise TypeError(f"Unexpected OmniAsyncGPUModelRunnerOutput kwargs: {sorted(kwargs)}")

        self._model_runner_output = None
        self._invalid_req_indices = invalid_req_indices

        self.async_copy_ready_event = torch.Event()
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors
        self._routed_experts = routed_experts

        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            # Keep sampled-token feedback identical to upstream async
            # scheduling. This tensor drives the next decode step, so avoid
            # changing its host-copy allocation semantics while building Omni
            # output asynchronously.
            self.sampled_token_ids_cpu = self._sampled_token_ids.to("cpu", non_blocking=True)
            self._logprobs_tensors_cpu = self._logprobs_tensors.to_cpu_nonblocking() if self._logprobs_tensors else None
            self._routed_experts_cpu = (
                self._routed_experts.to_cpu_nonblocking() if self._routed_experts is not None else None
            )
            self.async_copy_ready_event.record()

        self._model_runner_output_builder = model_runner_output_builder
        self._background_exception: BaseException | None = None
        self._background_thread: threading.Thread | None = None
        self._cuda_device = cuda_device
        self._background_thread = threading.Thread(
            target=self._build_output_in_background,
            daemon=True,
            name="omni-async-output-builder",
        )
        self._background_thread.start()

    def _build_model_runner_output_once(self) -> None:
        if self._model_runner_output is not None:
            return
        with record_function_or_nullcontext("omni_async_output:get_output/build_model_runner_output"):
            self._model_runner_output = self._model_runner_output_builder()
        self._model_runner_output_builder = None

    def _build_output_in_background(self) -> None:
        try:
            if self._cuda_device is not None:
                torch.cuda.set_device(self._cuda_device)
            self._build_model_runner_output_once()
        except BaseException as exc:  # noqa: BLE001 - re-raised by get_output().
            self._background_exception = exc

    def get_output(self) -> OmniModelRunnerOutput:
        background_thread = getattr(self, "_background_thread", None)
        if background_thread is not None:
            background_thread.join()
            self._background_thread = None
            background_exception = getattr(self, "_background_exception", None)
            if background_exception is not None:
                raise background_exception
        self._build_model_runner_output_once()
        with record_function_or_nullcontext("omni_async_output:get_output/finalize_async_sampled_tokens"):
            return super().get_output()


def _ensure_tensor_values(payload: dict[str, object]) -> dict[str, torch.Tensor]:
    """Convert a flattened payload to strictly ``dict[str, torch.Tensor]``.

    Non-tensor scalars (int, float) are wrapped with ``torch.tensor()``.
    Values that cannot be safely converted are dropped with a warning.
    This enforces the tensor-only invariant required by the
    ``OmniEngineCoreOutput.multimodal_output`` wire field and msgspec
    serialization.
    """
    result: dict[str, torch.Tensor] = {}
    for key, val in payload.items():
        if isinstance(val, torch.Tensor):
            result[key] = val
        elif isinstance(val, (int, float, bool)):
            result[key] = torch.tensor(val)
        elif isinstance(val, (list, tuple)):
            try:
                result[key] = torch.tensor(val)
            except (ValueError, TypeError, RuntimeError):
                logger.warning(
                    "Dropping non-tensorizable multimodal output key '%s' (type=%s) from wire payload.",
                    key,
                    type(val).__name__,
                )
        else:
            logger.warning(
                "Dropping non-tensor multimodal output key '%s' (type=%s) from wire payload.",
                key,
                type(val).__name__,
            )
    return result


class ExecuteModelState(NamedTuple):
    scheduler_output: SchedulerOutput
    logits: torch.Tensor | None
    spec_decode_metadata: Any
    spec_decode_common_attn_metadata: Any
    hidden_states: torch.Tensor
    hidden_states_cpu: torch.Tensor | None
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: Any
    cudagraph_stats: Any
    # OMNI: multimodal_outputs field for omni-specific multimodal handling
    multimodal_outputs: Any
    # slot_mappings for attention/drafter (aligned with upstream v1 API)
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None


class GPUARModelRunner(OmniGPUModelRunner, OmniConnectorModelRunnerMixin):
    """Autoregressive GPU model runner that returns hidden states per request.

    Follows the v0.12 two-phase execute/sample flow from GPUModelRunner, and
    reuses Omni hooks for additional_information / multimodal outputs. This
    class only overrides sample_tokens to expose hidden states + multimodal
    outputs per request while keeping Async output semantics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        # Initialize KV cache manager (preserve vllm_config fallback behavior)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
        self._async_chunk = getattr(self.model_config, "async_chunk", False)
        # Worker-connector init is gated by a per-`model_arch` allowlist
        # (covers both producer-side and consumer-side runners for the
        # arches below).  Consumer-wait stages must be registered
        # separately as `(model_arch, model_stage)` tuples in
        # `omni_scheduling_coordinator._FULL_PAYLOAD_INPUT_STAGES`;
        # forgetting that produces a Stage-1 hang on the consumer.
        _OMNI_CONNECTOR_INIT_ARCHS = {
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen2_5OmniForConditionalGeneration",
            "CovoAudioForConditionalGeneration",
            "MiMoAudioModel",
            "Qwen3TTSTalkerForConditionalGeneration",
            "Qwen3TTSCode2Wav",
            "CosyVoice3Model",
            "DyninOmniForConditionalGeneration",
            "IndexTTS2TalkerForConditionalGeneration",
        }
        if getattr(self.model_config, "model_arch", None) in _OMNI_CONNECTOR_INIT_ARCHS:
            self.init_omni_connectors(
                vllm_config=self.vllm_config,
                model_config=self.model_config,
                kv_transfer_manager=self.kv_transfer_manager,
            )
        self._downstream_payload_cache: dict[str, bool] = {}

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

    def _build_model_sampler_output_token_ids(self) -> list[list[int]]:
        """Build decoded-token history for custom model samplers.

        vLLM only populates sampling_metadata.output_token_ids when penalties or
        logits processors require it. CosyVoice3's custom RAS sampler also
        depends on this history, so we reconstruct it directly from the input
        batch for prefer_model_sampler models.
        """
        req_output_token_ids = getattr(self.input_batch, "req_output_token_ids", [])
        req_ids = list(getattr(self.input_batch, "req_ids", []))
        output_token_ids = [list(req_output_token_ids[idx] or []) for idx in range(len(req_ids))]

        sampled_token_ids_cpu = getattr(self.input_batch, "sampled_token_ids_cpu", None)
        async_copy_ready_event = getattr(self.input_batch, "async_copy_ready_event", None)
        prev_req_id_to_index = getattr(self.input_batch, "prev_req_id_to_index", None)
        if sampled_token_ids_cpu is None or not output_token_ids or prev_req_id_to_index is None:
            return output_token_ids

        sampled_token_ids: list[list[int]] | None = None
        for index, req_id in enumerate(req_ids):
            prev_index = prev_req_id_to_index.get(req_id)
            if prev_index is None:
                continue
            req_history = output_token_ids[index]
            if not req_history or req_history[-1] != -1:
                continue
            if sampled_token_ids is None:
                assert async_copy_ready_event is not None
                async_copy_ready_event.synchronize()
                sampled_token_ids = sampled_token_ids_cpu.tolist()
            new_ids = list(sampled_token_ids[prev_index])
            if not new_ids:
                continue
            num_sampled_ids = len(new_ids) if new_ids[-1] != -1 else new_ids.index(-1)
            first_placeholder = req_history.index(-1)
            num_placeholders = len(req_history) - first_placeholder
            num_to_replace = min(num_sampled_ids, num_placeholders)
            req_history[first_placeholder : first_placeholder + num_to_replace] = new_ids[:num_to_replace]

        for index, req_history in enumerate(output_token_ids):
            if -1 in req_history:
                output_token_ids[index] = req_history[: req_history.index(-1)]

        return output_token_ids

    def _sampling_metadata_for_model_sampler(self, sampling_metadata):
        if getattr(self.model, "skips_model_sampler_output_token_history", False):
            return sampling_metadata
        output_token_ids = self._build_model_sampler_output_token_ids()
        if output_token_ids == sampling_metadata.output_token_ids:
            return sampling_metadata
        return replace(sampling_metadata, output_token_ids=output_token_ids)

    def _request_final_stage_id(self, req_id: str) -> int | None:
        info = self.model_intermediate_buffer.get(req_id)
        if not isinstance(info, dict):
            req_state = self.requests.get(req_id)
            info = getattr(req_state, "additional_information_cpu", None)
        if not isinstance(info, dict):
            return None
        val = info.get("omni_final_stage_id")
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _request_needs_downstream_stage_payload(self, req_id: str) -> bool:
        cached = self._downstream_payload_cache.get(req_id)
        if cached is not None:
            return cached
        # Conservative default: keep payload if marker is missing.
        final_stage_id = self._request_final_stage_id(req_id)
        needs_payload = final_stage_id is None or final_stage_id > 0
        self._downstream_payload_cache[req_id] = needs_payload
        return needs_payload

    def _resolve_pooler_payload_req_ids(self, req_ids_output_copy: list[str]) -> tuple[str, list[str]]:
        downstream_req_ids = [rid for rid in req_ids_output_copy if self._request_needs_downstream_stage_payload(rid)]
        engine_output_type = (self.vllm_config.model_config.engine_output_type or "").lower()
        # Single-stage AR TTS models (e.g. VoxCPM2) finish on this stage but still
        # need multimodal payloads for final audio postprocess/output.
        if engine_output_type == "audio" and not downstream_req_ids:
            downstream_req_ids = req_ids_output_copy
        return engine_output_type, downstream_req_ids

    @staticmethod
    def _sparse_mm_req_ids(multimodal_outputs: Any) -> list[str] | None:
        if not isinstance(multimodal_outputs, dict):
            return None
        meta = multimodal_outputs.get("meta")
        req_ids = None
        sparse_audio = False
        if isinstance(meta, dict):
            req_ids = meta.get("req_id")
            sparse_audio = GPUARModelRunner._is_sparse_audio_marker(meta.get("sparse_audio"))
        if req_ids is None:
            req_ids = multimodal_outputs.get("meta.req_id")
            sparse_audio = GPUARModelRunner._is_sparse_audio_marker(multimodal_outputs.get("meta.sparse_audio"))
        if not sparse_audio:
            return None
        if not isinstance(req_ids, list):
            return None
        return [rid for rid in req_ids if isinstance(rid, str)]

    @staticmethod
    def _resolve_sparse_mm_routing(
        *,
        engine_output_type: str,
        req_ids_output_copy: list[str],
        downstream_req_ids: list[str],
        multimodal_outputs: Any,
    ) -> tuple[list[str], dict[str, int], bool]:
        sparse_mm_req_ids = GPUARModelRunner._sparse_mm_req_ids(multimodal_outputs)
        sparse_mm_index = {rid: i for i, rid in enumerate(sparse_mm_req_ids or [])}
        if engine_output_type != "audio" or sparse_mm_req_ids is None:
            return downstream_req_ids, sparse_mm_index, False

        sparse_req_id_set = set(sparse_mm_req_ids)
        sparse_downstream_req_ids = [rid for rid in req_ids_output_copy if rid in sparse_req_id_set]
        return sparse_downstream_req_ids, sparse_mm_index, True

    @staticmethod
    def _is_sparse_audio_marker(value: Any) -> bool:
        if isinstance(value, list):
            return any(str(item).lower() in ("1", "true", "yes", "on") for item in value)
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    def capture_model(self) -> int:
        result = super().capture_model()
        self._capture_talker_mtp_graphs()
        return result

    def shutdown(self) -> None:
        """Release omni-specific GPU resources before upstream shutdown.

        Order of operations (must match upstream's expectation):
          1. Unfreeze Python GC so model weights are collected immediately
             when self.model is set to None (upstream Worker.init_device
             calls gc.freeze() / freeze_gc_heap()).
          2. Destroy omni-specific CUDA graphs (talker MTP) so references to
             model parameters are released before self.model = None.
          3. Clear GPU-side buffers (input_ids, inputs_embeds) and per-request
             caches that may hold GPU tensor references.
          4. Call CUDAGraphWrapper.clear_all_graphs() unconditionally (not just
             on ROCm) to ensure all CUDA graphs including talker MTP are
             released before model weight teardown.
          5. Call BreakableCUDAGraphWrapper.clear_all_graphs() as well, to
             match the upstream ROCm-only pattern but also protect CUDA.
          6. Delegate to upstream GPUModelRunner.shutdown() which sets
             self.model = None, clears KV caches, resets workspace, etc.

        This prevents abrupt GPU memory release during EngineCore subprocess
        exit that can trigger GPU OOM signals when the parent process
        concurrently cleans up its own GPU state.
        """
        # 1. Unfreeze GC so model weights and GPU tensors are collected
        #    immediately when references are dropped (upstream Worker.shutdown
        #    also does this before any teardown).
        gc.unfreeze()

        # 2. Destroy talker MTP CUDA graph wrapper to release captured graphs.
        if hasattr(self, "talker_mtp") and self.talker_mtp is not None:
            self.talker_mtp = None
        self.has_talker_mtp = False

        # 3. Clear GPU-side buffers (small tensors, but every MiB helps).
        if hasattr(self, "input_ids") and self.input_ids is not None:
            self.input_ids = None
        if hasattr(self, "inputs_embeds") and self.inputs_embeds is not None:
            self.inputs_embeds = None

        # 4. Clear per-request caches that may hold GPU tensor references.
        if hasattr(self, "_downstream_payload_cache"):
            self._downstream_payload_cache.clear()
        if hasattr(self, "model_intermediate_buffer"):
            self.model_intermediate_buffer.clear()

        # 5. Release all CUDA graphs unconditionally (upstream only does this
        #    on ROCm; on CUDA the graphs are only freed by Python GC during
        #    interpreter shutdown, which is too late to prevent memory spikes).
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        CUDAGraphWrapper.clear_all_graphs()
        BreakableCUDAGraphWrapper.clear_all_graphs()

        # 6. Delegate to upstream shutdown (model = None, KV caches, workspace).
        super().shutdown()

    def _capture_talker_mtp_graphs(self) -> None:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        if not self.has_talker_mtp or not isinstance(self.talker_mtp, CUDAGraphWrapper):
            return

        from vllm.compilation.monitor import set_cudagraph_capturing_enabled
        from vllm.distributed.parallel_state import graph_capture

        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        num_warmups = self.compilation_config.cudagraph_num_of_warmups
        capture_sizes = sorted(capture_sizes, reverse=True)
        logger.info("Capturing talker_mtp graphs for sizes %s", capture_sizes)

        set_cudagraph_capturing_enabled(True)
        try:
            with torch.inference_mode(), graph_capture(device=self.device):
                for bsz in capture_sizes:
                    _, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
                        num_tokens=bsz,
                        num_reqs=bsz,
                        num_scheduled_tokens_np=np.ones(bsz, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                    )
                    n = batch_desc.num_tokens
                    ids = self.talker_mtp_input_ids.gpu[:n]
                    emb = self.talker_mtp_inputs_embeds.gpu[:n]
                    hid = self.last_talker_hidden.gpu[:n]
                    ts = self.text_step.gpu[:n]

                    for _ in range(num_warmups):
                        with set_forward_context(
                            None,
                            self.vllm_config,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=batch_desc,
                        ):
                            self.talker_mtp(ids, emb, hid, ts)

                    with set_forward_context(
                        None,
                        self.vllm_config,
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=batch_desc,
                    ):
                        self.talker_mtp(ids, emb, hid, ts)
                    torch.accelerator.synchronize()

            logger.info("Captured talker_mtp graphs for %d sizes", len(capture_sizes))
        except RuntimeError as e:
            raise RuntimeError(
                f"talker_mtp graph capture failed for a model that declared talker_mtp_graph_safe=True: {e}"
            ) from e
        finally:
            set_cudagraph_capturing_enabled(False)

    def _model_needs_full_prefix_hidden_states(self) -> bool:
        """Opt-out hook for models whose postprocess only consumes the tail.

        When False, we skip both the per-step GPU->CPU hidden-state write into
        OmniTensorPrefixCache and the merged-tensor reconstruction on hits;
        postprocess receives the normal scheduled-token slice instead. Models
        that need the full cached_prefix + new_tail span (default) are not
        affected.
        """
        model = getattr(self, "model", None)
        return bool(getattr(model, "requires_full_prefix_cached_hidden_states", True))

    def _deferred_prefix_cache_mm_keys(self) -> set[str]:
        """Model-declared multimodal keys whose prefix-cache writes are deferred."""
        model = getattr(self, "model", None)
        keys = getattr(model, "deferred_prefix_cache_mm_keys", ())
        return set(keys or ())

    def _maybe_update_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        hidden_states_cpu: torch.Tensor | None,
        multimodal_outputs: dict,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ):
        """If prefix caching is enabled and it's the last pipeline parallelism rank,
        retrieve the hidden states & multimodal outputs from the prefix cache based
        on our batch slot mappings.
        """
        # Cache hidden states if we've enabled hidden state prefix caching
        # unless this isn't the last pipeline parallelism rank.
        is_last_pp_rank = get_pp_group().is_last_rank
        if hidden_states_cpu is not None and not is_last_pp_rank:
            raise RuntimeError("hidden_states_cpu staging is only valid on the last pipeline parallel rank.")
        if self.omni_prefix_cache is not None and is_last_pp_rank:
            # If this happens, it generally means the model is not following the correct
            # interface yet and is therefore currently not compatible with prefix cache.
            if multimodal_outputs is not None and not isinstance(multimodal_outputs, Mapping):
                logger.warning_once(
                    "prefix caching expects mm outputs to be a dict, but got %s",
                    type(multimodal_outputs),
                )

            hs_for_cache = hidden_states if self._model_needs_full_prefix_hidden_states() else None
            # FIX: The .cpu attribute of slot_mapping is stale (not updated by the Triton
            # _compute_slot_mapping_kernel which only writes to .gpu). We must use .gpu and
            # sync back to CPU to get the correctly computed slot mapping.
            slot_mapping_gpu = self.input_batch.block_table[0].slot_mapping.gpu
            slot_mapping_cpu = slot_mapping_gpu[:num_tokens_padded].cpu()
            self.omni_prefix_cache.update_omni_tensor_prefix_cache(
                hidden_states=hs_for_cache,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                slot_mapping=slot_mapping_cpu,
                num_tokens_padded=num_tokens_padded,
                skip_mm_cache_keys=self._deferred_prefix_cache_mm_keys(),
                hidden_states_cpu=hidden_states_cpu,
            )

    def _maybe_get_combined_prefix_cache_tensors(
        self,
        hidden_states: torch.Tensor,
        hidden_states_cpu: torch.Tensor | None,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ) -> tuple[dict[str, torch.Tensor] | None, dict | None]:
        """If prefix caching is enabled, extract the merged hidden states and multimodal outputs for
        all requests in the batch (including those that aren't a hit on Prefix cache).
        """
        # Prior to applying the post-processing func, extract
        # the prefix cached hidden states and multimodal states.
        combined_hidden_states, combined_multimodal_outputs = None, None
        is_last_pp_rank = get_pp_group().is_last_rank
        if hidden_states_cpu is not None and not is_last_pp_rank:
            raise RuntimeError("hidden_states_cpu staging is only valid on the last pipeline parallel rank.")
        if self.omni_prefix_cache is not None:
            if not is_last_pp_rank:
                raise RuntimeError("Omni prefix-cache tensor merge is only valid on the last pipeline parallel rank.")
            if (
                not self._model_needs_full_prefix_hidden_states()
                and not self.omni_prefix_cache.has_prefix_cached_new_req_ids()
            ):
                return None, None
            if self._model_needs_full_prefix_hidden_states():
                combined_hidden_states = self.omni_prefix_cache.get_merged_hidden_states(
                    query_start_loc=self.query_start_loc.cpu,
                    input_batch=self.input_batch,
                    hidden_states=hidden_states,
                    hidden_states_cpu=hidden_states_cpu,
                    num_scheduled_tokens=num_scheduled_tokens,
                )
            combined_multimodal_outputs = self.omni_prefix_cache.get_merged_multimodal_states(
                query_start_loc=self.query_start_loc.cpu,
                input_batch=self.input_batch,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_scheduled_tokens=num_scheduled_tokens,
            )
        return combined_hidden_states, combined_multimodal_outputs

    def _stage_deferred_prefix_cache_mm_outputs(
        self,
        *,
        scheduler_output: SchedulerOutput,
        multimodal_outputs: Any,
        query_start_loc_cpu: Any,
    ) -> None:
        if self.omni_prefix_cache is None:
            return

        deferred_mm_cache_keys = self._deferred_prefix_cache_mm_keys()
        if not deferred_mm_cache_keys:
            return

        self.omni_prefix_cache.stage_deferred_mm_outputs(
            query_start_loc=query_start_loc_cpu,
            input_batch=self.input_batch,
            multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            deferred_mm_cache_keys=deferred_mm_cache_keys,
        )

    def _prepare_prefix_cache_pooler_payload_sources(
        self,
        *,
        hidden_states: torch.Tensor,
        staged_hidden_states_cpu: torch.Tensor | None,
        multimodal_outputs: Any,
        scheduler_output: SchedulerOutput,
        needs_scheduled_hidden_payload: bool,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None, dict | None]:
        hidden_states_cpu = None
        if needs_scheduled_hidden_payload:
            if staged_hidden_states_cpu is None:
                raise RuntimeError("Prefix-cache hidden-state payload requires staged CPU hidden states.")
            hidden_states_cpu = staged_hidden_states_cpu

        combined_hidden_states, combined_multimodal_outputs = self._maybe_get_combined_prefix_cache_tensors(
            hidden_states,
            staged_hidden_states_cpu,
            multimodal_outputs,
            scheduler_output.num_scheduled_tokens,
        )
        return hidden_states_cpu, combined_hidden_states, combined_multimodal_outputs

    @staticmethod
    def _build_combined_prefix_cache_mm_payload(
        combined_multimodal_outputs: dict,
        *,
        rid: str,
        idx: int,
    ) -> dict[str, object]:
        def _unwrap_lists(v):
            if isinstance(v, list):
                return v[idx] if idx < len(v) else v[0]
            if isinstance(v, dict):
                return {k: _unwrap_lists(sv) for k, sv in v.items()}
            return v

        return {
            mm_key: _unwrap_lists(combined_multimodal_outputs[mm_key][rid])
            for mm_key in combined_multimodal_outputs.keys()
        }

    def _build_omni_mm_payload(
        self,
        *,
        combined_multimodal_outputs: dict | None,
        mm_cpu: dict[str, object] | None,
        rid: str,
        idx: int,
        start: int,
        end: int,
        audio_sparse_output: bool,
        sparse_mm_index: dict[str, int],
        seq_len: int,
    ) -> dict[str, object]:
        if combined_multimodal_outputs:
            return self._build_combined_prefix_cache_mm_payload(
                combined_multimodal_outputs,
                rid=rid,
                idx=idx,
            )

        mm_payload: dict[str, object] = {}
        if not mm_cpu:
            return mm_payload

        for mm_key, mm_val in mm_cpu.items():
            if mm_key in {"meta.req_id", "meta.sparse_audio"}:
                continue
            if audio_sparse_output and isinstance(mm_val, list):
                sparse_idx = sparse_mm_index.get(rid)
                if sparse_idx is None:
                    continue
                if sparse_idx >= len(mm_val):
                    logger.warning(
                        "Sparse multimodal payload mismatch for request %s: index %d >= %d.",
                        rid,
                        sparse_idx,
                        len(mm_val),
                    )
                    continue
                sparse_val = mm_val[sparse_idx]
                mm_payload[mm_key] = sparse_val.clone() if isinstance(sparse_val, torch.Tensor) else sparse_val
                continue
            mm_payload[mm_key] = to_payload_element(
                element=mm_val,
                idx=idx,
                start=start,
                end=end,
                pass_lists_through=False,
                seq_len=seq_len,
            )
        return mm_payload

    def _build_omni_pooler_payload(
        self,
        *,
        rid: str,
        idx: int,
        start: int,
        end: int,
        hidden_states_cpu: torch.Tensor | None,
        req_hidden_states_cpu: dict[str, torch.Tensor] | None,
        combined_hidden_states: dict[str, torch.Tensor] | None,
        combined_multimodal_outputs: dict | None,
        mm_cpu: dict[str, object] | None,
        audio_sparse_output: bool,
        sparse_mm_index: dict[str, int],
        seq_len: int,
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        if not audio_sparse_output:
            if req_hidden_states_cpu is not None and combined_hidden_states is None:
                req_hidden_states = req_hidden_states_cpu[rid]
            else:
                req_hidden_states = self._resolve_req_hidden_states(
                    hidden_states_cpu,
                    combined_hidden_states,
                    rid,
                    start,
                    end,
                )
            if req_hidden_states is not None:
                payload["hidden"] = req_hidden_states

        mm_payload = self._build_omni_mm_payload(
            combined_multimodal_outputs=combined_multimodal_outputs,
            mm_cpu=mm_cpu,
            rid=rid,
            idx=idx,
            start=start,
            end=end,
            audio_sparse_output=audio_sparse_output,
            sparse_mm_index=sparse_mm_index,
            seq_len=seq_len,
        )
        payload.update(mm_payload)
        return payload

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        if self.routed_experts_initialized:
            self.routed_experts_capturer.clear_buffer()

        if not getattr(self, "_warmup_state_cleared", False):
            self._warmup_state_cleared = True
            if hasattr(self.model, "_clear_warmup_state"):
                self.model._clear_warmup_state()

        # Async-write pipeline: apply any pending GPU->CPU prefix-cache writes
        # whose copy event has already fired. Non-blocking — entries whose D2H
        # is still in flight stay queued and will be picked up on the next
        # step's drain. This guarantees any downstream read of
        # ``omni_prefix_cache.hidden_states_cache`` /
        # ``omni_prefix_cache.mm_outputs_cache`` in this step sees the
        # state produced no later than the previous forward step.
        if self.omni_prefix_cache is not None:
            self.omni_prefix_cache.drain_ready_async_writes()

        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        finished_reqs = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if finished_reqs and hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished_reqs.items():
                try:
                    # NOTE: seq_len is the same as num_computed_tokens_cpu in current
                    # async scheduling, since both exclude async placeholders. We use
                    # seq_len since we control it, just in case upstream async scheduler
                    # semantics change in the future.
                    num_computed = data.get("seq_len")

                    model_meta = self.model.get_kv_transfer_metadata(
                        req_id,
                        num_computed_tokens=num_computed,
                    )
                    if model_meta:
                        existing = data.get("custom_metadata") or {}
                        existing.update(model_meta)
                        data["custom_metadata"] = existing
                except Exception as e:
                    logger.warning(f"Failed to get custom metadata from model for {req_id}: {e}")
        self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished_reqs,
            kv_caches=self.kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
            request_id_resolver=self._resolve_global_request_id,
        )

        if hasattr(self, "_omni_connector"):
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
            if self._pending_full_payload_send:
                flush_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
                flush_ids.update({rid for rid in self._pending_full_payload_send if rid not in self.requests})
                if flush_ids:
                    self.flush_full_payload_outputs(flush_ids)

        if self.omni_prefix_cache is not None and scheduler_output.finished_req_ids:
            self.omni_prefix_cache.commit_deferred_mm_outputs(
                set(scheduler_output.finished_req_ids),
                self.input_batch,
            )

        if self.routed_experts_initialized:
            capturer = self.routed_experts_capturer
            if capturer is not None and hasattr(capturer, "finalize_pending_copy"):
                capturer.finalize_pending_copy()

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        if self.speculative_config is not None and self.speculative_config.use_ngram_gpu():
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = scheduler_output.scheduled_spec_decode_tokens.copy()
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            if kv_connector_metadata is not None:
                get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            deferred_state_corrections_fn = self._update_states(scheduler_output)

            # Notify model of finished requests for state cleanup
            if scheduler_output.finished_req_ids and hasattr(self.model, "on_requests_finished"):
                self.model.on_requests_finished(scheduler_output.finished_req_ids)

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)

                    kv_ids = self.kv_extracted_req_ids
                    self.kv_extracted_req_ids = None

                    output = make_empty_encoder_model_runner_output(scheduler_output)
                    if kv_ids:
                        output = copy(output)
                        output.kv_extracted_req_ids = kv_ids
                    return self.attach_omni_connector_output(output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    self._dummy_run(1)

                # Capture KV extraction results before early return;
                # sample_tokens() is skipped on this path so the IDs
                # would otherwise be silently overwritten next step.
                kv_ids = self.kv_extracted_req_ids
                self.kv_extracted_req_ids = None

                if not has_kv_transfer_group():
                    output = EMPTY_MODEL_RUNNER_OUTPUT
                else:
                    output = self.kv_connector_no_forward(scheduler_output, self.vllm_config)

                if kv_ids:
                    output = copy(output)
                    output.kv_extracted_req_ids = kv_ids

                return self.attach_omni_connector_output(output)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )
            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

            has_separate_kv_update = not all(
                all(g.backend.forward_includes_kv_cache_update for g in self.attn_groups[id])
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded if pad_attn or has_separate_kv_update else num_tokens_unpadded,
                num_reqs_padded=(num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            attn_metadata, spec_decode_common_attn_metadata = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                slot_mappings=slot_mappings_by_group,
            )
            self._maybe_attach_attention_metadata_extensions(
                attn_metadata=attn_metadata,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                max_query_len=max_num_scheduled_tokens,
                pad_attn=pad_attn,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(scheduler_output, num_tokens_padded, intermediate_tensors)

        # Let the model adjust inputs before forward (e.g. restore input_ids
        # for multimodal position detection, fix decode position offsets).
        if hasattr(self.model, "prepare_runner_inputs"):
            input_ids, positions = self.model.prepare_runner_inputs(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                req_ids=req_ids[:num_reqs],
                num_computed_tokens=[int(self.input_batch.num_computed_tokens_cpu[i]) for i in range(num_reqs)],
                num_scheduled_tokens=[int(num_scheduled_tokens_np[i]) for i in range(num_reqs)],
                input_ids_buffer=self.input_ids.gpu[:num_tokens_padded],
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        defer_kv_connector_finalize = self.speculative_config is not None
        with (
            nullcontext(),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,  # OMNI: required for KV cache operations
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                defer_finalize=defer_kv_connector_finalize,
            ) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
                sampling_metadata=self.input_batch.sampling_metadata,
                logits_index=logits_indices,
                sampler=self.sampler,
            )

            # [Omni] Map pending ropes metadata to req_ids.
            if hasattr(self.model, "flush_pending_metadata"):
                self.model.flush_pending_metadata(list(req_ids))

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(model_output)
            hidden_states_cpu = None

            # Async-write pipeline (replaces the per-step blocking
            # ``.to("cpu")`` + ``aten::index_put_`` on pageable host memory).
            # Schedules non-blocking GPU->CPU copies on a dedicated stream;
            # the actual CPU scatter into ``hidden_states_cache`` /
            # ``mm_outputs_cache`` happens in ``drain_ready_async_writes``
            # at the top of subsequent execute_model() calls.
            if self.omni_prefix_cache is not None and get_pp_group().is_last_rank:
                hs_for_cache = hidden_states if self._model_needs_full_prefix_hidden_states() else None
                # Some models (e.g. qwen3-tts-talker) opt out of full-hidden-state
                # prefix caching but the downstream pooler payload path still
                # needs a CPU hidden-states view. Materialize it synchronously
                # in that case; the legacy behavior is preserved.
                if hs_for_cache is None:
                    hidden_states_cpu = hidden_states[:num_tokens_unpadded].detach().to("cpu").contiguous()
                slot_mapping_gpu = self.input_batch.block_table[0].slot_mapping.gpu
                self.omni_prefix_cache.schedule_async_write(
                    hidden_states_gpu=hs_for_cache,
                    multimodal_outputs_gpu=(flatten_payload(multimodal_outputs) if multimodal_outputs else None),
                    slot_mapping_gpu=slot_mapping_gpu,
                    num_tokens_unpadded=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded,
                    skip_mm_cache_keys=self._deferred_prefix_cache_mm_keys(),
                )

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                sample_hidden_states = hidden_states[logits_indices.to(hidden_states.device)]
                # Try with sampling_metadata first; fall back to without for models that don't support it
                try:
                    logits = self.model.compute_logits(
                        sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                    )
                except TypeError:
                    logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices.to(hidden_states.device)]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(self.vllm_config, num_tokens_padded)
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    # Try with sampling_metadata first; fall back to without for models that don't support it
                    try:
                        logits = self.model.compute_logits(
                            sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                        )
                    except TypeError:
                        logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            hidden_states_cpu,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            multimodal_outputs,
            slot_mappings,  # OMNI: pass slot_mappings for drafter
        )
        self.kv_connector_output = kv_connector_output

        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        if self._should_return_omni_routed_experts() and hasattr(self, "_positions_cpu"):
            self._omni_routed_experts_d2h(scheduler_output)

        return None

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: Any,
    ):
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            model_sample = getattr(self.model, "sample", None)
            self.input_batch.update_async_output_token_ids()
            if logits is not None and callable(model_sample) and getattr(self.model, "prefer_model_sampler", False):
                # Apply logit bias (min_tokens, allowed_token_ids) before
                # the custom model sampler — the standard GPU sampler does
                # this internally, but prefer_model_sampler bypasses it.
                if hasattr(self.sampler, "logit_bias_state"):
                    self.sampler.logit_bias_state.apply_logit_bias(
                        logits,
                        self.input_batch.expanded_idx_mapping,
                        self.input_batch.idx_mapping_np,
                        self.input_batch.positions[self.input_batch.logits_indices],
                    )
                sampler_output = model_sample(
                    logits,
                    self._sampling_metadata_for_model_sampler(sampling_metadata),
                )
                if sampler_output is not None:
                    return sampler_output
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        return super()._sample(logits, spec_decode_metadata)

    @staticmethod
    def _resolve_req_hidden_states(
        hidden_states_cpu: torch.Tensor | None,
        combined_hidden_states: dict[str, torch.Tensor] | None,
        rid: str,
        start: int,
        end: int,
    ) -> torch.Tensor | None:
        if combined_hidden_states is not None:
            # We always have all request IDs for prefix cache, even for
            # partial cache misses, so this should never happen.
            if rid not in combined_hidden_states:
                raise RuntimeError("Request IDs in the batch are missing from the merged states!")
            return combined_hidden_states[rid]
        # Prefix caching is disabled. hidden_states_cpu may legitimately be
        # None (e.g. sparse audio output or no scheduled hidden payload);
        # callers must omit the "hidden" key in that case.
        if hidden_states_cpu is None:
            return None
        return hidden_states_cpu[start:end]

    def _build_multimodal_outputs(
        self,
        per_req_payloads: list[dict[str, object] | None] | None,
    ) -> list[dict[str, torch.Tensor] | None] | None:
        """Build per-request multimodal output payloads (dedicated channel).

        Reuses the per-request payloads assembled by the pooler-payload loop
        in sample_tokens() (which already handles prefix-cache merging,
        sparse audio output, and partial downstream batches) so the wire
        channel stays consistent with the full-payload accumulation path.
        Enforces the tensor-only invariant required by msgspec: scalars and
        lists are wrapped into tensors, and anything that cannot be safely
        converted is dropped.
        """
        if self.vllm_config.model_config.engine_output_type == "text":
            return None
        if per_req_payloads is None:
            return None
        wire_payloads: list[dict[str, torch.Tensor] | None] = []
        for payload in per_req_payloads:
            if not payload:
                wire_payloads.append(None)
            else:
                wire_payloads.append(_ensure_tensor_values(payload))
        if all(item is None for item in wire_payloads):
            return None
        return wire_payloads

    def _snapshot_query_start_loc_cpu(self) -> Any:
        query_start_loc_cpu = self.query_start_loc.cpu
        if callable(query_start_loc_cpu):
            query_start_loc_cpu = query_start_loc_cpu()
        if isinstance(query_start_loc_cpu, torch.Tensor):
            return query_start_loc_cpu.detach().cpu().clone()
        if isinstance(query_start_loc_cpu, np.ndarray):
            return query_start_loc_cpu.copy()
        if isinstance(query_start_loc_cpu, list):
            return list(query_start_loc_cpu)
        return query_start_loc_cpu

    @staticmethod
    def _snapshot_scheduler_output_for_async_omni_output(
        scheduler_output: SchedulerOutput,
    ) -> SchedulerOutput:
        updates: dict[str, Any] = {}
        for attr in ("num_scheduled_tokens", "scheduled_spec_decode_tokens"):
            val = getattr(scheduler_output, attr, None)
            if isinstance(val, dict):
                updates[attr] = val.copy()
            elif isinstance(val, list):
                updates[attr] = list(val)
        if not updates:
            return scheduler_output
        try:
            return replace(scheduler_output, **updates)
        except TypeError:
            return scheduler_output

    def _should_return_omni_routed_experts(self) -> bool:
        model_config = getattr(self, "model_config", None)
        if model_config is None:
            model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        return bool(getattr(model_config, "enable_return_routed_experts", False)) and bool(
            getattr(self, "routed_experts_initialized", False)
        )

    @staticmethod
    def _model_omni_flag(model: Any, name: str, default: bool = False) -> bool:
        return bool(getattr(model, name, default)) if model is not None else default

    def _runner_model_omni_flag(self, name: str, default: bool = False) -> bool:
        return self._model_omni_flag(getattr(self, "model", None), name, default)

    def _model_omni_pooler_payload_include_hidden(self) -> bool:
        return self._runner_model_omni_flag("omni_pooler_payload_include_hidden", default=True)

    def _should_use_async_omni_output(self) -> bool:
        if not self.use_async_scheduling:
            return False
        if self.omni_prefix_cache is not None:
            return False
        if self.speculative_config is not None:
            return False

        model_config = getattr(self, "model_config", None)
        if model_config is None:
            model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        if not bool(getattr(model_config, "async_chunk", False)):
            return False
        if bool(getattr(model_config, "enable_return_routed_experts", False)):
            return False

        model = getattr(self, "model", None)
        if not self._model_omni_flag(model, "use_async_omni_output"):
            return False
        if self._model_omni_flag(model, "has_postprocess") and not self._model_omni_flag(
            model, "eager_omni_postprocess_before_async_output"
        ):
            return False

        return True

    def _build_omni_async_snapshot_payload(
        self,
        *,
        hidden_states: torch.Tensor,
        staged_hidden_states_cpu: torch.Tensor | None,
        multimodal_outputs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"multimodal_outputs": multimodal_outputs}
        if self._model_omni_pooler_payload_include_hidden():
            payload["hidden_states"] = hidden_states
            payload["staged_hidden_states_cpu"] = staged_hidden_states_cpu
        return payload

    def _snapshot_omni_output_tensors_for_async_output(
        self,
        *,
        use_async_omni_output: bool,
        hidden_states: torch.Tensor,
        staged_hidden_states_cpu: torch.Tensor | None,
        multimodal_outputs: Any,
    ) -> _OmniOutputTensorSnapshot:
        if not use_async_omni_output:
            return _OmniOutputTensorSnapshot(
                hidden_states=hidden_states,
                staged_hidden_states_cpu=staged_hidden_states_cpu,
                multimodal_outputs=multimodal_outputs,
            )

        with record_function_or_nullcontext("omni_async_output:snapshot_cpu_payload"):
            async_payload_snapshot = _snapshot_tensor_payload_to_cpu_async(
                self._build_omni_async_snapshot_payload(
                    hidden_states=hidden_states,
                    staged_hidden_states_cpu=staged_hidden_states_cpu,
                    multimodal_outputs=multimodal_outputs,
                ),
                copy_stream=self._get_or_create_omni_payload_copy_stream(),
                pin_memory=bool(getattr(self, "pin_memory", False)),
            )

        payload = async_payload_snapshot.payload
        hidden_states_snapshot = payload.get("hidden_states")
        if hidden_states_snapshot is None:
            # Models that omit hidden from the async snapshot only need
            # multimodal payloads (for example, talker codes.audio).
            hidden_states_snapshot = hidden_states[:0]

        return _OmniOutputTensorSnapshot(
            hidden_states=hidden_states_snapshot,
            staged_hidden_states_cpu=payload.get("staged_hidden_states_cpu"),
            multimodal_outputs=payload["multimodal_outputs"],
            async_payload=async_payload_snapshot,
        )

    def _maybe_run_eager_omni_postprocess_before_async_output(
        self,
        *,
        hidden_states: torch.Tensor,
        multimodal_outputs: Any,
        num_scheduled_tokens_np: np.ndarray,
        scheduler_output: SchedulerOutput,
        req_ids_output_copy: list[str],
        query_start_loc_cpu: Any,
    ) -> bool:
        """Apply model postprocess on live GPU tensors before async payload D2H."""
        model = getattr(self, "model", None)
        if not self._model_omni_flag(model, "has_postprocess"):
            return False
        if not self._model_omni_flag(model, "eager_omni_postprocess_before_async_output"):
            return False

        _, downstream_req_ids = self._resolve_pooler_payload_req_ids(req_ids_output_copy)
        if not downstream_req_ids:
            return False

        with record_function_or_nullcontext("omni_output_builder:eager_postprocess"):
            self._process_additional_information_updates(
                hidden_states,
                multimodal_outputs,
                num_scheduled_tokens_np,
                scheduler_output,
                None,
                None,
                req_ids_filter=set(downstream_req_ids),
                req_ids=req_ids_output_copy,
                query_start_loc_cpu=query_start_loc_cpu,
            )
        return True

    def _get_or_create_omni_payload_copy_stream(self) -> torch.cuda.Stream:
        stream = getattr(self, "_omni_payload_copy_stream", None)
        if stream is None:
            stream = torch.cuda.Stream()
            self._omni_payload_copy_stream = stream
        return stream

    def _build_omni_model_runner_output_from_snapshot(
        self,
        *,
        scheduler_output: SchedulerOutput,
        hidden_states: torch.Tensor,
        staged_hidden_states_cpu: torch.Tensor | None,
        multimodal_outputs: Any,
        req_ids_output_copy: list[str],
        req_id_to_index_output_copy: dict[str, int],
        valid_sampled_token_ids: list[list[int]],
        logprobs_lists: Any,
        prompt_logprobs_dict: dict[str, Any],
        num_nans_in_logits: Any,
        kv_connector_output: Any,
        ec_connector_output: Any,
        cudagraph_stats: Any,
        kv_extracted_req_ids: list[str] | None,
        seq_len: int,
        num_scheduled_tokens_np: np.ndarray,
        query_start_loc_cpu: Any,
        postprocess_already_applied: bool = False,
    ) -> OmniModelRunnerOutput:
        combined_hidden_states = None
        combined_multimodal_outputs = None

        engine_output_type, downstream_req_ids = self._resolve_pooler_payload_req_ids(req_ids_output_copy)
        downstream_req_ids, sparse_mm_index, audio_sparse_output = self._resolve_sparse_mm_routing(
            engine_output_type=engine_output_type,
            req_ids_output_copy=req_ids_output_copy,
            downstream_req_ids=downstream_req_ids,
            multimodal_outputs=multimodal_outputs,
        )

        needs_pooler_payload = len(downstream_req_ids) > 0
        downstream_req_id_set = set(downstream_req_ids)
        hidden_states_cpu = None
        req_hidden_states_cpu: dict[str, torch.Tensor] | None = None
        include_hidden_payload = self._model_omni_pooler_payload_include_hidden()
        needs_scheduled_hidden_payload = (
            include_hidden_payload
            and needs_pooler_payload
            and (self.omni_prefix_cache is None or not self._model_needs_full_prefix_hidden_states())
        )
        self._stage_deferred_prefix_cache_mm_outputs(
            scheduler_output=scheduler_output,
            multimodal_outputs=multimodal_outputs,
            query_start_loc_cpu=query_start_loc_cpu,
        )

        if self.omni_prefix_cache is None and needs_scheduled_hidden_payload and not audio_sparse_output:
            num_valid_tokens = min(
                int(scheduler_output.total_num_scheduled_tokens),
                int(hidden_states.shape[0]),
            )
            if len(downstream_req_ids) == len(req_ids_output_copy):
                with record_function_or_nullcontext("omni_output_builder:hidden_d2h/scheduled"):
                    hidden_states_cpu = _to_cpu_contiguous(hidden_states[:num_valid_tokens])
            else:
                req_hidden_states_cpu = {}
                with record_function_or_nullcontext("omni_output_builder:hidden_d2h/per_request"):
                    for rid in downstream_req_ids:
                        idx = req_id_to_index_output_copy[rid]
                        start = int(query_start_loc_cpu[idx])
                        sched = int(num_scheduled_tokens_np[idx])
                        end = start + sched
                        req_hidden_states_cpu[rid] = _to_cpu_contiguous(hidden_states[start:end])

        # NOTE: pooler_output here is used only for the full-payload accumulation
        # path (accumulate_full_payload_output) and is NOT passed on the wire via
        # OmniModelRunnerOutput.pooler_output (which is set to None below).
        # The actual multimodal wire transport uses multimodal_outputs instead.
        pooler_output: list[dict[str, object]] | None = None
        if needs_pooler_payload:
            mm_cpu = None
            if self.omni_prefix_cache is not None:
                (
                    hidden_states_cpu,
                    combined_hidden_states,
                    combined_multimodal_outputs,
                ) = self._prepare_prefix_cache_pooler_payload_sources(
                    hidden_states=hidden_states,
                    staged_hidden_states_cpu=staged_hidden_states_cpu,
                    multimodal_outputs=multimodal_outputs,
                    scheduler_output=scheduler_output,
                    needs_scheduled_hidden_payload=needs_scheduled_hidden_payload,
                )
            if combined_multimodal_outputs is None:
                with record_function_or_nullcontext("omni_output_builder:build_mm_cpu"):
                    mm_cpu = build_mm_cpu(
                        flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs
                    )

            with record_function_or_nullcontext("omni_output_builder:process_additional_information"):
                if not postprocess_already_applied:
                    self._process_additional_information_updates(
                        hidden_states,
                        multimodal_outputs,
                        num_scheduled_tokens_np,
                        scheduler_output,
                        combined_hidden_states,
                        combined_multimodal_outputs,
                        req_ids_filter=downstream_req_id_set,
                        req_ids=req_ids_output_copy,
                        query_start_loc_cpu=query_start_loc_cpu,
                    )

            pooler_output = []
            with record_function_or_nullcontext("omni_output_builder:build_pooler_payloads"):
                for rid in req_ids_output_copy:
                    if rid not in downstream_req_id_set:
                        pooler_output.append({})
                        continue
                    idx = req_id_to_index_output_copy[rid]
                    start = int(query_start_loc_cpu[idx])
                    sched = int(num_scheduled_tokens_np[idx])
                    end = start + sched
                    payload = self._build_omni_pooler_payload(
                        rid=rid,
                        idx=idx,
                        start=start,
                        end=end,
                        hidden_states_cpu=hidden_states_cpu,
                        req_hidden_states_cpu=req_hidden_states_cpu,
                        combined_hidden_states=combined_hidden_states,
                        combined_multimodal_outputs=combined_multimodal_outputs,
                        mm_cpu=mm_cpu,
                        audio_sparse_output=audio_sparse_output,
                        sparse_mm_index=sparse_mm_index,
                        seq_len=seq_len,
                    )
                    pooler_output.append(flatten_payload(payload))

        pooler_output = pooler_output or []
        if self._async_chunk:
            pooler_inter, pooler_client = partition_payload_list(pooler_output)
        else:
            # Non-async-chunk still ships the full payload to the next stage (via
            # accumulate_full_payload_output and the inter_stage_outputs field); only
            # client mm keys are split out when async_chunk is enabled. #4527 set this
            # to (None, pooler_output), which skipped accumulation and starved the
            # downstream stage (300s connector-input timeout / empty audio). (PR #4792)
            pooler_inter, pooler_client = pooler_output, pooler_output

        if pooler_inter and self._should_accumulate_full_payload_output():
            with record_function_or_nullcontext("omni_output_builder:accumulate_full_payload_output"):
                for i, rid in enumerate(req_ids_output_copy):
                    req_state = self.requests.get(rid)
                    if req_state is not None and pooler_inter[i]:
                        self.accumulate_full_payload_output(rid, pooler_inter[i], req_state)

        with record_function_or_nullcontext("omni_output_builder:build_multimodal_outputs"):
            inter_stage_outputs = self._build_multimodal_outputs(pooler_inter)
            multimodal_outputs = self._build_multimodal_outputs(pooler_client)

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            routed_experts_lists = None
            if self._should_return_omni_routed_experts():
                routed_experts_lists = self._omni_extract_routed_experts(scheduler_output)
            output = OmniModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=None,
                multimodal_outputs=multimodal_outputs,
                inter_stage_outputs=inter_stage_outputs,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
            )
            output.kv_extracted_req_ids = kv_extracted_req_ids
            with record_function_or_nullcontext("omni_output_builder:get_omni_connector_output"):
                output.omni_connector_output = self.get_omni_connector_output()
            output.routed_experts = routed_experts_lists
        return output

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None

        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            return self.attach_omni_connector_output(
                OmniModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
            )

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            staged_hidden_states_cpu,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            multimodal_outputs,
            slot_mappings,  # OMNI: unpack slot_mappings for drafter
        ) = self.execute_model_state
        self.execute_model_state = None
        seq_len = hidden_states.shape[0]

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)

        # Correct padding values of prompt_token_ids to match the logits vocabulary size
        if logits is not None and not self.input_batch.sampling_metadata.no_penalties:
            smd = self.input_batch.sampling_metadata
            if smd.prompt_token_ids is not None:
                logits_vocab = logits.shape[-1]
                if self.input_batch.vocab_size > logits_vocab:
                    smd.prompt_token_ids = smd.prompt_token_ids.clamp(max=logits_vocab)

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.valid_sampled_token_count_gpu = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,  # OMNI: pass slot_mappings to drafter (upstream v1 API)
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            input_fits_in_drafter = self._input_fits_in_drafter(spec_decode_common_attn_metadata)
            use_gpu_toks = (
                spec_config.use_eagle() or spec_config.uses_draft_model() or spec_config.uses_extract_hidden_states()
            ) and not spec_config.disable_padded_drafter_batch
            if use_gpu_toks:
                assert isinstance(
                    self.drafter,
                    EagleProposer | DFlashProposer | DraftModelProposer | ExtractHiddenStatesProposer | Gemma4Proposer,
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(
                        self.optimistic_seq_lens_cpu,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                    self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)
                    # Since we couldn't run the drafter,
                    # just use zeros for the draft tokens.
                    self._draft_token_ids = torch.zeros(1, device=self.device, dtype=torch.int32).expand(
                        len(self.input_batch.req_ids), self.num_spec_tokens
                    )
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
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
            )

        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        # Finalize KV connector (wait_for_save + clear metadata) after
        # draft model runs. Deferred from target model forward to allow
        # draft model to also save its KV cache.
        if self.speculative_config is not None:
            self.finalize_kv_connector()

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        # kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids_output_copy],
                dtype=np.int32,
            )
        else:
            num_scheduled_tokens_np = np.asarray(num_scheduled_tokens_np, dtype=np.int32).copy()

        query_start_loc_cpu = self._snapshot_query_start_loc_cpu()
        scheduler_output_snapshot = self._snapshot_scheduler_output_for_async_omni_output(scheduler_output)
        req_ids_output_snapshot = list(req_ids_output_copy)
        req_id_to_index_output_snapshot = dict(req_id_to_index_output_copy)
        valid_sampled_token_ids_snapshot = [list(token_ids) for token_ids in valid_sampled_token_ids]
        logprobs_lists_snapshot = copy(logprobs_lists) if logprobs_lists is not None else None
        prompt_logprobs_dict_snapshot = dict(prompt_logprobs_dict) if prompt_logprobs_dict is not None else {}
        num_nans_in_logits_snapshot = (
            dict(num_nans_in_logits) if isinstance(num_nans_in_logits, dict) else num_nans_in_logits
        )

        use_async_omni_output = self._should_use_async_omni_output()
        omni_postprocess_already_applied = False
        if use_async_omni_output:
            omni_postprocess_already_applied = self._maybe_run_eager_omni_postprocess_before_async_output(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                scheduler_output=scheduler_output,
                req_ids_output_copy=req_ids_output_copy,
                query_start_loc_cpu=query_start_loc_cpu,
            )
        output_tensor_snapshot = self._snapshot_omni_output_tensors_for_async_output(
            use_async_omni_output=use_async_omni_output,
            hidden_states=hidden_states,
            staged_hidden_states_cpu=staged_hidden_states_cpu,
            multimodal_outputs=multimodal_outputs,
        )

        def output_builder() -> OmniModelRunnerOutput:
            if output_tensor_snapshot.async_payload is not None:
                with record_function_or_nullcontext("omni_async_output:wait_cpu_payload"):
                    output_tensor_snapshot.async_payload.wait()
            with record_function_or_nullcontext("omni_output_builder:total"):
                return self._build_omni_model_runner_output_from_snapshot(
                    scheduler_output=scheduler_output_snapshot,
                    hidden_states=output_tensor_snapshot.hidden_states,
                    staged_hidden_states_cpu=output_tensor_snapshot.staged_hidden_states_cpu,
                    multimodal_outputs=output_tensor_snapshot.multimodal_outputs,
                    req_ids_output_copy=req_ids_output_snapshot,
                    req_id_to_index_output_copy=req_id_to_index_output_snapshot,
                    valid_sampled_token_ids=valid_sampled_token_ids_snapshot,
                    logprobs_lists=logprobs_lists_snapshot,
                    prompt_logprobs_dict=prompt_logprobs_dict_snapshot,
                    num_nans_in_logits=num_nans_in_logits_snapshot,
                    kv_connector_output=kv_connector_output,
                    ec_connector_output=ec_connector_output,
                    cudagraph_stats=cudagraph_stats,
                    kv_extracted_req_ids=kv_extracted_req_ids,
                    seq_len=seq_len,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    query_start_loc_cpu=query_start_loc_cpu,
                    postprocess_already_applied=omni_postprocess_already_applied,
                )

        if not use_async_omni_output:
            output = output_builder()

            if not self.use_async_scheduling:
                return output
        with record_function_or_nullcontext("gpu_model_runner: AsyncGPUModelRunnerOutput"):
            async_output_cls = OmniAsyncGPUModelRunnerOutput if use_async_omni_output else AsyncGPUModelRunnerOutput
            async_output_kwargs = dict(
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
            if use_async_omni_output:
                async_output = async_output_cls(
                    model_runner_output_builder=output_builder,
                    cuda_device=self.device,
                    **async_output_kwargs,
                )
            else:
                async_output = async_output_cls(
                    model_runner_output=output,
                    **async_output_kwargs,
                )
        with record_function_or_nullcontext("gpu_model_runner: set_async_sampled_token_ids"):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def _resolve_global_request_id(self, req_id: str) -> str:
        """Resolve global request ID from request state."""
        req_state = self.requests.get(req_id)
        if not req_state:
            return req_id

        add_info = self.model_intermediate_buffer.get(req_id, {})
        global_id = add_info.get("global_request_id")
        if global_id:
            if isinstance(global_id, list) and global_id:
                global_id = global_id[0]
            if isinstance(global_id, bytes):
                return global_id.decode("utf-8")
            return str(global_id)
        return req_id
