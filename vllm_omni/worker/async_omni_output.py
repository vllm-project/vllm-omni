# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Async Omni output materialization shared by the AR model runners.

The AR runners build a per-request Omni payload (hidden states plus multimodal
outputs) after every decode step. Doing that inline serializes the payload
construction with the next forward pass, so the runners instead snapshot the
device tensors to pinned host memory on a dedicated copy stream and build the
payload on a background thread.

Both the CUDA runner (``vllm_omni/worker/gpu_ar_model_runner.py``) and the
Ascend runner (``vllm_omni/platforms/npu/worker/npu_ar_model_runner.py``) use
this machinery, so it lives here rather than in either platform's module.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.utils.config import stage_sends_async_output
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.utils.mm_outputs import build_mm_cpu, partition_payload_list, snapshot_mm_payload
from vllm_omni.worker.output.payload_build import build_omni_mm_payload
from vllm_omni.worker.sparse_audio import resolve_sparse_mm_routing

logger = init_logger(__name__)


def _ensure_tensor_values(payload: dict[str, object]) -> dict[str, torch.Tensor]:
    """Convert a flattened payload to strictly ``dict[str, torch.Tensor]``.

    Non-tensor scalars (int, float, bool) are wrapped with ``torch.tensor()``.
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


def _accel_module(device_type: str):
    """Return the torch accelerator submodule for a device type.

    NPU tensors report ``device.type == "npu"`` and need the ``torch.npu``
    stream/event APIs. Do not silently fall back to CUDA when the caller
    asked for NPU — that mixes Event/Stream APIs across devices.
    """
    if device_type == "npu":
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeError("requested npu accelerator but torch.npu is not available")
        return npu
    return torch.cuda


def _resolve_accel_module(
    *,
    device: torch.device | int | str | None,
    copy_stream: Any,
    sampled_token_ids: Any | None = None,
):
    """Pick ``torch.npu`` / ``torch.cuda`` from the runner device, not a default."""
    if device is not None:
        return _accel_module(torch.device(device).type)
    stream_device = getattr(copy_stream, "device", None)
    if stream_device is not None:
        device_type = stream_device.type if hasattr(stream_device, "type") else str(stream_device)
        return _accel_module(str(device_type))
    token_device = getattr(sampled_token_ids, "device", None)
    if token_device is not None and getattr(token_device, "type", "cpu") != "cpu":
        return _accel_module(token_device.type)
    raise RuntimeError(
        "OmniAsyncGPUModelRunnerOutput could not resolve the accelerator module; "
        "pass cuda_device=self.device from the runner."
    )


def _to_cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type == "cpu":
        return tensor.contiguous()
    return tensor.to("cpu").contiguous()


def _clone_accel_tensor_payload(value: Any, sources: list[torch.Tensor]) -> Any:
    """Clone accelerator tensors on the current stream before async CPU copies.

    The clone protects async Omni output snapshots from graph output buffers
    that may be reused by subsequent decode steps. CPU tensors are cloned
    synchronously because they are already host-owned snapshots.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            cloned = value.detach().clone()
            sources.append(cloned)
            return cloned
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_accel_tensor_payload(v, sources) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_accel_tensor_payload(v, sources) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_accel_tensor_payload(v, sources) for v in value)
    return value


def _copy_tensor_payload_to_cpu(value: Any, pin_memory: bool) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu":
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
        ready_event: Any | None,
        accel_sources: list[torch.Tensor],
    ) -> None:
        self.payload = payload
        self._ready_event = ready_event
        self._accel_sources = accel_sources
        self._waited = False

    def wait(self) -> None:
        if self._waited:
            return
        if self._ready_event is not None:
            self._ready_event.synchronize()
        self._accel_sources.clear()
        self._waited = True


def _snapshot_tensor_payload_to_cpu_async(
    value: Any,
    *,
    copy_stream: Any,
    pin_memory: bool,
) -> _AsyncCPUPayloadSnapshot:
    accel_sources: list[torch.Tensor] = []
    cloned = _clone_accel_tensor_payload(value, accel_sources)
    if not accel_sources:
        return _AsyncCPUPayloadSnapshot(cloned, None, accel_sources)

    accel = _accel_module(accel_sources[0].device.type)
    source_stream = accel.current_stream()
    ready_event = accel.Event()
    with accel.stream(copy_stream):
        copy_stream.wait_stream(source_stream)
        cpu_payload = _copy_tensor_payload_to_cpu(cloned, pin_memory)
        ready_event.record(copy_stream)
    return _AsyncCPUPayloadSnapshot(cpu_payload, ready_event, accel_sources)


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
        num_nans = kwargs.pop("num_nans", None)
        # Upstream AsyncGPUModelRunnerOutput added check_ep_fault / _has_fault
        # for EP all2all fault tolerance (PR #43637). Omni doesn't use this
        # feature but must consume the kwarg to prevent TypeError from stray
        # kwargs and initialize the attribute so super().get_output() works.
        kwargs.pop("check_ep_fault", False)
        if kwargs:
            raise TypeError(f"Unexpected OmniAsyncGPUModelRunnerOutput kwargs: {sorted(kwargs)}")

        # Built lazily by _build_model_runner_output_once(). Leave the parent
        # attribute unset so mypy keeps the ModelRunnerOutput type; assigning
        # None here would pin the field to None and reject the later builder().
        self._invalid_req_indices = invalid_req_indices

        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors
        self._routed_experts = routed_experts
        self._has_fault: torch.Tensor | None = None
        # Upstream b1e12d142d (PR #51304) added device-side NaN-in-logits
        # counts (num_nans) to AsyncGPUModelRunnerOutput. Omni keeps the
        # counts on the async copy stream and lets super().get_output()
        # populate num_nans_in_logits from the CPU copy.
        self._num_nans = num_nans

        accel = _resolve_accel_module(
            device=cuda_device,
            copy_stream=async_output_copy_stream,
            sampled_token_ids=sampled_token_ids,
        )
        self.async_copy_ready_event = accel.Event()
        default_stream = accel.current_stream()
        with accel.stream(async_output_copy_stream):
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
            self._num_nans_cpu = self._num_nans.to("cpu", non_blocking=True) if self._num_nans is not None else None
            self.async_copy_ready_event.record()

        self._model_runner_output_builder: Callable[[], OmniModelRunnerOutput] | None = model_runner_output_builder
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
        builder = self._model_runner_output_builder
        if builder is None or getattr(self, "_model_runner_output", None) is not None:
            return
        with record_function_or_nullcontext("omni_async_output:get_output/build_model_runner_output"):
            self._model_runner_output = builder()
        self._model_runner_output_builder = None

    def _build_output_in_background(self) -> None:
        try:
            if self._cuda_device is not None:
                device = torch.device(self._cuda_device)
                _accel_module(device.type).set_device(device)
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
        # Upstream AsyncGPUModelRunnerOutput.get_output() accesses
        # self._has_fault for EP all2all fault tolerance (PR #43637).
        # Ensure the attribute exists even when __init__ was bypassed
        # (e.g. unit tests using object.__new__).
        if not hasattr(self, "_has_fault"):
            self._has_fault = None
        # Upstream b1e12d142d (PR #51304) also touches _num_nans/_num_nans_cpu
        # in get_output(). Guard them the same way for object.__new__ tests.
        if not hasattr(self, "_num_nans"):
            self._num_nans = None
        if not hasattr(self, "_num_nans_cpu"):
            self._num_nans_cpu = None
        with record_function_or_nullcontext("omni_async_output:get_output/finalize_async_sampled_tokens"):
            return super().get_output()  # type: ignore[return-value]


class AsyncOmniOutputRunnerMixin:
    """Gate and snapshot helpers for deferred Omni output construction.

    Mixed into the AR runners, which provide the runner state these methods
    read (``use_async_scheduling``, ``omni_prefix_cache``, ``model``, ...) and
    the Omni hooks they call back into.
    """

    query_start_loc: Any
    use_async_scheduling: Any
    omni_prefix_cache: Any
    speculative_config: Any
    device: Any
    model: Any
    model_config: Any
    vllm_config: Any
    _async_chunk: bool
    requests: dict[str, Any]
    model_intermediate_buffer: Any
    _downstream_payload_cache: dict[str, bool]
    supports_mm_inputs: bool
    routed_experts_initialized: bool

    def _request_final_stage_id(self, req_id: str) -> int | None:
        info = self.model_intermediate_buffer.get(req_id)
        if not isinstance(info, dict):
            req_state = self.requests.get(req_id)
            info = getattr(req_state, "additional_information_cpu", None)
        if not isinstance(info, dict):
            return None
        val = info.get("omni_final_stage_id")
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _request_needs_downstream_stage_payload(self, req_id: str) -> bool:
        cached = self._downstream_payload_cache.get(req_id)
        if cached is not None:
            return cached
        final_stage_id = self._request_final_stage_id(req_id)
        if final_stage_id is None:
            # Conservative default while the marker is missing: keep the
            # payload, but do NOT memoize — the marker arrives via
            # `model_intermediate_buffer`, which may be unpopulated on the
            # first call (memoizing here pinned the request to True forever,
            # never refreshing once the marker landed).
            return True
        needs_payload = final_stage_id > 0
        self._downstream_payload_cache[req_id] = needs_payload
        return needs_payload

    def _resolve_pooler_payload_req_ids(self, req_ids: list[str]) -> tuple[str, list[str]]:
        downstream_req_ids = [rid for rid in req_ids if self._request_needs_downstream_stage_payload(rid)]
        engine_output_type = (self.vllm_config.model_config.engine_output_type or "").lower()
        if self._client_multimodal_output_keys():
            downstream_req_ids = req_ids
        # Single-stage AR TTS models (e.g. VoxCPM2) finish on this stage but still
        # need multimodal payloads for final audio postprocess/output.
        elif engine_output_type == "audio" and not downstream_req_ids:
            downstream_req_ids = req_ids
        return engine_output_type, downstream_req_ids

    def _process_additional_information_updates(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def _should_accumulate_full_payload_output(self) -> bool:
        raise NotImplementedError

    def _maybe_get_combined_prefix_cache_tensors(
        self,
        hidden_states: torch.Tensor,
        hidden_states_cpu: torch.Tensor | None,
        multimodal_outputs: Any,
        num_scheduled_tokens: Any,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any] | None]:
        raise NotImplementedError

    @staticmethod
    def _resolve_req_hidden_states(
        hidden_states_cpu: torch.Tensor | None,
        combined_hidden_states: dict[str, torch.Tensor] | None,
        rid: str,
        start: int,
        end: int,
    ) -> torch.Tensor | None:
        if combined_hidden_states is not None:
            if rid not in combined_hidden_states:
                raise RuntimeError("Request IDs in the batch are missing from the merged states!")
            return combined_hidden_states[rid]
        if hidden_states_cpu is None:
            return None
        return hidden_states_cpu[start:end]

    def _build_multimodal_outputs(
        self,
        per_req_payloads: Sequence[dict[str, object] | None] | None,
    ) -> list[dict[str, object]] | None:
        """Build per-request multimodal output payloads (dedicated channel).

        Reuses the per-request payloads assembled by the pooler-payload loop
        (prefix-cache merge, sparse audio, partial downstream batches) so the
        wire channel stays consistent with the full-payload accumulation path.
        Enforces the tensor-only msgspec invariant: scalars and lists become
        tensors; anything that cannot be converted is dropped.
        """
        if self.vllm_config.model_config.engine_output_type == "text" and not self._client_multimodal_output_keys():
            return None
        if per_req_payloads is None:
            return None
        wire_payloads: list[dict[str, object] | None] = []
        for payload in per_req_payloads:
            if not payload:
                wire_payloads.append(None)
            else:
                wire_payloads.append(_ensure_tensor_values(payload))
        if all(item is None for item in wire_payloads):
            return None
        return cast(list[dict[str, object]], wire_payloads)

    def _model_needs_full_prefix_hidden_states(self) -> bool:
        raise NotImplementedError

    def _stage_deferred_prefix_cache_mm_outputs(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def accumulate_full_payload_output(self, rid: str, payload: Any, request: Any) -> None:
        raise NotImplementedError

    def _omni_extract_routed_experts(self, scheduler_output: SchedulerOutput) -> Any:
        raise NotImplementedError

    def get_omni_connector_output(self) -> Any:
        raise NotImplementedError

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

    @staticmethod
    def _model_omni_flag(model: Any, name: str, default: bool = False) -> bool:
        return bool(getattr(model, name, default)) if model is not None else default

    def _runner_model_omni_flag(self, name: str, default: bool = False) -> bool:
        return self._model_omni_flag(getattr(self, "model", None), name, default)

    def _client_multimodal_output_keys(self) -> tuple[str, ...]:
        raw = getattr(
            getattr(self, "model", None),
            "omni_client_multimodal_output_keys",
            (),
        )
        if not isinstance(raw, tuple) or any(not isinstance(key, str) or not key for key in raw):
            raise TypeError("omni_client_multimodal_output_keys must be a tuple of non-empty strings")
        if len(raw) != len(set(raw)):
            raise ValueError("omni_client_multimodal_output_keys must not contain duplicates")
        return raw

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
                # NOTE: vLLM v0.24.0's GPUModelRunner no longer exposes a
                # ``self.pin_memory`` attribute (it uses a module-level
                # ``PIN_MEMORY`` constant instead), so the old
                # ``getattr(self, "pin_memory", False)`` silently fell back to
                # False. That allocated the async D2H snapshot destination in
                # *pageable* host memory, which turns ``copy_(non_blocking=True)``
                # into a fully synchronous, stream-stalling copy (~240 ms/step
                # on the 17.5k-token Thinker prefill). Resolve pinning from the
                # platform helper so the copy is a true async cudaMemcpyAsync.
                pin_memory=is_pin_memory_available(),
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
        """Apply model postprocess on live device tensors before payload D2H."""
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

    def _get_or_create_omni_payload_copy_stream(self) -> Any:
        stream = getattr(self, "_omni_payload_copy_stream", None)
        if stream is None:
            stream = _accel_module(self.device.type).Stream()
            self._omni_payload_copy_stream = stream
        return stream

    def _should_return_omni_routed_experts(self) -> bool:
        model_config = getattr(self, "model_config", None)
        if model_config is None:
            model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        return bool(getattr(model_config, "enable_return_routed_experts", False)) and bool(
            getattr(self, "routed_experts_initialized", False)
        )

    def _should_defer_full_payload_d2h(self) -> bool:
        """Keep opted-in full payloads on device until request completion."""
        return (
            getattr(self, "omni_prefix_cache", None) is None
            and self._runner_model_omni_flag("omni_payload_at_request_end")
            and self._should_accumulate_full_payload_output()
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
        hidden_seq_len: int,
        scheduled_seq_len: int,
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

        payload.update(
            build_omni_mm_payload(
                combined_multimodal_outputs=combined_multimodal_outputs,
                mm_cpu=mm_cpu,
                rid=rid,
                idx=idx,
                start=start,
                end=end,
                audio_sparse_output=audio_sparse_output,
                sparse_mm_index=sparse_mm_index,
                hidden_seq_len=hidden_seq_len,
                scheduled_seq_len=scheduled_seq_len,
            )
        )
        return payload

    def _build_omni_step_outputs(
        self,
        pooler_inter: Sequence[dict[str, object] | None] | None,
        pooler_client: Sequence[dict[str, object] | None] | None,
        *,
        defer_full_payload_d2h: bool,
    ) -> tuple[list[dict[str, object]] | None, list[dict[str, object]] | None]:
        if defer_full_payload_d2h:
            return None, None
        inter_stage_outputs = self._build_multimodal_outputs(pooler_inter)
        multimodal_outputs = (
            inter_stage_outputs if pooler_client is pooler_inter else self._build_multimodal_outputs(pooler_client)
        )
        return inter_stage_outputs, multimodal_outputs

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
        num_scheduled_tokens_np: np.ndarray,
        query_start_loc_cpu: Any,
        postprocess_already_applied: bool = False,
        omni_connector_output: Any | None = None,
        skip_accumulate_full_payload: bool = False,
    ) -> OmniModelRunnerOutput:
        combined_hidden_states = None
        combined_multimodal_outputs = None

        engine_output_type, downstream_req_ids = self._resolve_pooler_payload_req_ids(req_ids_output_copy)
        downstream_req_ids, sparse_mm_index, audio_sparse_output = resolve_sparse_mm_routing(
            engine_output_type=engine_output_type,
            req_ids_output_copy=req_ids_output_copy,
            downstream_req_ids=downstream_req_ids,
            multimodal_outputs=multimodal_outputs,
        )

        needs_pooler_payload = len(downstream_req_ids) > 0
        downstream_req_id_set = set(downstream_req_ids)
        defer_full_payload_d2h = needs_pooler_payload and self._should_defer_full_payload_d2h()
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

        pooler_output: list[dict[str, object]] | None = None
        if needs_pooler_payload:
            hidden_seq_len = int(hidden_states.shape[0])
            scheduled_seq_len = int(scheduler_output.total_num_scheduled_tokens)
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
                flat_mm = flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs
                if defer_full_payload_d2h:
                    with record_function_or_nullcontext("omni_output_builder:snapshot_mm_payload"):
                        mm_cpu = snapshot_mm_payload(flat_mm)
                else:
                    with record_function_or_nullcontext("omni_output_builder:build_mm_cpu"):
                        mm_cpu = build_mm_cpu(flat_mm)

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
                        hidden_seq_len=hidden_seq_len,
                        scheduled_seq_len=scheduled_seq_len,
                    )
                    pooler_output.append(flatten_payload(payload))

        pooler_output = pooler_output or []
        pooler_inter: Sequence[dict[str, object] | None] | None
        pooler_client: Sequence[dict[str, object] | None] | None
        if self._async_chunk and stage_sends_async_output(self.model_config):
            pooler_inter, pooler_client = partition_payload_list(pooler_output)
        else:
            pooler_inter, pooler_client = pooler_output, pooler_output

        client_output_keys = self._client_multimodal_output_keys()
        if client_output_keys:
            allowed = frozenset(client_output_keys)
            pooler_client = [
                {key: value for key, value in payload.items() if key in allowed} for payload in pooler_output
            ]

        if pooler_inter and not skip_accumulate_full_payload and self._should_accumulate_full_payload_output():
            with record_function_or_nullcontext("omni_output_builder:accumulate_full_payload_output"):
                for i, rid in enumerate(req_ids_output_copy):
                    req_state = self.requests.get(rid)
                    if req_state is not None and pooler_inter[i]:
                        self.accumulate_full_payload_output(rid, pooler_inter[i], req_state)

        with record_function_or_nullcontext("omni_output_builder:build_multimodal_outputs"):
            inter_stage_outputs, multimodal_outputs = self._build_omni_step_outputs(
                pooler_inter,
                pooler_client,
                defer_full_payload_d2h=defer_full_payload_d2h,
            )

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
            if omni_connector_output is None:
                with record_function_or_nullcontext("omni_output_builder:get_omni_connector_output"):
                    omni_connector_output = self.get_omni_connector_output()
            output.omni_connector_output = omni_connector_output
            output.routed_experts = routed_experts_lists
        return output
