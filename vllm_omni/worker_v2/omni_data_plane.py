# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from queue import Full, Queue
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import OmniPayload, unflatten_payload
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)
from vllm_omni.worker.payload_span import get_tensor_span, merge_tensor_spans
from vllm_omni.worker_v2.delivery import (
    DeliveryTimeoutError,
    OmniDeliveryManager,
)

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.utils.shm_readiness import ShmReadiness


@dataclass
class _NativeRequestState:
    request_id: str
    external_req_id: str
    prompt_token_ids: list[int]
    additional_information: Any = None
    sampling_params: Any = None
    num_computed_tokens: int = 0
    resumable: bool = False
    output_token_ids: list[int] = field(default_factory=list)
    finished: bool = False

    def snapshot(self, *, include_token_history: bool) -> SimpleNamespace:
        prompt = list(self.prompt_token_ids) if include_token_history else []
        output = list(self.output_token_ids) if include_token_history else []
        finished = self.finished
        request = SimpleNamespace(
            request_id=self.request_id,
            req_id=self.request_id,
            external_req_id=self.external_req_id,
            prompt_token_ids=prompt,
            output_token_ids=output,
            all_token_ids=prompt + output,
            output_token_count=len(self.output_token_ids),
            additional_information=self.additional_information,
            sampling_params=self.sampling_params,
            num_computed_tokens=self.num_computed_tokens,
            resumable=self.resumable,
        )
        request.is_finished = lambda: finished
        return request


@dataclass(frozen=True)
class _NativeOutputBatch:
    req_ids: list[str]
    inter_stage_outputs: list[Any | None] | None
    sampled_token_ids: list[list[int]] | None


_NATIVE_OUTPUT_STOP = object()
_NATIVE_OUTPUT_QUEUE_DEPTH = 8
_DEFAULT_DELIVERY_TIMEOUT_S = 30.0
_DEFAULT_SHUTDOWN_TIMEOUT_S = 5.0
logger = init_logger(__name__)


class OmniRunnerDataPlane(OmniConnectorModelRunnerMixin):
    """MRv2-owned stage transport and request-side payload state."""

    def __init__(self, vllm_config: Any, model_config: Any) -> None:
        self._recv_readiness: ShmReadiness | None = None
        self._native_requests: dict[str, _NativeRequestState] = {}
        self._native_output_lock = threading.RLock()
        self._native_send_lock = threading.Lock()
        self._native_outputs_in_flight: dict[str, int] = defaultdict(int)
        self._native_terminal_pending: set[str] = set()
        self.init_omni_connectors(model_config=model_config)
        self._delivery_manager = OmniDeliveryManager(
            delivery_timeout_s=self._connector_delivery_timeout(),
            shutdown_timeout_s=_DEFAULT_SHUTDOWN_TIMEOUT_S,
        )
        self._can_send = self._custom_process_func is not None
        self._start_output_worker(max_pending_batches=_NATIVE_OUTPUT_QUEUE_DEPTH)

    def _wake_chunk_receiver(self) -> None:
        notifier = getattr(self, "_recv_readiness", None)
        if notifier is not None:
            notifier.wake()

    def register_chunk_recv(self, request: Any) -> None:
        super().register_chunk_recv(request)
        self._wake_chunk_receiver()

    def _recv_loop(self) -> None:
        from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
        from vllm_omni.distributed.omni_connectors.utils.shm_readiness import ShmReadiness

        if not (
            os.getenv("VLLM_OMNI_CHUNK_RECV_EVENTS", "0") == "1"
            and self._async_chunk
            and self._model_mode != "ar"
            and self._stage_id > 0
            and self._from_tp == self._to_tp == 1
            and isinstance(self._omni_connector, SharedMemoryConnector)
        ):
            return super()._recv_loop()
        try:
            # Match the same sequential input edge used by _poll_single_request.
            # Other pipeline edges must not trigger reconciliation of this one.
            notifier = ShmReadiness(
                cohort_name=self._omni_connector.cohort_notification_name(str(self._stage_id - 1), str(self._stage_id))
            )
        except OSError:
            logger.warning("SHM publication notifications unavailable; retaining receiver polling", exc_info=True)
            return super()._recv_loop()
        self._recv_readiness = notifier
        logger.info("Native MRv2 Stage-%s receiver uses SHM publication events", self._stage_id)
        try:
            self._recv_ready_loop(notifier)
        except OSError:
            logger.warning("SHM publication watch failed; reverting to receiver polling", exc_info=True)
            if not self._stop_event.is_set():
                super()._recv_loop()
        finally:
            self._recv_readiness = None
            notifier.close()

    def _recv_ready_loop(self, notifier: Any) -> None:
        interested: set[str] = set()
        arrived: set[str] = set()
        rescan = False
        while not self._stop_event.is_set():
            with self._lock:
                # Only requests whose staged payload/metadata has been handed
                # off can receive another chunk. Rearm after consumption.
                keys = {
                    (
                        f"{self._request_ids_mapping.get(req_id, req_id)}"
                        f"_{self._stage_id - 1}_{self._get_req_chunk[req_id]}"
                    ): req_id
                    for req_id in self._pending_load_reqs
                    if not self._payload_is_consumable(
                        cast(OmniPayload | None, self._local_stage_payload_cache.get(req_id))
                    )
                    and req_id not in self._local_request_metadata
                    and req_id not in self._finished_load_reqs
                }
            # Probe newly armed keys once: data may predate registration, or
            # may have arrived while a previous chunk was staged/in flight.
            candidates = set(keys) if rescan else ((set(keys) - interested) | (arrived & keys.keys()))
            interested = set(keys)
            arrived = set()
            rescan = False
            if candidates:
                # Preserve registration order and publish a single cohort.
                self._poll_pending_requests_once([req_id for key, req_id in keys.items() if key in candidates])
                continue
            arrived, rescan = notifier.wait()

    def _connector_delivery_timeout(self) -> float:
        config = getattr(getattr(self, "_omni_connector", None), "config", None)
        extra = config.get("extra", {}) if isinstance(config, dict) else {}
        value = extra.get("delivery_timeout_s", _DEFAULT_DELIVERY_TIMEOUT_S)
        try:
            timeout = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid connector delivery_timeout_s: {value!r}") from error
        if timeout <= 0:
            raise ValueError("connector delivery_timeout_s must be positive")
        return timeout

    def _create_send_completion(self, *, request_id: str, put_key: str) -> Any:
        return self._delivery_manager.create_ticket(
            request_id=request_id,
            put_key=put_key,
        )

    def _start_output_worker(self, *, max_pending_batches: int) -> None:
        if max_pending_batches <= 0:
            raise ValueError("max_pending_batches must be positive")
        self._native_output_queue: Queue[_NativeOutputBatch | object] = Queue(maxsize=max_pending_batches)
        self._native_output_error_lock = threading.Lock()
        self._native_output_error: BaseException | None = None
        self._connector_send_error_sink = self._record_output_error
        self._native_output_closed = False
        stage_id = getattr(self, "_stage_id", "unknown")
        self._native_output_worker = threading.Thread(
            target=self._output_worker_loop,
            name=f"omni-native-output-{stage_id}",
            daemon=True,
        )
        self._native_output_worker.start()

    def _record_output_error(self, error: BaseException) -> None:
        first_error = False
        with self._native_output_error_lock:
            if self._native_output_error is None:
                self._native_output_error = error
                first_error = True
        if first_error:
            manager = getattr(self, "_delivery_manager", None)
            if manager is not None:
                manager.quarantine(error)

    def _raise_output_error(self) -> None:
        with self._native_output_error_lock:
            error = self._native_output_error
        if error is not None:
            raise RuntimeError(f"native output worker failed: {error}") from error

    def _output_worker_loop(self) -> None:
        while True:
            batch = self._native_output_queue.get()
            try:
                if batch is _NATIVE_OUTPUT_STOP:
                    return
                if self._native_output_error is not None:
                    continue
                assert isinstance(batch, _NativeOutputBatch)
                self.complete_outputs(
                    req_ids=batch.req_ids,
                    inter_stage_outputs=batch.inter_stage_outputs,
                    sampled_token_ids=batch.sampled_token_ids,
                )
            except BaseException as error:
                self._record_output_error(error)
            finally:
                self._native_output_queue.task_done()

    def enqueue_outputs(
        self,
        *,
        req_ids: list[str],
        inter_stage_outputs: list[Any | None] | None,
        sampled_token_ids: list[list[int]] | None,
    ) -> None:
        """Transfer one completed model batch to the ordered output worker."""
        self._raise_output_error()
        if self._native_output_closed:
            raise RuntimeError("native output worker is closed")
        batch = _NativeOutputBatch(
            req_ids=req_ids,
            inter_stage_outputs=inter_stage_outputs,
            sampled_token_ids=sampled_token_ids,
        )
        while True:
            try:
                self._native_output_queue.put(batch, timeout=0.1)
                break
            except Full:
                self._raise_output_error()
        self._raise_output_error()

    def drain_outputs(self) -> None:
        queue = self._native_output_queue
        pending_batches = max(1, queue.qsize() + 1)
        timeout_s = self._delivery_manager.delivery_timeout_s * pending_batches
        deadline = time.monotonic() + timeout_s
        while queue.unfinished_tasks:
            self._raise_output_error()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = DeliveryTimeoutError(
                    "native output queue drain timed out after "
                    f"{timeout_s:.3f}s with {queue.unfinished_tasks} "
                    "unfinished batch(es)"
                )
                self._record_output_error(error)
                self._raise_output_error()
            with queue.all_tasks_done:
                if queue.unfinished_tasks:
                    queue.all_tasks_done.wait(timeout=min(0.05, remaining))
        self._raise_output_error()

    def _stop_output_worker(self) -> None:
        worker = getattr(self, "_native_output_worker", None)
        if worker is None:
            return
        self._native_output_closed = True
        self._delivery_manager.shutdown(RuntimeError("MRv2 output worker shutdown"))
        deadline = time.monotonic() + self._delivery_manager.shutdown_timeout_s
        while worker.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._native_output_queue.put(
                    _NATIVE_OUTPUT_STOP,
                    timeout=min(0.05, remaining),
                )
                break
            except Full:
                continue
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            raise RuntimeError("MRv2 output worker did not stop within its shutdown deadline")
        self._native_output_worker = None

    def register_request(self, request_data: Any) -> None:
        self._raise_output_error()
        req_id = str(request_data.req_id)
        external_req_id = str(getattr(request_data, "external_req_id", None) or req_id)
        with self._native_output_lock:
            self._native_outputs_in_flight.pop(req_id, None)
            self._native_terminal_pending.discard(req_id)
            self._native_requests[req_id] = _NativeRequestState(
                request_id=req_id,
                external_req_id=external_req_id,
                prompt_token_ids=list(getattr(request_data, "prompt_token_ids", None) or []),
                additional_information=getattr(request_data, "additional_information", None),
                sampling_params=getattr(request_data, "sampling_params", None),
                num_computed_tokens=int(getattr(request_data, "num_computed_tokens", 0) or 0),
                resumable=bool(getattr(request_data, "resumable", False)),
            )

    def reserve_outputs(self, req_ids: list[str]) -> None:
        """Keep request state alive until deferred runner outputs are consumed."""
        self._raise_output_error()
        with self._native_output_lock:
            for req_id in dict.fromkeys(req_ids):
                if req_id in self._native_requests:
                    self._native_outputs_in_flight[req_id] += 1

    def _ready_terminal_requests(self) -> set[str]:
        return {
            req_id for req_id in self._native_terminal_pending if self._native_outputs_in_flight.get(req_id, 0) == 0
        }

    def _emit_ready_terminals(self) -> int:
        ready = self._ready_terminal_requests()
        if not ready:
            return 0
        emitted = self.emit_chunks(
            req_ids=[],
            inter_stage_outputs=None,
            sampled_token_ids=None,
            terminal_req_ids=ready,
        )
        self._native_terminal_pending.difference_update(ready)
        for req_id in ready:
            self._native_outputs_in_flight.pop(req_id, None)
        return emitted

    def request_terminal(self, req_ids: set[str]) -> int:
        """Emit terminal markers only after all deferred outputs are enqueued."""
        self._raise_output_error()
        with self._native_output_lock:
            self._native_terminal_pending.update(req_id for req_id in req_ids if req_id in self._native_requests)
            return self._emit_ready_terminals()

    def abort_requests(self, req_ids: set[str]) -> int:
        """Cancel deferred outputs and terminate each live request once."""
        with self._native_output_lock:
            active_req_ids = {req_id for req_id in req_ids if req_id in self._native_requests}
            if not active_req_ids:
                return 0
            for req_id in active_req_ids:
                self._native_outputs_in_flight.pop(req_id, None)
                self._native_terminal_pending.discard(req_id)
            return self.emit_chunks(
                req_ids=[],
                inter_stage_outputs=None,
                sampled_token_ids=None,
                terminal_req_ids=active_req_ids,
            )

    def complete_outputs(
        self,
        *,
        req_ids: list[str],
        inter_stage_outputs: list[Any | None] | None,
        sampled_token_ids: list[list[int]] | None,
    ) -> int:
        """Commit one deferred output batch, then release its lifecycle holds."""
        send_lock_held = False
        with self._native_output_lock:
            entries, finished_req_ids = self._build_chunk_entries(
                req_ids=req_ids,
                inter_stage_outputs=inter_stage_outputs,
                sampled_token_ids=sampled_token_ids,
                terminal_req_ids=set(),
            )
            # Claim connector ordering before publishing the updated request
            # state. request_terminal() can still register a pending terminal
            # while the enqueue runs, but abort cannot overtake this output.
            if entries:
                self._native_send_lock.acquire()
                send_lock_held = True
        try:
            emitted = self._send_chunk_entries(
                entries,
                send_lock_held=send_lock_held,
            )
        finally:
            if send_lock_held:
                self._native_send_lock.release()
        with self._native_output_lock:
            self._cleanup_finished_entries(finished_req_ids)
            for req_id in dict.fromkeys(req_ids):
                in_flight = self._native_outputs_in_flight.get(req_id, 0)
                if in_flight > 1:
                    self._native_outputs_in_flight[req_id] = in_flight - 1
                elif in_flight == 1:
                    self._native_outputs_in_flight.pop(req_id, None)
            return emitted + self._emit_ready_terminals()

    def register_receivers(self, handles: list[Any]) -> None:
        for handle in handles:
            self.register_chunk_recv(handle)

    def put_local_request_metadata(self, req_id: str, metadata: dict[str, Any]) -> None:
        if getattr(self, "payload_native", False):
            metadata = dict(metadata)
            codes = metadata.pop("code_predictor_codes", None)
            metadata["payload_ready"] = self._payload_value_has_content(codes)
        super().put_local_request_metadata(req_id, metadata)

    @classmethod
    def _copy_payload_structure(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._copy_payload_structure(item) for key, item in value.items()}
        if isinstance(value, list):
            return list(value)
        return value

    def _accumulate_payload(self, req_id: str, payload_data: OmniPayload) -> OmniPayload:
        """Accumulate chunk payloads (concat tensors, extend lists)."""
        if req_id not in self._send_side_request_payload:
            self._send_side_request_payload[req_id] = dict(payload_data)
            return dict(self._send_side_request_payload[req_id])

        origin = self._send_side_request_payload[req_id]
        merged = dict(origin)
        raw_ok = payload_data.get("meta", {}).get("override_keys", []) if isinstance(payload_data, dict) else []
        override_keys = {tuple(k) if isinstance(k, list) else k for k in raw_ok}

        for key, value in payload_data.items():
            if isinstance(value, dict):
                origin_sub = origin.get(key)
                merged_sub = dict(origin_sub) if isinstance(origin_sub, dict) else {}
                span_handled: set[str] = set()
                if key == "embed" and isinstance(origin_sub, dict):
                    for tk, sk, ek in (("decode", "decode_token_start", "decode_token_end"),):
                        if tk not in value or (key, tk) in override_keys:
                            continue
                        span = merge_tensor_spans(
                            get_tensor_span(origin_sub, tensor_key=tk, start_key=sk, end_key=ek),
                            get_tensor_span(value, tensor_key=tk, start_key=sk, end_key=ek),
                        )
                        if span is None:
                            continue
                        t, s, e = span
                        merged_sub[tk] = t
                        merged_sub[sk] = s
                        merged_sub[ek] = e
                        span_handled |= {tk, sk, ek}
                for qual, qval in value.items():
                    if qual in span_handled:
                        continue
                    if key == "meta" and qual == "finished":
                        merged_sub[qual] = qval
                        continue
                    if (key, qual) in override_keys:
                        merged_sub[qual] = qval
                        continue
                    osv = merged_sub.get(qual)
                    if isinstance(qval, torch.Tensor) and isinstance(osv, torch.Tensor):
                        merged_sub[qual] = torch.cat([osv, qval], dim=0)
                    elif isinstance(qval, list) and isinstance(osv, list):
                        merged_sub[qual] = osv + qval
                    else:
                        merged_sub[qual] = qval
                merged[key] = merged_sub
            else:
                if key in override_keys:
                    merged[key] = value
                    continue
                ov = origin.get(key)
                if isinstance(value, torch.Tensor) and isinstance(ov, torch.Tensor):
                    merged[key] = torch.cat([ov, value], dim=0)
                elif isinstance(value, list) and isinstance(ov, list):
                    merged[key] = ov + value
                else:
                    merged[key] = value

        self._send_side_request_payload[req_id] = merged
        return dict(merged)

    def pop_local_stage_payload(self, req_id: str) -> Any:
        """Hand one accumulated delta to the model and acknowledge its rows.

        Connector accumulation and ``model_intermediate_buffer`` are separate
        ownership domains. Once decode rows are handed to the model, remove
        them from connector accumulation so the next chunk cannot replay the
        same rows into Talker's ``cached_decode``.
        """
        with self._lock:
            payload = self._local_stage_payload_cache.pop(req_id, None)
            if payload is None:
                return None
            delivered = self._copy_payload_structure(payload)
            external_req_id = self._request_ids_mapping.get(req_id, req_id)
            accumulated = self._send_side_request_payload.get(external_req_id)
            if isinstance(accumulated, dict):
                embed = accumulated.get("embed")
                if isinstance(embed, dict):
                    embed.pop("decode", None)
                    embed.pop("decode_token_start", None)
                    embed.pop("decode_token_end", None)
                hidden_states = accumulated.get("hidden_states")
                if isinstance(hidden_states, dict):
                    hidden_states.pop("output", None)
                ids = accumulated.get("ids")
                if isinstance(ids, dict):
                    ids.pop("output", None)
            self._wake_chunk_receiver()
            return delivered

    def emit_chunks(
        self,
        *,
        req_ids: list[str],
        inter_stage_outputs: list[Any | None] | None,
        sampled_token_ids: list[list[int]] | None,
        terminal_req_ids: set[str],
    ) -> int:
        with self._native_output_lock:
            entries, finished_req_ids = self._build_chunk_entries(
                req_ids=req_ids,
                inter_stage_outputs=inter_stage_outputs,
                sampled_token_ids=sampled_token_ids,
                terminal_req_ids=terminal_req_ids,
            )
            emitted = self._send_chunk_entries(entries)
            self._cleanup_finished_entries(finished_req_ids)
            return emitted

    def _build_chunk_entries(
        self,
        *,
        req_ids: list[str],
        inter_stage_outputs: list[Any | None] | None,
        sampled_token_ids: list[list[int]] | None,
        terminal_req_ids: set[str],
    ) -> tuple[list[tuple[Any, Any | None]], list[str]]:
        if not getattr(self, "_can_send", True):
            finished_req_ids = [req_id for req_id in sorted(terminal_req_ids) if req_id in self._native_requests]
            return [], finished_req_ids
        payload_by_req = {
            req_id: inter_stage_outputs[index]
            for index, req_id in enumerate(req_ids)
            if inter_stage_outputs is not None and index < len(inter_stage_outputs)
        }
        sampled_by_req = {
            req_id: sampled_token_ids[index]
            for index, req_id in enumerate(req_ids)
            if sampled_token_ids is not None and index < len(sampled_token_ids)
        }
        entries: list[tuple[Any, Any | None]] = []
        finished_req_ids: list[str] = []
        for req_id in dict.fromkeys([*req_ids, *sorted(terminal_req_ids)]):
            state = self._native_requests.get(req_id)
            if state is None:
                continue
            state.output_token_ids.extend(int(token_id) for token_id in sampled_by_req.get(req_id, []))
            state.finished = req_id in terminal_req_ids
            payload = payload_by_req.get(req_id)
            if isinstance(payload, dict):
                payload = unflatten_payload(payload)
            if payload is None and not state.finished:
                continue
            include_token_history = bool(state.resumable or self._put_req_chunk.get(state.external_req_id, 0) == 0)
            entries.append(
                (
                    state.snapshot(include_token_history=include_token_history),
                    payload,
                )
            )
            if state.finished:
                finished_req_ids.append(req_id)
        return entries, finished_req_ids

    def _send_chunk_entries(
        self,
        entries: list[tuple[Any, Any | None]],
        *,
        send_lock_held: bool = False,
    ) -> int:
        if not entries:
            return 0
        context: AbstractContextManager
        if send_lock_held:
            context = nullcontext()
        else:
            context = self._native_send_lock
        with context:
            return self.send_chunks(entries, propagate_errors=True, wait_for_delivery=True)

    def _cleanup_finished_entries(self, finished_req_ids: list[str]) -> None:
        for req_id in finished_req_ids:
            self._native_requests.pop(req_id, None)
            self.cleanup_finished_request(req_id)

    def shutdown_omni_connectors(self) -> None:
        """Bound MRv2 shutdown even if a connector call cannot be cancelled.

        Connector ``close()`` runs concurrently with the I/O threads so a
        backend that supports cancellation can release a blocked ``put()``.
        Backends that do not support it are quarantined and left only on
        daemon threads after the bounded deadline; V1 keeps its existing
        shutdown implementation in the shared mixin.
        """
        timeout_s = self._delivery_manager.shutdown_timeout_s
        deadline = time.monotonic() + timeout_s
        self._delivery_manager.shutdown(RuntimeError("MRv2 connector shutdown"))
        self._stop_event.set()
        self._work_available.set()
        self._wake_chunk_receiver()

        close_errors: list[BaseException] = []
        connector = getattr(self, "_omni_connector", None)

        def close_connector() -> None:
            if connector is None:
                return
            try:
                connector.close()
            except BaseException as error:
                close_errors.append(error)

        close_thread = threading.Thread(
            target=close_connector,
            name=f"omni-native-connector-close-{getattr(self, '_stage_id', 'unknown')}",
            daemon=True,
        )
        close_thread.start()

        live_threads: list[str] = []
        for name, thread in (
            ("recv", getattr(self, "_recv_thread", None)),
            ("save", getattr(self, "_save_thread", None)),
            ("connector-close", close_thread),
        ):
            if thread is None:
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                live_threads.append(name)

        if close_errors:
            raise RuntimeError("MRv2 connector close failed") from close_errors[0]
        if live_threads:
            raise RuntimeError(
                f"MRv2 connector shutdown exceeded {timeout_s:.3f}s; blocked thread(s): {', '.join(live_threads)}"
            )

    def close(self) -> None:
        error: BaseException | None = None
        try:
            self.drain_outputs()
        except BaseException as exc:
            error = exc
        finally:
            self._delivery_manager.shutdown(error or RuntimeError("MRv2 data plane close"))
            try:
                self._stop_output_worker()
            except BaseException as exc:
                if error is None:
                    error = exc
            try:
                self.shutdown_omni_connectors()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error
