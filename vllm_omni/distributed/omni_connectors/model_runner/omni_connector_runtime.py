# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni connector lifecycle and shared model-runner transport state."""

from __future__ import annotations

import importlib
import inspect
import os
import threading
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import (
    ConnectorSpec,
    get_stage_connector_role,
)

logger = init_logger("vllm_omni.worker.omni_connector_model_runner_mixin")

if TYPE_CHECKING:
    from vllm_omni.config.model import OmniModelConfig
    from vllm_omni.distributed.omni_connectors.connectors.base import (
        OmniConnectorBase,
    )
    from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
        OmniKVTransferManager,
    )


def needs_omni_connector(model_config: Any) -> bool:
    """Whether a runner owns an input, output, or explicitly routed connector."""
    return (
        bool(getattr(model_config, "requires_full_payload_input", False))
        or bool(getattr(model_config, "custom_process_next_stage_input_func", None))
        or get_stage_connector_role(model_config) is not None
    )


def _should_create_payload_connector(model_config: Any) -> bool:
    """Whether this stage owns runner payload transport for its edge.

    Sender edges may instead be owned solely by KV transfer. Receivers still
    need a connector even though they do not declare a downstream payload hook.
    """
    if get_stage_connector_role(model_config) != "sender":
        return True

    next_stage_func = getattr(model_config, "custom_process_next_stage_input_func", None)
    return isinstance(next_stage_func, str) and bool(next_stage_func)


def should_accumulate_full_payload_output(model_config, custom_process_func) -> bool:
    """Producer-side structural gate.

    Fires iff the stage explicitly declares a downstream full-payload
    producer hook via ``custom_process_next_stage_input_func``.  Consumer
    stages may have ``custom_process_input_func`` values that can be
    mechanically derived to ``*_full_payload`` helper names in the same
    module; those are intentionally not enough to make the stage a producer.
    """
    if custom_process_func is None:
        return False
    if getattr(model_config, "async_chunk", False):
        return False
    if getattr(model_config, "final_output", False):
        return False
    next_stage_func = getattr(model_config, "custom_process_next_stage_input_func", None)
    if not isinstance(next_stage_func, str) or not next_stage_func:
        return False
    return getattr(model_config, "model_stage", None) is not None


class _OmniConnectorRuntimeMixin:
    """Own connector lifecycle, shared state, and KV transfer delegation."""

    _omni_connector: Any
    _kv_transfer_manager: Any
    _async_chunk: bool
    _model_mode: str
    _stage_id: int
    _next_stage_id: int
    _from_tp: int
    _to_tp: int
    _local_rank: int
    _custom_process_func_path: str | None
    _custom_process_func: Any
    _custom_process_supports_is_finished: bool | None
    _put_req_chunk: dict[str, int]
    _get_req_chunk: dict[str, int]
    _ramp_chunk_count: dict[str, int]
    _adaptive_states: dict[str, Any]
    _send_side_request_payload: dict[str, dict[str, Any]]
    _code_prompt_token_ids: dict[str, list[list[int]]]
    _cached_ic: dict[str, int]
    _request_ids_mapping: dict[str, str]
    _pending_load_reqs: dict[str, Any]
    _finished_load_reqs: set[str]
    _pending_save_reqs: dict[str, deque[Any]]
    _pending_save_counts: dict[str, int]
    _deferred_send_cleanup: set[str]
    _chunk_ready_req_ids: set[str]
    _chunk_finished_req_ids: set[str]
    _stage_recv_req_ids: set[str]
    _full_payload_pending_broadcast_req_ids: set[str]
    _async_chunk_updated_req_ids: set[str]
    _local_stage_payload_cache: dict[str, dict[str, Any]]
    _local_request_metadata: dict[str, dict[str, Any]]
    _chunk_stream_completed: set[str]
    _pending_full_payload_send: dict[str, tuple[Any, ...]]
    _kv_sent_req_ids: list[str]
    _kv_pending_transfers: dict[str, dict[str, Any]]
    _kv_active_transfers: set[str]
    _kv_completed_transfers: set[str]
    _kv_triggered_requests: set[str]
    _lock: Any
    _stop_event: Any
    _work_available: Any
    _recv_thread: Any
    _save_thread: Any
    _omni_connector_initialized: bool
    _full_payload_replace_keys_cached: frozenset[Any]
    _should_accumulate_full_payload_output_cached: bool

    vllm_config: Any
    requests: dict[str, Any]
    model_intermediate_buffer: dict[str, Any]

    _recv_loop: Any
    _save_loop: Any
    flush_full_payload_outputs: Any
    _custom_process_supports_is_finished_kwarg: Any
    _get_local_tp_group: Any

    # ------------------------------------------------------------------ #
    #  Init / Shutdown
    # ------------------------------------------------------------------ #

    def init_omni_connectors(
        self,
        model_config: OmniModelConfig,
        kv_transfer_manager: OmniKVTransferManager | None = None,
    ) -> None:
        """Initialize connectors and background threads.

        Args:
            model_config: Stage-level model config with connector settings.
            kv_transfer_manager: Existing KV transfer manager to delegate to.
        """
        self._omni_connector: OmniConnectorBase | None = (
            self._create_connector(model_config) if _should_create_payload_connector(model_config) else None
        )
        self._kv_transfer_manager = kv_transfer_manager

        self._async_chunk: bool = getattr(model_config, "async_chunk", False)
        self._model_mode: str = getattr(model_config, "worker_type", "ar")
        stage_id = getattr(model_config, "stage_id", 0)
        if isinstance(stage_id, str):
            stage_id = int(stage_id)
        self._stage_id: int = stage_id if isinstance(stage_id, int) else 0

        self._custom_process_func_path, self._custom_process_func = self._load_custom_func(model_config)
        self._custom_process_supports_is_finished = self._custom_process_supports_is_finished_kwarg()
        logger.debug(
            "[Stage-%s] init_omni_connectors: async_chunk=%s, custom_process_func=%s, connector=%s, func_path=%s",
            self._stage_id,
            self._async_chunk,
            self._custom_process_func,
            type(self._omni_connector).__name__ if self._omni_connector else None,
            self._custom_process_func_path,
        )

        # -- next stage ID (from connector config or default stage_id + 1) --
        self._next_stage_id: int = self._resolve_next_stage_id(model_config)

        # -- heterogeneous TP rank support --
        rank_cfg = self._parse_rank_mapping(model_config)
        if self._kv_transfer_manager is not None:
            topology = getattr(self._kv_transfer_manager, "tp_topology", None)
            effective_mapping = (
                getattr(topology, "source_tp_size", None),
                getattr(topology, "target_tp_size", None),
                getattr(topology, "local_rank", None),
            )
            if all(isinstance(value, int) for value in effective_mapping):
                rank_cfg = {
                    "from_tp": effective_mapping[0],
                    "to_tp": effective_mapping[1],
                    "local_rank": effective_mapping[2],
                }
        self._from_tp: int = rank_cfg["from_tp"]
        self._to_tp: int = rank_cfg["to_tp"]
        self._local_rank: int = rank_cfg["local_rank"]
        if self._kv_transfer_manager is not None:
            self._kv_transfer_manager.kv_send_key_builder = self.get_rank_aware_kv_send_keys
            self._kv_transfer_manager.kv_recv_key_builder = self.get_rank_aware_kv_keys
            self._kv_transfer_manager.kv_payload_merger = self._merge_rank_sharded_kv_payloads
            self._kv_transfer_manager.kv_payload_slicer = self._slice_rank_sharded_kv_payload

        # -- chunk index tracking (ported from OmniChunkTransferAdapter) --
        self._put_req_chunk: dict[str, int] = defaultdict(int)
        self._get_req_chunk: dict[str, int] = defaultdict(int)
        # Segment-local chunk counter: incremented alongside _put_req_chunk
        # and popped at request cleanup. Note: the mixin path (uniproc mode)
        # does not have segment boundary infrastructure; multi-segment support
        # is only available via chunk_transfer_adapter (distributed path).
        self._ramp_chunk_count: dict[str, int] = defaultdict(int)
        self._adaptive_states: dict[str, Any] = {}
        # Send-side async accumulation / staging buffer. Receive-side payload
        # ownership lives in ``_local_stage_payload_cache``.
        self._send_side_request_payload: dict[str, dict[str, Any]] = {}
        self._code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)
        self._cached_ic: dict[str, int] = {}
        self._request_ids_mapping: dict[str, str] = {}

        # -- async I/O state (shared by chunk + full_payload_mode) --
        self._pending_load_reqs: dict[str, Any] = {}
        self._finished_load_reqs: set[str] = set()
        self._pending_save_reqs: dict[str, deque] = {}
        self._pending_save_counts: dict[str, int] = defaultdict(int)
        self._deferred_send_cleanup: set[str] = set()
        # -- per-cycle output accumulator --
        self._chunk_ready_req_ids: set[str] = set()
        self._chunk_finished_req_ids: set[str] = set()
        self._stage_recv_req_ids: set[str] = set()
        self._full_payload_pending_broadcast_req_ids: set[str] = set()
        self._async_chunk_updated_req_ids: set[str] = set()

        # -- Model Runner local payload cache (RFC §2.4) --
        # Full stage payloads land here first on the recv side. We
        # intentionally do not write connector recv results straight into
        # `model_intermediate_buffer`: runner-owned runtime state is
        # materialized later by `_sync_local_stage_payloads()` on the
        # model thread. This keeps recv timing separate from execute-step
        # visibility and avoids mixing connector I/O with model runtime
        # ownership.
        self._local_stage_payload_cache: dict[str, dict[str, Any]] = {}
        # Lightweight scheduling metadata pending delivery to the Scheduler.
        self._local_request_metadata: dict[str, dict[str, Any]] = {}

        # -- persistent set of request IDs whose chunk stream is complete --
        # Prevents re-registration after the finish sentinel has been received.
        self._chunk_stream_completed: set[str] = set()

        # -- full_payload_mode: accumulate latest pooler_output per request,
        #    send only when the request finishes (next-cycle flush) --
        self._pending_full_payload_send: dict[str, tuple[Any, ...]] = {}

        # -- KV sent accumulator --
        self._kv_sent_req_ids: list[str] = []

        # -- KV transfer lifecycle (absorbed from scheduler) --
        # Requests marked for KV transfer: {req_id: {seq_len, block_ids}}
        self._kv_pending_transfers: dict[str, dict[str, Any]] = {}
        # Requests whose KV transfer has been submitted but not yet acked
        self._kv_active_transfers: set[str] = set()
        # Requests whose KV transfer is complete (acked by kv_extracted_req_ids)
        self._kv_completed_transfers: set[str] = set()
        # Dedup guard: requests that have already triggered KV transfer
        self._kv_triggered_requests: set[str] = set()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._work_available = threading.Event()

        # Start background threads only when there's a connector
        self._recv_thread: threading.Thread | None = None
        self._save_thread: threading.Thread | None = None
        if self._omni_connector is not None:
            self._recv_thread = threading.Thread(
                target=self._recv_loop,
                daemon=True,
                name="omni-mixin-recv",
            )
            self._recv_thread.start()
            self._save_thread = threading.Thread(
                target=self._save_loop,
                daemon=True,
                name="omni-mixin-save",
            )
            self._save_thread.start()

        # Explicit "fully initialised" marker so other parts of the runner
        # (e.g. _update_states cleanup) can branch on a stable contract
        # instead of probing for private mixin attribute names.  Must be set
        # only after every field above has been bound, so a partially
        # constructed mixin is never observable as initialised.
        self._omni_connector_initialized = True

    def shutdown_omni_connectors(self) -> None:
        """Stop background threads and release connector resources."""
        self._stop_event.set()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=5)
        if self._save_thread is not None:
            self._save_thread.join(timeout=5)
        if self._omni_connector is not None:
            try:
                self._omni_connector.close()
            except Exception:
                pass

    def cleanup_finished_request(self, req_id: str) -> None:
        """Clean up per-request state after a request is fully finished.

        Call this when a request is freed from the model runner to prevent
        memory leaks in the mixin's tracking dicts/sets.

        Two senders use different keys: ``send_chunk`` keys per-request
        state under the EXTERNAL id (after mapping resolution), while
        ``send_full_payload_outputs`` keys under the INTERNAL id. To cover
        both modes (and forward compat with id-rename scenarios) we attempt
        cleanup against both keys; the entry that doesn't exist for the
        active mode is a no-op pop.  Only the key that actually has pending
        saves is added to ``_deferred_send_cleanup`` so the bg save's
        decrement path drains it without leaving orphans.
        """
        # Force-flush any pending full-payload accumulator entry before
        # cleanup proceeds.  Without this, finished requests with no
        # downstream consumer (e.g. text-only on multi-modal arch) leave
        # the entry orphaned in _pending_full_payload_send across requests,
        # which empirically destabilises subsequent thinker forwards by
        # making prefix-cache reuse observe stale accumulator state.  The
        # flush is idempotent when the entry has already been flushed by the
        # scheduler-driven path, but this cleanup path runs for every request,
        # so skip it entirely when the request never accumulated a payload.
        if req_id in self._pending_full_payload_send:
            try:
                self.flush_full_payload_outputs({req_id})
            except Exception:
                # Cleanup must still proceed regardless of flush errors here --
                # we already gated on ``_omni_connector_initialized`` upstream,
                # so any exception here reflects a real connector-side issue
                # (shared memory corruption, background thread crash) worth
                # surfacing rather than silently swallowing.
                logger.warning(
                    "flush_full_payload_outputs(%s) raised during cleanup; continuing tear-down.",
                    req_id,
                    exc_info=True,
                )

        ext_id = self._request_ids_mapping.pop(req_id, None)
        keys_to_clean: list[str] = [req_id]
        if ext_id is not None and ext_id != req_id:
            keys_to_clean.append(ext_id)

        with self._lock:
            keys_pending = [k for k in keys_to_clean if self._pending_save_counts.get(k, 0)]
            for k in keys_pending:
                self._deferred_send_cleanup.add(k)
            for k in keys_to_clean:
                if k in keys_pending:
                    continue
                self._put_req_chunk.pop(k, None)
                self._send_side_request_payload.pop(k, None)
                self._code_prompt_token_ids.pop(k, None)
                self._cached_ic.pop(k, None)
                self._ramp_chunk_count.pop(k, None)
                self._adaptive_states.pop(k, None)
            self._kv_pending_transfers.pop(req_id, None)
            self._kv_active_transfers.discard(req_id)
            self._kv_completed_transfers.discard(req_id)
            self._kv_triggered_requests.discard(req_id)
        self._cleanup_recv_delivery_state(req_id)

    def drop_inactive_request_delivery_state(self, req_id: str) -> None:
        """Clear recv-side state for inactive requests."""
        ext_id = self._request_ids_mapping.pop(req_id, None)
        if hasattr(self, "_lock"):
            with self._lock:
                self._drop_send_side_payload_state(req_id, ext_id)
        else:
            self._drop_send_side_payload_state(req_id, ext_id)
        self._cleanup_recv_delivery_state(req_id)

    def _drop_send_side_payload_state(self, req_id: str, ext_id: str | None) -> None:
        if ext_id is not None:
            self._send_side_request_payload.pop(ext_id, None)
            self._cached_ic.pop(ext_id, None)
        self._send_side_request_payload.pop(req_id, None)
        self._cached_ic.pop(req_id, None)

    def _cleanup_recv_delivery_state(self, req_id: str) -> None:
        """Clear recv-side delivery-cycle state."""
        if hasattr(self, "_lock"):
            with self._lock:
                self._clear_recv_delivery_state(req_id)
        else:
            self._clear_recv_delivery_state(req_id)

    def _clear_recv_delivery_state(self, req_id: str) -> None:
        self._get_req_chunk.pop(req_id, None)
        self._pending_load_reqs.pop(req_id, None)
        self._finished_load_reqs.discard(req_id)
        self._chunk_ready_req_ids.discard(req_id)
        self._chunk_finished_req_ids.discard(req_id)
        self._chunk_stream_completed.discard(req_id)
        self._stage_recv_req_ids.discard(req_id)
        self._full_payload_pending_broadcast_req_ids.discard(req_id)
        self._async_chunk_updated_req_ids.discard(req_id)
        self._local_stage_payload_cache.pop(req_id, None)
        self._local_request_metadata.pop(req_id, None)

    def prune_inactive_requests(self, active_req_ids: Any) -> set[str]:
        """Drop connector state for requests that no longer exist locally.

        Preempted / unscheduled requests are expected to stay in
        ``self.requests`` and therefore remain untouched. This only prunes
        stale request IDs that have already fallen out of the active request
        map, preventing background recv/send bookkeeping from outliving the
        request lifecycle.
        """
        if active_req_ids is None:
            return set()

        active_req_ids = set(active_req_ids)
        pending_req_ids = set(getattr(self, "_pending_load_reqs", {}).keys())
        received_req_ids = set(getattr(self, "_stage_recv_req_ids", set()))
        received_req_ids.update(getattr(self, "_full_payload_pending_broadcast_req_ids", set()))
        received_req_ids.update(getattr(self, "_local_request_metadata", {}).keys())
        # Pending recv requests may not yet be in the caller's active set
        # (e.g. WAITING_FOR_CHUNK requests live in the coordinator's internal
        # queues, not in model runner self.requests). Protect them so that
        # legitimate waiting requests are not pruned.
        #
        # Likewise, a full payload can arrive on the background recv thread
        # after the scheduler_output snapshot for the current execute_model()
        # cycle was already materialized. Those requests may briefly live only
        # in recv-side buffers/local cache until the next scheduler cycle wakes
        # them up; pruning them here drops the payload before stage_recv can be
        # published.
        active_req_ids.update(pending_req_ids)
        active_req_ids.update(received_req_ids)
        stale_req_ids: set[str] = set()

        # NOTE: _pending_load_reqs is excluded from the scan list because
        # all its entries are unconditionally protected above.  The mixin
        # cannot distinguish a legitimately-waiting pending recv from an
        # orphaned one (only the coordinator/scheduler knows).
        #
        # Requests with freshly received full payloads / local stage payloads
        # are also protected above. Their scheduler wake-up may lag the recv
        # thread by one execute_model() cycle, especially when the request was
        # added after the current scheduler_output snapshot.
        #
        # Orphaned pending recv entries (e.g. from upstream stage crash) are
        # handled by collect_timed_out_request_ids() -- on
        # OmniSchedulingCoordinator for full-payload requests, and on
        # OmniChunkTransferAdapter for async-chunk ones -- which detect
        # wait-time violations.  The scheduler then removes the request from
        # its queues, sets FINISHED_ERROR, and calls _free_request() which
        # ultimately triggers cleanup_finished_request() here.
        for attr_name in (
            "_request_ids_mapping",
            "_get_req_chunk",
            "_finished_load_reqs",
            "_chunk_ready_req_ids",
            "_chunk_finished_req_ids",
            "_chunk_stream_completed",
            "_stage_recv_req_ids",
            "_full_payload_pending_broadcast_req_ids",
            "_async_chunk_updated_req_ids",
            "_local_stage_payload_cache",
            "_local_request_metadata",
            "_kv_pending_transfers",
            "_kv_active_transfers",
            "_kv_completed_transfers",
            "_kv_triggered_requests",
        ):
            state = getattr(self, attr_name, None)
            if isinstance(state, dict):
                stale_req_ids.update(req_id for req_id in state if req_id not in active_req_ids)
            elif isinstance(state, set):
                stale_req_ids.update(req_id for req_id in state if req_id not in active_req_ids)

        for req_id in stale_req_ids:
            self.cleanup_finished_request(req_id)

        return stale_req_ids

    def drop_inactive_request_runtime_state(self, req_id: str) -> None:
        """Clear inactive request state used by both the runner and mixin.

        This centralizes the model-runner-side cleanup pattern so
        ``OmniGPUModelRunner`` can reuse it instead of open-coding the same
        inactive-request state mutations.
        """
        if hasattr(self, "model_intermediate_buffer"):
            self.model_intermediate_buffer.pop(req_id, None)
        self.drop_inactive_request_delivery_state(req_id)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_connector(model_config: Any) -> OmniConnectorBase | None:
        """Create a connector from model_config, or None if unconfigured."""
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            return None

        if not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", None),
            }

        name = connector_config.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError("Invalid stage connector config: missing connector name")
        name = name.strip()

        extra = connector_config.get("extra")
        if extra is None:
            extra = {}
        elif not isinstance(extra, dict):
            raise RuntimeError(f"Invalid extra config for connector {name}: expected dict, got {type(extra).__name__}")

        spec = ConnectorSpec(name=name, extra=extra)
        try:
            return OmniConnectorFactory.create_connector(spec)
        except Exception as exc:
            raise RuntimeError(f"Failed to create connector {name}") from exc

    @classmethod
    def _load_custom_func(cls, model_config: Any) -> tuple[str | None, Any | None]:
        """Load the connector payload builder for the downstream stage.

        Preferred source is ``custom_process_next_stage_input_func``. Some
        full_payload_mode configs (async_chunk=false) only expose the next-stage prompt builder via
        ``custom_process_input_func`` (for example ``thinker2talker``), while the
        connector payload builder lives beside it as ``thinker2talker_full_payload``.
        In that case, derive the full_payload_mode builder path automatically.
        """
        candidates: list[str] = []

        next_stage_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if isinstance(next_stage_func, str) and next_stage_func:
            candidates.append(next_stage_func)

        if not getattr(model_config, "async_chunk", False):
            input_func = getattr(model_config, "custom_process_input_func", None)
            if isinstance(input_func, str) and input_func:
                try:
                    module_path, func_name = input_func.rsplit(".", 1)
                    if func_name.endswith("_full_payload") or func_name.endswith("_batch"):
                        candidates.append(f"{module_path}.{func_name}")
                    else:
                        candidates.append(f"{module_path}.{func_name}_full_payload")
                        candidates.append(f"{module_path}.{func_name}_batch")
                        candidates.append(input_func)
                except ValueError:
                    candidates.append(input_func)

        tried: set[str] = set()
        for func_path in candidates:
            if func_path in tried:
                continue
            tried.add(func_path)
            try:
                module_path, func_name = func_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                func = getattr(module, func_name, None)
                if callable(func):
                    if not cls._is_connector_payload_builder(func):
                        logger.debug(
                            "Skipping incompatible connector payload hook %s; signature=%s",
                            func_path,
                            inspect.signature(func),
                        )
                        continue
                    return func_path, func
            except Exception:
                logger.warning("Failed to load custom func: %s", func_path, exc_info=True)

        return None, None

    @staticmethod
    def _is_connector_payload_builder(func: Any) -> bool:
        """Whether *func* matches the mixin payload-builder contract."""
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return False

        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return True

        required = {"transfer_manager", "pooling_output", "request"}
        supported = {
            name
            for name, param in params.items()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        return required.issubset(supported)

    def _resolve_external_req_id(self, request: Any, fallback_req_id: str) -> str:
        """Resolve the external request ID consistently.

        Checks ``_request_ids_mapping`` first (populated by
        ``register_chunk_recv``), then falls back to the request's
        ``external_req_id`` attribute, and finally to the given
        ``fallback_req_id``.
        """
        mapped = self._request_ids_mapping.get(fallback_req_id)
        if mapped is not None:
            return mapped
        if request is not None:
            # external_req_id may be explicitly None; fall back.
            ext = getattr(request, "external_req_id", None)
            if ext is not None:
                return ext
        return fallback_req_id

    def _resolve_next_stage_id(self, model_config: Any) -> int:
        """Determine the downstream stage ID from connector config.

        Falls back to ``stage_id + 1`` when the config does not specify
        a ``to_stage`` explicitly.
        """
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is not None:
            if isinstance(connector_config, dict):
                to_stage = connector_config.get("to_stage")
            else:
                to_stage = getattr(connector_config, "to_stage", None)
            if isinstance(to_stage, int):
                return to_stage
            if isinstance(to_stage, str) and to_stage.strip():
                return int(to_stage)
        return self._stage_id + 1

    @staticmethod
    def _parse_rank_mapping(model_config: Any) -> dict[str, int]:
        """Parse rank_mapping from connector config (optional).

        Returns ``{"from_tp": int, "to_tp": int, "local_rank": int}``.
        When ``rank_mapping`` is absent, assumes 1:1 homogeneous mapping.
        """
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is not None and not isinstance(connector_config, dict):
            connector_config = getattr(connector_config, "__dict__", {})

        rank_mapping: dict = {}
        if isinstance(connector_config, dict):
            rank_mapping = connector_config.get("rank_mapping", {})

        from_tp = int(rank_mapping.get("from_tp", 1))
        to_tp = int(rank_mapping.get("to_tp", 1))

        local_rank = 0
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except (ValueError, TypeError):
            pass

        return {"from_tp": from_tp, "to_tp": to_tp, "local_rank": local_rank}

    # ------------------------------------------------------------------ #
    #  KV cache  (delegates to OmniKVTransferManager)
    # ------------------------------------------------------------------ #

    def send_kv_cache(
        self,
        finished_reqs: dict[str, dict[str, Any]],
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
        request_id_resolver: Any | None = None,
    ) -> list[str]:
        """Send KV cache for finished requests.

        Delegates to the existing ``OmniKVTransferManager``.
        """
        if self._kv_transfer_manager is None:
            return list(finished_reqs.keys()) if finished_reqs else []
        result = self._kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished_reqs,
            kv_caches=kv_caches,
            block_size=block_size,
            cache_dtype=cache_dtype,
            request_id_resolver=request_id_resolver,
        )
        if result:
            self._kv_sent_req_ids.extend(result)
        return result

    def recv_kv_cache(
        self,
        request_id: str,
        target_device: torch.device | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for a request.

        Delegates to the existing ``OmniKVTransferManager``.
        """
        if self._kv_transfer_manager is None:
            return None, 0
        return self._kv_transfer_manager.receive_kv_cache_for_request(
            request_id=request_id,
            target_device=target_device,
        )

    def receive_cfg_companion_kv_payloads(
        self,
        cfg_request_ids: dict[str, str],
        target_device: torch.device | None = None,
    ) -> dict[str, tuple[dict[str, Any] | None, int]]:
        """Receive raw CFG companion KV payloads keyed by role."""
        return {
            role: self.recv_kv_cache(companion_rid, target_device=target_device)
            for role, companion_rid in cfg_request_ids.items()
        }

    def receive_multi_kv_cache(
        self,
        req: Any,
        cfg_kv_collect_func: Any | None = None,
        target_device: torch.device | None = None,
    ) -> bool:
        """Receive primary and optional companion KV caches for a request.

        The mixin owns the runner-facing orchestration: primary KV receive,
        companion payload fetch, and applying any model-specific CFG fields back
        onto ``req.sampling_params``.
        """
        if self._kv_transfer_manager is None:
            return False

        request_id = getattr(req, "request_id", None)
        if not request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return False

        active_requests = getattr(self, "requests", None)
        if active_requests is not None and request_id not in active_requests:
            logger.debug("Skip receiving KV cache for inactive request %s", request_id)
            return False

        primary_ok = False
        data, _size = self.recv_kv_cache(request_id, target_device=target_device)
        if data:
            self._kv_transfer_manager.apply_kv_cache_to_request(req, data)
            primary_ok = True

        cfg_ids = getattr(getattr(req, "sampling_params", None), "cfg_kv_request_ids", None)
        if cfg_ids and cfg_kv_collect_func:
            try:
                cfg_role_payloads = self.receive_cfg_companion_kv_payloads(
                    cfg_ids,
                    target_device=target_device,
                )
                cfg_kvs = cfg_kv_collect_func(request_id, cfg_role_payloads)
                if cfg_kvs and hasattr(req, "sampling_params") and req.sampling_params is not None:
                    for key, value in cfg_kvs.items():
                        setattr(req.sampling_params, key, value)
                    logger.debug("Applied CFG KV caches: %s", list(cfg_kvs.keys()))
            except Exception:
                logger.exception("Failed to collect CFG KV caches for %s", request_id)

        return primary_ok

    # ------------------------------------------------------------------ #
    #  Rank-aware KV transfer routing
    # ------------------------------------------------------------------ #

    def get_rank_aware_kv_keys(
        self,
        req_id: str,
        from_stage: int,
        to_stage: int | None = None,
        chunk_id: int = 0,
    ) -> list[str]:
        """Build recv-side connector keys for all remote ranks this rank needs.

        For heterogeneous TP receive, the local rank is the target rank and must
        fetch one or more source-rank shards keyed as ``from_rank -> to_rank``.
        """
        if self._from_tp <= 1 and self._to_tp <= 1:
            resolved_to_stage = self._next_stage_id if to_stage is None else to_stage
            return [f"omni_{from_stage}_to_{resolved_to_stage}_kv_cache_{req_id}"]

        remote_ranks = self.get_kv_remote_ranks()
        return [
            self.get_kv_connector_key(
                req_id=req_id,
                from_stage=from_stage,
                chunk_id=chunk_id,
                from_rank=remote_rank,
                to_rank=self._local_rank,
            )
            for remote_rank in remote_ranks
        ]

    def get_kv_target_ranks_for_send(self) -> list[int]:
        """Determine which target ranks this local rank should send KV shards to."""
        self._validate_kv_tp_topology()
        if self._from_tp == self._to_tp:
            return [self._local_rank]
        if self._from_tp > self._to_tp:
            tp_ratio = self._from_tp // self._to_tp
            return [self._local_rank // tp_ratio]
        tp_ratio = self._to_tp // self._from_tp
        base_rank = self._local_rank * tp_ratio
        return [base_rank + i for i in range(tp_ratio)]

    def get_rank_aware_kv_send_keys(
        self,
        req_id: str,
        from_stage: int,
        to_stage: int | None = None,
        chunk_id: int = 0,
    ) -> list[str]:
        """Build send-side connector keys for this rank's KV shard(s)."""
        if self._from_tp <= 1 and self._to_tp <= 1:
            resolved_to_stage = self._next_stage_id if to_stage is None else to_stage
            return [f"omni_{from_stage}_to_{resolved_to_stage}_kv_cache_{req_id}"]

        target_ranks = self.get_kv_target_ranks_for_send()
        return [
            self.get_kv_connector_key(
                req_id=req_id,
                from_stage=from_stage,
                chunk_id=chunk_id,
                from_rank=self._local_rank,
                to_rank=target_rank,
            )
            for target_rank in target_ranks
        ]

    @staticmethod
    def _merge_rank_sharded_kv_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Merge multiple source-rank KV shards for one target rank."""
        payloads = [payload for payload in payloads if isinstance(payload, dict)]
        if not payloads:
            return None
        if len(payloads) == 1:
            return payloads[0]

        merged = dict(payloads[0])
        layer_blocks = merged.get("layer_blocks")
        if not isinstance(layer_blocks, dict):
            return merged

        def _merge_tensor_lists(name: str) -> list[torch.Tensor | None]:
            merged_list: list[torch.Tensor | None] = []
            cache_lists = [payload.get("layer_blocks", {}).get(name, []) for payload in payloads]
            max_len = max((len(cache_list) for cache_list in cache_lists), default=0)
            for idx in range(max_len):
                tensors = [cache_list[idx] for cache_list in cache_lists if idx < len(cache_list)]
                tensors = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
                if not tensors:
                    merged_list.append(None)
                elif len(tensors) == 1:
                    merged_list.append(tensors[0])
                else:
                    merged_list.append(torch.cat(tensors, dim=-2).contiguous())
            return merged_list

        merged["layer_blocks"] = {
            "key_cache": _merge_tensor_lists("key_cache"),
            "value_cache": _merge_tensor_lists("value_cache"),
        }
        metadata = dict(merged.get("metadata", {}))
        metadata["merged_remote_rank_count"] = len(payloads)
        merged["metadata"] = metadata
        return merged

    def _slice_rank_sharded_kv_payload(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """Slice a duplicated source-rank KV shard for ``from_tp < to_tp`` cases."""
        if payload is None or self._from_tp >= self._to_tp:
            return payload

        tp_ratio = self._to_tp // self._from_tp
        shard_index = self._local_rank % tp_ratio
        layer_blocks = payload.get("layer_blocks") if isinstance(payload, dict) else None
        if not isinstance(layer_blocks, dict):
            return payload

        def _slice_tensor_list(name: str) -> list[torch.Tensor | None]:
            sliced: list[torch.Tensor | None] = []
            for tensor in layer_blocks.get(name, []):
                if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
                    sliced.append(tensor)
                    continue
                head_dim = tensor.shape[-2]
                if head_dim % tp_ratio != 0:
                    sliced.append(tensor)
                    continue
                per_rank = head_dim // tp_ratio
                start = shard_index * per_rank
                sliced.append(tensor.narrow(-2, start, per_rank).contiguous())
            return sliced

        payload = dict(payload)
        payload["layer_blocks"] = {
            "key_cache": _slice_tensor_list("key_cache"),
            "value_cache": _slice_tensor_list("value_cache"),
        }
        metadata = dict(payload.get("metadata", {}))
        metadata["sliced_for_local_rank"] = self._local_rank
        payload["metadata"] = metadata
        return payload

    def should_replicate_payload(self) -> bool:
        """Whether non-KV payloads should be replicated across ranks.

        Data payloads (stage inputs, chunks) are identical after all-gather,
        so only rank 0 transfers them.  KV payloads are rank-specific and
        all ranks participate.
        """
        return self._local_rank != 0

    def get_kv_rank_mapping(self) -> dict[str, Any]:
        """Return the current rank mapping configuration.

        Useful for debugging and for downstream code that needs to know
        the TP topology without re-parsing model config.
        """
        return {
            "from_tp": self._from_tp,
            "to_tp": self._to_tp,
            "local_rank": self._local_rank,
            "remote_ranks": self.get_kv_remote_ranks(),
            "is_data_transfer_rank": self.is_data_transfer_rank(),
        }

    # ------------------------------------------------------------------ #
    #  KV transfer lifecycle (RFC – mixin-owned)
    # ------------------------------------------------------------------ #

    def mark_kv_transfer(
        self,
        req_id: str,
        seq_len: int,
        block_ids: list[int],
        custom_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a request as needing KV cache transfer.

        Called by the scheduler when a transfer trigger fires.  The mixin
        owns the lifecycle from this point: pending → active → completed.
        """
        if req_id in self._kv_pending_transfers:
            return
        self._kv_triggered_requests.add(req_id)
        transfer = {
            "seq_len": seq_len,
            "block_ids": block_ids,
        }
        if custom_metadata is not None:
            transfer["custom_metadata"] = custom_metadata
        self._kv_pending_transfers[req_id] = transfer

    def drain_pending_kv_transfers(self) -> dict[str, dict[str, Any]]:
        """Drain pending KV transfers and move them to active.

        Returns ``{req_id: {seq_len, block_ids}}`` for the model runner
        to submit to ``send_kv_cache``.
        """
        if not self._kv_pending_transfers:
            return {}
        pending = dict(self._kv_pending_transfers)
        self._kv_active_transfers.update(pending.keys())
        self._kv_pending_transfers.clear()
        return pending

    def ack_kv_transfers(self, req_ids: list[str] | set[str]) -> None:
        """Acknowledge completed KV transfers (from kv_extracted_req_ids).

        Moves requests from active to completed so the scheduler can
        safely free their blocks.
        """
        for req_id in req_ids:
            self._kv_active_transfers.discard(req_id)
            self._kv_completed_transfers.add(req_id)

    def drain_completed_kv_transfers(self) -> set[str]:
        """Drain and return completed KV transfer request IDs.

        The scheduler calls this to know which requests' blocks can be freed.
        """
        completed = set(self._kv_completed_transfers)
        self._kv_completed_transfers.clear()
        return completed

    def is_kv_transfer_triggered(self, req_id: str) -> bool:
        """Check if a request has already triggered KV transfer."""
        return req_id in self._kv_triggered_requests

    def has_pending_kv_work(self) -> bool:
        """True if any KV transfers are pending, active, or awaiting ack."""
        return bool(self._kv_pending_transfers or self._kv_active_transfers or self._kv_completed_transfers)

    # ------------------------------------------------------------------ #
    #  Heterogeneous TP rank support
    # ------------------------------------------------------------------ #

    def _validate_kv_tp_topology(self) -> None:
        """Reject heterogeneous TP mappings that cannot be routed losslessly."""
        if self._from_tp <= 0 or self._to_tp <= 0:
            raise ValueError(f"Invalid KV TP mapping: from_tp={self._from_tp}, to_tp={self._to_tp}")
        larger = max(self._from_tp, self._to_tp)
        smaller = min(self._from_tp, self._to_tp)
        if larger % smaller != 0:
            raise ValueError(
                f"KV TP mapping must be divisible for rank-aware routing: from_tp={self._from_tp}, to_tp={self._to_tp}"
            )

    def get_kv_remote_ranks(self) -> list[int]:
        """Determine which remote ranks this local rank exchanges KV with.

        Follows vLLM's ``TpKVTopology.get_target_remote_ranks()`` pattern:
        - ``from_tp > to_tp``: each to-rank reads from multiple from-ranks
        - ``from_tp < to_tp``: multiple to-ranks read from the same from-rank
        - ``from_tp == to_tp``: 1:1 mapping
        """
        self._validate_kv_tp_topology()
        if self._from_tp == self._to_tp:
            return [self._local_rank]

        if self._from_tp > self._to_tp:
            tp_ratio = self._from_tp // self._to_tp
            return [self._local_rank * tp_ratio + i for i in range(tp_ratio)]
        else:
            tp_ratio = self._to_tp // self._from_tp
            return [self._local_rank // tp_ratio]

    def is_data_transfer_rank(self) -> bool:
        """Whether this rank should participate in data (non-KV) transfer.

        Ordinary stage payloads are TP-identical, so exactly one TP rank
        should talk to the connector. When TP is initialized, use TP rank 0
        so the connector leader matches TP-local broadcast source rank.
        Otherwise fall back to LOCAL_RANK==0 for the single-rank case.
        """
        tp_group = self._get_local_tp_group()
        if tp_group is not None and getattr(tp_group, "world_size", 1) > 1:
            return getattr(tp_group, "rank_in_group", 0) == 0
        return self._local_rank == 0

    def get_kv_connector_key(
        self,
        req_id: str,
        from_stage: int,
        chunk_id: int,
        from_rank: int,
        to_rank: int,
    ) -> str:
        """Build connector key that includes rank info for KV transfers."""
        return f"{req_id}_{from_stage}_{chunk_id}_{from_rank}_{to_rank}"
