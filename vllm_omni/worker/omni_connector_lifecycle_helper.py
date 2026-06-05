# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle management helper for OmniConnectorModelRunnerMixin.

Handles initialization, shutdown, cleanup, and background I/O threads.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.connectors.base import (
        OmniConnectorBase,
    )
    from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
        OmniKVTransferManager,
    )

logger = init_logger(__name__)


class OmniConnectorLifecycleHelper:
    """Lifecycle management for connector initialization and cleanup."""

    def __init__(self, owner: Any):
        """Initialize lifecycle helper with a reference to the owner mixin."""
        self._owner = owner

    def init_omni_connectors(
        self,
        vllm_config: Any,
        model_config: Any,
        kv_transfer_manager: OmniKVTransferManager | None = None,
    ) -> None:
        """Initialize connectors and background threads.

        Args:
            vllm_config: Full vLLM config object.
            model_config: Stage-level model config with connector settings.
            kv_transfer_manager: Existing KV transfer manager to delegate to.
        """
        owner = self._owner
        owner._omni_connector: OmniConnectorBase | None = self._create_connector(model_config)
        owner._kv_transfer_manager = kv_transfer_manager

        owner._async_chunk: bool = getattr(model_config, "async_chunk", False)
        owner._model_mode: str = getattr(model_config, "worker_type", "ar")
        stage_id = getattr(model_config, "stage_id", 0)
        if isinstance(stage_id, str):
            stage_id = int(stage_id)
        owner._stage_id: int = stage_id if isinstance(stage_id, int) else 0

        owner._custom_process_func_path, owner._custom_process_func = self._load_custom_func(model_config)
        owner._custom_process_supports_is_finished = owner._custom_process_supports_is_finished_kwarg()
        logger.info(
            "[Stage-%s] init_omni_connectors: async_chunk=%s, custom_process_func=%s, connector=%s, func_path=%s",
            owner._stage_id,
            owner._async_chunk,
            owner._custom_process_func,
            type(owner._omni_connector).__name__ if owner._omni_connector else None,
            owner._custom_process_func_path,
        )

        # -- next stage ID (from connector config or default stage_id + 1) --
        owner._next_stage_id: int = self._resolve_next_stage_id(model_config)

        # -- heterogeneous TP rank support --
        rank_cfg = self._parse_rank_mapping(model_config)
        owner._from_tp: int = rank_cfg["from_tp"]
        owner._to_tp: int = rank_cfg["to_tp"]
        owner._local_rank: int = rank_cfg["local_rank"]
        if owner._kv_transfer_manager is not None:
            owner._kv_transfer_manager.kv_send_key_builder = owner.get_rank_aware_kv_send_keys
            owner._kv_transfer_manager.kv_recv_key_builder = owner.get_rank_aware_kv_keys
            owner._kv_transfer_manager.kv_payload_merger = owner._merge_rank_sharded_kv_payloads
            owner._kv_transfer_manager.kv_payload_slicer = owner._slice_rank_sharded_kv_payload

        # -- chunk index tracking (ported from OmniChunkTransferAdapter) --
        owner._put_req_chunk: dict[str, int] = defaultdict(int)
        owner._get_req_chunk: dict[str, int] = defaultdict(int)
        # Send-side async accumulation / staging buffer. Receive-side payload
        # ownership lives in ``_local_stage_payload_cache``.
        owner._send_side_request_payload: dict[str, dict[str, Any]] = {}
        owner._code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)
        owner._cached_ic: dict[str, int] = {}
        owner._request_ids_mapping: dict[str, str] = {}

        # -- async I/O state (shared by chunk + full_payload_mode) --
        owner._pending_load_reqs: dict[str, Any] = {}
        owner._finished_load_reqs: set[str] = set()
        owner._pending_save_reqs: dict[str, deque] = {}
        owner._pending_save_counts: dict[str, int] = defaultdict(int)
        owner._deferred_send_cleanup: set[str] = set()
        # -- per-cycle output accumulator --
        owner._chunk_ready_req_ids: set[str] = set()
        owner._chunk_finished_req_ids: set[str] = set()
        owner._stage_recv_req_ids: set[str] = set()
        owner._full_payload_pending_broadcast_req_ids: set[str] = set()
        owner._async_chunk_updated_req_ids: set[str] = set()

        # -- Model Runner local payload cache (RFC §2.4) --
        # Full stage payloads land here first on the recv side. We
        # intentionally do not write connector recv results straight into
        # `model_intermediate_buffer`: runner-owned runtime state is
        # materialized later by `_sync_local_stage_payloads()` on the
        # model thread. This keeps recv timing separate from execute-step
        # visibility and avoids mixing connector I/O with model runtime
        # ownership.
        owner._local_stage_payload_cache: dict[str, dict[str, Any]] = {}
        # Lightweight scheduling metadata pending delivery to the Scheduler.
        owner._local_request_metadata: dict[str, dict[str, Any]] = {}

        # -- persistent set of request IDs whose chunk stream is complete --
        # Prevents re-registration after the finish sentinel has been received.
        owner._chunk_stream_completed: set[str] = set()

        # -- full_payload_mode: accumulate latest pooler_output per request,
        #    send only when the request finishes (next-cycle flush) --
        owner._pending_full_payload_send: dict[str, tuple[Any, ...]] = {}

        # -- KV sent accumulator --
        owner._kv_sent_req_ids: list[str] = []

        # -- KV transfer lifecycle (absorbed from scheduler) --
        # Requests marked for KV transfer: {req_id: {seq_len, block_ids}}
        owner._kv_pending_transfers: dict[str, dict[str, Any]] = {}
        # Requests whose KV transfer has been submitted but not yet acked
        owner._kv_active_transfers: set[str] = set()
        # Requests whose KV transfer is complete (acked by kv_extracted_req_ids)
        owner._kv_completed_transfers: set[str] = set()
        # Dedup guard: requests that have already triggered KV transfer
        owner._kv_triggered_requests: set[str] = set()

        owner._lock = threading.Lock()
        owner._stop_event = threading.Event()
        owner._work_available = threading.Event()

        # Start background threads only when there's a connector
        owner._recv_thread: threading.Thread | None = None
        owner._save_thread: threading.Thread | None = None
        if owner._omni_connector is not None:
            owner._recv_thread = threading.Thread(
                target=self._recv_loop,
                daemon=True,
                name="omni-mixin-recv",
            )
            owner._recv_thread.start()
            owner._save_thread = threading.Thread(
                target=self._save_loop,
                daemon=True,
                name="omni-mixin-save",
            )
            owner._save_thread.start()

        # Explicit "fully initialised" marker so other parts of the runner
        # (e.g. _update_states cleanup) can branch on a stable contract
        # instead of probing for private mixin attribute names.  Must be set
        # only after every field above has been bound, so a partially
        # constructed mixin is never observable as initialised.
        owner._omni_connector_initialized = True

    def shutdown_omni_connectors(self) -> None:
        """Stop background threads and release connector resources."""
        owner = self._owner
        owner._stop_event.set()
        if owner._recv_thread is not None:
            owner._recv_thread.join(timeout=5)
        if owner._save_thread is not None:
            owner._save_thread.join(timeout=5)
        if owner._omni_connector is not None:
            try:
                owner._omni_connector.close()
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
        owner = self._owner
        # Force-flush any pending full-payload accumulator entry before
        # cleanup proceeds.  Without this, finished requests with no
        # downstream consumer (e.g. text-only on multi-modal arch) leave
        # the entry orphaned in _pending_full_payload_send across requests,
        # which empirically destabilises subsequent thinker forwards by
        # making prefix-cache reuse observe stale accumulator state.  The
        # flush is idempotent when the entry has already been flushed by the
        # scheduler-driven path, but this cleanup path runs for every request,
        # so skip it entirely when the request never accumulated a payload.
        if req_id in owner._pending_full_payload_send:
            try:
                owner.flush_full_payload_outputs({req_id})
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

        ext_id = owner._request_ids_mapping.pop(req_id, None)
        keys_to_clean: list[str] = [req_id]
        if ext_id is not None and ext_id != req_id:
            keys_to_clean.append(ext_id)

        with owner._lock:
            keys_pending = [k for k in keys_to_clean if owner._pending_save_counts.get(k, 0)]
            for k in keys_pending:
                owner._deferred_send_cleanup.add(k)
            for k in keys_to_clean:
                if k in keys_pending:
                    continue
                owner._put_req_chunk.pop(k, None)
                owner._send_side_request_payload.pop(k, None)
                owner._code_prompt_token_ids.pop(k, None)
                owner._cached_ic.pop(k, None)
            owner._kv_pending_transfers.pop(req_id, None)
            owner._kv_active_transfers.discard(req_id)
            owner._kv_completed_transfers.discard(req_id)
            owner._kv_triggered_requests.discard(req_id)
        self._cleanup_recv_delivery_state(req_id)

    def drop_inactive_request_delivery_state(self, req_id: str) -> None:
        """Clear recv-side state for inactive requests."""
        owner = self._owner
        ext_id = owner._request_ids_mapping.pop(req_id, None)
        if hasattr(owner, "_lock"):
            with owner._lock:
                self._drop_send_side_payload_state(req_id, ext_id)
        else:
            self._drop_send_side_payload_state(req_id, ext_id)
        self._cleanup_recv_delivery_state(req_id)

    def _drop_send_side_payload_state(self, req_id: str, ext_id: str | None) -> None:
        owner = self._owner
        if ext_id is not None:
            owner._send_side_request_payload.pop(ext_id, None)
            owner._cached_ic.pop(ext_id, None)
        owner._send_side_request_payload.pop(req_id, None)
        owner._cached_ic.pop(req_id, None)

    def _cleanup_recv_delivery_state(self, req_id: str) -> None:
        """Clear recv-side delivery-cycle state."""
        owner = self._owner
        if hasattr(owner, "_lock"):
            with owner._lock:
                self._clear_recv_delivery_state(req_id)
        else:
            self._clear_recv_delivery_state(req_id)

    def _clear_recv_delivery_state(self, req_id: str) -> None:
        owner = self._owner
        owner._get_req_chunk.pop(req_id, None)
        owner._pending_load_reqs.pop(req_id, None)
        owner._finished_load_reqs.discard(req_id)
        owner._chunk_ready_req_ids.discard(req_id)
        owner._chunk_finished_req_ids.discard(req_id)
        owner._chunk_stream_completed.discard(req_id)
        owner._stage_recv_req_ids.discard(req_id)
        owner._full_payload_pending_broadcast_req_ids.discard(req_id)
        owner._async_chunk_updated_req_ids.discard(req_id)
        owner._local_stage_payload_cache.pop(req_id, None)
        owner._local_request_metadata.pop(req_id, None)

    def prune_inactive_requests(self, active_req_ids: Any) -> set[str]:
        """Drop connector state for requests that no longer exist locally.

        Preempted / unscheduled requests are expected to stay in
        ``self.requests`` and therefore remain untouched. This only prunes
        stale request IDs that have already fallen out of the active request
        map, preventing background recv/send bookkeeping from outliving the
        request lifecycle.
        """
        owner = self._owner
        if active_req_ids is None:
            return set()

        active_req_ids = set(active_req_ids)
        pending_req_ids = set(getattr(owner, "_pending_load_reqs", {}).keys())
        received_req_ids = set(getattr(owner, "_stage_recv_req_ids", set()))
        received_req_ids.update(getattr(owner, "_full_payload_pending_broadcast_req_ids", set()))
        received_req_ids.update(getattr(owner, "_local_request_metadata", {}).keys())
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
        # Orphaned pending recv entries (e.g. from upstream stage crash)
        # are handled by OmniSchedulingCoordinator.collect_timed_out_request_ids()
        # which detects wait-time violations.  The scheduler then removes the
        # request from its queues, sets FINISHED_ERROR, and calls _free_request()
        # which ultimately triggers cleanup_finished_request() here.
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
            state = getattr(owner, attr_name, None)
            if isinstance(state, dict):
                stale_req_ids.update(req_id for req_id in state if req_id not in active_req_ids)
            elif isinstance(state, set):
                stale_req_ids.update(req_id for req_id in state if req_id not in active_req_ids)

        for req_id in stale_req_ids:
            self.cleanup_finished_request(req_id)

        return stale_req_ids

    # ------------------------------------------------------------------ #
    #  Background I/O threads
    # ------------------------------------------------------------------ #

    def _recv_loop(self) -> None:
        """Background thread: poll connector for incoming data."""
        owner = self._owner
        _recv_poll_count = 0
        while not owner._stop_event.is_set():
            with owner._lock:
                pending_ids = list(owner._pending_load_reqs.keys())

            if not pending_ids:
                owner._work_available.wait(timeout=0.01)
                owner._work_available.clear()
                continue

            _recv_poll_count += 1
            if _recv_poll_count % 5000 == 1:
                logger.info(
                    "[Stage-%s] _recv_loop: polling %s pending reqs: %s (poll#%s)",
                    owner._stage_id,
                    len(pending_ids),
                    pending_ids[:5],
                    _recv_poll_count,
                )

            made_progress = False
            for req_id in pending_ids:
                if owner._stop_event.is_set():
                    break
                try:
                    made_progress = owner._poll_single_request(req_id) or made_progress
                except Exception:
                    logger.warning("Error receiving data for %s", req_id, exc_info=True)

            if not made_progress and not owner._stop_event.is_set():
                owner._work_available.wait(timeout=0.005)
                owner._work_available.clear()

    _MAX_SEND_RETRIES = 3

    def _save_loop(self) -> None:
        """Background thread: send outgoing data via connector."""
        owner = self._owner
        while not owner._stop_event.is_set():
            task = None
            with owner._lock:
                for req_id in list(owner._pending_save_reqs.keys()):
                    dq = owner._pending_save_reqs[req_id]
                    if dq:
                        task = dq.popleft()
                        if not dq:
                            del owner._pending_save_reqs[req_id]
                        break
                    del owner._pending_save_reqs[req_id]

            if task is not None:
                success = False
                try:
                    success = owner._send_single_request(task)
                except Exception:
                    logger.error(
                        "Error saving data for %s",
                        task.get("request_id"),
                        exc_info=True,
                    )
                if not success:
                    self._requeue_or_drop_failed_send(task)
                continue

            owner._work_available.wait(timeout=0.01)
            owner._work_available.clear()

    def _requeue_or_drop_failed_send(self, task: dict) -> None:
        """Re-enqueue a failed send task or drop it after max retries."""
        owner = self._owner
        retry_count = task.get("_retry_count", 0) + 1
        req_id = task.get("request_id")
        if retry_count <= self._MAX_SEND_RETRIES:
            task["_retry_count"] = retry_count
            logger.warning(
                "[Stage-%s] Re-enqueuing failed send for %s (retry %d/%d)",
                getattr(owner, "_stage_id", "?"),
                req_id,
                retry_count,
                self._MAX_SEND_RETRIES,
            )
            with owner._lock:
                dq = owner._pending_save_reqs.setdefault(req_id, deque())
                dq.appendleft(task)
        else:
            logger.error(
                "[Stage-%s] Giving up on send for %s after %d retries",
                getattr(owner, "_stage_id", "?"),
                req_id,
                self._MAX_SEND_RETRIES,
            )
            self._decrement_pending_save_count(req_id)

    def _decrement_pending_save_count(self, request_id: str) -> None:
        """Decrement pending save count and run deferred cleanup if zero."""
        owner = self._owner
        cleanup_req_id = None
        with owner._lock:
            remaining = owner._pending_save_counts.get(request_id, 0)
            if remaining > 1:
                owner._pending_save_counts[request_id] = remaining - 1
            elif remaining == 1:
                owner._pending_save_counts.pop(request_id, None)
                if request_id in owner._deferred_send_cleanup:
                    owner._deferred_send_cleanup.remove(request_id)
                    cleanup_req_id = request_id
            if cleanup_req_id is not None:
                owner._put_req_chunk.pop(cleanup_req_id, None)
                owner._send_side_request_payload.pop(cleanup_req_id, None)
                owner._code_prompt_token_ids.pop(cleanup_req_id, None)
                owner._cached_ic.pop(cleanup_req_id, None)

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

    @staticmethod
    def _load_custom_func(model_config: Any) -> tuple[str | None, Any | None]:
        """Load the connector payload builder for the downstream stage.

        Preferred source is ``custom_process_next_stage_input_func``. Some
        full_payload_mode configs (async_chunk=false) only expose the next-stage prompt builder via
        ``custom_process_input_func`` (for example ``thinker2talker``), while the
        connector payload builder lives beside it as ``thinker2talker_full_payload``.
        In that case, derive the full_payload_mode builder path automatically.
        """
        import importlib
        import inspect

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
                    if not OmniConnectorLifecycleHelper._is_connector_payload_builder(func):
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
        import inspect

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

    def _resolve_next_stage_id(self, model_config: Any) -> int:
        """Determine the downstream stage ID from connector config.

        Falls back to ``stage_id + 1`` when the config does not specify
        a ``to_stage`` explicitly.
        """
        owner = self._owner
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
        return owner._stage_id + 1

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
