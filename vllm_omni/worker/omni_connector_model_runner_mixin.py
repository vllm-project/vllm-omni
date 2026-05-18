# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified data-plane communication mixin for Model Runners.

All connector.put()/get() calls are consolidated here. Background I/O
threads handle async_chunk and full_payload_mode transfers; KV cache is delegated to
the existing OmniKVTransferManager (to be absorbed later).

The mixin reports transfer results via OmniConnectorOutput so that the
Scheduler can make scheduling decisions without ever touching a connector.
"""

from __future__ import annotations

import importlib
import inspect
import os
import threading
from collections import defaultdict, deque
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
from vllm_omni.outputs import OmniConnectorOutput
from vllm_omni.worker.payload_span import (
    get_tensor_span,
    merge_tensor_spans,
)

_EMBED_SPAN_GROUPS: tuple[tuple[str, str, str], ...] = (("decode", "decode_token_start", "decode_token_end"),)

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.connectors.base import (
        OmniConnectorBase,
    )
    from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
        OmniKVTransferManager,
    )

logger = init_logger(__name__)


class OmniConnectorModelRunnerMixin:
    """Unified data-plane communication mixin for Model Runners.

    Provides three transfer modes through a single pair of bg I/O threads:
      - **full_payload_mode**: ``recv_full_payload_inputs`` / ``send_full_payload_outputs``
      - **Streaming (async_chunk)**: ``recv_chunk`` / ``send_chunk``
      - **KV cache**: ``send_kv_cache`` / ``recv_kv_cache`` (delegates to
        the existing ``OmniKVTransferManager``)

    The mixin owns connector instances and background threads.  It never
    touches scheduling queues -- readiness is communicated to the Scheduler
    via ``OmniConnectorOutput``.
    """

    # ------------------------------------------------------------------ #
    #  Init / Shutdown
    # ------------------------------------------------------------------ #

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
        self._omni_connector: OmniConnectorBase | None = self._create_connector(model_config)
        self._kv_transfer_manager = kv_transfer_manager

        self._async_chunk: bool = getattr(model_config, "async_chunk", False)
        self._model_mode: str = getattr(model_config, "worker_type", "ar")
        stage_id = getattr(model_config, "stage_id", 0)
        if isinstance(stage_id, str):
            stage_id = int(stage_id)
        self._stage_id: int = stage_id if isinstance(stage_id, int) else 0

        self._custom_process_func_path, self._custom_process_func = self._load_custom_func(model_config)
        self._custom_process_supports_is_finished = self._custom_process_supports_is_finished_kwarg()
        logger.info(
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
            state = getattr(self, attr_name, None)
            if isinstance(state, dict):
                stale_req_ids.update(req_id for req_id in state if req_id not in active_req_ids)
            elif isinstance(state, set):
                stale_req_ids.update(req_id for req_id in state if req_id not in active_req_ids)

        for req_id in stale_req_ids:
            self.cleanup_finished_request(req_id)

        return stale_req_ids

    # ------------------------------------------------------------------ #
    #  Local payload cache (RFC §2.4 – Model Runner ownership)
    # ------------------------------------------------------------------ #

    def put_local_stage_payload(self, req_id: str, payload: OmniPayload) -> None:
        """Store a full stage payload in the local cache."""
        self._local_stage_payload_cache[req_id] = payload

    def get_local_stage_payload(self, req_id: str) -> OmniPayload | None:
        """Read a stage payload without removing it."""
        return self._local_stage_payload_cache.get(req_id)

    def pop_local_stage_payload(self, req_id: str) -> OmniPayload | None:
        """Remove and return a stage payload (consume after use)."""
        return self._local_stage_payload_cache.pop(req_id, None)

    def put_local_request_metadata(self, req_id: str, metadata: dict[str, Any]) -> None:
        """Store lightweight scheduling metadata for a request."""
        self._local_request_metadata[req_id] = metadata

    def get_local_request_metadata(self, req_id: str) -> dict[str, Any] | None:
        """Retrieve scheduling metadata for a request."""
        return self._local_request_metadata.get(req_id)

    # ------------------------------------------------------------------ #
    #  Scheduling metadata extraction
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_scheduling_metadata(cls, payload: OmniPayload) -> dict[str, Any]:
        """Extract only the fields the scheduler needs from a full payload."""
        extracted: dict[str, Any] = {}
        meta = payload.get("meta") if isinstance(payload, dict) else None
        meta = meta if isinstance(meta, dict) else {}

        if "next_stage_prompt_len" in meta:
            extracted["next_stage_prompt_len"] = meta["next_stage_prompt_len"]
        elif "next_stage_prompt_len" in payload:
            logger.warning_once(
                "legacy flat 'next_stage_prompt_len' key in payload; expected 'meta.next_stage_prompt_len'"
            )
            extracted["next_stage_prompt_len"] = payload["next_stage_prompt_len"]

        audio_codes = cls._payload_audio_codes(payload)
        if audio_codes is not None:
            extracted["code_predictor_codes"] = audio_codes

        if "left_context_size" in meta:
            extracted["left_context_size"] = meta["left_context_size"]
        elif "left_context_size" in payload:
            logger.warning_once("legacy flat 'left_context_size' key in payload; expected 'meta.left_context_size'")

        return extracted

    _NON_CONSUMABLE_PAYLOAD_KEYS: set[tuple[str, str]] = {
        ("meta", "finished"),
        ("meta", "override_keys"),
        ("meta", "next_stage_prompt_len"),
        ("meta", "left_context_size"),
        ("ids", "output"),
        ("embed", "decode_token_start"),
        ("embed", "decode_token_end"),
    }

    @staticmethod
    def _payload_value_has_content(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, torch.Tensor):
            return value.numel() > 0
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) > 0
        return True

    @staticmethod
    def _payload_finished(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if "finished" in payload:
            logger.warning_once("legacy flat 'finished' key in payload; expected 'meta.finished'")
        meta = payload.get("meta")
        if not isinstance(meta, dict) or "finished" not in meta:
            return False
        flag = meta["finished"]
        if isinstance(flag, torch.Tensor):
            return flag.numel() == 1 and bool(flag.item())
        return bool(flag)

    @staticmethod
    def _payload_audio_codes(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return None
        if "code_predictor_codes" in payload:
            logger.warning_once("legacy flat 'code_predictor_codes' key in payload; expected 'codes.audio'")
        codes = payload.get("codes")
        if isinstance(codes, dict):
            return codes.get("audio")
        return None

    @classmethod
    def _payload_is_consumable(cls, payload: OmniPayload | None) -> bool:
        """Return True when an async payload can drive a real forward step.

        Metadata-only wake-ups should not transition WAITING_FOR_CHUNK requests
        back to schedulable state. In particular, a widened token horizon without
        any newly visible thinker decode embeds should not force a placeholder-only
        talker decode step.
        """
        if not isinstance(payload, dict) or not payload:
            return False

        embed = payload.get("embed")
        if isinstance(embed, dict):
            decode_embeddings = embed.get("decode")
            if isinstance(decode_embeddings, torch.Tensor):
                if decode_embeddings.ndim == 0:
                    return True
                return decode_embeddings.numel() > 0 and decode_embeddings.shape[0] > 0

        audio_codes = cls._payload_audio_codes(payload)
        if audio_codes is not None:
            if isinstance(audio_codes, torch.Tensor):
                return audio_codes.numel() > 0
            if hasattr(audio_codes, "__len__"):
                return len(audio_codes) > 0
            return True

        for key, value in payload.items():
            if isinstance(value, dict):
                for sk, sv in value.items():
                    if (key, sk) in cls._NON_CONSUMABLE_PAYLOAD_KEYS:
                        continue
                    if cls._payload_value_has_content(sv):
                        return True
                continue
            if cls._payload_value_has_content(value):
                return True
        return False

    @staticmethod
    def _get_local_tp_group() -> Any | None:
        """Return the local TP group when tensor parallelism is initialized."""
        try:
            return get_tp_group()
        except Exception:
            return None

    def _recv_ordinary_stage_result(
        self,
        connector: OmniConnectorBase,
        from_stage: str,
        to_stage: str,
        connector_get_key: str,
    ) -> Any:
        """Receive one ordinary non-KV stage payload on the local leader rank only."""
        tp_group = self._get_local_tp_group()
        if tp_group is None or getattr(tp_group, "world_size", 1) <= 1:
            return connector.get(from_stage, to_stage, connector_get_key)
        if not self.is_data_transfer_rank():
            return None
        return connector.get(from_stage, to_stage, connector_get_key)

    def _recv_full_payload_result(
        self,
        connector: OmniConnectorBase,
        from_stage: str,
        to_stage: str,
        connector_get_key: str,
    ) -> Any:
        """Receive one full-payload transfer on the local leader rank only."""
        return self._recv_ordinary_stage_result(
            connector,
            from_stage,
            to_stage,
            connector_get_key,
        )

    def _recv_async_chunk_result(
        self,
        connector: OmniConnectorBase,
        from_stage: str,
        to_stage: str,
        connector_get_key: str,
    ) -> Any:
        """Receive one ordinary async chunk on the local leader rank only."""
        return self._recv_ordinary_stage_result(
            connector,
            from_stage,
            to_stage,
            connector_get_key,
        )

    @staticmethod
    def _snapshot_payload(payload: Any) -> Any:
        if isinstance(payload, dict):
            return dict(payload)
        return payload

    def _broadcast_tp_payload_packet(self, packet: Any) -> Any:
        """Broadcast one ordinary payload packet from TP rank 0 when TP is active."""
        tp_group = self._get_local_tp_group()
        if tp_group is None or getattr(tp_group, "world_size", 1) <= 1:
            return packet
        leader_packet = packet if self.is_data_transfer_rank() else None
        return tp_group.broadcast_object(leader_packet, src=0)

    def _apply_staged_payloads_locked(self, staged_payloads: dict[str, Any]) -> None:
        for req_id, payload in staged_payloads.items():
            self._local_stage_payload_cache[req_id] = self._snapshot_payload(payload)

    def _collect_full_payload_results_locked(self) -> dict[str, Any] | None:
        if not self._full_payload_pending_broadcast_req_ids:
            return None
        results: dict[str, Any] = {}
        missing_req_ids: list[str] = []
        for req_id in tuple(self._full_payload_pending_broadcast_req_ids):
            payload = self._local_stage_payload_cache.get(req_id)
            if payload is None:
                missing_req_ids.append(req_id)
                continue
            results[req_id] = self._snapshot_payload(payload)
            self._full_payload_pending_broadcast_req_ids.discard(req_id)
        if missing_req_ids:
            logger.warning(
                "[Stage-%s] _collect_full_payload_results_locked: "
                "pending full-payload reqs missing from local cache: %s",
                self._stage_id,
                missing_req_ids,
            )
        return results or None

    def _collect_async_chunk_fanout_packet_locked(self) -> dict[str, Any] | None:
        payload_req_ids = set(self._async_chunk_updated_req_ids)
        payload_req_ids.update(self._finished_load_reqs)
        payload_req_ids.update(self._chunk_finished_req_ids)
        payload_req_ids.update(self._local_request_metadata)
        if not (
            payload_req_ids or self._finished_load_reqs or self._chunk_finished_req_ids or self._local_request_metadata
        ):
            return None

        staged_payloads = {
            req_id: self._snapshot_payload(self._local_stage_payload_cache[req_id])
            for req_id in payload_req_ids
            if req_id in self._local_stage_payload_cache
        }
        packet = {
            "staged_payloads": staged_payloads,
            "request_metadata": dict(self._local_request_metadata),
            "newly_finished": set(self._finished_load_reqs),
            "chunk_finished": set(self._chunk_finished_req_ids),
        }

        self._async_chunk_updated_req_ids.clear()
        self._finished_load_reqs.clear()
        self._chunk_finished_req_ids.clear()
        self._local_request_metadata.clear()

        for req_id in packet["chunk_finished"]:
            if req_id not in self._local_stage_payload_cache:
                continue
            ext_req_id = self._request_ids_mapping.get(req_id, req_id)
            self._send_side_request_payload.pop(ext_req_id, None)
            if ext_req_id != req_id:
                self._send_side_request_payload.pop(req_id, None)

        return packet

    def _apply_async_chunk_fanout_packet(self, packet: dict[str, Any]) -> None:
        staged_payloads = packet.get("staged_payloads", {})
        chunk_finished = set(packet.get("chunk_finished", ()))
        with self._lock:
            self._apply_staged_payloads_locked(staged_payloads)
            for req_id in chunk_finished:
                self._pending_load_reqs.pop(req_id, None)
                self._chunk_stream_completed.add(req_id)

    # ------------------------------------------------------------------ #
    #  full_payload_mode (recv_full_payload_inputs / send_full_payload_outputs)
    # ------------------------------------------------------------------ #

    def recv_full_payload_inputs(self, scheduler_output: Any) -> dict[str, Any] | None:
        """Check for incoming full_payload_mode stage inputs (non-blocking).

        Returns a dict mapping ``request_id -> engine_inputs`` for data
        that has arrived, or ``None`` if nothing is ready.  Stores full
        payloads in the local cache and extracts scheduling metadata.
        """
        # Fast path: when TP is trivial (no peer ranks waiting on a broadcast)
        # and the bg recv thread has not staged anything, skip the lock + TP
        # broadcast cycle entirely. _broadcast_tp_payload_packet already
        # returns its input unchanged under the same world_size<=1 condition,
        # so the original code path was a no-op here on every empty step.
        tp_group = self._get_local_tp_group()
        if (
            tp_group is None or getattr(tp_group, "world_size", 1) <= 1
        ) and not self._full_payload_pending_broadcast_req_ids:
            return None
        with self._lock:
            results = self._collect_full_payload_results_locked() if self.is_data_transfer_rank() else None
        results = self._broadcast_tp_payload_packet(results)
        if not results:
            return None
        with self._lock:
            self._stage_recv_req_ids.update(results.keys())
            for req_id in results:
                self._pending_load_reqs.pop(req_id, None)
            self._apply_staged_payloads_locked(results)
            for req_id, payload in results.items():
                self._local_request_metadata[req_id] = self._extract_scheduling_metadata(payload)
        logger.info(
            "[Stage-%s] recv_full_payload_inputs: consumed %s reqs: %s, stage_recv_req_ids now=%s",
            self._stage_id,
            len(results),
            list(results.keys()),
            self._stage_recv_req_ids,
        )
        return results

    def _get_model_config(self) -> Any:
        model_config = getattr(self, "model_config", None)
        if model_config is not None:
            return model_config
        return getattr(getattr(self, "vllm_config", None), "model_config", None)

    def _should_accumulate_full_payload_output(self) -> bool:
        """Gate send-side full-payload output accumulation only.

        Cached per instance: the result depends only on model_config /
        _custom_process_func, both of which are set at init time. Avoid
        the per-step dynamic import inside the model decode loop.
        """
        cached = getattr(self, "_should_accumulate_full_payload_output_cached", None)
        if cached is not None:
            return cached
        model_config = self._get_model_config()
        if model_config is None:
            self._should_accumulate_full_payload_output_cached = False
            return False
        if getattr(model_config, "model_arch", None) == "Qwen3OmniMoeForConditionalGeneration":
            from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
                should_accumulate_qwen3_omni_full_payload_output,
            )

            result = should_accumulate_qwen3_omni_full_payload_output(
                model_config,
                getattr(self, "_custom_process_func", None),
            )
            self._should_accumulate_full_payload_output_cached = result
            return result
        self._should_accumulate_full_payload_output_cached = False
        return False

    @staticmethod
    def _new_full_payload_accumulator(output: dict[str, Any]):
        chunks: dict[str, list[torch.Tensor]] = {}
        latest: dict[str, Any] = {}
        rows: dict[str, int] = {}
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                chunks[k] = [v]
                rows[k] = int(v.shape[0])
            else:
                latest[k] = v
        return chunks, latest, rows

    @staticmethod
    def _materialize_full_payload_entry(entry):
        if len(entry) == 2:
            return entry
        chunks, latest, _rows, request = entry
        output = dict(latest)
        for k, tensors in chunks.items():
            if tensors:
                output[k] = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        return output, request

    def accumulate_full_payload_output(
        self,
        req_id: str,
        pooler_output: Any,
        request: Any,
    ) -> None:
        """Accumulate pooler_output for a request across steps (full_payload_mode).

        Per-token tensors (2-D+, matching trailing dims) are concatenated
        along dim-0.  Scalar / global tensors (1-D or 0-D) are replaced
        with the latest value.

        Note: codec rows are NOT filtered for zero placeholders here. The
        downstream consumer ``_extract_qwen3_full_payload_codec_rows`` crops
        codec rows using ``output_token_ids`` as the authoritative source,
        which makes any sender-side zero filtering redundant. Skipping the
        sender-side ``t.any()`` scan also avoids a per-tensor GPU->CPU device
        sync that stalled the decode pipeline.

        The data is actually sent when ``flush_full_payload_outputs`` is called
        with the finished request IDs from the next scheduler cycle.
        """
        existing = self._pending_full_payload_send.get(req_id)

        if existing is None:
            chunks, latest, rows = self._new_full_payload_accumulator(pooler_output)
            self._pending_full_payload_send[req_id] = (chunks, latest, rows, request)
            return

        if len(existing) == 2:
            chunks, latest, rows = self._new_full_payload_accumulator(existing[0])
        else:
            chunks, latest, rows, _ = existing

        for k, v in pooler_output.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                if k in chunks and chunks[k] and v.shape[1:] == chunks[k][0].shape[1:]:
                    chunks[k].append(v)
                    rows[k] += int(v.shape[0])
                else:
                    latest.pop(k, None)
                    chunks[k] = [v]
                    rows[k] = int(v.shape[0])
            else:
                chunks.pop(k, None)
                rows.pop(k, None)
                latest[k] = v

        self._pending_full_payload_send[req_id] = (chunks, latest, rows, request)

    def flush_full_payload_outputs(self, finished_req_ids: set[str]) -> None:
        """Send accumulated full_payload outputs for requests that just finished."""
        logger.info(
            "[Stage-%s] flush_full_payload_outputs: finished_req_ids=%s, pending=%s",
            self._stage_id,
            finished_req_ids,
            list(self._pending_full_payload_send.keys()),
        )
        to_send: dict[str, tuple[Any, Any]] = {}
        for req_id in finished_req_ids:
            entry = self._pending_full_payload_send.pop(req_id, None)
            if entry is not None:
                to_send[req_id] = self._materialize_full_payload_entry(entry)
        logger.info("[Stage-%s] flush_full_payload_outputs: to_send=%s", self._stage_id, list(to_send.keys()))
        if to_send:
            self.send_full_payload_outputs(scheduler_output=None, outputs=to_send)

    def send_full_payload_outputs(
        self,
        scheduler_output: Any,
        outputs: dict[str, tuple[Any, Any] | Any],
    ) -> list[str]:
        """Send full_payload stage outputs to the next stage via connector.

        Args:
            outputs: Mapping of ``req_id`` to either a
                ``(pooling_output, request)`` tuple (preferred) or a raw
                payload dict.  When a tuple is supplied the request object
                is forwarded to ``custom_process_stage_input_func``.

        Returns list of request IDs successfully enqueued.
        """
        if self._omni_connector is None:
            logger.info("[Stage-%s] send_full_payload_outputs: connector is None, skip", self._stage_id)
            return []
        if not self.is_data_transfer_rank():
            logger.info(
                "[Stage-%s] send_full_payload_outputs: not data_transfer_rank (rank=%s), skip",
                self._stage_id,
                self._local_rank,
            )
            return list(outputs.keys())
        sent_ids: list[str] = []
        next_stage_id = self._next_stage_id
        for req_id, value in outputs.items():
            if isinstance(value, tuple) and len(value) == 2:
                raw_output, request = value
            else:
                raw_output, request = value, None

            payload = raw_output
            if self._custom_process_func is not None:
                payload = self._build_custom_process_payload(
                    request_id=req_id,
                    request=request,
                    pooling_output=raw_output,
                )
                if payload is None:
                    continue
            if payload is None:
                logger.info("[Stage-%s] send_full_payload_outputs: payload is None for %s", self._stage_id, req_id)
                continue
            if isinstance(payload, dict):
                audio_codes = self._payload_audio_codes(payload)
                if isinstance(audio_codes, torch.Tensor):
                    code_len = int(audio_codes.numel())
                elif hasattr(audio_codes, "__len__"):
                    code_len = len(audio_codes)
                else:
                    code_len = None
                meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                logger.info(
                    "[Stage-%s] send_full_payload_outputs: req=%s payload_keys=%s code_len=%s left_context_size=%s",
                    self._stage_id,
                    req_id,
                    sorted(payload.keys()),
                    code_len,
                    meta.get("left_context_size"),
                )

            external_req_id = self._resolve_external_req_id(request, req_id)
            chunk_id = self._put_req_chunk[req_id]
            self._put_req_chunk[req_id] += 1
            connector_put_key = f"{external_req_id}_{self._stage_id}_{chunk_id}"

            logger.info(
                "[Stage-%s] send_full_payload_outputs: enqueue req=%s put_key=%s next_stage=%s",
                self._stage_id,
                req_id,
                connector_put_key,
                next_stage_id,
            )
            task = {
                "stage_id": self._stage_id,
                "next_stage_id": next_stage_id,
                "put_key": connector_put_key,
                "data": payload,
                "request_id": req_id,
            }
            with self._lock:
                self._pending_save_reqs.setdefault(req_id, deque()).append(task)
                self._pending_save_counts[req_id] += 1
            sent_ids.append(req_id)
        if sent_ids:
            self._work_available.set()
        return sent_ids

    def recv_stage_inputs(self, scheduler_output: Any) -> dict[str, Any] | None:
        """Compatibility wrapper for ``recv_full_payload_inputs``."""
        return self.recv_full_payload_inputs(scheduler_output)

    def accumulate_batch_output(
        self,
        req_id: str,
        pooler_output: Any,
        request: Any,
    ) -> None:
        """Compatibility wrapper for ``accumulate_full_payload_output``."""
        self.accumulate_full_payload_output(req_id, pooler_output, request)

    def flush_batch_outputs(self, finished_req_ids: set[str]) -> None:
        """Compatibility wrapper for ``flush_full_payload_outputs``."""
        self.flush_full_payload_outputs(finished_req_ids)

    def send_stage_outputs(
        self,
        scheduler_output: Any,
        outputs: dict[str, tuple[Any, Any] | Any],
    ) -> list[str]:
        """Compatibility wrapper for ``send_full_payload_outputs``."""
        return self.send_full_payload_outputs(scheduler_output, outputs)

    # ------------------------------------------------------------------ #
    #  Streaming chunk mode  (recv_chunk / send_chunk)
    # ------------------------------------------------------------------ #

    def register_chunk_recv(self, request: Any) -> None:
        """Register a request for async chunk retrieval by the bg thread.

        Stage-0 has no upstream producer so this is a no-op there.
        Skips requests whose batch data has already been received to
        prevent the bg thread from polling for non-existent chunks.
        """
        if self._stage_id == 0:
            return
        request_id = request.request_id
        self._request_ids_mapping[request_id] = getattr(
            request,
            "external_req_id",
            request_id,
        )
        with self._lock:
            if request_id in self._stage_recv_req_ids:
                return
            # Don't re-register if the finish sentinel was already received
            if request_id in self._chunk_stream_completed:
                return
            self._pending_load_reqs[request_id] = request
        self._work_available.set()

    def recv_chunk(self) -> dict[str, Any]:
        """Collect chunks received by the bg thread since last call.

        Returns a dict ``{request_id: chunk_payload}`` for newly arrived
        chunks.  Empty dict when nothing is ready.

        This method reads from ``_finished_load_reqs`` without clearing
        it -- ``get_omni_connector_output()`` is the sole consumer that
        drains and resets ``_finished_load_reqs`` at the end of each
        ``execute_model`` cycle.

        Returns **shallow copies** of the cached payloads so that the
        caller can read them without racing against the background recv
        thread, which may concurrently mutate the live cache entries via
        ``dict.update()``.
        """
        with self._lock:
            finished = set(self._finished_load_reqs)
            if not finished:
                return {}
            # Snapshot the payloads under the lock to avoid racing with
            # _poll_single_request which does existing.update(payload_data)
            # on the same dict objects.
            result = {}
            for rid in finished:
                payload = self._local_stage_payload_cache.get(rid)
                result[rid] = dict(payload) if isinstance(payload, dict) else payload

        self._chunk_ready_req_ids.update(finished)
        return result

    def send_chunk(
        self,
        request: Any,
        pooling_output: Any | None = None,
    ) -> bool:
        """Derive and enqueue one chunk for async sending.

        Payload extraction runs in the caller thread (via
        ``custom_process_stage_input_func``); the actual
        ``connector.put()`` is done by the background save thread.
        Non-KV data is identical across TP ranks; only rank 0 sends.
        """
        if self._omni_connector is None:
            logger.warning("[Stage-%s] send_chunk: connector is None", self._stage_id)
            return False
        if not self.is_data_transfer_rank():
            return True
        raw_req_id = getattr(request, "request_id", None) or getattr(request, "req_id", None)
        request_id = self._resolve_external_req_id(request, raw_req_id)
        # Cache the internal→external mapping so that finish sentinels can
        # resolve the external ID even after the request is freed.
        if raw_req_id and raw_req_id != request_id:
            self._request_ids_mapping.setdefault(raw_req_id, request_id)
        chunk_id = self._put_req_chunk[request_id]

        payload_data = self._build_custom_process_payload(
            request_id=request_id,
            request=request,
            pooling_output=pooling_output,
        )
        if payload_data is None:
            if chunk_id == 0:
                logger.warning(
                    "[Stage-%s] send_chunk: payload is None for req=%s chunk=%s (process_func=%s)",
                    self._stage_id,
                    request_id,
                    chunk_id,
                    self._custom_process_func,
                )
            return False

        self._put_req_chunk[request_id] += 1
        next_stage_id = self._next_stage_id
        connector_put_key = f"{request_id}_{self._stage_id}_{chunk_id}"

        if chunk_id == 0:
            logger.info(
                "[Stage-%s] send_chunk: first chunk enqueued, req=%s key=%s",
                self._stage_id,
                request_id,
                connector_put_key,
            )

        task = {
            "stage_id": self._stage_id,
            "next_stage_id": next_stage_id,
            "put_key": connector_put_key,
            "data": payload_data,
            "request_id": request_id,
        }
        with self._lock:
            self._pending_save_reqs.setdefault(request_id, deque()).append(task)
            self._pending_save_counts[request_id] += 1
        self._work_available.set()
        return True

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

        request_id = getattr(req, "request_id", None) or (
            req.request_ids[0] if hasattr(req, "request_ids") and req.request_ids else None
        )
        if not request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return False

        active_requests = getattr(self, "requests", None)
        if active_requests is not None and request_id not in active_requests:
            logger.info("Skip receiving KV cache for inactive request %s", request_id)
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
                    logger.info("Applied CFG KV caches: %s", list(cfg_kvs.keys()))
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

    #  Output aggregation
    # ------------------------------------------------------------------ #

    def _empty_output_with_connector_signals(self) -> Any:
        """Return a minimal ModelRunnerOutput carrying pending connector signals.

        Used by early-return paths (e.g. ``num_scheduled_tokens == 0``)
        that still need to deliver ``omni_connector_output`` to the
        Scheduler so that WAITING_FOR_INPUT / WAITING_FOR_CHUNK
        transitions are not lost.
        """
        from vllm_omni.outputs import OmniModelRunnerOutput

        output = OmniModelRunnerOutput(req_ids=[], req_id_to_index={})
        output.omni_connector_output = self.get_omni_connector_output()
        return output

    def get_omni_connector_output(self) -> OmniConnectorOutput:
        """Collect and reset transfer results for this execute_model cycle.

        ``request_metadata`` carries only lightweight scheduling metadata.
        Full payloads remain owned by the Model Runner local cache for all
        paths.
        """
        if not hasattr(self, "_lock"):
            return OmniConnectorOutput()

        tp_group = self._get_local_tp_group()
        if self._async_chunk and tp_group is not None and getattr(tp_group, "world_size", 1) > 1:
            if self.is_data_transfer_rank():
                with self._lock:
                    fanout_packet = self._collect_async_chunk_fanout_packet_locked()
            else:
                fanout_packet = None
            fanout_packet = self._broadcast_tp_payload_packet(fanout_packet)
            if fanout_packet is None:
                newly_finished = set()
                chunk_finished = set()
                request_metadata = {}
            else:
                if not self.is_data_transfer_rank():
                    self._apply_async_chunk_fanout_packet(fanout_packet)
                newly_finished = set(fanout_packet["newly_finished"])
                chunk_finished = set(fanout_packet["chunk_finished"])
                request_metadata = dict(fanout_packet["request_metadata"])
        else:
            with self._lock:
                newly_finished = set(self._finished_load_reqs)
                self._finished_load_reqs.clear()
                chunk_finished = set(self._chunk_finished_req_ids)
                self._chunk_finished_req_ids.clear()
                request_metadata = dict(self._local_request_metadata)
                self._local_request_metadata.clear()
                # _send_side_request_payload is the async accumulation buffer for
                # future recv chunks. Clearing it on every consumable wake-up drops
                # intermediate
                # thinker decode spans before the model side can consume them.
                # Only terminal chunk_finished requests may release that buffer.
                for req_id in chunk_finished:
                    if req_id not in self._local_stage_payload_cache:
                        continue
                    ext_req_id = self._request_ids_mapping.get(req_id, req_id)
                    self._send_side_request_payload.pop(ext_req_id, None)
                    if ext_req_id != req_id:
                        self._send_side_request_payload.pop(req_id, None)
        self._chunk_ready_req_ids.update(newly_finished)

        output = OmniConnectorOutput(
            chunk_ready_req_ids=set(self._chunk_ready_req_ids),
            chunk_finished_req_ids=chunk_finished,
            request_metadata=request_metadata,
            kv_sent_req_ids=list(self._kv_sent_req_ids),
            stage_recv_req_ids=set(self._stage_recv_req_ids),
            has_pending_kv_work=self.has_pending_kv_work(),
        )
        if output.stage_recv_req_ids or chunk_finished or newly_finished:
            logger.info(
                "[Stage-%s] get_omni_connector_output: stage_recv=%s, chunk_finished=%s, chunk_ready=%s",
                self._stage_id,
                output.stage_recv_req_ids,
                chunk_finished,
                output.chunk_ready_req_ids,
            )
        self._chunk_ready_req_ids.clear()
        self._kv_sent_req_ids.clear()
        self._stage_recv_req_ids.clear()
        return output

    @staticmethod
    def _connector_output_has_signals(output: OmniConnectorOutput) -> bool:
        return bool(
            output.chunk_ready_req_ids
            or output.chunk_finished_req_ids
            or output.request_metadata
            or output.kv_sent_req_ids
            or output.stage_recv_req_ids
            or output.has_pending_kv_work
        )

    def attach_omni_connector_output(self, result: Any | None) -> Any:
        omni_output = self.get_omni_connector_output()
        if not self._connector_output_has_signals(omni_output):
            return result

        from copy import copy

        from vllm.v1.worker.gpu_model_runner import EMPTY_MODEL_RUNNER_OUTPUT

        wrapped = copy(result if result is not None else EMPTY_MODEL_RUNNER_OUTPUT)
        wrapped.omni_connector_output = omni_output
        return wrapped

    # ------------------------------------------------------------------ #
    #  Properties for compatibility with custom_process funcs that access
    #  transfer_manager.put_req_chunk / request_payload / code_prompt_token_ids
    # ------------------------------------------------------------------ #

    @property
    def put_req_chunk(self) -> dict[str, int]:
        return self._put_req_chunk

    @property
    def request_payload(self) -> dict[str, dict[str, Any]]:
        return self._send_side_request_payload

    @request_payload.setter
    def request_payload(self, value: dict[str, dict[str, Any]]) -> None:
        self._send_side_request_payload = value

    @property
    def code_prompt_token_ids(self) -> dict[str, list[list[int]]]:
        return self._code_prompt_token_ids

    @property
    def connector(self) -> Any | None:
        return self._omni_connector

    # ------------------------------------------------------------------ #
    #  Background I/O threads
    # ------------------------------------------------------------------ #

    def _recv_loop(self) -> None:
        """Background thread: poll connector for incoming data."""
        _recv_poll_count = 0
        while not self._stop_event.is_set():
            with self._lock:
                pending_ids = list(self._pending_load_reqs.keys())

            if not pending_ids:
                self._work_available.wait(timeout=0.01)
                self._work_available.clear()
                continue

            _recv_poll_count += 1
            if _recv_poll_count % 5000 == 1:
                logger.info(
                    "[Stage-%s] _recv_loop: polling %s pending reqs: %s (poll#%s)",
                    self._stage_id,
                    len(pending_ids),
                    pending_ids[:5],
                    _recv_poll_count,
                )

            made_progress = False
            for req_id in pending_ids:
                if self._stop_event.is_set():
                    break
                try:
                    made_progress = self._poll_single_request(req_id) or made_progress
                except Exception:
                    logger.warning("Error receiving data for %s", req_id, exc_info=True)

            if not made_progress and not self._stop_event.is_set():
                self._work_available.wait(timeout=0.005)
                self._work_available.clear()

    _MAX_SEND_RETRIES = 3

    def _save_loop(self) -> None:
        """Background thread: send outgoing data via connector."""
        while not self._stop_event.is_set():
            task = None
            with self._lock:
                for req_id in list(self._pending_save_reqs.keys()):
                    dq = self._pending_save_reqs[req_id]
                    if dq:
                        task = dq.popleft()
                        if not dq:
                            del self._pending_save_reqs[req_id]
                        break
                    del self._pending_save_reqs[req_id]

            if task is not None:
                success = False
                try:
                    success = self._send_single_request(task)
                except Exception:
                    logger.error(
                        "Error saving data for %s",
                        task.get("request_id"),
                        exc_info=True,
                    )
                if not success:
                    self._requeue_or_drop_failed_send(task)
                continue

            self._work_available.wait(timeout=0.01)
            self._work_available.clear()

    def _requeue_or_drop_failed_send(self, task: dict) -> None:
        """Re-enqueue a failed send task or drop it after max retries."""
        retry_count = task.get("_retry_count", 0) + 1
        req_id = task.get("request_id")
        if retry_count <= self._MAX_SEND_RETRIES:
            task["_retry_count"] = retry_count
            logger.warning(
                "[Stage-%s] Re-enqueuing failed send for %s (retry %d/%d)",
                getattr(self, "_stage_id", "?"),
                req_id,
                retry_count,
                self._MAX_SEND_RETRIES,
            )
            with self._lock:
                dq = self._pending_save_reqs.setdefault(req_id, deque())
                dq.appendleft(task)
        else:
            logger.error(
                "[Stage-%s] Giving up on send for %s after %d retries",
                getattr(self, "_stage_id", "?"),
                req_id,
                self._MAX_SEND_RETRIES,
            )
            self._decrement_pending_save_count(req_id)

    # ------------------------------------------------------------------ #
    #  Chunk-level poll / send  (ported from OmniChunkTransferAdapter)
    # ------------------------------------------------------------------ #

    def _poll_single_request(self, req_id: str) -> bool:
        """Poll connector for one chunk of a request (non-blocking)."""
        connector = self._omni_connector
        if connector is None:
            return False

        if self._async_chunk and self._model_mode != "ar":
            with self._lock:
                staged_payload = self._local_stage_payload_cache.get(req_id)
                metadata_in_flight = req_id in self._local_request_metadata
                scheduler_wakeup_pending = req_id in self._finished_load_reqs
            if self._payload_is_consumable(staged_payload) or metadata_in_flight or scheduler_wakeup_pending:
                logger.debug(
                    "[Stage-%s] delaying recv for req=%s until staged async payload is handed to scheduler",
                    self._stage_id,
                    req_id,
                )
                return False

        target_stage_id = self._stage_id - 1
        chunk_id = self._get_req_chunk[req_id]
        external_req_id = self._request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        if self._async_chunk:
            result = self._recv_async_chunk_result(
                connector,
                str(target_stage_id),
                str(self._stage_id),
                connector_get_key,
            )
        else:
            result = self._recv_full_payload_result(
                connector,
                str(target_stage_id),
                str(self._stage_id),
                connector_get_key,
            )

        if result is None:
            return False

        payload_data, _size = result
        if not payload_data:
            return False
        if isinstance(payload_data, dict):
            logger.info(
                "[Stage-%s] recv_chunk_result: req=%s ext=%s key=%s keys=%s finished=%s",
                self._stage_id,
                req_id,
                external_req_id,
                connector_get_key,
                sorted(payload_data.keys()),
                self._payload_finished(payload_data),
            )

        self._get_req_chunk[req_id] += 1

        if self._async_chunk:
            is_finished = self._payload_finished(payload_data)
            incoming_payload_consumable = self._payload_is_consumable(payload_data)

            if self._model_mode == "ar":
                payload_data = self._accumulate_payload(external_req_id, payload_data)
                payload_consumable = incoming_payload_consumable
            else:
                new_ids = self._payload_audio_codes(payload_data) or []
                if not new_ids and not is_finished:
                    return False
                payload_consumable = self._payload_is_consumable(payload_data)

            with self._lock:
                if is_finished:
                    self._chunk_finished_req_ids.add(req_id)
                    self._chunk_stream_completed.add(req_id)
                # Local cache (RFC §2.4) — merge, don't replace, so that
                # earlier chunk keys (e.g. thinker_prefill_embeddings from
                # chunk 0) are not overwritten by later chunks.
                existing = self._local_stage_payload_cache.get(req_id)
                if existing is not None and isinstance(existing, dict) and isinstance(payload_data, dict):
                    existing.update(payload_data)
                else:
                    self._local_stage_payload_cache[req_id] = payload_data
                staged_payload = self._local_stage_payload_cache[req_id]
                self._async_chunk_updated_req_ids.add(req_id)
                self.put_local_request_metadata(req_id, self._extract_scheduling_metadata(staged_payload))
                # A finish-only sentinel still needs one terminal wake-up so
                # the downstream stage can sync the merged local payload and
                # flush/finish even when the last recv carries no new
                # consumable chunk bytes.
                if payload_consumable or is_finished:
                    self._finished_load_reqs.add(req_id)
                if is_finished and not payload_consumable:
                    logger.debug(
                        "[Stage-%s] finish sentinel arrived for req=%s without new consumable payload",
                        self._stage_id,
                        req_id,
                    )
                elif not payload_consumable:
                    logger.debug(
                        "[Stage-%s] req=%s received metadata-only / non-consumable async payload; delaying wake-up",
                        self._stage_id,
                        req_id,
                    )
                if is_finished:
                    self._pending_load_reqs.pop(req_id, None)
        else:
            # full_payload_mode: the complete payload arrives in a single get(),
            # so always unregister immediately.
            if isinstance(payload_data, dict):
                engine_inputs = payload_data.get("engine_inputs", payload_data)
            else:
                engine_inputs = payload_data
            with self._lock:
                self._local_stage_payload_cache[req_id] = self._snapshot_payload(engine_inputs)
                # Publish full-payload readiness only after the aligned TP broadcast
                # path in recv_full_payload_inputs() has materialized the payload on all
                # local ranks. Publishing metadata / stage_recv from the background recv
                # thread can let the scheduler observe a request before the payload is
                # actually visible to the model thread.
                self._full_payload_pending_broadcast_req_ids.add(req_id)
                self._pending_load_reqs.pop(req_id, None)
            logger.info(
                "[Stage-%s] full_payload recv complete: req=%s key=%s payload_type=%s",
                self._stage_id,
                req_id,
                connector_get_key,
                type(engine_inputs).__name__,
            )

        logger.debug("[Stage-%s] Received data for key %s", self._stage_id, connector_get_key)
        return True

    def _build_custom_process_payload(
        self,
        request_id: str | None,
        request: Any | None,
        pooling_output: Any | None,
    ) -> Any | None:
        """Run the custom process hook with a best-effort finished kwarg."""
        if self._custom_process_func is None:
            return None

        kwargs = {
            "transfer_manager": self,
            "pooling_output": pooling_output,
            "request": request,
        }
        supports_is_finished = getattr(
            self,
            "_custom_process_supports_is_finished",
            self._custom_process_supports_is_finished_kwarg(),
        )
        is_finished_fn = getattr(request, "is_finished", None)
        if callable(is_finished_fn):
            try:
                if supports_is_finished is not False:
                    kwargs["is_finished"] = bool(is_finished_fn())
            except Exception:
                logger.debug("request.is_finished() failed for %s", request_id, exc_info=True)

        try:
            return self._custom_process_func(**kwargs)
        except TypeError as exc:
            if "is_finished" not in kwargs or not self._is_unexpected_is_finished_kwarg_error(exc):
                logger.exception("custom_process_stage_input_func failed for chunk %s", request_id)
                return None
            kwargs.pop("is_finished", None)
            try:
                return self._custom_process_func(**kwargs)
            except Exception:
                logger.exception("custom_process_stage_input_func failed for chunk %s", request_id)
                return None
        except Exception:
            logger.exception("custom_process_stage_input_func failed for chunk %s", request_id)
            return None

    def _custom_process_supports_is_finished_kwarg(self) -> bool | None:
        """Return whether the custom process hook accepts `is_finished`."""
        if self._custom_process_func is None:
            return None
        try:
            signature = inspect.signature(self._custom_process_func)
        except (TypeError, ValueError):
            return None

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True

        is_finished_param = signature.parameters.get("is_finished")
        if is_finished_param is None:
            return False
        return is_finished_param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )

    @staticmethod
    def _is_unexpected_is_finished_kwarg_error(exc: TypeError) -> bool:
        message = str(exc)
        return (
            "unexpected keyword argument 'is_finished'" in message
            or 'unexpected keyword argument "is_finished"' in message
            or "positional-only arguments passed as keyword arguments: 'is_finished'" in message
        )

    def _send_single_request(self, task: dict) -> bool:
        """Send one queued task via connector.put().

        Returns True on success.  On failure (put() raises or returns
        ``success=False``), returns False **without** decrementing
        ``_pending_save_counts`` so the caller can retry or clean up.
        """
        connector = self._omni_connector
        if connector is None:
            return True

        request_id = task.get("request_id")
        payload_data = task.get("data")
        if payload_data is None and task.get("request") is not None:
            payload_data = self._build_custom_process_payload(
                request_id=request_id,
                request=task.get("request"),
                pooling_output=task.get("pooling_output"),
            )
        put_key = task.get("put_key")

        success, _size, _metadata = connector.put(
            from_stage=str(task["stage_id"]),
            to_stage=str(task["next_stage_id"]),
            put_key=put_key,
            data=payload_data,
        )
        logger.info(
            "[Stage-%s] _send_single_request: put_key=%s success=%s size=%s",
            task["stage_id"],
            put_key,
            success,
            _size,
        )

        if not success:
            return False

        self._decrement_pending_save_count(request_id)
        return True

    def _decrement_pending_save_count(self, request_id: str) -> None:
        """Decrement pending save count and run deferred cleanup if zero."""
        cleanup_req_id = None
        with self._lock:
            remaining = self._pending_save_counts.get(request_id, 0)
            if remaining > 1:
                self._pending_save_counts[request_id] = remaining - 1
            elif remaining == 1:
                self._pending_save_counts.pop(request_id, None)
                if request_id in self._deferred_send_cleanup:
                    self._deferred_send_cleanup.remove(request_id)
                    cleanup_req_id = request_id
            if cleanup_req_id is not None:
                self._put_req_chunk.pop(cleanup_req_id, None)
                self._send_side_request_payload.pop(cleanup_req_id, None)
                self._code_prompt_token_ids.pop(cleanup_req_id, None)
                self._cached_ic.pop(cleanup_req_id, None)

    # ------------------------------------------------------------------ #
    #  Payload accumulation  (ported from OmniChunkTransferAdapter)
    # ------------------------------------------------------------------ #

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
                    for tk, sk, ek in _EMBED_SPAN_GROUPS:
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
    def _freeze_request_attr(value: Any) -> Any:
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, torch.Tensor):
            return value.clone()
        raw_list = getattr(value, "_x", None)
        if raw_list is not None:
            return list(raw_list)
        return value

    def _snapshot_request_for_send(self, request: Any, external_req_id: str) -> Any:
        finished = bool(getattr(request, "is_finished", lambda: False)())
        attrs: dict[str, Any] = {}
        try:
            attrs.update(vars(request))
        except TypeError:
            pass

        for name in (
            "request_id",
            "req_id",
            "external_req_id",
            "prompt_token_ids",
            "output_token_ids",
            "all_token_ids",
            "additional_information",
            "sampling_params",
            "multi_modal_data",
            "mm_hashes",
        ):
            if hasattr(request, name):
                attrs[name] = self._freeze_request_attr(getattr(request, name))

        attrs["external_req_id"] = external_req_id
        attrs["_frozen_is_finished"] = finished
        snapshot = SimpleNamespace(**attrs)
        snapshot.is_finished = lambda: finished
        return snapshot

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
                    if not OmniConnectorModelRunnerMixin._is_connector_payload_builder(func):
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
            return getattr(request, "external_req_id", fallback_req_id)
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
