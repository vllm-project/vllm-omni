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

from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.outputs import OmniConnectorOutput
from vllm_omni.worker.omni_connector_full_payload_helper import (
    OmniConnectorFullPayloadHelper,
)
from vllm_omni.worker.omni_connector_kv_helper import OmniConnectorKVHelper
from vllm_omni.worker.omni_connector_lifecycle_helper import OmniConnectorLifecycleHelper
from vllm_omni.worker.omni_connector_request_state_helper import (
    OmniConnectorRequestStateHelper,
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


# should_accumulate_full_payload_output is now imported from omni_connector_full_payload_helper


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
        # Initialize helper instances
        self._lifecycle_helper = OmniConnectorLifecycleHelper(self)
        self._full_payload_helper = OmniConnectorFullPayloadHelper(self)
        self._kv_helper = OmniConnectorKVHelper(self)
        self._request_state_helper = OmniConnectorRequestStateHelper(self)

        # Delegate to lifecycle helper for initialization
        self._lifecycle_helper.init_omni_connectors(
            vllm_config,
            model_config,
            kv_transfer_manager,
        )

    def shutdown_omni_connectors(self) -> None:
        """Stop background threads and release connector resources."""
        self._lifecycle_helper.shutdown_omni_connectors()

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
        self._lifecycle_helper.cleanup_finished_request(req_id)

    def drop_inactive_request_delivery_state(self, req_id: str) -> None:
        """Clear recv-side state for inactive requests."""
        self._lifecycle_helper.drop_inactive_request_delivery_state(req_id)

    def _drop_send_side_payload_state(self, req_id: str, ext_id: str | None) -> None:
        self._lifecycle_helper._drop_send_side_payload_state(req_id, ext_id)

    def _cleanup_recv_delivery_state(self, req_id: str) -> None:
        """Clear recv-side delivery-cycle state."""
        self._lifecycle_helper._cleanup_recv_delivery_state(req_id)

    def _clear_recv_delivery_state(self, req_id: str) -> None:
        self._lifecycle_helper._clear_recv_delivery_state(req_id)

    def prune_inactive_requests(self, active_req_ids: Any) -> set[str]:
        """Drop connector state for requests that no longer exist locally.

        Preempted / unscheduled requests are expected to stay in
        ``self.requests`` and therefore remain untouched. This only prunes
        stale request IDs that have already fallen out of the active request
        map, preventing background recv/send bookkeeping from outliving the
        request lifecycle.
        """
        return self._lifecycle_helper.prune_inactive_requests(active_req_ids)

    # ------------------------------------------------------------------ #
    #  Local payload cache (RFC §2.4 – Model Runner ownership)
    # ------------------------------------------------------------------ #

    def put_local_stage_payload(self, req_id: str, payload: OmniPayload) -> None:
        """Store a full stage payload in the local cache."""
        self._request_state_helper.put_local_stage_payload(req_id, payload)

    def get_local_stage_payload(self, req_id: str) -> OmniPayload | None:
        """Read a stage payload without removing it."""
        return self._request_state_helper.get_local_stage_payload(req_id)

    def pop_local_stage_payload(self, req_id: str) -> OmniPayload | None:
        """Remove and return a stage payload (consume after use)."""
        return self._request_state_helper.pop_local_stage_payload(req_id)

    def put_local_request_metadata(self, req_id: str, metadata: dict[str, Any]) -> None:
        """Store lightweight scheduling metadata for a request."""
        self._request_state_helper.put_local_request_metadata(req_id, metadata)

    def get_local_request_metadata(self, req_id: str) -> dict[str, Any] | None:
        """Retrieve scheduling metadata for a request."""
        return self._request_state_helper.get_local_request_metadata(req_id)

    # ------------------------------------------------------------------ #
    #  Scheduling metadata extraction
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_scheduling_metadata(cls, payload: OmniPayload) -> dict[str, Any]:
        """Extract only the fields the scheduler needs from a full payload."""
        return OmniConnectorRequestStateHelper._extract_scheduling_metadata(payload)

    _NON_CONSUMABLE_PAYLOAD_KEYS = OmniConnectorRequestStateHelper._NON_CONSUMABLE_PAYLOAD_KEYS

    @staticmethod
    def _payload_value_has_content(value: Any) -> bool:
        return OmniConnectorRequestStateHelper._payload_value_has_content(value)

    @staticmethod
    def _payload_finished(payload: Any) -> bool:
        return OmniConnectorRequestStateHelper._payload_finished(payload)

    @staticmethod
    def _payload_audio_codes(payload: Any) -> Any:
        return OmniConnectorRequestStateHelper._payload_audio_codes(payload)

    @classmethod
    def _payload_is_consumable(cls, payload: OmniPayload | None) -> bool:
        """Return True when an async payload can drive a real forward step.

        Metadata-only wake-ups should not transition WAITING_FOR_CHUNK requests
        back to schedulable state. In particular, a widened token horizon without
        any newly visible thinker decode embeds should not force a placeholder-only
        talker decode step.
        """
        return OmniConnectorRequestStateHelper._payload_is_consumable(payload)

    @staticmethod
    def _get_local_tp_group() -> Any | None:
        """Return the local TP group when tensor parallelism is initialized."""
        return OmniConnectorRequestStateHelper._get_local_tp_group()

    def _recv_ordinary_stage_result(
        self,
        connector: OmniConnectorBase,
        from_stage: str,
        to_stage: str,
        connector_get_key: str,
    ) -> Any:
        """Receive one ordinary non-KV stage payload on the local leader rank only."""
        return self._request_state_helper._recv_ordinary_stage_result(
            connector,
            from_stage,
            to_stage,
            connector_get_key,
        )

    def _recv_full_payload_result(
        self,
        connector: OmniConnectorBase,
        from_stage: str,
        to_stage: str,
        connector_get_key: str,
    ) -> Any:
        """Receive one full-payload transfer on the local leader rank only."""
        return self._request_state_helper._recv_full_payload_result(
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
        return self._request_state_helper._recv_async_chunk_result(
            connector,
            from_stage,
            to_stage,
            connector_get_key,
        )

    @staticmethod
    def _snapshot_payload(payload: Any) -> Any:
        return OmniConnectorRequestStateHelper._snapshot_payload(payload)

    def _broadcast_tp_payload_packet(self, packet: Any) -> Any:
        """Broadcast one ordinary payload packet from TP rank 0 when TP is active."""
        return self._request_state_helper._broadcast_tp_payload_packet(packet)

    def _apply_staged_payloads_locked(self, staged_payloads: dict[str, Any]) -> None:
        self._request_state_helper._apply_staged_payloads_locked(staged_payloads)

    def _collect_full_payload_results_locked(self) -> dict[str, Any] | None:
        return self._request_state_helper._collect_full_payload_results_locked()

    def _collect_async_chunk_fanout_packet_locked(self) -> dict[str, Any] | None:
        return self._request_state_helper._collect_async_chunk_fanout_packet_locked()

    def _apply_async_chunk_fanout_packet(self, packet: dict[str, Any]) -> None:
        self._request_state_helper._apply_async_chunk_fanout_packet(packet)

    # ------------------------------------------------------------------ #
    #  full_payload_mode (recv_full_payload_inputs / send_full_payload_outputs)
    # ------------------------------------------------------------------ #

    def recv_full_payload_inputs(self, scheduler_output: Any) -> dict[str, Any] | None:
        """Check for incoming full_payload_mode stage inputs (non-blocking).

        Returns a dict mapping ``request_id -> engine_inputs`` for data
        that has arrived, or ``None`` if nothing is ready.  Stores full
        payloads in the local cache and extracts scheduling metadata.
        """
        return self._full_payload_helper.recv_full_payload_inputs(scheduler_output)

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
        return self._full_payload_helper._should_accumulate_full_payload_output()

    @staticmethod
    def _new_full_payload_accumulator(output: dict[str, Any]):
        return OmniConnectorFullPayloadHelper._new_full_payload_accumulator(output)

    @staticmethod
    def _materialize_full_payload_entry(entry):
        return OmniConnectorFullPayloadHelper._materialize_full_payload_entry(entry)

    def _resolve_full_payload_replace_keys(self) -> frozenset:
        """Per-model REPLACE-key set for the full-payload accumulator.

        Looked up from the stage-input-processor module that ships the model's sync builder
        (`model_config.custom_process_input_func.__module__`).  The module
        declares ``_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str]``; if absent,
        returns the empty set.

        Cached per instance.  Keys in this set use REPLACE semantics in the
        accumulator (subsequent emissions discard prior chunks) instead of
        the default CONCAT semantics.  Use for tensors that carry the full
        result so far rather than per-step deltas (e.g. ``model_outputs``).
        """
        return self._full_payload_helper._resolve_full_payload_replace_keys()

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
        self._full_payload_helper.accumulate_full_payload_output(req_id, pooler_output, request)

    def flush_full_payload_outputs(self, finished_req_ids: set[str]) -> None:
        """Send accumulated full_payload outputs for requests that just finished."""
        self._full_payload_helper.flush_full_payload_outputs(finished_req_ids)

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
        return self._full_payload_helper.send_full_payload_outputs(scheduler_output, outputs)

    def recv_stage_inputs(self, scheduler_output: Any) -> dict[str, Any] | None:
        """Compatibility wrapper for ``recv_full_payload_inputs``."""
        return self._full_payload_helper.recv_stage_inputs(scheduler_output)

    def accumulate_batch_output(
        self,
        req_id: str,
        pooler_output: Any,
        request: Any,
    ) -> None:
        """Compatibility wrapper for ``accumulate_full_payload_output``."""
        self._full_payload_helper.accumulate_batch_output(req_id, pooler_output, request)

    def flush_batch_outputs(self, finished_req_ids: set[str]) -> None:
        """Compatibility wrapper for ``flush_full_payload_outputs``."""
        self._full_payload_helper.flush_batch_outputs(finished_req_ids)

    def send_stage_outputs(
        self,
        scheduler_output: Any,
        outputs: dict[str, tuple[Any, Any] | Any],
    ) -> list[str]:
        """Compatibility wrapper for ``send_full_payload_outputs``."""
        return self._full_payload_helper.send_stage_outputs(scheduler_output, outputs)

    # ------------------------------------------------------------------ #
    #  Streaming chunk mode  (recv_chunk / send_chunk)
    # ------------------------------------------------------------------ #

    def register_chunk_recv(self, request: Any) -> None:
        """Register a request for async chunk retrieval by the bg thread.

        Stage-0 has no upstream producer so this is a no-op there.
        Skips requests whose batch data has already been received to
        prevent the bg thread from polling for non-existent chunks.
        """
        self._request_state_helper.register_chunk_recv(request)

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
        return self._request_state_helper.recv_chunk()

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
        return self._request_state_helper.send_chunk(request, pooling_output)

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
        return self._kv_helper.send_kv_cache(
            finished_reqs,
            kv_caches,
            block_size,
            cache_dtype,
            request_id_resolver,
        )

    def recv_kv_cache(
        self,
        request_id: str,
        target_device: torch.device | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for a request.

        Delegates to the existing ``OmniKVTransferManager``.
        """
        return self._kv_helper.recv_kv_cache(request_id, target_device)

    def receive_cfg_companion_kv_payloads(
        self,
        cfg_request_ids: dict[str, str],
        target_device: torch.device | None = None,
    ) -> dict[str, tuple[dict[str, Any] | None, int]]:
        """Receive raw CFG companion KV payloads keyed by role."""
        return self._kv_helper.receive_cfg_companion_kv_payloads(cfg_request_ids, target_device)

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
        return self._kv_helper.receive_multi_kv_cache(req, cfg_kv_collect_func, target_device)

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
        return self._kv_helper.get_rank_aware_kv_keys(req_id, from_stage, to_stage, chunk_id)

    def get_kv_target_ranks_for_send(self) -> list[int]:
        """Determine which target ranks this local rank should send KV shards to."""
        return self._kv_helper.get_kv_target_ranks_for_send()

    def get_rank_aware_kv_send_keys(
        self,
        req_id: str,
        from_stage: int,
        to_stage: int | None = None,
        chunk_id: int = 0,
    ) -> list[str]:
        """Build send-side connector keys for this rank's KV shard(s)."""
        return self._kv_helper.get_rank_aware_kv_send_keys(req_id, from_stage, to_stage, chunk_id)

    @staticmethod
    def _merge_rank_sharded_kv_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Merge multiple source-rank KV shards for one target rank."""
        return OmniConnectorKVHelper._merge_rank_sharded_kv_payloads(payloads)

    def _slice_rank_sharded_kv_payload(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """Slice a duplicated source-rank KV shard for ``from_tp < to_tp`` cases."""
        return self._kv_helper._slice_rank_sharded_kv_payload(payload)

    def should_replicate_payload(self) -> bool:
        """Whether non-KV payloads should be replicated across ranks.

        Data payloads (stage inputs, chunks) are identical after all-gather,
        so only rank 0 transfers them.  KV payloads are rank-specific and
        all ranks participate.
        """
        return self._kv_helper.should_replicate_payload()

    def get_kv_rank_mapping(self) -> dict[str, Any]:
        """Return the current rank mapping configuration.

        Useful for debugging and for downstream code that needs to know
        the TP topology without re-parsing model config.
        """
        return self._kv_helper.get_kv_rank_mapping()

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
        self._kv_helper.mark_kv_transfer(req_id, seq_len, block_ids, custom_metadata)

    def drain_pending_kv_transfers(self) -> dict[str, dict[str, Any]]:
        """Drain pending KV transfers and move them to active.

        Returns ``{req_id: {seq_len, block_ids}}`` for the model runner
        to submit to ``send_kv_cache``.
        """
        return self._kv_helper.drain_pending_kv_transfers()

    def ack_kv_transfers(self, req_ids: list[str] | set[str]) -> None:
        """Acknowledge completed KV transfers (from kv_extracted_req_ids).

        Moves requests from active to completed so the scheduler can
        safely free their blocks.
        """
        self._kv_helper.ack_kv_transfers(req_ids)

    def drain_completed_kv_transfers(self) -> set[str]:
        """Drain and return completed KV transfer request IDs.

        The scheduler calls this to know which requests' blocks can be freed.
        """
        return self._kv_helper.drain_completed_kv_transfers()

    def is_kv_transfer_triggered(self, req_id: str) -> bool:
        """Check if a request has already triggered KV transfer."""
        return self._kv_helper.is_kv_transfer_triggered(req_id)

    def has_pending_kv_work(self) -> bool:
        """True if any KV transfers are pending, active, or awaiting ack."""
        return self._kv_helper.has_pending_kv_work()

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
        self._lifecycle_helper._recv_loop()

    _MAX_SEND_RETRIES = OmniConnectorLifecycleHelper._MAX_SEND_RETRIES

    def _save_loop(self) -> None:
        """Background thread: send outgoing data via connector."""
        self._lifecycle_helper._save_loop()

    def _requeue_or_drop_failed_send(self, task: dict) -> None:
        """Re-enqueue a failed send task or drop it after max retries."""
        self._lifecycle_helper._requeue_or_drop_failed_send(task)

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
        return self._request_state_helper._build_custom_process_payload(request_id, request, pooling_output)

    def _custom_process_supports_is_finished_kwarg(self) -> bool | None:
        """Return whether the custom process hook accepts `is_finished`."""
        return self._request_state_helper._custom_process_supports_is_finished_kwarg()

    @staticmethod
    def _is_unexpected_is_finished_kwarg_error(exc: TypeError) -> bool:
        return OmniConnectorRequestStateHelper._is_unexpected_is_finished_kwarg_error(exc)

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
        self._lifecycle_helper._decrement_pending_save_count(request_id)

    # ------------------------------------------------------------------ #
    #  Payload accumulation  (ported from OmniChunkTransferAdapter)
    # ------------------------------------------------------------------ #

    def _accumulate_payload(self, req_id: str, payload_data: OmniPayload) -> OmniPayload:
        """Accumulate chunk payloads (concat tensors, extend lists)."""
        return self._request_state_helper._accumulate_payload(req_id, payload_data)

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
        return OmniConnectorRequestStateHelper._freeze_request_attr(value)

    def _snapshot_request_for_send(self, request: Any, external_req_id: str) -> Any:
        return self._request_state_helper._snapshot_request_for_send(request, external_req_id)

    @staticmethod
    def _create_connector(model_config: Any) -> OmniConnectorBase | None:
        """Create a connector from model_config, or None if unconfigured."""
        return OmniConnectorLifecycleHelper._create_connector(model_config)

    @staticmethod
    def _load_custom_func(model_config: Any) -> tuple[str | None, Any | None]:
        """Load the connector payload builder for the downstream stage.

        Preferred source is ``custom_process_next_stage_input_func``. Some
        full_payload_mode configs (async_chunk=false) only expose the next-stage prompt builder via
        ``custom_process_input_func`` (for example ``thinker2talker``), while the
        connector payload builder lives beside it as ``thinker2talker_full_payload``.
        In that case, derive the full_payload_mode builder path automatically.
        """
        return OmniConnectorLifecycleHelper._load_custom_func(model_config)

    @staticmethod
    def _is_connector_payload_builder(func: Any) -> bool:
        """Whether *func* matches the mixin payload-builder contract."""
        return OmniConnectorLifecycleHelper._is_connector_payload_builder(func)

    def _resolve_external_req_id(self, request: Any, fallback_req_id: str) -> str:
        """Resolve the external request ID consistently.

        Checks ``_request_ids_mapping`` first (populated by
        ``register_chunk_recv``), then falls back to the request's
        ``external_req_id`` attribute, and finally to the given
        ``fallback_req_id``.
        """
        return self._request_state_helper._resolve_external_req_id(request, fallback_req_id)

    def _resolve_next_stage_id(self, model_config: Any) -> int:
        """Determine the downstream stage ID from connector config.

        Falls back to ``stage_id + 1`` when the config does not specify
        a ``to_stage`` explicitly.
        """
        return self._lifecycle_helper._resolve_next_stage_id(model_config)

    @staticmethod
    def _parse_rank_mapping(model_config: Any) -> dict[str, int]:
        """Parse rank_mapping from connector config (optional).

        Returns ``{"from_tp": int, "to_tp": int, "local_rank": int}``.
        When ``rank_mapping`` is absent, assumes 1:1 homogeneous mapping.
        """
        return OmniConnectorLifecycleHelper._parse_rank_mapping(model_config)

    # ------------------------------------------------------------------ #
    #  Heterogeneous TP rank support
    # ------------------------------------------------------------------ #

    def _validate_kv_tp_topology(self) -> None:
        """Reject heterogeneous TP mappings that cannot be routed losslessly."""
        self._kv_helper._validate_kv_tp_topology()

    def get_kv_remote_ranks(self) -> list[int]:
        """Determine which remote ranks this local rank exchanges KV with.

        Follows vLLM's ``TpKVTopology.get_target_remote_ranks()`` pattern:
        - ``from_tp > to_tp``: each to-rank reads from multiple from-ranks
        - ``from_tp < to_tp``: multiple to-ranks read from the same from-rank
        - ``from_tp == to_tp``: 1:1 mapping
        """
        return self._kv_helper.get_kv_remote_ranks()

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
        return self._kv_helper.get_kv_connector_key(req_id, from_stage, chunk_id, from_rank, to_rank)
