# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache transfer helper for OmniConnectorModelRunnerMixin.

Handles KV cache transfer, rank-aware routing, and KV lifecycle management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class OmniConnectorKVHelper:
    """KV cache transfer management for connector communication."""

    def __init__(self, owner: Any):
        """Initialize KV helper with a reference to the owner mixin."""
        self._owner = owner

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
        owner = self._owner
        if owner._kv_transfer_manager is None:
            return list(finished_reqs.keys()) if finished_reqs else []
        result = owner._kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished_reqs,
            kv_caches=kv_caches,
            block_size=block_size,
            cache_dtype=cache_dtype,
            request_id_resolver=request_id_resolver,
        )
        if result:
            owner._kv_sent_req_ids.extend(result)
        return result

    def recv_kv_cache(
        self,
        request_id: str,
        target_device: torch.device | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for a request.

        Delegates to the existing ``OmniKVTransferManager``.
        """
        owner = self._owner
        if owner._kv_transfer_manager is None:
            return None, 0
        return owner._kv_transfer_manager.receive_kv_cache_for_request(
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
        owner = self._owner
        if owner._kv_transfer_manager is None:
            return False

        request_id = getattr(req, "request_id", None)
        if not request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return False

        active_requests = getattr(owner, "requests", None)
        if active_requests is not None and request_id not in active_requests:
            logger.info("Skip receiving KV cache for inactive request %s", request_id)
            return False

        primary_ok = False
        data, _size = self.recv_kv_cache(request_id, target_device=target_device)
        if data:
            owner._kv_transfer_manager.apply_kv_cache_to_request(req, data)
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
        owner = self._owner
        remote_ranks = self.get_kv_remote_ranks()
        return [
            self.get_kv_connector_key(
                req_id=req_id,
                from_stage=from_stage,
                chunk_id=chunk_id,
                from_rank=remote_rank,
                to_rank=owner._local_rank,
            )
            for remote_rank in remote_ranks
        ]

    def get_kv_target_ranks_for_send(self) -> list[int]:
        """Determine which target ranks this local rank should send KV shards to."""
        owner = self._owner
        self._validate_kv_tp_topology()
        if owner._from_tp == owner._to_tp:
            return [owner._local_rank]
        if owner._from_tp > owner._to_tp:
            tp_ratio = owner._from_tp // owner._to_tp
            return [owner._local_rank // tp_ratio]
        tp_ratio = owner._to_tp // owner._from_tp
        base_rank = owner._local_rank * tp_ratio
        return [base_rank + i for i in range(tp_ratio)]

    def get_rank_aware_kv_send_keys(
        self,
        req_id: str,
        from_stage: int,
        to_stage: int | None = None,
        chunk_id: int = 0,
    ) -> list[str]:
        """Build send-side connector keys for this rank's KV shard(s)."""
        owner = self._owner
        target_ranks = self.get_kv_target_ranks_for_send()
        return [
            self.get_kv_connector_key(
                req_id=req_id,
                from_stage=from_stage,
                chunk_id=chunk_id,
                from_rank=owner._local_rank,
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
        owner = self._owner
        if payload is None or owner._from_tp >= owner._to_tp:
            return payload

        tp_ratio = owner._to_tp // owner._from_tp
        shard_index = owner._local_rank % tp_ratio
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
        metadata["sliced_for_local_rank"] = owner._local_rank
        payload["metadata"] = metadata
        return payload

    def should_replicate_payload(self) -> bool:
        """Whether non-KV payloads should be replicated across ranks.

        Data payloads (stage inputs, chunks) are identical after all-gather,
        so only rank 0 transfers them.  KV payloads are rank-specific and
        all ranks participate.
        """
        owner = self._owner
        return owner._local_rank != 0

    def get_kv_rank_mapping(self) -> dict[str, Any]:
        """Return the current rank mapping configuration.

        Useful for debugging and for downstream code that needs to know
        the TP topology without re-parsing model config.
        """
        owner = self._owner
        return {
            "from_tp": owner._from_tp,
            "to_tp": owner._to_tp,
            "local_rank": owner._local_rank,
            "remote_ranks": self.get_kv_remote_ranks(),
            "is_data_transfer_rank": owner.is_data_transfer_rank(),
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
        owner = self._owner
        if req_id in owner._kv_pending_transfers:
            return
        owner._kv_triggered_requests.add(req_id)
        transfer = {
            "seq_len": seq_len,
            "block_ids": block_ids,
        }
        if custom_metadata is not None:
            transfer["custom_metadata"] = custom_metadata
        owner._kv_pending_transfers[req_id] = transfer

    def drain_pending_kv_transfers(self) -> dict[str, dict[str, Any]]:
        """Drain pending KV transfers and move them to active.

        Returns ``{req_id: {seq_len, block_ids}}`` for the model runner
        to submit to ``send_kv_cache``.
        """
        owner = self._owner
        if not owner._kv_pending_transfers:
            return {}
        pending = dict(owner._kv_pending_transfers)
        owner._kv_active_transfers.update(pending.keys())
        owner._kv_pending_transfers.clear()
        return pending

    def ack_kv_transfers(self, req_ids: list[str] | set[str]) -> None:
        """Acknowledge completed KV transfers (from kv_extracted_req_ids).

        Moves requests from active to completed so the scheduler can
        safely free their blocks.
        """
        owner = self._owner
        for req_id in req_ids:
            owner._kv_active_transfers.discard(req_id)
            owner._kv_completed_transfers.add(req_id)

    def drain_completed_kv_transfers(self) -> set[str]:
        """Drain and return completed KV transfer request IDs.

        The scheduler calls this to know which requests' blocks can be freed.
        """
        owner = self._owner
        completed = set(owner._kv_completed_transfers)
        owner._kv_completed_transfers.clear()
        return completed

    def is_kv_transfer_triggered(self, req_id: str) -> bool:
        """Check if a request has already triggered KV transfer."""
        owner = self._owner
        return req_id in owner._kv_triggered_requests

    def has_pending_kv_work(self) -> bool:
        """True if any KV transfers are pending, active, or awaiting ack."""
        owner = self._owner
        return bool(owner._kv_pending_transfers or owner._kv_active_transfers or owner._kv_completed_transfers)

    # ------------------------------------------------------------------ #
    #  Heterogeneous TP rank support
    # ------------------------------------------------------------------ #

    def _validate_kv_tp_topology(self) -> None:
        """Reject heterogeneous TP mappings that cannot be routed losslessly."""
        owner = self._owner
        if owner._from_tp <= 0 or owner._to_tp <= 0:
            raise ValueError(f"Invalid KV TP mapping: from_tp={owner._from_tp}, to_tp={owner._to_tp}")
        larger = max(owner._from_tp, owner._to_tp)
        smaller = min(owner._from_tp, owner._to_tp)
        if larger % smaller != 0:
            raise ValueError(
                "KV TP mapping must be divisible for rank-aware routing: "
                f"from_tp={owner._from_tp}, "
                f"to_tp={owner._to_tp}"
            )

    def get_kv_remote_ranks(self) -> list[int]:
        """Determine which remote ranks this local rank exchanges KV with.

        Follows vLLM's ``TpKVTopology.get_target_remote_ranks()`` pattern:
        - ``from_tp > to_tp``: each to-rank reads from multiple from-ranks
        - ``from_tp < to_tp``: multiple to-ranks read from the same from-rank
        - ``from_tp == to_tp``: 1:1 mapping
        """
        owner = self._owner
        self._validate_kv_tp_topology()
        if owner._from_tp == owner._to_tp:
            return [owner._local_rank]

        if owner._from_tp > owner._to_tp:
            tp_ratio = owner._from_tp // owner._to_tp
            return [owner._local_rank * tp_ratio + i for i in range(tp_ratio)]
        else:
            tp_ratio = owner._to_tp // owner._from_tp
            return [owner._local_rank // tp_ratio]

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
