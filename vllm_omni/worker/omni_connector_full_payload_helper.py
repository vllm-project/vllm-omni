# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-payload transfer helper for OmniConnectorModelRunnerMixin.

Handles full_payload_mode: recv_full_payload_inputs / send_full_payload_outputs,
including accumulation and flushing logic.
"""

from __future__ import annotations

import importlib
from collections import deque
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


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


class OmniConnectorFullPayloadHelper:
    """Full-payload transfer management for connector communication."""

    def __init__(self, owner: Any):
        """Initialize full-payload helper with a reference to the owner mixin."""
        self._owner = owner

    def recv_full_payload_inputs(self, scheduler_output: Any) -> dict[str, Any] | None:
        """Check for incoming full_payload_mode stage inputs (non-blocking).

        Returns a dict mapping ``request_id -> engine_inputs`` for data
        that has arrived, or ``None`` if nothing is ready.  Stores full
        payloads in the local cache and extracts scheduling metadata.
        """
        owner = self._owner
        # Fast path: when TP is trivial (no peer ranks waiting on a broadcast)
        # and the bg recv thread has not staged anything, skip the lock + TP
        # broadcast cycle entirely. _broadcast_tp_payload_packet already
        # returns its input unchanged under the same world_size<=1 condition,
        # so the original code path was a no-op here on every empty step.
        tp_group = owner._get_local_tp_group()
        if (
            tp_group is None or getattr(tp_group, "world_size", 1) <= 1
        ) and not owner._full_payload_pending_broadcast_req_ids:
            return None
        with owner._lock:
            results = owner._collect_full_payload_results_locked() if owner.is_data_transfer_rank() else None
        results = owner._broadcast_tp_payload_packet(results)
        if not results:
            return None
        with owner._lock:
            owner._stage_recv_req_ids.update(results.keys())
            for req_id in results:
                owner._pending_load_reqs.pop(req_id, None)
            owner._apply_staged_payloads_locked(results)
            for req_id, payload in results.items():
                owner._local_request_metadata[req_id] = owner._extract_scheduling_metadata(payload)
        logger.info(
            "[Stage-%s] recv_full_payload_inputs: consumed %s reqs: %s, stage_recv_req_ids now=%s",
            owner._stage_id,
            len(results),
            list(results.keys()),
            owner._stage_recv_req_ids,
        )
        return results

    def _should_accumulate_full_payload_output(self) -> bool:
        """Gate send-side full-payload output accumulation only.

        Cached per instance: the result depends only on model_config /
        _custom_process_func, both of which are set at init time. Avoid
        the per-step dynamic import inside the model decode loop.
        """
        owner = self._owner
        if getattr(owner, "_omni_connector", None) is None:
            # No connector at all: send_full_payload_outputs would no-op.
            # Skip the per-step accumulator+build that would otherwise be
            # silently discarded.  Defends against a terminal stage whose
            # custom_process_input_func has a *_full_payload derivative in
            # the same module (e.g. dynin stage 2 token2image_to_token2audio
            # in pipelines that don't configure any connector at all).
            #
            # Known limitation: a *terminal-consumer* stage that has a
            # connector configured for receiving upstream input is NOT
            # caught here -- ``_omni_connector`` is non-None for it, and
            # ``_load_custom_func`` may still resolve a ``*_full_payload``
            # derivative from this stage's ``custom_process_input_func``.
            # In that case the accumulator builds payloads that
            # ``send_full_payload_outputs`` later drops via its own
            # connector-side checks (wasted CPU, not a functional bug).
            # A topology-aware gate (explicit producer field or pipeline
            # is_terminal info) would close the gap; that change is out
            # of scope for this PR.
            owner._should_accumulate_full_payload_output_cached = False
            return False
        cached = getattr(owner, "_should_accumulate_full_payload_output_cached", None)
        if cached is not None:
            return cached
        model_config = owner._get_model_config()
        if model_config is None:
            owner._should_accumulate_full_payload_output_cached = False
            return False
        result = should_accumulate_full_payload_output(
            model_config,
            getattr(owner, "_custom_process_func", None),
        )
        owner._should_accumulate_full_payload_output_cached = result
        return result

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
        owner = self._owner
        cached = getattr(owner, "_full_payload_replace_keys_cached", None)
        if cached is not None:
            return cached
        proc = getattr(owner, "_custom_process_func", None)
        if proc is None:
            owner._full_payload_replace_keys_cached = frozenset()
            return owner._full_payload_replace_keys_cached
        module_name = getattr(proc, "__module__", None)
        if module_name is None:
            owner._full_payload_replace_keys_cached = frozenset()
            return owner._full_payload_replace_keys_cached
        try:
            import sys as _sys

            mod = _sys.modules.get(module_name) or importlib.import_module(module_name)
            keys = getattr(mod, "_FULL_PAYLOAD_REPLACE_KEYS", frozenset())
        except ImportError:
            logger.debug(
                "Could not import stage input processor module %s while resolving "
                "_FULL_PAYLOAD_REPLACE_KEYS; using CONCAT semantics for all keys.",
                module_name,
                exc_info=True,
            )
            keys = frozenset()
        if not isinstance(keys, (frozenset, set)):
            logger.debug(
                "Ignoring non-set _FULL_PAYLOAD_REPLACE_KEYS from %s: %s",
                module_name,
                type(keys).__name__,
            )
            keys = frozenset()
        owner._full_payload_replace_keys_cached = frozenset(keys)
        logger.debug(
            "Resolved _FULL_PAYLOAD_REPLACE_KEYS for %s: %s",
            module_name,
            sorted(owner._full_payload_replace_keys_cached),
        )
        return owner._full_payload_replace_keys_cached

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
        owner = self._owner
        replace_keys = self._resolve_full_payload_replace_keys()
        existing = owner._pending_full_payload_send.get(req_id)

        if existing is None:
            chunks, latest, rows = self._new_full_payload_accumulator(pooler_output)
            owner._pending_full_payload_send[req_id] = (chunks, latest, rows, request)
            return

        if len(existing) == 2:
            chunks, latest, rows = self._new_full_payload_accumulator(existing[0])
        else:
            chunks, latest, rows, _ = existing

        for k, v in pooler_output.items():
            if v is None:
                continue
            if k in replace_keys:
                # Explicit REPLACE semantics: the new value supersedes any
                # prior chunks (e.g. `model_outputs` carries the full result
                # so far, not an appendable per-step delta).
                latest.pop(k, None)
                if isinstance(v, torch.Tensor) and v.dim() >= 2:
                    chunks[k] = [v]
                    rows[k] = int(v.shape[0])
                else:
                    chunks.pop(k, None)
                    rows.pop(k, None)
                    latest[k] = v
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

        owner._pending_full_payload_send[req_id] = (chunks, latest, rows, request)

    def flush_full_payload_outputs(self, finished_req_ids: set[str]) -> None:
        """Send accumulated full_payload outputs for requests that just finished."""
        owner = self._owner
        pending_req_ids = set(owner._pending_full_payload_send.keys())
        if not (finished_req_ids & pending_req_ids):
            return

        logger.info(
            "[Stage-%s] flush_full_payload_outputs: finished_req_ids=%s, pending=%s",
            owner._stage_id,
            finished_req_ids,
            list(owner._pending_full_payload_send.keys()),
        )
        to_send: dict[str, tuple[Any, Any]] = {}
        for req_id in finished_req_ids:
            entry = owner._pending_full_payload_send.pop(req_id, None)
            if entry is not None:
                to_send[req_id] = self._materialize_full_payload_entry(entry)
        logger.info("[Stage-%s] flush_full_payload_outputs: to_send=%s", owner._stage_id, list(to_send.keys()))
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
        owner = self._owner
        if owner._omni_connector is None:
            logger.info("[Stage-%s] send_full_payload_outputs: connector is None, skip", owner._stage_id)
            return []
        if not owner.is_data_transfer_rank():
            logger.info(
                "[Stage-%s] send_full_payload_outputs: not data_transfer_rank (rank=%s), skip",
                owner._stage_id,
                owner._local_rank,
            )
            return list(outputs.keys())
        sent_ids: list[str] = []
        next_stage_id = owner._next_stage_id
        for req_id, value in outputs.items():
            if isinstance(value, tuple) and len(value) == 2:
                raw_output, request = value
            else:
                raw_output, request = value, None

            payload = raw_output
            if owner._custom_process_func is not None:
                payload = owner._build_custom_process_payload(
                    request_id=req_id,
                    request=request,
                    pooling_output=raw_output,
                )
                if payload is None:
                    continue
            if payload is None:
                logger.info("[Stage-%s] send_full_payload_outputs: payload is None for %s", owner._stage_id, req_id)
                continue
            if isinstance(payload, dict):
                audio_codes = owner._payload_audio_codes(payload)
                if isinstance(audio_codes, torch.Tensor):
                    code_len = int(audio_codes.numel())
                elif hasattr(audio_codes, "__len__"):
                    code_len = len(audio_codes)
                else:
                    code_len = None
                meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                logger.info(
                    "[Stage-%s] send_full_payload_outputs: req=%s payload_keys=%s code_len=%s left_context_size=%s",
                    owner._stage_id,
                    req_id,
                    sorted(payload.keys()),
                    code_len,
                    meta.get("left_context_size"),
                )

            external_req_id = owner._resolve_external_req_id(request, req_id)
            chunk_id = owner._put_req_chunk[req_id]
            owner._put_req_chunk[req_id] += 1
            connector_put_key = f"{external_req_id}_{owner._stage_id}_{chunk_id}"

            logger.info(
                "[Stage-%s] send_full_payload_outputs: enqueue req=%s put_key=%s next_stage=%s",
                owner._stage_id,
                req_id,
                connector_put_key,
                next_stage_id,
            )
            task = {
                "stage_id": owner._stage_id,
                "next_stage_id": next_stage_id,
                "put_key": connector_put_key,
                "data": payload,
                "request_id": req_id,
            }
            with owner._lock:
                owner._pending_save_reqs.setdefault(req_id, deque()).append(task)
                owner._pending_save_counts[req_id] += 1
            sent_ids.append(req_id)
        if sent_ids:
            owner._work_available.set()
        return sent_ids

    # ------------------------------------------------------------------ #
    #  Compatibility wrappers
    # ------------------------------------------------------------------ #

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
