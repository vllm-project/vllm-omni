# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-state helper for OmniConnectorModelRunnerMixin.

Handles local payload cache, metadata, chunk tracking, and payload utilities.
"""

from __future__ import annotations

import inspect
from collections import deque
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.worker.payload_span import (
    get_tensor_span,
    merge_tensor_spans,
)

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.connectors.base import (
        OmniConnectorBase,
    )

logger = init_logger(__name__)

_EMBED_SPAN_GROUPS: tuple[tuple[str, str, str], ...] = (("decode", "decode_token_start", "decode_token_end"),)


class OmniConnectorRequestStateHelper:
    """Request-state management for connector communication."""

    def __init__(self, owner: Any):
        """Initialize request-state helper with a reference to the owner mixin."""
        self._owner = owner

    # ------------------------------------------------------------------ #
    #  Local payload cache (RFC §2.4 – Model Runner ownership)
    # ------------------------------------------------------------------ #

    def put_local_stage_payload(self, req_id: str, payload: OmniPayload) -> None:
        """Store a full stage payload in the local cache."""
        self._owner._local_stage_payload_cache[req_id] = payload

    def get_local_stage_payload(self, req_id: str) -> OmniPayload | None:
        """Read a stage payload without removing it."""
        return self._owner._local_stage_payload_cache.get(req_id)

    def pop_local_stage_payload(self, req_id: str) -> OmniPayload | None:
        """Remove and return a stage payload (consume after use)."""
        return self._owner._local_stage_payload_cache.pop(req_id, None)

    def put_local_request_metadata(self, req_id: str, metadata: dict[str, Any]) -> None:
        """Store lightweight scheduling metadata for a request."""
        self._owner._local_request_metadata[req_id] = metadata

    def get_local_request_metadata(self, req_id: str) -> dict[str, Any] | None:
        """Retrieve scheduling metadata for a request."""
        return self._owner._local_request_metadata.get(req_id)

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
        if not self._owner.is_data_transfer_rank():
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
        leader_packet = packet if self._owner.is_data_transfer_rank() else None
        return tp_group.broadcast_object(leader_packet, src=0)

    def _apply_staged_payloads_locked(self, staged_payloads: dict[str, Any]) -> None:
        owner = self._owner
        for req_id, payload in staged_payloads.items():
            owner._local_stage_payload_cache[req_id] = self._snapshot_payload(payload)

    def _collect_full_payload_results_locked(self) -> dict[str, Any] | None:
        owner = self._owner
        if not owner._full_payload_pending_broadcast_req_ids:
            return None
        results: dict[str, Any] = {}
        missing_req_ids: list[str] = []
        for req_id in tuple(owner._full_payload_pending_broadcast_req_ids):
            payload = owner._local_stage_payload_cache.get(req_id)
            if payload is None:
                missing_req_ids.append(req_id)
                continue
            results[req_id] = self._snapshot_payload(payload)
            owner._full_payload_pending_broadcast_req_ids.discard(req_id)
        if missing_req_ids:
            logger.warning(
                "[Stage-%s] _collect_full_payload_results_locked: "
                "pending full-payload reqs missing from local cache: %s",
                owner._stage_id,
                missing_req_ids,
            )
        return results or None

    def _collect_async_chunk_fanout_packet_locked(self) -> dict[str, Any] | None:
        owner = self._owner
        payload_req_ids = set(owner._async_chunk_updated_req_ids)
        payload_req_ids.update(owner._finished_load_reqs)
        payload_req_ids.update(owner._chunk_finished_req_ids)
        payload_req_ids.update(owner._local_request_metadata)
        if not (
            payload_req_ids
            or owner._finished_load_reqs
            or owner._chunk_finished_req_ids
            or owner._local_request_metadata
        ):
            return None

        staged_payloads = {
            req_id: self._snapshot_payload(owner._local_stage_payload_cache[req_id])
            for req_id in payload_req_ids
            if req_id in owner._local_stage_payload_cache
        }
        packet = {
            "staged_payloads": staged_payloads,
            "request_metadata": dict(owner._local_request_metadata),
            "newly_finished": set(owner._finished_load_reqs),
            "chunk_finished": set(owner._chunk_finished_req_ids),
        }

        owner._async_chunk_updated_req_ids.clear()
        owner._finished_load_reqs.clear()
        owner._chunk_finished_req_ids.clear()
        owner._local_request_metadata.clear()

        for req_id in packet["chunk_finished"]:
            if req_id not in owner._local_stage_payload_cache:
                continue
            ext_req_id = owner._request_ids_mapping.get(req_id, req_id)
            owner._send_side_request_payload.pop(ext_req_id, None)
            if ext_req_id != req_id:
                owner._send_side_request_payload.pop(req_id, None)

        return packet

    def _apply_async_chunk_fanout_packet(self, packet: dict[str, Any]) -> None:
        owner = self._owner
        staged_payloads = packet.get("staged_payloads", {})
        chunk_finished = set(packet.get("chunk_finished", ()))
        with owner._lock:
            self._apply_staged_payloads_locked(staged_payloads)
            for req_id in chunk_finished:
                owner._pending_load_reqs.pop(req_id, None)
                owner._chunk_stream_completed.add(req_id)

    # ------------------------------------------------------------------ #
    #  Streaming chunk mode  (recv_chunk / send_chunk)
    # ------------------------------------------------------------------ #

    def register_chunk_recv(self, request: Any) -> None:
        """Register a request for async chunk retrieval by the bg thread.

        Stage-0 has no upstream producer so this is a no-op there.
        Skips requests whose batch data has already been received to
        prevent the bg thread from polling for non-existent chunks.
        """
        owner = self._owner
        if owner._stage_id == 0:
            return
        request_id = request.request_id
        # Explicit external_req_id=None must fall back to request_id;
        # otherwise recv keys become `None_<stage>_<chunk>` and collide
        # across requests.
        ext = getattr(request, "external_req_id", None)
        owner._request_ids_mapping[request_id] = ext if ext is not None else request_id
        with owner._lock:
            if request_id in owner._stage_recv_req_ids:
                return
            # Don't re-register if the finish sentinel was already received
            if request_id in owner._chunk_stream_completed:
                return
            owner._pending_load_reqs[request_id] = request
        owner._work_available.set()

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
        owner = self._owner
        with owner._lock:
            finished = set(owner._finished_load_reqs)
            if not finished:
                return {}
            # Snapshot the payloads under the lock to avoid racing with
            # _poll_single_request which does existing.update(payload_data)
            # on the same dict objects.
            result = {}
            for rid in finished:
                payload = owner._local_stage_payload_cache.get(rid)
                result[rid] = dict(payload) if isinstance(payload, dict) else payload

        owner._chunk_ready_req_ids.update(finished)
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
        owner = self._owner
        if owner._omni_connector is None:
            logger.warning("[Stage-%s] send_chunk: connector is None", owner._stage_id)
            return False
        if not owner.is_data_transfer_rank():
            return True
        raw_req_id = getattr(request, "request_id", None) or getattr(request, "req_id", None)
        request_id = owner._resolve_external_req_id(request, raw_req_id)
        # Cache the internal→external mapping so that finish sentinels can
        # resolve the external ID even after the request is freed.
        if raw_req_id and raw_req_id != request_id:
            owner._request_ids_mapping.setdefault(raw_req_id, request_id)
        chunk_id = owner._put_req_chunk[request_id]

        payload_data = owner._build_custom_process_payload(
            request_id=request_id,
            request=request,
            pooling_output=pooling_output,
        )
        if payload_data is None:
            if chunk_id == 0:
                logger.warning(
                    "[Stage-%s] send_chunk: payload is None for req=%s chunk=%s (process_func=%s)",
                    owner._stage_id,
                    request_id,
                    chunk_id,
                    owner._custom_process_func,
                )
            return False

        owner._put_req_chunk[request_id] += 1
        next_stage_id = owner._next_stage_id
        connector_put_key = f"{request_id}_{owner._stage_id}_{chunk_id}"

        if chunk_id == 0:
            logger.info(
                "[Stage-%s] send_chunk: first chunk enqueued, req=%s key=%s",
                owner._stage_id,
                request_id,
                connector_put_key,
            )

        task = {
            "stage_id": owner._stage_id,
            "next_stage_id": next_stage_id,
            "put_key": connector_put_key,
            "data": payload_data,
            "request_id": request_id,
        }
        with owner._lock:
            owner._pending_save_reqs.setdefault(request_id, deque()).append(task)
            owner._pending_save_counts[request_id] += 1
        owner._work_available.set()
        return True

    # ------------------------------------------------------------------ #
    #  Payload accumulation  (ported from OmniChunkTransferAdapter)
    # ------------------------------------------------------------------ #

    def _accumulate_payload(self, req_id: str, payload_data: OmniPayload) -> OmniPayload:
        """Accumulate chunk payloads (concat tensors, extend lists)."""
        owner = self._owner
        if req_id not in owner._send_side_request_payload:
            owner._send_side_request_payload[req_id] = dict(payload_data)
            return dict(owner._send_side_request_payload[req_id])

        origin = owner._send_side_request_payload[req_id]
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

        owner._send_side_request_payload[req_id] = merged
        return dict(merged)

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

    def _build_custom_process_payload(
        self,
        request_id: str | None,
        request: Any | None,
        pooling_output: Any | None,
    ) -> Any | None:
        """Run the custom process hook with a best-effort finished kwarg."""
        owner = self._owner
        if owner._custom_process_func is None:
            return None

        kwargs = {
            "transfer_manager": owner,
            "pooling_output": pooling_output,
            "request": request,
        }
        supports_is_finished = getattr(
            owner,
            "_custom_process_supports_is_finished",
            owner._custom_process_supports_is_finished_kwarg(),
        )
        is_finished_fn = getattr(request, "is_finished", None)
        if callable(is_finished_fn):
            try:
                if supports_is_finished is not False:
                    kwargs["is_finished"] = bool(is_finished_fn())
            except Exception:
                logger.debug("request.is_finished() failed for %s", request_id, exc_info=True)

        try:
            return owner._custom_process_func(**kwargs)
        except TypeError as exc:
            if "is_finished" not in kwargs or not self._is_unexpected_is_finished_kwarg_error(exc):
                logger.exception("custom_process_stage_input_func failed for chunk %s", request_id)
                return None
            kwargs.pop("is_finished", None)
            try:
                return owner._custom_process_func(**kwargs)
            except Exception:
                logger.exception("custom_process_stage_input_func failed for chunk %s", request_id)
                return None
        except Exception:
            logger.exception("custom_process_stage_input_func failed for chunk %s", request_id)
            return None

    def _custom_process_supports_is_finished_kwarg(self) -> bool | None:
        """Return whether the custom process hook accepts `is_finished`."""
        owner = self._owner
        if owner._custom_process_func is None:
            return None
        try:
            signature = inspect.signature(owner._custom_process_func)
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

    def _resolve_external_req_id(self, request: Any, fallback_req_id: str) -> str:
        """Resolve the external request ID consistently.

        Checks ``_request_ids_mapping`` first (populated by
        ``register_chunk_recv``), then falls back to the request's
        ``external_req_id`` attribute, and finally to the given
        ``fallback_req_id``.
        """
        owner = self._owner
        mapped = owner._request_ids_mapping.get(fallback_req_id)
        if mapped is not None:
            return mapped
        if request is not None:
            # external_req_id may be explicitly None; fall back.
            ext = getattr(request, "external_req_id", None)
            if ext is not None:
                return ext
        return fallback_req_id
