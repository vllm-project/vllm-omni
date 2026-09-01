# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni connector payload transport for model runners."""

from __future__ import annotations

import importlib
import inspect
from collections import deque
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.parallel_state import get_tp_group

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_runtime import (
    _OmniConnectorRuntimeMixin,
    logger,
    should_accumulate_full_payload_output,
)
from vllm_omni.outputs import OmniConnectorOutput
from vllm_omni.utils.payload_span import (
    get_tensor_span,
    merge_tensor_spans,
)

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.connectors.base import (
        OmniConnectorBase,
    )


_EMBED_SPAN_GROUPS: tuple[tuple[str, str, str], ...] = (("decode", "decode_token_start", "decode_token_end"),)


class _OmniConnectorPayloadTransportMixin(_OmniConnectorRuntimeMixin):
    """Own payload caching, full/chunk transport, and connector I/O."""

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

    #  Output aggregation
    # ------------------------------------------------------------------ #

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
            logger.debug(
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

        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

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
    def ramp_chunk_count(self) -> dict[str, int]:
        return self._ramp_chunk_count

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
        logger.debug(
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
        if getattr(self, "_omni_connector", None) is None:
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
            self._should_accumulate_full_payload_output_cached = False
            return False
        cached = getattr(self, "_should_accumulate_full_payload_output_cached", None)
        if cached is not None:
            return cached
        model_config = self._get_model_config()
        if model_config is None:
            self._should_accumulate_full_payload_output_cached = False
            return False
        result = should_accumulate_full_payload_output(
            model_config,
            getattr(self, "_custom_process_func", None),
        )
        self._should_accumulate_full_payload_output_cached = result
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
        cached = getattr(self, "_full_payload_replace_keys_cached", None)
        if cached is not None:
            return cached
        proc = getattr(self, "_custom_process_func", None)
        if proc is None:
            self._full_payload_replace_keys_cached = frozenset()
            return self._full_payload_replace_keys_cached
        module_name = getattr(proc, "__module__", None)
        if module_name is None:
            self._full_payload_replace_keys_cached = frozenset()
            return self._full_payload_replace_keys_cached
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
        self._full_payload_replace_keys_cached = frozenset(keys)
        logger.debug(
            "Resolved _FULL_PAYLOAD_REPLACE_KEYS for %s: %s",
            module_name,
            sorted(self._full_payload_replace_keys_cached),
        )
        return self._full_payload_replace_keys_cached

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
        replace_keys = self._resolve_full_payload_replace_keys()
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

        self._pending_full_payload_send[req_id] = (chunks, latest, rows, request)

    def flush_full_payload_outputs(self, finished_req_ids: set[str]) -> None:
        """Send accumulated full_payload outputs for requests that just finished."""
        pending_req_ids = set(self._pending_full_payload_send.keys())
        if not (finished_req_ids & pending_req_ids):
            return

        logger.debug(
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
        logger.debug("[Stage-%s] flush_full_payload_outputs: to_send=%s", self._stage_id, list(to_send.keys()))
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
            logger.debug("[Stage-%s] send_full_payload_outputs: connector is None, skip", self._stage_id)
            return []
        if not self.is_data_transfer_rank():
            logger.debug(
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
                logger.debug("[Stage-%s] send_full_payload_outputs: payload is None for %s", self._stage_id, req_id)
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
                logger.debug(
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
            self._ramp_chunk_count[req_id] += 1
            connector_put_key = f"{external_req_id}_{self._stage_id}_{chunk_id}"

            logger.debug(
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
        # Explicit external_req_id=None must fall back to request_id;
        # otherwise recv keys become `None_<stage>_<chunk>` and collide
        # across requests.
        ext = getattr(request, "external_req_id", None)
        self._request_ids_mapping[request_id] = ext if ext is not None else request_id
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
        self._ramp_chunk_count[request_id] += 1
        next_stage_id = self._next_stage_id
        connector_put_key = f"{request_id}_{self._stage_id}_{chunk_id}"

        if chunk_id == 0:
            logger.debug(
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
                logger.debug(
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
            logger.debug(
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
            logger.debug(
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
        logger.debug(
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
                self._ramp_chunk_count.pop(cleanup_req_id, None)
                self._adaptive_states.pop(cleanup_req_id, None)

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
