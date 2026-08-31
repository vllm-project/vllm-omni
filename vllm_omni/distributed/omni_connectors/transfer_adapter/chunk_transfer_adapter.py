# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy
import importlib
import inspect
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import ConstantList

from vllm_omni.data_entry_keys import MetaStruct, OmniPayloadStruct, unflatten_payload

from ..adapter import construct_next_stage_streaming_input_prompt
from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec, stage_receives_chunks
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)


class _SenderGeneration:
    """Fence one external request generation without blocking cleanup."""

    def __init__(self) -> None:
        self.in_flight = False
        self.cancelled = False


class _LoadEntry:
    """Identify one receiver registration across queue and I/O boundaries."""

    __slots__ = ("request",)

    def __init__(self, request: Request) -> None:
        self.request = request

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def external_req_id(self) -> str:
        return self.request.external_req_id


def _resolve_talker_streaming_prompt_config(model_config: Any) -> tuple[int, int]:
    """Return the effective Talker context limit and recompute window."""
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    hf_config = getattr(model_config, "hf_config", None)
    tts_config = getattr(hf_config, "tts_config", None)
    if tts_config is None and getattr(hf_config, "model_type", None) == "minicpmtts":
        tts_config = hf_config
    tts_max_model_len = (
        tts_config.get("max_position_embeddings", 0)
        if isinstance(tts_config, Mapping)
        else getattr(tts_config, "max_position_embeddings", 0)
    )
    tts_max_model_len = int(tts_max_model_len or 0)
    if tts_max_model_len > 0:
        max_model_len = min(max_model_len, tts_max_model_len) if max_model_len else tts_max_model_len
    attention_type = (
        tts_config.get("attention_type")
        if isinstance(tts_config, Mapping)
        else getattr(tts_config, "attention_type", None)
    )
    # Official MiniCPMTTS supports both full_attention and
    # sliding_recompute. Only an explicit Talker-stage policy enables the
    # latter; native duplex alone must not override checkpoint semantics.
    previous_chunks = 1 if attention_type == "sliding_recompute" else 0
    return max_model_len, previous_chunks


class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    """Chunk-level transfer adapter for Omni connector pipelines.

    This class coordinates per-request chunk exchange between adjacent stages,
    and implements asynchronous get/put of chunks via background threads.
    It tracks per-request chunk indices for put/get, and accumulates
    payloads across chunks (concatenating tensors/lists in AR mode). It also
    caches prompt token ids and additional information for scheduler use.

    Scheduler integration is handled via WAITING_FOR_CHUNK transitions:
    requests are moved to waiting for chunk deque while polling, then restored
    to waiting/running queues once a chunk arrives. The requests will finish
    loading chunk util detecting the payload "finished" flag.

    The base class owns background recv/save loops; load/save only enqueue
    work and return immediately.
    """

    def __init__(self, vllm_config: Any):
        model_config = vllm_config.model_config
        # The base constructor starts the save thread, so sender-generation
        # state must exist before it can observe an enqueued task. The global
        # lock only protects this map and short state transitions. Slow payload
        # construction and connector writes never hold it.
        self._sender_state_lock = threading.Lock()
        # Serialize receiver registration/cleanup with the commit of a chunk
        # returned by connector.get(). The get itself stays outside the lock so
        # scheduler cleanup never waits for connector I/O.
        self._receiver_state_lock = threading.Lock()
        # Includes requests queued for polling and the request currently being
        # polled. Per-registration identity lets the receiver reject an old
        # segment's queued or in-flight work after the same request id resumes.
        self._registered_load_entries: dict[str, _LoadEntry] = {}
        self._sender_tokens: dict[str, _SenderGeneration] = {}
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._max_model_len, self._streaming_prompt_previous_chunks = _resolve_talker_streaming_prompt_config(
            model_config
        )
        active_stream_window = int(getattr(model_config, "active_stream_window", 0) or 0)
        model_max_num_seqs = int(getattr(model_config, "max_num_seqs", self.scheduler_max_num_seqs) or 0)
        if model_max_num_seqs <= 0:
            model_max_num_seqs = self.scheduler_max_num_seqs
        self._active_window = min(active_stream_window, model_max_num_seqs) if active_stream_window > 0 else 0
        if self._active_window > 0:
            logger.info(
                "Bounded active-stream window enabled: K=%d. "
                "Multi-replica deployments require sticky per-stream routing across Stage 1 "
                "replicas (each replica owns an independent active-set; without sticky routing, "
                "a stream can be active on one replica and non-active on another and both will "
                "race to evict it).",
                self._active_window,
            )
        self.connector = self.create_connector(model_config)
        self.receives_chunks = stage_receives_chunks(model_config)
        super().__init__(model_config)
        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        # State specific to Chunk management
        self.custom_process_next_stage_input_func: Callable[..., OmniPayloadStruct | None] | None = None
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        self._processor_accepts_step_tokens: dict[Callable[..., Any], bool] = {}
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        # Segment-local chunk counter: incremented alongside put_req_chunk
        # but popped at segment boundaries (unlike put_req_chunk which is
        # request-global for connector key continuity).
        self.ramp_chunk_count: dict[str, int] = defaultdict(int)
        self._adaptive_states: dict[str, Any] = {}
        self.upstream_exhausted_requests: set[str] = set()
        self.segment_finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()
        self.replaced_streaming_prompt_ids: set[str] = set()
        self.requests_origin_status = {}
        self._active_streams: dict[str, Any] = {}
        # Private hold-queue for non-active running requests. Restored to
        # running_queue inside restore_queues(). Avoids calling
        # waiting_queue.prepend_requests mid-step, which trips vllm's
        # per-step LogitsProcessor invariant
        # ("Cannot register new removed request after self.removed has
        #   been read").
        self._held_non_active: deque[Any] = deque()
        self.requests_num_chunks_sent: dict[str, int] = defaultdict(int)
        # Boundary tasks advance this before the background sender runs, so a
        # late old-segment frame cannot restore the previous watermark.
        self._segment_generation: dict[str, int] = defaultdict(int)
        self._pending_streaming_prefills: dict[str, dict] = {}
        # The recv thread only records AR payloads. Prompt mutation is finalized
        # by the scheduler after it observes the ready marker, where the request
        # token counters form a consistent snapshot.
        self._pending_ar_prompt_updates: dict[
            str,
            tuple[Request, dict[str, Any], bool, int | None, bool],
        ] = {}
        self._streaming_condition_lengths: dict[str, int] = {}
        self._streaming_condition_seqs: dict[str, int] = {}
        # Monotonic timestamp of when each request last began waiting for a
        # chunk, refreshed every time one arrives.  Read by
        # collect_timed_out_request_ids() so a stream that stops advancing
        # becomes a client-visible error instead of parking forever.  Mirrors
        # OmniSchedulingCoordinator._waiting_since on the full-payload path.
        self._waiting_since: dict[str, float] = {}

    @staticmethod
    def _is_truthy_scalar(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return value.numel() == 1 and bool(value.item())
        return bool(value) if value is not None else False

    @staticmethod
    def _confirmed_num_computed_tokens(request: Request) -> int:
        # vLLM async scheduling advances num_computed_tokens with output
        # placeholders before the corresponding token is committed. Connector
        # chunk send watermarks must use only committed tokens.
        num_computed = int(getattr(request, "num_computed_tokens", 0))
        num_placeholders = int(getattr(request, "num_output_placeholders", 0) or 0)
        return max(0, num_computed - num_placeholders)

    @staticmethod
    def _snapshot_processor_request(request: Request) -> Request:
        """Snapshot mutable processor inputs at save-queue admission time."""

        def snapshot_container(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: snapshot_container(item) for key, item in value.items()}
            if isinstance(value, list):
                return [snapshot_container(item) for item in value]
            if isinstance(value, tuple):
                return tuple(snapshot_container(item) for item in value)
            return value

        snapshot = copy.copy(request)
        for name in (
            "additional_information",
            "prompt_token_ids",
        ):
            if hasattr(request, name):
                setattr(snapshot, name, snapshot_container(getattr(request, name)))

        for private_name, public_name in (
            ("_all_token_ids", "all_token_ids"),
            ("_output_token_ids", "output_token_ids"),
        ):
            token_ids = getattr(request, private_name, None)
            if isinstance(token_ids, list):
                frozen_ids = token_ids.copy()
                setattr(snapshot, private_name, frozen_ids)
                setattr(snapshot, public_name, ConstantList(frozen_ids))
        return snapshot

    @staticmethod
    def _refresh_generation_chunk_prefill_state(request: Request) -> None:
        request.num_prompt_tokens = len(request.prompt_token_ids)
        if getattr(request, "prefill_stats", None) is None:
            request.prefill_stats = PrefillStats()

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def load_async(self, request: Request):
        """Register a request for asynchronous chunk retrieval.

        This method does not read from the connector directly. It records
        request metadata and enqueues the request id for the background
        receive loop to poll.

        Stage-0 has no upstream producer, so this call is a no-op there.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0 or not self.receives_chunks:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        with self._receiver_state_lock:
            request_id = request.request_id
            if request_id in self._registered_load_entries:
                return
            entry = _LoadEntry(request)
            self._cancelled_load_reqs.discard(request_id)
            self._registered_load_entries[request_id] = entry
            self._pending_load_reqs.append(entry)
        with self._recv_cond:
            self._recv_cond.notify()

    def _accepts_new_token_ids(self, processor: Callable[..., Any]) -> bool:
        cached = self._processor_accepts_step_tokens.get(processor)
        if cached is not None:
            return cached
        try:
            parameters = inspect.signature(processor).parameters.values()
        except (TypeError, ValueError):
            accepts = False
        else:
            accepts = any(
                parameter.name == "new_token_ids" or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        self._processor_accepts_step_tokens[processor] = accepts
        return accepts

    def save_async(
        self,
        multimodal_output: dict[str, Any] | None = None,
        request: Request | None = None,
        is_segment_finished: bool = False,
        new_token_ids: Iterable[int] | None = None,
        confirmed_num_computed_tokens: int | None = None,
        segment_generation: int | None = None,
    ):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction happens in ``_send_single_request`` on the
        background save_loop thread.

        For streaming input request ``is_segment_finished`` marks the end
        of the current realtime input segment. It is intentionally separate
        from ``request.is_finished()``: a resumable `/v1/realtime` session
        can finish one audio segment and later continue with another segment
        under the same external request id. For other requests, it is the same
        as ``request.is_finished()``.

        Args:
            multimodal_output: Per-request multimodal output dictionary
            request: Request object
            is_segment_finished: whether the segment of request is finished
            new_token_ids: immutable snapshot of tokens generated by this
                scheduler step, before resumable cleanup mutates the request
            confirmed_num_computed_tokens: committed token count captured
                before a streaming transition can mutate ``request``
            segment_generation: generation captured before a resumable stop
                can apply a queued update to the mutable request
        """
        is_finished = request.is_finished() and not request.resumable
        if not hasattr(self, "_segment_generation"):
            self._segment_generation = defaultdict(int)
        external_req_id = request.external_req_id
        if segment_generation is None:
            raw_generation = getattr(request, "_omni_segment_generation", 0)
            try:
                generation = int(raw_generation)
            except (TypeError, ValueError):
                generation = 0
        else:
            generation = int(segment_generation)
        expected_generation = self._segment_generation.get(external_req_id, generation)
        if generation < expected_generation:
            logger.warning(
                "Skip late save_async for request %s, segment_generation=%s, expected=%s",
                external_req_id,
                generation,
                expected_generation,
            )
            return
        self._segment_generation[external_req_id] = generation

        if confirmed_num_computed_tokens is None:
            confirmed_num_computed_tokens = self._confirmed_num_computed_tokens(request)
        processor_request = (
            self._snapshot_processor_request(request) if self.custom_process_next_stage_input_func else request
        )
        task = {
            "multimodal_output": multimodal_output,
            "request": processor_request,
            "is_finished": is_finished,
            "is_segment_finished": is_segment_finished,
            "new_token_ids": tuple(int(token_id) for token_id in (new_token_ids or ())),
            "segment_generation": generation,
        }

        reject_reason = None
        with self._sender_state_lock:
            sender_token = self._sender_tokens.get(external_req_id)
            if sender_token is not None and sender_token.cancelled:
                reject_reason = "previous sender generation is still draining"
            # If the request is preempted, skip the already saved chunks.
            elif not is_segment_finished and confirmed_num_computed_tokens < self.requests_num_chunks_sent.get(
                external_req_id, 0
            ):
                logger.warning(
                    f"Enqueue save_async for request {external_req_id}, "
                    f"request.num_computed_tokens={request.num_computed_tokens}, "
                    f"request.num_output_placeholders={getattr(request, 'num_output_placeholders', 0)}, "
                    f"previous_chunks_sent={self.requests_num_chunks_sent.get(external_req_id, 0)}"
                )
                return

            else:
                self.requests_num_chunks_sent[external_req_id] = confirmed_num_computed_tokens
                if sender_token is None:
                    sender_token = _SenderGeneration()
                    self._sender_tokens[external_req_id] = sender_token
                task["sender_token"] = sender_token
                self._pending_save_reqs.append(task)
                if is_segment_finished:
                    # The queued FIFO item now owns the old segment. Start the next
                    # segment's deduplication watermark before the worker sends it.
                    self._segment_generation[external_req_id] = generation + 1
                    self.requests_num_chunks_sent.pop(external_req_id, None)
        if reject_reason is not None:
            logger.error("Cannot enqueue %s: %s", external_req_id, reject_reason)
            self.record_send_failure(request.request_id, reject_reason)
            return
        with self._save_cond:
            self._save_cond.notify()

    def _poll_single_request(self, entry: _LoadEntry):
        request = entry.request
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        req_id = request.request_id
        with self._receiver_state_lock:
            if self._registered_load_entries.get(req_id) is not entry:
                return True
            chunk_id = self.get_req_chunk[req_id]
            external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        try:
            result = self.connector.get(
                str(target_stage_id),
                str(stage_id),
                connector_get_key,
            )
        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {connector_get_key}: {e}")
            with self._receiver_state_lock:
                if self._registered_load_entries.get(req_id) is not entry:
                    return True
            return False

        if result is None:
            with self._receiver_state_lock:
                if self._registered_load_entries.get(req_id) is not entry:
                    return True
            return False

        with self._receiver_state_lock:
            # cleanup_receiver() can run while connector.get() is in flight.
            # Treat cleanup as a commit barrier: a late chunk must not recreate
            # prompt/window state for the removed request.
            if self._registered_load_entries.get(req_id) is not entry:
                return True
            is_success = self._commit_received_chunk(
                request,
                result,
                stage_id=stage_id,
                req_id=req_id,
                chunk_id=chunk_id,
                connector_get_key=connector_get_key,
            )
            if is_success:
                self._registered_load_entries.pop(req_id, None)
            return is_success

    def _commit_received_chunk(
        self,
        request: Request,
        result: tuple[dict[str, Any], int],
        *,
        stage_id: int,
        req_id: str,
        chunk_id: int,
        connector_get_key: str,
    ) -> bool:
        """Commit a received connector chunk while receiver state is locked."""
        payload_data, size = result

        if payload_data:
            # Update connector state
            self.get_req_chunk[req_id] += 1

            meta = payload_data.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
                payload_data["meta"] = meta
            payload_finished = self._is_truthy_scalar(meta.get("finished"))
            payload_segment_finished = self._is_truthy_scalar(meta.get("is_segment_finished"))
            if self.model_mode == "ar":
                was_resumable = bool(getattr(request, "resumable", False))
                meta["streaming_prompt_recompute"] = False
                prompt_len = meta.get("next_stage_prompt_len")
                replace_prompt = meta.get("replace_streaming_prompt") is True
                payload_ids = payload_data.get("ids")
                has_prompt_payload = bool(
                    replace_prompt
                    or (isinstance(prompt_len, int) and not isinstance(prompt_len, bool) and prompt_len > 0)
                    or (isinstance(payload_ids, dict) and payload_ids.get("prompt"))
                )
                window_condition_len = (
                    prompt_len
                    if self._streaming_prompt_previous_chunks == 1
                    and isinstance(prompt_len, int)
                    and not isinstance(prompt_len, bool)
                    and prompt_len > 0
                    else None
                )
                update_prompt = bool(was_resumable and has_prompt_payload and (chunk_id > 0 or replace_prompt))
                self._pending_ar_prompt_updates[req_id] = (
                    request,
                    payload_data,
                    update_prompt,
                    window_condition_len,
                    was_resumable,
                )

                if payload_finished:
                    self.upstream_exhausted_requests.add(req_id)
                    request.resumable = False
                if payload_segment_finished:
                    self.segment_finished_requests.add(req_id)
            else:
                if payload_finished:
                    self.upstream_exhausted_requests.add(req_id)
                    request.resumable = False
                if payload_segment_finished:
                    self.segment_finished_requests.add(req_id)

                new_ids = payload_data.get("codes", {}).get("audio")
                has_tensor_codes = isinstance(new_ids, torch.Tensor)
                use_tensor_codes = has_tensor_codes and new_ids.ndim >= 2
                prompt_token_ids: list[int]
                if use_tensor_codes:
                    prompt_token_ids = [0] if new_ids.numel() > 0 else []
                elif has_tensor_codes:
                    new_ids = new_ids.tolist()
                    prompt_token_ids = new_ids
                elif new_ids is None:
                    new_ids = []
                    prompt_token_ids = new_ids
                else:
                    prompt_token_ids = new_ids
                request.prompt_token_ids = prompt_token_ids
                # Full-snapshot producers opt in explicitly; generation and
                # diffusion models keep their existing incremental merge.
                prev_info = getattr(request, "additional_information", None)
                replace_snapshot = meta.get("replace_runtime_additional_information") is True
                info = {} if replace_snapshot else (dict(prev_info) if isinstance(prev_info, dict) else {})
                for key, value in payload_data.items():
                    if key == "codes":
                        if isinstance(value, dict):
                            existing_sub = info.get(key)
                            merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                            for subkey, subvalue in value.items():
                                # A 1-D audio tensor is represented by the
                                # placeholder prompt above, but sibling fields
                                # such as the reference voice still belong in
                                # the current runtime snapshot.
                                if subkey == "audio" and not use_tensor_codes:
                                    continue
                                merged_sub[subkey] = subvalue
                            if merged_sub:
                                info[key] = merged_sub
                        continue
                    if isinstance(value, dict):
                        existing_sub = info.get(key)
                        merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                        for sk, sv in value.items():
                            if key == "meta" and sk == "finished":
                                continue
                            merged_sub[sk] = sv
                        info[key] = merged_sub
                        continue
                    info[key] = value
                request.additional_information = info
                request.num_computed_tokens = 0

                # Empty chunk with more data expected: keep polling.
                has_new_ids = bool(new_ids.numel()) if use_tensor_codes else bool(new_ids)
                if not has_new_ids and payload_segment_finished:
                    # Preserve an explicit scheduler boundary even when it
                    # contains no new codec frames.
                    request.prompt_token_ids = [0]
                if not has_new_ids and not payload_finished and not payload_segment_finished:
                    # The base recv loop treats False as "not ready yet" and
                    # requeues the request. Do not mark an empty non-terminal
                    # chunk as ready, otherwise Stage1 can consume before the
                    # first DAC frame arrives.
                    return False
                self._refresh_generation_chunk_prefill_state(request)

            # Mark as finished for consumption
            self._finished_load_reqs.add(req_id)
            logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
            return True

        return False

    def _send_single_request(self, task: dict):
        request = task["request"]
        external_req_id = request.external_req_id
        sender_token = task.get("sender_token")
        if sender_token is None:
            self._send_single_request_for_generation(task)
            return
        with self._sender_state_lock:
            is_current = self._sender_tokens.get(external_req_id) is sender_token
            if is_current and not sender_token.cancelled:
                sender_token.in_flight = True
            else:
                is_current = False
        if not is_current:
            logger.debug("Discarding stale queued chunk for aborted request %s", external_req_id)
            return
        try:
            self._send_single_request_for_generation(task, sender_token)
        finally:
            with self._sender_state_lock:
                if self._sender_tokens.get(external_req_id) is sender_token:
                    sender_token.in_flight = False
                    if sender_token.cancelled:
                        self._sender_tokens.pop(external_req_id, None)
                        self._clear_sender_state_locked(external_req_id)

    def _sender_generation_is_active(
        self,
        external_req_id: str,
        sender_token: _SenderGeneration,
    ) -> bool:
        with self._sender_state_lock:
            return self._sender_tokens.get(external_req_id) is sender_token and not sender_token.cancelled

    def _send_single_request_for_generation(
        self,
        task: dict,
        sender_token: _SenderGeneration | None = None,
    ):
        raw_mm = task["multimodal_output"]
        multimodal_output = unflatten_payload(raw_mm) if isinstance(raw_mm, Mapping) else raw_mm
        request = task["request"]
        is_finished = task["is_finished"]
        is_segment_finished = task["is_segment_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        external_req_id = request.external_req_id
        chunk_id = self.put_req_chunk[external_req_id]
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
        # Process payload in save_loop thread
        payload_data: OmniPayloadStruct | None = None
        if self.custom_process_next_stage_input_func:
            try:
                processor = self.custom_process_next_stage_input_func
                processor_kwargs = {
                    "transfer_manager": self,
                    "multimodal_output": multimodal_output,
                    "request": request,
                    # Existing processors use is_finished as a flush signal.
                    # Terminal stops no longer count as segment boundaries
                    # (is_segment_finished is False when the request finishes,
                    # see #5383), but the processor must still flush its
                    # accumulated tail on the terminal chunk — otherwise the
                    # downstream stage receives the finished marker without
                    # the final payload (#5413).
                    "is_finished": is_segment_finished or is_finished,
                }
                if self._accepts_new_token_ids(processor):
                    processor_kwargs["new_token_ids"] = task.get("new_token_ids", ())
                payload_data = processor(**processor_kwargs)

            except Exception as e:
                logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

        if payload_data is None:
            if not (is_segment_finished or is_finished):
                return
            # Segment/request finish markers must still reach downstream even when
            # the processor has no tensor payload.
            payload_data = OmniPayloadStruct()
        if payload_data.meta is None:
            payload_data.meta = MetaStruct()
        # A model-native codec stream may span many resumable scheduler
        # requests.  Its processor is the only layer that knows whether a
        # request finish is a transport wake or the lifetime of the codec
        # state itself.  Preserve an explicit streaming lifetime decision;
        # all ordinary producers keep the scheduler-derived default.
        processor_controls_codec_lifetime = payload_data.meta.finished is not None and self._is_truthy_scalar(
            payload_data.meta.codec_streaming
        )
        if not processor_controls_codec_lifetime:
            payload_data.meta.finished = torch.tensor(is_finished, dtype=torch.bool)
        if payload_data.meta.is_segment_finished is None:
            payload_data.meta.is_segment_finished = torch.tensor(is_segment_finished, dtype=torch.bool)

        if sender_token is not None and not self._sender_generation_is_active(external_req_id, sender_token):
            logger.debug("Skipping cancelled chunk for request %s before connector put", external_req_id)
            return

        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )

        if sender_token is not None and not self._sender_generation_is_active(external_req_id, sender_token):
            # cleanup_sender() may cancel this generation while put() is
            # blocked. Do not let that stale completion recreate state for a
            # request whose cleanup is already in progress.
            logger.debug("Ignoring completed put for cancelled request %s", external_req_id)
            return

        if success:
            self.put_req_chunk[external_req_id] += 1
            self.ramp_chunk_count[external_req_id] += 1
            logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")
            # Sender uses struct attr access here; the receive path in
            # `_load_one_request` / `_update_request_payload` reads dict keys.
            # That asymmetry is intentional: `OmniMsgpackDecoder` is type-erased
            # (no target type), so the wire round-trips struct -> dict. If you
            # change the schema, update both ends — see test_wire_round_trip.
            finished_flag = payload_data.meta.finished if payload_data.meta is not None else None
            is_payload_finished = False
            if isinstance(finished_flag, torch.Tensor):
                is_payload_finished = finished_flag.numel() == 1 and bool(finished_flag.item())
            elif finished_flag is not None:
                is_payload_finished = bool(finished_flag)

            # Processor-controlled codec ``finished`` can mark the end of one
            # native-duplex turn while the persistent request remains live.
            # Only the scheduler's whole-request terminal may reclaim shared
            # sender/receiver state; otherwise a late old-turn put can erase a
            # newly admitted turn for the same request.
            if is_payload_finished and is_finished:
                self.cleanup(request.request_id, external_req_id)
        else:
            # R1.2 of #4855. connector.put returning False is a silent drop: no
            # exception, no retry, and the caller's `if success:` block simply
            # does not run. /dev/shm exhaustion in SharedMemoryConnector.put is
            # one way to get here. Record it so the request fails now rather
            # than parking in WAITING_FOR_CHUNK until the deadline.
            logger.error(
                "Chunk send failed for %s (stage %s -> %s); giving up on this chunk",
                external_req_id,
                stage_id,
                next_stage_id,
            )
            # Key on the scheduler-side id. `external_req_id` is the user-facing id
            # (InputProcessor renames request_id to an internal UUID and keeps the
            # original in external_req_id, see async_omni_engine.py), while
            # `self.requests` -- and therefore `finish_requests` -- is keyed by the
            # internal one.
            self.record_send_failure(request.request_id, "connector.put reported failure")

        if is_segment_finished:
            self.code_prompt_token_ids.pop(external_req_id, None)
            self.ramp_chunk_count.pop(external_req_id, None)
            self._adaptive_states.pop(external_req_id, None)
            cached_ic = getattr(self, "_cached_ic", None)
            if cached_ic is not None:
                cached_ic.pop(external_req_id, None)

    def is_done_receiving_chunks(self, request_id: str) -> bool:
        """Return True if the request should stop polling upstream chunks.

        Covers both the whole-request marker (``upstream_exhausted_requests``)
        and the per-segment marker (``segment_finished_requests``) used while
        waiting for the next streaming input slice. Neither means this
        stage's own generation is done -- see vllm-project/vllm-omni#5349.
        """
        return request_id in self.upstream_exhausted_requests or request_id in self.segment_finished_requests

    ########################################################################
    # Cleanup
    ########################################################################

    def cleanup_receiver(self, request_id: str) -> None:
        """Reclaim receiver-side per-request state (keyed by internal id).

        Safe to call from the scheduler while ``connector.get()`` is in
        flight: cleanup and the post-get state commit are serialized, and a
        cancelled request cannot publish the late chunk.

        Must also purge the request from the chunk-parking deques
        (``waiting_for_chunk_waiting_requests`` / ``_running_requests`` /
        ``_held_non_active``): otherwise a caller that calls
        ``restore_queues()`` without ``scheduler_requests`` (e.g. a unit
        test, or any future caller not synced with the scheduler's own
        request-removal timing) would re-admit an already-finished
        request into the visible queue, which ``_promote_active_streams``
        would then FIFO-promote ahead of genuinely-waiting requests. See
        vllm-project/vllm-omni#5349's active-stream-window tests.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        with self._receiver_state_lock:
            self._clear_receiver_state_locked(request_id)

    def _clear_receiver_state_locked(self, request_id: str) -> None:
        """Clear receiver state while ``_receiver_state_lock`` is held."""
        self._active_streams.pop(request_id, None)
        self.upstream_exhausted_requests.discard(request_id)
        self.segment_finished_requests.discard(request_id)
        self.get_req_chunk.pop(request_id, None)
        self.requests_with_ready_chunks.discard(request_id)
        self.replaced_streaming_prompt_ids.discard(request_id)
        self.request_ids_mapping.pop(request_id, None)
        self.requests_origin_status.pop(request_id, None)
        self._pending_ar_prompt_updates.pop(request_id, None)
        self._streaming_condition_lengths.pop(request_id, None)
        self._streaming_condition_seqs.pop(request_id, None)
        self._registered_load_entries.pop(request_id, None)
        self._discard_from_chunk_deque(self.waiting_for_chunk_waiting_requests, request_id)
        self._discard_from_chunk_deque(self.waiting_for_chunk_running_requests, request_id)
        self._discard_from_chunk_deque(self._held_non_active, request_id)

        self._cancelled_load_reqs.add(request_id)
        self._finished_load_reqs.discard(request_id)
        self._waiting_since.pop(request_id, None)

    @staticmethod
    def _discard_from_chunk_deque(deque_list: deque[Any], request_id: str) -> None:
        if not deque_list:
            return
        for _ in range(len(deque_list)):
            request = deque_list.popleft()
            if request.request_id != request_id:
                deque_list.append(request)

    def cleanup_sender(self, external_req_id: str) -> None:
        """Reclaim sender-side per-request state (keyed by external id).

        Called after a terminal chunk is sent or when the scheduler aborts the
        request before a terminal chunk can be produced. In-flight sends are
        cancelled here and reclaim their own state in ``finally``; cleanup
        never waits for connector I/O on the scheduler thread.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        with self._sender_state_lock:
            sender_token = self._sender_tokens.get(external_req_id)
            if sender_token is None:
                self._clear_sender_state_locked(external_req_id)
                return
            sender_token.cancelled = True
            if sender_token.in_flight:
                # The sender's finally block is the first point where no
                # in-flight code can mutate this request's state. Let it own
                # cleanup without making the scheduler wait for connector I/O.
                return
            self._sender_tokens.pop(external_req_id, None)
            self._clear_sender_state_locked(external_req_id)

    def _clear_sender_state_locked(self, external_req_id: str) -> None:
        """Clear sender state while ``_sender_state_lock`` is held."""
        self.put_req_chunk.pop(external_req_id, None)
        self.request_payload.pop(external_req_id, None)
        self.code_prompt_token_ids.pop(external_req_id, None)
        self.requests_num_chunks_sent.pop(external_req_id, None)
        self._segment_generation.pop(external_req_id, None)
        self.ramp_chunk_count.pop(external_req_id, None)
        self._adaptive_states.pop(external_req_id, None)
        self._pending_streaming_prefills.pop(external_req_id, None)

        cached_ic = getattr(self, "_cached_ic", None)
        if cached_ic is not None:
            cached_ic.pop(external_req_id, None)

    def cleanup(
        self,
        request_id: str,
        external_req_id: str | None = None,
    ) -> None:
        """Reclaim all per-request state after a request finishes.

        Idempotent: calling with an already-cleaned or unknown id is safe.

        Args:
            request_id: Internal request id (receive / scheduler side key).
            external_req_id: External request id (send / payload side key).
                When *None*, looked up from ``request_ids_mapping``.
        """
        if external_req_id is None:
            external_req_id = self.request_ids_mapping.get(request_id, request_id)

        self.cleanup_receiver(request_id)
        self.cleanup_sender(external_req_id)

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        *,
        scheduler_requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Process pending chunks for waiting and running queues.

        When ``scheduler_requests`` is provided, purges any
        ``waiting_for_chunk_*_requests`` deque entries whose
        ``request_id`` is no longer tracked by it (e.g. after a
        mid-flight abort that ran ``Scheduler._free_request``) before
        processing chunks. Without this purge, ``restore_queues`` would
        later re-inject the freed ``Request`` onto ``running_queue`` and
        the worker's ``_update_states`` would crash with ``KeyError``
        reading ``self.requests[req_id]``. See vllm-project/vllm-omni#3736.

        ``scheduler_requests`` is keyword-only and optional; production
        schedulers always pass their live request map, while legacy
        callers that don't track aborts may omit it to keep the prior
        (unguarded) behaviour.
        """
        if not self.receives_chunks:
            return
        if self.connector.stage_id == 0:
            return

        # Purge deque entries whose request was freed mid-flight (abort →
        # Scheduler._free_request) before any chunk processing, so neither
        # the legacy nor the active-stream path can re-inject a zombie
        # Request onto the queues. See vllm-project/vllm-omni#3736.
        if scheduler_requests is not None:
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_waiting_requests, scheduler_requests)
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_running_requests, scheduler_requests)

        if self._active_window <= 0:
            self._process_chunk_queue_legacy(
                waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
            )
            self._process_chunk_queue_legacy(
                running_queue,
                self.waiting_for_chunk_running_requests,
                RequestStatus.RUNNING,
                self._finished_load_reqs,
            )
            self._apply_pending_ar_prompt_updates(scheduler_requests)
            self._requeue_replaced_prompts(waiting_queue, running_queue)
            while len(running_queue) > self.scheduler_max_num_seqs:
                request = running_queue.pop()
                request.status = RequestStatus.PREEMPTED
                waiting_queue.prepend_requests([request])
            return

        self._promote_active_streams(running_queue)
        self._promote_active_streams(waiting_queue)
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, self._finished_load_reqs
        )
        self._apply_pending_ar_prompt_updates(scheduler_requests)
        self._requeue_replaced_prompts(waiting_queue, running_queue)
        self._promote_active_streams(waiting_queue)
        self._preempt_non_active_running(waiting_queue, running_queue)

    def _apply_pending_ar_prompt_updates(self, scheduler_requests: dict[str, Request] | None) -> None:
        """Finalize ready AR prompt updates on the scheduler thread."""
        for request_id in tuple(self.requests_with_ready_chunks):
            with self._receiver_state_lock:
                pending = self._pending_ar_prompt_updates.pop(request_id, None)
                if pending is None:
                    continue
                (
                    fallback_request,
                    payload_data,
                    update_prompt,
                    window_condition_len,
                    was_resumable,
                ) = pending
                request = scheduler_requests.get(request_id) if scheduler_requests is not None else fallback_request
                if request is None:
                    continue

                request.additional_information = payload_data
                meta = payload_data.get("meta")
                meta = meta if isinstance(meta, dict) else {}
                previous_condition_seq = self._streaming_condition_seqs.get(request_id)
                condition_seq = previous_condition_seq + 1 if previous_condition_seq is not None else 0
                is_window_condition = window_condition_len is not None
                if is_window_condition:
                    meta["streaming_condition_seq"] = condition_seq
                if is_window_condition:
                    update_prompt = bool(
                        was_resumable
                        and (previous_condition_seq is not None or meta.get("replace_streaming_prompt") is True)
                    )
                try:
                    if update_prompt:
                        replaced = construct_next_stage_streaming_input_prompt(
                            payload_data,
                            request,
                            max_model_len=self._max_model_len,
                            previous_condition_len=self._streaming_condition_lengths.get(request_id),
                            previous_condition_seq=previous_condition_seq,
                            condition_seq=condition_seq,
                            recompute_previous_chunks=self._streaming_prompt_previous_chunks,
                        )
                        if replaced:
                            self.replaced_streaming_prompt_ids.add(request_id)
                except ValueError as exc:
                    # The connector chunk was consumed, so retrying would skip
                    # this transition and desynchronize Talker. Let the
                    # scheduler fail it.
                    self.record_receive_failure(request_id, str(exc))
                    continue

                if window_condition_len is not None:
                    self._streaming_condition_lengths[request_id] = window_condition_len
                    self._streaming_condition_seqs[request_id] = condition_seq

    def _requeue_replaced_prompts(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        """Move a replaced running prompt back through scheduler admission."""
        for request in list(running_queue):
            request_id = request.request_id
            if (
                request_id not in self.replaced_streaming_prompt_ids
                or request_id not in self.requests_with_ready_chunks
            ):
                continue
            running_queue.remove(request)
            request.status = RequestStatus.WAITING
            self.requests_origin_status[request_id] = RequestStatus.WAITING
            waiting_queue.add_request(request)

    def _promote_active_streams(self, queue: Any) -> None:
        if len(self._active_streams) >= self._active_window:
            return
        for request in list(queue):
            if len(self._active_streams) >= self._active_window:
                return
            request_id = request.request_id
            if request_id in self._active_streams:
                continue
            # Iterating the existing queue preserves FIFO admission.
            self._active_streams[request_id] = request

    def _ensure_active_stream(self, request: Request) -> bool:
        if self._active_window <= 0:
            return True
        request_id = request.request_id
        if request_id in self._active_streams:
            self._active_streams[request_id] = request
            return True
        if len(self._active_streams) >= self._active_window:
            return False
        self._active_streams[request_id] = request
        return True

    def collect_timed_out_request_ids(self, timeout_s: float) -> set[str]:
        """Return IDs whose chunk wait has exceeded *timeout_s*.

        The async-chunk path had no deadline of any kind: a request parks in
        ``WAITING_FOR_CHUNK`` and the receiver re-queues it on every failed
        poll (``transfer_adapter/base.py``) with no attempt counter, so a
        producer that crashed, gave up after its send retries, or simply never
        emitted a terminal chunk left the request waiting forever
        (vllm-project/vllm-omni#3833).  The full-payload path has had a net
        since ``OmniSchedulingCoordinator.collect_timed_out_request_ids``; this
        is its async-chunk counterpart, and both are driven by the same
        ``VLLM_OMNI_INPUT_WAIT_TIMEOUT_S``.

        The clock measures *stall* time, not stream lifetime: it starts when a
        request begins waiting for a chunk and resets each time one arrives, so
        a long but healthy stream is never failed.

        Clears ``_waiting_since`` for the expired IDs.  The caller marks them
        ``FINISHED_ERROR`` via the scheduler's ``finish_requests``, which routes
        back through this adapter's own ``finish_requests`` and releases the
        rest of the per-request state.
        """
        if timeout_s <= 0 or not self._waiting_since:
            return set()
        now = time.monotonic()
        timed_out_ids = {req_id for req_id, started in self._waiting_since.items() if now - started > timeout_s}
        for req_id in timed_out_ids:
            self._waiting_since.pop(req_id, None)
            logger.warning(
                "[Stage-%s] Request %s timed out waiting for a chunk (stalled > %.0fs)",
                self.connector.stage_id,
                req_id,
                timeout_s,
            )
        return timed_out_ids

    @property
    def num_running_waiting_for_chunk(self) -> int:
        """Count running requests temporarily removed while awaiting a chunk."""
        return len(self.waiting_for_chunk_running_requests)

    def _preempt_non_active_running(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        # Hold non-active running requests in a private deque rather than
        # routing them back through waiting_queue. Routing through the
        # vllm RequestQueue mid-step triggers
        #   "Cannot register new removed request after self.removed has
        #    been read"
        # in vllm.v1.sample.logits_processor.state when the persistent
        # batch was already snapshotted. They are returned to
        # running_queue in restore_queues() so the next scheduler tick
        # re-evaluates them through _promote_active_streams.
        index = len(running_queue) - 1
        while index >= 0:
            request = running_queue[index]
            if request.request_id in self._active_streams:
                index -= 1
                continue
            request = running_queue.pop(index)
            self._held_non_active.append(request)
            index -= 1

    def _process_chunk_queue_legacy(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if self.is_done_receiving_chunks(request.request_id):
                    request.additional_information = None
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
                self._waiting_since.setdefault(request.request_id, time.monotonic())
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    # A chunk landed: restart the clock for the next one, so the
                    # deadline measures stall time rather than total stream time.
                    self._waiting_since.pop(request.request_id, None)
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _purge_untracked_chunk_requests(
        self,
        deque_list: deque[Any],
        scheduler_requests: dict[str, Request],
    ) -> None:
        """Drop deque entries whose ``request_id`` is not in
        ``scheduler_requests`` and reclaim their receiver-side state.

        Handles requests that were aborted mid-flight while parked in a
        chunk-transfer deque: ``Scheduler._free_request`` deleted the
        entry from ``scheduler.requests`` but the deque still holds a
        reference to the now-freed ``Request``. Order of survivors is
        preserved.
        """
        if not deque_list:
            return
        for _ in range(len(deque_list)):
            request = deque_list.popleft()
            if request.request_id in scheduler_requests:
                deque_list.append(request)
            else:
                self.cleanup_receiver(request.request_id)

    def restore_queues(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        scheduler_requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.

        Re-runs the zombie purge first to close the race window where an
        abort fires *between* ``process_pending_chunks`` and the
        ``finally``-clause ``restore_queues`` call. Without the second
        purge, ``running_queue.extend(...)`` would still re-inject a
        freed ``Request`` and crash the worker on the next tick.

        ``scheduler_requests`` is optional for back-compat with legacy
        callers (older tests pass only the two queue arguments). When
        provided, it gates both the deque purge and the per-request
        admit checks below; when ``None``, the purge is skipped and
        every parked request is restored unconditionally (the
        pre-purge behavior).
        """
        if not self.receives_chunks:
            return
        if scheduler_requests is not None:
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_waiting_requests, scheduler_requests)
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_running_requests, scheduler_requests)
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            if scheduler_requests is None or request.request_id in scheduler_requests:
                waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            live_running_requests = [
                request
                for request in self.waiting_for_chunk_running_requests
                if scheduler_requests is None or request.request_id in scheduler_requests
            ]
            running_queue.extend(live_running_requests)
        self.waiting_for_chunk_running_requests = deque()

        if self._held_non_active:
            running_queue.extend(self._held_non_active)
            self._held_non_active = deque()

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        if not self.receives_chunks:
            return
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return

        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info
            if request and additional_info:
                request.additional_information = None

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if not self._ensure_active_stream(request):
                if target_status == RequestStatus.WAITING:
                    # A non-active placeholder must not remain visible to the
                    # scheduler: it has no connector payload yet, so running
                    # it would execute the downstream model with empty
                    # additional_information. Park it until restore_queues()
                    # and retry admission on the next scheduler tick.
                    queue.remove(request)
                    waiting_for_chunk_list.append(request)
                continue
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if self.is_done_receiving_chunks(request.request_id):
                    request.additional_information = None
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
                self._waiting_since.setdefault(request.request_id, time.monotonic())
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    # A chunk landed: restart the clock for the next one, so the
                    # deadline measures stall time rather than total stream time.
                    self._waiting_since.pop(request.request_id, None)
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)
                if req_data.req_id in self.replaced_streaming_prompt_ids:
                    external_req_id = self.request_ids_mapping.get(req_data.req_id)
                    if external_req_id is not None:
                        self.requests_num_chunks_sent.pop(external_req_id, None)
                    self.replaced_streaming_prompt_ids.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)

    def finish_requests(
        self, request_ids: Any, finished_status: RequestStatus, requests: dict[str, Request] | None = None
    ) -> list[tuple[str, int]]:
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = requests.keys()

        connector_owned_ids = {
            request.request_id
            for queue in (
                self.waiting_for_chunk_waiting_requests,
                self.waiting_for_chunk_running_requests,
                self._held_non_active,
            )
            for request in queue
        }

        # First pass: collect requests to remove from queues
        request_ids = set(request_ids)
        for req_id in request_ids:
            request = requests.get(req_id) if requests else None
            if request is None:
                # Invalid request ID.
                continue
            resumable_segment_stop = bool(
                getattr(request, "resumable", False) and request.status == RequestStatus.FINISHED_STOPPED
            )
            if request.is_finished() and not resumable_segment_stop:
                continue
            # Once restored to a scheduler queue, the saved origin is stale and
            # must not overwrite statuses such as WAITING_FOR_STREAMING_REQ.
            if req_id in self.requests_origin_status and req_id in connector_owned_ids:
                request.status = self.requests_origin_status.pop(req_id)

        # An abort can terminate a long-lived native codec stream before it
        # emits a terminal payload. Reclaim both sides of the adapter so a
        # later request reusing the same external id starts with empty state.
        for req_id in request_ids:
            request = requests.get(req_id) if requests else None
            external_req_id = (
                getattr(request, "external_req_id", None)
                if request is not None
                else self.request_ids_mapping.get(req_id)
            ) or req_id
            self.cleanup_receiver(req_id)
            self.cleanup_sender(external_req_id)

        self.waiting_for_chunk_waiting_requests = deque(
            request for request in self.waiting_for_chunk_waiting_requests if request.request_id not in request_ids
        )
        self.waiting_for_chunk_running_requests = deque(
            request for request in self.waiting_for_chunk_running_requests if request.request_id not in request_ids
        )
        self._held_non_active = deque(
            request for request in self._held_non_active if request.request_id not in request_ids
        )

        for req_id in request_ids:
            self.cleanup_receiver(req_id)

        return []
