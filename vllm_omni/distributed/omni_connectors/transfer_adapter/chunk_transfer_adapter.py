# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)


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
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.connector = self.create_connector(model_config)
        super().__init__(model_config)
        self.model_mode = getattr(model_config, "worker_type", "ar")
        # State specific to Chunk management
        self.custom_process_next_stage_input_func = None
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()

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
        request_id = request.request_id
        self.request_ids_mapping[request_id] = request.external_req_id

        if stage_id == 0:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        with self.lock:
            self._pending_load_reqs[request_id] = request

    def save_async(self, pooling_output: torch.Tensor | None = None, request: Request | None = None):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction is executed in the caller thread via
        ``custom_process_next_stage_input_func``

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
        """
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        request_id = request.external_req_id
        chunk_id = self.put_req_chunk[request_id]

        # Process payload in main thread to avoid race conditions on request state
        payload_data = None
        if self.custom_process_next_stage_input_func:
            try:
                payload_data = self.custom_process_next_stage_input_func(
                    transfer_manager=self,
                    pooling_output=pooling_output,
                    request=request,
                )

            except Exception as e:
                logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

        if not payload_data:
            return

        # Increment chunk_id
        self.put_req_chunk[request_id] += 1
        connector_put_key = f"{request_id}_{stage_id}_{chunk_id}"

        task = {
            "stage_id": stage_id,
            "next_stage_id": next_stage_id,
            "put_key": connector_put_key,
            "data": payload_data,
            "request_id": request_id,
        }

        with self.lock:
            if request_id not in self._pending_save_reqs:
                self._pending_save_reqs[request_id] = deque()
            self._pending_save_reqs[request_id].append(task)

    def _poll_single_request(self, req_id: str):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        chunk_id = self.get_req_chunk[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        result = self.connector.get(
            str(target_stage_id),
            str(stage_id),
            connector_get_key,
        )

        if result is None:
            return

        payload_data, size = result

        if payload_data:
            # Update connector state
            self.get_req_chunk[req_id] += 1
            req = self._pending_load_reqs[req_id]

            if self.model_mode == "ar":
                self._update_request_payload(external_req_id, payload_data)
                req.additional_information = payload_data
                if payload_data.get("finished"):
                    self.finished_requests.add(req_id)
            else:
                if payload_data.get("finished"):
                    self.finished_requests.add(req_id)
                    req.status = RequestStatus.FINISHED_STOPPED

                req.prompt_token_ids = payload_data.get("code_predictor_codes", [])
                req.num_computed_tokens = 0

            # Mark as finished for consumption
            with self.lock:
                self._finished_load_reqs.add(req_id)
                if req_id in self._pending_load_reqs:
                    del self._pending_load_reqs[req_id]
            logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")

    def _update_request_payload(self, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
        """Update the payload data for a request in the connector.

        Args:
            connector: OmniConnectorBase instance
            req_id: Request ID to update
            payload_data: New payload data to store
        """
        if req_id not in self.request_payload:
            self.request_payload[req_id] = payload_data
            return payload_data
        origin_payload = self.request_payload[req_id]
        for key, value in payload_data.items():
            if key == "finished":
                continue
            elif isinstance(value, torch.Tensor) and key in origin_payload:
                payload_data[key] = torch.cat([origin_payload[key], value], dim=0)
            elif isinstance(value, list) and key in origin_payload:
                payload_data[key] = origin_payload[key] + value

        self.request_payload[req_id] = payload_data
        return payload_data

    def _send_single_request(self, task: dict):
        connector_put_key = task["put_key"]
        stage_id = task["stage_id"]
        next_stage_id = task["next_stage_id"]
        payload_data = task["data"]

        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )

        if success:
            logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> None:
        """
        Process pending chunks for waiting and running queues.
        """
        if self.connector.stage_id == 0:
            return
        finished_load_reqs = self.get_finished_requests()
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, finished_load_reqs
        )
        while len(running_queue) > self.scheduler_max_num_seqs:
            request = running_queue.pop()
            waiting_queue.prepend_requests([request])

    def restore_queues(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.
        """
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            running_queue.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
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

    def _process_chunk_queue(
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
                if request.request_id in self.finished_requests:
                    request.additional_information = {}
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)

    def get_finished_requests(self):
        with self.lock:
            finished_load = set(self._finished_load_reqs)
            self._finished_load_reqs = set()
        return finished_load
