"""Auto-Regressive stage strategy."""

from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

from .base import ReceiveOutput, StageStrategy

logger = init_logger(__name__)


class ARStrategy(StageStrategy):
    """Strategy for AR stages (Thinker, Talker)."""

    def should_send_chunk(self, request, pooler_output, token_ids, processor, state_manager) -> bool:
        req_id = state_manager.get_global_request_id(request.request_id)
        state = state_manager.get_state(req_id)

        # Skip during prefill
        if processor.is_prefill(pooler_output, request):
            return False

        # Skip first chunk (orchestrator handles it)
        if processor.should_skip_first_chunk and state.first_chunk_after_prefill:
            state.first_chunk_after_prefill = False
            return False

        return True

    def receive_chunk(self, active_requests, connector, stage_id, processor, state_manager) -> ReceiveOutput:
        stopped_running = set()
        stopped_preempted = set()
        received_count = 0
        prev_stage = stage_id - 1

        for request in active_requests:
            req_id = state_manager.get_global_request_id(request.request_id)
            state = state_manager.get_state(req_id)

            # This means it is in prefill-mode
            # Shouldn't receive chunks during prefill mode
            if len(request.prompt_token_ids) > request.num_computed_tokens:
                continue

            # AR stages: when upstream finishes, just keep running
            # (AR stages like Talker continue independently after Thinker is done)
            if state.upstream_finished:
                request.status = RequestStatus.RUNNING
                continue

            # Skip if pending chunk exists and not yet consumed
            if getattr(request, "pending_chunk", None):
                request.status = RequestStatus.RUNNING
                continue

            # Try to retrieve chunk
            key = self.prepare_connector_key(state.received_chunks, prev_stage, req_id)
            chunk = connector.get_chunk(str(prev_stage), str(stage_id), key)

            if chunk and chunk[0]:
                payload, _ = chunk
                if payload:
                    # Apply incoming chunk using processor logic
                    # processor.apply_incoming_chunk expects request_state, passing None for now
                    # as it's typically used for cached data which might be on the request itself
                    processor.apply_incoming_chunk(payload, request, None)

                    state_manager.increment_chunk_received(req_id)
                    received_count += 1
                    request.status = RequestStatus.RUNNING

                    if payload.get("last_chunk", False):
                        state_manager.mark_upstream_finished(req_id)
            else:
                request.status = RequestStatus.WAITING_FOR_CHUNK

        return ReceiveOutput(stopped_running, stopped_preempted, received_count)
