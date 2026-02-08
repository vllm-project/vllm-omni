"""Generation stage strategy."""

from vllm.v1.request import RequestStatus

from .base import ReceiveOutput, StageStrategy


class GenerationStrategy(StageStrategy):
    """Strategy for Generation stages (Code2Wav)."""

    def should_send_chunk(self, request, pooler_output, token_ids, processor, state_manager) -> bool:
        # Generation stages typically don't send chunks downstream
        # Code2Wav consumes chunks to produce audio, but doesn't send "chunks" to another vLLM stage
        # (Audio output handling is separate)
        # If a new model in future requires to send chunks, this can be implemented then
        return False

    def receive_chunk(self, active_requests, connector, stage_id, processor, state_manager) -> ReceiveOutput:
        stopped_running = set()
        stopped_preempted = set()
        received_count = 0
        prev_stage = stage_id - 1

        for request in active_requests:
            req_id = state_manager.get_global_request_id(request.request_id)
            state = state_manager.get_state(req_id)

            # If the model_executor is slow and hasn't consumed the chunk yet
            pending = getattr(request, "pending_chunk", None)
            if pending is not None and len(pending) > 0:
                request.status = RequestStatus.RUNNING
                continue

            if state.upstream_finished:
                # If upstream is finished, mark request as finished
                should_stop = processor.on_upstream_finished(request)
                if should_stop:
                    # Capture current status before changing
                    if request.status == RequestStatus.RUNNING:
                        stopped_running.add(request)
                    else:
                        stopped_preempted.add(request)

                    request.status = RequestStatus.FINISHED_STOPPED
                continue

            # Try to retrieve chunk
            key = self.prepare_chunk_key(state.received_chunks, prev_stage, req_id)
            chunk = connector.get_chunk(key, req_id)

            if chunk and chunk[0]:
                payload, _ = chunk
                if payload:
                    processor.apply_incoming_chunk(payload, request, None)
                    state_manager.increment_chunk_received(req_id)

                    # For generation stages (Code2Wav), receiving a chunk often means
                    # we start a new generation step from scratch or continue.
                    # As per original OmniGenerationScheduler:
                    request.num_computed_tokens = 0
                    request.status = RequestStatus.RUNNING
                    received_count += 1

                    if payload.get("last_chunk", False):
                        state_manager.mark_upstream_finished(req_id)
            else:
                request.status = RequestStatus.WAITING_FOR_CHUNK

        return ReceiveOutput(stopped_running, stopped_preempted, received_count)
