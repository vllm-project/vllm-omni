from __future__ import annotations

from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus, StreamingUpdate


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    def _free_input_coordinator_request(self, request_id: str) -> None:
        """Prune full-payload coordinator state for a completed request."""
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            input_coordinator.free_finished_request(request_id)

    def _replace_session_with_streaming_update(
        self,
        session: Request,
        update: StreamingUpdate,
    ) -> None:
        """For streaming input: Replace an existing streaming session payload with the latest update."""
        session._output_token_ids.clear()
        session._all_token_ids.clear()
        new_prompt = update.prompt_token_ids or ()
        session._all_token_ids.extend(new_prompt)
        session.num_computed_tokens = 0
        session.prompt_token_ids = update.prompt_token_ids or ()
        session.additional_information = update.additional_information or None
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)
