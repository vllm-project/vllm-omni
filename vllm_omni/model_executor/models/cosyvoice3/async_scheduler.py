# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Engine-owned routing for host-ready codec output; no GPU waits here."""

from __future__ import annotations

import queue
from dataclasses import dataclass

import torch
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs


@dataclass
class AudioRequestState:
    client_index: int
    terminal: OmniEngineCoreOutput | None = None
    done: bool = False
    cancelled: bool = False


class CosyVoice3AsyncScheduler(OmniARAsyncScheduler):
    """Keep client completion until both AR termination and final PCM arrive.

    Audio routes outlive AR slots. Terminals and audio completion are
    tracked separately because either may arrive first; cancellation fences
    already queued PCM until the base scheduler produces its terminal output.
    """

    def __init__(self, *args, **kwargs):
        self.codec = None
        self.audio_requests: dict[str, AudioRequestState] = {}
        self.abort_outputs = []
        super().__init__(*args, **kwargs)

    def schedule(self, throttle_prefills=False):
        if self.codec is None:
            raise RuntimeError("CosyVoice3 async codec requires an in-process UniProc executor")
        self.codec.check_error()
        # Finished AR requests keep their codec capacity until final PCM is
        # consumed. Preserve capacity for active streams instead of admitting
        # an unbounded succession of detached tails.
        original = self.max_num_active_reqs
        detached = sum(state.terminal is not None for state in self.audio_requests.values())
        self.max_num_active_reqs = max(0, original - detached)
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self.max_num_active_reqs = original
        # Restarting a partially emitted waveform on KV preemption is unsafe.
        if output.preempted_req_ids:
            self.finish_requests(output.preempted_req_ids, RequestStatus.FINISHED_ERROR)
        return output

    def retire_request(self, req_id):
        """Finish client/codec bookkeeping; AR resources are owned by the base scheduler."""
        self.audio_requests.pop(req_id, None)
        self.codec.release(req_id)

    def finish_requests(self, request_ids, finished_status):
        if request_ids is not None and not isinstance(request_ids, str):
            request_ids = list(request_ids)
        ids = (
            set(self.audio_requests)
            if request_ids is None
            else ({request_ids} if isinstance(request_ids, str) else set(request_ids))
        )
        for req_id in ids & self.audio_requests.keys():
            self.codec.cancel(req_id)
            state = self.audio_requests[req_id]
            state.cancelled = True
            if state.terminal is not None:
                output = state.terminal
                output.finish_reason = (
                    FinishReason.ABORT if finished_status == RequestStatus.FINISHED_ABORTED else FinishReason.ERROR
                )
                self.abort_outputs.append((state.client_index, output))
                self.retire_request(req_id)
        return super().finish_requests(request_ids, finished_status)

    def get_num_unfinished_requests(self):
        return super().get_num_unfinished_requests() + sum(
            state.terminal is not None for state in self.audio_requests.values()
        )

    def add_request(self, request):
        if request.request_id in self.audio_requests:
            raise ValueError("Request ID is still draining codec audio")
        super().add_request(request)
        if request.request_id in self.requests:
            self.audio_requests[request.request_id] = AudioRequestState(request.client_index)

    def has_requests(self):
        return bool(self.audio_requests or self.abort_outputs) or super().has_requests()

    def has_unfinished_requests(self):
        return bool(self.audio_requests or self.abort_outputs) or super().has_unfinished_requests()

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        self.codec.check_error()
        # Retain normal stop metadata/tokens, while freeing AR/KV through the
        # unchanged base algorithm. Only the client terminal is deferred.
        for client_output in result.values():
            kept = []
            for output in client_output.outputs:
                req_id = output.request_id
                if req_id in self.audio_requests and output.finish_reason is not None:
                    if output.finish_reason in (FinishReason.ABORT, FinishReason.ERROR):
                        self.codec.cancel(req_id)
                        self.retire_request(req_id)
                        kept.append(output)
                    else:
                        self.audio_requests[req_id].terminal = output
                else:
                    kept.append(output)
            client_output.outputs = kept
            if client_output.finished_requests:
                client_output.finished_requests.difference_update(self.audio_requests)
        for client, output in self.abort_outputs:
            batch = result.setdefault(client, OmniEngineCoreOutputs(outputs=[]))
            batch.outputs.append(output)
            if batch.finished_requests is None:
                batch.finished_requests = set()
            batch.finished_requests.add(output.request_id)
        self.abort_outputs.clear()
        ready: dict[str, list[torch.Tensor]] = {}
        while True:
            try:
                audio = self.codec.outputs.get_nowait()
            except queue.Empty:
                break
            req_id = audio.request_id
            state = self.audio_requests.get(req_id)
            if state is None or state.cancelled:
                continue
            if audio.pcm is not None and audio.pcm.numel():
                ready.setdefault(req_id, []).append(audio.pcm)
            if audio.final:
                state.done = True
        completed = {key for key, state in self.audio_requests.items() if state.done and state.terminal is not None}
        for req_id in ready.keys() | completed:
            state = self.audio_requests[req_id]
            client = state.client_index
            batch = result.setdefault(client, OmniEngineCoreOutputs(outputs=[]))
            final = state.done and state.terminal is not None
            output = state.terminal if final else OmniEngineCoreOutput(request_id=req_id, new_token_ids=[])
            pieces = ready.get(req_id)
            if pieces:
                output.multimodal_output = {
                    "audio": pieces[0] if len(pieces) == 1 else torch.cat(pieces),
                    "sr": torch.tensor(self.audio_sample_rate, dtype=torch.int32),
                }
            existing = next((o for o in batch.outputs if o.request_id == req_id), None)
            if existing is not None:
                output.new_token_ids = existing.new_token_ids + output.new_token_ids
                batch.outputs.remove(existing)
            batch.outputs.append(output)
            if final:
                if batch.finished_requests is None:
                    batch.finished_requests = set()
                batch.finished_requests.add(req_id)
                self.retire_request(req_id)
        return result
