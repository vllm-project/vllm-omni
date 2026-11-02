# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep MOSS codec tails alive after the talker releases its KV blocks."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.engine import OmniEngineCoreOutput


class MossFusedScheduler(OmniARAsyncScheduler):
    """Selected only by the single-stage MOSS Local pipeline.

    The base scheduler owns generation, KV blocks and ordinary outputs. This
    scheduler holds terminal outputs until the in-process codec finishes, and
    leases capacity for detached codec state until its tail has been emitted.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.omni_deferred_finishes: dict[str, dict[str, Any]] = {}
        self.omni_tail_outputs: deque[tuple[str, dict[str, Any] | None, bool, str | None]] = deque()

    def bind_model_executor(self, executor: UniProcExecutor) -> None:
        parallel = executor.vllm_config.parallel_config
        if (
            not isinstance(executor, UniProcExecutor)
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
        ):
            raise ValueError("MOSS fused audio requires an in-process TP1/PP1 executor")
        state = executor.driver_worker.worker.model_runner.model_state
        if not state.set_tail_audio_sender(self):
            raise ValueError("MOSS fused scheduler requires a fused codec model state")

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        capacity = self.max_num_active_reqs
        self.max_num_active_reqs = max(0, capacity - len(self.omni_deferred_finishes))
        try:
            return super().schedule(throttle_prefills)
        finally:
            self.max_num_active_reqs = capacity

    def finish_requests(self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus):
        if request_ids is None:
            deferred_ids = tuple(self.omni_deferred_finishes)
        else:
            request_ids = (request_ids,) if isinstance(request_ids, str) else tuple(request_ids)
            deferred_ids = self.omni_deferred_finishes.keys() & request_ids
        for req_id in deferred_ids:
            self.omni_deferred_finishes.pop(req_id)
            self.finished_req_ids.add(req_id)
        return super().finish_requests(request_ids, finished_status)

    def _preempt_request(self, request: Request, timestamp: float, drop_stale_output: bool = False) -> None:
        super()._preempt_request(request, timestamp, drop_stale_output)
        # Recomputing the talker cannot recreate an already-streamed codec prefix.
        self.waiting.remove_requests([request])
        request.status = RequestStatus.FINISHED_ERROR
        request.resumable = False
        request.stop_reason = "Fused audio cannot resume after KV-cache preemption"
        self._free_request(request)
        self.omni_deferred_finishes[request.request_id] = {
            "client_index": request.client_index,
            "finish_reason": FinishReason.ERROR,
            "stop_reason": request.stop_reason,
        }
        self.omni_tail_outputs.append((request.request_id, None, True, request.stop_reason))

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]:
        detached = tuple(self.omni_deferred_finishes)
        result = super().update_from_output(scheduler_output, model_runner_output)
        for client_index, batch in result.items():
            for output in batch.outputs:
                if output.finish_reason in (FinishReason.STOP, FinishReason.LENGTH):
                    self.omni_deferred_finishes[output.request_id] = {
                        "client_index": client_index,
                        "finish_reason": output.finish_reason,
                        "stop_reason": output.stop_reason,
                    }
                    output.finish_reason = None
                    output.stop_reason = None

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        mm_outputs = getattr(model_runner_output, "multimodal_outputs", None)
        if mm_outputs:
            for req_id in detached:
                self._forward_deferred_audio(outputs, req_id, mm_outputs, model_runner_output)
        # Preserve the order: ordinary step, in-flight step after stop, codec tail.
        self._emit_tail_outputs(outputs)
        for client_index, items in outputs.items():
            result.setdefault(client_index, EngineCoreOutputs()).outputs.extend(items)
        return result

    def _forward_deferred_audio(
        self,
        outputs: dict[int, list[EngineCoreOutput]],
        req_id: str,
        mm_outputs: list[Any],
        model_runner_output: ModelRunnerOutput,
    ) -> None:
        held = self.omni_deferred_finishes.get(req_id)
        index = model_runner_output.req_id_to_index.get(req_id)
        if held is None or index is None or index >= len(mm_outputs) or not mm_outputs[index]:
            return
        outputs[held["client_index"]].append(
            OmniEngineCoreOutput(request_id=req_id, new_token_ids=[], multimodal_output=mm_outputs[index])
        )

    def _emit_tail_outputs(self, outputs: dict[int, list[EngineCoreOutput]]) -> None:
        while self.omni_tail_outputs:
            req_id, payload, final, error = self.omni_tail_outputs.popleft()
            held = self.omni_deferred_finishes.get(req_id)
            if held is None or (payload is None and not final):
                continue
            outputs[held["client_index"]].append(
                OmniEngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=[],
                    multimodal_output=payload,
                    finish_reason=FinishReason.ERROR if error is not None else held["finish_reason"] if final else None,
                    stop_reason=error if error is not None else held["stop_reason"] if final else None,
                )
            )
            if final:
                self.omni_deferred_finishes.pop(req_id)
                if error is not None:
                    self.finished_req_ids.add(req_id)

    def has_requests(self) -> bool:
        return bool(self.omni_deferred_finishes) or super().has_requests()
