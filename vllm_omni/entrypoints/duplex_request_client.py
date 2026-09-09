# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, MutableMapping
from typing import Protocol
from weakref import WeakSet

from vllm_omni.engine.duplex.contracts import (
    duplex_data_plane_request_info,
    duplex_resource_request_generation,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.lease import DuplexLeaseActivity
from vllm_omni.engine.duplex.messages import DuplexFence
from vllm_omni.engine.messages import ErrorMessage, OutputMessage
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.duplex import get_duplex_output_decision

_RETIRED_DUPLEX_ROUTE = object()


class DuplexEnginePort(Protocol):
    """Engine control methods required by the experimental request client.

    The protocol keeps request preregistration and output collection testable
    without coupling this component to ``AsyncOmniEngine``.
    """

    async def open_duplex_session_async(self, session_id: str, **kwargs) -> dict[str, object]: ...

    async def append_duplex_input_async(self, session_id: str, **kwargs) -> dict[str, object]: ...

    async def signal_duplex_turn_async(self, session_id: str, **kwargs) -> dict[str, object]: ...

    async def close_duplex_session_async(self, session_id: str, **kwargs) -> dict[str, object]: ...

    async def touch_duplex_session_async(self, session_id: str, **kwargs) -> dict[str, object]: ...

    async def resume_duplex_session_async(self, session_id: str, **kwargs) -> dict[str, object]: ...


class DuplexRequestOutputPort:
    """Narrow output-processing surface owned by the AsyncOmni entrypoint."""

    def __init__(
        self,
        *,
        request_states: MutableMapping[str, ClientRequestState],
        num_stages: int,
        log_stats: bool,
        start_output_handler: Callable[[], object],
        process_single_result: Callable[..., OmniRequestOutput | None],
    ) -> None:
        self.request_states = request_states
        self.num_stages = num_stages
        self.log_stats = log_stats
        self._start_output_handler = start_output_handler
        self._process_single_result = process_single_result

    def start_output_handler(self) -> None:
        self._start_output_handler()

    def process_single_result(self, *args, **kwargs) -> OmniRequestOutput | None:
        return self._process_single_result(*args, **kwargs)


class DuplexRequestClient:
    """Owns duplex request identity, preregistration and output collection."""

    def __init__(self, engine: DuplexEnginePort, output_port: DuplexRequestOutputPort) -> None:
        self.engine = engine
        self.output_port = output_port
        self._resource_generations: dict[tuple[str, int, int], int] = {}
        self._append_locks: dict[tuple[str, int, int], asyncio.Lock] = {}
        # Retired routes can still be held by collectors. Weak membership
        # fences those readers without retaining one tombstone per session.
        self._retired_output_routes: WeakSet[ClientRequestState] = WeakSet()

    @staticmethod
    def request_info(result: dict[str, object]) -> tuple[str | None, int | None]:
        return duplex_data_plane_request_info(result)

    @staticmethod
    def _resource_key(fence: DuplexFence) -> tuple[str, int, int]:
        return fence.session_id, fence.incarnation, fence.epoch

    @staticmethod
    def _physical_request_id(fence: DuplexFence, generation: int) -> str:
        role = "stage0" if generation == 0 else f"stage0g{generation}"
        return duplex_resource_request_id(fence, role)

    def _ensure_output_route(self, request_id: str) -> tuple[ClientRequestState, bool]:
        request_state = self.output_port.request_states.get(request_id)
        created = request_state is None
        if request_state is None:
            request_state = ClientRequestState(request_id)
            self.output_port.request_states[request_id] = request_state
        if request_state.metrics is None:
            wall_start_ts = time.time()
            request_state.metrics = OrchestratorMetrics(
                self.output_port.num_stages,
                self.output_port.log_stats,
                wall_start_ts,
                max(0, self.output_port.num_stages - 1),
            )
            request_state.request_arrival_ts = wall_start_ts
        return request_state, created

    def _remove_output_route(self, request_id: str, request_state: ClientRequestState) -> None:
        if self.output_port.request_states.get(request_id) is request_state:
            self.output_port.request_states.pop(request_id)
            self._retired_output_routes.add(request_state)
            try:
                request_state.queue.put_nowait(_RETIRED_DUPLEX_ROUTE)
            except asyncio.QueueFull:
                # A full queue cannot have a blocked get; its next reader
                # observes retirement before processing the queued output.
                pass

    def _clear_fence_routes(self, fence: DuplexFence) -> None:
        for request_id in tuple(self.output_port.request_states):
            if duplex_resource_request_generation(request_id, fence, "stage0") is not None:
                self._remove_output_route(request_id, self.output_port.request_states[request_id])
        key = self._resource_key(fence)
        self._resource_generations.pop(key, None)
        self._append_locks.pop(key, None)

    async def open(
        self,
        session_id: str,
        *,
        session_mode: str,
        capabilities: dict[str, object] | None,
        session_config: dict[str, object] | None,
        runtime_config: dict[str, object] | None,
        fence: DuplexFence,
        timeout: float | None,
    ) -> dict[str, object]:
        # Lifecycle expiry messages share the engine output channel. Start its
        # sole dispatcher before open so an idle session can be reaped even if
        # it never submits an append request.
        self.output_port.start_output_handler()
        kwargs: dict[str, object] = {
            "session_mode": session_mode,
            "capabilities": capabilities,
            "session_config": session_config,
            "fence": fence,
            "timeout": timeout,
        }
        if runtime_config is not None:
            kwargs["runtime_config"] = runtime_config
        result = await self.engine.open_duplex_session_async(session_id, **kwargs)
        self._resource_generations.setdefault(self._resource_key(fence), 0)
        return result

    async def append(
        self,
        session_id: str,
        *,
        mode: str,
        payload: object,
        operation_id: str | None,
        final: bool,
        expected_epoch: int | None,
        fence: DuplexFence,
        timeout: float | None,
        collect_outputs: bool,
    ) -> dict[str, object]:
        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs: dict[str, object] = {
            "mode": mode,
            "payload": payload,
            "final": final,
            "expected_epoch": expected_epoch,
            "timeout": timeout,
            "fence": fence,
        }
        if operation_id is not None:
            kwargs["operation_id"] = operation_id

        key = self._resource_key(fence)
        append_lock = self._append_locks.setdefault(key, asyncio.Lock())
        if deadline is None:
            await append_lock.acquire()
        else:
            await asyncio.wait_for(append_lock.acquire(), timeout=max(0.0, deadline - time.monotonic()))
        try:
            self._require_append_owner(key, append_lock)
            if deadline is not None:
                kwargs["timeout"] = max(0.0, deadline - time.monotonic())
                if kwargs["timeout"] == 0.0:
                    raise TimeoutError("duplex append deadline expired before submission")
            current_generation = self._resource_generations.get(key, 0)
            candidate_generations = (current_generation, current_generation + 1)
            routes: dict[int, tuple[str, ClientRequestState, bool]] = {}
            for generation in candidate_generations:
                candidate_request_id = self._physical_request_id(fence, generation)
                request_state, created = self._ensure_output_route(candidate_request_id)
                routes[generation] = (candidate_request_id, request_state, created)
            self.output_port.start_output_handler()
            try:
                result = await self.engine.append_duplex_input_async(session_id, **kwargs)
            except BaseException as exc:
                # A failed KV rebuild consumed the attempted physical identity;
                # the engine never reuses it, so the following retry must be
                # pre-registered for the next generation.  A timeout or an
                # explicitly uncertain operation may already have emitted, so
                # retain both aliases until retry/cancel/close.  Definite
                # failures discard only routes created by this attempt.
                error_code = getattr(exc, "code", None)
                if error_code == "kv_recovery_failed" and self._append_locks.get(key) is append_lock:
                    self._resource_generations[key] = current_generation + 1
                    for candidate_request_id, request_state, _ in routes.values():
                        self._remove_output_route(candidate_request_id, request_state)
                elif not isinstance(exc, TimeoutError) and error_code not in {"timeout", "uncertain_operation"}:
                    for candidate_request_id, request_state, created in routes.values():
                        if created and request_state.queue.empty():
                            self._remove_output_route(candidate_request_id, request_state)
                raise

            self._require_append_owner(key, append_lock)
            request_id, response_stage_id = duplex_data_plane_request_info(result)
            if request_id is None:
                for candidate_request_id, request_state, created in routes.values():
                    if created and request_state.queue.empty():
                        self._remove_output_route(candidate_request_id, request_state)
                return result

            returned_generation = duplex_resource_request_generation(request_id, fence, "stage0")
            if returned_generation is None or returned_generation not in candidate_generations:
                for candidate_request_id, request_state, created in routes.values():
                    if created and request_state.queue.empty():
                        self._remove_output_route(candidate_request_id, request_state)
                expected_ids = tuple(item[0] for item in routes.values())
                raise RuntimeError(
                    f"duplex data-plane request id mismatch: expected one of {expected_ids!r}, got {request_id!r}"
                )

            # No await between validating and publishing the generation: an
            # append for this fence either sees the old or the new identity.
            self._resource_generations[key] = returned_generation
            _, request_state, _ = routes[returned_generation]
            for candidate_generation, (candidate_request_id, candidate_state, _) in routes.items():
                if candidate_generation != returned_generation:
                    self._remove_output_route(candidate_request_id, candidate_state)
            if not collect_outputs:
                return result
            outputs = await self.collect_outputs(
                request_id,
                request_state,
                response_stage_id=response_stage_id,
                timeout=None if deadline is None else max(0.0, deadline - time.monotonic()),
            )
            self._require_append_owner(key, append_lock)
            if outputs:
                result = dict(result)
                result["data_plane_outputs"] = outputs
            return result
        finally:
            append_lock.release()

    def _require_append_owner(self, key: tuple[str, int, int], owner: asyncio.Lock) -> None:
        if self._append_locks.get(key) is not owner:
            raise RuntimeError("duplex append superseded by a successful close or cancel")

    async def collect_registered_outputs(
        self,
        request_id: str,
        *,
        response_stage_id: int | None,
        timeout: float | None,
    ) -> list[OmniRequestOutput]:
        self.output_port.start_output_handler()
        request_state = self.output_port.request_states.get(request_id)
        if request_state is None:
            return []
        return await self.collect_outputs(
            request_id,
            request_state,
            response_stage_id=response_stage_id,
            timeout=timeout,
        )

    async def signal(
        self,
        session_id: str,
        *,
        event: str,
        fence: DuplexFence,
        next_fence: DuplexFence | None,
        session_config: dict[str, object] | None,
        runtime_config: dict[str, object] | None,
        timeout: float | None,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {"event": event, "fence": fence, "timeout": timeout}
        if next_fence is not None:
            kwargs["next_fence"] = next_fence
        if session_config is not None:
            kwargs["session_config"] = session_config
        if runtime_config is not None:
            kwargs["runtime_config"] = runtime_config
        result = await self.engine.signal_duplex_turn_async(session_id, **kwargs)
        if event in {"barge_in", "input.cancel", "response.cancel"}:
            self._clear_fence_routes(fence)
        return result

    async def close(
        self,
        session_id: str,
        *,
        reason: str,
        fence: DuplexFence,
        timeout: float | None,
    ) -> dict[str, object]:
        result = await self.engine.close_duplex_session_async(
            session_id,
            reason=reason,
            fence=fence,
            timeout=timeout,
        )
        self._clear_fence_routes(fence)
        return result

    async def touch(
        self,
        session_id: str,
        *,
        fence: DuplexFence,
        activity: DuplexLeaseActivity,
        timeout: float | None,
    ) -> dict[str, object]:
        return await self.engine.touch_duplex_session_async(
            session_id,
            fence=fence,
            activity=activity,
            timeout=timeout,
        )

    async def resume(
        self,
        session_id: str,
        *,
        fence: DuplexFence,
        expected_lease_generation: int,
        timeout: float | None,
    ) -> dict[str, object]:
        return await self.engine.resume_duplex_session_async(
            session_id,
            fence=fence,
            expected_lease_generation=expected_lease_generation,
            timeout=timeout,
        )

    async def collect_outputs(
        self,
        request_id: str,
        request_state: ClientRequestState,
        *,
        response_stage_id: int | None,
        timeout: float | None,
    ) -> list[OmniRequestOutput]:
        deadline = None if timeout is None else time.monotonic() + timeout
        wall_start_ts = request_state.request_arrival_ts or time.time()
        final_stage_id = response_stage_id if response_stage_id is not None else max(0, self.output_port.num_stages - 1)
        num_stages = max(self.output_port.num_stages, final_stage_id + 1)
        metrics = getattr(request_state, "metrics", None) or OrchestratorMetrics(
            num_stages,
            self.output_port.log_stats,
            wall_start_ts,
            final_stage_id,
        )
        stage_event_cursor = metrics.stage_event_cursor(request_id)
        request_start_ts = {request_id: wall_start_ts}
        outputs: list[OmniRequestOutput] = []
        while True:
            if request_state in self._retired_output_routes:
                return []
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                break
            try:
                message = await asyncio.wait_for(request_state.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if message is _RETIRED_DUPLEX_ROUTE:
                # Relay the single marker to any other already-waiting reader.
                request_state.queue.put_nowait(_RETIRED_DUPLEX_ROUTE)
            if request_state in self._retired_output_routes:
                return []
            if isinstance(message, ErrorMessage):
                raise RuntimeError(message.error)
            if not isinstance(message, OutputMessage):
                continue
            engine_outputs = message.engine_outputs
            is_response_stage = response_stage_id is None or message.stage_id >= response_stage_id
            is_direct_response = self.is_direct_response(engine_outputs)
            output_to_collect = None
            if isinstance(engine_outputs, OmniRequestOutput):
                output_to_collect = engine_outputs
                if message.metrics is not None:
                    existing_metrics = dict(output_to_collect.metrics)
                    metrics.process_stage_metrics(
                        result={"metrics": message.metrics},
                        stage_type="",
                        stage_id=message.stage_id,
                        req_id=request_id,
                        engine_outputs=engine_outputs,
                        finished=message.finished,
                        final_output_type=engine_outputs.final_output_type,
                        output_to_yield=output_to_collect,
                        event_cursor=stage_event_cursor,
                    )
                    generated_metrics = dict(output_to_collect.metrics)
                    existing_stage_metrics = existing_metrics.get("stage_metrics")
                    generated_stage_metrics = generated_metrics.get("stage_metrics")
                    if isinstance(existing_stage_metrics, dict) and isinstance(generated_stage_metrics, dict):
                        generated_metrics["stage_metrics"] = {
                            **existing_stage_metrics,
                            **generated_stage_metrics,
                        }
                    output_to_collect.metrics = {
                        **existing_metrics,
                        **generated_metrics,
                    }
            elif is_response_stage:
                output_to_collect = self.output_port.process_single_result(
                    message,
                    message.stage_id,
                    metrics,
                    request_start_ts,
                    wall_start_ts,
                    final_stage_id,
                    stage_event_cursor,
                )
            if output_to_collect is not None and (is_response_stage or is_direct_response):
                if message.finished:
                    output_to_collect.finished = True
                outputs.append(output_to_collect)
            if message.finished or outputs:
                break
        return outputs

    @classmethod
    def is_direct_response(cls, output: object) -> bool:
        multimodal_output = cls.multimodal_output(output)
        if not multimodal_output:
            return False
        return multimodal_output.get("duplex_direct_response") is True or multimodal_output.get(
            "duplex_native_decision"
        ) in {"listen", "speak"}

    @classmethod
    def multimodal_output(cls, output: object) -> dict[str, object]:
        if isinstance(output, OmniRequestOutput):
            decision = get_duplex_output_decision(output)
            if decision is not None:
                return dict(decision.metadata)
            private_output = getattr(output, "_multimodal_output", None)
            multimodal_output = output.multimodal_output
            if isinstance(multimodal_output, dict) and multimodal_output:
                return multimodal_output
            inner_output = getattr(output, "request_output", None)
            if inner_output is not None and inner_output is not output:
                inner_multimodal_output = cls.multimodal_output(inner_output)
                if inner_multimodal_output:
                    return inner_multimodal_output
            if isinstance(private_output, dict):
                return private_output
            return multimodal_output if isinstance(multimodal_output, dict) else {}
        multimodal_output = getattr(output, "multimodal_output", None)
        if isinstance(multimodal_output, dict):
            return multimodal_output
        outputs = getattr(output, "outputs", None)
        completion = outputs[0] if isinstance(outputs, list) and outputs else None
        multimodal_output = getattr(completion, "multimodal_output", None) if completion is not None else None
        return multimodal_output if isinstance(multimodal_output, dict) else {}


__all__ = ["DuplexEnginePort", "DuplexRequestClient", "DuplexRequestOutputPort"]
