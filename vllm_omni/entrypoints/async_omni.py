# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
AsyncOmni - Refactored async orchestrator using AsyncOmniEngine.

This is the new implementation that uses AsyncOmniEngine (which manages
StageEngineCoreClient instances) instead of OmniStage with worker processes.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import anyio
import msgspec
from vllm import TokensPrompt
from vllm.config.profiler import validate_profile_iteration_bounds
from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import SupportedTask

from vllm_omni.diffusion.data import CuMemTag, OmniACK, OmniSleepTask, OmniWakeTask
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import ErrorMessage, NextStageInputMessage, OutputMessage
from vllm_omni.entrypoints.async_omni_base import ABORT_TIMEOUT_S, AsyncOmniBase
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.omni_base import OmniEngineDeadError
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.errors import client_error_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams, disable_watermarking
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.v1.engine import PauseMode
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.kv_hints import KvHintsEnvelope

    from vllm_omni.inputs.data import OmniInteractionPrompt, OmniPromptType

logger = init_logger(__name__)


class _SubmitRevokedError(Exception):
    """Internal: a tracked streaming submit was revoked (request aborted).

    Stops the input pump without reporting an input error or sending the
    final empty update — the revoke already closed the request out.
    """


CACHE_RESET_TIMEOUT_S = 60.0


class AsyncOmni(AsyncOmniBase, EngineClient):
    """Asynchronous unified entry point for multi-stage pipelines using AsyncOmniEngine.

    This is the refactored version that uses AsyncOmniEngine instead of
    OmniStage workers. It provides the same interface as AsyncOmni but with
    a cleaner architecture.

    Args:
        model: Model name or path to load.
        **kwargs: Additional keyword arguments.
            - deploy_config: Optional path to a deploy YAML. If None,
              configurations are resolved from the model pipeline factory.
            - log_stats: Whether to enable statistics logging.
            - stage_init_timeout: Timeout for per-stage initialization.
            - init_timeout: Total timeout for orchestrator startup.
            - async_chunk: Whether to enable async chunk mode.
            - output_modalities: Requested output modalities.
            - Additional keyword arguments passed to stage engines.

    Example:
        >>> async_omni = AsyncOmni(model="Qwen/Qwen2.5-Omni-7B")
        >>> async for output in async_omni.generate(
        ...     prompt="Hello",
        ...     request_id="req-1",
        ...     sampling_params_list=[SamplingParams(), SamplingParams()]
        ... ):
        ...     print(output)
    """

    engine: AsyncOmniEngine

    def _create_engine(self, **engine_kwargs: Any) -> AsyncOmniEngine:
        return AsyncOmniEngine(**engine_kwargs)

    def __init__(self, model: str = "", *args: Any, **kwargs: Any) -> None:
        tts_max_instructions_length = kwargs.get("tts_max_instructions_length", None)
        AsyncOmniBase.__init__(self, *args, model=model, **kwargs)
        self.tts_max_instructions_length = tts_max_instructions_length
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False
        # In-flight EngineCore submits (non-streaming add_request, or each
        # streaming ADD/update/final marker). sleep() waits for this to hit
        # zero so a pipelined request cannot race into EngineCore during
        # drain/offload. Streaming generate does not hold a slot while
        # waiting for the next client chunk.
        self._admitting: int = 0
        # In-flight add_request_async tasks awaiting queue room, keyed by
        # internal request id. A full request queue against a paused engine
        # used to wedge the event loop (sync_q.put blocks the calling thread),
        # so sleep() could never observe _admitting==0. pause_generation(mode=
        # "abort")/sleep() cancel these so the admission slots release instead
        # of pinning the loop (#8678), and abort(request_id) settles a
        # request's own submit so a queue-parked ADD cannot be admitted after
        # the abort returns.
        self._pending_submits: dict[str, asyncio.Task[None]] = {}
        # True after pause_generation() or AR EngineCore sleep; wake_up must
        # not reopen generate() until resume_generation(). Diffusion-only
        # sleep uses _paused as a temporary admission gate and clears it
        # on wake so sleep → wake → generate keeps working.
        self._hold_admission_until_resume: bool = False
        # Stages whose scheduler pause_generation closed (AR in any mode,
        # diffusion with mode="keep"); admission stays closed until
        # resume_generation has reopened all of them.
        self._paused_stage_ids: set[int] = set()
        self._sleeping_tags: set[str] = set()
        self._stage_sleeping_tags: dict[int, set[str]] = {}
        self._level2_sleeping: bool = False
        # Paced interaction ingress is stamped before ordered worker delivery.
        self._interaction_reservations: dict[tuple[str, str], float | None] = {}

    # ==================== Generate Method ====================

    async def generate(
        self,
        prompt: OmniPromptType | AsyncGenerator[StreamingInput, None] | list[OmniPromptType],
        sampling_params: Any = None,
        request_id: str = "",
        *,
        prompt_text: str | None = None,
        lora_request: Any = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        sampling_params_list: Sequence[OmniSamplingParams] | None = None,
        output_modalities: list[str] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        session_id: str | None = None,
        kv_hints: KvHintsEnvelope | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        arrival_time: float | None = None,
        submitted: asyncio.Future[None] | None = None,
        entry_stage_id: int = 0,
        watermarking: bool | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt(s) asynchronously.

        Coordinates multi-stage pipeline execution. Processes the prompt
        through all stages in the pipeline and yields outputs as they become
        available.

        ``session_id`` is accepted for EngineClient protocol compatibility
        and is not duplex-session plumbing.

        **Diffusion batching:**
        Diffusion stages accept only a single prompt per request.  Passing a
        ``list`` of prompts to a diffusion stage will raise ``ValueError``.
        To batch multiple diffusion prompts, submit each as an independent
        request; the scheduler will automatically co-batch compatible requests.

        Args:
            prompt: A single prompt **or** a list of prompts.  For diffusion
                stages, only a single prompt is accepted; a list will be
                rejected with an error.
            request_id: Unique identifier for this request. If one is not provided,
                a random one will be generated.
            sampling_params_list: List of SamplingParams, one per stage.
                Must have the same length as the number of stages.
                If *None*, uses default sampling params for each stage.
            output_modalities: Optional list of output modalities.
            entry_stage_id: Stage the request is submitted to; > 0 bypasses
                stage 0 and leaves the prompt raw for that stage. Streaming
                prompts always enter at stage 0.
            watermarking: Whether configured watermarking applies to this request
                (None means enabled); False disables it on every stage.

        Yields:
            OmniRequestOutput objects as they are produced by each stage.

        Raises:
            ValueError: If sampling_params_list has incorrect length, or
                if a list prompt is submitted to a diffusion stage.
        """
        # Append a random UUID suffix to the request_id to ensure it is unique
        # and non-empty, similar to vLLM's input processor. The suffix is used
        # only for internal tracking throughout the request's life.
        external_request_id = request_id
        request_id = self._get_unique_request_id(external_request_id)

        # Wait until generation is resumed if the engine is paused. Non-streaming
        # generate holds an admission slot until add_request completes so sleep()
        # cannot race a just-unblocked generate into EngineCore during offload.
        # Streaming generate does **not** hold the slot while waiting for the
        # first client chunk; the input pump acquires it immediately before each
        # EngineCore ADD/update.
        # The wait also covers partial wakes: after wake_up(tags=["weights"]) +
        # resume_generation() the kv_cache tag can still be asleep, and a submit
        # against a partially-asleep engine would only be rejected EngineCore
        # side. Park here until the full wake instead of erroring (#8678).
        streaming_input = isinstance(prompt, AsyncGenerator)
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused and not getattr(self, "_sleeping_tags", None))
            if not streaming_input:
                self._admitting = getattr(self, "_admitting", 0) + 1
        admitting = not streaming_input

        logger.debug(f"[AsyncOmni] generate() called for request {external_request_id}")

        input_stream_task: asyncio.Task | None = None
        try:
            # Backstop: the admission gate above waits out sleeping tags, so
            # this only fires if tags appeared after the gate passed (e.g. a
            # sleep() racing this submit) — fail fast rather than submit into
            # an offloading engine.
            _sleeping_tags = getattr(self, "_sleeping_tags", None)
            if _sleeping_tags:
                raise RuntimeError(
                    f"Generation rejected: Engine is partially or fully asleep. "
                    f"Currently sleeping tags: {list(_sleeping_tags)}. "
                    f"Please perform a full wake_up before generating."
                )

            # Reject diffusion list-prompt early with a clear API error.
            if isinstance(prompt, list) and any(
                stage_config.stage_type == "diffusion" for stage_config in self.engine.stage_configs
            ):
                raise ValueError(
                    "Diffusion stages accept only a single prompt per request. "
                    "Submit multiple independent requests to use scheduler batching."
                )

            # Start final output dispatcher on the first call to generate()
            self._final_output_handler()

            # Forward bare sampling_params (e.g. from /v1/completions) as the stage-0 entry.
            if sampling_params_list is None and sampling_params is not None:
                if self.num_stages == 1:
                    sampling_params_list = [sampling_params]
                else:
                    default = list(self.default_sampling_params_list)
                    default[0] = sampling_params
                    sampling_params_list = default

            # Expand sampling params for PD disaggregation (user may provide N-1 params)
            if (
                sampling_params_list is not None
                and isinstance(sampling_params_list, Sequence)
                and not isinstance(sampling_params_list, (str, bytes))
            ):
                sampling_params_list = self._maybe_expand_sampling_params(list(sampling_params_list))

            # Set the output kind to delta output if sampling params were omitted,
            # since AsyncOmni is typically used for streaming.
            sampling_params_list = self.resolve_sampling_params_list(
                sampling_params_list,
                allow_delta_coercion=True,
            )
            if watermarking is False:
                sampling_params_list = disable_watermarking(sampling_params_list)

            # Track per-request metrics
            wall_start_ts = float(arrival_time) if arrival_time is not None else time.time()
            req_start_ts: dict[str, float] = {}

            # Determine the final stage for E2E stats
            final_stage_id_for_e2e = self._compute_final_stage_id(output_modalities)
            final_output_stage_ids = self._compute_final_output_stage_ids(output_modalities) or [final_stage_id_for_e2e]

            metrics = OrchestratorMetrics(
                self.num_stages,
                self.log_stats,
                wall_start_ts,
                final_stage_id_for_e2e,
                transfer_emitter=getattr(self, "transfer_metrics", None),
                replica_resolver=self._resolve_transfer_replica,
            )

            req_state = ClientRequestState(
                request_id=request_id,
                external_request_id=external_request_id,
                final_stage_id=final_stage_id_for_e2e,
            )
            req_state.metrics = metrics
            req_state.request_arrival_ts = wall_start_ts
            self.request_states[request_id] = req_state

            # PD disaggregation: modify prefill-stage sampling params per request
            req_sp_list = list(sampling_params_list)
            for index, params in enumerate(req_sp_list):
                if isinstance(params, OmniDiffusionSamplingParams) and params.streaming_buffer_seconds is not None:
                    self.validate_streaming_playback_request()
                    req_sp_list[index] = replace(params, streaming_output_ack=True)
            pd_pair = self._get_pd_separation_pair()
            if pd_pair is not None:
                p_id = pd_pair[0]
                req_sp_list[p_id] = self._prepare_prefill_sampling_params(request_id, req_sp_list[p_id])

            # Add request(s) to stage 0. For streaming inputs, submit
            # chunks incrementally through streaming_update. The helper
            # returns as soon as the pump task is created; each ADD/update
            # takes its own admission slot inside the pump.
            if streaming_input:
                first_chunk_submitted = asyncio.get_running_loop().create_future()
                input_stream_task = await self._add_streaming_input_request(
                    request_id=request_id,
                    input_stream=cast(AsyncGenerator, prompt),
                    sampling_params_list=req_sp_list,
                    final_stage_id=final_stage_id_for_e2e,
                    final_output_stage_ids=final_output_stage_ids,
                    arrival_time=wall_start_ts,
                    lora_request=lora_request,
                    first_chunk_submitted=first_chunk_submitted,
                    **({"kv_hints": kv_hints} if kv_hints is not None else {}),
                )
                await first_chunk_submitted
            else:
                revoked_submit = await self._submit_tracked(
                    request_id,
                    lambda: self.engine.add_request_async(
                        request_id=request_id,
                        prompt=prompt,
                        sampling_params_list=req_sp_list,
                        final_stage_id=final_stage_id_for_e2e,
                        final_output_stage_ids=final_output_stage_ids,
                        arrival_time=wall_start_ts,
                        lora_request=lora_request,
                        entry_stage_id=entry_stage_id,
                        **({"kv_hints": kv_hints} if kv_hints is not None else {}),
                    ),
                    hold_admission=False,
                    on_revoke=lambda: self._finish_revoked_submit(request_id),
                )
                if revoked_submit:
                    # _finish_revoked_submit already enqueued the terminal
                    # abort output; the results loop below drains it and the
                    # generator finishes as a normal aborted request.
                    logger.debug("[AsyncOmni] Submit for %s was revoked before admission", request_id)
            submit_ts = time.time()
            stage_first_ts = cast(list[float | None], req_state.metrics.stage_first_ts)
            stage_first_ts[0] = submit_ts
            req_start_ts[request_id] = submit_ts
            if admitting:
                await self._release_generate_admission()
                admitting = False
            # Callers wait on this before abort so Stage0 observes the sender cache.
            if submitted is not None and not submitted.done():
                submitted.set_result(None)
            # Refresh gauges on arrival.
            self._publish_request_gauges(len(self.request_states))

            # Process results based on mode
            # Both sequential and async_chunk modes read the same message stream
            # from Orchestrator; stage-transfer behavior differs inside
            # Orchestrator._route_output().
            async for output in self._process_orchestrator_results(
                request_id,
                metrics,
                final_stage_id_for_e2e,
                req_start_ts,
                wall_start_ts,
            ):
                yield output
                metadata = (getattr(output, "multimodal_output", None) or {}).get("metadata", {})
                stream = metadata.get("stream", {}) if isinstance(metadata, dict) else {}
                chunk_index = stream.get("output_chunk_index") if isinstance(stream, dict) else None
                if chunk_index is not None:
                    await self._engine_core_rpc(
                        "release_streaming_output", stage_ids=[0], args=(request_id, chunk_index)
                    )

            logger.debug(f"[AsyncOmni] Request {request_id} completed")

        except (asyncio.CancelledError, GeneratorExit):
            self._record_request_failure_once(request_id, reason="client_disconnect")
            # ASGI cancellation also affects subsequent awaits. Shield the
            # bounded engine abort before removing the local request state.
            with anyio.CancelScope(shield=True):
                await self._abort_internal_requests(request_id, timeout=ABORT_TIMEOUT_S)
            logger.info(f"[AsyncOmni] Request {request_id} aborted.")
            raise
        except Exception as e:
            self._record_request_failure_once(request_id, reason="stage_error")
            await self._abort_internal_requests(request_id, timeout=ABORT_TIMEOUT_S)
            logger.info(f"[AsyncOmni] Request {request_id} failed (input error): {e}")
            raise
        finally:
            if input_stream_task is not None and not input_stream_task.done():
                input_stream_task.cancel()
            if admitting:
                await self._release_generate_admission()
            reservations = getattr(self, "_interaction_reservations", {})
            for key in [key for key in reservations if key[0] == request_id]:
                reservations.pop(key, None)
            self._log_summary_and_cleanup(request_id)

    async def _release_generate_admission(self) -> None:
        """Drop one in-flight generate admission slot held across add_request."""
        async with self._pause_cond:
            self._admitting = max(getattr(self, "_admitting", 1) - 1, 0)
            self._pause_cond.notify_all()

    def _cancel_pending_submits(self) -> None:
        """Cancel generate() coroutines still waiting for request-queue room.

        Their queue puts suspend on a paused engine; without the cancel they
        would hold their admission slots forever and sleep() would wait on
        ``_admitting == 0`` indefinitely. Cancellation unwinds through
        generate()'s normal abort handling, releasing the slot. Callers hold
        ``_pause_cond`` (or run before any generate can pass the gate), so a
        submit cannot register after this snapshot; the ``_admitting == 0``
        barrier the callers wait on afterwards settles the rest.
        """
        for task in list(getattr(self, "_pending_submits", {}).values()):
            task.cancel()

    async def _submit_tracked(
        self,
        request_id: str,
        submit_factory: Callable[[], Awaitable[None]],
        *,
        hold_admission: bool,
        on_revoke: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        """Gate + run one engine submit, tracked for pause/sleep/abort revoke.

        The submit runs as a task that registers here before any wait — so
        the cancel snapshot inside ``pause_generation()``/``sleep()`` cannot
        miss it. Streaming pumps (``hold_admission=True``) re-check admission
        per submit and take a per-submit slot: their caller drops generate()'s
        outer slot while waiting for the next client chunk, so the ADD must
        not run while the engine is paused or partially asleep. A
        non-streaming submit re-checks nothing and takes no slot: its caller
        holds the outer slot across add_request, and sleep()'s ``_admitting
        == 0`` barrier waits behind that slot — re-waiting on ``_paused``
        here would deadlock ``sleep(mode="wait")`` against it, since only
        ``mode="abort"`` cancels parked submits. When the submit is revoked
        (the task is cancelled while the caller was not), ``on_revoke``
        closes the request out as an abort instead of letting the
        CancelledError escape into a caller that never cancelled anything.
        Returns whether that happened — a True return means the request is
        already closed out and the caller must stop submitting for it (the
        streaming pump stops; generate() drains the terminal abort output
        below).
        """
        pending = getattr(self, "_pending_submits", None)
        if pending is None:
            pending = self._pending_submits = {}

        async def _gated_submit() -> None:
            if hold_admission:
                async with self._pause_cond:
                    # Same full-wake condition as generate()'s gate: streaming
                    # ADDs must not race a partially-asleep engine either (#8678).
                    await self._pause_cond.wait_for(
                        lambda: not self._paused and not getattr(self, "_sleeping_tags", None)
                    )
                    self._admitting = getattr(self, "_admitting", 0) + 1
            try:
                await submit_factory()
            finally:
                if hold_admission:
                    await self._release_generate_admission()

        submit_task = asyncio.get_running_loop().create_task(_gated_submit())
        pending[request_id] = submit_task
        try:
            await submit_task
        except asyncio.CancelledError:
            # Distinguish a revoked submit from a cancelled caller: the gated
            # task is cancelled only by pause/sleep/abort, while a
            # cancellation of the generate()/pump task itself leaves it
            # running and marks this task as cancelling instead. (The gated
            # task may still be unwinding its slot release when the error
            # reaches us, so submit_task.cancelled() is not reliable here.)
            caller = asyncio.current_task()

            if caller is not None and caller.cancelling() > 0:  # type: ignore[attr-defined]
                # Stop the gated submit too, so one parked at the gate or the
                # queue cannot run later for a request whose caller is gone.
                submit_task.cancel()
                raise
            if on_revoke is not None:
                await on_revoke()
            return True
        finally:
            pending.pop(request_id, None)
        return False

    async def _finish_revoked_submit(self, request_id: str) -> None:
        """Close out a submit revoked before admission as an aborted request.

        The abort-retry RL client handles a normal generator end with
        ``finish_reason="abort"`` (an admitted-then-aborted request ends
        exactly that way); a CancelledError instead would mark the caller
        task cancelled and kill a ``gather`` over the rollout batch.
        """
        self._record_request_failure_once(request_id, reason="client_abort")
        await self._enqueue_synthetic_abort(request_id)

    async def _add_streaming_input_request(
        self,
        *,
        request_id: str,
        input_stream: AsyncGenerator[StreamingInput, None],
        sampling_params_list: Sequence[OmniSamplingParams],
        final_stage_id: int,
        final_output_stage_ids: Sequence[int],
        arrival_time: float,
        lora_request: Any = None,
        first_chunk_submitted: asyncio.Future[None] | None = None,
        kv_hints: KvHintsEnvelope | None = None,
    ) -> asyncio.Task:
        """Submit a streaming input generator as incremental stage-0 updates."""
        if not sampling_params_list:
            raise ValueError("sampling_params_list cannot be empty for streaming input")
        # only check thinker's sampling params now
        stage0_params = sampling_params_list[0]
        self._validate_streaming_input_sampling_params(stage0_params)
        stage0_params = cast(SamplingParams, stage0_params)
        req_state = self.request_states[request_id]
        has_submitted_first_chunk = False
        input_error_reported = False

        # NOTE: InputProcessor in vLLM should generally do this too, but for
        # now we do it defensively. TODO (Alex) ensure clones/copying are optimized
        if not stage0_params.skip_clone:
            stage0_params = stage0_params.clone()
            stage0_params.skip_clone = True

        def _mark_first_chunk_submitted() -> None:
            if first_chunk_submitted is not None and not first_chunk_submitted.done():
                first_chunk_submitted.set_result(None)

        async def _report_input_error(error: Exception) -> None:
            nonlocal input_error_reported
            if input_error_reported:
                return
            input_error_reported = True
            status_code, error_type = client_error_metadata(error)
            await req_state.queue.put(
                ErrorMessage(
                    request_id=request_id,
                    error=str(error),
                    status_code=status_code,
                    error_type=error_type,
                )
            )

        async def _revoke_streaming() -> None:
            # pause/sleep/abort revoked the parked submit: wake generate()
            # (it may still wait on first_chunk_submitted) and end the
            # request as aborted instead of unwinding the pump as a client
            # disconnect.
            _mark_first_chunk_submitted()
            await self._finish_revoked_submit(request_id)

        async def _submit_streaming_chunk(submit_factory) -> None:
            """One streaming ADD/update: admission-gated and revoke-tracked."""
            if await self._submit_tracked(request_id, submit_factory, hold_admission=True, on_revoke=_revoke_streaming):
                # The revoke already ended this request (synthetic abort
                # enqueued); stop the pump instead of sending further chunk
                # or final updates for it.
                raise _SubmitRevokedError

        async def handle_inputs() -> None:
            nonlocal has_submitted_first_chunk
            cancelled = False
            # Reported only after the terminal submit below: once generate()
            # sees the error it aborts the request, which would revoke a
            # terminal update still waiting to run as its tracked task.
            input_error: Exception | None = None
            try:
                async for chunk in input_stream:
                    chunk_params = getattr(chunk, "sampling_params", None) or stage0_params
                    self._validate_streaming_input_sampling_params(chunk_params)
                    chunk_sampling_params_list = list(sampling_params_list)
                    chunk_sampling_params_list[0] = chunk_params
                    chunk_prompt = chunk.prompt
                    prompt_text, _, _ = extract_prompt_components(self.model_config, chunk_prompt)

                    if not has_submitted_first_chunk:
                        await _submit_streaming_chunk(
                            lambda: self.engine.add_request_async(
                                request_id=request_id,
                                prompt=chunk_prompt,
                                prompt_text=prompt_text,
                                sampling_params_list=chunk_sampling_params_list,
                                final_stage_id=final_stage_id,
                                final_output_stage_ids=final_output_stage_ids,
                                arrival_time=arrival_time,
                                lora_request=lora_request,
                                resumable=True,
                                **({"kv_hints": kv_hints} if kv_hints is not None else {}),
                            )
                        )
                        has_submitted_first_chunk = True
                        _mark_first_chunk_submitted()
                    else:
                        await _submit_streaming_chunk(
                            lambda: self.engine.add_streaming_update_async(
                                request_id=request_id,
                                prompt=chunk_prompt,
                                prompt_text=prompt_text,
                                sampling_params_list=chunk_sampling_params_list,
                                final_stage_id=final_stage_id,
                                final_output_stage_ids=final_output_stage_ids,
                                arrival_time=arrival_time,
                                lora_request=lora_request,
                                resumable=True,
                            )
                        )
            except _SubmitRevokedError:
                cancelled = True
            except (asyncio.CancelledError, GeneratorExit):
                cancelled = True
            except Exception as error:
                input_error = error
            finally:
                try:
                    if not cancelled:
                        # Send empty final request to indicate that inputs have
                        # finished. Don't send if canceled (session was aborted).
                        final_sampling_params_list = list(sampling_params_list)
                        final_sampling_params_list[0] = stage0_params
                        final_prompt = TokensPrompt(prompt_token_ids=[0])

                        if has_submitted_first_chunk:
                            await _submit_streaming_chunk(
                                lambda: self.engine.add_streaming_update_async(
                                    request_id=request_id,
                                    prompt=final_prompt,
                                    prompt_text=None,
                                    sampling_params_list=final_sampling_params_list,
                                    final_stage_id=final_stage_id,
                                    final_output_stage_ids=final_output_stage_ids,
                                    arrival_time=arrival_time,
                                    lora_request=lora_request,
                                    resumable=False,
                                )
                            )
                        else:
                            await _submit_streaming_chunk(
                                lambda: self.engine.add_request_async(
                                    request_id=request_id,
                                    prompt=final_prompt,
                                    prompt_text=None,
                                    sampling_params_list=final_sampling_params_list,
                                    final_stage_id=final_stage_id,
                                    final_output_stage_ids=final_output_stage_ids,
                                    arrival_time=arrival_time,
                                    lora_request=lora_request,
                                    resumable=False,
                                    **({"kv_hints": kv_hints} if kv_hints is not None else {}),
                                )
                            )
                            has_submitted_first_chunk = True
                except _SubmitRevokedError:
                    # Revoke of the final update already closed the request
                    # with a synthetic abort; do not also report its error.
                    pass
                except Exception as error:
                    if input_error is None:
                        input_error = error
                finally:
                    if input_error is not None:
                        await _report_input_error(input_error)
                    # Unblock generate() even on cancel / empty stream / submit
                    # failure so it can observe the queued terminal result or error.
                    _mark_first_chunk_submitted()

        input_stream_task = asyncio.create_task(handle_inputs())
        req_state.input_stream_task = input_stream_task
        return input_stream_task

    @staticmethod
    def _validate_streaming_input_sampling_params(params: OmniSamplingParams) -> None:
        if (
            not isinstance(params, SamplingParams)
            or params.n > 1
            or params.output_kind == RequestOutputKind.FINAL_ONLY
            or params.stop
        ):
            raise ValueError(
                "Input streaming is currently supported only for SamplingParams "
                "with n == 1, output_kind != FINAL_ONLY, and without stop strings."
            )

    async def run_entry_stage(
        self, prompt: OmniPromptType, sampling_params_list: Sequence[OmniSamplingParams], *, request_id: str
    ) -> NextStageInputMessage | OutputMessage:
        """Run the entry stage as its own request.

        Returns the next stage's input, or the raw output if the entry stage is the final stage.
        """
        final_stage_id = self.num_stages - 1
        # Each call returns one complete output, so we use FINAL_ONLY
        sampling_params_list = coerce_param_message_types(list(sampling_params_list), is_streaming=False)
        async with self._stage_request(request_id, stage_id=0) as req_state:
            await self.engine.add_request_async(
                request_id=req_state.request_id,
                prompt=prompt,
                sampling_params_list=sampling_params_list,
                final_stage_id=final_stage_id,
                final_output_stage_ids=[final_stage_id],
                yield_stage_id=0 if final_stage_id > 0 else None,
            )
            return await self._stage_response(req_state)

    async def run_downstream_stage(
        self, stage_input: NextStageInputMessage, *, request_id: str
    ) -> NextStageInputMessage | OutputMessage:
        """Run the receiver stage of a next stage input as its own request."""
        async with self._stage_request(request_id, stage_id=stage_input.receiver_stage_id) as req_state:
            await self.engine.add_next_stage_input_async(
                msgspec.structs.replace(stage_input, request_id=req_state.request_id)
            )
            return await self._stage_response(req_state)

    @asynccontextmanager
    async def _stage_request(
        self, external_request_id: str, *, stage_id: int
    ) -> AsyncGenerator[ClientRequestState, None]:
        """Track one stage call; abort it if the caller is cancelled and always drop its state."""
        request_id = f"{self._get_unique_request_id(external_request_id)}-{stage_id}"
        req_state = ClientRequestState(request_id, external_request_id=external_request_id)
        self.request_states[request_id] = req_state
        self._final_output_handler()
        try:
            yield req_state
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                await self._abort_internal_requests(request_id, timeout=ABORT_TIMEOUT_S)
            raise
        finally:
            self.request_states.pop(request_id, None)

    async def _stage_response(self, req_state: ClientRequestState) -> NextStageInputMessage | OutputMessage:
        """Wait for a stage call's response: the next stage input, or the finished output.

        A final stage that decodes emits an unfinished output per step; those are skipped.
        """
        # This is analogous to _process_orchestrator_results.
        while True:
            response = await req_state.queue.get()
            if isinstance(response, ErrorMessage):
                if response.fatal:
                    raise OmniEngineDeadError(response.error, error_stage_id=response.stage_id)
                self._raise_nonfatal_error_message(response)
            if not isinstance(response, OutputMessage) or response.finished:
                return response

    def _route_engine_message(self, msg: Any) -> bool:
        if not isinstance(msg, NextStageInputMessage):
            return False
        req_state = self.request_states.get(msg.request_id)
        if req_state is not None:
            req_state.queue.put_nowait(msg)
        return True

    async def encode(
        self,
        prompt: Any,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """EngineClient.encode() stub.

        Omni pipeline currently exposes only generate() API at orchestrator level.
        """
        raise NotImplementedError("AsyncOmni.encode is not implemented.")

    # ==================== Control Methods ====================

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        stage_ids: list[int] | None = None,
    ) -> list[Any]:
        """Execute a best-effort control RPC on selected stages.

        Unsupported stages currently return a TODO-style result dict instead of
        failing the entire call. This keeps AsyncOmni usable while the orchestrator
        control plane is still being filled out.
        """
        results = await self.engine.collective_rpc_async(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            stage_ids=stage_ids,
        )

        unsupported_results = [
            index for index, result in enumerate(results) if isinstance(result, dict) and result.get("todo")
        ]
        if unsupported_results:
            logger.warning(
                "[AsyncOmni] collective_rpc(%s) has TODO support in replica result(s) %s (requested stages: %s)",
                method,
                unsupported_results,
                stage_ids,
            )

        return results

    async def _engine_core_rpc(
        self,
        method: str,
        *,
        stage_ids: list[int],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> list[Any]:
        """Call EngineCore helpers and reject failures from any replica."""
        timeout_kwargs = {"timeout": timeout} if timeout is not None else {}
        results = await self.collective_rpc(
            method=method,
            args=args,
            kwargs=kwargs,
            stage_ids=stage_ids,
            **timeout_kwargs,
        )

        def check_result(result: Any) -> None:
            if isinstance(result, (list, tuple)):
                for item in result:
                    check_result(item)
            elif isinstance(result, dict):
                if result.get("error") or result.get("todo") or result.get("supported") is False:
                    raise RuntimeError(f"{method} failed: {result}")

        for result in results:
            check_result(result)
        return results

    @staticmethod
    def _coerce_stage_bool(result: Any) -> bool:
        """Reduce a stage RPC result to a boolean.

        Some stage RPCs may return worker-level lists like ``[True]``;
        diffusion wrappers usually return a plain bool.
        """
        if isinstance(result, list):
            return all(bool(item) for item in result)
        return bool(result)

    async def abort(self, request_id: str | Iterable[str], *, timeout: float | None = None) -> None:
        """Abort request(s) via the Orchestrator."""
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        # Map the external user request IDs to internal IDs used by the Orchestrator.
        # NOTE: If the user request_id matches multiple requests, all of them will be
        # aborted. This is also what happens in this case in vLLM's output processor.
        internal_ids = [s.request_id for s in self.request_states.values() if s.external_request_id in request_ids]
        # A submit parked on a full request queue would be admitted after this
        # abort returned; revoke it before delegating to the Orchestrator.
        await self._settle_pending_submits(internal_ids)
        await self._abort(internal_ids, timeout=timeout)

    async def update_streaming_playback(self, request_id: str, position_seconds: float) -> None:
        """Forward external-session playback progress to the diffusion engine."""
        self.validate_streaming_playback_request()
        for state in list(self.request_states.values()):
            if state.external_request_id == request_id:
                await self._engine_core_rpc(
                    "update_streaming_playback", stage_ids=[0], args=(state.request_id, position_seconds)
                )

    def validate_streaming_playback_request(self) -> None:
        """Validate pacing topology before a realtime video request starts."""
        if self.num_stages != 1 or self.engine.get_stage_metadata(0).stage_type != "diffusion":
            raise ValueError("playback feedback requires single-stage diffusion")

    def _resolve_interaction_request(self, request_id: str) -> str:
        """Validate and map an external streaming id to its internal id."""
        if self.num_stages != 1:
            raise ValueError("interaction requires single-stage diffusion")
        stage_meta = self.engine.get_stage_metadata(0)
        if stage_meta.stage_type != "diffusion":
            raise ValueError("interaction requires a diffusion stage")

        internal_ids = [s.request_id for s in self.request_states.values() if s.external_request_id == request_id]
        if not internal_ids:
            raise ValueError(f"No active request for interaction: {request_id!r}")
        if len(internal_ids) > 1:
            raise ValueError(
                f"interaction requires exactly one active request for {request_id!r}, found {len(internal_ids)}"
            )
        return internal_ids[0]

    def _interaction_reservations_map(self) -> dict[tuple[str, str], float | None]:
        """Lazily provide state for lightweight objects that skip ``__init__``."""
        reservations = getattr(self, "_interaction_reservations", None)
        if reservations is None:
            reservations = {}
            self._interaction_reservations = reservations
        return reservations

    async def reserve_streaming_interaction(self, request_id: str, event_id: str) -> float | None:
        """Reserve an engine-clock ingress timestamp without waiting for a worker RPC."""
        internal_id = self._resolve_interaction_request(request_id)
        reservations = self._interaction_reservations_map()
        reservation_key = (internal_id, event_id)
        reservations[reservation_key] = None
        try:
            stamps = await self._engine_core_rpc(
                "track_streaming_interaction", stage_ids=[0], args=(internal_id, event_id)
            )
        except BaseException:
            # Keep the placeholder so the caller's cancellation/error path can
            # release the engine-side reservation exactly once.
            raise
        stamp = next((value for value in stamps if value is not None), None)
        if stamp is not None:
            reservations[reservation_key] = float(stamp)
        else:
            reservations.pop(reservation_key, None)
        return stamp

    async def release_streaming_interaction(self, request_id: str, event_id: str) -> None:
        """Release a reservation that will not be delivered to the worker."""
        internal_id = self._resolve_interaction_request(request_id)
        reservations = self._interaction_reservations_map()
        reservation_key = (internal_id, event_id)
        if reservation_key not in reservations:
            return
        reservations.pop(reservation_key, None)
        await self._engine_core_rpc("track_streaming_interaction", stage_ids=[0], args=(internal_id, event_id, False))

    async def submit_interaction_async(
        self,
        request_id: str,
        *,
        interaction: OmniInteractionPrompt,
        track_playback: bool = False,
    ) -> float | None:
        """Apply a midway interaction to an active streaming diffusion request.

        ``request_id`` is the external id created by the server-side session,
        matching the value passed to :meth:`generate`.
        """
        event = interaction.get("event")
        prompt = event.get("prompt") if isinstance(event, dict) else None
        if isinstance(event, dict) and "prompt" in event and (not isinstance(prompt, str) or not prompt):
            raise ValueError("prompt must be non-empty")
        transition_chunks = interaction.get("transition_chunks")
        if transition_chunks is not None and transition_chunks < 0:
            raise ValueError("transition_chunks must be >= 0")

        internal_id = self._resolve_interaction_request(request_id)
        event_id = interaction.get("event_id", "")
        reservations = self._interaction_reservations_map()
        reservation_key = (internal_id, event_id)
        received_at = reservations.get(reservation_key)
        reserved = reservation_key in reservations
        reservation_attempted = track_playback and not reserved
        try:
            if reservation_attempted:
                received_at = await self.reserve_streaming_interaction(request_id, event_id)
                reserved = received_at is not None
            # ``received_at`` is an engine-owned field. A direct caller may put
            # one in the TypedDict, but it must never reach the runner unchanged.
            interaction = {**interaction}
            interaction.pop("received_at", None)
            if received_at is not None:
                interaction["received_at"] = received_at
            await self.engine.submit_interaction_async(internal_id, interaction=interaction)
        except BaseException:
            if reserved or reservation_attempted:
                try:
                    await self.release_streaming_interaction(request_id, event_id)
                except Exception:
                    logger.debug("Failed to release interaction reservation", exc_info=True)
            raise
        else:
            reservations.pop(reservation_key, None)
        if self.log_stats:
            logger.info("[AsyncOmni] Queued interaction for request %s", request_id)
        return received_at

    def _split_stage_ids_by_type(self, stage_ids: list[int] | None = None) -> tuple[list[int], list[int]]:
        """Split stage ids into AR/LLM (EngineCore) vs diffusion (worker RPC)."""
        n_stages = len(self.engine.stage_configs)
        if stage_ids is None:
            stage_ids = list(range(n_stages))
        else:
            invalid = [sid for sid in stage_ids if not isinstance(sid, int) or sid < 0 or sid >= n_stages]
            if invalid:
                raise ValueError(
                    f"Invalid stage_ids {invalid}; valid range is 0..{n_stages - 1}"
                    if n_stages
                    else f"Invalid stage_ids {invalid}; this engine has no stages"
                )
        ar_stage_ids: list[int] = []
        diffusion_stage_ids: list[int] = []
        for sid in stage_ids:
            stage_config = self.engine.stage_configs[sid]
            if stage_config.stage_type == "diffusion":
                diffusion_stage_ids.append(sid)
            else:
                ar_stage_ids.append(sid)
        return ar_stage_ids, diffusion_stage_ids

    def _sleeping_tags_for_stages(self, stage_ids: list[int]) -> set[str]:
        """Union of sleeping tags recorded for the given stages."""
        per_stage = getattr(self, "_stage_sleeping_tags", None) or {}
        tags: set[str] = set()
        for sid in stage_ids:
            tags.update(per_stage.get(sid, set()))
        return tags

    def _refresh_union_sleeping_tags(self) -> None:
        """Keep ``_sleeping_tags`` as the engine-wide union of per-stage tags."""
        per_stage = getattr(self, "_stage_sleeping_tags", None) or {}
        union: set[str] = set()
        for stage_tags in per_stage.values():
            union.update(stage_tags)
        self._sleeping_tags = union

    def _record_stage_sleep(self, stage_ids: list[int], tags: Iterable[str]) -> None:
        per_stage = getattr(self, "_stage_sleeping_tags", None)
        if per_stage is None:
            per_stage = {}
            self._stage_sleeping_tags = per_stage
        tag_set = set(tags)
        for sid in stage_ids:
            per_stage.setdefault(sid, set()).update(tag_set)
        self._refresh_union_sleeping_tags()

    def _clear_stage_sleep(self, stage_ids: list[int], tags: Iterable[str]) -> None:
        per_stage = getattr(self, "_stage_sleeping_tags", None)
        if per_stage is None:
            self._sleeping_tags = set()
            return
        tag_set = set(tags)
        for sid in stage_ids:
            remaining = per_stage.get(sid)
            if remaining is None:
                continue
            remaining.difference_update(tag_set)
            if not remaining:
                per_stage.pop(sid, None)
        self._refresh_union_sleeping_tags()

    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
        stage_ids: list[int] | None = None,
    ) -> None:
        """Pause generation, mirroring vLLM AsyncLLM.pause_generation.

        1. Stop frontend admission (``_paused``).
        2. For AR/LLM stages, call EngineCore.pause_scheduler via the
           Orchestrator loop (abort/wait/keep + optional cache clear).
        3. For diffusion stages, ``mode="keep"`` pauses the DiffusionEngine
           scheduler and returns once the batch that was running has finished
           on every worker; that batch is delivered before any control RPC
           issued after this call runs, so the documented pause -> sleep order
           is safe. Queued requests stay queued until
           :meth:`resume_generation`. Other modes pause frontend admission
           only.

        Note: ``sleep()`` already pauses the AR scheduler internally (same as
        vLLM EngineCore.sleep). Call this API when you need pause *without*
        freeing GPU memory (e.g. weight sync).
        """
        if wait_for_inflight_requests:
            mode = "wait"

        async with self._pause_cond:
            # Keep running EngineCore pause + cache clear even when frontend
            # admission is already paused (sleep or a prior pause_generation).
            self._paused = True
            self._hold_admission_until_resume = True
            # Cancel under the lock, matching sleep(): with the gate closed,
            # a generate that has not passed it cannot register a submit the
            # snapshot would miss.
            if mode == "abort":
                self._cancel_pending_submits()

        ar_stage_ids, diffusion_stage_ids = self._split_stage_ids_by_type(stage_ids)
        if mode != "keep":
            diffusion_stage_ids = []
        # Recorded before the RPCs so a failed or cancelled pause can still
        # be undone with an explicit resume_generation.
        self._paused_stage_ids.update(ar_stage_ids, diffusion_stage_ids)
        if ar_stage_ids:
            logger.info(
                "[%s] Pausing AR stage(s) %s via EngineCore.pause_scheduler(mode=%s)",
                self._name,
                ar_stage_ids,
                mode,
            )
            # Same API name as vLLM AsyncMPClient.pause_scheduler_async; routed
            # through collective_rpc so it runs on the orchestrator event loop.
            await self._engine_core_rpc(
                "pause_scheduler",
                stage_ids=ar_stage_ids,
                kwargs={"mode": mode, "clear_cache": clear_cache},
            )
        if diffusion_stage_ids:
            logger.info(
                "[%s] Pausing diffusion stage(s) %s via DiffusionEngine pause_scheduler(mode=keep)",
                self._name,
                diffusion_stage_ids,
            )
            await self._engine_core_rpc(
                "pause_scheduler",
                stage_ids=diffusion_stage_ids,
                kwargs={"mode": "keep"},
            )

        # Frontend / sender-side cache clear (P0). EngineCore.pause_scheduler
        # already clears AR-side caches when clear_cache=True.
        if clear_cache and 0 in ar_stage_ids:
            await self._clear_frontend_mm_cache()

    async def resume_generation(self, stage_ids: list[int] | None = None) -> None:
        """Resume generation after :meth:`pause_generation`."""
        ar_stage_ids, diffusion_stage_ids = self._split_stage_ids_by_type(stage_ids)
        if ar_stage_ids:
            logger.info("[%s] Resuming AR stage(s) %s via EngineCore", self._name, ar_stage_ids)
            await self._engine_core_rpc("resume_scheduler", stage_ids=ar_stage_ids)
            self._paused_stage_ids.difference_update(ar_stage_ids)
        diffusion_stage_ids = [sid for sid in diffusion_stage_ids if sid in self._paused_stage_ids]
        if diffusion_stage_ids:
            logger.info("[%s] Resuming diffusion stage(s) %s via DiffusionEngine", self._name, diffusion_stage_ids)
            await self._engine_core_rpc("resume_scheduler", stage_ids=diffusion_stage_ids)
            self._paused_stage_ids.difference_update(diffusion_stage_ids)

        if self._paused_stage_ids:
            # Reopening admission now would let new requests queue on a stage
            # whose scheduler is still closed.
            logger.info(
                "[%s] Admission stays paused: stage(s) %s are still paused",
                self._name,
                sorted(self._paused_stage_ids),
            )
            return

        async with self._pause_cond:
            self._paused = False
            self._hold_admission_until_resume = False
            self._pause_cond.notify_all()

    async def is_paused(self) -> bool:
        """Check if frontend admission is paused."""
        async with self._pause_cond:
            return self._paused

    # EngineClient exposes async operations; OmniBase implements the sync API.
    async def start_profile(  # type: ignore[override]
        self,
        profile_prefix: str | None = None,
        stages: list[int] | None = None,
        *,
        delay_iterations: int | None = None,
        max_iterations: int | None = None,
    ) -> list[Any]:
        """Start profiling specified stages.

        Uses vLLM-compatible profile(is_start=True, profile_prefix) interface.

        Args:
            profile_prefix: Optional prefix for the trace file names.
            stages: List of stage IDs to profile. If None, profiles all stages.
            delay_iterations: Optional per-session delayed start (vLLM compat).
            max_iterations: Optional per-session iteration cap (vLLM compat).
        """
        validate_profile_iteration_bounds(delay_iterations, max_iterations)
        # Mirror upstream ExecutorBase.profile: forward the per-session
        # controls as kwargs only when set, so workers without them keep
        # working with plain (is_start, profile_prefix) args.
        kwargs: dict[str, int] = {}
        if delay_iterations is not None:
            kwargs["delay_iterations"] = delay_iterations
        if max_iterations is not None:
            kwargs["max_iterations"] = max_iterations
        return await self.collective_rpc(
            method="profile",
            args=(True, profile_prefix),
            kwargs=kwargs or None,
            stage_ids=stages,
        )

    async def stop_profile(self, stages: list[int] | None = None) -> list[Any]:  # type: ignore[override]
        """Stop profiling specified stages.

        Uses vLLM-compatible profile(is_start=False) interface.

        Args:
            stages: List of stage IDs to profile. If None, stops all stages.
        """
        return await self.collective_rpc(method="profile", args=(False, None), stage_ids=stages)

    async def _clear_frontend_mm_cache(self) -> None:
        """Clear P0 through the renderer's serialized multimodal executor."""
        renderer = self.renderer
        if renderer is not None:
            await renderer.clear_mm_cache_async()

    async def reset_mm_cache(
        self, *, stage_ids: list[int] | None = None, timeout: float = CACHE_RESET_TIMEOUT_S
    ) -> None:
        """Reset sender and receiver MM caches on selected AR stages.

        By default all AR stages are selected; diffusion stages are skipped.
        Stage 0's sender is the frontend renderer. Downstream senders are
        cleared by the orchestrator before resetting their engine cores.
        Call while generation is paused to avoid racing new inputs.
        """
        ar_stage_ids, _ = self._split_stage_ids_by_type(stage_ids)
        if 0 in ar_stage_ids:
            await asyncio.wait_for(self._clear_frontend_mm_cache(), timeout=timeout)
        if ar_stage_ids:
            await self._engine_core_rpc("reset_mm_cache", stage_ids=ar_stage_ids, timeout=timeout)

    async def reset_encoder_cache(
        self, *, stage_ids: list[int] | None = None, timeout: float = CACHE_RESET_TIMEOUT_S
    ) -> None:
        """Reset encoder caches on selected AR stages (all by default).

        Diffusion stages are skipped. RPC failures are raised to the caller.
        """
        ar_stage_ids, _ = self._split_stage_ids_by_type(stage_ids)
        if ar_stage_ids:
            await self._engine_core_rpc("reset_encoder_cache", stage_ids=ar_stage_ids, timeout=timeout)

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
        *,
        stage_ids: list[int] | None = None,
        timeout: float = CACHE_RESET_TIMEOUT_S,
    ) -> bool:
        """Reset prefix caches on selected AR stages (all by default).

        Diffusion stages are skipped. Return False if a stage cannot reset
        its cache; unsupported operations and RPC failures raise instead of
        silently retaining KV computed under previous model weights.
        """
        ar_stage_ids, _ = self._split_stage_ids_by_type(stage_ids)
        if not ar_stage_ids:
            return True
        results = await self._engine_core_rpc(
            "reset_prefix_cache",
            stage_ids=ar_stage_ids,
            args=(reset_running_requests, reset_connector),
            timeout=timeout,
        )
        return all(self._coerce_stage_bool(result) for result in results)

    async def release_kv_cache_memory(
        self, *, stage_ids: list[int] | None = None, timeout: float = CACHE_RESET_TIMEOUT_S
    ) -> None:
        """Discard AR-stage KV memory after a completed generation pause.

        EngineCore checks that the scheduler is paused and all executor
        memory is resident. Diffusion stages have no KV cache to release.
        Restore KV memory with ``wake_up(tags=["kv_cache"])`` before resuming.
        """
        ar_stage_ids, _ = self._split_stage_ids_by_type(stage_ids)
        if not ar_stage_ids:
            return
        async with self._pause_cond:
            if not self._paused:
                raise RuntimeError("release_kv_cache_memory() requires a completed pause first")
            await self._pause_cond.wait_for(lambda: getattr(self, "_admitting", 0) == 0)
        if 0 in ar_stage_ids:
            await asyncio.wait_for(self._clear_frontend_mm_cache(), timeout=timeout)
        await self._engine_core_rpc("release_kv_cache_memory", stage_ids=ar_stage_ids, timeout=timeout)
        self._record_stage_sleep(ar_stage_ids, [CuMemTag.KV_CACHE.value])
        self._hold_admission_until_resume = True

    async def sleep(
        self, stage_ids: list[int] | None = None, level: int = 2, mode: PauseMode = "abort"
    ) -> list[OmniACK]:
        """Put stages to sleep.

        AR/LLM stages use EngineCore.sleep (pause scheduler, wait idle, then
        offload/discard memory) — matching vLLM AsyncLLM.sleep.

        Diffusion stages keep the worker-level handle_sleep_task RPC, which
        does not stop the DiffusionEngine scheduler; quiesce a busy diffusion
        stage first with ``pause_generation(mode="keep")`` (or abort it).

        Frontend admission is blocked at the start of this call (``_paused``)
        so pipelined :meth:`generate` cannot race into stages while sleep is
        in flight. This does **not** invoke EngineCore.pause_scheduler again
        (sleep already pauses the AR scheduler).

        For AR / mixed engines, ``wake_up`` does **not** clear ``_paused``;
        callers must :meth:`resume_generation` when ready (typical trainer
        order: pause → abort → sleep → train → wake → resume). Diffusion-only
        engines have no EngineCore pause to hold, so ``wake_up`` restores
        admission and ``sleep → wake → generate`` keeps working.
        """
        # Validate before touching admission state. Raising after
        # ``_paused = True`` would wedge generate() for good: nothing was
        # slept, so wake_up() has no tags to restore and never reaches the
        # block that clears ``_paused``.
        ar_stage_ids, diffusion_stage_ids = self._split_stage_ids_by_type(stage_ids)

        # Block admission before any sleep RPC so generate() waits on
        # _pause_cond during the drain/offload window. Wait until generate()
        # coroutines that already passed the pause check have submitted (or
        # failed) so EngineCore does not see ADD frames while sleeping.
        async with self._pause_cond:
            self._paused = True
            # Cancel inside the lock, after the gate closed (matching
            # pause_generation's ordering): a generate that has not passed
            # the gate cannot register a submit this snapshot would miss.
            if mode == "abort":
                self._cancel_pending_submits()
            await self._pause_cond.wait_for(lambda: getattr(self, "_admitting", 0) == 0)
        # EngineCore.sleep resets receiver caches itself; only clear P0 here.
        if 0 in ar_stage_ids:
            await self._clear_frontend_mm_cache()

        self._final_output_handler()
        sleep_tags = [CuMemTag.WEIGHTS.value, CuMemTag.KV_CACHE.value]
        final_acks: list[OmniACK] = []
        if ar_stage_ids:
            self._hold_admission_until_resume = True
            logger.info(
                "[%s] Sleeping AR stage(s) %s via EngineCore.sleep(level=%s, mode=%s)",
                self._name,
                ar_stage_ids,
                level,
                mode,
            )
            await self._engine_core_rpc(
                "sleep",
                stage_ids=ar_stage_ids,
                args=(level, mode),
            )
            # EngineCore.sleep has no OmniACK handshake; emit stage-level SUCCESS
            # markers so callers/tests that count ACKs keep a stable API.
            task_id = f"engine_core-sleep-{uuid.uuid4().hex[:8]}"
            final_acks.extend(
                OmniACK(
                    task_id=task_id,
                    status="SUCCESS",
                    stage_id=sid,
                    rank=0,
                    metadata={"path": "engine_core", "level": level, "mode": mode},
                )
                for sid in ar_stage_ids
            )
            self._record_stage_sleep(ar_stage_ids, sleep_tags)
            if level == 2:
                self._level2_sleeping = True

        if diffusion_stage_ids:
            acks: list[OmniACK] = []
            try:
                acks = await self._sleep_diffusion(diffusion_stage_ids, level)
            finally:
                # A failed sleep may have released part of a stage, so wake_up must still reach it.
                self._record_stage_sleep(diffusion_stage_ids, sleep_tags)
                # Level 2 has discarded weights only where a replica finished its sleep.
                if level == 2 and any(self._diffusion_ack_error(ack) is None for ack in acks):
                    self._level2_sleeping = True
            self._raise_on_diffusion_errors("handle_sleep_task", acks)
            final_acks.extend(acks)

        return final_acks

    async def _sleep_diffusion(self, stage_ids: list[int], level: int) -> list[OmniACK]:
        """Worker-level sleep RPC for diffusion stages only."""
        # Diffusion reports one summary ACK at rank 0 regardless of TP.
        total_workers = len(stage_ids)
        task_id = str(uuid.uuid4())
        self.event_resolver.watch_task(task_id, expected_count=total_workers)
        logger.info("[%s] Sleep (diffusion) initiated (Task: %s).", self._name, task_id)
        task = OmniSleepTask(level=level, task_id=task_id)
        rpc_results = await self.collective_rpc(method="handle_sleep_task", args=(task,), stage_ids=stage_ids)
        return await self._resolve_diffusion_acks(rpc_results)

    async def _resolve_diffusion_acks(self, rpc_results: list[Any]) -> list[OmniACK]:
        """Resolve the ACKs of a diffusion worker RPC."""
        final_acks: list[OmniACK] = []
        for stage_res in rpc_results:
            worker_acks = stage_res if isinstance(stage_res, list) else [stage_res]
            for ack in worker_acks:
                if ack is None:
                    continue
                # StagePool's result for a failed RPC has no task_id to resolve.
                if not isinstance(ack, dict) or "task_id" in ack:
                    await self.event_resolver.resolve(ack)
                final_acks.append(ack)
        return final_acks

    @classmethod
    def _raise_on_diffusion_errors(cls, method: str, acks: list[OmniACK]) -> None:
        errors = [error for ack in acks if (error := cls._diffusion_ack_error(ack))]
        if errors:
            raise RuntimeError(f"{method} failed: {'; '.join(errors)}")

    @staticmethod
    def _diffusion_ack_error(ack: OmniACK | dict[str, Any]) -> str | None:
        # In-process stages return OmniACK, subprocess stages return its dict form,
        # and StagePool returns {"supported": False, "error": ...} when the RPC failed.
        if isinstance(ack, dict):
            status, error_msg, rpc_error = ack.get("status"), ack.get("error_msg"), ack.get("error")
            # A subprocess RPC error arrives as {"error": True, "reason": ...}.
            if rpc_error is True:
                rpc_error = ack.get("reason") or "RPC failed"
            if not rpc_error and (ack.get("todo") or ack.get("supported") is False):
                rpc_error = str(ack)
        else:
            status, error_msg, rpc_error = getattr(ack, "status", None), getattr(ack, "error_msg", None), None
        if status == "ERROR":
            return error_msg or "worker reported ERROR"
        return rpc_error

    async def wake_up(self, stage_ids: list[int] | None = None, tags: list[str] | None = None) -> list[OmniACK]:
        """Wake stages after sleep.

        AR/LLM stages use EngineCore.wake_up (restore memory, auto-resume
        scheduler). Diffusion stages keep the worker-level wake RPC.

        Does **not** clear the frontend ``_paused`` admission gate when
        :meth:`pause_generation` ran or AR stages were slept — call
        :meth:`resume_generation` when the trainer is ready to admit new
        requests. Diffusion-only ``sleep`` uses ``_paused`` only as a race
        guard; this method restores admission after a successful wake.
        """
        self._final_output_handler()

        if getattr(self, "_level2_sleeping", False):
            raise NotImplementedError(
                "wake_up() after sleep(level=2) is not yet implemented: weights were "
                "discarded from GPU and reloading from disk is not yet supported. "
                "Use sleep(level=1) instead, which offloads weights to CPU RAM "
                "and supports fast DMA restore."
            )
        ar_stage_ids, diffusion_stage_ids = self._split_stage_ids_by_type(stage_ids)
        target_stage_ids = ar_stage_ids + diffusion_stage_ids
        _current_tags = self._sleeping_tags_for_stages(target_stage_ids)
        per_stage = getattr(self, "_stage_sleeping_tags", None) or {}
        if not _current_tags and not per_stage:
            _current_tags = set(getattr(self, "_sleeping_tags", set()))
        if tags is None:
            requested_tags = list(_current_tags)
        else:
            requested_tags = [t for t in tags if t in _current_tags]
        if not requested_tags:
            logger.info(f"[{self._name}] Requested tags {tags} are already warm. Skipping wake_up.")
            return []

        final_acks: list[OmniACK] = []
        if ar_stage_ids:
            logger.info("[%s] Waking AR stage(s) %s via EngineCore", self._name, ar_stage_ids)
            await self._engine_core_rpc(
                "wake_up",
                stage_ids=ar_stage_ids,
                kwargs={"tags": requested_tags},
            )
            task_id = f"engine_core-wake-{uuid.uuid4().hex[:8]}"
            final_acks.extend(
                OmniACK(
                    task_id=task_id,
                    status="SUCCESS",
                    stage_id=sid,
                    rank=0,
                    metadata={"path": "engine_core", "tags": list(requested_tags)},
                )
                for sid in ar_stage_ids
            )
            self._clear_stage_sleep(ar_stage_ids, requested_tags)

        for sid in diffusion_stage_ids:
            final_acks.extend(await self._wake_diffusion([sid], requested_tags))
            self._clear_stage_sleep([sid], requested_tags)
        if diffusion_stage_ids:
            await asyncio.sleep(0.1)

        # Only clear the level-2 flag once all tags are warm, in case partial
        # wake support (e.g. tags=["kv_cache"] only) is added in the future.
        if not getattr(self, "_sleeping_tags", None):
            self._level2_sleeping = False
        logger.info(
            "[%s] Wake-up complete for stage(s) %s.",
            self._name,
            ar_stage_ids + diffusion_stage_ids,
        )
        # generate() coroutines parked on the full wake (sleeping tags in the
        # admission-gate predicate) may now pass the gate (#8678).
        async with self._pause_cond:
            self._pause_cond.notify_all()
        # Diffusion-only sleep uses `_paused` as a race guard. Restore
        # generate() admission after memory is back. AR/mixed sleep and
        # pause_generation keep the trainer hold until resume_generation.
        if not getattr(self, "_hold_admission_until_resume", False):
            async with self._pause_cond:
                self._paused = False
                self._pause_cond.notify_all()
        return final_acks

    async def _wake_diffusion(self, stage_ids: list[int], requested_tags: list[str]) -> list[OmniACK]:
        """Worker-level wake RPC for diffusion stages only."""
        total_workers = len(stage_ids)
        task_id = str(uuid.uuid4())
        self.event_resolver.watch_task(task_id, expected_count=total_workers)
        logger.info("[%s] Wake-up (diffusion) initiated (Task: %s).", self._name, task_id)
        task = OmniWakeTask(tags=requested_tags, task_id=task_id)
        rpc_results = await self.collective_rpc(method="handle_wake_task", args=(task,), stage_ids=stage_ids)
        acks = await self._resolve_diffusion_acks(rpc_results)
        self._raise_on_diffusion_errors("handle_wake_task", acks)
        return acks

    async def is_sleeping(self, stage_ids: list[int] | None = None) -> bool:
        """Return whether all stages are sleeping, or with ``stage_ids``,
        whether any of those stages is recorded as sleeping.

        TODO(AsyncOmni): query the orchestrator once all stage backends expose
        a real sleeping-state RPC. For now we track the requested state locally.
        """
        if stage_ids is not None:
            return bool(self._sleeping_tags_for_stages(stage_ids))
        return bool(getattr(self, "_sleeping_tags", None))

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into all stages.

        Returns True only if all concretely-implemented stages report success.
        """
        results = await self.collective_rpc(method="add_lora", args=(lora_request,))
        concrete_results = [r for r in results if not (isinstance(r, dict) and r.get("todo"))]
        return all(self._coerce_stage_bool(r) for r in concrete_results) if concrete_results else False

    async def remove_lora(self, adapter_id: int) -> bool:
        """Remove a LoRA adapter from all stages.

        TODO(AsyncOmni): add richer per-stage error reporting to the public API.
        """
        results = await self.collective_rpc(method="remove_lora", args=(adapter_id,))
        concrete_results = [r for r in results if not (isinstance(r, dict) and r.get("todo"))]
        return all(self._coerce_stage_bool(r) for r in concrete_results) if concrete_results else False

    async def list_loras(self) -> list[int]:
        """List all loaded LoRA adapter IDs across stages."""
        results = await self.collective_rpc(method="list_loras")
        merged: set[int] = set()
        for result in results:
            if isinstance(result, dict) and result.get("todo"):
                continue
            if isinstance(result, (list, set)):
                for item in result:
                    if isinstance(item, (list, set)):
                        merged.update(item)
                    elif isinstance(item, int):
                        merged.add(item)
            elif isinstance(result, int):
                merged.add(result)
        return sorted(merged)

    async def pin_lora(self, adapter_id: int) -> bool:
        """Pin a LoRA adapter across stages."""
        results = await self.collective_rpc(method="pin_lora", args=(adapter_id,))
        concrete_results = [r for r in results if not (isinstance(r, dict) and r.get("todo"))]
        return all(self._coerce_stage_bool(r) for r in concrete_results) if concrete_results else False

    # ==================== Properties ====================

    async def get_input_preprocessor(self) -> InputProcessor:
        """Get input preprocessor."""
        return self.input_processor

    async def get_tokenizer(self) -> TokenizerLike:
        """Get tokenizer for the comprehension stage."""
        stage_index = self._get_comprehension_stage_index()
        if stage_index is not None:
            processor = self.engine.output_processors[stage_index]
            assert processor is not None
            tokenizer = processor.tokenizer
            if tokenizer is not None:
                return tokenizer
        processor = self.input_processor
        assert processor is not None
        return processor.tokenizer

    async def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return False

    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Notify engine that a KV-transfer request was rejected before admission.

        Omni does not currently use KV-transfer pre-admission resources,
        so this is a no-op.
        """
        logger.debug(
            "KV-transfer request rejected (no-op in omni): request_id=%s",
            request_id,
        )

    def _weight_transfer_stage_ids(self) -> list[int]:
        """Positional stage indices that have weight transfer configured.

        Mirrors the deploy semantics: pipeline-wide ``weight_transfer_config``
        enables every stage; a stage-level override restricts it to that stage.
        Stages without the feature must not receive lifecycle RPCs — their
        workers would reject the methods.

        Stage types expose the resolved config differently:
        - AR/generation: ``pool.stage_vllm_config`` is an upstream
          ``VllmConfig`` (its ``weight_transfer_config`` field).
        - Diffusion (inline): ``pool.stage_client.od_config``.
        - Diffusion (subprocess): ``pool.stage_client.proc_manager.od_config``.
        """
        enabled: list[int] = []
        for index, pool in enumerate(getattr(self.engine, "stage_pools", [])):
            cfg = getattr(pool, "stage_vllm_config", None)
            if getattr(cfg, "weight_transfer_config", None) is not None:
                enabled.append(index)
                continue
            diffusion_cfg = getattr(cfg, "diffusion_config", None)
            if getattr(diffusion_cfg, "weight_transfer_config", None) is not None:
                enabled.append(index)
                continue
            client = getattr(pool, "stage_client", None)
            od_config = getattr(client, "od_config", None)
            if od_config is None:
                od_config = getattr(getattr(client, "proc_manager", None), "od_config", None)
            if getattr(od_config, "weight_transfer_config", None) is not None:
                enabled.append(index)
        return enabled

    @property
    def weight_transfer_enabled(self) -> bool:
        """Whether any stage was configured with weight transfer.

        Used to gate the weight-transfer HTTP routes: the admin endpoints are
        only mounted when the feature is actually enabled via
        ``weight_transfer_config`` in the deploy config.
        """
        return bool(self._weight_transfer_stage_ids())

    async def _weight_transfer_rpc(self, method: str, args: tuple[Any, ...] = ()) -> list[Any]:
        """Run a weight-transfer lifecycle RPC on the enabled stages, failing on errors.

        ``StagePool.collective_rpc`` reports per-stage failures as
        ``{"supported": False, "error": ...}`` result dicts instead of raising,
        so a naive call could return 200 to the HTTP client while one stage
        silently failed. Any error result therefore raises here; each worker's
        lifecycle methods reset their own per-stage state on failure, so the
        surviving stages are left in a consistent idle state and the trainer
        can retry the full protocol.
        """
        stage_ids = self._weight_transfer_stage_ids()
        if not stage_ids:
            raise RuntimeError("No stage has weight transfer enabled. Set weight_transfer_config in the deploy config.")
        results = await self.collective_rpc(method=method, args=args, stage_ids=stage_ids)
        for result in results:
            if isinstance(result, dict) and (result.get("error") is not None or result.get("supported") is False):
                raise RuntimeError(f"{method} failed on a stage: {result.get('error', result)}")
        return results

    async def init_weight_transfer_engine(self, init_info: dict) -> None:
        """Initialize weight transfer mechanism across all stages.

        Args:
            init_info: Backend-specific initialization information
        """
        await self._weight_transfer_rpc("init_weight_transfer_engine", (init_info,))
        logger.info("[%s] Weight transfer engine initialized", self._name)

    async def start_weight_update(self) -> None:
        """Start a new weight update session across configured stages.

        Each worker delegates preparation to its configured WTE backend,
        which owns whether the update uses layerwise reload.
        """
        await self._weight_transfer_rpc("start_weight_update")
        logger.info("[%s] Weight update session started", self._name)

    async def update_weights(self, update_info: dict) -> None:
        """Send weight update chunk to all stages.

        Args:
            update_info: Backend-specific update information containing weight data
        """
        await self._weight_transfer_rpc("update_weights", (update_info,))
        logger.debug("[%s] Weight chunk forwarded to stages", self._name)

    async def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the current weight update session across all stages.

        Calls ``finish_weight_update()`` on every stage's worker, finalizing
        the weight transfer and resetting the update target.

        Args:
            weight_version: Accepted for upstream ``EngineClient`` protocol
                compatibility (RLHF weight-transfer routers pass it positionally).
                Currently unused by omni stages.
        """
        await self._weight_transfer_rpc("finish_weight_update")
        logger.info("[%s] Weight update session finished", self._name)

    async def do_log_stats(self) -> None:
        """Log statistics.

        TODO: Forward to Orchestrator process via message.
        """
        pass

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Return the task set exposed by the orchestrator-backed engine."""
        return tuple(self.engine.supported_tasks)
