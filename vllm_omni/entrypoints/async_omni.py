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
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from vllm import TokensPrompt
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
from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.entrypoints.async_omni_base import AsyncOmniBase
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.errors import client_error_metadata
from vllm_omni.inputs.data import OmniSamplingParams
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.v1.engine import PauseMode
    from vllm.v1.engine.input_processor import InputProcessor

    from vllm_omni.inputs.data import OmniInteractionPrompt, OmniPromptType

logger = init_logger(__name__)


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
        # True after pause_generation() or AR EngineCore sleep; wake_up must
        # not reopen generate() until resume_generation(). Diffusion-only
        # sleep uses _paused as a temporary admission gate and clears it
        # on wake so sleep → wake → generate keeps working.
        self._hold_admission_until_resume: bool = False
        self._sleeping_tags: set[str] = set()
        self._stage_sleeping_tags: dict[int, set[str]] = {}
        self._level2_sleeping: bool = False

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
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        arrival_time: float | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt(s) asynchronously.

        Coordinates multi-stage pipeline execution. Processes the prompt
        through all stages in the pipeline and yields outputs as they become
        available.

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
        streaming_input = isinstance(prompt, AsyncGenerator)
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)
            if not streaming_input:
                self._admitting = getattr(self, "_admitting", 0) + 1
        admitting = not streaming_input

        logger.debug(f"[AsyncOmni] generate() called for request {external_request_id}")

        input_stream_task: asyncio.Task | None = None
        try:
            _sleeping_tags = getattr(self, "_sleeping_tags", None)
            if _sleeping_tags:
                raise RuntimeError(
                    f"Generation rejected: Engine is partially or fully asleep. "
                    f"Currently sleeping tags: {list(_sleeping_tags)}. "
                    f"Please perform a full wake_up before generating."
                )

            # Reject diffusion list-prompt early with a clear API error.
            if isinstance(prompt, list) and any(
                getattr(client, "stage_type", "") == "diffusion" for client in getattr(self.engine, "stage_clients", [])
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
            )
            req_state.metrics = metrics
            req_state.request_arrival_ts = wall_start_ts
            self.request_states[request_id] = req_state

            # PD disaggregation: modify prefill-stage sampling params per request
            req_sp_list = list(sampling_params_list)
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
                    input_stream=prompt,
                    sampling_params_list=req_sp_list,
                    final_stage_id=final_stage_id_for_e2e,
                    final_output_stage_ids=final_output_stage_ids,
                    arrival_time=wall_start_ts,
                    lora_request=lora_request,
                    first_chunk_submitted=first_chunk_submitted,
                )
                await first_chunk_submitted
            else:
                await self.engine.add_request_async(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params_list=req_sp_list,
                    final_stage_id=final_stage_id_for_e2e,
                    final_output_stage_ids=final_output_stage_ids,
                    arrival_time=wall_start_ts,
                    lora_request=lora_request,
                )
            submit_ts = time.time()
            req_state.metrics.stage_first_ts[0] = submit_ts
            req_start_ts[request_id] = submit_ts
            if admitting:
                await self._release_generate_admission()
                admitting = False
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

            logger.debug(f"[AsyncOmni] Request {request_id} completed")

        except (asyncio.CancelledError, GeneratorExit):
            self._record_request_failure_once(request_id, reason="client_disconnect")
            await self._abort_internal_requests(request_id)
            logger.info(f"[AsyncOmni] Request {request_id} aborted.")
            raise
        except Exception as e:
            self._record_request_failure_once(request_id, reason="stage_error")
            await self._abort_internal_requests(request_id)
            logger.info(f"[AsyncOmni] Request {request_id} failed (input error): {e}")
            raise
        finally:
            if input_stream_task is not None and not input_stream_task.done():
                input_stream_task.cancel()
            if admitting:
                await self._release_generate_admission()
            self._log_summary_and_cleanup(request_id)

    async def _release_generate_admission(self) -> None:
        """Drop one in-flight generate admission slot held across add_request."""
        async with self._pause_cond:
            self._admitting = max(getattr(self, "_admitting", 1) - 1, 0)
            self._pause_cond.notify_all()

    async def _submit_with_admission(self, awaitable):
        """Wait for resume, hold one admission slot for a single EngineCore submit."""
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)
            self._admitting = getattr(self, "_admitting", 0) + 1
        try:
            return await awaitable
        finally:
            await self._release_generate_admission()

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
    ) -> asyncio.Task:
        """Submit a streaming input generator as incremental stage-0 updates."""
        if not sampling_params_list:
            raise ValueError("sampling_params_list cannot be empty for streaming input")
        # only check thinker's sampling params now
        stage0_params = sampling_params_list[0]
        self._validate_streaming_input_sampling_params(stage0_params)
        req_state = self.request_states[request_id]
        has_submitted_first_chunk = False

        # NOTE: InputProcessor in vLLM should generally do this too, but for
        # now we do it defensively. TODO (Alex) ensure clones/copying are optimized
        if not stage0_params.skip_clone:
            stage0_params = stage0_params.clone()
            stage0_params.skip_clone = True

        def _mark_first_chunk_submitted() -> None:
            if first_chunk_submitted is not None and not first_chunk_submitted.done():
                first_chunk_submitted.set_result(None)

        async def handle_inputs() -> None:
            nonlocal has_submitted_first_chunk
            cancelled = False
            try:
                async for chunk in input_stream:
                    chunk_params = getattr(chunk, "sampling_params", None) or stage0_params
                    self._validate_streaming_input_sampling_params(chunk_params)
                    chunk_sampling_params_list = list(sampling_params_list)
                    chunk_sampling_params_list[0] = chunk_params
                    chunk_prompt = chunk.prompt
                    prompt_text, _, _ = extract_prompt_components(self.model_config, chunk_prompt)

                    if not has_submitted_first_chunk:
                        await self._submit_with_admission(
                            self.engine.add_request_async(
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
                        has_submitted_first_chunk = True
                        _mark_first_chunk_submitted()
                    else:
                        await self._submit_with_admission(
                            self.engine.add_streaming_update_async(
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
            except (asyncio.CancelledError, GeneratorExit):
                cancelled = True
            except Exception as error:
                status_code, error_type = client_error_metadata(error)
                await req_state.queue.put(
                    ErrorMessage(
                        request_id=request_id,
                        error=str(error),
                        status_code=status_code,
                        error_type=error_type,
                    )
                )
            finally:
                try:
                    if not cancelled:
                        # Send empty final request to indicate that inputs have
                        # finished. Don't send if canceled (session was aborted).
                        final_sampling_params_list = list(sampling_params_list)
                        final_sampling_params_list[0] = stage0_params
                        final_prompt = TokensPrompt(prompt_token_ids=[0])

                        if has_submitted_first_chunk:
                            await self._submit_with_admission(
                                self.engine.add_streaming_update_async(
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
                            await self._submit_with_admission(
                                self.engine.add_request_async(
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
                            has_submitted_first_chunk = True
                finally:
                    # Unblock generate() even on cancel / empty stream / submit
                    # failure so it can observe a terminal abort or empty result.
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

        unsupported_stage_ids: list[int] = []
        effective_stage_ids = stage_ids or list(range(len(results)))
        for index, result in enumerate(results):
            if isinstance(result, dict) and result.get("todo"):
                unsupported_stage_ids.append(effective_stage_ids[index])

        if unsupported_stage_ids:
            logger.warning(
                "[AsyncOmni] collective_rpc(%s) has TODO support on stage(s): %s",
                method,
                unsupported_stage_ids,
            )

        return results

    async def _engine_core_rpc(
        self,
        method: str,
        *,
        stage_ids: list[int],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Call an AR EngineCore helper via collective_rpc (orchestrator loop).

        StagePool resolves ``{method}_async`` on the AR client when present
        (vLLM AsyncMPClient convention). Raises if any replica reports failure.
        """
        results = await self.collective_rpc(
            method=method,
            args=args,
            kwargs=kwargs,
            stage_ids=stage_ids,
        )
        for result in results:
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(f"{method} failed: {result['error']}")
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

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort request(s) via the Orchestrator."""
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        # Map the external user request IDs to internal IDs used by the Orchestrator.
        # NOTE: If the user request_id matches multiple requests, all of them will be
        # aborted. This is also what happens in this case in vLLM's output processor.
        internal_ids = [s.request_id for s in self.request_states.values() if s.external_request_id in request_ids]
        await self._abort(internal_ids)

    async def submit_interaction_async(
        self,
        request_id: str,
        *,
        interaction: OmniInteractionPrompt,
    ) -> None:
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

        await self.engine.submit_interaction_async(
            internal_ids[0],
            interaction=interaction,
        )
        if self.log_stats:
            logger.info("[AsyncOmni] Queued interaction for request %s", request_id)

    def _split_stage_ids_by_type(self, stage_ids: list[int] | None = None) -> tuple[list[int], list[int]]:
        """Split stage ids into AR/LLM (EngineCore) vs diffusion (worker RPC)."""
        n_stages = len(self.engine.stage_clients)
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
            client = self.engine.stage_clients[sid]
            if getattr(client, "stage_type", "llm") == "diffusion":
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
        3. Diffusion stages have no EngineCore scheduler — only frontend
           admission is paused for them.

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

        ar_stage_ids, _diffusion_stage_ids = self._split_stage_ids_by_type(stage_ids)
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

        # Frontend / sender-side cache clear (P0). EngineCore.pause_scheduler
        # already clears AR-side caches when clear_cache=True.
        if clear_cache:
            await self.reset_prefix_cache(
                reset_running_requests=not wait_for_inflight_requests,
                reset_connector=True,
            )
            await self.reset_mm_cache()
            await self.reset_encoder_cache()

    async def resume_generation(self, stage_ids: list[int] | None = None) -> None:
        """Resume generation after :meth:`pause_generation`."""
        ar_stage_ids, _diffusion_stage_ids = self._split_stage_ids_by_type(stage_ids)
        if ar_stage_ids:
            logger.info("[%s] Resuming AR stage(s) %s via EngineCore", self._name, ar_stage_ids)
            await self._engine_core_rpc("resume_scheduler", stage_ids=ar_stage_ids)

        async with self._pause_cond:
            self._paused = False
            self._hold_admission_until_resume = False
            self._pause_cond.notify_all()

    async def is_paused(self) -> bool:
        """Check if frontend admission is paused."""
        async with self._pause_cond:
            return self._paused

    async def start_profile(
        self,
        profile_prefix: str | None = None,
        stages: list[int] | None = None,
    ) -> list[Any]:
        """Start profiling specified stages.

        Uses vLLM-compatible profile(is_start=True, profile_prefix) interface.

        Args:
            profile_prefix: Optional prefix for the trace file names.
            stages: List of stage IDs to profile. If None, profiles all stages.
        """
        return await self.collective_rpc(method="profile", args=(True, profile_prefix), stage_ids=stages)

    async def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        """Stop profiling specified stages.

        Uses vLLM-compatible profile(is_start=False) interface.

        Args:
            stages: List of stage IDs to profile. If None, stops all stages.
        """
        return await self.collective_rpc(method="profile", args=(False, None), stage_ids=stages)

    async def reset_mm_cache(self) -> None:
        """Reset the frontend (P0) multimodal processor cache.

        ``EngineCore.sleep(level>=1)`` already clears the P1 receiver cache.
        Clearing P0 avoids hash-only follow-up requests after that reset.
        """
        renderer = self.renderer
        if renderer is not None:
            await renderer.clear_mm_cache_async()

    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache for all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmni] reset_encoder_cache not yet supported with Orchestrator process")

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        """Reset the prefix cache for all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmni] reset_prefix_cache not yet supported with Orchestrator process")
        return True

    async def sleep(
        self, stage_ids: list[int] | None = None, level: int = 2, mode: PauseMode = "abort"
    ) -> list[OmniACK]:
        """Put stages to sleep.

        AR/LLM stages use EngineCore.sleep (pause scheduler, wait idle, then
        offload/discard memory) — matching vLLM AsyncLLM.sleep.

        Diffusion stages keep the existing worker-level handle_sleep_task RPC
        because StageDiffusionProc does not expose EngineCore.pause_scheduler.

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
        # Block admission before any sleep RPC so generate() waits on
        # _pause_cond during the drain/offload window. Wait until generate()
        # coroutines that already passed the pause check have submitted (or
        # failed) so EngineCore does not see ADD frames while sleeping.
        async with self._pause_cond:
            self._paused = True
            await self._pause_cond.wait_for(lambda: getattr(self, "_admitting", 0) == 0)

        # P0 sender cache must drop hashes before EngineCore.sleep clears P1.
        await self.reset_mm_cache()

        self._final_output_handler()
        ar_stage_ids, diffusion_stage_ids = self._split_stage_ids_by_type(stage_ids)
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

        if diffusion_stage_ids:
            final_acks.extend(await self._sleep_diffusion(diffusion_stage_ids, level))

        self._record_stage_sleep(
            ar_stage_ids + diffusion_stage_ids,
            [CuMemTag.WEIGHTS.value, CuMemTag.KV_CACHE.value],
        )
        if level == 2:
            self._level2_sleeping = True
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
        final_acks: list[OmniACK] = []
        for stage_res in rpc_results:
            worker_acks = stage_res if isinstance(stage_res, list) else [stage_res]
            for ack in worker_acks:
                if ack is not None:
                    await self.event_resolver.resolve(ack)
                    final_acks.append(ack)
        return final_acks

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

        if diffusion_stage_ids:
            final_acks.extend(await self._wake_diffusion(diffusion_stage_ids, requested_tags))

        self._clear_stage_sleep(target_stage_ids, requested_tags)
        # Only clear the level-2 flag once all tags are warm, in case partial
        # wake support (e.g. tags=["kv_cache"] only) is added in the future.
        if not getattr(self, "_sleeping_tags", None):
            self._level2_sleeping = False
        logger.info(
            "[%s] Wake-up complete for stage(s) %s.",
            self._name,
            ar_stage_ids + diffusion_stage_ids,
        )
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
        final_acks: list[OmniACK] = []
        for stage_res in rpc_results:
            worker_acks = stage_res if isinstance(stage_res, list) else [stage_res]
            for ack in worker_acks:
                if ack is not None:
                    await self.event_resolver.resolve(ack)
                    final_acks.append(ack)
        await asyncio.sleep(0.1)
        return final_acks

    async def is_sleeping(self) -> bool:
        """Return whether all stages are sleeping.

        TODO(AsyncOmni): query the orchestrator once all stage backends expose
        a real sleeping-state RPC. For now we track the requested state locally.
        """
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
            tokenizer = self.engine.output_processors[stage_index].tokenizer
            if tokenizer is not None:
                return tokenizer
        return self.input_processor.tokenizer  # type: ignore[return-value]

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

    async def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """Start a new weight update.

        Omni does not currently support weight transfer, so this is a no-op.
        """
        logger.debug("Weight update start requested (no-op in omni)")

    async def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the current weight update.

        Omni does not currently support weight transfer, so this is a no-op.
        ``weight_version`` is accepted for upstream ``EngineClient`` protocol
        compatibility (RLHF weight-transfer routers pass it positionally).
        """
        logger.debug("Weight update finish requested (no-op in omni)")

    async def do_log_stats(self) -> None:
        """Log statistics.

        TODO: Forward to Orchestrator process via message.
        """
        pass

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Return the task set exposed by the orchestrator-backed engine."""
        return tuple(self.engine.supported_tasks)
