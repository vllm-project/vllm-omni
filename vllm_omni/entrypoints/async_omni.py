"""
AsyncOmni - Refactored async orchestrator using AsyncOmniEngine.

This is the new implementation that uses AsyncOmniEngine (which manages
StageEngineCoreClient instances) instead of OmniStage with worker processes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from vllm import TokensPrompt
from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.tokenizers import TokenizerLike
    from vllm.v1.engine import PauseMode

    from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams

logger = init_logger(__name__)
_FINAL_OUTPUT_IDLE_SLEEP_S = 0.001


class AsyncOmni(EngineClient, OmniBase):
    """Asynchronous unified entry point for multi-stage pipelines using AsyncOmniEngine.

    This is the refactored version that uses AsyncOmniEngine instead of
    OmniStage workers. It provides the same interface as AsyncOmni but with
    a cleaner architecture.

    Args:
        model: Model name or path to load.
        **kwargs: Additional keyword arguments.
            - stage_configs_path: Optional path to YAML file containing stage
              configurations. If None, configurations are resolved from model
              pipeline factory.
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

    def __init__(self, *args: Any, model: str = "", **kwargs: Any) -> None:
        OmniBase.__init__(self, model=model, **kwargs)
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False
        self._is_sleeping: bool = False
        self.final_output_task: asyncio.Task | None = None

        self.config_path = self.engine.config_path
        self.stage_configs = self.engine.stage_configs
        self.tts_max_instructions_length = kwargs.get("tts_max_instructions_length", None)
        self.input_processor = self.engine.input_processor

        stage_index = self._get_comprehension_stage_index()
        if stage_index is None:
            self.io_processor = None
        else:
            vllm_config = self.engine.stage_vllm_configs[stage_index]
            io_processor_plugin = vllm_config.model_config.io_processor_plugin
            renderer = self.renderer
            if renderer is None:
                from vllm.renderers import renderer_from_config

                renderer = renderer_from_config(vllm_config)
            self.io_processor = get_io_processor(vllm_config, renderer, io_processor_plugin)

    def _get_comprehension_stage_index(self) -> int | None:
        fallback_idx: int | None = None
        for idx, stage_client in enumerate(self.engine.stage_clients):
            stage_vllm_config = self.engine.stage_vllm_configs[idx]
            if stage_vllm_config is None:
                continue
            if fallback_idx is None:
                fallback_idx = idx
            if stage_client.is_comprehension:
                return idx
        return fallback_idx

    @property
    def renderer(self):
        """Return the renderer from the engine input processor when available."""
        if self.input_processor is None:
            return None
        return self.input_processor.renderer

    @property
    def vllm_config(self):
        """Return the vLLM config for the comprehension stage when present."""
        stage_index = self._get_comprehension_stage_index()
        if stage_index is None:
            return None
        return self.engine.stage_vllm_configs[stage_index]

    async def get_vllm_config(self) -> Any:
        """Compatibility helper for call sites expecting async vllm config access."""
        return self.vllm_config

    def get_diffusion_od_config(self) -> Any | None:
        """Return the diffusion-stage config when the pipeline has one."""
        for stage_client in self.engine.stage_clients:
            if getattr(stage_client, "stage_type", None) != "diffusion":
                continue

            od_config = getattr(stage_client, "od_config", None)
            if od_config is not None:
                return od_config

            inner_engine = getattr(stage_client, "_engine", None)
            od_config = getattr(inner_engine, "od_config", None)
            if od_config is not None:
                return od_config

        return None

    @property
    def model_config(self):
        """Return the model config for the comprehension stage when present."""
        vllm_config = self.vllm_config
        if vllm_config is None:
            return None
        return vllm_config.model_config

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
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt(s) asynchronously.

        Coordinates multi-stage pipeline execution. Processes the prompt
        through all stages in the pipeline and yields outputs as they become
        available.

        **Batch mode (diffusion only):**
        When *prompt* is a ``list``, all prompts are dispatched in a single
        ``DiffusionEngine.step()`` call at the diffusion stage.  The combined
        result is yielded as one ``OmniRequestOutput`` with all generated
        images.  Only a single *request_id* is used for the whole batch.

        Args:
            prompt: A single prompt **or** a list of prompts.  A list
                triggers batch mode when the diffusion stage is reached.
            request_id: Unique identifier for this request.
            sampling_params_list: List of SamplingParams, one per stage.
                Must have the same length as the number of stages.
                If *None*, uses default sampling params for each stage.
            output_modalities: Optional list of output modalities.

        Yields:
            OmniRequestOutput objects as they are produced by each stage.
            In batch mode the diffusion stage yields one output containing
            all generated images.

        Raises:
            ValueError: If sampling_params_list has incorrect length.
        """
        # Wait until generation is resumed if the engine is paused
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        logger.debug(f"[AsyncOmni] generate() called for request {request_id}")

        input_stream_task: asyncio.Task | None = None
        try:
            # Start final output dispatcher on the first call to generate()
            self._final_output_handler()

            sampling_params_list = self.resolve_sampling_params_list(sampling_params_list)

            # Track per-request metrics
            wall_start_ts = time.time()
            req_start_ts: dict[str, float] = {}

            # Determine the final stage for E2E stats
            final_stage_id_for_e2e = self._compute_final_stage_id(output_modalities)

            metrics = OrchestratorMetrics(
                self.num_stages,
                self.log_stats,
                wall_start_ts,
                final_stage_id_for_e2e,
            )
            req_state = ClientRequestState(request_id)
            req_state.metrics = metrics
            self.request_states[request_id] = req_state

            # Add request(s) to stage 0. For streaming inputs, submit
            # chunks incrementally through streaming_update.
            if isinstance(prompt, AsyncGenerator):
                input_stream_task = await self._add_streaming_input_request(
                    request_id=request_id,
                    input_stream=prompt,
                    sampling_params_list=sampling_params_list,
                    final_stage_id=final_stage_id_for_e2e,
                )
            else:
                await self.engine.add_request_async(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params_list=sampling_params_list,
                    final_stage_id=final_stage_id_for_e2e,
                )
            submit_ts = time.time()
            req_state.metrics.stage_first_ts[0] = submit_ts
            req_start_ts[request_id] = submit_ts

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

            self._log_summary_and_cleanup(request_id)

        except (asyncio.CancelledError, GeneratorExit):
            if input_stream_task is not None and not input_stream_task.done():
                input_stream_task.cancel()
            await self.abort(request_id)
            logger.info(f"[AsyncOmni] Request {request_id} aborted.")
            raise
        except Exception as e:
            await self.abort(request_id)
            logger.info(f"[AsyncOmni] Request {request_id} failed (input error): {e}")
            raise

    async def _add_streaming_input_request(
        self,
        *,
        request_id: str,
        input_stream: AsyncGenerator[StreamingInput, None],
        sampling_params_list: Sequence[OmniSamplingParams],
        final_stage_id: int,
    ) -> asyncio.Task:
        """Submit a streaming input generator as incremental stage-0 updates."""
        if not sampling_params_list:
            raise ValueError("sampling_params_list cannot be empty for streaming input")
        # only check thinker's sampling params now
        stage0_params = sampling_params_list[0]
        self._validate_streaming_input_sampling_params(stage0_params)

        req_state = self.request_states[request_id]

        if not stage0_params.skip_clone:
            stage0_params = stage0_params.clone()
            stage0_params.skip_clone = True
        stage0_params.output_kind = RequestOutputKind.DELTA

        has_submitted_first_chunk = False

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
                        await self.engine.add_request_async(
                            request_id=request_id,
                            prompt=chunk_prompt,
                            prompt_text=prompt_text,
                            sampling_params_list=chunk_sampling_params_list,
                            final_stage_id=final_stage_id,
                            resumable=True,
                        )
                        has_submitted_first_chunk = True
                    else:
                        await self.engine.add_streaming_update_async(
                            request_id=request_id,
                            prompt=chunk_prompt,
                            prompt_text=prompt_text,
                            sampling_params_list=chunk_sampling_params_list,
                            final_stage_id=final_stage_id,
                            resumable=True,
                        )
            except (asyncio.CancelledError, GeneratorExit):
                cancelled = True
            except Exception as error:
                await req_state.queue.put({"request_id": request_id, "error": error})
            finally:
                if not cancelled:
                    # Send empty final request to indicate that inputs have
                    # finished. Don't send if canceled (session was aborted).
                    final_sampling_params_list = list(sampling_params_list)
                    final_sampling_params_list[0] = stage0_params
                    final_prompt = TokensPrompt(prompt_token_ids=[0])

                    if has_submitted_first_chunk:
                        await self.engine.add_streaming_update_async(
                            request_id=request_id,
                            prompt=final_prompt,
                            prompt_text=None,
                            sampling_params_list=final_sampling_params_list,
                            final_stage_id=final_stage_id,
                            resumable=False,
                        )
                    else:
                        await self.engine.add_request_async(
                            request_id=request_id,
                            prompt=final_prompt,
                            prompt_text=None,
                            sampling_params_list=final_sampling_params_list,
                            final_stage_id=final_stage_id,
                            resumable=False,
                        )

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

    # ==================== Processing Methods ====================

    async def _process_orchestrator_results(
        self,
        request_id: str,
        metrics: OrchestratorMetrics,
        final_stage_id_for_e2e: int,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Read results from the Orchestrator (via the request's asyncio.Queue)
        and yield OmniRequestOutput objects.

        The Orchestrator handles all stage-to-stage transfers. This method
        only processes final outputs that arrive on the per-request queue.
        """
        req_state = self.request_states.get(request_id)
        if req_state is None:
            return

        while True:
            result = await req_state.queue.get()

            stage_id = result.get("stage_id", 0)

            # Check for errors
            if "error" in result:
                logger.error(
                    "[AsyncOmni] Orchestrator error for req=%s stage-%s: %s",
                    request_id,
                    stage_id,
                    result["error"],
                )
                raise RuntimeError(result)

            # Process the result (constructs OmniRequestOutput)
            output_to_yield = self._process_single_result(
                result,
                stage_id,
                metrics,
                req_start_ts,
                wall_start_ts,
                final_stage_id_for_e2e,
            )

            if output_to_yield:
                logger.debug(
                    "[AsyncOmni] req=%s stage-%s yielding final_output_type=%s",
                    request_id,
                    stage_id,
                    getattr(output_to_yield, "final_output_type", None),
                )
                yield output_to_yield

            # The Orchestrator sets "finished" when the final stage is done
            if result.get("finished"):
                break

    # ==================== Output Handler ====================

    def _final_output_handler(self) -> None:
        """Start the final output handler if not already running.

        This handler reads messages from the Orchestrator output queue and
        routes them to per-request asyncio.Queues.
        """
        if self.final_output_task is not None:
            return

        engine = self.engine

        async def _final_output_loop():
            """Background coroutine that dispatches final outputs to request queues."""
            try:
                while True:
                    msg = await engine.try_get_output_async()
                    if msg is None:
                        await asyncio.sleep(_FINAL_OUTPUT_IDLE_SLEEP_S)
                        continue

                    should_continue, _, stage_id, req_state = self._handle_output_message(msg)
                    if should_continue:
                        continue

                    req_state.stage_id = stage_id

                    # Route to the per-request queue
                    await req_state.queue.put(msg)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[AsyncOmni] final_output_loop failed.")
                for req_state in list(self.request_states.values()):
                    error_msg = {"request_id": req_state.request_id, "error": str(e)}
                    await req_state.queue.put(error_msg)
                self.final_output_task = None

        self.final_output_task = asyncio.create_task(_final_output_loop())
        logger.debug("[AsyncOmni] Final output handler started")

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
        await self.engine.abort_async(request_ids)
        for req_id in request_ids:
            self.request_states.pop(req_id, None)
        if self.log_stats:
            logger.info("[AsyncOmni] Aborted request(s) %s", ",".join(request_ids))

    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause generation."""
        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        # TODO: Implement request draining if wait_for_inflight_requests

        if clear_cache:
            # Clear caches for all stages.
            await self.reset_prefix_cache(
                reset_running_requests=not wait_for_inflight_requests,
                reset_connector=True,
            )
            await self.reset_mm_cache()
            await self.reset_encoder_cache()

    async def resume_generation(self) -> None:
        """Resume generation."""
        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()

    async def is_paused(self) -> bool:
        """Check if paused."""
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
        """Reset the multi-modal cache for all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmni] reset_mm_cache not yet supported with Orchestrator process")

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

    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        """Sleep all stages.

        Best-effort: unsupported stages will emit a TODO result.
        """
        self._is_sleeping = True
        await self.collective_rpc(method="sleep", args=(level,))

    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up all stages.

        Best-effort: unsupported stages will emit a TODO result.
        """
        self._is_sleeping = False
        await self.collective_rpc(method="wake_up", args=(tags,))

    async def is_sleeping(self) -> bool:
        """Return whether all stages are sleeping.

        TODO(AsyncOmni): query the orchestrator once all stage backends expose
        a real sleeping-state RPC. For now we track the requested state locally.
        """
        return self._is_sleeping

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
            if isinstance(result, list):
                merged.update(result)
        return sorted(merged)

    async def pin_lora(self, adapter_id: int) -> bool:
        """Pin a LoRA adapter across stages."""
        results = await self.collective_rpc(method="pin_lora", args=(adapter_id,))
        concrete_results = [r for r in results if not (isinstance(r, dict) and r.get("todo"))]
        return all(self._coerce_stage_bool(r) for r in concrete_results) if concrete_results else False

    # ==================== Properties ====================

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return self.final_output_task is not None and not self.final_output_task.done()

    @property
    def errored(self) -> bool:
        """Whether orchestrator thread has stopped unexpectedly."""
        return not self.engine.is_alive()

    @property
    def is_stopped(self) -> bool:
        """EngineClient abstract property implementation."""
        return self.errored

    @property
    def dead_error(self) -> BaseException:
        """EngineClient abstract property implementation."""
        return EngineDeadError()

    # ==================== EngineClient Interface ====================

    async def get_input_preprocessor(self) -> InputPreprocessor:
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

    async def do_log_stats(self) -> None:
        """Log statistics.

        TODO: Forward to Orchestrator process via message.
        """
        pass

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Return the task set exposed by the orchestrator-backed engine."""
        return tuple(self.engine.supported_tasks)

    async def check_health(self) -> None:
        """Check engine health by verifying the Orchestrator process is alive."""
        OmniBase.check_health(self)

    # ==================== Shutdown ====================

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown the engine."""
        if self.final_output_task is not None:
            self.final_output_task.cancel()
            self.final_output_task = None
        OmniBase.shutdown(self)
