"""Inline Stage Diffusion Client for vLLM-Omni multi-stage runtime.

Runs DiffusionEngine in a ThreadPoolExecutor inside the Orchestrator process
instead of spawning a separate StageDiffusionProc subprocess, eliminating ZMQ
IPC overhead. Used when there is only a single diffusion stage.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import DiffusionRequestAbortedError
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.engine.stage.stage_core_client import StageCoreClientBase
from vllm_omni.engine.stage.stage_core_types import (
    StageDiffusionCoreOutput,
    StageDiffusionCoreOutputs,
)
from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.errors import client_error_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniInteractionPrompt
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.engine.stage.stage_core_types import StageDiffusionCoreRequest
    from vllm_omni.inputs.data import OmniPromptType

logger = init_logger(__name__)


class InlineStageDiffusionClient(StageCoreClientBase):
    """Runs DiffusionEngine in a thread executor inside the Orchestrator.

    Conforms to the :class:`StageCoreClientBase` contract so the pool drives it
    identically to the out-of-process ``StageDiffusionCoreClient``: requests
    arrive as a typed ``StageDiffusionCoreRequest`` and outputs are drained as a
    ``StageDiffusionCoreOutputs`` batch. The only difference is that execution
    runs in-process on a thread executor rather than a subprocess over ZMQ.
    """

    stage_type: str = "diffusion"
    replica_id: int = 0
    is_comprehension: bool = False

    def __init__(
        self,
        model: str,
        od_config: OmniDiffusionConfig,
        metadata: StageMetadata,
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.od_config = od_config
        self.stage_id = metadata.stage_id
        self.replica_id = metadata.replica_id
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.model_stage = getattr(metadata, "model_stage", None)
        self.default_sampling_params = metadata.default_sampling_params
        self.requires_multimodal_data = metadata.requires_multimodal_data
        self.custom_process_input_func = metadata.custom_process_input_func
        self.engine_input_source = metadata.engine_input_source
        self.batch_size = batch_size

        self._enrich_config()
        self._engine = DiffusionEngine.make_engine(self.od_config)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inline-diffusion")

        self._output_queue: asyncio.Queue[StageDiffusionCoreOutput] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task] = {}
        self._engine_dead = False
        self._shutting_down = False
        self._shutdown_complete = False
        self._shutdown_lock = threading.Lock()

        self._engine.executor.register_failure_callback(self._mark_engine_dead)

        logger.info(
            "[InlineStageDiffusionClient] stage-%s [rep-%s] initialized inline (batch_size=%d)",
            self.stage_id,
            self.replica_id,
            self.batch_size,
        )

    def _enrich_config(self) -> None:
        """Load model metadata from HuggingFace and populate od_config fields."""
        self.od_config.enrich_config()

    def _mark_engine_dead(self) -> None:
        if self._engine_dead:
            return
        self._engine_dead = True
        logger.error(
            "[InlineStageDiffusionClient] stage-%s [rep-%s] diffusion executor died unexpectedly.",
            self.stage_id,
            self.replica_id,
        )

    # ------------------------------------------------------------------
    # Request processing
    # ------------------------------------------------------------------

    async def add_request_async(self, request: StageDiffusionCoreRequest) -> None:
        # The pool hands the inline client the original
        # ``OmniDiffusionSamplingParams`` unmodified (no process boundary). Each
        # request mutates its sampling state while it is normalized and
        # executed, and callers commonly reuse one params object for concurrent
        # requests, so take a copy synchronously before either task starts.
        # ``generator`` and ``modules`` are kept by reference: advancing
        # per-output generator state and passing live component modules through
        # unchanged is the reason the inline path bypasses the lossy wire form.
        # For robustness we still accept the plain-dict wire form (used by the
        # out-of-process client) and reconstruct the dataclass in-process; in
        # that case a stripped ``generator`` is recreated from ``seed`` by the
        # engine, matching the out-of-process path.
        sp = request.sampling_params
        if isinstance(sp, OmniDiffusionSamplingParams):
            memo: dict[int, Any] = {id(sp.modules): sp.modules}
            generators = sp.generator if isinstance(sp.generator, list) else [sp.generator]
            for gen in generators:
                if gen is not None:
                    memo[id(gen)] = gen
            sampling_params = copy.deepcopy(sp, memo)
        else:
            sampling_params = OmniDiffusionSamplingParams(**sp)
        logger.debug(
            "[InlineStageDiffusionClient] stage-%s [rep-%s] add request: %s",
            self.stage_id,
            self.replica_id,
            request.request_id,
        )
        task = asyncio.create_task(
            self._dispatch_request(
                request.request_id,
                request.prompt,
                sampling_params,
                request.kv_sender_info,
            )
        )
        self._tasks[request.request_id] = task

    async def _dispatch_request(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> None:
        try:
            request = OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                kv_sender_info=kv_sender_info,
            )

            if self.od_config.streaming_output:
                async for results in self._engine.step_streaming(request):
                    result = results[0]
                    if not result.request_id:
                        result.request_id = request_id
                    self._enqueue_result(request_id, result)
            else:
                # Non-streaming callers share the streaming engine path but
                # only publish the final output.
                result = None
                async for results in self._engine.step_streaming(request):
                    result = results[0]
                if result is None:
                    raise RuntimeError("Diffusion execution finished without output.")
                if not result.request_id:
                    result.request_id = request_id
                self._enqueue_result(request_id, result)
        except DiffusionRequestAbortedError as e:
            logger.info("request_id: %s aborted: %s", request_id, str(e))
        except Exception as e:
            logger.exception("Diffusion request %s failed: %s", request_id, e)
            status_code, error_type = client_error_metadata(e)
            # Mirror the out-of-process client: the error rides the wire struct
            # and the pool materializes it into an error ``OmniRequestOutput``.
            self._output_queue.put_nowait(
                StageDiffusionCoreOutput(
                    request_id=request_id,
                    finished=True,
                    output=None,
                    error=str(e),
                    status_code=status_code,
                    error_type=error_type,
                )
            )
        finally:
            self._tasks.pop(request_id, None)

    def _enqueue_result(self, request_id: str, result: OmniRequestOutput) -> None:
        """Wrap a successful engine output in the typed wire struct and queue it."""
        self._output_queue.put_nowait(
            StageDiffusionCoreOutput(
                request_id=request_id,
                finished=bool(getattr(result, "finished", True)),
                output=result,
            )
        )

    def get_outputs_nowait(self) -> StageDiffusionCoreOutputs | None:
        """Drain all ready diffusion outputs into one batch, or ``None`` if idle."""
        collected: list[StageDiffusionCoreOutput] = []
        while True:
            try:
                collected.append(self._output_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if collected:
            return StageDiffusionCoreOutputs(outputs=collected)
        if self._engine_dead:
            raise EngineDeadError(f"Stage-{self.stage_id} inline diffusion engine is dead")
        return None

    async def get_outputs_async(self) -> StageDiffusionCoreOutputs:
        """Await the next batch of diffusion outputs.

        Poll-based to mirror the out-of-process client: return as soon as a batch
        is ready, an empty batch once shutting down, and propagate
        ``EngineDeadError`` (raised by ``get_outputs_nowait``) if the engine dies.
        """
        while True:
            if self._shutting_down:
                return StageDiffusionCoreOutputs(outputs=[])
            batch = self.get_outputs_nowait()
            if batch is not None:
                return batch
            await asyncio.sleep(0.05)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            task = self._tasks.pop(rid, None)
            if task:
                task.cancel()
            self._engine.abort(rid)

    async def submit_interaction_async(
        self,
        request_id: str,
        interaction: OmniInteractionPrompt,
        timeout: float | None = None,
    ) -> Any:
        """Apply a midway interaction to an active streaming request."""
        logger.debug(
            "[InlineStageDiffusionClient] stage-%s [rep-%s] interaction: %s",
            self.stage_id,
            self.replica_id,
            request_id,
        )
        return await self.collective_rpc_async(
            "submit_interaction",
            timeout=timeout,
            args=(request_id, interaction),
        )

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        loop = asyncio.get_running_loop()

        if method == "profile":
            is_start = args[0] if args else True
            profile_prefix = args[1] if len(args) > 1 else None
            if is_start and profile_prefix is None:
                profile_prefix = f"stage_{self.stage_id}_rep_{self.replica_id}_diffusion_{int(time.time())}"
            return await loop.run_in_executor(
                self._executor,
                self._engine.profile,
                is_start,
                profile_prefix,
            )

        kwargs = kwargs or {}

        # LoRA methods
        if method == "add_lora":
            lora_request = args[0] if args else kwargs.get("lora_request")
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "add_lora",
                timeout,
                (),
                {"lora_request": lora_request},
                None,
            )
            return all(results) if isinstance(results, list) else results

        if method == "remove_lora":
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "remove_lora",
                timeout,
                args,
                kwargs,
                None,
            )
            return all(results) if isinstance(results, list) else results

        if method == "list_loras":
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "list_loras",
                timeout,
                (),
                {},
                None,
            )
            if not isinstance(results, list):
                return results or []
            merged: set[int] = set()
            for part in results:
                merged.update(part or [])
            return sorted(merged)

        if method == "pin_lora":
            lora_id = args[0] if args else kwargs.get("adapter_id")
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "pin_lora",
                timeout,
                (),
                {"adapter_id": lora_id},
                None,
            )
            return all(results) if isinstance(results, list) else results

        return await loop.run_in_executor(
            self._executor,
            self._engine.collective_rpc,
            method,
            timeout,
            args,
            kwargs,
            None,
        )

    def check_health(self) -> None:
        """Check if the inline diffusion engine and its workers are healthy.

        Overrides the base template to actively probe the executor (rather than
        only reporting an already-detected death via ``_engine_dead_reason``).
        """
        if self._shutting_down:
            raise EngineDeadError("InlineStageDiffusionClient is shutting down")
        try:
            self._engine.executor.check_health()
        except EngineDeadError:
            self._mark_engine_dead()
            raise

    def _engine_dead_reason(self) -> str | None:
        if self._engine_dead:
            return f"Stage-{self.stage_id} inline diffusion engine is dead"
        return None

    def shutdown(self, timeout: float | None = None) -> None:
        # ``timeout`` is part of the StageCoreClientBase contract; inline shutdown
        # is synchronous and deterministic, so it is accepted but unused.
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._shutting_down = True

            # Cancel all pending tasks
            for task in self._tasks.values():
                task.cancel()

            try:
                # Stop the engine first so any control RPC running in the thread
                # pool can observe shutdown instead of keeping stage teardown
                # blocked while the executor waits for that RPC.
                self._engine.close()
            except Exception:
                pass

            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._shutdown_complete = True
