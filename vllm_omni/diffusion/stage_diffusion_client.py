"""Stage Diffusion Client for vLLM-Omni multi-stage runtime.

Wraps AsyncOmniDiffusion to expose the same interface the Orchestrator
expects from any stage client.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

logger = init_logger(__name__)


class StageDiffusionClient:
    """Wraps AsyncOmniDiffusion for use inside the Orchestrator.

    Exposes the same attributes and async methods the Orchestrator
    uses on StageEngineCoreClient, but routes execution through
    DiffusionEngine instead of vLLM EngineCore.
    """

    stage_type: str = "diffusion"

    def __init__(
        self,
        model: str,
        od_config: OmniDiffusionConfig,
        metadata: StageMetadata,
        batch_size: int = 1,
    ) -> None:
        self.stage_id = metadata.stage_id
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.custom_process_input_func = metadata.custom_process_input_func
        self.engine_input_source = metadata.engine_input_source

        self._engine = AsyncOmniDiffusion(model=model, od_config=od_config, batch_size=batch_size)
        self._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task] = {}

        logger.info("[StageDiffusionClient] Stage-%s initialized (batch_size=%d)", self.stage_id, batch_size)

    async def add_request_async(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
    ) -> None:
        task = asyncio.create_task(
            self._run(request_id, prompt, sampling_params),
            name=f"diffusion-{request_id}",
        )
        self._tasks[request_id] = task

    async def _run(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
    ) -> None:
        try:
            result = await self._engine.generate(prompt, sampling_params, request_id)
            await self._output_queue.put(result)
        except Exception as e:
            logger.exception(
                "[StageDiffusionClient] Stage-%s req=%s failed: %s",
                self.stage_id,
                request_id,
                e,
            )
        finally:
            self._tasks.pop(request_id, None)

    # TODO(Long): Temporary solution to boost performance of diffusion stages.
    # Remove this after scheduling algorithm is implemented
    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
    ) -> None:
        """Submit a list of prompts as a single batched engine call.

        All prompts are processed in one ``DiffusionEngine.step()`` call
        and the combined result is placed on the output queue with a single
        *request_id*.
        """
        task = asyncio.create_task(
            self._run_batch(request_id, prompts, sampling_params),
            name=f"diffusion-batch-{request_id}",
        )
        self._tasks[request_id] = task

    async def _run_batch(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
    ) -> None:
        try:
            result = await self._engine.generate_batch(
                prompts,
                sampling_params,
                request_id,
            )
            await self._output_queue.put(result)
        except Exception as e:
            logger.exception(
                "[StageDiffusionClient] Stage-%s batch req=%s failed: %s",
                self.stage_id,
                request_id,
                e,
            )
        finally:
            self._tasks.pop(request_id, None)

    def get_diffusion_output_async(self) -> OmniRequestOutput | None:
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            task = self._tasks.pop(rid, None)
            if task:
                task.cancel()

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Best-effort control RPC shim for diffusion stages.

        TODO(AsyncOmni): add dedicated wrappers on AsyncOmniDiffusion for the
        remaining control APIs instead of reaching into its underlying engine.
        """
        kwargs = kwargs or {}

        # Handle profile method: inject stage_id into profile_prefix for diffusion stages
        if method == "profile":
            target = getattr(self._engine, method, None)
            if target is None:
                return {
                    "supported": False,
                    "todo": True,
                    "reason": f"AsyncOmniDiffusion.{method} is not implemented",
                }
            # Extract is_start and profile_prefix from args
            is_start = args[0] if args else True
            profile_prefix = args[1] if len(args) > 1 else None
            # Generate profile_prefix with stage_id if starting and no prefix provided
            if is_start and profile_prefix is None:
                profile_prefix = f"stage_{self.stage_id}_diffusion_{int(time.time())}"
            result = target(is_start, profile_prefix)
            if timeout is not None:
                return await asyncio.wait_for(result, timeout=timeout)
            return await result

        if method in {"add_lora", "remove_lora", "list_loras", "pin_lora"}:
            target = getattr(self._engine, method, None)
            if target is None:
                return {
                    "supported": False,
                    "todo": True,
                    "reason": f"AsyncOmniDiffusion.{method} is not implemented",
                }
            result = target(*args, **kwargs)
            if timeout is not None:
                return await asyncio.wait_for(result, timeout=timeout)
            return await result

        if method in {"sleep", "wake_up"}:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._engine._executor,
                self._engine.engine.collective_rpc,
                method,
                timeout,
                args,
                kwargs,
                None,
            )

        return {
            "supported": False,
            "todo": True,
            "reason": f"Diffusion stage collective_rpc method {method} is not implemented yet",
        }

    def shutdown(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._engine.close()
