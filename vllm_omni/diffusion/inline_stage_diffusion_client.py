"""Inline Stage Diffusion Client for vLLM-Omni multi-stage runtime.

Runs DiffusionEngine in a ThreadPoolExecutor inside the Orchestrator process
instead of spawning a separate StageDiffusionProc subprocess, eliminating ZMQ
IPC overhead. Used when there is only a single diffusion stage.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionRequestAbortedError
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniPromptType

logger = init_logger(__name__)


class InlineStageDiffusionClient:
    """Runs DiffusionEngine in a thread executor inside the Orchestrator."""

    stage_type: str = "diffusion"

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
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.custom_process_input_func = metadata.custom_process_input_func
        self.engine_input_source = metadata.engine_input_source
        self.batch_size = batch_size

        self._enrich_config()
        self._engine = DiffusionEngine.make_engine(self.od_config)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inline-diffusion")

        self._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task] = {}
        self._shutting_down = False

        logger.info(
            "[InlineStageDiffusionClient] Stage-%s initialized inline (batch_size=%d)",
            self.stage_id,
            self.batch_size,
        )

    def _enrich_config(self) -> None:
        """Load model metadata from HuggingFace and populate od_config fields."""
        self.od_config.enrich_config()

    # ------------------------------------------------------------------
    # Request processing
    # ------------------------------------------------------------------

    async def add_request_async(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        task = asyncio.create_task(
            self._dispatch_request(
                request_id,
                prompt,
                sampling_params,
                kv_sender_info,
            )
        )
        self._tasks[request_id] = task

    async def _dispatch_request(
        self,
        request_id: str,
        prompt: Any,
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> None:
        try:
            request = OmniDiffusionRequest(
                prompts=[prompt],
                sampling_params=sampling_params,
                request_ids=[request_id],
                request_id=request_id,
                kv_sender_info=kv_sender_info,
            )

            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(self._executor, self._engine.step, request)
            result = results[0]
            if not result.request_id:
                result.request_id = request_id

            self._output_queue.put_nowait(result)
        except DiffusionRequestAbortedError as e:
            logger.info("request_id: %s aborted: %s", request_id, str(e))
        except Exception as e:
            logger.exception("Diffusion request %s failed: %s", request_id, e)
            error_output = OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=[],
            )
            error_output.error = str(e)
            self._output_queue.put_nowait(error_output)
        finally:
            self._tasks.pop(request_id, None)

    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        task = asyncio.create_task(
            self._dispatch_batch(
                request_id,
                prompts,
                sampling_params,
                kv_sender_info,
            )
        )
        self._tasks[request_id] = task

    async def _dispatch_batch(
        self,
        request_id: str,
        prompts: list[Any],
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> None:
        try:
            request = OmniDiffusionRequest(
                prompts=prompts,
                sampling_params=sampling_params,
                request_ids=[f"{request_id}-{i}" for i in range(len(prompts))],
                request_id=request_id,
                kv_sender_info=kv_sender_info,
            )

            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(self._executor, self._engine.step, request)

            all_images: list = []
            merged_mm: dict[str, Any] = {}
            merged_metrics: dict[str, Any] = {}
            merged_durations: dict[str, float] = {}
            merged_custom: dict[str, Any] = {}
            peak_mem = 0.0
            latents = None
            trajectory_latents: list[torch.Tensor] | None = None
            trajectory_timesteps: list[torch.Tensor] | None = None
            trajectory_log_probs: torch.Tensor | None = None
            trajectory_decoded: list[Image.Image] | None = None
            final_output_type = "image"

            for r in results:
                all_images.extend(r.images)
                merged_mm.update(r._multimodal_output)
                merged_metrics.update(r.metrics)
                merged_durations.update(r.stage_durations)
                merged_custom.update(r._custom_output)
                peak_mem = max(peak_mem, r.peak_memory_mb)
                if latents is None and r.latents is not None:
                    latents = r.latents
                if trajectory_latents is None:
                    trajectory_latents = r.trajectory_latents
                if trajectory_timesteps is None:
                    trajectory_timesteps = r.trajectory_timesteps
                if trajectory_log_probs is None:
                    trajectory_log_probs = r.trajectory_log_probs
                if trajectory_decoded is None:
                    trajectory_decoded = r.trajectory_decoded
                if r.final_output_type != "image":
                    final_output_type = r.final_output_type

            result = OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=all_images,
                prompt=prompts[0] if len(prompts) == 1 else None,
                metrics=merged_metrics,
                latents=latents,
                trajectory_latents=trajectory_latents,
                trajectory_timesteps=trajectory_timesteps,
                trajectory_log_probs=trajectory_log_probs,
                trajectory_decoded=trajectory_decoded,
                custom_output=merged_custom or None,
                multimodal_output=merged_mm or None,
                final_output_type=final_output_type,
                stage_durations=merged_durations,
                peak_memory_mb=peak_mem,
            )

            self._output_queue.put_nowait(result)
        except DiffusionRequestAbortedError as e:
            logger.info("request_id: %s aborted: %s", request_id, str(e))
        except Exception as e:
            logger.exception("Batch diffusion request %s failed: %s", request_id, e)
            error_output = OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=[],
            )
            error_output.error = str(e)
            self._output_queue.put_nowait(error_output)
        finally:
            self._tasks.pop(request_id, None)

    def get_diffusion_output_nowait(self) -> OmniRequestOutput | None:
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            task = self._tasks.pop(rid, None)
            if task:
                task.cancel()
            self._engine.abort(rid)

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
                profile_prefix = f"stage_{self.stage_id}_diffusion_{int(time.time())}"
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

    def shutdown(self) -> None:
        self._shutting_down = True

        # Cancel all pending tasks
        for task in self._tasks.values():
            task.cancel()

        try:
            # Cancel queued futures and wait for the running one to complete deterministically
            self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass

        try:
            self._engine.close()
        except Exception:
            pass
