# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subprocess entry point for lightweight diffusion submodule stages."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.utils.system_utils import get_mp_context

from vllm_omni.diffusion.data import DiffusionOutput, DiffusionRequestAbortedError
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc, complete_diffusion_handshake
from vllm_omni.diffusion.worker.submodule_worker import SubModuleWorker
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class _SubModuleEngine:
    """Lightweight engine for one diffusion submodule worker."""

    def __init__(
        self,
        worker: SubModuleWorker,
        od_config: OmniDiffusionConfig,
        executor: ThreadPoolExecutor,
    ) -> None:
        self.worker = worker
        self.od_config = od_config
        self.executor = executor

    async def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        if len(request.prompts) != 1:
            raise ValueError(f"StageSubModuleProc only supports one prompt per request, got {len(request.prompts)}.")

        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(self.executor, self.worker.execute_submodule, request)
        return [self._to_request_output(request, output)]

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.worker.profile(is_start=is_start, profile_prefix=profile_prefix)

    def collective_rpc(
        self,
        method: str,
        _timeout: float | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *_: Any,
    ) -> Any:
        return getattr(self.worker, method)(*args, **(kwargs or {}))

    def abort(self, _request_id: str) -> None:
        return None

    def close(self) -> None:
        self.worker.shutdown()

    def _to_request_output(
        self,
        request: OmniDiffusionRequest,
        output: DiffusionOutput,
    ) -> OmniRequestOutput:
        if output.aborted:
            raise DiffusionRequestAbortedError(output.abort_message or "Diffusion submodule request aborted.")
        if output.error:
            raise RuntimeError(output.error)

        request_id = request.request_id
        prompt = request.prompts[0] if request.prompts else None
        output_data = output.output
        images: list[Image.Image] = []
        latents: torch.Tensor | None = None
        final_output_type = f"stage_{self.od_config.model_stage}"

        if isinstance(output_data, Image.Image):
            images = [output_data]
            final_output_type = "image"
        elif isinstance(output_data, list):
            images = output_data
            final_output_type = "image"
        elif isinstance(output_data, torch.Tensor):
            latents = output_data
            final_output_type = "latents"

        return OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=images,
            prompt=prompt,
            metrics={},
            latents=latents,
            multimodal_output=output.multimodal_output,
            custom_output=output.custom_output,
            final_output_type=final_output_type,
            stage_durations=output.stage_durations,
            peak_memory_mb=output.peak_memory_mb,
        )


class StageSubModuleProc(StageDiffusionProc):
    """StageDiffusionProc variant backed by a lightweight submodule worker."""

    def initialize(self) -> None:
        self._enrich_config()
        worker = SubModuleWorker(
            local_rank=0,
            rank=0,
            od_config=self._od_config,
        )
        worker.load_model(load_format=self._od_config.diffusion_load_format)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._engine = _SubModuleEngine(worker, self._od_config, self._executor)
        logger.info(
            "StageSubModuleProc initialized with model=%s stage=%s",
            self._od_config.model,
            self._od_config.model_stage,
        )


def spawn_submodule_proc(
    model: str,
    od_config: OmniDiffusionConfig,
) -> tuple[BaseProcess, str, str, str]:
    handshake_address = get_open_zmq_ipc_path()
    request_address = get_open_zmq_ipc_path()
    response_address = get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageSubModuleProc.run_diffusion_proc,
        name="StageSubModuleProc",
        kwargs={
            "model": model,
            "od_config": od_config,
            "handshake_address": handshake_address,
            "request_address": request_address,
            "response_address": response_address,
        },
    )
    proc.start()
    deadline = time.monotonic() + 10
    while not proc.is_alive():
        if proc.exitcode is not None:
            raise RuntimeError(f"StageSubModuleProc failed to start (exit code {proc.exitcode})")
        if time.monotonic() > deadline:
            raise TimeoutError("StageSubModuleProc did not become alive within 10s")
        time.sleep(0.01)
    return proc, handshake_address, request_address, response_address


def complete_submodule_handshake(
    proc: BaseProcess,
    handshake_address: str,
    handshake_timeout: int,
) -> None:
    complete_diffusion_handshake(proc, handshake_address, handshake_timeout)
