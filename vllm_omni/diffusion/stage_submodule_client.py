# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage client for lightweight submodule stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.diffusion.stage_submodule_proc import (
    complete_submodule_handshake,
    spawn_submodule_proc,
)
from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


class StageSubModuleClient(StageDiffusionClient):
    """StageDiffusionClient variant backed by StageSubModuleProc."""

    def __init__(
        self,
        model: str,
        od_config: OmniDiffusionConfig,
        metadata: StageMetadata,
        stage_init_timeout: int,
        batch_size: int = 1,
    ) -> None:
        proc, handshake_address, request_address, response_address = spawn_submodule_proc(model, od_config)
        complete_submodule_handshake(proc, handshake_address, stage_init_timeout)
        self._initialize_client(metadata, request_address, response_address, proc=proc, batch_size=batch_size)

    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        if len(prompts) == 1:
            await self.add_request_async(request_id, prompts[0], sampling_params, kv_sender_info=kv_sender_info)
            return

        self.check_health()
        self._output_queue.put_nowait(
            OmniRequestOutput.from_error(
                request_id,
                f"StageSubModuleClient only supports single-prompt batches, got {len(prompts)} prompts.",
            )
        )
