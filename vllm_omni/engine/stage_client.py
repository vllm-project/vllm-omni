"""Shared stage-client typing for vLLM-Omni runtime surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType, OmniTokensPrompt
    from vllm_omni.outputs import OmniRequestOutput

from vllm_omni.engine.output_modality import FinalOutputModalityType
from vllm_omni.inputs.data import OmniSamplingParams


class StageClient(Protocol):
    """Shared metadata and lifecycle surface for all stage clients.

    This protocol intentionally covers the common attributes consumed by
    orchestration and entrypoint code. Backend-specific request APIs stay
    typed with the concrete client union.
    """

    stage_id: int
    replica_id: int
    stage_type: str
    final_output: bool
    final_output_type: FinalOutputModalityType | None
    default_sampling_params: OmniSamplingParams
    requires_multimodal_data: bool
    custom_process_input_func: Any | None
    engine_input_source: Sequence[int]
    is_comprehension: bool

    def check_health(self) -> None: ...

    def shutdown(self) -> None: ...


class StageClientBase:
    """Runtime nominal base for stage clients.

    Keeping the runtime base separate from the protocol avoids Protocol-related
    MRO interference with concrete client super() calls.
    """

    pass


class StagePoolClient(StageClient, Protocol):
    """Common pool-facing client surface shared by every stage backend."""

    async def abort_requests_async(self, request_ids: list[str]) -> None: ...

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any: ...


class StagePoolLLMClient(StagePoolClient, Protocol):
    """Pool-facing API for LLM-style stages."""

    model_stage: str | None

    async def add_request_async(self, request: EngineCoreRequest) -> None: ...

    async def get_output_async(self) -> EngineCoreOutputs: ...

    def set_engine_outputs(self, engine_outputs: EngineCoreOutput) -> None: ...

    def process_engine_inputs(
        self,
        source_outputs: list[Any],
        prompt: Any = None,
        streaming_context: Any | None = None,
    ) -> list[OmniTokensPrompt]: ...

    def get_kv_sender_info(
        self,
        *,
        base_port: int = ...,
        kv_transfer_port_offset: int = ...,
    ) -> dict[str, Any] | None: ...


class StagePoolDiffusionClient(StagePoolClient, Protocol):
    """Pool-facing API for diffusion stages."""

    async def add_request_async(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None: ...

    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None: ...

    def get_diffusion_output_nowait(self) -> OmniRequestOutput | None: ...
