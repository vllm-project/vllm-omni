"""Contract layer for the vLLM-Omni stage process/client model.

``StageCoreClientBase`` (``abc.ABC``) is the runtime base shared by every backend
(LLM, diffusion). It declares the surface common to both backends — including the
request/output exchange (``add_request_async`` / ``get_outputs_async`` /
``get_outputs_nowait``), each typed to the field-free marker types so a concrete
backend narrows them to its own wire types. Backend-only methods
(``process_core_inputs`` / ``get_kv_sender_info`` for LLM) live on the respective
subclasses, so a backend only implements what it actually supports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.engine.stage.stage_core_types import StageCoreOutputs, StageCoreRequest
    from vllm_omni.engine.stage_init_utils import StageMetadata

from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.inputs.data import OmniSamplingParams
from vllm_omni.outputs.output_metadata import FinalOutputModalityType


class StageCoreClientBase(ABC):
    """Runtime base shared by every stage client (LLM and diffusion).

    Declares only the backend-common contract. It intentionally does not declare
    any LLM-only method, nor anything whose name would shadow an
    ``AsyncMPClient`` / ``DPLBAsyncMPClient`` implementation in a concrete LLM
    client's MRO (see the OOP counter-design for the shadowing rule).
    """

    # ---- shared metadata (populated by __init__ from ``metadata``) ----
    stage_id: int
    # replica_id is assigned in __init__ from ``metadata.replica_id`` and is
    # required to be present, integral, and non-negative when ``metadata`` is
    # given (validated by ``_validate_replica_id``) — the master allocates it
    # during the proc's registration handshake, which always precedes client
    # construction, so there is no late-binding seam. The ``None`` default only
    # covers the metadata-less construction path.
    replica_id: int | None = None
    stage_type: str
    model_stage: str | None
    final_output: bool
    final_output_type: FinalOutputModalityType | None
    default_sampling_params: OmniSamplingParams
    prompt_expand_func: Callable | None
    requires_multimodal_data: bool
    custom_process_input_func: Callable | None
    engine_input_source: Sequence[int]
    is_comprehension: bool

    def __init__(self, *args: Any, metadata: StageMetadata | None = None, **kwargs: Any) -> None:
        """Populate the shared stage metadata, then continue cooperative init.

        ``metadata`` is consumed here to set the backend-common metadata fields
        (including ``replica_id``, which is always concrete by construction
        time); any remaining positional/keyword arguments are forwarded via
        ``super().__init__`` so a concrete LLM client can initialize its mixed-in
        transport base (``AsyncMPClient``) through the MRO.
        """
        if metadata is not None:
            self.stage_id = metadata.stage_id
            self.replica_id = self._validate_replica_id(metadata)
            self.stage_type = metadata.stage_type
            self.model_stage = metadata.model_stage
            self.final_output = metadata.final_output
            self.final_output_type = metadata.final_output_type
            self.default_sampling_params = metadata.default_sampling_params
            self.prompt_expand_func = metadata.prompt_expand_func
            self.requires_multimodal_data = getattr(metadata, "requires_multimodal_data", False)
            self.custom_process_input_func = getattr(metadata, "custom_process_input_func", None)
            self.engine_input_source = getattr(metadata, "engine_input_source", [])
            self.is_comprehension = getattr(metadata, "is_comprehension", False)
        super().__init__(*args, **kwargs)

    def check_health(self) -> None:
        """Raise ``EngineDeadError`` if the backing stage process is dead.

        Template method: each backend reports liveness (and may cache a detected
        death as a side effect) via :meth:`_engine_dead_reason`; the raise site is
        shared here.
        """
        reason = self._engine_dead_reason()
        if reason is not None:
            raise EngineDeadError(reason)

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown."""

    @abstractmethod
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        "Abort requests."

    @abstractmethod
    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """RPC"""

    @abstractmethod
    async def add_request_async(self, request: StageCoreRequest) -> None:
        """Add request. Backends accept their own concrete request type."""

    @abstractmethod
    async def get_outputs_async(self) -> StageCoreOutputs:
        """Await and return the next batch of stage outputs.

        Backends narrow the return to their own concrete outputs type.
        """

    @abstractmethod
    def get_outputs_nowait(self) -> StageCoreOutputs | None:
        """Return the next batch of stage outputs, or ``None`` if none are ready.

        Backends narrow the return to their own concrete outputs type.
        """

    @staticmethod
    def _validate_replica_id(metadata: StageMetadata) -> int:
        """Return the concrete ``replica_id`` from ``metadata`` or raise.

        The master assigns ``replica_id`` during the proc's registration
        handshake, which always precedes client construction, so a valid
        (present, integral, non-negative) id must be available here. A missing
        or negative value (e.g. the auto-assign sentinel leaking through) is a
        wiring bug, not a state to tolerate.
        """
        replica_id = metadata.replica_id
        if replica_id is None:
            raise ValueError(
                f"metadata.replica_id must be provided for stage {metadata.stage_id}; "
                "it is assigned by the master before the stage client is constructed."
            )
        try:
            replica_id = int(replica_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"metadata.replica_id must be an integer for stage {metadata.stage_id}, got {metadata.replica_id!r}."
            ) from exc
        if replica_id < 0:
            raise ValueError(f"metadata.replica_id must be >= 0 for stage {metadata.stage_id}, got {replica_id}.")
        return replica_id

    @abstractmethod
    def _engine_dead_reason(self) -> str | None:
        """Return an error message if the backing engine/subprocess is dead.

        Returns ``None`` when healthy. Implementations may update internal
        dead-state (e.g. cache a detected death) as a side effect.
        """
