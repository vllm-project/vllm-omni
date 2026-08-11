"""
Stage LLM Core Client for vLLM-Omni multi-stage runtime.

``StageLLMCoreClientBase`` is an **adapter**: it implements the stage-domain
contract (``StageCoreClientBase``) on top of vLLM's ``AsyncMPClient``. The
vLLM-named methods (``add_request_async``, ``abort_requests_async``,
``collective_rpc_async``, ``shutdown``) are reused via cooperative ``super()``
dispatch through the concrete client's MRO — never redeclared on a superclass of
``AsyncMPClient`` — so ``DPLBAsyncMPClient``'s DP-aware overrides stay in effect.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import vllm.v1.engine.core_client as _vllm_core_client_module
from vllm.logger import init_logger
from vllm.v1.engine.core_client import AsyncMPClient, DPLBAsyncMPClient

from vllm_omni.distributed.omni_connectors.utils.config import (
    TRANSFER_ENGINE_CONNECTOR_NAMES,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    KV_TRANSFER_PORT_OFFSET,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import kv_zmq_port
from vllm_omni.engine.stage.stage_core_client import StageCoreClientBase
from vllm_omni.engine.stage.stage_core_types import (
    StageLLMCoreOutputs,
    StageLLMCoreRequest,
)
from vllm_omni.engine.stage_init_utils import StageMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.outputs import RequestOutput
    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.executor import Executor

    from vllm_omni.engine.orchestrator import StreamingInputState
    from vllm_omni.engine.stage.stage_llm_core_proc_manager import StageLLMCoreProcManager
    from vllm_omni.inputs.data import OmniPromptType, OmniTokensPrompt

logger = init_logger(__name__)


class StageLLMCoreClientBase(StageCoreClientBase):
    """Adapter mixin: implements ``StageCoreClientBase`` over ``AsyncMPClient``.

    Not instantiated directly — the concrete subclasses mix in the vLLM async MP
    client (plain or DP load-balancing) that supplies ZMQ transport, the output
    queue, and the vLLM-named methods.

    ``replica_id`` is set from ``metadata`` by ``StageCoreClientBase.__init__``:
    the master assigns it during the proc's registration handshake, which always
    precedes client construction, so it is already concrete here. The KV-transfer
    sender endpoint (keyed on ``replica_id`` + ``stage_id``) is built in
    ``__init__`` right after ``super().__init__()`` populates both.
    """

    @staticmethod
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool = False,
        metadata: StageMetadata | None = None,
        client_addresses: dict[str, str] | None = None,
        proc_manager: StageLLMCoreProcManager | None = None,
        coordinator: DPCoordinator | None = None,
    ) -> StageLLMCoreClient | DPLBStageLLMCoreClient:
        """Create the appropriate stage async client for the DP mode."""
        parallel_config = vllm_config.parallel_config
        client_args = dict(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            metadata=metadata,
            client_addresses=client_addresses,
            proc_manager=proc_manager,
            coordinator=coordinator,
        )

        if parallel_config.data_parallel_size > 1 and not parallel_config.data_parallel_external_lb:
            return DPLBStageLLMCoreClient(**client_args)

        return StageLLMCoreClient(**client_args)

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        *,
        metadata: StageMetadata | None = None,
        proc_manager: StageLLMCoreProcManager | None = None,
        coordinator: DPCoordinator | None = None,
    ):
        """Create an async EngineCore client for a single stage.

        Heavy init (config extraction, plugin loading, device setup) is done by
        the orchestrator via ``stage_init_utils``. This constructor stores
        metadata and delegates transport setup to the vLLM client.

        The output decoder built inside ``super().__init__()`` reads
        ``EngineCoreOutputs`` from ``vllm.v1.engine.core_client`` module scope;
        ``vllm_omni.patch`` rebinds that name to ``StageLLMCoreOutputs`` at
        import time. This constructor no longer mutates global module state — it
        only asserts the precondition (read-only) before the decoder is created.
        """
        # Stage metadata is populated by StageCoreClientBase.__init__ (below).
        self.client_addresses = dict(client_addresses or {})
        self._omni_kv_config = getattr(getattr(vllm_config, "model_config", None), "omni_kv_config", None)
        self._core_host = self._resolve_core_host()
        self._kv_sender_info: dict[str, Any] | None = None
        self._kv_sender_initialized = False

        client_name = self.__class__.__name__

        # Precondition: patch.py must have rebound the output-decoder type before
        # any stage client is constructed (it runs at ``import vllm_omni``, long
        # before serving). Assert read-only rather than rebinding here (SRP).
        assert _vllm_core_client_module.EngineCoreOutputs is StageLLMCoreOutputs, (
            "vllm.v1.engine.core_client.EngineCoreOutputs is not StageLLMCoreOutputs; "
            "ensure vllm_omni.patch is imported before constructing the stage client."
        )

        try:
            # StageCoreClientBase.__init__ sets the shared stage metadata from
            # ``metadata``, then forwards the transport args to AsyncMPClient via
            # cooperative super().__init__ (MRO: StageLLMCoreClientBase ->
            # StageCoreClientBase -> AsyncMPClient).
            super().__init__(
                vllm_config,
                executor_class,
                log_stats=log_stats,
                client_addresses=client_addresses,
                metadata=metadata,
            )
            if proc_manager is not None:
                # vLLM's background resources field is still named ``engine_manager``.
                self.resources.engine_manager = proc_manager
                self.start_engine_core_monitor()
            if coordinator is not None:
                self.resources.coordinator = coordinator
        except Exception:
            logger.exception(
                "[%s] stage-%s [rep-%s] EngineCore init failed",
                client_name,
                self.stage_id,
                self.replica_id,
            )
            try:
                self.shutdown()
            except Exception as shutdown_error:
                logger.warning(
                    "[%s] stage-%s [rep-%s] cleanup after init failure failed: %s",
                    client_name,
                    self.stage_id,
                    self.replica_id,
                    shutdown_error,
                )
            raise

        # stage_id and replica_id are now set by StageCoreClientBase.__init__
        # (via super() above), so the KV-transfer sender endpoint — keyed on
        # both — can be built here.
        self._initialize_kv_sender_endpoint()

        logger.info(
            "[%s] stage-%s [rep-%s] EngineCore running",
            client_name,
            self.stage_id,
            self.replica_id,
        )

    async def get_outputs_async(self) -> StageLLMCoreOutputs:
        """Await the next batch of stage outputs.

        Adapts vLLM's ``AsyncMPClient.get_output_async`` — reached via ``super()``
        because the name differs and is not shadowed by ``StageCoreClientBase``.
        """
        return await super().get_output_async()

    def get_outputs_nowait(self) -> StageLLMCoreOutputs | None:
        """Return the next batch of stage outputs, or ``None`` if none ready.

        Non-blocking poll of the async output queue. The background output-queue
        task is started lazily by ``add_request_async`` / ``get_outputs_async``;
        this method does not start it (that would call ``asyncio.create_task`` and
        require a running loop), so before any request is added it simply reports
        ``None``.
        """
        assert self.outputs_queue is not None
        try:
            outputs = self.outputs_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    # ---- vLLM-named methods: delegate PAST the StageCoreClientBase abstract ----
    #
    # These four are declared ``@abstractmethod`` on ``StageCoreClientBase``,
    # which sits *above* the vLLM client in the concrete MRO
    # (``StageLLMCoreClient`` -> ``StageLLMCoreClientBase`` -> ``StageCoreClientBase``
    # -> ``AsyncMPClient`` ...). A plain ``super().<m>()`` would resolve to the
    # abstract stub (a no-op), NOT to ``AsyncMPClient`` / ``DPLBAsyncMPClient``.
    # ``super(StageCoreClientBase, self)`` starts MRO lookup *after*
    # ``StageCoreClientBase``, so it reaches the real vLLM implementation —
    # preserving ``DPLBAsyncMPClient``'s data-parallel overrides for the DPLB
    # client. Providing these concrete overrides also clears the abstract
    # methods so the concrete clients can be instantiated.

    async def add_request_async(self, request: StageLLMCoreRequest) -> None:
        """Add a request to the stage engine core."""
        logger.debug(
            "[%s] stage-%s [rep-%s] add request: %s",
            self.__class__.__name__,
            self.stage_id,
            self.replica_id,
            request.request_id,
        )
        await super(StageCoreClientBase, self).add_request_async(request)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        """Abort in-flight requests on the stage engine core."""
        await super(StageCoreClientBase, self).abort_requests_async(request_ids)

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Forward a control RPC to the underlying stage engine core."""
        return await super(StageCoreClientBase, self).collective_rpc_async(method, timeout, args, kwargs)

    def shutdown(self, timeout: float | None = None) -> None:
        """Shut down the stage engine core and its transport."""
        super(StageCoreClientBase, self).shutdown(timeout)

    def process_core_inputs(
        self,
        source_outputs: list[RequestOutput],
        prompt: OmniPromptType | list[OmniPromptType] | None = None,
        streaming_context: StreamingInputState | None = None,
    ) -> list[OmniTokensPrompt]:
        """Transform upstream stage outputs into this stage's inputs."""
        if self.custom_process_input_func is not None:
            signature = inspect.signature(self.custom_process_input_func)
            if len(signature.parameters) >= 4:
                return self.custom_process_input_func(
                    source_outputs,
                    prompt,
                    self.requires_multimodal_data,
                    streaming_context,
                )
            return self.custom_process_input_func(
                source_outputs,
                prompt,
                self.requires_multimodal_data,
            )

        if not self.engine_input_source:
            raise ValueError(f"engine_input_source empty for stage {self.stage_id}")
        return self._default_process_core_inputs(source_outputs, prompt, self.requires_multimodal_data)

    def get_kv_sender_info(
        self,
        *,
        base_port: int = 50051,
        kv_transfer_port_offset: int = KV_TRANSFER_PORT_OFFSET,
    ) -> dict[str, Any] | None:
        """Build sender bootstrap info for diffusion KV transfer receivers."""
        if self.replica_id is None:
            raise RuntimeError(
                f"Stage-{self.stage_id} KV sender info requested but replica_id is unset; "
                "the client must be constructed with metadata carrying a concrete replica_id."
            )
        if self._kv_sender_info is not None:
            return dict(self._kv_sender_info)

        if self._core_host is None:
            self._core_host = self._resolve_core_host()
        if self._core_host is None:
            return None
        return {
            "host": self._core_host,
            "zmq_port": kv_zmq_port(
                base_port - KV_TRANSFER_PORT_OFFSET + kv_transfer_port_offset,
                int(self.stage_id),
                local_rank=0,
                replica_id=self.replica_id,
            ),
        }

    def _engine_dead_reason(self) -> str | None:
        """Report engine-core death (drives the base ``check_health``)."""
        if self.resources.engine_dead:
            return f"Stage-{self.stage_id} engine core is dead"
        return None

    @staticmethod
    def _default_process_core_inputs(
        source_outputs: list[RequestOutput],
        prompt: OmniPromptType | list[OmniPromptType] | None,
        requires_multimodal_data: bool,
    ) -> list[OmniTokensPrompt]:
        from vllm_omni.inputs.data import OmniTokensPrompt

        if not isinstance(prompt, list):
            prompt = [prompt]

        mm_data = {so.request_id: p.get("multi_modal_data") for so, p in zip(source_outputs, prompt)}

        return [
            OmniTokensPrompt(
                prompt_token_ids=so.outputs[0].token_ids,
                multi_modal_data=(mm_data[so.request_id] if requires_multimodal_data else None),
            )
            for so in source_outputs
        ]

    # ==================== KV sender endpoint ====================

    @staticmethod
    def _detect_local_ip() -> str | None:
        """Best-effort local IP detection for cross-node connector bootstrap."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return None

    def _resolve_core_host(self) -> str | None:
        """Resolve a routable host for this stage from its client addresses."""
        replica_host = self.client_addresses.get("replica_host")
        if replica_host:
            return replica_host
        for key in ("input_address", "output_address", "stats_update_address"):
            address = self.client_addresses.get(key)
            if not address:
                continue
            host = urlparse(address).hostname
            if host in {None, "", "*", "0.0.0.0", "::"}:
                continue
            if host in {"localhost", "127.0.0.1"}:
                detected = self._detect_local_ip()
                if detected:
                    return detected
                continue
            return host
        return self._detect_local_ip()

    def _get_kv_connector_config(self) -> dict[str, Any] | None:
        omni_kv_config = getattr(self, "_omni_kv_config", None)
        if not isinstance(omni_kv_config, dict):
            return None
        connector_config = omni_kv_config.get("connector_config")
        if not isinstance(connector_config, dict):
            return None
        return connector_config

    def _resolve_sender_host_from_config(self, connector_config: dict[str, Any]) -> str | None:
        host = connector_config.get("sender_host") or connector_config.get("host")
        if host in {None, "", "auto", "*", "0.0.0.0", "::"}:
            return self._resolve_core_host()
        return str(host)

    def _initialize_kv_sender_endpoint(self) -> None:
        if self._kv_sender_initialized:
            return
        if self.replica_id is None:
            # No replica_id (metadata-less construction) — nothing to key on.
            return
        self._kv_sender_initialized = True
        connector_config = self._get_kv_connector_config()
        if connector_config is None or connector_config.get("role") != "sender":
            return

        sender_host = self._resolve_sender_host_from_config(connector_config)
        if sender_host is not None:
            self._core_host = sender_host

        connector_type = connector_config.get("type")
        sender_port = connector_config.get("sender_zmq_port")
        if connector_type in TRANSFER_ENGINE_CONNECTOR_NAMES or sender_port is None:
            base_port = connector_config.get("zmq_port")
            if base_port is None:
                return
            base_port = os.path.expandvars(str(base_port))

            omni_kv_config = getattr(self, "_omni_kv_config", None)
            from_stage = self.stage_id
            if isinstance(omni_kv_config, dict):
                from_stage = omni_kv_config.get("omni_from_stage", from_stage)

            try:
                sender_port = kv_zmq_port(
                    int(base_port),
                    int(from_stage),
                    local_rank=0,
                    replica_id=self.replica_id,
                )
            except (TypeError, ValueError):
                logger.warning(
                    "[StageLLMCoreClient] stage-%s [rep-%s] could not resolve sender_zmq_port "
                    "from base_port=%s and from_stage=%s",
                    self.stage_id,
                    self.replica_id,
                    base_port,
                    from_stage,
                )
                return

        if self._core_host is None:
            return

        self._kv_sender_info = {
            "host": str(self._core_host),
            "zmq_port": int(sender_port),
        }


class StageLLMCoreClient(StageLLMCoreClientBase, AsyncMPClient):
    """Stage async client backed by vLLM's ``AsyncMPClient``."""


class DPLBStageLLMCoreClient(StageLLMCoreClientBase, DPLBAsyncMPClient):
    """Stage async client backed by vLLM's ``DPLBAsyncMPClient``."""
