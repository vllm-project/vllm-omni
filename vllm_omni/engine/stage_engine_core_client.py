"""
Stage Engine Core Client for vLLM-Omni multi-stage runtime.

Directly inherits from vLLM's AsyncMPClient to reuse EngineCore architecture.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import AsyncMPClient

from vllm_omni.distributed.omni_connectors.utils.initialization import KV_TRANSFER_PORT_OFFSET
from vllm_omni.engine.stage_init_utils import StageMetadata

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutput

    from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


class StageEngineCoreClient(AsyncMPClient):
    """Stage async client that inherits from vLLM's AsyncMPClient.

    Fully reuses AsyncMPClient for:
    - ZMQ setup, sockets
    - outputs_queue, output_queue_task
    - All utility methods (get_output_async, abort_requests_async, etc.)

    The subprocess is spawned externally via ``spawn_stage_core`` /
    ``complete_stage_handshake`` from *stage_engine_core_proc.py*.
    """

    def __init__(
        self,
        vllm_config: Any,
        executor_class: type,
        log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        proc: Any = None,
        client_count: int = 1,
        client_index: int = 0,
        *,
        metadata: StageMetadata | None = None,
        engine_manager: Any = None,
        coordinator: Any = None,
    ):
        """Create an async EngineCore client for a single stage.

        All heavy init (config extraction, plugin loading, device setup,
        engine args building, device locking) is done by the Orchestrator
        via helpers in stage_init_utils.py. This constructor just stores metadata
        and calls super().__init__().

        The subprocess is spawned externally via ``spawn_stage_core`` /
        ``complete_stage_handshake`` (see *stage_engine_core_proc.py*).
        The resulting ``proc`` handle is passed in so this client can
        manage the process lifecycle on shutdown.
        """
        # -------- Stage metadata (public fields used at runtime) --------
        if metadata is not None:
            self.stage_id = metadata.stage_id
            self.stage_type = metadata.stage_type
            self.engine_output_type = metadata.engine_output_type
            self.is_comprehension = metadata.is_comprehension
            self.requires_multimodal_data = metadata.requires_multimodal_data
            self.engine_input_source = metadata.engine_input_source
            self.final_output = metadata.final_output
            self.final_output_type = metadata.final_output_type
            self.default_sampling_params = metadata.default_sampling_params
            self.custom_process_input_func = metadata.custom_process_input_func
            self.model_stage = metadata.model_stage

        self.engine_outputs: Any = None
        self._proc = proc
        self.client_addresses = dict(client_addresses or {})
        self._omni_kv_config = getattr(getattr(vllm_config, "model_config", None), "omni_kv_config", None)
        self._kv_sender_host = self._resolve_contact_host()
        self._kv_sender_info: dict[str, Any] | None = None
        self._kv_sender_initialized = False

        logger.info(
            "[StageEngineCoreClient] Stage-%s initializing EngineCore",
            self.stage_id,
        )
        try:
            super().__init__(
                vllm_config,
                executor_class,
                log_stats=log_stats,
                client_addresses=client_addresses,
                client_count=client_count,
                client_index=client_index,
            )
        except Exception:
            logger.exception(
                "[StageEngineCoreClient] Stage-%s EngineCore init failed",
                self.stage_id,
            )
            try:
                self.shutdown()
            except Exception as shutdown_error:
                logger.warning(
                    "[StageEngineCoreClient] Stage-%s cleanup after init failure failed: %s",
                    self.stage_id,
                    shutdown_error,
                )
            raise
        self._initialize_kv_sender_endpoint()
        logger.info(
            "[StageEngineCoreClient] Stage-%s EngineCore running",
            self.stage_id,
        )

    # ==================== Overrides ====================

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        """Add request to the stage engine core."""
        logger.info(f"[StageEngineCoreClient] Stage-{self.stage_id} adding request: {request.request_id}")
        await super().add_request_async(request)

    # ==================== Stage Methods ====================

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

    def _resolve_contact_host(self) -> str | None:
        """Resolve a routable host for this stage from its client addresses."""
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
            return self._resolve_contact_host()
        return str(host)

    def _initialize_kv_sender_endpoint(self) -> None:
        if self._kv_sender_initialized:
            return
        self._kv_sender_initialized = True
        connector_config = self._get_kv_connector_config()
        if connector_config is None or connector_config.get("role") != "sender":
            return

        sender_host = self._resolve_sender_host_from_config(connector_config)
        if sender_host is not None:
            self._kv_sender_host = sender_host

        sender_port = connector_config.get("sender_zmq_port")
        if sender_port is None:
            base_port = connector_config.get("zmq_port")
            if base_port is None:
                return

            omni_kv_config = getattr(self, "_omni_kv_config", None)
            from_stage = self.stage_id
            if isinstance(omni_kv_config, dict):
                from_stage = omni_kv_config.get("omni_from_stage", from_stage)

            try:
                sender_port = int(base_port) + KV_TRANSFER_PORT_OFFSET + int(from_stage)
            except (TypeError, ValueError):
                logger.warning(
                    "[StageEngineCoreClient] Stage-%s could not resolve sender_zmq_port "
                    "from base_port=%s and from_stage=%s",
                    self.stage_id,
                    base_port,
                    from_stage,
                )
                return

        if self._kv_sender_host is None:
            return

        self._kv_sender_info = {
            "host": str(self._kv_sender_host),
            "zmq_port": int(sender_port),
        }

    def get_kv_sender_info(
        self,
        *,
        base_port: int = 50051,
        kv_transfer_port_offset: int = KV_TRANSFER_PORT_OFFSET,
    ) -> dict[str, Any] | None:
        """Build sender bootstrap info for diffusion KV transfer receivers.

        ``base_port`` and ``kv_transfer_port_offset`` are only used by the
        legacy fallback path when no connector-level sender endpoint is
        configured in ``omni_kv_config``.
        """
        if self._kv_sender_info is not None:
            return dict(self._kv_sender_info)

        if self._kv_sender_host is None:
            self._kv_sender_host = self._resolve_contact_host()
        if self._kv_sender_host is None:
            return None
        return {
            "host": self._kv_sender_host,
            "zmq_port": base_port + kv_transfer_port_offset + int(self.stage_id),
        }

    def set_engine_outputs(self, engine_outputs: EngineCoreOutput) -> None:
        """Set engine outputs (called by orchestrator)."""
        self.engine_outputs = engine_outputs

    def process_engine_inputs(
        self,
        stage_list: list[Any],
        prompt: OmniTokensPrompt | list[OmniTokensPrompt] | None = None,
    ) -> list[OmniTokensPrompt]:
        """Process inputs from upstream stages."""
        from vllm_omni.inputs.data import OmniTokensPrompt

        if self.custom_process_input_func is not None:
            return self.custom_process_input_func(
                stage_list,
                self.engine_input_source,
                prompt,
                self.requires_multimodal_data,
            )

        if not self.engine_input_source:
            raise ValueError(f"engine_input_source empty for stage {self.stage_id}")

        source_id = self.engine_input_source[0]
        source_outputs = stage_list[source_id].engine_outputs

        if not isinstance(prompt, list):
            prompt = [prompt]

        mm_data = {so.request_id: p.get("multi_modal_data") for so, p in zip(source_outputs, prompt)}

        return [
            OmniTokensPrompt(
                prompt_token_ids=so.outputs[0].token_ids,
                multi_modal_data=(mm_data[so.request_id] if self.requires_multimodal_data else None),
            )
            for so in source_outputs
        ]

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Forward control RPCs to the underlying AsyncMPClient stage engine.

        Each ``StageEngineCoreClient`` already represents one logical stage, so
        stage-scoped control operations should be executed here and then fanned
        in-core across the workers managed by this EngineCore client.
        """
        return await super().collective_rpc_async(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    def shutdown(self) -> None:
        """Shutdown ZMQ connections and the subprocess."""
        super().shutdown()
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
