# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""DuplexOmniEngine: the duplex engine sibling of ``AsyncOmniEngine``.

It is deliberately small. The engine base owns the stage processes, the
orchestrator thread and the correlated RPC transport; this class only knows
how duplex session messages enter the engine (open/close/resume/touch through
correlated RPC, session commands one-way) and which orchestrator to build
(``DuplexOrchestrator`` with the model plugin loaded).
"""

from __future__ import annotations

import asyncio
import queue
import uuid
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionError,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    TouchDuplexSessionMessage,
)
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, EncodeAudio, load_duplex_plugin
from vllm_omni.engine.messages import EngineQueueMessage
from vllm_omni.engine.omni_engine_base import OmniEngineBase
from vllm_omni.engine.orchestrator import OrchestratorBase

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.commands import DuplexCommand

logger = init_logger(__name__)

_DEFAULT_CONTROL_TIMEOUT_S = 10.0
_COMMAND_PUT_TIMEOUT_S = 30.0


class DuplexOmniEngine(OmniEngineBase):
    """Engine for full-duplex models; sessions run inside ``DuplexOrchestrator``."""

    plugin: DuplexModelPlugin | None = None
    #: Set by ``_validate_deployment`` from the deploy config before any stage starts.
    duplex_session_config: DuplexSessionRuntimeConfig

    # Any: **kwargs is forwarded verbatim to ``OmniEngineBase.__init__``.
    def __init__(self, model: str, *, duplex_audio_encoder: EncodeAudio, **kwargs: Any) -> None:
        # Injected by ``DuplexOmni`` (entrypoint layer): the engine package
        # knows how to hand model audio to the plugin, not how to encode it.
        self._audio_encoder = duplex_audio_encoder
        super().__init__(model, **kwargs)

    # ---- orchestrator construction (orchestrator thread) ----

    def _validate_deployment(self) -> None:
        """Load the plugin and check the scheduler contract before any stage process starts."""
        pipeline_config = self.pipeline_config
        plugin_path = getattr(pipeline_config, "duplex_plugin", None) if pipeline_config is not None else None
        if pipeline_config is None:
            raise ValueError(
                f"{self.model!r}: the model pipeline could not be resolved (models with custom HF code "
                "such as MiniCPM-o need trust_remote_code=True)"
            )
        if not plugin_path:
            raise ValueError(f"{self.model!r} is not a duplex model: the pipeline declares no duplex_plugin")
        deploy_config = self.deploy_config
        if deploy_config is None:
            raise ValueError("DuplexOmniEngine requires a deploy config with session_mode: duplex (none resolved)")
        if getattr(deploy_config, "session_mode", "turn") != "duplex":
            raise ValueError(
                "DuplexOmniEngine requires a deploy config with session_mode: duplex "
                f"(got {getattr(deploy_config, 'session_mode', None)!r})"
            )
        self.duplex_session_config = deploy_config.duplex_session
        self.plugin = load_duplex_plugin(plugin_path, self._audio_encoder)

    # Any: signature of the ``OmniEngineBase._create_orchestrator`` seam it overrides.
    def _create_orchestrator(self, **orchestrator_kwargs: Any) -> OrchestratorBase:
        from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator

        assert self.plugin is not None, "_validate_deployment() must run before the orchestrator is created"
        stage0_vllm_config = self.stage_pools[0].stage_vllm_config if self.stage_pools else None
        model_config = getattr(stage0_vllm_config, "model_config", None)
        return DuplexOrchestrator(
            plugin=self.plugin,
            duplex_session_config=self.duplex_session_config,
            model_config=model_config,
            **orchestrator_kwargs,
        )

    # ---- deployment facts ----

    @property
    def duplex_capabilities(self) -> DuplexCapabilities:
        if self.plugin is None:
            raise RuntimeError("duplex plugin is not loaded")
        return self.plugin.capabilities(max_sessions=self.duplex_session_config.max_sessions)

    # ---- session message surface ----

    def _execute_control(
        self,
        message: EngineQueueMessage,
        *,
        control_id: str,
        operation: str,
        session_id: str,
        timeout: float | None,
    ) -> DuplexControlResultMessage:
        try:
            result = self.rpc_client.execute(
                ("duplex", control_id),
                message,
                timeout=timeout,
                timeout_message=f"duplex {operation} timed out for session {session_id}",
            )
        except TimeoutError as exc:
            raise DuplexSessionError(str(exc), code="timeout", retryable=True, session_id=session_id) from exc
        except RuntimeError as exc:
            # The RPC router is closed or the orchestrator reported a terminal error.
            raise DuplexSessionError(str(exc), code="engine_error", session_id=session_id) from exc
        if not isinstance(result, DuplexControlResultMessage):
            raise DuplexSessionError(
                f"unexpected duplex control result: {type(result).__name__}",
                code="internal_error",
                session_id=session_id,
            )
        if not result.ok:
            raise DuplexSessionError(
                result.error_message or f"duplex {operation} failed",
                code=result.error_code or "internal_error",
                retryable=result.error_retryable,
                session_id=session_id,
            )
        return result

    def _open_session(
        self,
        session_id: str,
        session_config: DuplexSessionConfig,
        *,
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexControlResultMessage:
        control_id = uuid.uuid4().hex
        return self._execute_control(
            OpenDuplexSessionMessage(
                control_id=control_id,
                session_id=session_id,
                session_config=session_config,
            ),
            control_id=control_id,
            operation="open",
            session_id=session_id,
            timeout=timeout,
        )

    async def open_session_async(
        self,
        session_id: str,
        session_config: DuplexSessionConfig,
        *,
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexControlResultMessage:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._open_session(session_id, session_config, timeout=timeout))

    def _close_session(
        self,
        session_id: str,
        *,
        reason: str = "client_close",
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexControlResultMessage:
        control_id = uuid.uuid4().hex
        return self._execute_control(
            CloseDuplexSessionMessage(
                control_id=control_id,
                session_id=session_id,
                reason=reason,
            ),
            control_id=control_id,
            operation="close",
            session_id=session_id,
            timeout=timeout,
        )

    async def close_session_async(
        self,
        session_id: str,
        *,
        reason: str = "client_close",
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexControlResultMessage:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._close_session(session_id, reason=reason, timeout=timeout),
        )

    def _resume_session(
        self,
        session_id: str,
        *,
        expected_lease_generation: int,
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexControlResultMessage:
        control_id = uuid.uuid4().hex
        return self._execute_control(
            ResumeDuplexSessionMessage(
                control_id=control_id,
                session_id=session_id,
                expected_lease_generation=expected_lease_generation,
            ),
            control_id=control_id,
            operation="resume",
            session_id=session_id,
            timeout=timeout,
        )

    async def resume_session_async(
        self,
        session_id: str,
        *,
        expected_lease_generation: int,
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexControlResultMessage:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._resume_session(
                session_id,
                expected_lease_generation=expected_lease_generation,
                timeout=timeout,
            ),
        )

    def _touch_session(
        self,
        session_id: str,
        *,
        activity: str,
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexControlResultMessage:
        control_id = uuid.uuid4().hex
        return self._execute_control(
            TouchDuplexSessionMessage(
                control_id=control_id,
                session_id=session_id,
                activity=activity,
            ),
            control_id=control_id,
            operation="touch",
            session_id=session_id,
            timeout=timeout,
        )

    async def touch_session_async(
        self,
        session_id: str,
        *,
        activity: str,
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexControlResultMessage:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._touch_session(session_id, activity=activity, timeout=timeout),
        )

    def _submit_command(self, session_id: str, command: DuplexCommand) -> None:
        """One-way: enqueue a session command in caller order (blocks only on queue backpressure)."""
        if not self.is_alive():
            raise DuplexSessionError("engine is not alive", code="engine_dead", session_id=session_id)
        message = DuplexSessionCommandMessage(session_id=session_id, command=command)
        try:
            self.request_queue.sync_q.put(message, timeout=_COMMAND_PUT_TIMEOUT_S)
        except queue.Full as exc:
            raise DuplexSessionError(
                "engine request queue is full", code="engine_backpressure", retryable=True, session_id=session_id
            ) from exc

    async def submit_command_async(self, session_id: str, command: DuplexCommand) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._submit_command(session_id, command))


__all__ = ["DuplexOmniEngine"]
