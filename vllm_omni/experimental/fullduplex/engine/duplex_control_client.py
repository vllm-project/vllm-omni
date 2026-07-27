# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import uuid
from collections.abc import Callable

from vllm_omni.experimental.fullduplex.engine.contracts import CorrelatedRpcTransport
from vllm_omni.experimental.fullduplex.engine.lease import DuplexLeaseActivity
from vllm_omni.experimental.fullduplex.engine.messages import (
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexFence,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    SignalDuplexTurnMessage,
    TouchDuplexSessionMessage,
)

DuplexControlMessage = (
    OpenDuplexSessionMessage
    | AppendDuplexInputMessage
    | SignalDuplexTurnMessage
    | CloseDuplexSessionMessage
    | TouchDuplexSessionMessage
    | ResumeDuplexSessionMessage
)


class DuplexControlRequestError(RuntimeError):
    """Typed client-side failure returned by the duplex control plane."""

    def __init__(self, result: dict[str, object]) -> None:
        error = result.get("error")
        error_data = error if isinstance(error, dict) else {}
        self.result = result
        self.code = str(error_data.get("code") or "internal_error")
        self.retryable = bool(error_data.get("retryable", False))
        accepted_fence = result.get("accepted_fence")
        self.accepted_fence = accepted_fence if isinstance(accepted_fence, DuplexFence) else None
        lease_generation = result.get("lease_generation")
        self.lease_generation = lease_generation if isinstance(lease_generation, int) else None
        super().__init__(f"duplex {result.get('operation')} failed: {result}")


class DuplexControlClient:
    """Builds duplex commands on top of an engine-owned RPC transport."""

    def __init__(
        self,
        transport: CorrelatedRpcTransport,
        *,
        control_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._transport = transport
        self._control_id_factory = control_id_factory or (lambda: uuid.uuid4().hex)

    def execute(self, message: DuplexControlMessage, *, timeout: float | None) -> dict[str, object]:
        result_message = self._transport.execute(
            ("duplex", message.control_id),
            message,
            timeout=timeout,
            timeout_message=f"duplex control timed out after {timeout} seconds",
        )
        if not isinstance(result_message, DuplexControlResultMessage):
            raise RuntimeError(f"unexpected duplex control result type: {type(result_message).__name__}")
        if result_message.fence != message.fence:
            raise RuntimeError(
                f"duplex control fence mismatch: expected {message.fence!r}, got {result_message.fence!r}"
            )
        result = {
            "fence": result_message.fence,
            "operation": result_message.operation,
            "session_id": result_message.session_id,
            "ok": result_message.ok,
            "stage_results": list(result_message.stage_results),
            "unsupported_count": result_message.unsupported_count,
            "error_count": result_message.error_count,
            "accepted_fence": result_message.accepted_fence,
            "lease_generation": result_message.lease_generation,
            "error": (
                {
                    "code": result_message.error.code,
                    "message": result_message.error.message,
                    "retryable": result_message.error.retryable,
                }
                if result_message.error is not None
                else None
            ),
        }
        if not result_message.ok:
            raise DuplexControlRequestError(result)
        return result

    def open(
        self,
        session_id: str,
        *,
        session_mode: str,
        capabilities: dict[str, object] | None,
        session_config: dict[str, object] | None,
        runtime_config: dict[str, object] | None,
        fence: DuplexFence,
        timeout: float | None,
    ) -> dict[str, object]:
        return self.execute(
            OpenDuplexSessionMessage(
                control_id=self._control_id_factory(),
                fence=fence,
                session_id=session_id,
                session_mode=session_mode,
                capabilities=dict(capabilities or {}),
                session_config=dict(session_config or {}),
                runtime_config=dict(runtime_config or {}),
            ),
            timeout=timeout,
        )

    def append(
        self,
        session_id: str,
        *,
        mode: str,
        payload: object,
        operation_id: str | None,
        final: bool,
        expected_epoch: int | None,
        fence: DuplexFence,
        timeout: float | None,
    ) -> dict[str, object]:
        if expected_epoch is not None and expected_epoch != fence.epoch:
            raise ValueError("expected_epoch must match fence.epoch")
        return self.execute(
            AppendDuplexInputMessage(
                control_id=self._control_id_factory(),
                operation_id=operation_id,
                fence=fence,
                session_id=session_id,
                expected_epoch=expected_epoch,
                mode=mode,
                payload=payload,
                final=final,
            ),
            timeout=timeout,
        )

    def signal(
        self,
        session_id: str,
        *,
        event: str,
        fence: DuplexFence,
        next_fence: DuplexFence | None,
        session_config: dict[str, object] | None,
        runtime_config: dict[str, object] | None,
        timeout: float | None,
    ) -> dict[str, object]:
        return self.execute(
            SignalDuplexTurnMessage(
                control_id=self._control_id_factory(),
                fence=fence,
                session_id=session_id,
                event=event,
                next_fence=next_fence,
                session_config=dict(session_config) if session_config is not None else None,
                runtime_config=dict(runtime_config) if runtime_config is not None else None,
            ),
            timeout=timeout,
        )

    def close(
        self,
        session_id: str,
        *,
        reason: str,
        fence: DuplexFence,
        timeout: float | None,
    ) -> dict[str, object]:
        return self.execute(
            CloseDuplexSessionMessage(
                control_id=self._control_id_factory(),
                fence=fence,
                session_id=session_id,
                reason=reason,
            ),
            timeout=timeout,
        )

    def touch(
        self,
        session_id: str,
        *,
        fence: DuplexFence,
        activity: DuplexLeaseActivity,
        timeout: float | None,
    ) -> dict[str, object]:
        return self.execute(
            TouchDuplexSessionMessage(
                control_id=self._control_id_factory(),
                fence=fence,
                session_id=session_id,
                activity=activity.value,
            ),
            timeout=timeout,
        )

    def resume(
        self,
        session_id: str,
        *,
        fence: DuplexFence,
        expected_lease_generation: int,
        timeout: float | None,
    ) -> dict[str, object]:
        return self.execute(
            ResumeDuplexSessionMessage(
                control_id=self._control_id_factory(),
                fence=fence,
                session_id=session_id,
                expected_lease_generation=expected_lease_generation,
            ),
            timeout=timeout,
        )


__all__ = ["CorrelatedRpcTransport", "DuplexControlClient", "DuplexControlMessage"]
