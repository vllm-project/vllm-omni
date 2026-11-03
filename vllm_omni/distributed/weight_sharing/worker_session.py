# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-2/3 seam: installed by the model loader, called on its owner thread.

The loader reads Bootstrap from additional_config[CONFIG_KEY], verifies the
local rank/device via SessionContext, prepares/imports weights, and installs
this object as runner.weight_sharing. The runner must advertise
supports_weight_sharing=True before loading; otherwise ordinary loading is
rejected, rather than allocating a second backbone. detach must drop model/graph references
after device work has drained. private_bytes measures this rank's non-shared
resident allocations (including KV); the configured budgets reserve peaks too.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from threading import current_thread

import msgspec

from .backend import ExportSession, ImportedWeights
from .contracts import SessionContext, ThreadOwnershipError, WeightSharingBackend
from .runtime import CONFIG_KEY, Bootstrap, DeviceBudget, RankResult


def require_sharing_loader(runner: object, additional_config: Mapping[str, object]) -> None:
    if CONFIG_KEY in additional_config and getattr(runner, "supports_weight_sharing", False) is not True:
        raise RuntimeError("weight sharing requested but the stage-2 runner integration is unavailable")


def rank_bootstrap(
    additional_config: Mapping[str, object], *, role: str, rank: int, device_index: int, device_uuid: str
) -> tuple[SessionContext, DeviceBudget, bytes]:
    """Stage-2 loader entry, before any shared/private model allocation."""
    bootstrap = msgspec.convert(additional_config[CONFIG_KEY], type=Bootstrap)
    if bootstrap.role != role or role not in ("provider", "consumer"):
        raise ValueError("incorrect sharing bootstrap role")
    if not 0 <= rank < len(bootstrap.identities) or len(bootstrap.budgets) != len(bootstrap.identities):
        raise ValueError("incomplete sharing bootstrap rank map")
    identity = bootstrap.identities[rank]
    if identity.rank != rank or identity.tp_size != len(bootstrap.identities) or identity.device_uuid != device_uuid:
        raise ValueError("sharing bootstrap does not match this rank's physical GPU")
    budget = bootstrap.budgets[rank]
    budget.validate()
    if len(bootstrap.bundles) != (len(bootstrap.identities) if role == "consumer" else 0):
        raise ValueError("incomplete sharing bootstrap bundles")
    return SessionContext(identity, device_index), budget, bootstrap.bundles[rank] if role == "consumer" else b""


class WorkerSharingSession:
    def __init__(
        self,
        backend: WeightSharingBackend,
        context: SessionContext,
        handle: ExportSession | ImportedWeights,
        *,
        private_bytes: Callable[[], int],
        detach: Callable[[], None],
    ):
        self.backend = backend
        self.context = context
        self.handle = handle
        self.private_bytes = private_bytes
        self.detach = detach
        self._owner = current_thread()
        self._detached = False

    def control(self, operation: str, messages: tuple[bytes, ...] = ()) -> dict[str, object]:
        if current_thread() is not self._owner:
            raise ThreadOwnershipError("sharing controls must run on the model loader's owner thread")
        payload = b""
        receipt = self.handle.receipt if isinstance(self.handle, ImportedWeights) else b""
        try:
            if operation in ("confirm_import", "confirm_release"):
                if len(messages) != self.context.identity.tp_size:
                    raise ValueError("sharing receipt vector does not cover all ranks")
                message = messages[self.context.identity.rank]
                if operation == "confirm_import":
                    self.backend.confirm_import(message)
                else:
                    self.backend.confirm_release(message)
            elif operation == "claim":
                payload = self.backend.claim_export(self.handle.session_id)
            elif operation == "close":
                if not self._detached:
                    self.detach()
                    self._detached = True
                receipt = self.backend.close_import(self.handle.session_id).receipt or b""
            elif operation == "release":
                self.backend.release_export(self.handle.session_id)
            elif operation == "fail":
                self.backend.mark_failed(self.handle)
            elif operation != "status":
                raise ValueError(f"unknown sharing control: {operation}")
            status = self.backend.session_status(self.handle)
            return msgspec.to_builtins(
                RankResult(
                    self.context.identity,
                    status.state,
                    status.shared_nbytes,
                    self.private_bytes(),
                    os.getpid(),
                    payload,
                    receipt,
                )
            )
        except Exception:
            # Includes model detach failures outside the backend's own calls.
            self.backend.mark_failed(self.handle)
            raise


def control_runner(runner: object, operation: str, messages: tuple[bytes, ...]) -> dict[str, object]:
    session = getattr(runner, "weight_sharing", None)
    if not isinstance(session, WorkerSharingSession):
        raise RuntimeError("stage-2 loader did not install a WorkerSharingSession")
    return session.control(operation, messages)
