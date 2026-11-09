# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt the existing loader sessions to stage RPCs on the CUDA owner thread."""

from __future__ import annotations

import gc
from collections.abc import Callable
from threading import current_thread

import msgspec

from .backend import ExportSession, ImportedWeights
from .contracts import SessionContext, ThreadOwnershipError, WeightSharingBackend
from .runtime import CONFIG_KEY, Bootstrap, DeviceBudget, RankResult
from .session import SharedWeightGroupConfig, build_session_context


def resolve_worker_sharing(
    config, *, role: str, rank: int | None = None
) -> tuple[SharedWeightGroupConfig, DeviceBudget] | None:
    """Read rank-local bootstrap without mutating the config.

    Runners may omit rank after TP initialization; workers pass their rank
    explicitly before initialization. The IPC codec checks the physical GPU.
    """
    if CONFIG_KEY not in config.additional_config:
        return None
    if rank is None:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    bootstrap = msgspec.convert(config.additional_config[CONFIG_KEY], type=Bootstrap)
    if role not in ("provider", "consumer") or not 0 <= rank < len(bootstrap.groups):
        raise ValueError("incorrect sharing bootstrap role or rank")
    if len(bootstrap.budgets) != len(bootstrap.groups):
        raise ValueError("incomplete sharing bootstrap rank map")
    group = bootstrap.groups[rank]
    if group.role != role or group.rank != rank or group.tp_size != len(bootstrap.groups):
        raise ValueError("sharing bootstrap does not match this rank")
    if bool(group.payload) != (role == "consumer"):
        raise ValueError("incorrect sharing bootstrap payload")
    if not all((group.model_id, group.checkpoint_revision, group.layout)):
        raise ValueError("missing sharing model agreement")
    budget = bootstrap.budgets[rank]
    budget.validate()
    return group, budget


def sharing_memory_request(
    budget: DeviceBudget | None, *, role: str, rank: int, kv_cache_bytes: int | None, free_bytes: int
) -> int | None:
    """Check the known KV lower bound; weight/private peaks are checked after warmup.

    Fixed KV sizes skip automatic profiling, so this envelope does not allocate
    memory or size the KV cache. None preserves ordinary stage ratio handling.
    """
    if budget is None:
        return None
    expected = budget.kv_cache_bytes if role == "provider" else budget.consumer_kv_cache_bytes
    if not kv_cache_bytes or kv_cache_bytes != expected:
        raise ValueError("shared stage requires the fixed KV cache budget admitted by the sharing group")
    required = kv_cache_bytes + budget.reserve_bytes
    if free_bytes < required:
        raise ValueError(
            f"rank {rank}: insufficient free memory for shared {role} startup "
            f"(free={free_bytes / 1024**3:.3f}, required={required / 1024**3:.3f} GiB)"
        )
    return min(budget.capacity_bytes, free_bytes - budget.reserve_bytes)


def install_runner_session(
    runner, config, *, role: str, rank: int, device_index: int, detach: Callable[[], None]
) -> None:
    """Reuse the backend and handle already created by the model loader."""
    sharing = resolve_worker_sharing(config, role=role, rank=rank)
    if sharing is None:
        return
    group, _ = sharing
    if getattr(runner, "weight_sharing", None) is not None:
        raise RuntimeError("sharing session is already installed")
    backend = runner._weight_share_backend
    handle = runner._weight_share_session if group.role == "provider" else runner.pipeline._weight_share_import
    if handle is None:
        raise RuntimeError("shared loading completed without a session handle")
    import torch

    owned_peak = max(
        torch.accelerator.max_memory_allocated(device_index),
        torch.accelerator.max_memory_reserved(device_index),
    )
    shared = backend.session_status(handle).shared_nbytes if group.role == "provider" else 0

    def memory_usage() -> tuple[int, int, int]:
        nonlocal owned_peak
        from vllm_omni.platforms import current_omni_platform

        # IPC imports are not owned caching-allocator allocations. Only the
        # provider subtracts shared storage bytes; allocator slack stays private.
        resident = current_omni_platform.memory_reserved(device_index)
        owned_peak = max(
            owned_peak,
            torch.accelerator.max_memory_allocated(device_index),
            torch.accelerator.max_memory_reserved(device_index),
            resident,
        )
        return max(0, owned_peak - shared), max(0, resident - shared), current_omni_platform.get_free_memory()

    runner.weight_sharing = WorkerSharingSession(
        backend,
        build_session_context(group, device_index),
        handle,
        memory_usage=memory_usage,
        detach=detach,
    )


def detach_diffusion_runner(runner) -> None:
    """Drop model references after the stage has drained; backend detects leftovers."""
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.synchronize()
    runner.release_captured_graphs()
    runner.state_cache.clear()
    runner.vllm_config.compilation_config.static_forward_context.clear()
    # Drop consumers that retain the pipeline; shared loading forbids offload,
    # LoRA and weight updates. KV connector shutdown remains worker-owned.
    runner.cache_backend = None
    runner.prompt_embed_cache = None
    runner._interaction_coordinator = None
    runner.input_batch = None
    runner.pipeline = None
    gc.collect()


class WorkerSharingSession:
    def __init__(
        self,
        backend: WeightSharingBackend,
        context: SessionContext,
        handle: ExportSession | ImportedWeights,
        *,
        memory_usage: Callable[[], tuple[int, int, int | None]],
        detach: Callable[[], None],
    ):
        self.backend = backend
        self.context = context
        self.handle = handle
        self.memory_usage = memory_usage
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
            elif operation == "synchronize":
                from vllm_omni.platforms import current_omni_platform

                current_omni_platform.synchronize()
            elif operation != "status":
                raise ValueError(f"unknown sharing control: {operation}")
            status = self.backend.session_status(self.handle)
            # Memory admission is irrelevant to cleanup. A failed device query
            # must not prevent reporting a completed release to the provider.
            peak, resident, free = (
                self.memory_usage() if operation in ("claim", "status", "confirm_import") else (0, 0, None)
            )
            return msgspec.to_builtins(
                RankResult(
                    self.context.identity,
                    status.state,
                    status.shared_nbytes,
                    peak,
                    payload,
                    receipt,
                    free,
                    resident,
                )
            )
        except Exception:
            # Includes model detach failures outside the backend's own calls.
            self.backend.mark_failed(self.handle)
            raise


def control_runner(runner: object, operation: str, messages: tuple[bytes, ...]) -> dict[str, object]:
    session = getattr(runner, "weight_sharing", None)
    if not isinstance(session, WorkerSharingSession):
        # Loading may fail after creating the backend and session, before returning
        # a pipeline or handle. Mark the partial session as failed as well.
        backend = getattr(runner, "_weight_share_backend", None)
        session_id = getattr(runner, "_weight_share_session_id", None)
        if operation == "fail" and backend is not None and session_id is not None:
            backend.mark_failed(session_id)
            return {}
        raise RuntimeError("model loader did not install a WorkerSharingSession")
    return session.control(operation, messages)
