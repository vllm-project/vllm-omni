# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-rank failure agreement and tensor fan-out shared by MiniMax H3 code.

These helpers live outside ``pipeline_minimax_h3`` so sibling modules (notably
``timeline_guide_encoding``) can use them without importing the pipeline, which
imports them back. ``pipeline_minimax_h3`` re-exports every name here, so
existing ``pipeline_minimax_h3.<helper>`` references and monkeypatch targets
keep resolving.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.errors import client_error_from_metadata, is_client_error_status

logger = init_logger(__name__)


def _dit_rank_world() -> tuple[Any, int, int]:
    if not dist.is_initialized():
        return None, 0, 1
    group = get_world_group().device_group
    return group, dist.get_rank(group), dist.get_world_size(group)


def _broadcast_rank0_exception(exc: Exception | None) -> None:
    """Synchronize a rank-0-only exception across every DiT rank.

    H3 reference-video preparation runs only on rank 0; the other DiT ranks
    return ``None`` without touching disk. When rank 0 raises inside that
    path it exits :meth:`prepare_encode` before reaching the downstream
    ``dist.broadcast`` calls, and non-zero ranks then hang on those
    collectives forever. Every rank calls this helper right after the
    rank-0-only work, before any subsequent collective, so all ranks either
    raise the same error together or all continue.

    Unlike :func:`_synchronize_any_rank_exception` this deliberately reports
    only rank 0. The asymmetry is intentional: the guarded work runs on rank 0
    alone, so there is no other rank whose failure could outrank it and no
    cross-rank precedence to resolve.
    """
    group, rank, world_size = _dit_rank_world()
    if world_size == 1:
        if exc is not None:
            raise exc
        return
    if rank == 0:
        if exc is None:
            payload: list[Any] = [None]
        else:
            payload = [
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "status_code": getattr(exc, "status_code", None),
                    "error_type": getattr(exc, "error_type", None),
                }
            ]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0, group=group)
    info = payload[0]
    if info is None:
        return
    if rank == 0:
        assert exc is not None
        raise exc
    # Rebuild a matching client-facing error on non-zero ranks so the runner's
    # per-request try/except records the same 4xx status as rank 0. The exact
    # subclass need not survive the wire; the message and status suffice.
    status_code = info.get("status_code")
    error_type = info.get("error_type")
    message = f"[rank 0] {info['type']}: {info['message']}"
    if status_code is not None:
        raise client_error_from_metadata(
            message,
            status_code=int(status_code),
            error_type=error_type,
        )
    raise RuntimeError(message)


def _synchronize_any_rank_exception(exc: Exception | None) -> None:
    """Agree on any rank's failure after local work and residency cleanup.

    An unknown/fatal failure on *any* rank outranks a client error on a lower
    rank. Raising the first non-``None`` entry would let a rank-0 4xx mask a
    rank-1 ``RuntimeError``, and the serving boundary treats an
    origin-qualified 4xx as proof that the worker rejected the request
    cleanly - which would release guided uploads while another rank is still
    in an unknown state. This mirrors the all-ranks rule already enforced by
    ``MultiprocDiffusionExecutor._unwrap_rpc_result_envelope``.
    """
    group, _, world_size = _dit_rank_world()
    if world_size == 1:
        if exc is not None:
            raise exc
        return
    local_error = (
        None
        if exc is None
        else {
            "type": type(exc).__name__,
            "message": str(exc),
            "status_code": getattr(exc, "status_code", None),
            "error_type": getattr(exc, "error_type", None),
        }
    )
    errors: list[Any] = [None] * world_size
    dist.all_gather_object(errors, local_error, group=group)
    failed = [(rank, info) for rank, info in enumerate(errors) if info is not None]
    if not failed:
        return

    client_failure = all(is_client_error_status(info["status_code"]) for _, info in failed)
    if client_failure:
        selected_rank, selected = failed[0]
    else:
        selected_rank, selected = next(
            (rank, info) for rank, info in failed if not is_client_error_status(info["status_code"])
        )

    if len(failed) > 1:
        # Other ranks stay out of the client-visible message: the serving
        # layer echoes it verbatim (see the RPC sanitization in
        # ``multiproc_executor``).
        logger.error(
            "MiniMax H3 failed on DiT rank(s) %s; reporting rank %d. Details: %s",
            [rank for rank, _ in failed],
            selected_rank,
            "; ".join(f"[rank {rank}] {info['type']}: {info['message']}" for rank, info in failed),
        )

    message = f"[rank {selected_rank}] {selected['type']}: {selected['message']}"
    if client_failure:
        raise client_error_from_metadata(
            message,
            status_code=int(selected["status_code"]),
            error_type=selected["error_type"],
        )
    raise RuntimeError(message)


def _broadcast_tensor(
    tensor: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    group, rank, world_size = _dit_rank_world()
    if world_size == 1:
        if tensor is None:
            raise ValueError("source tensor is required for single-rank execution")
        return tensor.to(device=device, dtype=dtype)

    shape = torch.zeros(5, dtype=torch.long, device=device)
    if rank == 0:
        if tensor is None:
            raise ValueError("rank 0 must provide a tensor to broadcast")
        shape[0] = tensor.ndim
        shape[1 : tensor.ndim + 1] = torch.tensor(
            tensor.shape,
            device=device,
        )
    dist.broadcast(shape, src=0, group=group)
    ndim = int(shape[0].item())
    tensor_shape = tuple(int(v) for v in shape[1 : ndim + 1].tolist())
    if rank == 0:
        output = tensor.to(device=device, dtype=dtype).contiguous()
    else:
        output = torch.empty(tensor_shape, device=device, dtype=dtype)
    dist.broadcast(output, src=0, group=group)
    return output
