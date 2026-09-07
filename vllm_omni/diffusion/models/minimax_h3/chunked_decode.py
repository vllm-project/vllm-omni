# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-local coordination for MiniMax H3 temporal chunk decoding.

This boundary owns released-code compatibility and reconciles preflight and
callback failures at fixed VAE-group rendezvous points. Cropping,
representation conversion, transport, and encoding remain runtime
responsibilities.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

import torch
import torch.distributed as dist
import torch.nn as nn

from .temporal_chunks import (
    MiniMaxH3TemporalChunkCompatibilityError,
    MiniMaxH3TemporalDecodePlan,
    _TemporalChunkCallbackSignature,
    decode_minimax_h3_temporal_chunks_compat,
    prepare_minimax_h3_temporal_chunks_compat,
)


class MiniMaxH3VideoChunkCallback(_TemporalChunkCallbackSignature, Protocol):
    """Consume one ordered, temporally committed H3 video chunk.

    ``frames`` is an owned, contiguous float32 RGB tensor in ``BCTHW`` layout
    and the ``[0, 1]`` value range. The callback runs synchronously on the
    producer's current device stream. Before reading it on another stream, the
    consumer must establish ordering from the producer stream (for example via
    an event) and retain the allocation for that stream's lifetime. The
    consumer may retain or mutate the tensor; it does not alias the returned
    complete decode.
    """


class MiniMaxH3ChunkedDecodeUnsupportedError(RuntimeError):
    """Raised when loaded MiniMax H3 code cannot honor the chunk contract."""


class MiniMaxH3ChunkCallbackPeerError(RuntimeError):
    """Raised on peer VAE ranks when the output-rank callback fails."""


class _MiniMaxH3ChunkDecodeHost(Protocol):
    """Narrow VAE capabilities required by the chunk coordinator."""

    model: nn.Module

    def _denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor: ...

    def _decode_tiling_context(
        self,
        latent: torch.Tensor,
    ) -> AbstractContextManager: ...

    def _normalize_decoded_frames(
        self,
        decoded: torch.Tensor,
    ) -> torch.Tensor: ...


class _MiniMaxH3ChunkDecodeCoordinator:
    """Run source-gated chunk decode with group-local failure semantics.

    The coordinator is a plain Python object. It caches only successful source
    validation; request callbacks, tensors, metadata, and process groups remain
    local to :meth:`decode`.

    Errors raised inside the checkpoint decoder after native collectives begin
    cannot be reconciled here: another rank may already be blocked inside that
    collective. Resolving such a failure requires native collective handling or
    supervisor-level timeout and worker-group teardown; the process group cannot
    be assumed reusable.
    """

    def __init__(self) -> None:
        self._sources_validated = False

    @staticmethod
    def _build_execution_signature(
        latent: torch.Tensor,
        plan: MiniMaxH3TemporalDecodePlan,
    ) -> tuple[int, ...]:
        """Describe every shape and plan field used by the temporal schedule."""

        dtype_codes: dict[torch.dtype | None, int] = {
            None: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
            torch.float32: 3,
            torch.float64: 4,
        }
        return (
            # Keep the layout versioned so adding a field cannot silently make
            # an older and newer worker appear compatible.
            1,
            *[int(size) for size in latent.shape],
            dtype_codes.get(latent.dtype, -1),
            *[int(size) for size in plan.latent.shape],
            dtype_codes.get(plan.latent.dtype, -1),
            int(plan.num_chunks),
            int(plan.total_frames),
            int(plan.pad_frames),
            int(plan.output_frames),
            len(plan.chunk_frames),
            *[int(frame_count) for frame_count in plan.chunk_frames],
            dtype_codes.get(plan.temporal_dtype, -1),
        )

    @staticmethod
    def _is_output_rank(
        chunk_callback: MiniMaxH3VideoChunkCallback | None,
        *,
        group: dist.ProcessGroup | None,
        device: torch.device,
    ) -> bool:
        if group is None:
            return chunk_callback is not None

        group_rank = dist.get_rank(group)
        # A valid configuration has one contribution: rank 0 contributes 1.
        # A non-zero owner contributes at least 2, and multiple owners sum to
        # at least 3, so every rank makes the same decision from one reduction.
        callback_owner = torch.tensor(
            [group_rank + 1 if chunk_callback is not None else 0],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(callback_owner, op=dist.ReduceOp.SUM, group=group)
        if int(callback_owner.item()) != 1:
            raise ValueError("MiniMax-H3 chunk decode requires a callback on exactly VAE group rank 0")
        return group_rank == 0

    @staticmethod
    def _synchronize_callback_error(
        callback_error: BaseException | None,
        *,
        group: dist.ProcessGroup | None,
        device: torch.device,
    ) -> None:
        if group is None:
            if callback_error is not None:
                raise callback_error
            return

        is_output_rank = dist.get_rank(group) == 0
        failed = torch.tensor(
            [int(is_output_rank and callback_error is not None)],
            dtype=torch.int32,
            device=device,
        )
        dist.broadcast(
            failed,
            src=dist.get_global_rank(group, 0),
            group=group,
        )
        if int(failed.item()) == 0:
            return
        if is_output_rank:
            assert callback_error is not None
            raise callback_error
        raise MiniMaxH3ChunkCallbackPeerError("MiniMax-H3 video chunk callback failed on this VAE group's output rank")

    @staticmethod
    def _synchronize_preflight_error(
        preflight_error: Exception | None,
        temporal_dtype: torch.dtype | None,
        execution_signature: tuple[int, ...] | None,
        *,
        group: dist.ProcessGroup | None,
        device: torch.device,
    ) -> None:
        if group is None:
            if preflight_error is not None:
                raise preflight_error
            return
        failed = torch.tensor(
            [int(preflight_error is not None)],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
        if int(failed.item()) == 0:
            dtype_codes = {
                None: 0,
                torch.float16: 1,
                torch.bfloat16: 2,
                torch.float32: 3,
            }
            code = torch.tensor(
                [dtype_codes.get(temporal_dtype, -1)],
                dtype=torch.int32,
                device=device,
            )
            lowest = code.clone()
            highest = code.clone()
            dist.all_reduce(lowest, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(highest, op=dist.ReduceOp.MAX, group=group)
            if int(lowest.item()) == int(highest.item()) and int(lowest.item()) >= 0:
                if execution_signature is None:
                    raise RuntimeError("MiniMax-H3 temporal decode preflight produced no execution signature")

                # The plan length is shape-dependent, so first agree on a
                # common tensor size and then compare the complete signatures.
                # This keeps all ranks in tensor collectives and catches a
                # schedule mismatch before entering the decoder's collectives.
                signature_length = torch.tensor(
                    [len(execution_signature)],
                    dtype=torch.int64,
                    device=device,
                )
                max_signature_length = signature_length.clone()
                dist.all_reduce(
                    max_signature_length,
                    op=dist.ReduceOp.MAX,
                    group=group,
                )
                padded_signature = torch.full(
                    (int(max_signature_length.item()),),
                    -2,
                    dtype=torch.int64,
                    device=device,
                )
                padded_signature[: len(execution_signature)] = torch.tensor(
                    execution_signature,
                    dtype=torch.int64,
                    device=device,
                )
                gathered_signatures = [torch.empty_like(padded_signature) for _ in range(dist.get_world_size(group))]
                dist.all_gather(gathered_signatures, padded_signature, group=group)
                if any(not torch.equal(padded_signature, peer) for peer in gathered_signatures):
                    raise MiniMaxH3ChunkedDecodeUnsupportedError(
                        "MiniMax-H3 VAE ranks built incompatible temporal decode plans"
                    )
                return
            raise MiniMaxH3ChunkedDecodeUnsupportedError(
                "MiniMax-H3 VAE ranks selected incompatible temporal concat dtypes"
            )
        if preflight_error is not None:
            raise preflight_error
        raise MiniMaxH3ChunkedDecodeUnsupportedError("A peer VAE rank rejected the MiniMax-H3 temporal chunk preflight")

    def decode(
        self,
        host: _MiniMaxH3ChunkDecodeHost,
        latent: torch.Tensor,
        chunk_callback: MiniMaxH3VideoChunkCallback | None,
        *,
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        """Decode one latent while publishing committed normalized chunks."""

        plan: MiniMaxH3TemporalDecodePlan | None = None
        preflight_error: Exception | None = None
        if latent.ndim != 5:
            preflight_error = MiniMaxH3ChunkedDecodeUnsupportedError(
                f"MiniMax-H3 video latent must be rank 5, got {tuple(latent.shape)}"
            )
        else:
            try:
                denormalized = host._denormalize_latent(latent)
                plan = prepare_minimax_h3_temporal_chunks_compat(
                    host.model,
                    denormalized,
                    validate_sources=not self._sources_validated,
                )
                self._sources_validated = True
                del denormalized
            except MiniMaxH3TemporalChunkCompatibilityError as exc:
                preflight_error = MiniMaxH3ChunkedDecodeUnsupportedError(str(exc))
            except Exception as exc:  # noqa: BLE001
                # Every VAE rank must reach the preflight reduction even when
                # one node's private planner or environment fails locally.
                exc.__traceback__ = None
                preflight_error = exc
        self._synchronize_preflight_error(
            preflight_error,
            plan.temporal_dtype if plan is not None else None,
            self._build_execution_signature(latent, plan) if plan is not None else None,
            group=group,
            device=latent.device,
        )
        assert plan is not None
        is_output_rank = self._is_output_rank(
            chunk_callback,
            group=group,
            device=latent.device,
        )
        callback_error: BaseException | None = None
        next_chunk_index = 0
        expected_total_chunks: int | None = None
        next_frame_start = 0
        saw_final = False

        def publish(
            decoded_chunk: torch.Tensor,
            *,
            chunk_index: int,
            total_chunks: int,
            frame_start: int,
            is_final: bool,
        ) -> None:
            nonlocal callback_error, expected_total_chunks
            nonlocal next_chunk_index, next_frame_start, saw_final
            if not is_output_rank or callback_error is not None:
                return
            try:
                if expected_total_chunks is None:
                    expected_total_chunks = total_chunks
                if (
                    saw_final
                    or total_chunks <= 0
                    or total_chunks != expected_total_chunks
                    or chunk_index != next_chunk_index
                    or chunk_index >= total_chunks
                    or frame_start != next_frame_start
                    or is_final != (chunk_index + 1 == total_chunks)
                ):
                    raise RuntimeError(
                        "MiniMax-H3 temporal backend emitted discontinuous "
                        "chunk metadata: "
                        f"index={chunk_index}/{next_chunk_index}, "
                        f"total={total_chunks}/{expected_total_chunks}, "
                        f"frame_start={frame_start}/{next_frame_start}, "
                        f"final={is_final}, after_final={saw_final}"
                    )
                frames = host._normalize_decoded_frames(decoded_chunk)
                frames = frames.clone(memory_format=torch.contiguous_format)
                frame_count = int(frames.shape[2])
                if frame_count <= 0:
                    raise RuntimeError("MiniMax-H3 temporal backend emitted an empty chunk")
                next_chunk_index += 1
                next_frame_start += frame_count
                saw_final = is_final
                assert chunk_callback is not None
                chunk_callback(
                    frames,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    frame_start=frame_start,
                    is_final=is_final,
                )
            except BaseException as exc:  # noqa: BLE001
                # Keep a failed GPU chunk and its callback locals out of the
                # traceback while peer ranks finish the remaining collectives.
                exc.__traceback__ = None
                callback_error = exc

        # Do not catch and reduce exceptions from this region. A peer may
        # already be blocked in one of the checkpoint's native tile
        # collectives, in which case a new adapter-level collective would also
        # deadlock. A native backend or supervisor must handle timeout and
        # worker-group teardown.
        with host._decode_tiling_context(latent):
            decoded = decode_minimax_h3_temporal_chunks_compat(
                host.model,
                plan,
                publish,
            )
        frames = host._normalize_decoded_frames(decoded)
        if (
            is_output_rank
            and callback_error is None
            and (not saw_final or next_chunk_index != expected_total_chunks or next_frame_start != int(frames.shape[2]))
        ):
            callback_error = RuntimeError(
                "MiniMax-H3 temporal backend did not reconstruct the full "
                "decode: "
                f"final={saw_final}, "
                f"chunks={next_chunk_index}/{expected_total_chunks}, "
                f"frames={next_frame_start}/{frames.shape[2]}"
            )
        self._synchronize_callback_error(
            callback_error,
            group=group,
            device=frames.device,
        )
        return frames


__all__ = [
    "MiniMaxH3ChunkCallbackPeerError",
    "MiniMaxH3ChunkedDecodeUnsupportedError",
    "MiniMaxH3VideoChunkCallback",
]
