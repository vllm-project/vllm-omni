# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frames-only callback coordination for MiniMax-H3 VAE decoding."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from .temporal_chunks import decode_temporal_chunks

MiniMaxH3VideoChunkCallback = Callable[[torch.Tensor], None]


def decode_h3_chunks(
    host,
    latent: torch.Tensor,
    callback: MiniMaxH3VideoChunkCallback | None,
    *,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Run the local temporal loop, synchronizing callback errors across ranks."""
    streaming = callback is not None
    owner = streaming
    if group is not None:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        # Ownership is positional: every rank supplies the callback and rank 0
        # publishes. Reduce only whether the ranks agree that this decode
        # streams, so the verdict is identical everywhere -- a rank-local
        # verdict would let a peer fall through and enter the decoder
        # collectives alone, hanging the stage instead of failing it.
        census = torch.tensor([int(streaming)], dtype=torch.int64, device=latent.device)
        dist.all_reduce(census, group=group)
        supplied = int(census.item())
        if supplied not in (0, world_size):
            raise ValueError(
                "MiniMax-H3 chunk callback must be supplied on every rank of the VAE "
                f"group or on none of them; got {supplied} of {world_size}"
            )
        # No rank supplied a callback: every rank runs a plain full decode.
        streaming = supplied == world_size
        owner = streaming and rank == 0

    error: BaseException | None = None

    def publish(raw: torch.Tensor) -> None:
        nonlocal error
        if not owner or error is not None:
            return
        try:
            processor = getattr(host.model, "processor", None)
            decoded = raw if processor is None else processor.revert_tensor(raw)
            frames = host._normalize_decoded_frames(decoded).contiguous()
            assert callback is not None
            callback(frames)
        except BaseException as exc:  # noqa: BLE001
            error = exc

    # Peer ranks run the temporal loop without materializing a second video.
    sink = publish if owner else ((lambda _frames: None) if streaming else None)
    result = decode_temporal_chunks(
        host.model,
        host._denormalize_latent(latent),
        sink,
    )

    if group is not None:
        failed = torch.tensor([int(rank == 0 and error is not None)], dtype=torch.int32, device=latent.device)
        dist.broadcast(failed, src=dist.get_global_rank(group, 0), group=group)
        if int(failed.item()):
            if rank == 0:
                assert error is not None
                raise error
            raise RuntimeError("MiniMax-H3 video chunk callback failed on rank zero")
    elif error is not None:
        raise error

    if not result.numel():
        return result
    processor = getattr(host.model, "processor", None)
    decoded = result if processor is None else processor.revert_tensor(result)
    return host._normalize_decoded_frames(decoded)
