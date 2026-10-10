# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in, lossless final VAE pixel assembly on the replica primary.

Only the decoder's final output uses this policy. Halo/global-attention
collectives and every rank's temporal cache remain under the original VAE.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist

MODES = ("native", "root_float", "root_uint8_allgather", "root_uint8_p2p")
_POLICY: ContextVar[PixelOutputPolicy | None] = ContextVar("lingbot_pixel_output", default=None)


class _CompletionEvent(Protocol):
    def query(self) -> bool: ...


_PENDING_PIXEL_OWNERS: list[tuple[_CompletionEvent, tuple[object, ...]]] = []


def _retain_until_complete(owners, device):
    if device.type != "npu":
        return
    # Work.wait installs a stream dependency; it need not block the host.
    # An event after reassembly also covers all receive-buffer consumers.
    _PENDING_PIXEL_OWNERS[:] = [(event, held) for event, held in _PENDING_PIXEL_OWNERS if not event.query()]
    if len(_PENDING_PIXEL_OWNERS) >= 32:
        raise RuntimeError("Final-pixel completion owners failed to retire; restart eight workers")
    event = torch.npu.Event()
    event.record(torch.npu.current_stream(device))
    _PENDING_PIXEL_OWNERS.append((event, tuple(owners)))


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f"lingbot_npu_pixel_output_mode must be one of {MODES}")
    return mode


@dataclass(frozen=True)
class PixelOutputPolicy:
    mode: str
    group: object
    rank: int
    world_size: int

    def __post_init__(self):
        validate_mode(self.mode)
        if self.mode == "native" or self.world_size != 8 or not 0 <= self.rank < 8:
            raise ValueError("Pixel output policy requires a non-native eight-rank mode")

    @property
    def planar_uint8(self):
        return self.mode.startswith("root_uint8_")

    @property
    def produce_output(self):
        return self.rank == 0


@contextmanager
def pixel_output_context(policy):
    if _POLICY.get() is not None:
        raise RuntimeError("Nested pixel-output execution is not supported")
    token = _POLICY.set(policy)
    try:
        yield
    finally:
        _POLICY.reset(token)


def current_pixel_output_policy():
    return _POLICY.get()


def exact_planar_pixels(video):
    # Decoder.decode_chunk ordinarily clamps [-1,1] before _uint8_frames.
    # Move these elementwise operations before spatial/temporal concatenation
    # while retaining every decoder-dtype and FP32 rounding boundary.
    from .lingbot_pixels import planar_uint8_tensor

    return planar_uint8_tensor(video.clamp(-1.0, 1.0))


def gather_final_pixels(local_video, *, policy, split_dim, expected_extent):
    """Collect planar uint8 shards. No fallback after cache writes/communication."""
    if split_dim not in ("width", "height"):
        raise ValueError("Pixel output requires a spatial split")
    if local_video.ndim != 5 or local_video.shape[:2] != (1, 3):
        raise ValueError("Final pixel shard must be floating [1,3,F,H,W]")
    if not local_video.is_floating_point() or expected_extent is None or expected_extent <= 0:
        raise ValueError("Final pixel shard has invalid dtype/extent metadata")
    # Complete owners include the original decoder output and every send/recv
    # allocation until Work.wait has installed the stream completion dependency.
    owners = [local_video]
    try:
        pixels = exact_planar_pixels(local_video).contiguous()
        owners.append(pixels)
        dim = 3 if split_dim == "width" else 2
        if pixels.shape[dim] * policy.world_size < expected_extent:
            raise ValueError("Final pixel shards cannot cover the authoritative full extent")
        if policy.mode == "root_uint8_allgather":
            received = [torch.empty_like(pixels) for _ in range(policy.world_size)]
            owners.extend(received)
            dist.all_gather(received, pixels, group=policy.group)
        elif policy.mode == "root_uint8_p2p":
            received = [None] * policy.world_size
            ops = []
            if policy.produce_output:
                received[0] = pixels
                for peer in range(1, policy.world_size):
                    received[peer] = torch.empty_like(pixels)
                    owners.append(received[peer])
                    global_peer = dist.get_global_rank(policy.group, peer)
                    ops.append(dist.P2POp(dist.irecv, received[peer], global_peer, policy.group))
            else:
                root = dist.get_global_rank(policy.group, 0)
                ops.append(dist.P2POp(dist.isend, pixels, root, policy.group))
            works = dist.batch_isend_irecv(ops)
            owners.extend(works)
            for work in works:
                work.wait()
        else:
            raise ValueError("Byte assembly requires an explicit uint8 mode")
        if not policy.produce_output:
            _retain_until_complete(owners, pixels.device)
            return pixels.new_empty(0)
        result = torch.cat(received, dim=dim)
        if result.shape[dim] != expected_extent:
            result = result.narrow(dim, 0, expected_extent).contiguous()
        owners.append(result)
        _retain_until_complete(owners, result.device)
        return result
    except BaseException as error:
        from .lingbot_sp8_fatal import report_failure

        report_failure("final_pixel_output", str(error), owners)
        raise


def planar_pixels_to_host(pixels):
    from .lingbot_pixels import _interleave_planar

    if pixels.dtype != torch.uint8 or pixels.ndim != 4 or pixels.shape[0] != 3:
        raise ValueError("Pixel assembly returned an invalid planar arena")
    return _interleave_planar(pixels.cpu().numpy())
