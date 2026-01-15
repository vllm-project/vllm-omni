# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_rank_world(
    group_getter: Callable[[], dist.ProcessGroup] | None,
) -> tuple[int | None, int | None]:
    if not dist.is_initialized():
        return None, None

    group: dist.ProcessGroup | None = None
    if group_getter is not None:
        try:
            group = group_getter()
        except Exception:
            group = None

    try:
        if group is None:
            return dist.get_rank(), dist.get_world_size()
        return dist.get_rank(group), dist.get_world_size(group)
    except Exception:
        return None, None


class VaeDecodeProfiler:
    """Lightweight VAE decode profiler wrapper.

    This is meant to be installed as an instance-level override of `vae.decode`
    so pipelines don't need model-specific code paths.
    """

    def __init__(
        self,
        vae: Any,
        *,
        label: str,
        group_getter: Callable[[], dist.ProcessGroup] | None,
    ) -> None:
        self._vae = vae
        self._label = label
        self._group_getter = group_getter
        self._orig_decode = vae.decode

    def decode(self, z: torch.Tensor, *args: Any, **kwargs: Any):
        device = getattr(z, "device", None)
        is_cuda = bool(device is not None and device.type == "cuda")

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = self._orig_decode(z, *args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - t0) * 1000

        dist_rank, dist_world_size = _get_rank_world(self._group_getter)
        rank_str = dist_rank if dist_rank is not None else "na"
        world_str = dist_world_size if dist_world_size is not None else "na"

        if is_cuda:
            peak_alloc_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
            logger.debug(
                "%s VAE decode profile: rank=%s/%s time_ms=%.3f peak_alloc_gib=%.3f peak_reserved_gib=%.3f",
                self._label,
                rank_str,
                world_str,
                dt_ms,
                peak_alloc_gib,
                peak_reserved_gib,
            )
        else:
            logger.debug(
                "%s VAE decode profile: rank=%s/%s time_ms=%.3f",
                self._label,
                rank_str,
                world_str,
                dt_ms,
            )

        return out


def maybe_install_vae_decode_profiler(
    pipeline: Any,
    *,
    enabled: bool,
    group_getter: Callable[[], dist.ProcessGroup] | None = None,
) -> None:
    if not enabled:
        return

    vae = getattr(pipeline, "vae", None)
    if vae is None or not hasattr(vae, "decode"):
        return

    if getattr(vae, "_vllm_vae_decode_profiler_installed", False):
        return

    wrapper = VaeDecodeProfiler(
        vae,
        label=type(pipeline).__name__,
        group_getter=group_getter,
    )

    vae._vllm_vae_decode_profiler_installed = True  # type: ignore[attr-defined]
    vae._vllm_vae_decode_profiler_original_decode = vae.decode  # type: ignore[attr-defined]
    vae.decode = wrapper.decode  # type: ignore[assignment]

