# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Process and platform setup for isolated collective test workers."""

from collections.abc import Callable
from multiprocessing.context import BaseContext

import torch


def collective_context(device_kind: str) -> BaseContext:
    """Reuse imports in a clean CPU forkserver; CUDA workers use spawn."""
    if device_kind == "cpu" and "forkserver" in torch.multiprocessing.get_all_start_methods():
        torch.multiprocessing.set_forkserver_preload(["tests.helpers.cpu_collective_server"])
        return torch.multiprocessing.get_context("forkserver")
    return torch.multiprocessing.get_context("spawn")


def start_collective_workers(
    worker: Callable[..., None], *, args: tuple[object, ...], nprocs: int, device_kind: str
) -> None:
    context = collective_context(device_kind)
    if context.get_start_method() == "spawn":
        torch.multiprocessing.spawn(worker, args=args, nprocs=nprocs)
    else:
        torch.multiprocessing.start_processes(worker, args=args, nprocs=nprocs, start_method=context.get_start_method())


def configure_cpu_collective_worker() -> None:
    """Select CPU platforms inside a spawned worker that only uses Gloo.

    Upstream coordinators resolve local ranks through the selected
    platform's visible devices. CPU ranks can outnumber visible GPUs, so they
    must use the CPU platform even when the test host has CUDA installed.
    The worker still creates real process groups and runs real collectives.
    """
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform

    from vllm_omni.diffusion.distributed import comm, group_coordinator, parallel_state
    from vllm_omni.platforms.interface import UnspecifiedOmniPlatform

    class CpuCollectivePlatform(UnspecifiedOmniPlatform):
        dist_backend = "gloo"

        @classmethod
        def get_device_count(cls) -> int:
            # CPU tensors share one logical device regardless of process rank.
            return 1

        @classmethod
        def set_device(cls, device: torch.device) -> None:
            assert device.type == "cpu"

        @classmethod
        def synchronize(cls) -> None:
            pass

    vllm.platforms.current_platform = CpuPlatform()
    platform = CpuCollectivePlatform()
    for module in (comm, group_coordinator, parallel_state):
        module.current_omni_platform = platform
