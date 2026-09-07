# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the XPU all-gather landing buffer in the platform layer.

Communication libraries register the device memory handed to them and keep that
registration alive. Allocating a fresh output for every all-gather therefore
leaks registrations whenever the caching allocator hands back a different
address -- which any `empty_cache()` on the request path guarantees.
`XPUOmniPlatform.all_gather_into_tensor` lands the collective in a reused
buffer and copies out, so the registered address is stable while callers keep
owning their result.

That reuse lives on the XPU platform only, because the copy-out is not free and
the address churn is specific to the XPU collective backend. These tests
therefore drive both platforms: the reuse bodies point
`group_coordinator.current_omni_platform` at `XPUOmniPlatform`, and
`_body_default_path_unchanged` pins that the default `OmniPlatform`
implementation still allocates and gathers with no landing buffer and no copy.

These run on gloo/CPU (no accelerator), matching `test_comm.py`.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import os
import sys
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed import group_coordinator as gc_module
from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.platforms.interface import OmniPlatform

_REAL_PLATFORM = gc_module.current_omni_platform


class _AbsentKernelsFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Satisfy `vllm_xpu_kernels` imports with empty modules.

    `vllm_omni.platforms.xpu.platform` imports vLLM's `XPUPlatform`, which
    imports the compiled `vllm_xpu_kernels` extensions at module scope. Those
    ship only in an XPU build, so on a CPU runner the real class is otherwise
    unimportable. Nothing under test touches a kernel -- the landing buffer is
    plain `torch.empty`/`view`/`clone` and runs on CPU tensors over gloo -- so
    stubbing the missing extensions keeps the class under test the real
    `XPUOmniPlatform` rather than a re-implementation of it.
    """

    _NAME = "vllm_xpu_kernels"

    def find_spec(self, name, path=None, target=None):
        if name == self._NAME or name.startswith(f"{self._NAME}."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        pass


def _xpu_platform() -> type[OmniPlatform]:
    """Import the real `XPUOmniPlatform`, stubbing absent XPU kernel packages.

    Called inside the spawned worker so the module-scope side effects of
    `vllm_omni.platforms.xpu.platform` (an accelerator memory-info override and
    a sampler env var) stay in the child process.
    """
    try:
        from vllm_omni.platforms.xpu.platform import XPUOmniPlatform
    except ModuleNotFoundError:
        sys.meta_path.insert(0, _AbsentKernelsFinder())
        from vllm_omni.platforms.xpu.platform import XPUOmniPlatform

    return XPUOmniPlatform


@contextlib.contextmanager
def _platform(platform):
    """Point the coordinator's module-level platform at `platform`.

    The hooks are classmethods, so the class itself is everything the
    coordinator needs -- no instance and no accelerator. Yields the platform it
    replaced and puts it back on exit.
    """
    saved = gc_module.current_omni_platform
    gc_module.current_omni_platform = platform
    try:
        yield saved
    finally:
        gc_module.current_omni_platform = saved


def _buffers() -> dict:
    """The XPU platform's landing buffers -- class state, not per coordinator."""
    return _xpu_platform()._all_gather_buffers


def _run(
    body: Callable[[GroupCoordinator, int, int], None],
    local_rank: int,
    world_size: int,
    master_port: int,
    as_xpu: bool = True,
) -> None:
    """Build a gloo group and a coordinator directly.

    Deliberately *not* via `init_distributed_environment`: that helper does
    `get_rank() % get_device_count()` and therefore needs at least one visible
    accelerator, which an L1 CPU runner does not have. Everything under test
    here is device-agnostic -- CPU tensors over gloo exercise the same code
    path -- so the coordinator is constructed directly and stays runnable with
    zero accelerators.

    The coordinator is built *before* the platform is swapped: `__init__` asks
    the platform for a torch device and the XPU answer is an `xpu` device this
    runner has none of. Only `all_gather` runs under the platform being tested,
    which is the code path these tests are about.
    """
    for key, value in {
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(master_port),
    }.items():
        os.environ[key] = value

    dist.init_process_group(backend="gloo", init_method="env://", world_size=world_size, rank=local_rank)
    xpu_platform = _xpu_platform()
    # Class state must not carry into (or out of) a case.
    xpu_platform.reset_all_gather_buffers()
    try:
        group = GroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=local_rank,
            torch_distributed_backend="gloo",
        )
        with _platform(xpu_platform if as_xpu else OmniPlatform):
            body(group, local_rank, world_size)
    finally:
        xpu_platform.reset_all_gather_buffers()
        dist.destroy_process_group()


def _rank_tensor(local_rank: int, rows: int = 3, cols: int = 4) -> torch.Tensor:
    """Rank-distinguishable payload, so a wrong gather cannot look right."""
    return torch.full((rows, cols), float(local_rank + 1)) + torch.arange(rows * cols, dtype=torch.float32).reshape(
        rows, cols
    )


def _expected(world_size: int, rows: int = 3, cols: int = 4) -> torch.Tensor:
    return torch.cat([_rank_tensor(r, rows, cols) for r in range(world_size)], dim=0)


# ---------------------------------------------------------------------------
# worker bodies
# ---------------------------------------------------------------------------


def _body_parity(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """Reused buffer must produce exactly what a fresh allocation produces."""
    out = group.all_gather(_rank_tensor(local_rank))
    torch.testing.assert_close(out, _expected(world_size), rtol=0, atol=0)


def _body_landing_address_is_stable(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """The property that actually fixes the leak: one registered address.

    Three same-shape gathers must all land in the same buffer. Without reuse the
    platform allocates a new output each time and this dict stays empty.
    """
    seen = []
    for _ in range(3):
        group.all_gather(_rank_tensor(local_rank))
        buffers = _buffers()
        assert buffers, "reuse is enabled but no landing buffer was cached"
        assert len(buffers) == 1, f"expected one buffer per (dtype, device), got {len(buffers)}"
        seen.append(next(iter(buffers.values())).data_ptr())
    assert len(set(seen)) == 1, f"landing buffer moved between calls: {seen}"


def _body_reset_drops_the_buffers(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """`reset_all_gather_buffers` must really drop the class state.

    After the reset the dict is empty and the next gather lands somewhere else;
    a reset that only looked like it cleared would keep serving the old buffer.
    The old buffer is held alive on purpose, so the allocator cannot hand its
    address straight back and make the second half of this vacuous.
    """
    platform = _xpu_platform()
    group.all_gather(_rank_tensor(local_rank))
    kept_alive = next(iter(_buffers().values()))
    first_ptr = kept_alive.data_ptr()

    platform.reset_all_gather_buffers()
    assert _buffers() == {}, "reset left landing buffers behind"

    group.all_gather(_rank_tensor(local_rank))
    buffers = _buffers()
    assert len(buffers) == 1, f"the gather after reset must cache one buffer, got {len(buffers)}"
    assert next(iter(buffers.values())).data_ptr() != first_ptr, "the gather after reset reused the dropped buffer"
    assert kept_alive.numel() > 0, "kept_alive must stay referenced until here"


def _body_result_is_private(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """Callers still own their result -- the copy-out must not be skipped.

    A caller may hold the previous result across the next gather; if we handed
    back the landing buffer itself, the second gather would rewrite the first
    result in place.
    """
    first = group.all_gather(_rank_tensor(local_rank))
    first_snapshot = first.clone()
    first.mul_(-1.0)  # a caller mutating what it owns must not corrupt the buffer

    second = group.all_gather(_rank_tensor(local_rank))
    torch.testing.assert_close(second, _expected(world_size), rtol=0, atol=0)
    torch.testing.assert_close(first, -first_snapshot, rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr(), "two results must not alias each other"


def _body_non_zero_dim(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """`dim != 0` reshapes the gathered result; reuse must not disturb it."""
    out = group.all_gather(_rank_tensor(local_rank), dim=1)
    expected = torch.cat([_rank_tensor(r) for r in range(world_size)], dim=1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _body_separate_tensors(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """`separate_tensors=True` returns per-rank views; they must stay correct."""
    parts = group.all_gather(_rank_tensor(local_rank), separate_tensors=True)
    assert len(parts) == world_size
    for rank, part in enumerate(parts):
        torch.testing.assert_close(part, _rank_tensor(rank), rtol=0, atol=0)


def _body_grows_for_larger_shape(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """A bigger gather grows the single buffer instead of adding another."""
    group.all_gather(_rank_tensor(local_rank, rows=2))
    small = next(iter(_buffers().values())).numel()
    group.all_gather(_rank_tensor(local_rank, rows=8))
    buffers = _buffers()
    assert len(buffers) == 1, f"buffer count must stay 1 per (dtype, device), got {len(buffers)}"
    grown = next(iter(buffers.values())).numel()
    assert grown > small, "buffer did not grow for the larger gather"
    # Going back below the record high must not shrink or re-key the buffer:
    # that is what bounds the residency by the largest gather instead of by the
    # number of shapes, and what keeps the address stable afterwards.
    group.all_gather(_rank_tensor(local_rank, rows=2))
    buffers = _buffers()
    assert len(buffers) == 1, f"a smaller gather added a second buffer: {len(buffers)}"
    assert next(iter(buffers.values())).numel() == grown, "buffer shrank below its record high"


def _body_default_path_unchanged(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """Off XPU, `all_gather` must be exactly what main does.

    Two properties together pin that: the default `OmniPlatform` implementation
    creates no landing buffer, and the tensor handed back to the caller *is* the
    tensor the collective wrote into -- i.e. there is no extra clone on the hot
    path.
    """
    assert gc_module.current_omni_platform is OmniPlatform, "this body must run on the default implementation"

    real_all_gather_into_tensor = dist.all_gather_into_tensor
    written_to: list[int] = []

    def recording(output, input_, **kwargs):
        written_to.append(output.data_ptr())
        return real_all_gather_into_tensor(output, input_, **kwargs)

    dist.all_gather_into_tensor = recording
    try:
        for _ in range(3):
            out = group.all_gather(_rank_tensor(local_rank))
            torch.testing.assert_close(out, _expected(world_size), rtol=0, atol=0)
            assert out.data_ptr() == written_to[-1], "off XPU the result must not be a copy"
            assert _buffers() == {}, "off XPU no landing buffer may be allocated"
    finally:
        dist.all_gather_into_tensor = real_all_gather_into_tensor

    # Note deliberately *not* asserted: that the three outputs sit at three
    # distinct addresses. Off XPU each call does allocate afresh, but the
    # caching allocator is free to hand back the block the previous result
    # released, so identical addresses are legal and would make this flaky.


def _body_pipeline_subclass_inherits_all_gather(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """The PP coordinator inherits `all_gather` without calling `super().__init__`.

    `PipelineGroupCoordinator` re-implements `__init__`, so any per-instance
    state the base constructor sets up is absent there while the inherited
    `all_gather` still runs -- which is exactly why the landing buffer belongs
    to the platform class rather than to a coordinator. Build one through the
    real factory (with the real platform restored, since the factory asks it for
    a torch device) and exercise the inherited method.
    """
    from vllm_omni.diffusion.distributed.parallel_state import init_model_parallel_group

    with _platform(_REAL_PLATFORM):
        pp_group = init_model_parallel_group(
            group_ranks=[list(range(world_size))],
            local_rank=local_rank,
            backend="gloo",
            parallel_mode="pipeline",
        )
    out = pp_group.all_gather(_rank_tensor(local_rank))
    torch.testing.assert_close(out, _expected(world_size), rtol=0, atol=0)
    assert len(_buffers()) == 1, "the inherited all_gather must use the platform's landing buffer"


# ---------------------------------------------------------------------------
# spawn wrappers
# ---------------------------------------------------------------------------


_BODIES = {
    "parity": _body_parity,
    "pipeline": _body_pipeline_subclass_inherits_all_gather,
    "stable": _body_landing_address_is_stable,
    "reset": _body_reset_drops_the_buffers,
    "private": _body_result_is_private,
    "dim1": _body_non_zero_dim,
    "separate": _body_separate_tensors,
    "grow": _body_grows_for_larger_shape,
    "default": _body_default_path_unchanged,
}


def _entry(local_rank: int, world_size: int, master_port: int, body_name: str, as_xpu: bool) -> None:
    _run(_BODIES[body_name], local_rank, world_size, master_port, as_xpu=as_xpu)


def _spawn(world_size: int, master_port: int, body_name: str, as_xpu: bool = True) -> None:
    torch.multiprocessing.spawn(
        _entry,
        args=(world_size, master_port, body_name, as_xpu),
        nprocs=world_size,
    )


# ---------------------------------------------------------------------------
# CPU: gloo collectives on CPU tensors (no accelerator)
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_matches_fresh_allocation(world_size: int):
    _spawn(world_size, 29660 + world_size, "parity")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_pipeline_coordinator_inherits_all_gather(world_size: int):
    """MRO regression: the subclass that skips `super().__init__` must still work."""
    _spawn(world_size, 29740 + world_size, "pipeline")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_landing_buffer_address_is_stable(world_size: int):
    _spawn(world_size, 29680 + world_size, "stable")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_reset_all_gather_buffers_drops_them(world_size: int):
    """The process-level teardown hook must free the class state, not fake it."""
    _spawn(world_size, 29770 + world_size, "reset")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_result_does_not_alias_landing_buffer(world_size: int):
    _spawn(world_size, 29690 + world_size, "private")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_non_zero_dim(world_size: int):
    _spawn(world_size, 29700 + world_size, "dim1")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_separate_tensors(world_size: int):
    _spawn(world_size, 29710 + world_size, "separate")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_landing_buffer_grows_instead_of_multiplying(world_size: int):
    _spawn(world_size, 29720 + world_size, "grow")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_default_platform_all_gather_is_unchanged(world_size: int):
    """The other half: no landing buffer and no copy on the default platform."""
    _spawn(world_size, 29750 + world_size, "default", as_xpu=False)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_default_platform_all_gather_matches_fresh_allocation(world_size: int):
    """Correctness parity on the default path, for completeness."""
    _spawn(world_size, 29760 + world_size, "parity", as_xpu=False)
