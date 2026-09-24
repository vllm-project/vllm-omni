# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exactness coverage for the fused mHC post-residual kernel.

Ported from #7545's test matrix: int-view (bit-level) equality of the fused
kernel against the native formula across output dtypes, token counts, seeds,
and special values, plus dispatch/failure-cache/OOM contracts.

The kernel reproduces the native numerics structurally (see
``_mhc_post_residual_kernel``): asm-barrier adds, libdevice expf, IEEE
div_rn, ascending column sums, interleaved row-sum pairing, and a
bit-exact sigmoid formulation. These are properties of the current PyTorch
CUDA reductions and Triton libdevice (verified on torch 2.13.0, triton
3.7.1); the tests re-derive the native reference on every run, so a future
torch reduction-order change fails them — the trigger to re-derive the
kernel's orders.
"""

from __future__ import annotations

import unittest.mock

import pytest
import torch

from vllm_omni.diffusion.layers import mhc

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

DEVICES = [pytest.param("cuda", marks=pytest.mark.cuda), pytest.param("musa", marks=pytest.mark.musa)]

_SCALE = 0.125
_ITERATIONS = 20
_EPSILON = 1e-12


def _clear_caches():
    mhc._FAILED_MHC_KERNELS.clear()
    mhc._WARNED_MHC_KERNELS.clear()


@pytest.fixture(autouse=True)
def _isolated_failure_caches():
    _clear_caches()
    yield
    _clear_caches()


def _device(name):
    api = getattr(torch, name, None)
    if api is None or not api.is_available():
        pytest.skip(f"requires {name}")
    return torch.device(name)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor.contiguous().view(torch.int32)
    return tensor.contiguous().view(torch.int16)


def _make_inputs(tokens: int, seed: int, device) -> tuple:
    """Strided split views of the shared [T, 24] projection, as compute_logits produces."""
    generator = torch.Generator(device=device).manual_seed(seed)
    packed = torch.randn(tokens, 24, device=device, dtype=torch.float32, generator=generator) * 3
    return (
        packed[:, 4:8],
        packed[:, 8:].view(-1, 4, 4),
        torch.randn(1, device=device, dtype=torch.float32, generator=generator),
        torch.randn(4, device=device, dtype=torch.float32, generator=generator),
        torch.randn(1, device=device, dtype=torch.float32, generator=generator),
        torch.randn(4, 4, device=device, dtype=torch.float32, generator=generator),
    )


def _reference(args, out_dtype):
    post_logits, residual_logits, ap, bp, ar, br = args
    return mhc.MHCPostResidual.forward_native(
        post_logits,
        residual_logits,
        ap,
        bp,
        ar,
        br,
        scale=_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=out_dtype,
    )


def _launch_spy():
    """Wrap the post-residual kernel so tests can count real launches."""
    original = mhc._mhc_post_residual_kernel
    launches = []

    class Spy:
        def __getitem__(self, grid):
            launches.append(grid)
            return original[grid]

    return unittest.mock.patch.object(mhc, "_mhc_post_residual_kernel", Spy()), launches


def _failing_kernel(error: Exception):
    """A kernel stand-in that raises when launched via ``kernel[grid](...)``."""

    class Failing:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise error

            return launch

    return Failing()


def _fused(args, out_dtype):
    return mhc.MHCPostResidual()(
        *args,
        scale=_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=out_dtype,
    )


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens",
    [1, 2, 3, 7, 15, 16, 17, 31, 255, 256, 1024, 4096, 65537],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_mhc_post_residual_bit_exact(device_name, out_dtype, tokens, seed):
    dev = _device(device_name)
    args = _make_inputs(tokens, seed, dev)
    fused_post, fused_residual = _fused(args, out_dtype)
    ref_post, ref_residual = _reference(args, out_dtype)
    assert torch.equal(_bits(fused_post), _bits(ref_post))
    assert torch.equal(_bits(fused_residual), _bits(ref_residual))
    # A launch failure would have been cached; a clean cache proves the
    # kernel really produced these outputs.
    assert not mhc._FAILED_MHC_KERNELS


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_mhc_post_residual_special_values(device_name, out_dtype):
    dev = _device(device_name)
    args = _make_inputs(64, 7, dev)
    post_logits, residual_logits, ap, bp, ar, br = args
    residual_logits[0] = 0.0
    residual_logits[1] = -0.0
    residual_logits[2] = 5e-45  # smallest fp32 subnormal
    residual_logits[3] = 80.0  # exp overflow boundary after the amax shift
    residual_logits[4] = -100.0
    residual_logits[5] = 3.4e38
    special = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1e-40, 1e30, -1e30]
    for i, value in enumerate(special):
        residual_logits[8 + i, i % 4, (i * 3) % 4] = value
    br[0, 0] = float("inf")
    br[1, 1] = float("nan")

    fused_post, fused_residual = _fused(args, out_dtype)
    ref_post, ref_residual = _reference(args, out_dtype)
    # NaN payloads: compare bit patterns, not value equality.
    assert torch.equal(_bits(fused_post), _bits(ref_post))
    assert torch.equal(_bits(fused_residual), _bits(ref_residual))
    assert not mhc._FAILED_MHC_KERNELS


@pytest.mark.cuda
def test_mhc_post_residual_launches_the_kernel_on_strided_views():
    """compute_logits-style split views must take the fused path (T > 1)."""
    if not torch.cuda.is_available():
        pytest.skip("requires cuda")
    dev = torch.device("cuda")
    args = _make_inputs(97, 0, dev)
    assert not args[1].is_contiguous()

    spy, launches = _launch_spy()
    with spy:
        fused_post, fused_residual = _fused(args, torch.bfloat16)
    torch.accelerator.synchronize()
    assert len(launches) == 1
    assert not mhc._FAILED_MHC_KERNELS
    ref_post, ref_residual = _reference(args, torch.bfloat16)
    assert torch.equal(_bits(fused_post), _bits(ref_post))
    assert torch.equal(_bits(fused_residual), _bits(ref_residual))


@pytest.mark.cuda
def test_mhc_post_residual_propagates_out_of_memory():
    """A transient OOM must propagate, not permanently disable the kernel."""
    if not torch.cuda.is_available():
        pytest.skip("requires cuda")
    dev = torch.device("cuda")
    args = _make_inputs(64, 0, dev)

    oom_kernel = _failing_kernel(torch.OutOfMemoryError("CUDA out of memory (simulated)"))
    with unittest.mock.patch.object(mhc, "_mhc_post_residual_kernel", oom_kernel):
        with pytest.raises(torch.OutOfMemoryError):
            _fused(args, torch.bfloat16)
    assert not mhc._FAILED_MHC_KERNELS


@pytest.mark.cuda
def test_mhc_post_residual_caches_runtime_failure():
    """One launch failure falls back to native (bit-exact) and stops retrying."""
    if not torch.cuda.is_available():
        pytest.skip("requires cuda")
    dev = torch.device("cuda")
    args = _make_inputs(64, 0, dev)
    calls = []

    class CountingFailing:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(1)
                raise RuntimeError("simulated launch failure")

            return launch

    with unittest.mock.patch.object(mhc, "_mhc_post_residual_kernel", CountingFailing()):
        first_post, first_residual = _fused(args, torch.bfloat16)
        second_post, second_residual = _fused(args, torch.bfloat16)
    assert len(calls) == 1  # second call served by the failure cache
    ref_post, ref_residual = _reference(args, torch.bfloat16)
    assert torch.equal(_bits(first_post), _bits(ref_post))
    assert torch.equal(_bits(first_residual), _bits(ref_residual))
    assert torch.equal(_bits(second_post), _bits(ref_post))
    assert torch.equal(_bits(second_residual), _bits(ref_residual))


@pytest.mark.cuda
@pytest.mark.parametrize("grad_context", [torch.inference_mode, torch.no_grad])
def test_mhc_post_residual_accepts_parameters_without_active_autograd(grad_context):
    """nn.Parameter inputs under inference contexts must fuse (regression:
    keying on raw requires_grad flags rejected every production call)."""
    if not torch.cuda.is_available():
        pytest.skip("requires cuda")
    dev = torch.device("cuda")
    args = _make_inputs(128, 0, dev)
    args = (
        args[0],
        args[1],
        torch.nn.Parameter(args[2].detach().clone()),
        args[3],
        torch.nn.Parameter(args[4].detach().clone()),
        args[5],
    )
    spy, launches = _launch_spy()
    with spy:
        with grad_context():
            fused_post, fused_residual = _fused(args, torch.bfloat16)
    torch.accelerator.synchronize()
    assert len(launches) == 1
    assert not mhc._FAILED_MHC_KERNELS
    ref_post, ref_residual = _reference(args, torch.bfloat16)
    assert torch.equal(_bits(fused_post), _bits(ref_post))
    assert torch.equal(_bits(fused_residual), _bits(ref_residual))
