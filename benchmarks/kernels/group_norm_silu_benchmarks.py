# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark for the fused GroupNorm+SiLU operators and their split reduction.

Both operators launch one program per ``(batch, group)`` pair unless the spatial
axis is split. Every GroupNorm in the diffusion stack uses 32 groups and the VAE
decodes at batch 1, so without a split the largest activations in the pipeline
run on 32 CTAs regardless of how many SMs the device has. This script measures
what that costs and what the split recovers.

Usage:
    # Compare eager / no-split / split at the HunyuanImage3 decode shapes:
    python benchmarks/kernels/group_norm_silu_benchmarks.py

    # Sweep the CTA-waves target that sizes the split, to pick the default:
    python benchmarks/kernels/group_norm_silu_benchmarks.py --sweep-waves 1,2,4,8

    # Just one operator, one dtype:
    python benchmarks/kernels/group_norm_silu_benchmarks.py --op adaptive --dtype bfloat16

Two things to read carefully in the output:

* ``split off`` is *this* kernel with the split disabled, but still on the
  current autotune space. That is not the same as the kernel as it shipped:
  the config space changed too. Pass ``--baseline released`` for the true
  before/after, which restores the shipped ``BLOCK_SIZE=4096, num_warps=16``
  space as well as disabling the split.
* ``% copy`` is measured against a device-to-device copy of the same footprint,
  not a datasheet peak, because a copy is the realistic ceiling for a streaming
  kernel. The copy itself does not reach hardware peak (~480 of 600 GB/s on an
  A10G), so values slightly over 100% mean "streaming as fast as a copy", not
  "faster than memory".
"""

import argparse
import contextlib
import os
import sys

import torch
import torch.nn.functional as F
from vllm.triton_utils import triton

from vllm_omni.model_executor.models.common.ops import (
    fused_adaptive_group_norm_silu,
    fused_group_norm_silu,
)
from vllm_omni.model_executor.models.common.ops._group_norm_reduction import (
    _SPLIT_WAVES_ENV,
    pick_split,
)

# (label, channels, spatial HW). The first six walk the HunyuanImage3 VAE
# decoder ladder for a 1024x1024 image (32x spatial compression, channels
# halving as resolution doubles); the last two are DiT ResBlock activations,
# where the operator is launch-bound and the split must not make things worse.
SHAPES = [
    ("vae/1024^2", 128, (1024, 1024)),
    ("vae/512^2", 256, (512, 512)),
    ("vae/256^2", 512, (256, 256)),
    ("vae/128^2", 1024, (128, 128)),
    ("vae/64^2", 1024, (64, 64)),
    ("vae/32^2", 1024, (32, 32)),
    ("dit/64^2", 1024, (64, 64)),
    ("dit/32^2", 2048, (32, 32)),
]

NUM_GROUPS = 32
EPS = 1e-6


def _bench(fn) -> float:
    """Median milliseconds for one call."""
    return triton.testing.do_bench(fn, warmup=25, rep=100)


# The autotune space these operators shipped with before the split reduction:
# one block size, one warp count, varying only the pipeline depth. Kept here so
# the "released vs now" numbers in the PR can be reproduced from this script
# rather than taken on trust -- it is also the space under which the fused
# operator loses to eager on small activations.
_RELEASED_CONFIGS = [triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=s) for s in (1, 2, 4, 6)]


@contextlib.contextmanager
def _released_kernels():
    """Re-decorate both operators' kernels with the pre-split autotune space."""
    import vllm_omni.model_executor.models.common.ops._group_norm_reduction as reduction

    plain = sys.modules["vllm_omni.model_executor.models.common.ops.fused_group_norm_silu"]
    adaptive = sys.modules["vllm_omni.model_executor.models.common.ops.fused_adaptive_group_norm_silu"]
    saved = (
        plain._group_norm_silu_kernel,
        adaptive._adaptive_group_norm_silu_kernel,
        reduction.group_norm_partial_stats_kernel,
    )
    tune = triton.autotune(configs=_RELEASED_CONFIGS, key=reduction.SPLIT_REDUCTION_KEY)
    plain._group_norm_silu_kernel = tune(saved[0].fn)
    adaptive._adaptive_group_norm_silu_kernel = tune(saved[1].fn)
    reduction.group_norm_partial_stats_kernel = tune(saved[2].fn)
    try:
        # Released behaviour is one CTA per (batch, group), so also disable the split.
        with _waves(0):
            yield
    finally:
        plain._group_norm_silu_kernel = saved[0]
        adaptive._adaptive_group_norm_silu_kernel = saved[1]
        reduction.group_norm_partial_stats_kernel = saved[2]


def _copy_bandwidth(device, dtype, nbytes: int) -> float:
    """Achievable GB/s for a same-size device-to-device copy, as the reference.

    A datasheet peak is not the right yardstick: it is unreachable, and the
    fraction of it a kernel hits depends on the access pattern. A plain copy of
    the same footprint is the honest ceiling for a streaming kernel.
    """
    n = nbytes // torch.tensor([], dtype=dtype).element_size()
    src = torch.empty(n, dtype=dtype, device=device)
    dst = torch.empty_like(src)
    ms = _bench(lambda: dst.copy_(src))
    return (2 * nbytes) / (ms * 1e-3) / 1e9


def _make_case(op: str, batch: int, channels: int, hw, dtype, device):
    x = torch.randn(batch, channels, *hw, device=device, dtype=dtype)
    weight = torch.randn(channels, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype)
    if op == "plain":
        eager = lambda: F.silu(F.group_norm(x, NUM_GROUPS, weight, bias, EPS))  # noqa: E731
        fused = lambda: fused_group_norm_silu(x, weight, bias, NUM_GROUPS, EPS)  # noqa: E731
    else:
        scale = torch.randn(batch, channels, device=device, dtype=dtype) * 0.1
        shift = torch.randn(batch, channels, device=device, dtype=dtype) * 0.1
        bc = (batch, channels) + (1,) * len(hw)

        def eager():
            normed = F.group_norm(x, NUM_GROUPS, weight, bias, EPS)
            return F.silu(normed * (1.0 + scale.reshape(bc)) + shift.reshape(bc))

        fused = lambda: fused_adaptive_group_norm_silu(  # noqa: E731
            x, weight, bias, scale, shift, NUM_GROUPS, EPS
        )
    return x, eager, fused


@contextlib.contextmanager
def _waves(waves):
    """Size the split for ``waves`` CTA waves (0 = no split) inside the block."""
    prev = os.environ.get(_SPLIT_WAVES_ENV)
    os.environ[_SPLIT_WAVES_ENV] = str(waves)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_SPLIT_WAVES_ENV, None)
        else:
            os.environ[_SPLIT_WAVES_ENV] = prev


def _with_waves(waves, fn):
    """Run ``fn`` with the split sized for ``waves`` CTA waves (0 = no split)."""
    with _waves(waves):
        return fn()


def run_compare(op, batch, dtype, device, sms, released=False):
    baseline = "released" if released else "split off"
    print(
        f"\n### op={op} batch={batch} dtype={str(dtype).replace('torch.', '')} groups={NUM_GROUPS} baseline={baseline}"
    )
    header = (
        f"{'shape':<12} {'C':>5} {'MB':>7} {'split':>6} {'CTAs':>6} "
        f"{'eager ms':>9} {baseline:>10} {'split':>9} {'GB/s':>8} {'% copy':>7} {'vs base':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, channels, hw in SHAPES:
        x, eager, fused = _make_case(op, batch, channels, hw, dtype, device)
        nbytes = x.numel() * x.element_size()
        # 2 reads + 1 write is what the fused kernel actually moves.
        traffic = 3 * nbytes

        split, _ = pick_split(x[0, 0].numel(), batch, NUM_GROUPS, device)
        ms_eager = _bench(eager)
        ms_off = _with_waves(0, lambda: _bench(fused))
        ms_split = _bench(fused)

        gbs = traffic / (ms_split * 1e-3) / 1e9
        copy_gbs = _copy_bandwidth(device, dtype, nbytes)
        print(
            f"{label:<12} {channels:>5} {nbytes / 2**20:>7.1f} {split:>6} {batch * NUM_GROUPS * split:>6} "
            f"{ms_eager:>9.3f} {ms_off:>10.3f} {ms_split:>9.3f} {gbs:>8.1f} "
            f"{100 * gbs / copy_gbs:>6.1f}% {ms_off / ms_split:>7.2f}x"
        )
        x = eager = fused = None
        torch.accelerator.empty_cache()
    print(f"(device has {sms} SMs; without a split every row above would run on {batch * NUM_GROUPS} CTAs)")


def run_sweep(op, batch, dtype, device, waves_list):
    print(f"\n### waves sweep: op={op} batch={batch} dtype={str(dtype).replace('torch.', '')}")
    header = f"{'shape':<12} {'C':>5} " + "".join(f"{'w=' + str(w):>12}" for w in waves_list)
    print(header)
    print("-" * len(header))
    for label, channels, hw in SHAPES:
        x, _eager, fused = _make_case(op, batch, channels, hw, dtype, device)
        spatial = x[0, 0].numel()
        cells = []
        best_ms, best_w = None, None
        for w in waves_list:
            ms = _with_waves(w, lambda: _bench(fused))  # noqa: B023
            split = _with_waves(w, lambda: pick_split(spatial, batch, NUM_GROUPS, device)[0])  # noqa: B023
            cells.append((w, ms, split))
            if best_ms is None or ms < best_ms:
                best_ms, best_w = ms, w
        row = f"{label:<12} {channels:>5} "
        for w, ms, split in cells:
            mark = "*" if w == best_w else " "
            row += f"{ms:>8.3f}/{split:<2d}{mark}"
        print(row)
        x = _eager = fused = None
        torch.accelerator.empty_cache()
    print("cells are  ms/split ; * marks the fastest waves value for that shape")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", choices=["plain", "adaptive", "both"], default="both")
    parser.add_argument(
        "--baseline",
        choices=["split-off", "released"],
        default="split-off",
        help=(
            "what to compare against: 'split-off' is this kernel with the split disabled; "
            "'released' also restores the pre-split BLOCK_SIZE=4096/num_warps=16 autotune "
            "space, i.e. the operator as it shipped"
        ),
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16,float32")
    parser.add_argument(
        "--sweep-waves",
        default=None,
        help="comma-separated CTA-waves targets to sweep instead of the compare table, e.g. 1,2,4,8",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA/ROCm device.")
    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    print(f"device: {torch.cuda.get_device_name(0)} ({sms} SMs)")

    ops = ["plain", "adaptive"] if args.op == "both" else [args.op]
    dtypes = [getattr(torch, d.strip()) for d in args.dtype.split(",")]

    for op in ops:
        for dtype in dtypes:
            if args.sweep_waves:
                run_sweep(op, args.batch, dtype, device, [int(w) for w in args.sweep_waves.split(",")])
            else:
                run_compare(op, args.batch, dtype, device, sms, released=args.baseline == "released")


if __name__ == "__main__":
    main()
