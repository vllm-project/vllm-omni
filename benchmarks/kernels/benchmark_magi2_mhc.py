# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare native and fused MAGI-2 mHC post-processing on CUDA or MUSA.

Run from the checkout root with PyTorch and the selected backend dependencies.
torchada is loaded automatically only on MUSA::

    python -m benchmarks.kernels.benchmark_magi2_mhc \
        --tokens 1 4 16 64 129 4096 --hidden-size 3072 --num-tests 30

The released config supplies hidden=3072, four streams and 20 Sinkhorn steps.
Token counts are synthetic; supply per-rank sizes from the target trace when
comparing production shapes. FP32 post/residual logits are strided views of the
same [tokens, 24] projection, as in MHCHandler.compute_logits. Mix inputs use
the selected dtype. Both mix implementations consume identical coefficients.

The three comparisons cover coefficient preparation, stream mixing, and their
combined pipeline (including intermediate allocations, excluding projection,
normalization and apply_pre). This is an eager operator diagnostic, without
compile or graph capture. It does not measure full model performance.
No CUDA/MUSA device (or no PyTorch) produces a JSON skip record and exit status zero.
"""

import argparse
import math
from functools import partial
from typing import Any

from benchmarks.kernels._magi2_musa_bench import (
    add_arguments,
    check_pair,
    emit,
    load_musa,
    measure_pair,
    report_environment,
)


def benchmark_shape(torch: Any, mhc: Any, args: argparse.Namespace, tokens: int) -> None:
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    packed = torch.randn(tokens, 24, device=device, dtype=torch.float32)
    # Preserve the token stride of the model's shared projection allocation.
    post_inputs = (
        packed[:, 4:8],
        packed[:, 8:].view(tokens, 4, 4),
        torch.tensor([0.8], device=device),
        torch.randn(4, device=device),
        torch.tensor([1.1], device=device),
        torch.randn(4, 4, device=device),
    )
    streams = torch.randn(tokens, 4, args.hidden_size, device=device, dtype=dtype)
    branch = torch.randn(tokens, args.hidden_size, device=device, dtype=dtype)
    post_op, mix_op = mhc.MHCPostResidual(), mhc.MHCMix()
    scale = 1 / math.sqrt(4 * args.hidden_size)
    kwargs = dict(scale=scale, iterations=args.sinkhorn_iterations, epsilon=1e-12, out_dtype=dtype)
    prepare_reference = partial(post_op.forward_native, *post_inputs, **kwargs)
    prepare_fused = partial(post_op, *post_inputs, **kwargs)
    coefficients = prepare_reference()
    mix_reference = partial(mix_op.forward_native, streams, branch, *coefficients)
    mix_fused = partial(mix_op, streams, branch, *coefficients)
    post_kernel, mix_kernel = "_mhc_post_residual_kernel", "_mhc_mix_kernel"
    cases = {
        "post_residual": (prepare_reference, prepare_fused, (post_kernel,)),
        "mix": (mix_reference, mix_fused, (mix_kernel,)),
        "pipeline": (
            lambda: mix_op.forward_native(streams, branch, *prepare_reference()),
            lambda: mix_op(streams, branch, *prepare_fused()),
            (post_kernel, mix_kernel),
        ),
    }
    rtol = {torch.bfloat16: 1.6e-2, torch.float16: 2e-3, torch.float32: 1e-5}[dtype]
    for name, (reference, fused, kernels) in cases.items():
        if args.op != "all" and args.op != name:
            continue
        # BF16 MUSA custom ops can quantize values near zero to the adjacent
        # representable value; retain the strict relative gate while allowing
        # one BF16 ulp in the absolute comparison.
        atol = (
            (1e-2 if dtype == torch.bfloat16 else 1e-6)
            if name == "post_residual"
            else (1e-2 if dtype == torch.bfloat16 else 1e-5)
        )
        parity = check_pair(torch, reference, fused, mhc, kernels, rtol=rtol, atol=atol)
        timing = measure_pair(torch, reference, fused, args)
        # Logical minimum traffic: input reads plus output writes. Native
        # intermediates and repeat parameter loads can add actual DRAM traffic.
        element_bytes = streams.element_size()
        prepare_bytes = tokens * 20 * (4 + element_bytes) + 22 * 4
        mix_bytes = tokens * (9 * args.hidden_size + 20) * element_bytes
        minimum_bytes = {"post_residual": prepare_bytes, "mix": mix_bytes, "pipeline": prepare_bytes + mix_bytes}[name]
        # Sigmoid/exp costs and max reductions are not represented as FLOPs.
        prepare_flops = tokens * (72 + 64 * args.sinkhorn_iterations)
        mix_flops = tokens * args.hidden_size * 36
        scalar_flops = {"post_residual": prepare_flops, "mix": mix_flops, "pipeline": prepare_flops + mix_flops}[name]
        emit(
            status="pass",
            op=name,
            tokens=tokens,
            hidden_size=args.hidden_size,
            num_streams=4,
            dtype=args.dtype,
            logits_dtype="float32",
            packed_shape=list(packed.shape),
            post_stride=list(post_inputs[0].stride()),
            residual_stride=list(post_inputs[1].stride()),
            scale=scale,
            sinkhorn_iterations=args.sinkhorn_iterations,
            epsilon=1e-12,
            minimum_io_bytes=minimum_bytes,
            scalar_flops_excluding_exp_and_max=scalar_flops,
            effective_fused_gbps=minimum_bytes / timing["fused"]["p50_us"] / 1000,
            **parity,
            **timing,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--op", choices=["all", "post_residual", "mix", "pipeline"], default="all")
    args = parser.parse_args()
    if args.hidden_size <= 0 or not 0 <= args.sinkhorn_iterations <= 64:
        parser.error("hidden-size must be positive; sinkhorn-iterations must be in [0, 64]")
    torch = load_musa(args, parser)
    if torch is None:
        return
    cuda = getattr(torch, "cuda")
    from vllm_omni.diffusion.layers import mhc

    report_environment(torch, args, mhc, "MAGI2_USE_FUSED_MHC")
    with torch.inference_mode():
        for tokens in args.tokens:
            benchmark_shape(torch, mhc, args, tokens)
            cuda.empty_cache()


if __name__ == "__main__":
    main()
