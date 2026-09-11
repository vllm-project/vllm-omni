# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare native and fused MAGI-2 SwiGLU7 operators on CUDA or MUSA.

Run from the checkout root in an environment with PyTorch and the selected
backend dependencies.  torchada is loaded automatically only on MUSA::

    python -m benchmarks.kernels.benchmark_magi2_swiglu7 \
        --tokens 1 4 16 64 129 4096 --intermediate-size 1280 --num-tests 30

The released MAGI-2 config supplies the 1280 intermediate width. Token counts
are a synthetic sweep; pass per-rank sizes from the target trace for production
comparisons. Input is contiguous interleaved gate/up [tokens, 2 * intermediate].
Both paths evaluate FP32 arithmetic with one cast to the requested output dtype.
This measures eager operator latency, not compiled serving performance.
No CUDA/MUSA device (or no PyTorch) produces a JSON skip record and exit status zero.
"""

import argparse
from functools import partial

from benchmarks.kernels._magi2_musa_bench import (
    add_arguments,
    check_pair,
    emit,
    load_musa,
    measure_pair,
    report_environment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    parser.add_argument("--intermediate-size", type=int, default=1280)
    parser.add_argument("--out-dtype", choices=["bfloat16", "float16", "float32"], default=None)
    args = parser.parse_args()
    if args.intermediate_size <= 0:
        parser.error("intermediate-size must be positive")
    torch = load_musa(args, parser)
    if torch is None:
        return
    cuda = getattr(torch, "cuda")
    from vllm_omni.diffusion.layers import swiglu7

    report_environment(torch, args, swiglu7, "MAGI2_USE_FUSED_SWIGLU7")
    dtype = getattr(torch, args.dtype)
    out_dtype = getattr(torch, args.out_dtype or args.dtype)
    device = torch.device("cuda")
    rtol = {torch.bfloat16: 1.6e-2, torch.float16: 2e-3, torch.float32: 1e-5}[out_dtype]
    atol = 1e-5 if out_dtype == torch.float32 else 2e-3
    op = swiglu7.SwiGLU7()
    with torch.inference_mode():
        for tokens in args.tokens:
            # Values outside both clamp boundaries exercise the released formula.
            x = (torch.randn(tokens, 2 * args.intermediate_size, device=device) * 10).to(dtype)
            reference = partial(op.forward_native, x, out_dtype=out_dtype)
            fused = partial(op, x, out_dtype=out_dtype)
            parity = check_pair(torch, reference, fused, swiglu7, ("_swiglu7_kernel",), rtol=rtol, atol=atol)
            timing = measure_pair(torch, reference, fused, args)
            output_elements = tokens * args.intermediate_size
            minimum_io_bytes = (
                x.numel() * x.element_size() + output_elements * torch.empty((), dtype=out_dtype).element_size()
            )
            emit(
                status="pass",
                op="swiglu7",
                input_shape=list(x.shape),
                input_stride=list(x.stride()),
                dtype=args.dtype,
                out_dtype=args.out_dtype or args.dtype,
                alpha=1.702,
                limit=7.0,
                minimum_io_bytes=minimum_io_bytes,
                # Six scalar arithmetic operations and one exp per output;
                # clamps and transcendental costs are not modeled as FLOPs.
                scalar_flops_excluding_exp_and_clamps=6 * output_elements,
                effective_fused_gbps=minimum_io_bytes / timing["fused"]["p50_us"] / 1000,
                **parity,
                **timing,
            )
            del x, reference, fused
            cuda.empty_cache()


if __name__ == "__main__":
    main()
