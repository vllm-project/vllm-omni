# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure SnakeBeta copy removal on real UpSample1d views (no model weights).

    python benchmarks/kernels/benchmark_snake_activation.py --dtype float32
    python benchmarks/kernels/benchmark_snake_activation.py --dtype bfloat16 --include-compiled

The copy baseline explicitly materializes the input as the old wrapper did.
Both paths use the same kernel arithmetic. The alias_free scope runs the real
upsample/activation/downsample module, not a full pretrained speech decoder.
Times are CUDA Graph replay measurements; compilation/capture are excluded.
"""

import argparse
import json
import statistics

import torch
from vllm.triton_utils import triton

from vllm_omni.model_executor.models.common.alias_free_activation import AliasFreeActivation1d
from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--include-compiled", action="store_true")
    args = parser.parse_args()
    if min(args.batch, args.channels, args.rounds, args.rep_ms, *args.lengths) <= 0:
        parser.error("shapes, rounds, and rep-ms must be positive")
    return args


def extra_peak_bytes(fn):
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    before = torch.accelerator.memory_allocated()
    result = fn()
    torch.accelerator.synchronize()
    peak = torch.accelerator.max_memory_allocated() - before
    del result
    return peak


@torch.inference_mode()
def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    print(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__, **vars(args)}))
    torch.manual_seed(42)
    for length in args.lengths:
        module = AliasFreeActivation1d(SnakeBeta(args.channels)).to(device="cuda", dtype=dtype).eval()
        x = torch.randn(args.batch, args.channels, length, device="cuda", dtype=dtype)
        view = module.upsample(x)
        assert view.stride(-1) == 1 and not view.is_contiguous()
        assert module.act._init_triton()
        module.act.precompute_exp_cache()

        def activation_copy():
            return module.act._triton_forward(view.contiguous())

        def activation_view():
            return module.act._triton_forward(view)

        def alias_copy():
            return module.downsample(module.act._triton_forward(module.upsample(x).contiguous()))

        def alias_view():
            return module(x)

        scopes = {
            "activation": {"copy": activation_copy, "view": activation_view},
            "alias_free": {"copy": alias_copy, "view": alias_view},
        }
        if args.include_compiled:
            compiled = torch.compile(module.act._eager_forward, fullgraph=True)
            scopes["activation"]["compiled_native"] = lambda: compiled(view)

        for scope, implementations in scopes.items():
            reference = implementations["copy"]()
            torch.testing.assert_close(implementations["view"](), reference, atol=0, rtol=0)
            if "compiled_native" in implementations:
                tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}[dtype]
                torch.testing.assert_close(
                    implementations["compiled_native"](), reference, atol=tolerance, rtol=tolerance
                )
            del reference
            # Prewarm all paths, including optional compilation, before timing.
            for fn in implementations.values():
                for _ in range(5):
                    fn()
            assert module.act._triton_kernel is not False
            samples = {name: [] for name in implementations}
            for round_index in range(args.rounds):
                order = list(implementations)
                if round_index % 2:
                    order.reverse()
                for name in order:
                    samples[name].append(
                        triton.testing.do_bench_cudagraph(implementations[name], rep=args.rep_ms, return_mode="median")
                        * 1000
                    )
            medians = {name: statistics.median(values) for name, values in samples.items()}
            print(
                json.dumps(
                    {
                        "scope": scope,
                        "input_shape": list(x.shape),
                        "activation_shape": list(view.shape),
                        "activation_stride": list(view.stride()),
                        "round_us": samples,
                        "median_us": medians,
                        "copy_over_view": medians["copy"] / medians["view"],
                        "extra_peak_bytes": {name: extra_peak_bytes(fn) for name, fn in implementations.items()},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
