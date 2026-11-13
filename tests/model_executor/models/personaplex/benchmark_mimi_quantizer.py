# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare full and 8-codebook Mimi RVQ on CUDA without model checkpoints.

Run: python tests/model_executor/models/personaplex/benchmark_mimi_quantizer.py

This measures the HF quantizer on random centroids and synthetic latents, not
full codec/LM latency, audio quality, or serving concurrency. Timings include
host launches and synchronized CUDA execution. No CPU fallback is allowed.
"""

import hashlib
import inspect
import json
import statistics
from datetime import datetime, timezone

import torch
import transformers
from torch.utils.benchmark import Timer
from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiSplitResidualVectorQuantizer


def main() -> None:
    # Explicit device allocation below fails rather than falling back to CPU.
    torch.set_num_threads(1)
    torch.manual_seed(314159)
    config = MimiConfig()
    assert config.num_quantizers == 32 and config.num_semantic_quantizers == 1
    quantizer = MimiSplitResidualVectorQuantizer(config).eval().to(device="cuda:0")
    for rvq in (quantizer.semantic_residual_vector_quantizer, quantizer.acoustic_residual_vector_quantizer):
        for layer in rvq.layers:
            layer.codebook.embed_sum.normal_()
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "device": str(next(quantizer.parameters()).device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "dtype": "float32",
        "scope": "HF Mimi split RVQ only; random centroids and synthetic latents; no pretrained weights or serving",
        "config": {
            key: getattr(config, key)
            for key in (
                "hidden_size",
                "vector_quantization_hidden_dimension",
                "codebook_dim",
                "codebook_size",
                "num_quantizers",
                "num_semantic_quantizers",
            )
        },
        "source_sha256": hashlib.sha256(inspect.getsource(MimiSplitResidualVectorQuantizer).encode()).hexdigest(),
        "warmup_calls_per_path": 16,
        "rounds": 6,
        "calls_per_round": 16,
        "timing": "torch.utils.benchmark.Timer; synchronizes accelerators and adds its own warmup",
    }
    batches = []
    for batch in (1, 8, 32):
        inputs = [torch.randn(batch, config.hidden_size, 1, device="cuda") for _ in range(32)]
        for x in inputs:
            full = quantizer.encode(x)
            prefix = quantizer.encode(x, num_quantizers=8)
            assert torch.equal(full[:8], prefix), "8-codebook prefix differs from full RVQ"
        for _ in range(16):
            quantizer.encode(inputs[0])
            quantizer.encode(inputs[0], num_quantizers=8)
        timers = {
            count: Timer(
                stmt="for x in inputs: quantizer.encode(x, num_quantizers=count)",
                globals={"quantizer": quantizer, "inputs": inputs[:16], "count": count},
                num_threads=1,
            )
            for count in (32, 8)
        }
        timings: dict[int, list[float]] = {32: [], 8: []}
        for repeat in range(6):
            for count in (32, 8) if repeat % 2 == 0 else (8, 32):
                timings[count].append(timers[count].timeit(number=1).mean * 1000 / 16)
        full_ms = statistics.median(timings[32])
        prefix_ms = statistics.median(timings[8])
        row = {
            "batch": batch,
            "frames_per_call": 1,
            "parity_cases": len(inputs),
            "prefix_bit_identical": True,
            "full_ms": timings[32],
            "prefix_ms": timings[8],
            "full_median_ms": full_ms,
            "prefix_median_ms": prefix_ms,
            "module_speedup": full_ms / prefix_ms,
        }
        batches.append(row)
        print("BATCH " + json.dumps(row), flush=True)
    result["batches"] = batches
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
