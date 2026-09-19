# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A/B benchmark for CosyVoice3 TensorRT Euler-session host overhead."""

import argparse
import json
import random
import statistics
import time

import torch
from omegaconf import DictConfig

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
    CausalConditionalCFM,
)
from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import (
    build_flow_estimator_trt,
)


def make_case(length):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20260918 + length)
    x = 0.1 * torch.randn((1, 80, length), device="cuda", generator=gen)
    mu = 0.1 * torch.randn((1, 80, length), device="cuda", generator=gen)
    mask = torch.ones((1, 1, length), device="cuda")
    spks = 0.1 * torch.randn((1, 80), device="cuda", generator=gen)
    cond = 0.1 * torch.randn((1, 80, length), device="cuda", generator=gen)
    t_span = torch.linspace(0, 1, 11, device="cuda")
    return x, t_span, mu, mask, spks, cond


def baseline_solve(cfm, x, t_span, mu, mask, spks, cond):
    t, dt = t_span[0].unsqueeze(0), t_span[1] - t_span[0]
    batch_size = int(x.size(0))
    estimator_batch = 2 * batch_size
    estimator_dtype = spks.dtype if spks is not None else x.dtype

    x_in = torch.zeros((estimator_batch, 80, x.size(2)), device=x.device, dtype=estimator_dtype)
    mask_in = torch.zeros((estimator_batch, 1, x.size(2)), device=x.device, dtype=estimator_dtype)
    mu_in = torch.zeros((estimator_batch, 80, x.size(2)), device=x.device, dtype=estimator_dtype)
    t_in = torch.zeros((estimator_batch,), device=x.device, dtype=estimator_dtype)
    spks_in = torch.zeros((estimator_batch, 80), device=x.device, dtype=estimator_dtype)
    cond_in = torch.zeros((estimator_batch, 80, x.size(2)), device=x.device, dtype=estimator_dtype)

    for step in range(1, len(t_span)):
        x_in[:batch_size] = x
        x_in[batch_size:] = x
        mask_in[:batch_size] = mask
        mask_in[batch_size:] = mask
        mu_in[:batch_size] = mu
        t_in[:] = t
        if spks is not None:
            spks_in[:batch_size] = spks
        if cond is not None:
            cond_in[:batch_size] = cond
        dphi_dt = cfm.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [batch_size, batch_size], dim=0)
        dphi_dt = (1.0 + cfm.inference_cfg_rate) * dphi_dt - cfm.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    return x.float()


def measure(fn):
    torch.accelerator.synchronize()
    caller = torch.cuda.current_stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(caller)
    host_start = time.perf_counter_ns()
    out = fn()
    host_ms = (time.perf_counter_ns() - host_start) / 1e6
    end_event.record(caller)
    end_event.synchronize()
    gpu_ms = start_event.elapsed_time(end_event)
    return out, host_ms, gpu_ms


def summary(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_median_ci(values, samples=5000, seed=12345):
    rng = random.Random(seed)
    n = len(values)
    medians = [statistics.median(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    medians.sort()
    return [
        medians[int(0.025 * samples)],
        medians[min(samples - 1, int(0.975 * samples))],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TensorRT benchmark")
    wrapper = build_flow_estimator_trt(args.onnx, device="cuda")
    cfm = CausalConditionalCFM(
        in_channels=80,
        cfm_params=DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        ),
        n_spks=1,
        spk_emb_dim=80,
        estimator=wrapper,
    )
    case = make_case(args.length)

    def baseline_fn():
        return baseline_solve(cfm, *case)

    def optimized_fn():
        return cfm.solve_euler(*case)

    for _ in range(args.warmup):
        baseline_fn()
        optimized_fn()
    torch.accelerator.synchronize()

    baseline_ref = baseline_fn().clone()
    optimized_ref = optimized_fn().clone()
    torch.accelerator.synchronize()

    baseline_host = []
    optimized_host = []
    baseline_gpu = []
    optimized_gpu = []
    host_delta = []
    gpu_delta = []
    host_relative_percent = []
    gpu_relative_percent = []

    for index in range(args.repeats):
        if index % 2 == 0:
            _, bh, bg = measure(baseline_fn)
            _, oh, og = measure(optimized_fn)
        else:
            _, oh, og = measure(optimized_fn)
            _, bh, bg = measure(baseline_fn)
        baseline_host.append(bh)
        optimized_host.append(oh)
        baseline_gpu.append(bg)
        optimized_gpu.append(og)
        host_delta.append(bh - oh)
        gpu_delta.append(bg - og)
        host_relative_percent.append(100.0 * (bh - oh) / bh)
        gpu_relative_percent.append(100.0 * (bg - og) / bg)

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "io_dtype": str(wrapper.io_dtype),
        "length": args.length,
        "steps": 10,
        "repeats": args.repeats,
        "baseline_host_ms": summary(baseline_host),
        "optimized_host_ms": summary(optimized_host),
        "host_delta_ms": summary(host_delta),
        "host_delta_median_95ci": bootstrap_median_ci(host_delta),
        "host_positive_rounds": sum(value > 0 for value in host_delta),
        "host_relative_percent": summary(host_relative_percent),
        "host_relative_median_95ci": bootstrap_median_ci(host_relative_percent),
        "baseline_gpu_ms": summary(baseline_gpu),
        "optimized_gpu_ms": summary(optimized_gpu),
        "gpu_delta_ms": summary(gpu_delta),
        "gpu_delta_median_95ci": bootstrap_median_ci(gpu_delta),
        "gpu_positive_rounds": sum(value > 0 for value in gpu_delta),
        "gpu_relative_percent": summary(gpu_relative_percent),
        "gpu_relative_median_95ci": bootstrap_median_ci(gpu_relative_percent),
        "baseline_nonfinite": int((~torch.isfinite(baseline_ref)).sum().item()),
        "optimized_nonfinite": int((~torch.isfinite(optimized_ref)).sum().item()),
        "exact_output": bool(torch.equal(baseline_ref, optimized_ref)),
        "max_abs_error": float((baseline_ref - optimized_ref).abs().max().item()),
    }
    print(json.dumps(result, indent=2), flush=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
