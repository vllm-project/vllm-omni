# Pi0.5 S0 benchmark and results

This follows the repository's Pi0.5 benchmark layout. The scripts are limited to the four-layer S0 measurements and are kept separate from generated run output.

- [`bench_pi05.py`](bench_pi05.py): one entry point for model-only, in-process pipeline, `uni`/`mp` engine latency, request-level attribution, and OpenPI websocket E2E.
- Raw request records, aggregate JSON, checksums, and the historical instrumentation patch are retained in the local experiment bundle and are not included in this PR.

## Reproduce the measurements

The reported numbers were collected at source commit `1c11eef1a5a093f064562c47c9cae8b0b5dfbd49` with the instrumentation patch retained in the local experiment bundle. The checked-out branch is newer; historical measurements are not claimed as measurements of the branch head.

Use `vllm_omni/deploy/pi05.yaml` as the base config. For BF16, create a local config outside the repository:

```bash
sed 's/^dtype: float32$/dtype: bfloat16/' vllm_omni/deploy/pi05.yaml > /tmp/pi05-bf16.yaml
```

Set the matching attention options and run from the repository root. Repeat each command three times, changing the output names for each run:

```bash
export ATTENTION_BACKEND=torch
export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_DISABLE_COMPILE_CACHE=1

python benchmarks/pi05/bench_pi05.py --mode uni \
  --deploy-config /tmp/pi05-bf16.yaml --views 1 2 3 --steps 10 \
  --warmup 3 --iters 30 --out /tmp/pi05-uni.json --raw-out /tmp/pi05-uni.jsonl
python benchmarks/pi05/bench_pi05.py --mode mp \
  --deploy-config /tmp/pi05-bf16.yaml --views 1 2 3 --steps 10 \
  --warmup 3 --iters 30 --out /tmp/pi05-mp.json --raw-out /tmp/pi05-mp.jsonl
```

For the OpenPI E2E layer, start the server with the matching dtype config and run the command three times for three 30-request rounds. For the supplementary FP32 runs, use the checked-in default `vllm_omni/deploy/pi05.yaml` instead of `/tmp/pi05-bf16.yaml`.

```bash
vllm serve lerobot/pi05_base --omni --host 0.0.0.0 --port 8100 \
  --deploy-config /tmp/pi05-bf16.yaml --enforce-eager --disable-log-stats
python benchmarks/pi05/bench_pi05.py --mode e2e --host 127.0.0.1 --port 8100 --requests 30 --steps 10
```

For the exact historical instrumentation patch, retrieve `per-request-timing.patch` from the local experiment bundle and apply it to the recorded source revision. Do not use the historical measurements as a before/after comparison against the current branch.

## Recorded results

# Pi0.5 S0 BS=1 Profiling Results: RTX 4090 and RTX PRO 6000

**Status: measurement matrix complete; S0 acceptance is pending the BF16 correctness decision.**

This report records the S0 result for the fixed source revision and instrumentation patch below. It documents a negative optimization result: the tested inline/single-stage direction did not show a reproducible benefit, so no extra scheduler or buffer-reuse complexity is proposed here.

## Reproduction identity

- vLLM-Omni source: `1c11eef1a5a093f064562c47c9cae8b0b5dfbd49`
- Per-request instrumentation patch: `6d633fa7be71de0b87cbade84bcfaf4d569eecdf5bf809aad6b04aab12f14bc1`
- Workload: BS=1, action horizon 50, 10 flow steps, seed 42, 200-token padded language input, 1/2/3 image views
- Precision: BF16 as the primary issue workload; FP32 as a supplementary reference
- Engine matrix per dtype: `uni` and production `mp`, each view count, 3 independent runs, each run with 3 warmups and 30 warm requests
- OpenPI E2E: 3 rounds × 30 requests per view and dtype; first cold request reported separately from warm percentiles
- Backends: `ATTENTION_BACKEND=torch`, `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`, eager execution
- Environment on Pro6000: Python 3.12.3, PyTorch 2.13.0+cu130, vLLM/vLLM-Omni 0.32.0, Transformers 5.16.1, tokenizers 0.23.2, Triton 3.7.1

## Pro6000 latency results

Warm p50 in ms; the value is the median of three independent runs.

### BF16

| Views | Model-only | In-process pipeline | `uni` engine | Production `mp` engine | OpenPI E2E p50 / p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 376.7 | 340.3 | 357.6 | 333.8 | 334.3 / 340.4 |
| 2 | 378.8 | 343.3 | 356.8 | 334.6 | 334.7 / 337.4 |
| 3 | 376.8 | 342.2 | 356.0 | 331.6 | 333.9 / 337.4 |

### FP32 supplementary results

| Views | Model-only | In-process pipeline | `uni` engine | Production `mp` engine | OpenPI E2E p50 / p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 411.2 | 375.8 | 388.7 | 368.2 | 371.5 / 376.2 |
| 2 | 403.4 | 382.1 | 390.0 | 368.8 | 370.9 / 373.7 |
| 3 | 412.0 | 379.6 | 392.6 | 367.3 | 371.5 / 374.1 |

### Request-level engine segments (Pro6000, production `mp`, one view)

| Segment | BF16 (ms) | FP32 (ms) |
|---|---:|---:|
| orchestrator setup | 0.029 | 0.030 |
| queue wait | 0.394 | 0.369 |
| stage submission | 0.600 | 0.594 |
| scheduler wait | 0.233 | 0.239 |
| executor run | 331.900 | 365.198 |
| output routing | 0.074 | 0.073 |
| response handling | 0.380 | 0.368 |
| stage 0 generation total | 332.143 | 365.435 |

Non-executor engine work is about 1.5–1.7 ms. No stable queue/scheduler/routing segment above 10 ms was observed. `mp` was faster than `uni` in this workload; this does not support landing an inline/direct path or additional allocation-reuse complexity for S0.

## Cross-GPU comparison

The same frozen source, patch, workload, and matrix were used on a single RTX 4090 and GPU0 of a 4× RTX PRO 6000 Blackwell Server Edition system. The comparison is descriptive only; GPU-specific latency is not interpreted as an S0 before/after optimization result.

| Dtype | Views | Pro6000 `mp` p50 (ms) | 4090 `mp` p50 (ms) | Pro6000 E2E p50 / p95 (ms) | 4090 E2E p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| BF16 | 1 | 333.8 | 150.5 | 334.3 / 340.4 | 152.1 / 154.3 |
| BF16 | 2 | 334.6 | 151.3 | 334.7 / 337.4 | 152.4 / 155.5 |
| BF16 | 3 | 331.6 | 150.5 | 333.9 / 337.4 | 152.6 / 155.0 |
| FP32 | 1 | 368.2 | 215.7 | 371.5 / 376.2 | 216.1 / 227.0 |
| FP32 | 2 | 368.8 | 215.3 | 370.9 / 373.7 | 216.9 / 228.8 |
| FP32 | 3 | 367.3 | 216.0 | 371.5 / 374.1 | 218.0 / 229.8 |

On this eager Torch SDPA path, Pro6000 was approximately 2.2× slower than 4090 for BF16 and 1.7× slower for FP32. Pro6000 monitoring showed approximately 2415 MHz SM clock, 139–154 W power, about 13% utilization, and no observed thermal throttling. This cross-card observation should be investigated as a backend/architecture path difference; it is not evidence of engine overhead or an S0 optimization gain.

## Correctness status and open decisions

- FP32 parity against LeRobot with identical inputs and fixed noise: `max_abs=0` for 1/2/3 views.
- BF16 full 32-dimension output with `atol=5e-2`:
  - 1 view: `max_abs=3.142e-2` — pass.
  - 2 views: `max_abs=7.625e-2` — **fail**.
  - 3 views: `max_abs=4.081e-2` — pass.
- In the two-view BF16 case, 50/3200 elements exceed `5e-2`, all in action dimension 9. The maximum error over the seven robot-consumed dimensions is `6.687e-3`. Dimension 9 is nonzero and participates in flow matching, so this report does not silently exclude it from the full-output criterion.
- The 4090 auxiliary BF16 result (`2.69e-2`) used a three-camera CPU path; it is not the same two-view workload and does not clear the two-view case.
- Maintainer decision required: retain full-32 `atol=5e-2` and fix the two-view mismatch, or explicitly approve a different documented acceptance policy.

## S0 conclusion

The measurement and request-level attribution matrix is complete on both GPUs at the frozen revision. No tested engine-path change produced a reproducible improvement that justifies landing extra S0 complexity. The optimization direction can be closed without an inline/direct implementation if maintainers accept the negative result. **S0 remains pending until the BF16 correctness policy is resolved.**

## Implementation and evidence

The runnable benchmark harness is in [`benchmarks/pi05/`](.). Raw request records, the historical instrumentation patch, and the aggregate JSON remain in the local experiment bundle and are not part of this PR. The exact historical measurement source remains `1c11eef1a5a093f064562c47c9cae8b0b5dfbd49`; the current branch is rebased on a newer `main` commit, so these numbers are not presented as measurements of the branch head.

The original full bundle is `Pi0.5-Pro6000-Full-20261022-bundle`. This PR includes the final measurement records and selected environment evidence needed to inspect/reaggregate the results; it excludes model weights, virtual environments, caches, and retry/debug logs.
