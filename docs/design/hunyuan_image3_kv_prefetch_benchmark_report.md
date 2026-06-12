# HunyuanImage-3.0 KV Async Prefetch Benchmark Report

## Test Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX Pro 6000 Blackwell Server Edition (sm_120) × 4 |
| Model | HunyuanImage-3.0 (fp8 quantization) |
| Deploy config | `hunyuan_image_3_moe_async_kv.yaml` (SharedMemoryConnector) |
| Pipeline | Stage 0: AR (GPU 0,1, TP=2) → Stage 1: DiT (GPU 2,3, TP=2, EP) |
| Denoising steps | 50 |
| Guidance scale | 5.0 |
| Seed | 42 |
| Benchmark script | `examples/offline_inference/hunyuan_image3/kv_prefetch_bench.py` |

## Test Methodology

Two submission modes are compared:

- **Sequential mode** (`--mode sequential`): Requests are submitted one at a time. When the DiT scheduler builds the `prefetch_stub`, no next request exists in the waiting queue yet. Every request shows MISS — this serves as the sync baseline.

- **Pipeline mode** (`--mode pipeline`): All requests are submitted to the orchestrator at once. While DiT denoises request N (~25s), the AR stage has already finished request N+1 and enqueued it to the DiT waiting queue. The scheduler sees the next request and populates `prefetch_stub`, so `start_load_kv()` fires during request N's step 0 — subsequent requests should show HIT.

```bash
# Sequential (sync baseline, all MISS)
python -m examples.offline_inference.hunyuan_image3.kv_prefetch_bench \
    --model /root/autodl-tmp/HunyuanImage-3.0/ \
    --quantization fp8 --steps 50 --num-requests 4 --mode sequential

# Pipeline (prefetch enabled, HIT on req 2+)
python -m examples.offline_inference.hunyuan_image3.kv_prefetch_bench \
    --model /root/autodl-tmp/HunyuanImage-3.0/ \
    --quantization fp8 --steps 50 --num-requests 4 --mode pipeline
```

## Results

### 1. KV Prefetch Hit/Miss Status

| Request | Sequential | Pipeline |
|---------|-----------|----------|
| Warmup-0 | MISS | MISS |
| Req-1 | MISS | MISS |
| Req-2 | MISS | **HIT** |
| Req-3 | MISS | **HIT** |
| Req-4 | MISS | **HIT** |

- Sequential: all MISS — no next request exists when the scheduler builds the prefetch stub.
- Pipeline: Req-1 is MISS due to an architectural limitation (see Section 4), Req-2+ all HIT.

### 2. KV Receive Latency

| Request | Sequential (sync_recv) | Pipeline (apply / sync_recv) | Speedup |
|---------|----------------------|----------------------------|---------|
| Warmup-0 | 472.5ms / 516.3ms | 364.5ms / 504.3ms | ~1.0× |
| Req-1 | 313.3ms / 496.9ms | 167.8ms / 169.9ms | ~1.9× |
| Req-2 | 317.3ms / 338.1ms | **36.7ms** (HIT apply) | **9.2×** |
| Req-3 | 343.6ms / 358.0ms | **18.7ms / 41.2ms** (HIT apply) | **8.7×** |
| Req-4* | 2124.0ms / 2464.3ms | **33.8ms / 34.1ms** (HIT apply) | **72.4×** |

> *Req-4 in sequential mode is an outlier — see Section 5.

**Key finding:** HIT requests have KV apply latency of 18–41ms, which is **4–12× faster** than the sync_recv latency of 167–500ms.

### 3. End-to-End Latency (Sequential mode only)

| Request | e2e Time | KV recv % of e2e |
|---------|----------|------------------|
| Warmup-0 | 30.91s | ~1.5% |
| Req-1 | 27.89s | ~1.1% |
| Req-2 | 28.35s | ~1.1% |
| Req-3 | 27.09s | ~1.3% |
| Req-4* | 114.80s | ~1.8% |

Pipeline mode does not have per-request e2e measurements because requests overlap in execution.

### 4. Req-1 MISS in Pipeline Mode: Root Cause

In HunyuanImage-3.0's request mode (`step_execution=False`), the engine loop alternates between `schedule()` and `execute_model()`:

```
schedule() → execute_model() (50 steps, ~25s) → schedule() → execute_model() ...
```

When Req-0 is executing, Req-1 enters the waiting queue (AR has already finished it), but `schedule()` is **not called** during `execute_model()`. The prefetch stub is only built inside `schedule()`, so Req-1 never gets a prefetch opportunity while Req-0 runs. After Req-0 completes, the next `schedule()` immediately moves Req-1 from waiting→running without ever building a prefetch stub for it.

This is an architectural limitation of request mode, not a bug. Potential fixes:
1. Trigger prefetch in `add_request()` when a request enters the waiting queue
2. Switch to step mode (where `schedule()` is called between denoising steps)

### 5. Anomaly: Req-4 AR Output Explosion

| Metric | Req-1/2/3 (normal) | Req-4 (anomaly) |
|--------|-------------------|-----------------|
| AR generated tokens | 99–179 | **8192** (max_tokens) |
| KV size | 87.6–92.8 MB | **617.9 MB** |
| sync_recv latency | 313–343 ms | **2124–2464 ms** |
| e2e latency | 27–28 s | **114.80 s** |

Req-4's AR stage generated 8192 tokens (hitting the max_tokens limit), causing KV size to balloon 6–7× and sync_recv latency to increase 6–7×. This is a model behavior issue (AR failed to stop at the expected token), unrelated to the KV prefetch optimization. Notably, in pipeline mode with prefetch HIT, the same request's KV apply latency is only 33.8ms — demonstrating that prefetch effectively hides even large KV transfer delays.

## Summary

1. **Prefetch is effective**: In pipeline mode, Req-2+ all show HIT. KV receive latency drops from 300–500ms to 18–41ms — a **4–12× speedup**.

2. **Req-1 MISS is an architectural limitation**: Request mode's serial schedule→execute loop means `schedule()` never fires during `execute_model()`, so Req-1 entering the waiting queue during Req-0's denoising never triggers a prefetch stub build.

3. **Prefetch contribution to e2e**: For normal requests (~90MB KV), KV sync_recv accounts for ~1.5% of e2e (300ms / 28s). The absolute e2e gain is modest (~300ms saved per request). For large-KV scenarios (e.g., the 617MB outlier), the benefit is more significant (~2.1s saved).

4. **AR output explosion** (Req-4 hitting max_tokens) is a separate issue that warrants independent investigation.
