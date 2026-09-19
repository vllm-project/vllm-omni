# MammothModa2 AR prefix-cache benchmark

This benchmark compares three cache states for the MammothModa2 AR stage:

- **A**: automatic prefix caching disabled.
- **B1**: caching enabled, but reset after warmup and before every measured
  request. Every sample must be a verified cache miss.
- **B2**: caching enabled and warmed with the target request. Every measured
  sample must be a verified cache hit.

The harness records cache accounting, TTFT, end-to-end AR latency, generated
token throughput, device-wide peak GPU memory, and tensor-accounted CPU prefix
cache memory. It uses the AR-only deploy profiles and does not measure DiT
execution.

## Prerequisites

- One isolated NVIDIA GPU with enough memory for MammothModa2-Preview.
- The model available locally or from Hugging Face.
- `pynvml`, PyTorch, vLLM, and vLLM-Omni installed.
- No other process using the measured GPU because memory is sampled
  device-wide through NVML.

## Reproduce A/B1/B2

Run each scenario in a separate process. The commands below use the 5,627-token
synthetic stress prompt from the reported result:

```bash
python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario a \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 \
  --iterations 10 \
  --output /tmp/mammoth_prefix_a.json

python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario b1 \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 \
  --iterations 10 \
  --output /tmp/mammoth_prefix_b1.json

python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario b2 \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 \
  --iterations 10 \
  --output /tmp/mammoth_prefix_b2.json
```

The harness fails rather than reporting a mislabeled scenario when cache
accounting does not satisfy these contracts:

- A: `cached_tokens == 0` and `cache_creation_tokens == 0`.
- B1: `cached_tokens == 0` and `cache_creation_tokens > 0`.
- B2: cached and newly created tokens match the prompt length and configured
  block size. In particular, an exact block boundary legitimately recomputes
  the final block needed to produce logits (for example, a 32-token prompt with
  block size 16 reports 16 cached and 16 cache-creation tokens).

For B1, the model and kernels are warmed first. The scheduler is then paused,
its prefix cache is cleared, and it is resumed before each timed request. Cache
reset time is excluded from request latency.

The `prefix_cache_cpu_memory` result is read directly from the AR worker. It
reports the allocated hidden-state and multimodal cache tensors, transient
pending-write tensors, and the pinned-memory subset in bytes. Scenario A
reports the cache as disabled; B1 and B2 allocate the same static CPU cache
regardless of how many blocks are currently populated. This is distinct from
`peak_gpu_memory_mib`, which is sampled device-wide through NVML.

## Diagnostic profiler

Profiler traces are diagnostic artifacts and must not be used as authoritative
latency measurements. Collect one request after warmup:

```bash
python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario a \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 80 \
  --iterations 1 \
  --profile-dir /tmp/mammoth_profile_a \
  --output /tmp/mammoth_profile_a.json

python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario b2 \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 80 \
  --iterations 1 \
  --profile-dir /tmp/mammoth_profile_b2 \
  --output /tmp/mammoth_profile_b2.json
```

Analyze the exported rank-0 traces:

```bash
python .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  /tmp/mammoth_profile_a/**/trace_rank0.json \
  --min-gap-ms 1 \
  --topn 20

python .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  /tmp/mammoth_profile_b2/**/trace_rank0.json \
  --min-gap-ms 1 \
  --topn 20
```

## Reported environment

- GPU: NVIDIA A800-SXM4-80GB
- vLLM: 0.28.0
- PyTorch: 2.13.0+cu129
- Precision: BF16
- Execution: eager
- Image grid: 256 x 256
- Prompt tokens: 5,627
- Generated tokens: 273
- Warmup: one request
- Measurements: 10 requests per scenario

## Result summary

For B1, all 10 requests reported zero cached tokens and 5,616 cache-creation
tokens. For B2, all 10 requests reported 5,616 cached tokens and zero
cache-creation tokens.

- A mean TTFT: 383.02 ms; mean AR latency: 5,986.45 ms.
- B1 mean TTFT: 392.92 ms; mean AR latency: 5,997.09 ms.
- B2 mean TTFT: 60.72 ms; mean AR latency: 5,634.94 ms.
- B2 versus B1: TTFT decreased by 84.5%, AR latency decreased by 6.0%,
  and throughput increased by 6.4%.
- Peak device memory was 40,408.8 MiB for A and 40,410.8 MiB for B1/B2.
- The B1/B2 CPU hidden-state cache allocation was 1,062,813,696 bytes
  (1,013.58 MiB), all pinned; pending-write and multimodal-cache bytes were
  zero after warmup.

The shorter 587-token profiler workload reduced scheduled prefill from 587
tokens to 11. Prefill CUDA time decreased from 42.11 ms to 31.09 ms, while
total GEMM and GPU busy time remained nearly unchanged because autoregressive
decode was unaffected. D2H transfer volume increased by about 1.94 MiB and
transfer time increased by about 1.4 ms.

## Full AR to DiT attribution

`benchmark_ar2dit_e2e.py` uses the two-stage deploy configs and records
unprofiled client E2E, AR TTFT/decode/total, AR-to-DiT reconstruction, handoff
serialization/submission, DiT total, and unattributed orchestration time. It
also reports the exact float32 hidden-state tensor payload size; transport RX
fields remain zero for the current same-host in-process handoff.

```bash
python benchmarks/mammoth_moda2/benchmark_ar2dit_e2e.py \
  --scenario a \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 \
  --iterations 10 \
  --output /tmp/mammoth_full_a.json

python benchmarks/mammoth_moda2/benchmark_ar2dit_e2e.py \
  --scenario b2 \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 \
  --iterations 10 \
  --output /tmp/mammoth_full_b2.json
```

Collect stage-isolated diagnostic traces in separate runs. Profiler timings
must not be mixed with the authoritative unprofiled measurements:

```bash
python benchmarks/mammoth_moda2/benchmark_ar2dit_e2e.py \
  --scenario b2 --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 --iterations 1 --profile-stage 0 \
  --profile-dir /tmp/mammoth_full_stage0 \
  --output /tmp/mammoth_full_stage0.json

python benchmarks/mammoth_moda2/benchmark_ar2dit_e2e.py \
  --scenario b2 --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 --iterations 1 --profile-stage 1 \
  --profile-dir /tmp/mammoth_full_stage1 \
  --output /tmp/mammoth_full_stage1.json
```

Analyze each exported `trace_rank0.json` with `trace_analyzer.py` as shown
above, then gzip and checksum the raw artifacts before upload:

```bash
python .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  /tmp/mammoth_full_stage0/*/trace_rank0.json \
  --min-gap-ms 1 --topn 20

python .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  /tmp/mammoth_full_stage1/*/trace_rank0.json \
  --min-gap-ms 1 --topn 20

gzip -9 /tmp/mammoth_full_stage0/*/trace_rank0.json
gzip -9 /tmp/mammoth_full_stage1/*/trace_rank0.json

sha256sum /tmp/mammoth_full_stage0/*/trace_rank0.json.gz
sha256sum /tmp/mammoth_full_stage1/*/trace_rank0.json.gz
```

The generated benchmark JSON files contain the raw per-request samples and
environment metadata. Stage 0 contains AR prefill/decode and the latent D2H
copy; stage 1 contains conditioning H2D, DiT denoising, and VAE decode. The CPU-side
`full_hidden_states.float().contiguous()` reconstruction is measured by
`ar2dit_reconstruction_ms`, outside the GPU trace windows.

For the same 5,627-token, 50-step workload used above, the 10-request
unprofiled attribution was:

- A: E2E 20,948.66 ± 91.07 ms; AR 5,965.55 ± 81.73 ms; handoff
  108.69 ± 4.15 ms; DiT 14,859.15 ± 15.21 ms.
- B2: E2E 20,804.66 ± 40.45 ms; AR 5,754.00 ± 37.61 ms; handoff
  99.29 ± 17.71 ms; DiT 14,934.20 ± 19.76 ms.
- The float32 AR-to-DiT hidden-state payload was 84,568,064 bytes.

The stage-isolated traces are diagnostic. Stage 0 showed 3.137 s GPU busy in
a 9.066 s span; pinned D2H copies totaled 3.202 ms. Stage 1 showed 13.894 s
GPU busy in a 16.389 s span; pageable H2D copies totaled 8.606 ms. Attention
and GEMM dominate the DiT path, while gaps of at least 1 ms totaled 0.126 s;
bulk conditioning transfer is negligible.

## Image-quality comparison

`benchmark_image_quality.py` compares cache-disabled A with warm-hit B2 for
three prompts and three seeds. It requires request-local DiT seed propagation;
the reported run used the independent `fix/mammothmoda2-dit-seed` branch on
top of this branch. Metric calculation also requires `numpy`, `scikit-image`,
and `transformers`. Run each generation scenario in a separate process, then
compute pixel and CLIP metrics:

```bash
python benchmarks/mammoth_moda2/benchmark_image_quality.py generate \
  --scenario a --model /root/models/MammothModa2-Preview \
  --output-dir /tmp/mammoth_quality

python benchmarks/mammoth_moda2/benchmark_image_quality.py generate \
  --scenario b2 --model /root/models/MammothModa2-Preview \
  --output-dir /tmp/mammoth_quality

python benchmarks/mammoth_moda2/benchmark_image_quality.py compare \
  --output-dir /tmp/mammoth_quality \
  --output /tmp/mammoth_quality_metrics.json
```

Across the nine matched pairs, mean image-image CLIP cosine similarity was
0.9800 ± 0.0152, SSIM was 0.9121 ± 0.0960, and PSNR was
28.28 ± 7.08 dB. Mean text-image CLIP cosine was 0.2214 for A and 0.2242
for B2 (B2 minus A: +0.0028).
