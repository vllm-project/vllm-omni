# Qwen2.5-Omni embedding microbenchmark

Measure the host-observed latency of `embed_input_ids` before and after the
redundant text embedding removal in PR #7477. Both variants use the same real
checkpoint embedding table, dependencies, inputs and batch shapes. The baseline
wrapper is extracted from the specified Git commit; the fixed wrapper comes from
the current checkout. No full model is initialized.

## Run

Use Linux, a CUDA GPU, the project dependencies, and a local
`Qwen/Qwen2.5-Omni-7B` checkpoint with its safetensors index and weights.
From the repository root:

```bash
git fetch origin 01a2f93256975c7ff9565c5414b0fe0225bc4765
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 python benchmarks/qwen2_5_omni/benchmark_embeddings.py \
  --model /path/to/Qwen2.5-Omni-7B \
  --baseline-ref 01a2f93256975c7ff9565c5414b0fe0225bc4765 \
  --cpu-core 2 --lengths 1172 --warmup 50 --blocks 20 --iterations 20 \
  --output-dir /tmp/qwen25-embedding-results
```

Choose a CPU core in your allowed affinity set. For the original CUDA length
sweep, use `--lengths 512 1172 4096 16384`.

## Method

- BF16 checkpoint embeddings, TP=1, CUDA int32 token IDs, CPU multimodal masks.
- One pinned CPU core, one PyTorch thread, garbage collection disabled.
- 50 warmups per variant, then 20 paired blocks of 20 calls with balanced,
  randomized variant order. Summary: median of block medians.
- Host latency includes synchronization inside the wrapper. Explicit GPU
  synchronization before and after the timed call is excluded from host latency.
  Calling-thread CPU time and synchronized completion time are recorded separately.
- CPU frequency is sampled at block boundaries; clocks are not locked.
- Validation checks one text embedding call for the fixed wrapper, two for the
  multimodal baseline, and `torch.equal` for identical inputs and batch shapes.
  This does not establish batch invariance or full-model output equality.

Interleaved audio/video uses a valid synthetic multimodal mask. This benchmark
measures wrapper latency, not E2E throughput.

## Reference results

Original environment: H100 80GB; Python 3.12.3; vLLM 0.29.0+cu129;
PyTorch 2.13.0+cu129; Transformers 5.14.0; Qwen2.5-Omni-7B BF16.
At 1172 tokens, the September 13 measurements were:

| Input | Before (μs) | After (μs) | Host latency reduction |
| --- | ---: | ---: | ---: |
| Text | 18.81 | 18.66 | 0.80% |
| Audio | 104.32 | 85.30 | 18.23% |
| Image | 104.72 | 86.27 | 17.62% |
| Video | 104.73 | 85.76 | 18.11% |
| Audio + image + video | 567.92 | 549.54 | 3.24% |
| Interleaved audio/video | 622.76 | 597.84 | 4.00% |

Results vary between runs and environments. These local savings did not produce
a consistent E2E throughput improvement in the separate repeated E2E experiment.

## Output

The output directory contains `metadata.json`, `micro-raw.json`,
`validation.json`, and `summary.json`. Raw records include per-call timings,
block timestamps and CPU frequency samples. Timer overhead is measured separately.
