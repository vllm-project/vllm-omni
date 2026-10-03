# CosyVoice3 B2 payload-only cleanup

This is the intermediate study. The subsequent [completion report](../completion/README.md)
resolves the current-main C=4 parity gap and localizes the historical 0.28
differences to Talker generation before the transport boundary.

This implements the [maintainer's revised B2 scope](https://github.com/vllm-project/vllm-omni/issues/6870#issuecomment-5676714631):
omit unused Talker hidden-state payloads in async-chunk mode and keep output
materialization inline. CosyVoice3 does not opt into async materialization,
and the shared GPU runner has no B2 changes.

The chunk processor declares that it needs sampled-token updates even without
a tensor payload. The scheduler preserves its existing Stage-0-final exclusion
for these updates. Prefill conditioning and the existing packed-inference
payload policy are retained.

## Isolated single-GPU measurements

Measured on 2026-10-03 using one RTX 4090 (49,140 MiB reported), vLLM
0.28.0+cu129, PyTorch 2.13.0+cu129, and the Torch flow backend. These numbers
use the historical pinned baseline and a payload-only backport. They are
separate from the current-main correctness checks below.

Both stages use device 0, seed 0, async chunk, AR async scheduling, disabled
prefix caching, and 25-token chunks. The model revision, inputs, and reference
audio match the earlier study. Each run has eight measured requests after two
full warmup batches at its measured concurrency. Fixed-length runs use
`min_tokens=max_tokens=128`, giving 1,016 measured AR intervals per run.
The original GPU-0 serving process was stopped for the experiment and restored
afterward; GPU 1's services remained running. CPU isolation was not enforced.

| Workload | Paired rounds | Baseline AR ITL (ms/token) | Cleanup AR ITL (ms/token) | Mean change |
| --- | ---: | ---: | ---: | ---: |
| C=1, natural EOS | 1 | 4.912 | 4.889 | -0.48% |
| C=1, fixed 128 tokens | 3 | 4.974 | 4.942 | -0.66% |
| C=4, fixed 128 tokens | 3 | 7.553 | 7.461 | -1.22% |

The fixed C=1 paired deltas were -0.08650, +0.01300, and -0.02475 ms/token.
The approximate paired-t 95% interval is [-0.15753, +0.09203] ms/token.
All 32 C=1 paired waveforms and chunk boundaries matched exactly across the
natural and fixed workloads.

The fixed C=4 paired deltas were -0.03150, -0.098625, and -0.147375 ms/token.
The approximate paired-t 95% interval is [-0.23703, +0.05203] ms/token.
All pairs matched workload and audio lengths, but only 6/24 paired waveforms
matched exactly. Baseline waveforms were identical across these three repeats;
concurrent waveform parity is therefore not established by this experiment.
Flow uses worker-RNG noise (`torch.randn_like(mu)`), so chunk ordering can
affect noise assignment, but codec/noise-order tracing is needed to establish
the cause of these differences. No independent speech-quality claim is made.

| Fixed workload | Mean of run-median TTFA, baseline → cleanup (ms) | Mean last-audio time (ms) | Audio throughput (audio-s/s) |
| --- | ---: | ---: | ---: |
| C=1 | 463.39 → 466.15 | 979.41 → 961.29 | 5.233 → 5.321 |
| C=4 | 1475.52 → 1478.78 | 3095.62 → 3106.66 | 6.609 → 6.585 |

Both AR intervals span zero with only three independent paired rounds.
These observations support a small local cleanup, without establishing a
general speedup or a solution to the RFC's larger bottleneck. There is no
consistent end-to-end benefit across the measured workloads.

All unprofiled runs peaked at approximately 24,760–24,766 MiB. The monitor
saw at most three GPU processes, with no co-resident GPU-0 serving workload.
Startup and shutdown time is excluded from request latency and throughput.
Two earlier harness setup attempts failed before producing measurements;
their logs were retained, and the restoration handler recovered the service.

## Current-main correctness

The submission is ported onto main `ee8fdab1dc26ea8f09d7910cdc98a17c4b0ea02a`.
Its offline GPU check uses the official vLLM 0.30.0+cu129 wheel in an isolated
virtual environment, with the same PyTorch 2.13.0+cu129 installation.

- Pinned 0.28 backport: **273 CPU regression tests passed**.
- Current main plus cleanup, vLLM 0.30: **326 CPU regression tests passed**,
  including Stage-0-final token-update exclusion and packed payload policy.
- Current-main GPU streaming comparison, C=1, fixed 128 tokens: **8/8 paired
  waveforms and streaming chunk boundaries matched exactly**. This single
  pair is a correctness check, not a current-main performance claim.
- Separate backport profile: **128 bookkeeping/output-build events, zero
  async snapshot events, and zero background-materialization build events**.
  Profiling is separate from the performance measurements above.

TensorRT, Hopper packed inference, online serving, and broader input coverage
were not included in the GPU comparison. Packed-policy compatibility is
covered by a CPU regression test.

## Reproduction

The benchmark now accepts `--fixed-tokens` and `--warmup-batch-size`. On a
matching runtime, select the source with both working directory and
`PYTHONPATH`. Make the model available under the validation directory's
`models/Fun-CosyVoice3-0.5B-2512`, using container-visible paths.

```bash
cd /path/to/source
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
export COSYVOICE3_TRT=0
python /path/to/benchmark.py \
  --validation-dir /path/to/validation \
  --label fixed-c4-baseline-r1 \
  --requests 8 --warmups 2 --concurrency 4 \
  --fixed-tokens 128 --warmup-batch-size 4
```

For historical 0.28 reproduction, create detached checkouts at the baseline
recorded in [manifest.json](manifest.json), apply [v028.patch](v028.patch) to
one with `git apply --unidiff-zero v028.patch`, and run the same benchmark
against both in alternating order. The patch
is a reproduction artifact; current main uses the implementation in production
source. [results.json](results.json) retains every paired observation, while
the separately attached evidence archive contains the full controller scripts,
resource samples, logs, deployment YAMLs, per-request metrics, and raw trace.

AI assistance: Codex assisted with implementation, validation, analysis,
and drafting this report.
