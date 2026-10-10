# CosyVoice3 B1: V1 RAS sampler validation

This change uses live request parameters from the existing V1 runner interface
and batches mixed greedy/random sampling. Greedy rows consume no RNG, and RAS
draws a replacement only for rejected rows. The finite-logits check and one
batched rejection readback remain. MRv2 and standard sampling are unchanged.

SamplingMetadata follows V1 InputBatch's contract: parameter tensors are on
the sampling device with one entry per request, and absent top-p/k tensors use
the model defaults. The sampler does not repair malformed tensor shapes.
Missing host request parameters retain tensor-based routing. Float32 rounding
matches InputBatch at the penalty and greedy boundaries.

## Run the sampler comparison

Use a Linux CUDA development environment with this checkout's pinned
**vLLM 0.31.0**, CosyVoice3 dependencies and pytest. The tested machine uses an
RTX 4090, driver 580.65.06, Python 3.12.3 and PyTorch 2.13.0+cu130.

```bash
python -m pip install -e . --no-deps
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.tts.validate_cosyvoice3_b1 \
  --output-dir /tmp/cosyvoice3-b1-new-run
```

Use a fresh output directory. Add `--quick` for a smoke matrix; the test suite
still runs in full. No weights are needed. The command records the environment,
runs CPU/real-runner and CUDA tests, checks token/RNG trajectories, then measures
paired sampler timings and captures separate traces. Missing or skipped tests
fail validation; CUDA requests never silently fall back to CPU.

The reference is frozen from commit
`4c5541cfc17143f80bdb89bbb7a5840b08bb52c6`. Both arms bind the same installed
`random_sample` after Omni initialization, including its seeded-RNG patch.
The benchmark verifies callable identity and records source hashes.

To run only the sampler after tests pass:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.tts.benchmark_cosyvoice3_b1 \
  --trace --output /tmp/cosyvoice3-b1-replay/sampler.json
```

The full matrix covers 112 cases: batch sizes 1/4/8, mixed/all-random requests,
float32/bfloat16, full/partial/no request seeds, and four history/logit cases.
Seeded tokens and per-request RNG states are compared for 40 steps. Unseeded
rows are checked for valid support, not identical trajectories.

Timings alternate 30 AB/BA pairs, each with 16 calls, after five warmup blocks.
Cloning and RNG resets are outside timing; host parameter gathering is inside.
`completed_ms` includes GPU completion. The reported confidence interval
resamples paired rounds; it does not eliminate external contention. Traces run
after timing. A `passed` report means validation completed, not that all cases
are faster. Inspect default all-random cases as well as mixed batches.

## Serving validation and evidence boundary

Sampler replay does not measure whole-request latency. Use the existing
serving benchmark on each checkout with the same environment and V1 deploy
profile (`vllm_omni/deploy/cosyvoice3.yaml`):

```bash
python benchmarks/tts/bench_cosyvoice3_concurrency_matrix.py \
  --model /path/to/Fun-CosyVoice3-0.5B-2512 \
  --ref-audio /path/to/reference.wav --ref-text 'reference transcript' \
  --concurrency 1 4 --step-size 25 --num-prompts 16 --warmup 4 \
  --trt 0 --seed 42 --output-dir /tmp/cosyvoice3-b1-serving
```

The simplified source passed 113 tests on the RTX 4090: 74 sampler/runner/CUDA
and harness tests, plus 39 adjacent tests affected by the shared fixture.
All 112 correctness scenarios and paired timings completed. Median per-case
sampler latency reductions versus the frozen baseline were 9.27–11.16% for
all-random batches and 28.44–59.43% for mixed batches. All within-run paired
confidence intervals were positive. These numbers describe sampler replay;
they do not establish E2E gains or isolate the benefit of simplification alone.

The pre-simplification real-model experiment used V1 RAS, Torch Flow, chunk 25, seed 42,
two texts and four independent AB/BA server pairs per concurrency. Each server
ran four warmups and 16 measured requests. All 256 measured requests succeeded:

| Concurrency | Baseline/head matching PCM | Mean E2E baseline → head |
| --- | --- | --- |
| 1 | 64/64 | 1413.76 → 1413.37 ms |
| 4 | 9/64 | 4754.22 → 4895.08 ms |

The serial paired saving was 0.39 ms, with a 95% interval of [-30.79, 20.44] ms:
no significant E2E improvement. Concurrent raw latency increased 2.96%, with
different outputs and some different lengths; the baseline itself was not
repeatable across concurrent runs. This result does not establish attribution
or absence of regression. Full-chain performance acceptance remains open.

Request seed controls token sampling; default Flow uses global process noise.
Controlled cross-version audio equivalence is distinct from arbitrary-concurrency
determinism. Use fixed work and full-batch warmups to isolate performance, then
validate natural-EOS serving separately. Temporary PCM capture and analysis
scripts are kept with the experiment records, outside this production patch.
