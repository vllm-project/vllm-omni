# LLaMA-Omni2 Code2Wav Batching Validation

## Summary

This report validates the LLaMA-Omni2 Code2Wav batching work rebuilt for
vLLM-Omni PR #5556. The comparison uses the real
`ICTNLP/LLaMA-Omni2-0.5B` Thinker → Talker → Code2Wav pipeline on two NVIDIA
A100 80 GB PCIe GPUs.

The controlled end-to-end gate passed:

- concurrency 1 median audio TTFP changed by `-0.79%` and median audio RTF by
  `-9.44%`, so neither latency metric regressed;
- concurrency 4 audio throughput improved by `28.90%`;
- concurrency 8 audio throughput improved by `70.30%`;
- all 72 measured requests completed with non-empty audio and no request
  failures;
- a Stage 2 profiler trace captured Flow and HiFT executing with batch size 2;
- the real decoder passed deterministic single-request versus batch-2 waveform
  parity at `rtol=1e-4`, `atol=1e-5`.

## Compared Revisions

| Item | Revision or artifact |
| --- | --- |
| Final upstream base | `8ecd1f6d5cc91aab8a475a861213720b336e2f65` |
| Benchmark upstream base | `728640f4d9bb7c1b68d646e2a4c59ce1ce45de9c` |
| Sequential reference source | Code2Wav and deploy content match `d15cd7dd`; Stage 2 was overridden to `max_num_seqs: 1` |
| Batched implementation | `961d88dd` |
| Stage 2 batch admission | `d15cd7dd` |
| Strict benchmark gate | `b4b23d15` |
| Profiler ranges | `7ccfc515` |
| Repeated-EOS sequence fix | `b4331460` |
| Lazy sampler fallback fix | `efdc4978` |
| Final formatted source | `a81c22a3` |

The measured final-after source contained the same Code2Wav and repeated-EOS
logic as `a81c22a3`. The later lazy-sampler change only avoids constructing an
unused fallback sampler and was covered by focused regression tests; it does
not change Code2Wav execution.

The two commits added between the benchmark base and the final upstream base
only modify MiniMax-H3 NPU files and do not overlap this PR's code paths.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-14 |
| Host | `n232-195-203` |
| GPUs | 2 × NVIDIA A100 80 GB PCIe |
| GPU placement | physical GPUs 5 and 7 |
| Driver | `535.261.03` |
| Python | `3.11.2` |
| PyTorch | `2.11.0` |
| CUDA reported by PyTorch | `13.1` |
| vLLM | `0.26.0` |
| Container image | `sha256:65bbb59a4e86e4206337cb5c32973efff3baccf015a3cd840290b99dc08ff70a` |
| Model snapshot | `a16aa9a4ea3f2f363c3db728e8e83ee08e60922c` |
| Decoder snapshot | `7ff21e8e641b00cff2e0492651d654d153b21211` |

The only serving configuration difference was Stage 2 admission:

```diff
-    max_num_seqs: 1
+    max_num_seqs: 8
```

Both configurations used eager float32 Code2Wav execution on the same second
GPU. Thinker, Talker, model snapshots, prompts, row order, warmup count, output
length, and GPU placement were unchanged.

## Benchmark Method

Each implementation served the same fixed eight-row speech-to-speech JSONL
dataset. Every concurrency point used one warmup request followed by eight
measured requests, with `--output-len 12`. The complete c1/c4/c8 matrix was
repeated three times.

The measured command inside the validation container was:

```bash
python benchmarks/tts/bench_tts.py \
  --model ICTNLP/LLaMA-Omni2-0.5B \
  --served-model-name /models/root-cache/snapshots/a16aa9a4ea3f2f363c3db728e8e83ee08e60922c \
  --task speech_to_speech \
  --dataset-path /bench/fixed-container.jsonl \
  --concurrency 1 4 8 \
  --num-prompts 8 \
  --num-warmups 1 \
  --output-len 12 \
  --host 127.0.0.1 \
  --port 18092 \
  -- --disable-tqdm --save-detailed
```

The final gate was computed with:

```bash
python benchmarks/tts/summarize_llama_omni2_runs.py \
  --label c1 \
  --before before-run{1,2,3}/*_c1_*.json \
  --after final-after-run{1,2,3}/*_c1_*.json \
  --label c4 \
  --before before-run{1,2,3}/*_c4_*.json \
  --after final-after-run{1,2,3}/*_c4_*.json \
  --label c8 \
  --before before-run{1,2,3}/*_c8_*.json \
  --after final-after-run{1,2,3}/*_c8_*.json \
  --output llama-omni2-final-gate.json
```

The summarizer rejects fewer than three runs, incomplete request sets, failed
requests, and non-positive audio duration, TTFP, RTF, or throughput before
computing the gate.

## End-to-End Results

Values are median across the three independent runs, followed by sample
standard deviation in parentheses. Relative change is after versus before;
lower is better for TTFP and RTF, while higher is better for throughput.

| Concurrency | Metric | Before | After | Relative change |
| ---: | --- | ---: | ---: | ---: |
| 1 | median audio TTFP (ms) | 161.63 (19.67) | 160.36 (4.75) | -0.79% |
| 1 | median audio RTF | 11.6339 (0.6680) | 10.5359 (0.0401) | -9.44% |
| 1 | audio throughput | 0.08363 (0.00403) | 0.09474 (0.00045) | +13.28% |
| 4 | median audio TTFP (ms) | 23540.64 (2656.29) | 1826.93 (47.40) | -92.24% |
| 4 | median audio RTF | 41.1119 (7.2657) | 37.0967 (3.2199) | -9.77% |
| 4 | audio throughput | 0.08341 (0.00120) | 0.10751 (0.00668) | +28.90% |
| 8 | median audio TTFP (ms) | 50531.90 (7872.92) | 5653.38 (217.82) | -88.81% |
| 8 | median audio RTF | 53.1751 (5.5151) | 51.8532 (5.8090) | -2.49% |
| 8 | audio throughput | 0.08697 (0.00281) | 0.14810 (0.01600) | +70.30% |

The large high-concurrency TTFP change comes from removing the
single-sequence Stage 2 admission bottleneck. Throughput, rather than TTFP
alone, is used as the primary performance claim.

### Gate Decision

| Gate | Requirement | Result |
| --- | --- | --- |
| c1 TTFP | regression no greater than 5% | PASS, `-0.79%` |
| c1 RTF | regression no greater than 5% | PASS, `-9.44%` |
| c4 or c8 throughput/RTF | at least 10% improvement | PASS, c4 throughput `+28.90%`, c8 throughput `+70.30%` |
| Three runs per point | required | PASS |
| Complete positive-audio results | required | PASS |

## Correctness

### Online Pipeline

Across the nine final-after result files:

- `72/72` measured requests completed;
- `0` requests failed and every per-request error string was empty;
- all result files reported positive audio duration, audio frames, TTFP, RTF,
  and throughput;
- all three run directories passed the independent strict validation script;
- the final benchmark logs contained no traceback, fatal-engine, CUDA,
  device-side-assert, internal-server-error, or monotonic-`chunk_seq` pattern.

The repeated-EOS regression discovered during profiler warmup is covered by
tests that prove:

- one or more Talker EOS tokens are stripped from codec units;
- an early EOS-only update does not send an empty chunk or increment
  `chunk_seq`;
- the actual scheduler-finished update emits exactly one terminal chunk;
- codec tokens after EOS remain an error;
- invalid updates do not mutate request state.

### Real Decoder Parity

The real `ICTNLP/cosy2_decoder` Flow and HiFT modules were loaded on one A100.
One fixed 32-unit codec sequence was decoded once as a single request and once
as two equal-shape requests in one batch.

CosyVoice2 CFM calls `torch.randn_like(mu)`, whose CUDA sequence depends on
tensor shape. For a meaningful deterministic pointwise comparison, the parity
run disabled TF32 and cuDNN benchmarking, enabled deterministic algorithms,
and supplied the same request-stable CFM noise to every row.

| Field | Result |
| --- | ---: |
| Sample rate | 24,000 Hz |
| Output samples per request | 30,720 |
| Finite samples per request | 30,720 |
| Maximum absolute difference | `8.8513e-06` |
| Mean absolute difference | `5.8815e-07` |
| RMSE | `1.0097e-06` |
| Cosine similarity | `0.9999999999565` |
| `torch.testing.assert_close` | PASS at `rtol=1e-4`, `atol=1e-5` |

Both batch rows matched the single-request reference at the stated tolerance.
Uncontrolled stochastic runs are not expected to be pointwise identical
because changing batch shape changes the CFM noise draw.

## Profiler Attribution

The Stage 2 trace was captured only after the benchmark entered its measured
run. Profiling used a short scheduled window with shape recording disabled to
avoid the memory growth observed when profiling the complete request.

| Range | batch=1 count | batch=2 count |
| --- | ---: | ---: |
| `llama_omni2.code2wav.flow` | 20 | 20 |
| `llama_omni2.code2wav.hift` | 20 | 20 |
| `llama_omni2.code2wav.d2h` | 19 | 19 |

The trace contains 3,665,143 events and is 69,121,019 bytes compressed.
`flow_max_batch=2` and `hift_max_batch=2`, proving that the real Flow and HiFT
paths executed more than one request in a GPU call. The trace is attribution
evidence only; its profiled request latency is not used in the A/B table.

## Regression Validation

The final source was validated in the same runtime image:

- LLaMA-Omni2 model tests plus MiniCPM-o Code2Wav regression:
  `136 passed`;
- touched scheduler, connector, output, finish-reason, worker, and benchmark
  tests: `276 passed`;
- LLaMA-Omni2 missing-HF-config engine-args test: `1 passed`;
- focused terminal-EOS and benchmark-gate tests: `43 passed`;
- Ruff `0.14.10` lint: PASS;
- Ruff `0.14.10` format check: PASS;
- `git diff --check`: PASS.

The complete `tests/engine/test_arg_utils.py` file could not run cleanly in
the offline validation container because 12 unrelated tests require the
default `Qwen/Qwen3-0.6B` snapshot, which is not present in that image. The
LLaMA-Omni2-specific test from that file passed independently.

## Raw Artifacts

Raw artifacts are intentionally ignored by Git under:

```text
benchmarks/tts/results/llama_omni2/20260814-a100/
```

Important files:

- `before-run{1,2,3}/*.json`
- `final-after-run{1,2,3}/*.json`
- `llama-omni2-final-gate.json`
- `environment.txt`
- `configs/before.yaml`
- `configs/after.yaml`
- `parity/real-code2wav-deterministic-metrics.json`
- `parity/service-log-scan.txt`
- `profiler/stage2/summary.json`
- `profiler/stage2/trace_rank0.json.gz`

## Limitations

- The workload contains eight fixed prompts and 12 output text tokens. It
  validates the batching path and queueing behavior but is not a capacity
  curve for long conversations.
- The profiler captured exact-shape batches up to size 2. It proves true
  multi-request execution, not that every scheduler step reaches batch 8.
- Results are specific to two A100 80 GB PCIe GPUs, the listed snapshots, and
  this deployment topology.
- Numerical waveform parity requires deterministic request-stable CFM noise;
  normal diffusion sampling remains stochastic.
- The final lazy-sampler fix was covered by regression tests after the E2E
  matrix. It does not alter Thinker, Talker, or Code2Wav math.
