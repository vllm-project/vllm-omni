# CosyVoice3 B2 implementation and acceptance

This is the completion report for
[RFC #6870, B2](https://github.com/vllm-project/vllm-omni/issues/6870), following
the [maintainer's revised scope](https://github.com/vllm-project/vllm-omni/issues/6870#issuecomment-5676714631).
The production implementation omits unused Talker hidden payloads in the
default async-chunk pipeline, retains prompt conditioning, and delivers
sampled codec IDs on token-only steps. Output materialization remains inline.

The detailed [design contract](../../../../docs/design/feature/cosyvoice3_talker_output_payloads.md)
describes prefill, decode, completion, request ownership, legacy compatibility,
and the scheduler's Stage-0-final exclusion. This implementation reuses the
runner's existing payload policy; it does not add another materialization
thread or modify sampling/synthesis computations.

## Acceptance coverage

| Check | Result | Evidence |
| --- | --- | --- |
| Hidden payload omission | Passed | CPU tests verify the hidden CPU-staging helper is not invoked for async chunks; conditioning and codec IDs are retained |
| Async and sync AR scheduling | Passed | Model/runner policy tests cover both scheduling modes |
| Legacy non-async-chunk payload policy | Passed at unit level | Legacy hidden staging and conditioning remain available |
| Concurrent request isolation | Passed | Interleaved requests keep their conditioning, codec prefixes, and EOF state separate |
| Stage-0-final exclusion | Passed | Token-only updates do not send chunks to a nonexistent downstream consumer |
| Current-main regression suite | Passed | 331 selected CPU tests, 18 warnings, 7.25 seconds |
| Current-main C=1 streaming | Passed | 24/24 paired waveforms and chunk boundaries match across three unprofiled paired rounds |
| Current-main C=4 streaming | Passed | 24/24 paired waveforms and chunk boundaries match across three unprofiled paired rounds |
| Concurrent boundary diagnostics | Passed | Talker codec streams, all 24 sender/receiver prefixes, conditioning, and flow inputs match on current main |
| Diagnostic RNG control | Passed | Current-main comparisons with request-scoped diagnostic RNG also match; the control is absent from production |

GPU validation uses the default async-chunk pipeline with the Torch flow
backend on one RTX 4090. TensorRT and Hopper packed GPU execution are outside
the GPU scope; the existing packed payload policy has CPU regression coverage.
The baseline legacy GPU path has a separate conditioning-concatenation failure,
documented below rather than reported as a passing compatibility test.

## Current-main performance

The baseline is main `ee8fdab1dc26ea8f09d7910cdc98a17c4b0ea02a`; the measured
cleanup snapshot is `63c924168c79754f0467443563def38bfbb44ac8`. Subsequent
submission updates add tests, documentation, and benchmark metric collection;
the three production files retain the measured implementation. Hashes and
environment details are in [manifest.json](manifest.json).

Both variants use vLLM 0.30.0+cu129, PyTorch 2.13.0+cu129, Transformers
5.14.1, the same official CosyVoice3 weights and reference WAV, seed 0,
128 output tokens, 25-token codec chunks, async chunk, AR async scheduling,
and disabled prefix caching. Each run has eight measured requests after two
full-batch warmups at its measured concurrency. The A/B order alternates
between paired rounds.

The original GPU-0 service was temporarily stopped for each GPU reservation
and automatically restored. GPU 1's services remained running. CPU affinity
was not pinned. NVML process and memory samples accompany each run.

| Concurrency | Paired rounds | Baseline AR ITL (ms/token) | B2 AR ITL (ms/token) | Mean reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3 | 4.852 | 4.718 | 2.75% |
| 4 | 3 | 6.118 | 6.107 | 0.19% |

For C=1, B2-minus-baseline deltas were -0.26895, -0.04847, and -0.08287
ms/token. The approximate paired-t 95% interval is [-0.42808, +0.16122]
ms/token. All three pairs improved locally, with substantial variation.

For C=4, deltas were -0.03535, +0.02378, and -0.02343 ms/token. The
approximate paired-t 95% interval is [-0.08934, +0.06601] ms/token. One pair
regressed. These are three independent paired rounds, not 24 independent
timing experiments; both intervals include zero.

| Concurrency | Mean of run-median TTFA, baseline → B2 (ms) | Mean last-audio time (ms) | Audio throughput (audio-s/s) |
| --- | ---: | ---: | ---: |
| 1 | 505.25 → 500.70 | 1045.54 → 1010.84 | 4.899 → 5.067 |
| 4 | 1533.68 → 1514.76 | 3143.75 → 3090.86 | 6.512 → 6.622 |

The data supports completing B2 as payload cleanup. It does not establish a
hardware-independent speedup or resolve the RFC's larger sampler/host gap and
flow-batching workstreams.

### Native timing collection

Current main emits detailed metrics tables only at DEBUG. The benchmark reads
the already-collected `output.metrics["stage_metrics"]["0"]["vllm_itls_ms"]`
arrays instead of enabling DEBUG or a profiler. Each measured request has
127 decode intervals, giving 1,016 intervals per run. Warmup arrays are not
saved with the measured requests.

The analyzer uses those native arrays when present and retains the historical
log-table parser for 0.28 reproduction. Model execution is not instrumented
for these performance rounds. Diagnostic tensor hashing and RNG controls
run separately and supply no latency claims.

## Resolving the earlier C=4 discrepancy

The previous 0.28 measurements showed waveform differences despite matched
token counts and audio lengths. The new diagnostic follows the entire boundary:

1. Record each Talker's final valid codec sequence.
2. Match every sender prefix to that sequence.
3. Match every Code2Wav receiver prefix to its sender.
4. Compare prompt conditioning, flow encoder inputs, and generated flow noise.
5. Repeat with diagnostic request-scoped RNG, then inspect AR sampling history.

On current main, ordinary and RNG-controlled comparisons each match all
eight final codec sequences, all 24 chunk payloads/conditioning/flow inputs,
and all eight waveforms. The separate uninstrumented C=4 rounds also match
all 24 paired waveforms, so parity is not confined to instrumented runs.

On 0.28, every sender and receiver prefix still matches its own Talker output,
and conditioning matches across variants. Flow noise matches as well, but
the Talker codec sequences already differ. Request-scoped flow RNG does not
remove that divergence. Even the unchanged 0.28 baseline changes some codec
sequences when only downstream diagnostic timing/RNG conditions change.

The history diagnostic verifies 1,024 sampling histories in each 0.28 variant:
every history contains exactly the already generated prefix for its request.
In diverging requests, the first token difference precedes the recorded
generator-state and subsequent history differences. The discrepancy is
therefore upstream of the connector/flow boundary, not token loss, swapped
conditioning, or simply reassigned flow noise. Its precise old-runtime
sampling/numerical cause is not established by these traces.

Consequently, the historical 0.28 C=4 delta is not a clean estimate of the
payload change alone. The current-main paired acceptance results replace it
for this submission. Historical reports remain available for provenance.
Details are in [diagnostics.json](diagnostics.json) and
[sampling_history.json](sampling_history.json).

## Legacy GPU check

A non-async-chunk GPU check fails on the unmodified current-main baseline
with an `embed.speech_feat` concatenation error: expected length 174 but
received 1500. This occurs before the benchmark produces measured requests.
The baseline failure cannot be counted as successful legacy GPU validation.
The B2 branch produces the same error and the same shape mismatch, confirming
this failure exists independently of B2. [compatibility.json](compatibility.json)
records both runs; unit tests verify that B2 preserves legacy policy.

## Reproduce

Use the matching current-main runtime and make the official weights available
under the validation directory's `models/Fun-CosyVoice3-0.5B-2512`. Select the
source with both working directory and `PYTHONPATH`:

```bash
cd /path/to/source
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
export COSYVOICE3_TRT=0
python /path/to/benchmark.py \
  --validation-dir /path/to/validation \
  --label c4-baseline-r1 --requests 8 --warmups 2 --concurrency 4 \
  --fixed-tokens 128 --warmup-batch-size 4
```

Run the same command against the B2 checkout and alternate order across three
pairs. For C=1, set concurrency and warmup batch size to 1. The analyzer reads
native metrics from `metrics.json` and compares the saved NumPy waveforms.
Use `--no-async-chunk` only for the separately labelled legacy check.

CPU regression commands and prerequisites are in the submission's operation
notes. The attached completion archive contains controller scripts, diagnostic
source patches, trace JSONL, metric arrays, per-run resource samples, logs,
deployment YAMLs, and compatibility evidence. It excludes weights, environments,
credentials, and generated audio arrays.

AI assistance: Codex assisted with implementation, tests, diagnostics,
execution, analysis, and drafting these materials.
