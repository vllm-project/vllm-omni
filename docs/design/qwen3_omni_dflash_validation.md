# Qwen3-Omni DFlash validation

Validation for [PR #9595](https://github.com/vllm-project/vllm-omni/pull/9595).
The payload lifecycle is described in
[Async Omni Output Materialization](feature/omni_async_output_materialization.md).

## Method

**Revision:** all results below were rerun using a clean frozen tree of `caf99e8c80814dcf749c85bf6bc1739e7b1fa262`, rebased onto upstream `7b3c84a5e`. Subsequent validation-document commits do not change runtime code. These results supersede the earlier frozen `529ff4c6` + `d45ed8d5` measurements. Source diff SHA256: `563e5c0235371f1476d654971fab6a239ed6ef41db17c92fe07e544ed1aa8d9c`.

**Hardware/software:** 3 × NVIDIA A100-SXM4-40GB: Thinker TP2 on GPU0/1; ordinary Talker and Code2Wav on GPU2; fourth GPU unused. vLLM 0.33.0 wheel at `4ba787ec9959e566d3b587a796dff07673324da5`, Torch 2.14.0+cu130, MRv1, FlashAttention, Qwen3-Omni-30B-A3B-Instruct, four speculative tokens, prefix caching disabled.

**Draft weights:** public Hugging Face checkpoint [`yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8`](https://huggingface.co/yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8), the block-8 `DFlashDraftModel` used with the Qwen3-Omni target.

**Dataset:** LibriSpeech test-clean; ModelScope `DiffSynth-Studio/example_image_dataset` and `DiffSynth-Studio/example_video_dataset`. Every timed batch has the same 60 inputs: 40 audio, 10 images, 10 videos. Images/videos each reuse five files: 50 distinct media files total. Audio prompts request transcription; image/video prompts request short descriptions.

| Path | Output | Thinker async scheduling | Payload materialization | async_chunk |
| --- | --- | --- | --- | --- |
| Text-only | Text | false | No speech payload delivered | false |
| A | Full-payload speech | false | Inline | false |
| B | Chunked speech | false | Inline | true |
| C | Chunked speech | true | Deferred consumed-input span, subject to guards | true |

Each baseline disables Thinker speculation; its DFlash counterpart enables it. Comparing B and C changes scheduling and materialization together and cannot isolate either contribution.

**Eager performance protocol:** target/drafter `enforce_eager=true`; closed-loop concurrency 1/4/8; per-modality and per-concurrency warmup; one measured 60-request batch per configuration/concurrency. Eight cells × three concurrencies × 60 = 1,440 timed requests. Fixed service order: A baseline, A DFlash, B DFlash, B baseline, C baseline, C DFlash. Text-only cells share the A services. Thinker greedy/top-k 1/max_tokens 96/seed 42; Talker temperature 0.9/top-k 50/repetition penalty 1.05/seed 42 with stochastic residual codebooks; Code2Wav greedy. Fixed seeds do not guarantee equal audio across different RNG-consumption patterns.

Performance services have no audit hooks. Throughput is successful requests divided by batch wall time, including post-response audio validation. Per-request E2E excludes that validation; input encoding is outside timing. TTFA is POST to first nonempty audio SSE packet, not GPU first frame or playback. Packet gaps pool adjacent events within requests. Output lengths differ; observed ratios are single-round observations, not statistically established equal-work speedups.

**Controlled correctness protocol:** five inputs (three audio, one image, one video, including previously divergent samples), text-only/speech, two repetitions, c1, A/B/C baseline/DFlash. Thinker and Talker main codebook greedy; Talker residual codebooks explicitly `subtalker_sampling_params: {do_sample: false}`. Compare final Thinker IDs and all 16 codec codebooks after removing repeated chunk left context. Generated-speech ASR uses the same model as a screening reference, not independent ASR/listening. Audit services run separately and their timings are excluded from performance.

## Current-head eager results

All 1,440 timed requests, 120 controlled main requests and 30 generated-speech
ASR checks completed successfully. All six performance and six correctness
service logs had no ERROR/Traceback entries. The failed preliminary audit
attempt used an invalid request-ID field in local instrumentation; it was
corrected and rerun, and its results are excluded.

### Throughput

| Path | c | Baseline req/s | DFlash req/s | Ratio | Mean E2E baseline → DFlash (s) | Success baseline / DFlash |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Text-only | 1 | 0.708 | 1.180 | 1.666× | 1.412 → 0.847 | 60/60 / 60/60 |
| Text-only | 4 | 2.538 | 4.037 | 1.590× | 1.530 → 0.948 | 60/60 / 60/60 |
| Text-only | 8 | 4.668 | 7.692 | 1.648× | 1.592 → 0.945 | 60/60 / 60/60 |
| A | 1 | 0.420 | 0.568 | 1.353× | 2.376 → 1.756 | 60/60 / 60/60 |
| A | 4 | 1.475 | 1.915 | 1.298× | 2.593 → 2.046 | 60/60 / 60/60 |
| A | 8 | 2.548 | 3.167 | 1.243× | 2.925 → 2.305 | 60/60 / 60/60 |
| B | 1 | 0.466 | 0.589 | 1.265× | 2.142 → 1.692 | 60/60 / 60/60 |
| B | 4 | 1.664 | 1.926 | 1.157× | 2.340 → 2.011 | 60/60 / 60/60 |
| B | 8 | 2.811 | 3.213 | 1.143× | 2.684 → 2.325 | 60/60 / 60/60 |
| C | 1 | 0.477 | 0.590 | 1.238× | 2.093 → 1.690 | 60/60 / 60/60 |
| C | 4 | 1.642 | 1.961 | 1.195× | 2.344 → 1.960 | 60/60 / 60/60 |
| C | 8 | 2.837 | 3.315 | 1.169× | 2.683 → 2.270 | 60/60 / 60/60 |

## Latency (P50 / P95 in milliseconds)

| Path | c | TTFT baseline → DFlash | TTFA baseline → DFlash | E2E baseline → DFlash | Audio packet gap baseline → DFlash |
| --- | ---: | --- | --- | --- | --- |
| Text-only | 1 | 77.6 / 259.1 → 111.5 / 463.5 | N/A → N/A | 1327.0 / 2899.8 → 644.3 / 2050.8 | N/A |
| Text-only | 4 | 120.8 / 284.3 → 141.3 / 328.9 | N/A → N/A | 1416.7 / 3021.1 → 708.9 / 2442.7 | N/A |
| Text-only | 8 | 125.7 / 294.7 → 145.3 / 438.0 | N/A → N/A | 1494.6 / 3103.0 → 742.0 / 2421.0 | N/A |
| A | 1 | 109.2 / 455.8 → 102.0 / 393.6 | 2205.5 / 4987.0 → 1528.8 / 3655.4 | 2205.7 / 4987.1 → 1529.0 / 3655.6 | N/A |
| A | 4 | 123.8 / 377.2 → 138.4 / 377.0 | 2376.8 / 5246.7 → 1755.5 / 4261.4 | 2385.8 / 5246.9 → 1755.7 / 4261.6 | N/A |
| A | 8 | 130.9 / 379.8 → 156.7 / 490.8 | 2604.4 / 5908.4 → 1942.1 / 4873.4 | 2604.6 / 5908.5 → 1942.3 / 4873.6 | N/A |
| B | 1 | 112.3 / 454.7 → 125.0 / 494.7 | 437.6 / 788.0 → 495.3 / 872.6 | 1894.0 / 4452.8 → 1540.3 / 3198.2 | 299.8 / 1516.6 → 296.6 / 977.9 |
| B | 4 | 127.4 / 361.3 → 119.0 / 396.1 | 489.3 / 710.1 → 515.2 / 851.1 | 2037.5 / 4677.2 → 1805.2 / 4082.4 | 366.5 / 1576.4 → 384.4 / 1031.3 |
| B | 8 | 138.1 / 430.5 → 151.2 / 364.7 | 524.5 / 819.7 → 581.3 / 854.2 | 2407.7 / 5799.5 → 1967.1 / 4511.2 | 496.9 / 1678.7 → 498.0 / 1227.4 |
| C | 1 | 110.0 / 478.5 → 130.0 / 529.2 | 431.0 / 763.4 → 491.8 / 837.0 | 1890.6 / 4410.1 → 1525.4 / 3244.3 | 295.0 / 1496.9 → 296.2 / 950.6 |
| C | 4 | 186.8 / 412.1 → 197.8 / 367.7 | 540.9 / 702.8 → 574.7 / 739.7 | 2126.1 / 4947.6 → 1766.8 / 3747.1 | 373.8 / 1558.5 → 378.0 / 1109.2 |
| C | 8 | 183.6 / 367.5 → 209.8 / 395.5 | 564.4 / 733.3 → 609.9 / 809.6 | 2434.1 / 5562.5 → 1982.2 / 4375.3 | 494.2 / 1631.7 → 510.2 / 1375.4 |

## Audio-transcription reference WER

| Path | c | Baseline edits / words (WER) | DFlash edits / words (WER) |
| --- | ---: | --- | --- |
| Text-only | 1 | 3/722 (0.42%) | 3/722 (0.42%) |
| Text-only | 4 | 3/722 (0.42%) | 3/722 (0.42%) |
| Text-only | 8 | 3/722 (0.42%) | 3/722 (0.42%) |
| A | 1 | 3/722 (0.42%) | 3/722 (0.42%) |
| A | 4 | 3/722 (0.42%) | 3/722 (0.42%) |
| A | 8 | 3/722 (0.42%) | 3/722 (0.42%) |
| B | 1 | 3/722 (0.42%) | 3/722 (0.42%) |
| B | 4 | 3/722 (0.42%) | 3/722 (0.42%) |
| B | 8 | 3/722 (0.42%) | 3/722 (0.42%) |
| C | 1 | 3/722 (0.42%) | 3/722 (0.42%) |
| C | 4 | 3/722 (0.42%) | 3/722 (0.42%) |
| C | 8 | 3/722 (0.42%) | 3/722 (0.42%) |

## Controlled greedy comparison

| Path | Text-only Thinker tokens identical | Speech Thinker tokens identical | Speech codec frames identical |
| --- | ---: | ---: | ---: |
| A | 4/10 | 4/10 | 0/10 |
| B | 4/10 | 4/10 | 4/10 |
| C | 4/10 | 4/10 | 4/10 |

| Service | Main requests | Token repeats | Speech-frame repeats | Same-model generated-speech ASR WER |
| --- | ---: | ---: | ---: | --- |
| full_baseline | 20/20 | 10/10 | 5/5 | 1/149 (0.67%) |
| full_dflash | 20/20 | 10/10 | 4/5 | 1/153 (0.65%) |
| syncchunk_baseline | 20/20 | 10/10 | 5/5 | 1/149 (0.67%) |
| syncchunk_dflash | 20/20 | 10/10 | 5/5 | 2/153 (1.31%) |
| chunk_baseline | 20/20 | 10/10 | 5/5 | 1/149 (0.67%) |
| chunk_dflash | 20/20 | 10/10 | 5/5 | 2/153 (1.31%) |

## Payload and position audits

Correctness-only instrumentation performs one additional batched embedding
lookup on every TP rank and compares it with the prepared rows. This audit
is excluded from performance measurements. There were no embedding-value
mismatches in 768 / 770 / 764 TP preparation records for A / B / C,
respectively. These counts include both TP ranks and the ASR requests;
they are not independent request counts.

At the receiving processor, all 150 B and 148 C decode spans retained their
ID/embedding row hashes. A separate full-payload audit reran 20 main requests
and five ASR checks: all ten speech requests retained the ordered decode-row
hashes after accumulation. Prompt rows were checked separately from decode
rows; a full-payload `ids.output` field describes the latest step, not the
entire accumulated prompt and output.

Both TP ranks' recorded prepared sequences preserve each scheduler-final
sequence through the stop boundary (25 requests per mode/rank, including
ASR). Prepared verifier batches may contain additional candidates after the
stop token. This is a stop-prefix audit, not a claim that every GPU candidate
is delivered or that final payloads are raw concatenations of sampler rows.
Terminal and tool-boundary filtering remain separate responsibilities.

Real image requests have 594 prompt tokens but final prompt MRoPE coordinates
`[41, 41, 41]` (delta -552). The first generated input has corrected linear
position 594; later positions advance. The preparation records and runner
regression jointly cover the independent linear position used for first-token
seeding. Multi-step tests cover zero-draft first decode followed by speculative
decode, without a missing or repeated conditioning row.

## CUDA Graph follow-up

An additional C-path comparison sets both target and drafter
`enforce_eager: false`, preserving the eager matrix's other settings.
The default custom-all-reduce baseline completed 180/180 timed requests
at concurrency 1/4/8 (0.926 / 2.324 / 3.402 req/s). The corresponding
DFlash service failed before health readiness, during CUDA Graph memory
profiling (`PIECEWISE=49`, `FULL=38`), with:

```text
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:164
'an illegal memory access was encountered'
```

No DFlash performance or correctness result is available for that default
graph configuration. The location of the error does not establish its root
cause. A separate comparison disables custom all-reduce on both baseline
and DFlash; its results must remain distinct from the default configuration.

### Graph comparison with custom all-reduce disabled

Both services set stage 0 `disable_custom_all_reduce: true` and target/drafter
`enforce_eager: false`. Other C-path settings and workload are unchanged.
All **360/360 timed requests** succeeded; timings exclude audit hooks.

| c | Baseline req/s | DFlash req/s | Ratio | Mean E2E baseline → DFlash (s) | TTFA P50/P95 baseline → DFlash (ms) |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.928 | 0.912 | 0.983× | 1.073 → 1.092 | 192.4 / 655.9 → 230.2 / 691.8 |
| 4 | 2.278 | 2.500 | 1.097× | 1.714 → 1.560 | 284.5 / 604.5 → 282.4 / 655.6 |
| 8 | 3.594 | 3.798 | 1.057× | 2.107 → 1.983 | 352.6 / 613.5 → 358.2 / 729.9 |

DFlash throughput is 1.7% lower at c1 and 9.7% / 5.7% higher at c4/c8.
Its embedding/payload work and changed generation shapes are included in the
timing, but no profiling isolates the c1 regression. The baseline itself is
much faster with graphs; eager ratios do not carry over to this configuration.
This single fixed-order round does not establish statistical significance.

The graph correctness services completed 40/40 main requests and ten ASR
checks. Both services repeated Thinker tokens in 10/10 cases and speech codec
frames in 5/5. Across baseline and DFlash, Thinker tokens matched 8/10
text-only and 8/10 speech cases; complete speech codec frames matched 6/10.
These are distinct from the eager results and do not establish equivalence.

The graph DFlash preparation audit found zero lookup mismatches in 744 TP
records; all 146 receiving decode spans retained row hashes. Actual stage-0
FULL and PIECEWISE graph replay events were recorded, including DFlash;
these results do not merely reflect successful graph configuration/capture.

Generated-speech same-model ASR WER was 1/154 (0.65%) for the graph baseline
and 20/153 (13.07%) for graph DFlash. The largest error is the first audio
sample: its ASR transcript stops after “It was this observation that drew
from Douglass,” while the generated text continues. This is an unresolved
screening regression; the present evidence does not distinguish truncated
generated speech from an ASR-model error. Passing payload row checks does
not resolve it, and graph speech quality is not accepted on these results.

### Default custom-all-reduce startup diagnosis

A startup-only interception on the unmodified frozen runtime found a replay
of the 320-token PIECEWISE graph while custom all-reduce's capture context
was still open (`_IS_CAPTURING=true`, `_capture_registered=true`). DFlash
clamps profiling inputs to `max_num_seqs * (1 + num_speculative_tokens)` =
320, so distinct larger profiling sizes can map to the same draft graph.
The graph wrapper replays the cached entry, but custom all-reduce registers
its captured IPC buffers only on exit from the outer context. This is a
reproduced ordering hazard in the pinned upstream startup path, preceding
sampling and speculative payload preparation.

The original error is reported at `cudaIpcGetMemHandle`, not necessarily at
the first invalid device access; asynchronous CUDA error reporting prevents
attributing the device fault solely to that line. The diagnostic intercept
stops before replay. No production fix or upstream-version comparison was
applied; disabling custom all-reduce is a tested workaround, not a fix for
the default configuration. Default graph startup remains unsupported by this
validation.

## Interpretation and limits

Strict greedy equivalence is not established: the controlled A/B/C inputs
still match the baseline in only 4/10 Thinker token sequences. Full-payload
DFlash repeats identical Thinker tokens but differs in codec frames for one
image request (4/5 frame-repeat agreement). Same-prefix logits/argmax-margin
analysis has not been completed; these differences are not attributed to
BF16 or numerics without measurements. The five-input set includes known
divergent cases and is not a population accuracy estimate.

Low reference WER and successful HTTP completion do not establish complete
speech-content or naturalness correctness. Generated-speech ASR uses the same
model and is not independent ASR or listening. Performance output volume is
retained in local JSON; output lengths differ, and no stable equal-work gain
or isolated async-materialization benefit is claimed.

MRv2, duplex/resumable input, exhaustive cancellation/request-ID reuse,
preemption/long-context boundaries, model-level stochastic distribution
equivalence, and independent speech listening are outside this validation.
Prefix-cache combinations are not covered by this matrix. Earlier tool-call
checks are historical evidence and were not rerun here.

The related CPU regression suite passed with 286 passed, 6 skipped and
3 warnings. Repository-wide tests and maintainer CUDA L2/Buildkite were not
run. Local manifests, exact deploy configurations, frozen sources, per-request
JSONL, token/frame traces, WAV/ASR files and service logs are retained locally;
the benchmark harness and raw media are not included in this document.
