# World Model Validation

## Freeze the comparison

Record Omni base/head, official source, checkpoint/adapter revisions, extra
patches, asset hashes, prompt/controls, seed/generator, dtype, attention,
geometry, schedule, topology, and decoder. Verify actual import locations.

| Comparison | Question answered |
| --- | --- |
| Unmodified official entry point versus Omni | Does the port preserve the reference behavior? |
| Omni offline versus session/stepwise | Does the serving mode change the result? |
| Same latents, one-shot versus streaming decode | Does incremental decoding preserve time and content? |

Keep diagnostic noise changes, alternative decoders, and precision studies
separate. If the official environment cannot run, retain its concrete failure,
complete available internal checks, and leave canonical quality unqualified.

## Capability-driven test matrix

Use [vllm-omni-test](../../vllm-omni-test/SKILL.md) for helpers, markers,
levels, and copy-paste local/CI-like commands. Select conditional rows only
when the integration implements or advertises that capability.

| Layer | Minimum meaningful evidence |
| --- | --- |
| CPU contracts | Real registry/pre/post resolution, explicit config precedence, public/model parameter propagation, early invalid-config failures. |
| Loader | Real tiny shards, duplicate/missing/unexpected/shape cases, actual loaded sets, released-index audit, constructed device/dtype. |
| Reference | Independently constructed official and Omni models, short trajectory first, then boundary-crossing latents and complete float output. |
| Offline/API | Actual public entry point, decoded returned media, correct controls and geometry. |
| Continuation/streaming | Multiple chunks, updates, stable output, normal flush, complete metadata, per-chunk cadence. |
| Lifecycle | A/B/A; interleaved sessions where supported; finish/abort/fault/reset followed by fresh generation; state inspected on each rank. |
| Distribution/offload | Full-output error and memory for each advertised layout against the working baseline. |
| A/V | First/tail content, buffered short chunks, tracks, timestamps, saved media, cancellation, real codec decoding. |
| Graph/compile | Actual dispatch and replay, first/subsequent shapes, changed prompt/bucket, stable addresses, and recovery. |

The smallest usable baseline can be a supported parallel/offload layout when
the model does not fit one device. Mock external dependencies and inject
failures as needed; separately exercise the real loader, scheduler, and core
computation being claimed.

## Numerical and temporal quality

Choose exact or tolerance criteria before running. Report complete-output
MAE, MSE/RMSE, maximum absolute error, and any PSNR/SSIM definition, data range,
frame count, and aggregation. Zero MSE yields infinite PSNR.

Locate the first divergence through noise draws, prompt tokens/cropping,
camera conditions, timestep/sigma, x0/re-noising precision, clean KV, and
decode. An integer timestep label can repeat and is not a substitute for the
reference sigma lookup.

Cross every state boundary touched by the change and inspect error versus
chunk index plus the full temporal result. Finite tensors, expected shapes,
and opening frames are smoke checks. Establish A/A variation when the
baseline drifts; interpret TP/SP differences against its direct-path baseline.
Qualify approximate decoders, quantization, and caches independently.

## Failure and resource checks

Inject faults at relevant prepare, denoise, clean-commit, decode, metadata,
and transport boundaries. Inspect engine/model sessions, KV references, VAE
history, and request-owned Futures/executors on every rank after release.
Exercise abort during hold/pacing waits and update/timeout races where relevant.

Confirm reset or new-request recovery using the current protocol, including
the declared A/B/A output criterion. Distinguish legitimate resident model or
graph caches from session leaks. Zero session counts alone do not establish
long-running memory stability.

## Performance protocol

Measure only requested, quality-qualified configurations. Start with a small
A/B and recheck the baseline; use A/B/A or interleaved AB/BA when variation is
material. Warm up actual geometry, prompt buckets, audio phases, graphs, and
allocator lifetimes. Changing-context chunks are not independent repetitions.

| Metric | Required boundary and context |
| --- | --- |
| SDK E2E | Call start to complete CPU output; state whether loading, compile, codec, and disk I/O are included. |
| API E2E | POST to complete MP4 bytes or first decodable chunk; include polling/network conditions. |
| Streaming | First chunk, subsequent per-chunk distribution, finalization, and evidenced explanations for stalls. |
| Generation rate | Actual delivered frames divided by the declared time; distinguish playback FPS and concurrent throughput. |
| GPU memory | Per-rank allocated/reserved, measurement stage, and collection tool. |
| Host memory | RSS/PSS, process-tree scope, and sampling interval, reported separately from GPU memory. |
| Quality | Accuracy and full-content evidence for the same configuration. |

Handle asynchronous device work and the multi-rank critical path. Do not sum
overlapped decode/commit stages. Keep instrumented dispatch/profiling runs
separate from clean timing. Place hashing, memory RPC, codec, and disk work
outside the timer unless the requested metric includes them. Match official
and Omni per-request cache clearing and warmup lifetimes.

Report repetitions, raw samples, variation, and percentile method. Speedup is
`baseline / candidate`; latency reduction is `1 - candidate / baseline`.
With five samples, nearest-rank p95 is the maximum. Treat gains within
observed baseline drift as preliminary evidence.

## Contribution evidence

Reuse existing tests, examples, and benchmarks. Keep one-off artifacts outside
the source tree according to checkout rules. Bind results to their tested SHA,
inputs, and extra patches, then revalidate affected behavior after fixes.

Use [add-recipe](../../add-recipe/SKILL.md) for setup and qualification
instructions and [precheck-pr](../../precheck-pr/SKILL.md) for submission.
Include numerical metrics and generated comparison clips, scoped performance
tables, per-GPU memory, and remaining gaps. Case-study measurements are
historical evidence, not results of running this skill on a new model.
