# Performance and Accuracy Verification

Activate this reference when the PR adds a model or claims or intentionally
changes latency, throughput, memory, scaling, precision, or output quality.

Official docs: [profiling](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/profiling/),
[serving benchmarks](https://docs.vllm.ai/projects/vllm-omni/en/latest/cli/bench/serve/),
and [metrics](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/metrics/).

## Comparable A/B contract

Compare frozen base and head for an existing path. For a new model absent from
base, compare head with a pinned canonical reference implementation. Use the same:

- hardware, driver, software versions, model/checkpoint, and dependencies;
- input or dataset, seed, precision, batch/concurrency, topology, and feature
  flags;
- warmup, measured repetitions, synchronization, timing scope, and memory
  collection method.

Keep exact commands and report variability rather than a cherry-picked run.
Use the repository's benchmark when it covers the claim; otherwise use the
smallest reproducible workload that reaches the changed production path.

## Isolation per claim

Isolate one claim per comparison. When a PR stacks several changes (a fused or
restructured load path, a distilled step schedule, a new cache), request one
column per claim at equal steps/workload instead of one blended base-vs-head
delta, and name the unchanged stage or metric that pins equivalence.

When the changed path is runtime-switchable (feature flag or environment
variable), run every column on the same head and toggle the switch rather than
diffing against a base commit that cannot separate the stacked changes:
base, head with the switch off, head with the switch on, then head with the
switch on plus each further claim.

## Evidence table

| Dimension | Typical evidence |
| --- | --- |
| Latency | End-to-end latency and, when available, TTFT plus a defined per-stage and transfer-time split. |
| Throughput/scaling | Requests, tokens, frames, or audio duration per second across the claimed concurrency/topology. |
| Memory | Peak allocated/reserved device memory and OOM boundary if relevant. |
| Quality/accuracy | Repository metric or known-good output comparison with an explicit tolerance; paired samples when no metric exists. |

Every optimization needs correctness or quality evidence. A faster result on a
different workload, precision, seed, or topology is not a valid comparison.

For diffusion accuracy evidence, use the reference-alignment and calibration
contract in [diffusion-accuracy-test](../../../diffusion-accuracy-test/SKILL.md)
to assess the claimed comparison scope and threshold evidence.

## Required output artifacts by modality

Provide the actual output files or responses and the applicable raw benchmark
records for each model, modality, and serving path covered by the claim. Bind
them to the revisions, checkpoint, input, seed, and runtime configuration. For
a comparison, provide both sides. Apply every applicable row for multimodal
claims. For a shared change, name the affected model families, explain the
representative cases, and list untested combinations.

| Output | Required quality or correctness artifacts | Performance artifacts when claimed |
| --- | --- | --- |
| Text and multimodal understanding | Actual returned text or response payload with the input and task score or parity check. | TTFT, decode latency or token throughput at stated input and output lengths. |
| Image generation or editing | Decodable generated or edited image files, conditioning inputs, and the applicable quality check. | End-to-end latency and peak memory at matching dimensions and steps. |
| Video and audio-video generation | Playable generated video files with frame count, FPS, and duration. Audio-video requires the generated audio track and alignment check. | End-to-end latency and peak memory at matching frames, resolution, and steps. |
| Text-to-speech | Generated audio and the exact synthesis text. For streaming, include received chunks and the final audio. | End-to-end latency and real-time factor. For streaming, include the first-packet and continuity measures below. |
| Multimodal Omni | Actual returned text and generated audio. Include the actual files for each additional claimed output modality. | End-to-end time and the applicable first-token, first-audio, and stage timings. |
| Music or sound generation | Generated audio with sample rate, duration, and the applicable quality check. | End-to-end latency and real-time factor. |
| Multi-stage or interactive outputs | Final outputs and the relevant stage or session trace, including terminal events. | End-to-end time and the applicable stage, handoff, or per-tick timings. |

Give the reviewer access to the actual artifacts through attachments or stable
links. Open the media or response and confirm its run identity. Hashes, metric
summaries, and screenshots can supplement the required files. A missing or
inaccessible required artifact triggers the zero-score rule in
[review-score.md](review-score.md#hard-scoring-requirements).

## E2e stage-attribution table

For an end-to-end latency claim, request the stage-attribution table the
repository's perf PRs use:

- Rows carry the benchmark's own field names: total e2e, per-stage execution,
  postprocess, response encoding, and a request-to-output roll-up. Add a stage
  handoff/idle row when the pipeline spans stages — connector transfer and
  scheduling gaps are invisible in per-stage latency deltas.
- Sanity rows pin comparability: inference steps, output frames/pixels/samples,
  and generated tokens held identical across columns.
- Report P50 and P100 from the stated warmup and measured counts. One
  population per number: never mix cold-start and warm samples into one value.
- Name the baseline and candidate commit SHAs, and close with an isolation
  statement naming the metric that stayed unchanged ("X remained effectively
  unchanged, isolating the gain to Y").

When the serving path caches, captures, or regenerates at runtime (CUDA-graph
bucket growth, cache rebuilds, recompilation), medians hide the resulting tail.
Also request the event count and the maximum observed stall.

Component-level tables pair with, not replace, the e2e table: per-op reference
vs candidate timings (with the numerical-equality result per op) plus one
complete-pipeline check under the production shapes.

## Proportionality

When a PR adds a model and explicitly labels its numbers as smoke observations
rather than claims, do not demand the attribution machinery. Ask only for
evidence protecting the model's stated operating contract — for a
realtime/duplex model that is sustained cadence over a realistic session
(enough ticks to leave warmup), with any mid-session capture/regeneration
event and its stall reported, since a cadence that collapses mid-session is a
functional failure, not a perf claim.

## Realtime/duplex world models

Request the operating-contract evidence as serving metrics, not claims:

- Sustained per-tick latency over one realistic session: median, max, and the
  argmax tick.
- Stall events with their causes.
- Peak memory at first versus last tick — session-owned state must not grow
  per tick.
- Control latency from a submitted action to the tick that applies it.
- Realtime-versus-offline parity under the same controls, using the
  repository's video-similarity helpers. Parity plotted over tick index is the
  drift-over-horizon curve and the honest measure of session lifetime;
  distribution-level suites (FVD, VBench) belong to model evaluation, not the
  serving integration.

Watch the contract boundaries the model documents: if a control input is
tick-API-only, the parity pair must use a horizon both paths can run.

## Speech-generation (TTS) models

The streaming contract outranks the average — a synthesis that pauses
mid-stream is broken for realtime use regardless of its averages. Request:

- The benchmark's own speech metrics over one long utterance (enough audio to
  leave warmup, tens of seconds): `audio_ttfp` (time to first packet),
  `audio_rtf` (real-time factor; above 1.0 cannot keep up), `e2el`, and
  `ttft`/`tpot` where tokens precede audio.
- Mid-utterance stalls as the benchmark's continuity fields, not prose:
  `audio_underrun` percentiles and the `audio_continuity_ok` rate
  (`vllm_omni/benchmarks/metrics/metrics.py`), computed by
  `vllm_omni/benchmarks/audio_continuity.py` — the companion to `audio_ttfp`
  and `audio_rtf`. A synthesis that bursts chunks fast enough on average but
  underruns mid-stream fails the streaming contract even when the averages
  pass.
- Quality as a paired comparison against the reference implementation under
  the same text, speaker prompt, and seed; report sample rate and frame
  count, and assert output completeness — truncated or dropped trailing audio
  is a known failure class, not a quality nuance.
- For realtime or duplex TTS: interruption (barge-in) latency and the
  per-session memory trend across turns.

## Omni models

Apply the e2e stage-attribution table directly to the voice-to-voice path of
omni (speech-to-speech and multimodal AR) models:

- TTFT for text output, `audio_ttfp` for audio output, and per-stage
  ITL/TPOT cadence across the perception-understanding-generation chain.
- Cross-modal parity — the omni analogue of realtime-versus-offline parity:
  the same semantic request through text and audio inputs, compared on output
  content or timing.
- For duplex omni sessions: turn-taking latency, interruption handling, the
  concurrent session count actually validated, and a per-turn memory trend —
  session-owned KV state must not grow per turn. When either turn produces
  audio, continuity is judged with the same `audio_underrun` and
  `audio_continuity_ok` fields as TTS, since turn-taking latency alone does
  not expose mid-utterance underruns.

## Reviewer verification ladder

1. Run base/head A/B on suitable hardware when affordable.
2. If only one side can run, verify the script and request a comparable pair.
3. If hardware or assets are unavailable, audit methodology, code path, and
   contributor evidence; name the exact unverified claim.

Classify discrepancies before reporting: implementation regression, benchmark
bug, environmental drift, noise, or unsupported claim. Explain material
regressions against the PR's stated goal or repository contract; do not invent
universal percentage thresholds.

Report claimed and measured values together, bind them to the frozen SHAs and
environment, and keep unavailable hardware as a validation gap rather than a
fabricated pass.
