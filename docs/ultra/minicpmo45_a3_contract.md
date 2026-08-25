# `vllm-omni-v0-ultra` M0 Contract and Timeline

Status: **已实现；本地静态与隔离验证已通过**. Full repository pytest still
requires a runnable vLLM/Torch environment. This document freezes the M0
diagnostic contract; it does not claim an A3 speedup, complete quality pass, or
submission readiness.

## Frozen competition baseline

| Field | Value |
| --- | --- |
| Integration branch | `vllm-omni-v0-ultra` |
| M0 branch | `ultra/00-contract-timeline` |
| Upstream base | `vllm-project/vllm-omni:minicpm-challenge@4105c717fe9fdab70285f4d23036768b7814ba78` |
| Target model | `OpenBMB/MiniCPM-o-4_5` |
| Target hardware | one Atlas A3 / Ascend 910C |
| Target image | `quay.io/ascend/vllm-omni:v0.25.0-a3` |
| Competition rules | submission guide revision 12, checked 2026-08-21 |
| Scored workload | Seed-TTS Chinese through `/v1/chat/completions`, concurrency `c=1`, two warmup requests |
| Compatibility workload | current official JSON's Seed-TTS English `c=1`, 32 prompts |
| Non-scoring guardrails | `c=4/8` success, continuity, memory, and mean/P95 regression checks |
| Official config | `tests/dfx/perf/tests/test_minicpmo_4_5.json` |
| Existing default pipeline | async chunking; Stage 0/1 PIECEWISE graph; Stage 2 eager; 25 codec frames with 3 left-context frames |
| Official score order | mean audio RTF first, mean `audio_ttfp` second, mean TTFT third |
| Official score inputs preserved by M0 | runner-produced TTFT, `audio_ttfp`, audio RTF, and existing aggregation |

Before a formal A3 run, re-check the upstream branch SHA, target image, model
weights, test script, and current competition rules. A rules or environment
change requires a new evidence run rather than overwriting an old one.

The submission guide and the current pinned performance JSON are not identical:
the guide specifies Chinese Seed-TTS at `c=1`, while the JSON currently contains
English `c=1/4/8` sweeps. Formal optimization decisions therefore use Chinese
`c=1` as the scored workload and repeat the JSON's English `c=1` cell as a
compatibility check. The candidate source is installed with
`pip install -e . --no-build-isolation`, but deploy config and server arguments
come from the official baseline. A candidate-only YAML change is not evidence of
an effective competition optimization.

## M0 hypothesis and scope

**Hypothesis:** a default-off, host-only request timeline makes the
client-visible latency path and existing server stage snapshots diagnosable
without changing inference semantics.

M0 changes only the benchmark client and documentation. It does not change the
public API, prompt construction, sampling, dtype, codec chunk policy, model
state, deployment YAML, official result JSON, or metric aggregation.

`audio_ttfp` intentionally retains its historical behavior: it is assigned on
the first SSE item with `modality == "audio"`, including an empty or malformed
audio payload. The timeline adds a separate
`first_nonempty_decodable_pcm` diagnostic only after a non-empty WAV packet has
successfully been parsed and yielded PCM frames. It is not an official score
replacement.

## Enable diagnostics

The baseline path is used when `VLLM_OMNI_ULTRA_TIMELINE` is unset or false.
That path does not create a recorder, open files, request stage metrics, invoke
the timeline clock, synchronize a device, or materialize a tensor.

For an A3 evidence run, keep raw evidence outside Git:

```bash
export BENCHMARK_DIR=work/perf-evidence/<run-id>/benchmark
export VLLM_OMNI_ULTRA_TIMELINE=1
export VLLM_OMNI_ULTRA_TIMELINE_DIR=work/perf-evidence/<run-id>/timeline
```

`VLLM_OMNI_ULTRA_TIMELINE_PATH=/path/to/events.jsonl` overrides the directory
setting. If neither destination is supplied, the opt-in sink is
`./ultra-timeline/events.<pid>.jsonl`. Use the directory form for
multi-process evidence so every process gets its own PID-named JSONL file.

Set `VLLM_OMNI_ULTRA_TIMELINE_CAPTURE_RAW=1` only when the evidence location is
access-controlled. It writes raw SSE fragments and PCM sidecars under
`raw/`; the JSONL normally stores only byte counts and SHA-256 digests. Prompts,
generated text, audio payloads, and raw PCM are never embedded in the JSONL.

Normal JSONL records are buffered per request and flushed after its terminal
event, so diagnostic file I/O does not run between streamed messages. Explicit
raw capture is intentionally more intrusive and is for diagnosis only, never a
formal scoring run.

Enabling the timeline also requests the existing
`return_stage_metrics=true` response field for `openai-chat-omni` and
`daily-omni` benchmark runs. This is an explicit diagnostic-mode wire change;
leaving the flag unset restores the baseline request payload.

## Event schema and timing interpretation

Every JSONL object contains these fields:

```text
schema_version, pid, seq,
request_id, turn_id, chunk_id, stage, event,
monotonic_ns, stream, shape, bytes, error
```

Optional fields are metadata-only: `sha256`, bounded `details`, and, when raw
capture is explicitly enabled, a relative `raw_path`. `monotonic_ns` is a host
timestamp and is not a device-kernel duration.

| Event | M0 meaning |
| --- | --- |
| `request_start`, `request_sent`, `request_finished`, `request_error` | Client request lifecycle and retry outcome |
| `sse_fragment_received`, `sse_message_received` | TCP/SSE framing boundaries, with payload size/hash only by default |
| `stage_metrics_observed` | Existing server-provided aggregate snapshot was observed; not a new internal clock |
| `first_text_token` | First client-observed text token used by the existing TTFT path, once per HTTP attempt |
| `first_audio_sse` | First audio-modality SSE per attempt, even if empty or malformed; mirrors the historical `audio_ttfp` endpoint |
| `audio_decode_error`, `audio_decode_deferred` | Invalid WAV, incompatible WAV parameters, empty PCM, or deferred non-WAV decoding |
| `first_nonempty_decodable_pcm` | First non-empty, parsed WAV PCM per attempt; diagnostic TTFP endpoint |
| `first_nonempty_decodable_pcm_deferred` | Non-WAV encoded audio observed; no PCM claim is made until a decoder is added |
| `last_audio` | Last non-empty audio payload of the successful attempt, with whether it parsed to PCM |

M0 intentionally does **not** claim exact `first_codec_token`, connector
`put/get`, CFM, HiFT, H2D/D2H, or NPU-kernel timestamps. Those cross-process
boundaries require later default-off hooks in the Stage 1/2 bridge and Code2Wav
implementation. Until then, `stage_metrics_observed` is the only server-side
coarse timing evidence and must not be relabeled as those internal events.

## Validation and promotion

M0 CPU checks cover disabled behavior, schema, payload privacy, raw sidecars,
write failure isolation, TCP-fragmented SSE, stage-metric opt-in, the legacy
empty-audio `audio_ttfp` behavior, invalid WAV, and valid PCM detection.

The next performance PR may be created only after M0's local checks pass. A3
promotion requires clean service restarts, fixed inputs and the alternating
`B-C-C-B-B-C` sequence with at least three repetitions per arm and the official
two-request warmup. Promotion follows the competition's lexicographic score:

- Prefer a `c=1` mean audio RTF improvement of at least 2%, with TTFP and TTFT
  regressions no larger than 1%.
- A TTFP-focused change may advance only when the RTF confidence interval is
  non-inferior within 0.5% and TTFP improves by at least 5%.
- A TTFT-focused change may advance only when RTF and TTFP remain non-inferior
  and TTFT improves by at least 3%.

At `c=4/8`, mean/P95, success, continuity, and memory are regression guardrails,
not competition score targets. Success rate, streaming continuity, and
decodable-audio rate must remain 100%, and peak memory may not increase by more
than 5% by default.

Numerical experiments remain isolated from the integration branch until they
also pass VideoMME, Daily-Omni, ASV, WER, Demo, and stability gates. The
conservative gates combine the stricter value from the submission guide and the
current repository tests: Daily-Omni >= 78.0%, Video-MME >= 68.0%, Seed-TTS
ASV SIM >= 0.689, and Seed-TTS WER <= 1.56%. The guide's reproduced F16
baselines are 79.5%, 69.0%, 0.709, and 1.414%, respectively. Re-validate the
guide revision and upstream SHA before every submission evidence run.

## Fallback and evidence hygiene

Rollback is immediate: unset `VLLM_OMNI_ULTRA_TIMELINE` and restart the
benchmark client. If the diagnostic code itself must be removed, revert the
single M0 commit and repeat the same baseline workload.

Keep each run in a new untracked directory such as
`work/perf-evidence/<timestamp>-m0-timeline/`, recording the commit, image,
hardware, model hash, benchmark command, environment, raw events, results and
quality outputs. Profiler runs are diagnostic-only and must not be mixed into
formal scoring samples.
