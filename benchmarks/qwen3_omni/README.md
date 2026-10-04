# Qwen3-Omni Seed-TTS Realtime turn-trigger comparison

The performance configuration at
[`tests/dfx/perf/tests/test_qwen3_omni_seed_tts.json`](../../tests/dfx/perf/tests/test_qwen3_omni_seed_tts.json)
compares the two ways a Realtime turn can be ended for the same Seed-TTS audio
and target text:

| Case | Input and trigger |
| --- | --- |
| `explicit` | WebSocket `/v1/realtime?duplex=1`; send text and paced speech PCM, then `input_audio_buffer.commit` and `response.create` at the end of the speech |
| `vad` | Same text and speech PCM, followed by the silent tail; server VAD ends the turn and creates the response, without a client commit or `response.create` |

Both run on `qwen3_omni_duplex.yaml` with async chunking, temperature 0 for text
generation, a 256-token output limit, the first four English Seed-TTS entries in
fixed order, and concurrency 1. Each entry is an independent session carrying one
utterance. The standard benchmark warmup runs separately from the four measured
requests.

`--seed-tts-reference-as-input` sends the dataset's reference speech as actual
user audio, rather than as `ref_audio` voice-cloning metadata. The instruction
asks the model to read the target text, not transcribe or answer the audio.
Reference audio is normalized once to mono 24 kHz PCM16 with a one-second silent
tail (`SEED_TTS_SILENT_TAIL_MS`); both cases send the identical speech samples in
200 ms chunks at real-time speed.

**Only VAD streams the silent tail.** The tail exists so server VAD can detect
the endpoint, and its 800 ms silence threshold must stay shorter than the tail or
the turn never ends. An explicit client ends the turn itself when the speaker
stops, so making it stream a second of silence first would charge it for latency
no real caller would pay — and would understate how much the VAD endpoint
actually costs. VAD may trim audio at its detected speech boundaries before model
execution. The public VAD contract requires `interrupt_response=true` and
`barge_in_on_speech`; these flags are accepted, but this workload never sends
overlapping speech, so it does not exercise or measure interruption.

This checks turn-trigger performance for one response per input. It does not
measure interruptions, overlapping speech, cross-turn context, or load scaling.
A VAD split before the reference audio ends is rejected rather than included as
a successful single-turn comparison.

## Run

Use the repository environment and reserve GPUs according to your host rules.
On this shared NVIDIA host:

```bash
gpu run --gpus 2 --nonblock --timeout 3h --note "Qwen3 Seed-TTS triggers" -- \
  uv run --no-sync pytest -s -v tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_seed_tts.json \
  --run-level full_model
```

The checked-in CUDA nightly job runs both cases on two H100/B200 GPUs.
`BENCHMARK_DIR` selects the output directory (default `tests/dfx/perf/results`).
For local cached weights/data, copy the JSON and replace `server_params.model`
and every `benchmark_params[].dataset_path` with local paths. Preserve sample
count, order, prompt, and generation settings across cases.

## Read the measurements

Every timing below starts at the same client-side origin: immediately before the
text item is submitted, and therefore **before** the paced audio upload. Both
cases share that origin and the same input pacing, so they are comparable to each
other.

- **TTFT / audio TTFP:** origin to first text / first audio packet received. This
  includes the real-time audio upload and, for VAD, endpoint detection, so the
  absolute values are dominated by input delivery rather than by model latency.
- **E2EL:** the same origin to `response.done` reception.
- **Audio RTF:** origin to the last audio packet divided by output audio duration.
  It includes input delivery; it is not model-only compute RTF.
- **Audio duration:** generated audio length, received as PCM16.

To separate input delivery and endpoint detection from the remaining response
latency, read the per-request rows in `duplex_request_metrics`, which carry
utterance identity, trigger, `input_audio_ms`, `input_upload_ms`,
`session_setup_ms`, `input_content_end_to_first_audio_ms`,
`explicit_commit_to_first_audio_ms` (explicit only), and
`vad_stop_received_ms` / `vad_stop_to_first_audio_ms` (VAD only). The VAD pair
starts at the client's receipt of `speech_stopped`, not at a server GPU timestamp.

TTFT/TTFP here are client event timestamps. The server also attaches its own
`response_request_metrics` to the first text delta, measured from "accepted
native-append start" — a different origin, and one that is not defined for the
VAD case at all. On one explicit turn — the first request against a cold
server, so both numbers carry warmup — it reported `ttft_ms` 935 where the
client observed 5616 for that same response. The MiniCPM-o Seed-TTS duplex benchmark
(`test_minicpmo_4_5_duplex_seed_tts.json`) prefers those server values and
derives TPOT from Stage-0 engine metrics. **The two configurations do not share
a metric origin, so their TTFT/TTFP/RTF numbers are not comparable across
models.**

TPOT/ITL are not reported here, and cannot currently be: this path returns
`stage_metrics: {}` on every delta (verified against a live Qwen3-Omni duplex
server), so there is no engine token timing to read. `num_tpot_samples` is
therefore `0`, which is what a configured TPOT baseline would trip on rather
than silently comparing a value derived from `latency - ttft`.

Why the origin matters, recomputed both ways over the same four measured
requests (2x L20X, warm):

| RTF origin | `explicit` mean | `vad` mean | worst single request |
| --- | --- | --- | --- |
| end of reference speech (used here) | 0.13 | 0.21 | 0.24 |
| session start | 0.71 | 0.84 | 0.98 |

Timing from session start does not fail the `< 1` SLO outright, but roughly 85%
of what it reports is the client's own real-time upload, so the model's share is
diluted into a near-constant offset. Over these requests a hypothetical 100 ms
generation regression moves the reported RTF by 8.9% from the speech-end origin
and by 1.6% from session start — a 5.7x difference in how visible a regression
is. The worst measured VAD request already sits at 0.98 under the session-start
origin, with the margin consumed by upload rather than by the model, so a batch
of longer reference clips would cross the line for no model-side reason.

The test requires all requests to complete, exactly one completed response with
text and audio per request, and, for VAD, actual speech-start and speech-stop
events. No performance regression baseline is set: the numbers below were taken
on 2x L20X, not on the H100/B200 the nightly job targets, and four requests are
a small smoke test rather than a statistically stable latency study.

Reference run, 2x L20X, one warmup then four requests at concurrency 1:

| | `explicit` | `vad` |
| --- | --- | --- |
| Mean TTFT | 347 ms | 912 ms |
| Mean audio TTFP | 419 ms | 1035 ms |
| Mean audio RTF | 0.13 | 0.21 |
| Mean E2EL | 1213 ms | 1864 ms |

The roughly 600 ms that VAD adds is the endpoint it has to detect — the
`silence_duration_ms` wait that an explicit commit skips. That difference is the
measurement this configuration exists to produce. Note the spread within each
run (explicit median TTFT 54 ms against a mean of 347 ms): the first measured
request is still paying warmup, so one `--num-warmups` is not enough to read
per-request numbers, only the means.
