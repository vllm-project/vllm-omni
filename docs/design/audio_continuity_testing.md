# Streaming audio continuity: definitions and validation

Audio continuity is a simulated FIFO player measurement. It describes delivery
stalls rather than waveform quality, audio-device behavior, or first-packet
latency. Read it alongside TTFP and RTF.

## Playback accounting

For chunk arrival `t_i` and decoded PCM duration `d_i`, playback starts at the
first nonempty chunk: `E_0 = t_0 + d_0`. For subsequent chunks:

```text
g_i = max(0, t_i - E_(i-1))
E_i = max(t_i, E_(i-1)) + d_i
D = sum(d_i)
audio_max_single_underrun_s = max(g_i, default=0)
audio_underrun_s = total_underrun_s = sum(g_i)
audio_underrun_event_count = count(g_i > 1e-9)
audio_underrun_ratio = total_underrun_s / (D + total_underrun_s)
audio_underrun_events_per_minute = audio_underrun_event_count * 60 / D
audio_continuity_ok = total_underrun_s <= threshold_s + 1e-9
```

Stalls at or below `1e-9` seconds are ignored. PCM duration is
`bytes / (sample_rate * sample_width * channels)`; WAV headers and base64 text
are excluded. The last chunk contributes to `D`.

The existing `audio_underrun_s` and `ContinuityStats.max_underrun_s` names retain
their cumulative-deficit meaning. The old maximum cumulative deficit equals
total playback stalls on valid timelines. The new single-stall field is
additive, so existing benchmark JSON and Prometheus dashboards remain
comparable. Continuity still uses the total budget; the default is 100 ms.

The benchmark environment variable
`VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S` configures this budget. Invalid
or nonfinite values use 100 ms; negative values clamp to zero. It does not
change the server-side default.

## Sample population and aggregation

Chat/Speech contributes one measured response per successful audio request.
Duplex contributes one sample per completed `response_id`, including multiple
responses on one connection. Each response has a fresh playback buffer.
First-packet wait, inter-response listening, and waiting for EOF or
`response.done` after the final PCM chunk are excluded. Gaps within a response
are included.

Failed, cancelled, incomplete, LISTEN-only, empty, malformed, or unsupported
encoded-audio responses are unmeasured. Unmeasured responses do not increase
the continuity denominator. With no samples, JSON stores `null` for the rate
and the CLI displays `N/A`; unavailable distributions are omitted.

Means and percentiles weight measured responses equally, including ratios and
events per minute. They are not duration-weighted session statistics.
`total_audio_underrun_events` sums the actual event counts. Each new metric has
its own sample count; older adapters missing new fields do not contribute
fabricated zero samples.

Server-side Chat/Speech streaming finalize uses the same helper and bounded
`model_name`, `stage`, `replica` labels. Response IDs are diagnostic JSON keys,
never Prometheus labels. Server output timestamps and benchmark receive
timestamps measure different boundaries; network buffering can separate them.
This change does not add full-duplex server lifecycle instrumentation.

## Deterministic regression coverage

The tests cover hand-computed timelines, exact rational playback schedules,
time-origin shifts, PCM sample rate/width/channel count, threshold boundaries,
invalid input, retry isolation, interleaved/cancelled replies, response-weighted
aggregation, CLI/JSON units, and once-per-stream Prometheus observation.

For arrivals `[.3,.45,.85]` and durations `[.1,.2,.2]`, the expected values are:
single stall `.2s`, total `.25s`, two events, ratio `1/3`, and 240 events per
audio minute. For `[0,.5,.6]` with three `.1s` chunks, there is one new stall,
not two observations of an already accumulated deficit.

From an installed development checkout:

```bash
python -m pytest tests/benchmarks/test_audio_continuity.py \
  tests/benchmarks/test_audio_continuity_correctness.py \
  tests/benchmarks/test_audio_continuity_realtime.py \
  tests/benchmarks/patch/test_audio_continuity_backends.py \
  tests/benchmarks/patch/test_benchmark_result.py \
  tests/benchmarks/metrics/test_metrics.py \
  tests/metrics/test_modality.py \
  tests/entrypoints/openai_api/test_serving_speech_metrics.py \
  -m "core_model and cpu" --run-level core_model
```

These deterministic tests need project test dependencies and a matching vLLM
installation, but no model weights, GPU, network, or wall-clock sleeps. The
hand-computed/helper suite can also run directly with Python and the existing
`pybase64` dependency:

```bash
python tests/benchmarks/test_audio_continuity_correctness.py
```

For real-service acceptance, retain per-chunk monotonic receive times and PCM
durations and recompute each response independently. Compare Prometheus
increments for the same labels, and interpret output-side and receive-side
measurements separately.
