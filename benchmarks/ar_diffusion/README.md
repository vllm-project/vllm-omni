# Realtime AR-Diffusion session benchmark

Multi-session load harness for realtime AR-Diffusion serving: session arrivals,
playout deadlines, and the metrics a streaming video deployment is actually
gated on.

## Load modes

| Mode | Question it answers | Deadlines |
|---|---|---|
| `saturating` | How fast can this configuration go at all? Aggregate generated FPS, frames per GPU-second. | none |
| `paced` | Does it keep up with playback? CPR, TTFC, worst-case per-chunk latency. | yes |

Running only `saturating` produces a throughput number no deployment can use.
Running only `paced` hides the ceiling, so a deadline miss cannot be attributed
to insufficient capacity rather than to scheduling. Report both.

## Load parameters

Session **arrival rate** is the only valid open-loop axis. Per-session tick rate
is not: ticks of one session are strictly ordered and state-dependent, so tick
`k + 1` cannot be issued before chunk `k` returns and raising the rate only
grows a queue. Concurrently active sessions are derived state, reported
alongside every measurement as `peak_concurrent_sessions`.

## Metrics

Per session: `ttfc_s`, chunk latency P50/P95/P99, and

```
RTF = wall(K chunks) / (K * frames_per_chunk / target_fps)
```

Aggregate: `continuous_play_ratio` (CPR, the fraction of chunks delivered before
their playout deadline, macro-averaged over sessions), `worst_case_chunk_latency_s`
across all active sessions, `generated_fps`, and `frames_per_gpu_second` — the
only unit comparable across replica counts, TP degrees and hardware.

CPR is not redundant with RTF. RTF is a ratio of aggregates: a session can hold
`RTF = 0.9` over a minute and still stutter visibly, because chunks arrive in
bursts that individually miss their playout instants.

Deadlines assume a declared playout grid. Playback starts once `--buffer-chunks`
chunks are buffered; from that instant the player consumes continuously, so

```
deadline(i) = t_ready(buffer_chunks - 1) + cumulative_frames(i) / target_fps
```

Chunks inside the prebuffer have no deadline — their cost is start latency,
which TTFC reports. Walking cumulative frames rather than multiplying a constant
period is what lets a deeper buffer grant real slack, and what keeps a causal
decoder's shorter opening chunk from being credited with a full period.

## The vertical path

A session's tick returns latents, which nothing can watch. `DecodingSession`
wraps a latent-producing session so each tick also decodes its chunk
incrementally, which puts three things inside the measurement:

- **TTFF rather than time-to-first-chunk.** The gate this RFC proposes is time
  to first *frame*, so the decode has to be on the measured path.
- **`decode_share`**, the fraction of chunk wall time spent decoding. With
  generation and decode serialized this is the headroom overlapping them can
  recover; once they overlap it is the number that has to shrink. Overlap work
  without this baseline has nothing to beat.
- **`resident_decoder_bytes_per_session`**, which admission needs and which no
  other part of the run reports.

Delivered frames then come from the decoder rather than from the profile, so
the harness *measures* the temporal geometry instead of assuming it. Sessions
that only produce latents keep working: the stage split is reported as `None`
and the profile supplies the frame counts.

## Model neutrality

The chunk shape comes from the pipeline's declared `ARDiffusionKVCacheSpec`. No
model name appears in this directory.

One conversion is not derivable from the capability today: every frame count in
the spec is in *latent* frames and nothing converts them to delivered frames, so
the playout grid cannot be computed from the spec alone. Supply
`--vae-temporal-factor` (default `1`, correct for a decoder that does not
compress time).

**The conversion is not a multiplication.** A causal video decoder expands a
session's first latent frame to one raw frame and every later one to the full
factor, so `n` latent frames become `(n - 1) * factor + 1` raw frames the first
time and `n * factor` every time after:

| | latent frames | raw frames |
|---|---|---|
| chunk 0 | 3 | **9** |
| chunk k > 0 | 3 | 12 |
| K chunks | 3K | **12K − 3** |

So `chunks x frames_per_chunk` over-counts every run. The profile carries
`frames_per_first_chunk` alongside `frames_per_chunk`, and both `generated_fps`
and `RTF` sum the per-chunk delivery. Pass `--non-causal-decoder` for a decoder
that expands every latent frame identically.

This is also why a single declared integer could not close the gap: the missing
piece is a *mapping*, not a factor.

## Usage

```bash
# Ceiling
python -m benchmarks.ar_diffusion.run_realtime_benchmark \
    --model <checkpoint> --prompt "..." \
    --num-sessions 1 --mode saturating --chunks 32 \
    --vae-temporal-factor 4 \
    --note "checkpoint=<...>" --note "hw=1xH200" --note "res=480x832" \
    --note "steps=4" --output runs/n1.json

# State concurrency N, execution concurrency 1
python -m benchmarks.ar_diffusion.run_realtime_benchmark \
    ... --num-sessions 2 --output runs/n2.json
```

At least one `--note` is required: a latency without model, checkpoint,
resolution, denoising steps and hardware is not a result.

## Reading N=1 against N=2

With state concurrency `N` but execution concurrency 1, ticks serialize, so:

```
aggregate generated FPS(N)  ~=  aggregate generated FPS(1)
per-session RTF(N)          ~=  N x RTF(1)
```

`compare_runs()` reports `generated_fps_ratio` and `per_tick_switching_cost_s`,
the latter being latency beyond what pure queueing predicts. A measurable drop
in aggregate FPS is per-tick switching cost — state bind/unbind, KV pool paging,
conditioning rebuild — and that number is the baseline any cross-session
batching work has to beat.

## Tests

`tests/diffusion/ar_diffusion/test_realtime_benchmark.py` drives the whole load
model on a virtual clock against fake sessions: CPU only, no engine, no device,
no checkpoint. `engine_binding.py` is the only module not covered.
