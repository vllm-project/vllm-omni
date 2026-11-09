# Bound Models

## Contents
- RTF definitions
- Closed-loop capacity (Little's law)
- Shared-device throughput bound
- Low-concurrency pipeline bound
- Per-stage physical bounds
- End-to-end RTF: fixed and per-audio-second cost
- First-token and first-packet latency
- Sweep and capacity audits
- Target efficiency and profiled completions
- Reading the ledger

## RTF definitions

For request `r` with generation time `T_r` (request start to last audio
packet) and audio duration `d_r`:

- per-request RTF: `rtf_r = T_r / d_r`; `mean_audio_rtf` averages it.
- aggregate RTF: `sum_r T_r / sum_r d_r`.

Short utterances weigh more in the mean. Under load the two can differ by
10% or more; state which one a target uses.

## Closed-loop capacity

`vllm bench serve --max-concurrency C --request-rate inf` keeps `C`
requests in flight. In steady state Little's law gives
`C = lambda * mean(T)`, and audio throughput is
`X = lambda * mean(d)`. Therefore

```text
aggregate RTF(C) = mean(T) / mean(d) = C / X
```

`X` is `audio_throughput` in the result JSON. The relation holds only while
concurrency is actually held: ramp-up, ramp-down, failures, and too few
prompts break it. `ceiling_report.py` checks the residual.

Consequence: at fixed `C`, lowering RTF is the same problem as raising
audio throughput. Once `X` saturates, `RTF(C)` grows linearly with `C`.

## Shared-device throughput bound

Let `g_i(B)` be GPU-seconds that stage `i` consumes per second of produced
audio when it runs at batch `B`. With all stages time-sharing `N` devices
and no idle time,

```text
X <= N / sum_i g_i(B)
aggregate RTF(C) >= C * sum_i g_i(B) / N
```

With one device per stage, the bound becomes `X <= min_i 1 / g_i(B)`; the
stage with the largest `g_i` binds and the others idle.

Measured `g_i` from utilisation: `g_i = u_i * wall / sum(d)`, where `u_i` is
the stage's busy share of the device during the measured window. Idle share
`1 - sum_i u_i` is the scheduling and host headroom.

`g_i(B)` falls with `B` until the stage saturates the device. A flat
`g_i(B)` means batching is exhausted for that stage; a falling curve that
the deployment cannot reach (capacity, graph batch limits, micro-batching)
is configuration or code headroom.

## Low-concurrency pipeline bound

At concurrency 1 the stages overlap through streaming hand-off. With
per-request busy time `t_i` at batch 1,

```text
RTF(1) >= max_i t_i / d + (fill + drain) / d
```

`fill` is the time before the slowest stage receives its first input;
`drain` is the work left after the upstream stage finishes. The per-stage
`stage_gen_time` printed by `--print-stage` is wall time inside the stage
(including waits for upstream chunks), so it is an upper estimate of `t_i`.
When `RTF(1)` is close to `max_i stage_gen_time_i / d`, the slowest stage's
single-batch latency is the binding term.

## Per-stage physical bounds

Measure device peaks on the target device (`scripts/measure_peaks.py`); do not
use data-sheet numbers, and do not reuse peaks measured on another
accelerator. See [npu-ceiling.md](npu-ceiling.md) for the Ascend variant.

Autoregressive decode at batch `B` is bandwidth bound while
`B * flops_per_token / bytes_per_step` stays below the device's
FLOP-per-byte ridge:

```text
step_time >= (weight_bytes + B * kv_bytes_per_seq) / HBM_bandwidth
g_decode(B) >= step_time / B * tokens_per_audio_second
```

`tokens_per_audio_second` is the codec rate for a talker (codec tokens per
second of audio) and text tokens per audio second for a thinker. `weight_bytes`
is what one step streams: every parameter except embedding tables, which are
gathered by row. A parameter sum overstates it; MiniCPM-o's Talker carries a
0.22 GiB text embedding table next to 0.36 GiB of streamed weights. Each token
also costs `flops_per_token / peak_flops(dtype)` whatever the batch, which
becomes the decode row's compute term at large batch.

Compute-bound stages (flow matching, vocoder, encoders):

```text
g_compute >= flops_per_audio_second / peak_flops(dtype)
```

Count work on the checkpoint's reference implementation
(`scripts/count_reference_work.py`), not on the engine being measured. The
reference defines the model's algorithm: its decode step, its streaming chunk,
its caches and overlaps. The engine's code path may add padding, recomputed
overlap or extra CFG rows, and a bound counted on it would call that waste
necessary. Take FLOPs, streamed weight bytes and KV bytes from the reference.
Do not take its activation bytes: it does not fuse, so they overstate what an
implementation must move. Use the peak for the dtype that actually runs;
reduced-precision float32 (TF32 on CUDA, HF32 on Ascend) is a different peak
from IEEE FP32.

That spread is device-specific: bf16 is about 15x IEEE fp32 on an L20X and
about 3.4x on a 910B1. A stage a CUDA ledger flags as precision bound can
already sit near its roofline on Ascend, so recompute the term instead of
carrying the conclusion across.

Each kernel has both terms, and its floor is the roofline maximum, not the
sum. A stage runs its kernels one after another on its own stream, so its
floor is the sum of per-kernel maxima:

```text
g_device_i = sum_k max(flops_k / peak_flops(dtype_k), bytes_k / HBM_bandwidth)
          >= max(sum_k flops_k / peak, sum_k bytes_k / HBM_bandwidth)
```

The stage-level maximum on the right is also a floor, but a looser one: it
lets a compute-bound kernel hide a memory-bound one that actually runs after
it. `ceiling_report.py` treats each bound flag as one row `k`, so split a
stage into kernel groups with different bound classes (Code2Wav's flow model
in TF32 and its vocoder in FP32; a decoder's weight-streaming GEMMs and its
attention) and give each its own flag. Rows must not share kernels: a decode
row and a compute row for the same GEMMs would count them twice.

Pass the bytes of a compute row (activations, weights per micro-batch) when
they are known; a row whose memory term wins is not helped by lower precision
matmuls.

### Launch term

Each kernel costs a launch latency `alpha` (measured by `measure_peaks.py`:
eager PyTorch dispatch per op, or per node of a replayed graph). A stage runs
one step at a time in its own process, so at batch `B`

```text
l_i = alpha(mode) * kernels_per_step * steps_per_audio_second / B
serial_i = max(g_device_i, l_i)
X <= min(N / sum_i g_device_i, min_i 1 / serial_i)
```

`steps_per_audio_second` is per sequence: codec tokens per audio second for a
talker, text tokens for a thinker, chunks times sampling steps for a flow
model. Launch time is host or front-end time, not device busy time, so it
enters the stage's serial floor and leaves the shared-device sum alone. It is
the term that binds at concurrency 1 and in stages that fall back to eager
above their graph batch limit; use `alpha(eager)` for those.

Count `kernels_per_step` from an nsys trace of that stage (kernels between two
step markers), not from the module tree; fused and library kernels differ.
For an eager stage, PyTorch ops per step is the closer count, since `alpha(eager)`
is per op.

### What the bound does not model

Kernel execution and launch latency only. Host scheduling, connector and
shared-memory transfer between stages, request queueing, chunk hand-off and
Python output processing are not priced. A cell whose measured RTF is far above
this bound while the device idles points at those costs; say so in the report
instead of attributing the gap to kernels.

## End-to-end RTF: fixed and per-audio-second cost

The per-audio-second terms above only bound how fast audio streams once it
flows. Every request also pays costs that do not grow with its audio: Thinker
prefill, the first Thinker token before the Talker starts, the first codec
chunk before Code2Wav starts, and host and transfer work. Model each request as

```text
T_r = a + b * d_r
rtf_r = b + a / d_r
mean RTF      = b + a * mean(1 / d)
aggregate RTF = b + a / mean(d)
```

`b` is the per-audio-second cost and `a` the fixed cost per request. The mean
weights `a` by `mean(1/d)`, which exceeds `1/mean(d)` (Jensen), so short
utterances make the mean RTF larger than the aggregate. The gap between the
two is exactly `a * (mean(1/d) - 1/mean(d))`.

### Measured split

The result JSON carries both RTFs and each request's audio duration
(`request_stage_metrics`), so the two equations solve for `a` and `b`:

```text
a = (mean RTF - aggregate RTF) / (mean(1/d) - 1/mean(d))
b = aggregate RTF - a / mean(d)
```

`ceiling_report.py` prints the split. It is a two-point solve: noisy when the
two RTFs differ by a few percent (C1), skipped when `1/d` barely varies, and
warned when fewer than 90% of requests carry stage durations or `a` or `b`
comes out negative. A per-request regression of `T_r` on `d_r` would be
sturdier, but the benchmark does not save per-request E2EL.

### Bound split

Per-request terms come in two kinds, each `max(flops / peak, bytes / bandwidth,
alpha * kernels)` in latency:

- **fixed**: work a request adds beyond its audio (Thinker prefill, encoders).
  One step reads the weights once for the whole batch, so the device work a
  request adds is `max(flops / peak, weight_bytes / (bandwidth * C))`.
- **fill**: latency before a downstream stage can start whose work is already
  in the per-audio terms (one Thinker decode step before the Talker, the first
  codec chunk of Talker steps before Code2Wav). It adds latency, not work.

Each term gates a stage: it must finish before that stage, and every stage
after it, sees its first input. A term gating `out` is the last stage's work on
the first packet. Three families of floors follow, each of the form
`rtf_r >= a / d_r + b`:

```text
shared device:      a = C * W / N,                     b = C * sum_i g_device_i(C) / N
stage j throughput: a = 0,                             b = C * serial_j(C)
stage j path:       a = sum of latencies gating <= j,  b = serial_j(1)
```

`W` is the per-request fixed work.

- **Shared device** is work conservation:
  `X <= N / (sum_i g_device_i + W / mean(d))`, so by Little's law
  `aggregate RTF >= C * (sum_i g_device_i + W / mean(d)) / N`.
- **Stage throughput**: one process runs its steps serially with at most `C`
  sequences per step, so `X <= 1 / serial_j(C)` and
  `aggregate RTF >= C * serial_j(C)`.
- **Stage path** is the critical path of one request: stage `j` cannot start
  before its gating terms finish, and then needs at least `serial_j(1)` per
  audio second. A step carrying this request never runs faster than at
  batch 1: weights are read once per step whatever the batch, its own FLOPs
  and launches are still paid.

```text
aggregate RTF bound = max over all floors of (a / mean(d) + b)
mean RTF bound      = mean_r max over per-request floors of (a / d_r + b)
```

Shared device and stage throughput are averages over requests, so above
C = 1 they bound the aggregate only. Do not write `b = C * serial_j(C)` into a
per-request path: it assumes every step of this request carries `C`
sequences, which a compute-bound stage running smaller batches beats. At
C = 1 all three hold per request. Drain after the binding stage is left out,
which keeps every floor a lower bound. `T_r` also ends after the first packet,
so the first-packet latency below is one more per-request floor with `b = 0`.

With no per-request terms, the aggregate bound reduces to
`max(C * sum_i g_device_i / N, C * max_i serial_i)`.

Reading it: compare the measured `a` with the largest floor `a`, and the
measured `b` with the largest floor `b`. A fixed-cost gap is prefill,
first-chunk scheduling, connector waits and host work, moved by chunk sizes,
admission and hand-off; a per-audio gap is kernels and batching. Under load,
`a` also absorbs queueing before a request's first step, which tracks
`AUDIO_TTFP`.

Prefix caching shrinks prefill FLOPs: count only tokens that can miss the
cache. With block size `k` and a shared template no longer than the shortest
prompt, at least `input_r - floor(min_input / k) * k` tokens per request miss.
The weight read stays, since one prefill step reads every weight.

## First-token and first-packet latency

The bench measures TTFT from request start to the first Thinker text delta,
and `AUDIO_TTFP` to the first audio packet. Both reuse the per-request terms:

```text
TTFT_r >= sum of latencies gating stage 0
TTFP_r >= sum of latencies of all terms (fixed and fill, gating any stage or out)
```

For a Thinker, Talker and Code2Wav pipeline with streaming chunks, the first
packet waits for, in order:

1. Encoders and Thinker prefill on uncached tokens (`--fixed-bound ...@0`); the
   first text token comes out of prefill, so these alone bound TTFT.
2. The Thinker steps the Talker needs before it can start (`--fill-bound ...@1`).
3. `codec_chunk_frames` Talker steps before Code2Wav sees a chunk
   (`--fill-bound ...@2`).
4. Code2Wav's work on that chunk (`--fill-bound ...@out`), one row per kernel
   group: the chunk is `codec_chunk_frames / codec_rate` seconds of audio, so
   its FLOPs are that times the per-audio-second count.

Each latency is `max(flops / peak, bytes / bandwidth, alpha * kernels)` at
batch 1. These are serial dependencies of one request, so they add; within one
term the roofline still takes the maximum.

The floor is the same at every concurrency: queueing is not a physical
floor. Read it at C = 1, where the measured value has no queueing. Under load,
`measured - bound` is queueing before the first step plus host, scheduling and
hand-off; a TTFP that grows with C while the C = 1 gap is small points at
admission and scheduling, not kernels.

## Sweep and capacity audits

Laws a concurrency sweep of one configuration obeys in a closed loop, checked
by `--series`:

- audio throughput does not fall as C rises: past saturation it stays flat;
- aggregate RTF `= C / X` does not fall;
- mean TTFT and TTFP do not fall, since more requests in flight only add
  waiting.

A violation beyond noise (5%) means the cells differ in more than C: warm-up,
prefix-cache state, configuration, or thrashing (preemption, eager fallback).
Fix the cells before reading the sweep.

Capacity: a stage's KV cache holds
`floor(KV_cache_tokens / (ceil(tokens_per_request / block) * block))`
requests. `KV_cache_tokens` is the `GPU KV cache size` startup log line, and
`tokens_per_request` is prompt plus output at that stage, at the largest
request of the workload. Below C, the cell measures queueing or preemption.
The startup line `Maximum concurrency for <max_model_len> tokens per request`
is the same formula at `max_model_len`, a worst case rather than the workload.

## Target efficiency and profiled completions

Two readings from [mindie-sim](https://github.com/mindie-agent/sim) sit beside the
bound. Neither replaces it.

### Target efficiency

`fit_efficiency_targets` asks what the time becomes when each pipe runs at a
stated fraction of its peak. With `η <= 1` a fraction of the measured peak:

```text
t_row(η) = max(compute_time / η_compute, memory_time / η_memory)
projection = C * sum_i t_row(η) / N + C * W / (N * mean(d))
```

The max is taken per row before summing, same as the device floor. At `η = 1`
the per-audio part equals the shared-device floor. Launch time does not scale,
and neither do the per-request fixed and fill terms: the projection says how
the streaming rows move. It is not a lower bound. An implementation can beat
it, down to the bound. Split the gap:

- `measured - projection`: time left even after every streaming row reaches the
  target. Host, scheduling, transfer, queueing, or work the rows do not count.
- `projection - bound`: the price of stopping at the target instead of the
  measured peak.

A projection above the measured aggregate means the cell already beats the
target, and the report warns.

### Profiled completion

mindie-sim treats a measured primitive duration as a completion constraint on
that implementation, and lets the physical program win when it is slower. The
same rule, per row:

```text
implementation floor = max(roofline, profiled)
```

The roofline wins when it is slower, because a measured duration does not
delete FLOPs or bytes. The floor constrains the profiled implementation only,
so it never enters the bound. A profile below its roofline means the work
count, the peak, or the profile's coverage is wrong.

### What is not borrowed

mindie-sim's event engine schedules shared pipes and predicts a completion
time, including overlap and contention. Contention only adds time, so the sum
of per-row rooflines is already the lower bound on device time; the schedule
would be a prediction. Rows are assumed not to overlap. Where two pipes of one
kernel do overlap, count the kernel as one row (see
[npu-ceiling.md](npu-ceiling.md)).

## Reading the ledger

- `sum_i g_i_measured` versus `N / X`: the difference is idle time.
- % of SOL per stage, `g_device_i / g_i_measured` (at most 100%): how close
  a stage is to its roofline. Below about 30%, kernel or batching work is
  likely to pay; near 100%, only algorithmic changes (fewer steps, smaller
  model, lower precision) help. Above 100% means an input is wrong.
- TTFT and TTFP % of SOL, bound / measured mean: read at C = 1; under load
  the gap is mostly queueing.
- Cell % of SOL, `RTF_bound / aggregate_RTF`: the headline. Name the binding
  term next to it (shared device or one stage's serial floor) and that
  stage's bound class (compute, memory or launch).
- Large `g_i` with low batch reached: raise admission or graph batch
  limits before touching kernels.
- Launch-bound stage: raise graph coverage (capture sizes, graph batch
  limits) or fuse; kernel speedups will not move it.
