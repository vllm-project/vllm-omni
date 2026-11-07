# Bound Models

## Contents
- RTF definitions
- Closed-loop capacity (Little's law)
- Shared-device throughput bound
- Low-concurrency pipeline bound
- Per-stage physical bounds
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

Measure device peaks on the target GPU (`scripts/measure_peaks.py`); do not
use data-sheet numbers.

Autoregressive decode at batch `B` is bandwidth bound while
`B * flops_per_token / bytes_per_step` stays below the device's
FLOP-per-byte ridge:

```text
step_time >= (weight_bytes + B * kv_bytes_per_seq) / HBM_bandwidth
g_decode(B) >= step_time / B * tokens_per_audio_second
```

`tokens_per_audio_second` is the codec rate for a talker (codec tokens per
second of audio) and text tokens per audio second for a thinker.

Compute-bound stages (flow matching, vocoder, encoders):

```text
g_compute >= flops_per_audio_second / peak_flops(dtype)
```

Count FLOPs with the torch profiler (`with_flops`) on that stage only, or
analytically from layer shapes, sampling steps and CFG duplication. Use the
peak for the dtype that actually runs (TF32 versus IEEE FP32 differs by an
order of magnitude on many GPUs).

## Reading the ledger

- `sum_i g_i_measured` versus `N / X`: the difference is idle time.
- `g_i_measured / g_i_bound`: how far a stage is from its roofline. Above
  about 3x, kernel or batching work is likely to pay; near 1x, only
  algorithmic changes (fewer steps, smaller model, lower precision) help.
- Large `g_i` with low batch reached: raise admission or graph batch
  limits before touching kernels.
