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
second of audio) and text tokens per audio second for a thinker.

Compute-bound stages (flow matching, vocoder, encoders):

```text
g_compute >= flops_per_audio_second / peak_flops(dtype)
```

Count FLOPs with the torch profiler (`with_flops`) on that stage only, or
analytically from layer shapes, sampling steps and CFG duplication. Use the
peak for the dtype that actually runs; reduced-precision float32 (TF32 on
CUDA, HF32 on Ascend) is a different peak from IEEE FP32.

That spread is device-specific: bf16 is about 15x IEEE fp32 on an L20X and
about 3.4x on a 910B1. A stage a CUDA ledger flags as precision bound can
already sit near its roofline on Ascend, so recompute the term instead of
carrying the conclusion across.

A stage has both terms; its device floor is the roofline maximum, not the sum:

```text
g_device_i = max(flops_i / peak_flops(dtype), bytes_i / HBM_bandwidth)
```

Pass the bytes of a compute stage (activations, weights per micro-batch) when
they are known; a stage whose memory term wins is not helped by lower
precision matmuls.

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

## Reading the ledger

- `sum_i g_i_measured` versus `N / X`: the difference is idle time.
- % of SOL per stage, `g_device_i / g_i_measured` (at most 100%): how close
  a stage is to its roofline. Below about 30%, kernel or batching work is
  likely to pay; near 100%, only algorithmic changes (fewer steps, smaller
  model, lower precision) help. Above 100% means an input is wrong.
- Cell % of SOL, `RTF_bound / aggregate_RTF`: the headline. Name the binding
  term next to it (shared device or one stage's serial floor) and that
  stage's bound class (compute, memory or launch).
- Large `g_i` with low batch reached: raise admission or graph batch
  limits before touching kernels.
- Launch-bound stage: raise graph coverage (capture sizes, graph batch
  limits) or fuse; kernel speedups will not move it.
