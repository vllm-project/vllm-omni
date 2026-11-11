# Calibrated Roofline Models

## Scope and terminology

This tool reports measured performance beside a **calibrated estimate**, not
an achievable target or a strict physical bound. The CLI `--*-bound` flags and
JSON `*_bound`, `bound_floors`, and `pct_of_sol` keys keep their legacy names;
`bound_kind=calibrated_estimate` states their interpretation. The report labels
these estimates and leaves mean RTF unavailable without per-request work.

Keep three different kinds of evidence separate:

- **Hardware theoretical bounds** require independently justified upper limits
  for each resource, unavoidable work counts, and valid dependency constraints.
  This tool does not certify such bounds.
- **Empirical calibration** from `measure_peaks.py` measures copy and GEMM
  throughput at particular sizes and dispatch costs on one host. Other shapes,
  kernels, clocks, or dispatch implementations can beat those measurements.
- **Implementation measurements** are durations of one implementation under a
  stated batch, shape, precision, and execution mode. They do not constrain
  all future implementations.

All estimates below assume the supplied work counts, cache policy, batching,
precision and serial dependencies. Measure peaks on the target device and
record shapes and conditions. A ratio above 100% calls for investigation of
calibration and coverage; it is not a physical impossibility.

## RTF definitions and closed-loop capacity

For request `r`, let `T_r` run from request start to last audio packet and
`d_r` be audio seconds:

```text
mean RTF      = mean_r(T_r / d_r)
aggregate RTF = sum_r T_r / sum_r d_r
```

In a closed loop with C requests actually in flight, Little's law gives
`C = lambda * mean(T)` and `X = lambda * mean(d)`, hence
`aggregate RTF = C / X`. Ramp-up, drain, failures and too few prompts break
this approximation. The report checks the residual; a small residual checks
population accounting, not the validity of the roofline model.

## Per-stage calibrated work

Count reference-implementation FLOPs, streamed weights and KV traffic; omit
embedding tables gathered by row unless tied to a streamed output head.
Record deployment precision, algorithm differences, and workload token rates.
Do not count unfused reference activation traffic as unavoidable traffic.
Reference FLOPs also describe an algorithm, not a proof that every equivalent
implementation must perform those operations.

For an autoregressive stage at batch B:

```text
memory = (weight_bytes + B * KV_bytes_per_sequence) / BW / B * tokens_per_audio_s
compute = FLOPs_per_token / P(dtype) * tokens_per_audio_s
```

For a compute row, use FLOPs/audio-s divided by its dtype's calibrated peak
and, if known, bytes/audio-s divided by bandwidth. Within a serial stage:

```text
g_device_i = sum_k max(compute_k, memory_k)
l_i = alpha(mode) * kernels_per_step * steps_per_audio_s / B
serial_i = max(g_device_i, l_i)
```

Each row must cover disjoint work. Use the same NAME for serial groups of one
stage, for example flow and vocoder rows named `code2wav@2`. The sum assumes
those groups do not overlap; it is not valid across independent streams.

`alpha(eager)` includes PyTorch dispatch for a tiny op. Graph replay amortizes
host launch across nodes. Neither is a universal per-kernel minimum. Profile
actual kernel or op counts, match the calibration's unit, and label launch
estimates as implementation-specific.

## Shared-device resource constraints

Stages on a shared device can overlap, including through CUDA MPS. Do not add
stage roofline durations as if each exclusively owns the entire GPU. Instead
accumulate normalized demand separately for each resource q:

```text
D_memory = sum of every stage's bytes/audio-s / BW
D_compute:dtype = sum of that dtype's FLOPs/audio-s / P(dtype)
D = max_q D_q
shared aggregate RTF estimate = C * D / N
```

Compute modes remain separate constraints. Without a device-specific mapping,
FP32 and tensor modes are not assumed to consume one serial pipe. This is a
conservative relaxation, not a claim that different dtypes always overlap.
The tool also keeps each stage's serial throughput constraint
`aggregate RTF estimate >= C * serial_i(C)`.

For example, 1 ms of compute in one stage and 1 ms of memory in another need
not consume 2 ms of exclusive device time. The resource constraint is 1 ms;
two serial rows in the same stage still require 2 ms. Two stages each demanding
1 ms of the same compute resource contribute 2 ms to that resource.

N denotes identical devices for the shared capacity relaxation. Stage serial
constraints still apply. Placement, replicas, heterogeneous devices and
communication need additional modeling before interpreting a deployment target.

## Fixed work, fill and aggregate estimates

`--fixed-bound` adds per-request work (prefill, encoders). Retain its compute
and memory demands separately: compute `F/P`, memory `weight_bytes/(BW*C)`.
`--fill-bound` adds only path latency for work already counted in streaming
rows. `@out` denotes the last stage's first audio chunk.

Fixed work must join the matching streaming resource **before** the maximum:

```text
W_q = sum of fixed work on resource q per request
shared estimate = C / N * max_q(D_q + W_q / mean(d))
lat_k = max(F_k/P, bytes_k/BW, alpha*kernels_k)
```

The aggregate estimate is the maximum of:

- the shared resource constraint above;
- each stage's throughput constraint `C * serial_i(C)`;
- each stage's average path estimate `gate_latency_i / mean(d) + serial_i(1)`;
- the first-packet path estimate `sum_k lat_k / mean(d)`.

Stage path estimates assume the supplied averaged work and serial gates. Drain
is omitted. The shared constraint is rendered as a cell-specific rate with
zero intercept; it already includes fixed work divided by that cell's mean
length. Do not interpret its intercept as an absence of prefill work.

### Why mean RTF is unavailable

Audio durations alone do not establish per-request token counts, contexts,
uncached work or chunk costs. A workload-average token/audio ratio cannot be
applied to each request, even at C=1. With durations `[1, 9]`, tokens `[1, 81]`
and 1 ms/token, aggregate RTF is 0.0082 but mean RTF is 0.005. Reusing 8.2
tokens/audio-s for both requests would incorrectly estimate mean RTF as 0.0082.

The current CLI accepts aggregate work counts, so `rtf_bound_mean` and
`pct_of_sol_mean` are omitted and the report explains the missing evidence.
A future per-request estimate must join each request's work to its duration,
compute its path, and only then average. Concurrency one does not remove
workload heterogeneity.

### Fitted fixed/rate split

If all requests obey `T_r = a + b*d_r` with common a and b, then

```text
a = (mean RTF - aggregate RTF) / (mean(1/d) - 1/mean(d))
b = aggregate RTF - a / mean(d)
```

The report retains this descriptive fit, skips nearly equal durations, and
warns on negative coefficients or incomplete duration coverage. Positive
coefficients do not validate the model. Correlated request work and queueing
can alter both coefficients; neither identifies a causal host/kernel split.

## First-token and first-packet estimates

TTFT includes encoder/prefill terms gating stage 0. TTFP adds Thinker steps
before Talker, codec steps before Code2Wav, and the first Code2Wav chunk:

```text
TTFT estimate = sum of lat_k gating stage 0
TTFP estimate = sum of all lat_k
```

Only sum proven serial dependencies. These are calibrated workload estimates,
not universal per-request guarantees. They omit queueing, transfer and most
host work. At any concurrency, measured minus estimate may include inefficient
kernels, additional work, host, transfer, and queueing. Slow kernels themselves
can cause queueing. Use traces and controlled experiments to attribute the gap;
C1 is a useful baseline, not a proof that kernel inefficiency is absent.

## Efficiency targets and profiled measurements

`--efficiency-target compute=F,memory=F` is a counterfactual, not a measurement
of achieved efficiency. Scale streaming resource demands separately, add fixed
work at full calibration, then take the maximum:

```text
projection = C / N * max_q(D_q / efficiency_q + W_q / mean(d))
```

This is a shared-capacity projection only; stage and latency estimates remain
separate. At efficiency 1 it equals the shared constraint, which need not be
the binding end-to-end estimate. Neither `measured - projection` nor
`projection - estimate` identifies causal overhead. An 80% target does not
show the kernels achieved 80%, and the residual cannot be assigned to host or
scheduling without profiling. A projection above the measured value is a
model/calibration discrepancy, not a measurement of achieved kernel efficiency.

`--profiled NAME=SECONDS_PER_AUDIO_S` displays both the calibrated roofline and
the supplied implementation duration, plus their maximum as a legacy
implementation estimate. A profile below the roofline may simply outperform
calibration; the measured duration remains evidence and is not invalidated.
This comparison does not enter the aggregate estimate or certify a hard floor.

No event scheduler is borrowed from external simulators. Serial stage rows
and resource-wise shared constraints account for different obligations;
contention and overlap need explicit modeling for a completion-time prediction.

## Sweep, capacity and attribution audits

`--series` flags decreases above 5% in throughput, aggregate RTF, TTFT or TTFP.
These are diagnostics to investigate, not universal monotonicity laws: batching,
cache state and contention can change efficiency as concurrency changes.

KV capacity is `floor(KV_tokens / (ceil(request_tokens / block) * block))`.
Use prompt plus output tokens at the workload's largest request. Capacity below
C can cause queueing or preemption; the startup max_model_len estimate is a
worst case rather than a workload count.

Utilization-derived `g_i = u_i / X` is an observed busy-time proxy. Summed
per-process busy fractions need not equal device busy time under overlap, and
missing readings invalidate absolute attribution. GPU idle time is a symptom,
not a measured host-cost decomposition. Stage wall time includes upstream waits
and must not be substituted for exclusive device time.

Report estimate/measured ratios and their assumptions, not guaranteed speedup
factors. Validate an optimization with repeated matched runs, quality checks,
and a trace tying the proposed change to the observed critical path.
