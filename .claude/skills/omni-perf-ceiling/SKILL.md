---
name: omni-perf-ceiling
description: Estimate the best achievable serving performance of a vLLM-Omni multi-stage omni model (Qwen3-Omni, MiniCPM-o, and similar thinker/talker/code2wav pipelines) and how much headroom the current implementation leaves. Use when asked for the lowest reachable AUDIO_RTF or first-audio latency at a given concurrency, the maximum audio throughput on given hardware, whether a model "still has optimization room", or which stage bounds a target. Produces measured, configuration-envelope, and physical-bound numbers with a per-stage headroom ledger, on CUDA GPUs or Ascend NPUs. Use `omni-perf-opt` afterwards to implement optimizations.
---

# Omni Performance Ceiling

Answer "how fast can this get, and where is the remaining gap" with three
numbers per target, not one:

1. **Measured**: the shipped configuration on the target hardware.
2. **Envelope**: the best value over configuration-only changes (capacity,
   graph budgets, batch knobs, MRv2/V1, MPS, placement), plus any existence
   proof (another engine or the reference implementation on the same
   hardware, workload, and output mode).
3. **Bound**: a physical floor derived from per-stage work and measured
   device peaks.

`envelope - bound` is code headroom; `measured - envelope` is configuration
headroom. Report both, per stage.

## First questions

Fill in what is not already known; the `omni-perf-opt` first questions apply.
In addition, fix:

- Target cells, for example `AUDIO_RTF` at concurrency 1 and 64.
- RTF definition: per-request mean (`mean_audio_rtf`) or aggregate
  (sum of generation time over sum of audio). They differ under load.
- Output mode: streaming chunks or a single final audio. Streaming adds
  first-chunk work; compare engines only within one mode.
- Device budget: one shared device, or one device per stage.
- Accelerator: CUDA GPU or Ascend NPU. On Ascend read
  [references/npu-ceiling.md](references/npu-ceiling.md) first; the MPS lever
  and per-process SM attribution are not available there.
- Whether numerics-changing levers (TF32, FP16 CFM, quantization) are inside
  the envelope. Default: report them in a separate column.

## Two regimes

Concurrency 1 is a latency problem; high concurrency is a capacity problem.
Do not use one model for both. Derivations:
[references/bound-models.md](references/bound-models.md).

- **Low concurrency**: stages pipeline, so
  `RTF(1) >= max_i t_i / audio + fill_and_drain / audio`, where `t_i` is the
  stage's per-request busy time at batch 1. If measured RTF(1) already
  equals the slowest stage share, scheduling has nothing left; only that
  stage's single-batch latency can move the number.
- **High concurrency**: closed-loop load obeys Little's law,
  `aggregate RTF(C) = C / audio_throughput`. The floor is therefore set by
  the maximum audio throughput:
  `audio_throughput <= N_gpu / sum_i g_i(B)`, where `g_i(B)` is GPU-seconds
  that stage `i` spends per second of audio at batch `B`. Measure `g_i`,
  bound each `g_i`, add them up. A stage also runs its steps serially in one
  process, so `audio_throughput <= 1 / max(g_i, l_i)` per stage, where `l_i`
  is its kernel-launch time per audio second. Small batches and eager
  fallbacks make that launch term bind before the device does.

Both regimes also pay a fixed cost per request. End-to-end,
`T_r = a + b * d_r`: `b` is the per-audio-second cost the terms above bound,
`a` is prefill, first-chunk fill before each downstream stage, host work and,
under load, queueing. So `aggregate RTF = b + a / mean(d)` and
`mean RTF = b + a * mean(1/d)`; a mean-RTF target pays more for `a` because
short utterances weigh more. Report the measured and bound `a` and `b`
separately: they move with different levers.

## Workflow

1. **Freeze the workload and commands.** Use the `omni-perf-opt`
   measurement templates. Record commit, hardware, deploy YAML plus overlay,
   output mode, dataset, prompt count, and `--extra-body`.

2. **Check admission before measuring capacity.** Every stage's
   `max_num_seqs`, session caps (`max_sessions` for duplex), and per-stage
   graph batch limits must be at least the target concurrency. Otherwise the
   high-concurrency cell measures queueing, not compute. Read startup logs
   for graph capture budgets and eager fallbacks at the target batch.

3. **Measure the cells.** Warm up with at least `C` requests, then run
   `10 x C` (minimum 128) prompts per cell with `--print-stage`,
   `--save-result --save-detailed`, and device utilisation sampling
   (`nvidia-smi --query-gpu=utilization.gpu -lms 500`, or `npu-smi info watch`
   on Ascend). Keep profilers off.

4. **Build the ledger.** Run
   `python .claude/skills/omni-perf-ceiling/scripts/ceiling_report.py`
   on the result JSON and bench log of each cell. It checks Little's law,
   derives aggregate RTF, per-stage time shares, GPU idle share, and the
   implied throughput floor. A Little's-law mismatch above 10% means the run
   did not hold concurrency (ramp, failures, or too few prompts); fix that
   before reading anything else.

5. **Attribute device time per stage.** Pick one:
   - one device per stage (`*_3gpu.yaml`): per-device utilisation is the
     stage's `g_i` directly;
   - CUDA only: one shared device with MPS disabled
     (`platforms.cuda.cuda_mps: false`) and `nvidia-smi pmon -s u`, giving
     per-process SM share. Ascend has no per-process AI Core counter;
   - stage isolation microbenchmarks at the target batch.
   Experiment recipes:
   [references/ceiling-experiments.md](references/ceiling-experiments.md).

6. **Explore the envelope.** Change one configuration lever per column
   (capacity, graph batch limits, micro-batch size, runner, MPS, replicas,
   placement). Keep the best column per cell. Quality must stay within
   run-to-run variance (`omni-perf-opt` quality gate) for the envelope to
   count.

7. **Compute bounds.** Run `scripts/measure_peaks.py` on the target device for
   memory bandwidth, dense matmul throughput and per-kernel launch latency
   (eager and graph replay). Give each stage three terms: compute (FLOPs per
   audio second from `FlopCounterMode` or analytic counts), memory (weights
   plus KV read per decode step, or bytes moved), and launch (kernels per step
   times steps per audio second, at the stage's eager or graph mode). The
   stage floor is `max(compute, memory)` on the device and `max(device,
   launch)` in its own process; the larger term names the stage's bound class.
   Write each term's formula with the numbers substituted so it can be
   re-checked, and state which inputs are measured and which analytic.
   Then add the per-request terms: `--fixed-bound` for work every request
   adds (Thinker prefill on uncached tokens, encoders) and `--fill-bound` for
   latency before a downstream stage can start (first Thinker step before the
   Talker, the first codec chunk before Code2Wav). The report turns them into
   aggregate and mean RTF floors of the form `a / d + b`.

8. **Correlate measured against bound.** Pass `NAME@STAGE` on the bound
   flags so `ceiling_report.py` joins each term to that stage's measured
   `g_i` and reports its % of SOL (bound / measured, at most 100%). A stage
   far below SOL with a compute or memory bound needs kernel or batching
   work; a launch-bound stage needs graph coverage or fewer, larger kernels.
   Measured RTF far above the whole bound, with high GPU idle, is host,
   scheduling or inter-stage transfer, which the bound does not price.
   Compare the measured fixed cost `a` and per-audio cost `b` (solved from
   mean and aggregate RTF) with the largest floor `a` and `b`: a fixed-cost
   gap points at prefill, first-chunk scheduling, connector waits and
   queueing; a per-audio gap points at kernels and batching.

9. **Report.** Fill the template below. Hand the largest gap to
   `omni-perf-opt` as its first candidate.

## Rules

- Do not quote a ceiling without the measured cell beside it on the same
  commit and hardware.
- Historical numbers are context, not evidence for the current revision.
- An existence proof must match output mode, reference audio policy,
  dataset subset, and RTF definition. Otherwise list it as context.
- Physical bounds are lower limits on time, not predictions. Name the
  dominant term and the assumptions it hides (perfect batching, no host
  overhead, peak bandwidth).
- Peaks and launch latency come from `measure_peaks.py` on the device in
  hand, never from memory or a data sheet. Structural inputs (weight bytes,
  KV bytes, codec rate, FLOPs) come from the checkpoint and the result JSON.
  A term that cannot be grounded is written as "not bounded: <reason>",
  never estimated to fill the column.
- A stage measured below its own bound (% of SOL above 100%) means the work
  count, the peak or the attribution is wrong. Fix the input before reading
  the ledger.
- Bound the RTF the target names. The aggregate floor uses `mean(d)`; the
  mean floor uses each request's `d` and, above C = 1, only the per-request
  critical path, so it can sit below the aggregate floor.
- Report the regime for each cell. Do not extrapolate a high-concurrency
  RTF from low-concurrency runs, or the reverse.
- A cell whose throughput is still rising with concurrency is not saturated;
  its aggregate RTF floor is the next cell's, not this one's.
- A ceiling belongs to one accelerator. Bounds scale with that device's
  measured peaks and the envelope with the levers it actually has, so never
  carry a CUDA ceiling, or a CUDA lever's delta, to Ascend or back.
- The best configuration can differ per cell (a latency-optimized stage at
  concurrency 1 can saturate first at high concurrency). Report the envelope
  configuration next to each cell instead of one "best" profile.
- A lever that changes nothing is a free repeat; use it to state run-to-run
  noise when repeats were not budgeted.

## Report template

```markdown
## Ceiling summary
| Cell | Measured | Envelope (config) | Bound | % of SOL | Binding term |
|------|----------|-------------------|-------|----------|--------------|
| RTF C1 (mean / aggregate) | | | | | |
| RTF C64 (mean / aggregate) | | | | | |
| Audio throughput C64 (audio-s/s) | | | | | |

## Setup
- vllm-omni <sha>, vllm <ver>, <GPU>, driver
- Deploy: <yaml> + <overlay>; session mode; output mode
- Workload: dataset, prompts, warmup, extra body

## Headroom ledger (C<high>)
| Stage | Time share | g_i (GPU-s per audio-s) | Bound g_i | % of SOL | Bound class | Launch floor | Next lever |
|-------|-----------|-------------------------|-----------|----------|-------------|--------------|------------|
| Stage 0 | | | | | | | |
| Stage 1 | | | | | | | |
| Stage 2 | | | | | | | |
| GPU idle / host | | | 0 | | host | | |

## End-to-end split (T = a + b * d)
| Cell | Measured a (ms) | Floor a (ms) | Measured b | Floor b | Fixed share of RTF |
|------|-----------------|--------------|------------|---------|--------------------|
| C1 | | | | | |
| C<high> | | | | | |

## Bound arithmetic
[Per stage: each term's formula with numbers substituted, its source
(measured peak, checkpoint, result JSON, FLOP counter, analytic), and any
term left "not bounded: <reason>"]

## Checks
- Little's law: C / audio_throughput vs aggregate RTF
- Admission: stage capacities >= C; graph fallbacks at C
- Quality: <metric> within run-to-run variance, or not measured

## Not measured
[Bounds left analytic, cells skipped, and why]
```

## References

- [references/bound-models.md](references/bound-models.md): pipeline and
  capacity models, Little's law, roofline bounds for decode and compute
  stages
- [references/ceiling-experiments.md](references/ceiling-experiments.md):
  admission overlays, per-stage device attribution, envelope sweeps, peak
  measurement
- [references/npu-ceiling.md](references/npu-ceiling.md): what the method keeps
  on Ascend NPU, what MPS and pmon are replaced by, reservation and
  `npu-smi` sampling, and how the bound terms shift
- `omni-perf-opt`: measurement protocol, quality gate, optimization levers
- `.claude/skills/vllm-omni/references/models/`: per-model stage maps and
  deploy YAMLs
