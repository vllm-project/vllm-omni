---
name: omni-perf-ceiling
description: Estimate the best achievable serving performance of a vLLM-Omni multi-stage omni model (Qwen3-Omni, MiniCPM-o, and similar thinker/talker/code2wav pipelines) and how much headroom the current implementation leaves. Use when asked for the lowest reachable AUDIO_RTF or first-audio latency at a given concurrency, the maximum audio throughput on given hardware, whether a model "still has optimization room", or which stage bounds a target. Produces measured, configuration-envelope, and physical-bound numbers with a per-stage headroom ledger. Use `omni-perf-opt` afterwards to implement optimizations.
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
- GPU budget: one shared device, or one device per stage.
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
  bound each `g_i`, add them up.

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
   `--save-result --save-detailed`, and GPU utilisation sampling
   (`nvidia-smi --query-gpu=utilization.gpu -lms 500`). Keep profilers off.

4. **Build the ledger.** Run
   `python .claude/skills/omni-perf-ceiling/scripts/ceiling_report.py`
   on the result JSON and bench log of each cell. It checks Little's law,
   derives aggregate RTF, per-stage time shares, GPU idle share, and the
   implied throughput floor. A Little's-law mismatch above 10% means the run
   did not hold concurrency (ramp, failures, or too few prompts); fix that
   before reading anything else.

5. **Attribute GPU time per stage.** Pick one:
   - one device per stage (`*_3gpu.yaml`): per-device utilisation is the
     stage's `g_i` directly;
   - one shared device with MPS disabled (`platforms.cuda.cuda_mps: false`)
     and `nvidia-smi pmon -s u`: per-process SM share;
   - stage isolation microbenchmarks at the target batch.
   Experiment recipes:
   [references/ceiling-experiments.md](references/ceiling-experiments.md).

6. **Explore the envelope.** Change one configuration lever per column
   (capacity, graph batch limits, micro-batch size, runner, MPS, replicas,
   placement). Keep the best column per cell. Quality must stay within
   run-to-run variance (`omni-perf-opt` quality gate) for the envelope to
   count.

7. **Compute bounds.** Run `scripts/measure_peaks.py` on the target GPU for
   memory bandwidth and dense matmul throughput. Bound autoregressive decode
   by bytes read per step, compute stages by FLOPs per audio second (torch
   profiler `with_flops` on that stage, or analytic counts). State which
   bound is analytic and which is measured.

8. **Report.** Fill the template below. Hand the largest gap to
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
- Report the regime for each cell. Do not extrapolate a high-concurrency
  RTF from low-concurrency runs, or the reverse.
- A cell whose throughput is still rising with concurrency is not saturated;
  its aggregate RTF floor is the next cell's, not this one's.
- The best configuration can differ per cell (a latency-optimized stage at
  concurrency 1 can saturate first at high concurrency). Report the envelope
  configuration next to each cell instead of one "best" profile.
- A lever that changes nothing is a free repeat; use it to state run-to-run
  noise when repeats were not budgeted.

## Report template

```markdown
## Ceiling summary
| Cell | Measured | Envelope (config) | Bound | Binding stage |
|------|----------|-------------------|-------|---------------|
| RTF C1 (mean / aggregate) | | | | |
| RTF C64 (mean / aggregate) | | | | |
| Audio throughput C64 (audio-s/s) | | | | |

## Setup
- vllm-omni <sha>, vllm <ver>, <GPU>, driver
- Deploy: <yaml> + <overlay>; session mode; output mode
- Workload: dataset, prompts, warmup, extra body

## Headroom ledger (C<high>)
| Stage | Time share | g_i (GPU-s per audio-s) | Bound g_i | Gap | Next lever |
|-------|-----------|-------------------------|-----------|-----|------------|
| Stage 0 | | | | | |
| Stage 1 | | | | | |
| Stage 2 | | | | | |
| GPU idle / host | | | 0 | | |

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
  admission overlays, per-stage GPU attribution, envelope sweeps, peak
  measurement
- `omni-perf-opt`: measurement protocol, quality gate, optimization levers
- `.claude/skills/vllm-omni/references/models/`: per-model stage maps and
  deploy YAMLs
