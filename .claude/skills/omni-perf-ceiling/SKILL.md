---
name: omni-perf-ceiling
description: Estimate idealized serving performance of a vLLM-Omni multi-stage omni model (Qwen3-Omni, MiniCPM-o, and similar thinker/talker/code2wav pipelines) and how much headroom the current implementation leaves. Use when asked for the lowest reachable AUDIO_RTF, TTFT or first-audio latency (AUDIO_TTFP) at a given concurrency, the maximum audio throughput on given hardware, whether a model "still has optimization room", or which stage bounds a target. Produces measured and calibrated roofline estimates with a per-stage gap ledger, on CUDA GPUs or Ascend NPUs. Use `omni-perf-opt` afterwards to implement optimizations.
---

# Omni Performance Ceiling

Answer "how fast can this get, and where is the remaining gap" with two
numbers per target, not one:

1. **Measured**: the shipped configuration on the target hardware.
2. **Calibrated estimate**: reference work priced with device microbenchmarks.
   It is conditional on the workload, precision, cache and serial-path assumptions.
   It is neither a strict physical bound nor an achievable performance target.

`measured - estimate` is an unattributed gap, not guaranteed optimization
headroom. Report it per stage and require profiling before assigning a cause.
The legacy CLI `--*-bound` and JSON `*_bound` names remain for compatibility;
`bound_kind=calibrated_estimate` identifies their semantics.

## First questions

Fill in what is not already known; the `omni-perf-opt` first questions apply.
In addition, fix:

- Target cells, for example `AUDIO_RTF` at concurrency 1 and 64, and TTFT or
  `AUDIO_TTFP` at concurrency 1.
- RTF definition: per-request mean (`mean_audio_rtf`) or aggregate
  (sum of generation time over sum of audio). They differ under load.
- Output mode: streaming chunks or a single final audio. Streaming adds
  first-chunk work; compare engines only within one mode.
- Device budget: one shared device, or one device per stage.
- Accelerator: CUDA GPU or Ascend NPU. On Ascend read
  [references/npu-ceiling.md](references/npu-ceiling.md) first; MPS and
  per-process SM attribution are not available there.

## Two regimes

Concurrency 1 is a latency problem; high concurrency is a capacity problem.
Do not use one model for both. Derivations:
[references/bound-models.md](references/bound-models.md).

- **Low concurrency**: account for serial stage work and the gates before
  each stage starts. Stage wall time includes upstream waits, so it does not
  establish exclusive kernel time or exclude scheduling overhead.
- **High concurrency**: Little's law relates aggregate RTF to audio throughput
  while concurrency is held. Accumulate shared demand per resource (memory
  and each compute dtype), take the maximum, then combine with each stage's
  serial constraint. Do not sum durations of stages that can overlap via MPS
  or independent streams. Serial rows inside one stage may still be summed.

Fixed request work joins the matching resource before the maximum. The
aggregate shared estimate is `C/N * max_q(D_q + W_q/mean(d))`; D is streaming
resource demand and W is fixed demand. Different dtype constraints remain
separate unless a hardware resource mapping proves they share a serial pipe.

The fitted `T_r = a + b*d_r` split is descriptive and assumes common a and b.
Its coefficients do not identify causal fixed/streaming overhead. Workload
averages do not bound individual requests: the current report leaves mean
RTF unavailable, even at C1, because it lacks per-request work counts.

First-token and first-packet latency estimates describe the head of `a`. TTFT includes
the work before the Thinker's first token (encoders, prefill on uncached
tokens); TTFP adds every hand-off before the first audio packet (Thinker steps
the Talker needs, the first codec chunk, Code2Wav on that chunk). The estimate
has no queueing term, so judge it at concurrency 1.

## Workflow

1. **Freeze the workload and commands.** Use the `omni-perf-opt`
   measurement templates. Record commit, hardware, deploy YAML plus overlay,
   output mode, dataset, prompt count, and `--extra-body`.

2. **Check admission before measuring capacity.** Every stage's
   `max_num_seqs`, session caps (`max_sessions` for duplex), and per-stage
   graph batch limits must be at least the target concurrency. Otherwise the
   high-concurrency cell measures queueing, not compute. Read startup logs
   for graph capture budgets and eager fallbacks at the target batch. Check
   KV capacity at the workload too (`--kv-capacity`): each stage's
   `GPU KV cache size` divided by its largest request's tokens must be at
   least the target concurrency.

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
   before reading anything else. Pass `--series` with the cells of one
   configuration. Decreases in throughput, aggregate RTF, TTFT or TTFP
   are diagnostics to investigate; batching can change efficiency with C.

5. **Attribute device time per stage.** Pick one:
   - one device per stage (`*_3gpu.yaml`): per-device utilisation is the
     stage's `g_i` directly;
   - CUDA only: one shared device with MPS disabled
     (`platforms.cuda.cuda_mps: false`) and `nvidia-smi pmon -s u`, giving
     per-process SM share. Ascend has no per-process AI Core counter;
   - stage isolation microbenchmarks at the target batch.
   Experiment recipes:
   [references/ceiling-experiments.md](references/ceiling-experiments.md).

6. **Compute calibrated estimates.** Run `scripts/measure_peaks.py` on the target device for
   memory bandwidth, dense matmul throughput and per-kernel launch latency
   (eager and graph replay). Run `scripts/count_reference_work.py` on the
   checkpoint for work counts: it builds the reference transformers model on
   the meta device and counts each decoder's prefill and decode FLOPs per
   token, streamed weight bytes and KV bytes per token, plus Code2Wav FLOPs per
   chunk from the reference streaming path, and prints the bound flags. Give
   each stage three terms: compute (FLOPs per audio second), memory (weights
   plus KV read per decode step, or bytes moved), and launch (kernels per step
   times steps per audio second, at the stage's eager or graph mode). Split a
   stage into kernel groups with different bound classes, one flag each; the
   serial stage estimate is `sum over groups of max(compute, memory)` and
   `max(device, launch)` in its own process; the larger term names the
   stage's bound class.
   Write each term's formula with the numbers substituted so it can be
   re-checked, and state which inputs are measured and which analytic.
   Then add the per-request terms: `--fixed-bound` for work every request
   adds (Thinker prefill on uncached tokens, encoders) and `--fill-bound` for
   latency before a downstream stage can start (first Thinker step before the
   Talker, the first codec chunk before Code2Wav, and `@out` for Code2Wav on
   that chunk). The report produces an aggregate RTF estimate, a TTFT
   estimate (terms gating stage 0) and a TTFP estimate (all serial gates).
   Mean RTF estimation requires per-request work and is unavailable.

7. **Compare measurement and estimate.** Pass `NAME@STAGE` so the report
   joins the calibration to measured stage busy time. Estimate/measured is a
   diagnostic ratio; values above 100% can reflect calibration limits.
   Treat kernel efficiency, extra work, host, transfer and queueing as
   hypotheses. Use nsys/msprof timelines and kernel counters to identify the
   critical path. Slow kernels can cause queueing and downstream idle time.
   The fitted a/b split and an idle percentage cannot establish causality.
   `--efficiency-target compute=F,memory=F` projects shared resource demand at
   a hypothetical fraction of calibration. It does not measure achieved
   efficiency or attribute the residual. `--profiled NAME=SECONDS` compares
   a supplied implementation duration with its calibrated roofline. Neither
   comparison is a guarantee of an attainable speedup.

8. **Report.** Fill the template below. Hand the largest gap to
   `omni-perf-opt` as its first candidate.

## Rules

- Do not quote a ceiling without the measured cell beside it on the same
  commit and hardware.
- Historical numbers are context, not evidence for the current revision.
- Label empirical rooflines as calibrated estimates. Strict hardware bounds
  require justified resource limits and unavoidable work; this tool does not
  certify them. Name perfect batching, cache, precision and serial assumptions.
- Calibrate on the serving device and host with `measure_peaks.py`. A single
  GEMM shape or PyTorch tiny-op launch cost is not a universal hardware limit.
- Shared devices use separate resource constraints; summing serial stage
  durations across independent streams overstates the shared requirement.
- Work counts (FLOPs, streamed weight bytes, KV bytes) come from the
  checkpoint's reference implementation, never from the vLLM-Omni code path
  being measured: counting the engine's own path folds its waste (padding,
  recomputed overlap, extra CFG rows) into the bound and hides the headroom.
  Take only FLOPs, weights and KV from the reference, not activation bytes;
  the reference does not fuse. Workload (tokens per audio second, context,
  uncached tokens) comes from the result JSON.
- Numerics and algorithm constants follow the reference unless the measured
  deployment differs. Then price the deployment's dtype in the bound, and its
  chunk size and steps as an algorithm lever, with the reference's setting
  reported beside it.
  A term that cannot be grounded is written as "not bounded: <reason>",
  never estimated to fill the column.
- Investigate ratios above 100%: calibration may be exceeded, or work coverage
  and attribution may differ. Do not declare physical impossibility.
- Report aggregate and mean RTF separately. Workload-average work supports only
  aggregate estimates. Mean RTF stays unavailable without per-request counts.
- TTFT/TTFP estimates omit queueing. At every C their residual may also include
  inefficient kernels or extra work; profile before assigning host/scheduler cost.
- Report the regime for each cell. Do not extrapolate a high-concurrency
  RTF from low-concurrency runs, or the reverse.
- A cell whose throughput is still rising has not established saturation.
  Keep its estimate tied to its own concurrency and workload.
- A ceiling belongs to one accelerator. Bounds scale with that device's
  measured peaks, so never carry a CUDA ceiling to Ascend or back.

## Report template

```markdown
## Ceiling summary
| Cell | Measured | Calibrated estimate | Estimate/measured | Binding term |
|------|----------|-------|----------|--------------|
| Aggregate RTF C1 | | | | |
| Mean RTF C1 | | unavailable: per-request work missing | | |
| Aggregate RTF C64 | | | | |
| Mean RTF C64 | | unavailable: per-request work missing | | |
| Audio throughput C64 (audio-s/s) | | | | |
| TTFT C1 (ms, mean) | | | | |
| AUDIO_TTFP C1 (ms, mean) | | | | |

## Setup
- vllm-omni <sha>, vllm <ver>, <GPU>, driver
- Deploy: <yaml> + <overlay>; session mode; output mode
- Workload: dataset, prompts, warmup, extra body

## Gap ledger (C<high>)
| Stage | Time share | g_i (GPU-s per audio-s) | Estimated g_i | Estimate/measured | Resource class | Launch estimate | Next lever |
|-------|-----------|-------------------------|-----------|----------|-------------|--------------|------------|
| Stage 0 | | | | | | | |
| Stage 1 | | | | | | | |
| Stage 2 | | | | | | | |
| GPU idle (cause unassigned) | | | | | | | |

## End-to-end split (T = a + b * d)
| Cell | Fitted a (ms) | Estimate a (ms) | Fitted b | Estimate b | Fixed share of RTF |
|------|-----------------|--------------|------------|---------|--------------------|
| C1 | | | | | |
| C<high> | | | | | |

## First-packet path (C1)
| Term | Gates | Latency estimate (ms) | Formula |
|------|-------|--------------------|---------|
| Thinker prefill | stage 0 (TTFT) | | |
| Thinker steps before Talker | stage 1 | | |
| First codec chunk | stage 2 | | |
| Code2Wav first chunk | first packet | | |
| **TTFP estimate** / measured | | | |

## Estimate arithmetic
[Per stage: each term's formula with numbers substituted, its source
(measured peak, checkpoint, result JSON, FLOP counter, analytic), and any
term left "not bounded: <reason>"]

## Checks
- Little's law: C / audio_throughput vs aggregate RTF
- Admission: stage capacities and KV capacity >= C; graph fallbacks at C
- Sweep: investigate changes in throughput, aggregate RTF, TTFT and TTFP
- Attribution: trace evidence, or gap remains unattributed
- Interpretation: calibrated estimates, no guaranteed speedup
- Quality: <metric> within run-to-run variance, or not measured

## Not measured
[Bounds left analytic, cells skipped, and why]
```

## References

- [references/bound-models.md](references/bound-models.md): pipeline and
  capacity models, Little's law, roofline bounds for decode and compute
  stages
- [references/ceiling-experiments.md](references/ceiling-experiments.md):
  admission overlays, per-stage device attribution, peak measurement
- [references/npu-ceiling.md](references/npu-ceiling.md): what the method keeps
  on Ascend NPU, what MPS and pmon are replaced by, reservation and
  `npu-smi` sampling, and how the bound terms shift
- `omni-perf-opt`: measurement protocol, quality gate, optimization levers
- `.claude/skills/vllm-omni/references/models/`: per-model stage maps and
  deploy YAMLs
