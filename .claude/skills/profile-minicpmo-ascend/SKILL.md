---
name: profile-minicpmo-ascend
description: Capture, analyze, compare, and interpret MiniCPM-o 4.5 online-serving profiles on Ascend NPU in vLLM-Omni. Use when locating Thinker, Talker, Code2Wav, connector, runtime API, kernel, synchronization, memory, batching, or first-chunk bottlenecks; establishing a profiler baseline; comparing a performance candidate; or producing evidence-backed MiniCPM Ascend optimization suggestions.
---

# Profile MiniCPM-o on Ascend

Use profiling to choose one controlled experiment. Never use profiler timing as
a score measurement.

## Required Context

Before profiling competition work, read:

1. `../optimize-minicpmo-ascend/SKILL.md` for gates and measurement integrity.
2. `../optimize-minicpmo-ascend/references/repo-map.md` for stage ownership.
3. [references/analysis-guide.md](references/analysis-guide.md) before interpreting a trace.

Read `../run-minicpmo-ascend-perf-cycle/SKILL.md` when the profile will lead to
an implementation candidate, commit, or push.

## Workflow

### 1. Start from an Unprofiled Baseline

- Confirm the functional/effect gate passes.
- Record Git SHA, model revision, deploy config, workload, seed, warmups,
  concurrency, environment, and unprofiled metrics.
- Identify one symptom and the stage most likely to own it.

Do not profile cold start, model loading, or graph compilation unless startup is
the explicit target.

### 2. Choose the Narrowest Capture

- Stage 0: text TTFT, multimodal preprocessing, prefill, or decode.
- Stage 1: codec generation, Talker scheduling, or first-codec delay.
- Stage 2: first audio, chunk cadence, Flow/CFM/HiFT, or vocoder throughput.
- All stages: only when queue ownership or inter-stage gaps are unclear.

Keep the default capture to one request after two unprofiled warmups. Long or
high-concurrency captures create excessive data and distort scheduling.

### 3. Run the Repository Chain

```bash
MODEL=/path/to/MiniCPM-o-4_5 \
PROFILE_ID=<candidate-id>-stage2 \
PROFILE_STAGES=2 \
bash benchmarks/competition/minicpmo_ascend/run_profile.sh
```

The runner generates a profiler-enabled deploy config, starts a clean server,
warms outside the capture, calls `/start_profile`, runs the deterministic
request, calls `/stop_profile`, shuts down, and writes raw traces plus
`profile_analysis.json` and `profile_analysis.md`.

Use `PROFILE_INPUT_MODALITY`, `PROFILE_OUTPUT_MODE`, `PROFILE_MEDIA`,
`PROFILE_WARMUPS`, `PROFILE_REQUESTS`, `THINKER_MAX_TOKENS`, and
`TALKER_MAX_TOKENS` only when the candidate record fixes the same values for
baseline and candidate.

### 4. Check Capture Integrity

- Require successful warmups, request completion, non-empty output, and clean
  profiler stop.
- Confirm the selected stages produced exported CSV and a timeline.
- Keep raw trace paths and the capture JSON with the candidate record.
- Reject a capture that contains failures, different output length, changed
  modalities, or unmatched profiler settings.

### 5. Analyze Before Editing

Use `profile_analysis.md` to rank device kernels, runtime APIs, Torch operators,
and the <=50 us kernel fraction. Open `trace_view.json` in Perfetto or the CANN
database in MindStudio when aggregate tables cannot distinguish compute from
queue gaps.

State:

1. bottleneck evidence;
2. stage and code path;
3. one primary variable;
4. expected unprofiled metric impact;
5. correctness, memory, and stability guardrails;
6. rollback condition.

### 6. Compare Matching Captures

After an unprofiled A/B/A benchmark, compare diagnostic traces only when the
workload, stage selection, environment, and profiler configuration match:

```bash
.venv/bin/python -m benchmarks.competition.minicpmo_ascend.profile_analysis compare \
  <baseline>/profile_analysis.json <candidate>/profile_analysis.json \
  --output <comparison>/profile_comparison.json
```

Accept or reject the candidate from unprofiled benchmark and gate results. Use
the profile comparison only to explain the mechanism.

## Output Contract

Report the unprofiled baseline, capture scope, top evidence, hypothesis,
guardrails, artifact paths, and unresolved ambiguity. Label every profile
number as diagnostic and every non-official workload as a local proxy.
