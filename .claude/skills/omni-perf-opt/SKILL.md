---
name: omni-perf-opt
description: Diagnose and optimize serving performance of multi-stage autoregressive omni models in vLLM-Omni (Qwen3-Omni, Qwen2.5-Omni, MiniCPM-o, and similar thinker/talker/code2wav pipelines). Use when asked to lower TTFT or first-audio latency (AUDIO_TTFP), keep AUDIO_RTF below 1 under load, raise throughput or concurrency, fit stages on fewer GPUs, fix choppy streaming audio, attribute latency to a stage or connector, tune deploy YAML performance knobs (batching, CUDA graphs, async_chunk codec chunking, MRv2, replicas), or plan and validate omni performance PRs. Diffusion workloads use `diffusion-perf-opt`.
---

# Omni Model Performance Optimization

Run a disciplined loop: freeze commands, measure a low-overhead baseline,
attribute the gap to one stage or hop, change one lever, and prove the gain
without losing quality. Profiler traces diagnose; they never supply the
final latency number.

## First questions

Ask for anything not already known:

- Hardware: GPU model and count, NVLink, whether stages may share a GPU.
- Model, checkpoint, vLLM and vLLM-Omni commits, and the deploy YAML actually
  loaded (default or `--deploy-config`, its `base_config` chain, platform
  section, `--stage-overrides`).
- Session mode: `turn` (HTTP) or `duplex` (realtime). They have different
  defaults and different MRv2 support.
- Workload: input modalities and sizes, output modalities (text only or text
  plus audio), output length, concurrency or request rate, streaming or not.
- Target: TTFT, AUDIO_TTFP, AUDIO_RTF, audio continuity, throughput at a
  given concurrency, GPU count, or memory. Name one primary metric and its
  percentile (mean, P90, P99).
- Quality tolerance: bitwise, within run-to-run variance, or a stated
  accuracy and WER budget. Default to quality-preserving.

## Workflow

1. **Freeze three commands.** Write them as scripts with every variable
   explicit:
   - server: `vllm serve <model> --omni --deploy-config ... [--stage-overrides ...]`
   - single request: one fixed `/v1/chat/completions` (or `/v1/audio/speech`,
     `/v1/realtime`) payload with fixed media, seed, and `modalities`
   - benchmark: `vllm bench serve --omni ...` with fixed dataset, prompt
     count, concurrency or rate, `--extra-body`, `--percentile-metrics`,
     `--print-stage`, `--save-result`
   Templates: [references/measurement-protocol.md](references/measurement-protocol.md).

2. **Collect a real baseline.** No torch profiler, no orchestrator monitor.
   Warm up (graph capture and first-request compilation distort small runs).
   Sweep concurrency at 1, a middle value, and the target. Record mean and
   P99 TTFT, AUDIO_TTFP, AUDIO_RTF, E2EL, throughput, failures, and
   `audio_continuity_ok_rate`. Run the accuracy baseline on the same build.

3. **Attribute the gap to a stage or hop.** Re-run with `--log-stats`
   (per-stage wall time, `stage_gen_time_ms`, connector `tx_time_ms`,
   `rx_decode_time_ms`, `in_flight_time_ms`) and `--print-stage`. Classify
   with the bottleneck table below before touching code.

4. **Pick a lever.** Use [references/optimization-playbook.md](references/optimization-playbook.md).
   Try configuration-only levers before code changes, and existing model
   options before new kernels.

5. **A/B one variable at a time.** Same hardware, commits, dataset, seed,
   prompt count, concurrency, and `--extra-body`. When the change is behind a
   switch, run every column on the same head and toggle the switch. Stacked
   changes get one column each.

6. **Pass the quality gate.** See below. A failed or inconclusive gate blocks
   a ready claim.

7. **Trace only after narrowing.** Profile the stage you already suspect:
   per-stage `profiler_config` through `--stage-overrides`, warm up, then
   `POST /start_profile` and `POST /stop_profile` with `{"stages": [...]}`.
   Keep operator/shape traces and host-stack traces separate. For timelines,
   reuse `.claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py`.

8. **Write the plan.** Classify candidates P0/P1/P2, tie each to evidence,
   and let the user choose before implementing anything above P0.

## Bottleneck table

| Evidence | Likely bottleneck | First levers |
|----------|-------------------|--------------|
| High TTFT, thinker prefill dominates | Thinker prefill or multimodal encoding | Encoder CUDA graphs, `max_num_batched_tokens`, thinker TP, dedicated thinker GPU |
| Good TTFT, high AUDIO_TTFP | Talker first chunk or code2wav first decode | `initial_codec_chunk_frames`, MRv2 `talker_first_audio` + `codec_chunk_ramp`, first-chunk express |
| AUDIO_RTF ≥ 1 as concurrency rises | Talker or code2wav saturated | Stage `max_num_seqs`, CUDA graphs, talker replicas, code2wav graph batch sizes |
| Good mean RTF, low `audio_continuity_ok_rate` | Bursty chunk delivery | `codec_chunk_frames`, `codec_left_context_frames`, first-chunk express slack |
| High connector `in_flight_time_ms` or `rx_decode_time_ms` | Transport or payload size | Connector type, payload trimming, co-location |
| High latency while GPUs idle | Orchestrator loop or queue backlog | `--enable-orch-monitor`, event-driven orchestrator, replica balance |
| OOM or cannot co-locate | Memory split | `gpu_memory_utilization`, `kv_cache_memory_bytes`, graph budgets, MPS |
| Client time far above server time | Client, streaming encode, or network | Response format, chunk size, bench client placement |

## Quality gate

Every optimization needs correctness evidence on the same workload, including
"math-preserving" changes.

- Text: thinker greedy output unchanged, or accuracy (Daily-Omni, Video-MME)
  within baseline run-to-run variance.
- Speech: Seed-TTS WER (`--seed-tts-wer-eval`) within variance; for
  bitwise-claimed changes, identical codec ids and waveform for a fixed seed.
- Streaming: same duration, no truncation or repetition, and
  `audio_continuity_ok_rate` not worse.
- Concurrency: per-request outputs at concurrency 2+ equal single-request
  outputs.
- Graphs: graph mode matches `--enforce-eager` for touched stages.
- Precision changes (TF32, fused kernels, cuDNN algorithm search, short KV,
  quantization) must state that rounding changes and show the metric delta.

## Priority rules

- **P0**: measurement fixes and configuration-only changes with low risk
  (warmup, frozen commands, batch limits, placement, replicas, existing
  opt-in deploy profiles, chunk sizes).
- **P1**: contained code changes (graph capture for a new stage path, host
  sync removal, payload trimming, scheduler batching, fused existing ops).
- **P2**: numerics-changing or invasive work (new custom kernels, TF32 or
  FP8 paths, short-KV attention, algorithm changes, MRv2 enablement for a new
  model).

Present the plan as a table (priority, stage, lever, evidence, expected gain,
validation, quality risk), then ask which item to implement. Proceed alone
only on P0 measurement or instrumentation fixes.

## Omni heuristics

- `async_chunk` trades total compute for first-audio latency. It cuts
  AUDIO_TTFP sharply but can raise E2EL and RTF; offline batch jobs may be
  faster without it. Judge it against the stated primary metric.
- Streaming output and `async_chunk` depend on each other: without
  `async_chunk` streaming cannot start early, and without streaming the
  client does not see the earlier first packet.
- Put the thinker on its own GPU first; talker and code2wav are smaller.
  When the talker saturates, add talker replicas before raising thinker batch
  limits.
- Prefix caching stays off for the full Qwen3-Omni and MiniCPM-o pipelines;
  the MRv2 first-frame path also requires it off.
- MRv2 output is not bitwise identical to V1. Re-measure MRv2 numbers after
  any runner change; MiniCPM-o's earlier MRv2 numbers were invalidated. MRv2
  downstream stages support turn mode; duplex needs its own profile and its
  own measurement.
- Stages sharing one GPU under MRv2 should run under NVIDIA MPS.
- Graph caches have fixed budgets. When a budget fills, uncached shapes run
  eagerly: correct but slower. Check startup capture logs after changing
  batch sizes or graph caps.
- The orchestrator is event-driven by default (`VLLM_OMNI_EVENT_DRIVEN_ORCH`
  unset). Orchestrator monitor counts are not comparable across loop modes.
- Never report profiler-enabled latency as a result.

## Report template

```markdown
## Summary
[Outcome, primary metric delta, and the stage it came from]

## Setup
- vllm-omni <sha>, vllm <sha>, <GPU x N>, driver/CUDA
- Deploy: <yaml> + <overrides>; session mode <turn|duplex>
- Commands: serve / request / bench (scripts or exact lines)

## Results (warmup <w>, measured <n> requests, concurrency <c>)
| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| TTFT mean / P99 (ms) | | | |
| AUDIO_TTFP mean / P99 (ms) | | | |
| AUDIO_RTF mean / P99 | | | |
| Throughput (req/s or audio-s/s) | | | |
| audio_continuity_ok_rate | | | |
| Failures | | | |

## Stage attribution
| Row | Baseline | Candidate |
|-----|----------|-----------|
| Stage 0 gen (ms) | | |
| Stage 1 gen (ms) | | |
| Stage 2 gen (ms) | | |
| Connector in-flight (ms) | | |
Isolation: [metric that stayed unchanged, isolating the gain to <stage>]

## Quality
[Accuracy / WER / parity result versus baseline variance]

## Next steps
[Candidate table or chosen item]
```

## References

- [references/measurement-protocol.md](references/measurement-protocol.md):
  command templates, metric definitions, diagnostics, A/B rules
- [references/optimization-playbook.md](references/optimization-playbook.md):
  levers by layer with knobs, evidence, and validation
- `.claude/skills/vllm-omni/references/models/qwen3-omni.md`,
  `.claude/skills/vllm-omni/references/models/minicpm-o.md`: model profiles
- `docs/design/qwen3_omni_tts_performance_optimization.md`,
  `docs/design/qwen3_omni_mrv2_performance.md`,
  `docs/design/minicpm_o45_mrv2_performance.md`: design evidence
- `docs/contributing/profiling.md` (sections on Omni AR/TTS stages and the
  orchestrator monitor), `docs/contributing/metrics.md`
- `.claude/skills/review-pr/references/checks/perf-verification.md`: what
  reviewers will ask for
- `quantization` skill for precision work; `vllm-omni-test` for perf JSON rows
  under `tests/dfx/perf/tests/`
