# Multi-stage benchmark data statistics

This section describes additional benchmark design for multi stage models.

## Overview

The `vllm bench serve --omni` command prints basic benchmark metrics with overall calculations. With adding `--print-stage` parameter, stage wise benchmark data will also be printed. This feature helps you track the performances of each stage, especially internal stages like talker in Qwen3-Omni. As these internal stages will not send outputs to the client, it is not as simple as stages with outputs for clients to track performances directly from the response.

For Qwen3-Omni: these results will be printed after end-to-end benchmark data:
<pre>
============= Stage Benchmark Result =============
=============== Stage 0 (thinker) ================
……
================ Stage 1 (talker) ================
……
=============== Stage 2 (code2wav) ===============
……
==================================================
</pre>

For HunyuanImage-3.0, stage wise metrics will be like:
<pre>
============= Stage Benchmark Result =============
================== Stage 0 (AR) ==================
……
================= Stage 1 (dit) ==================
……
==================================================
</pre>

The name of stages are fetched from `StagePipelineConfig.model_stage`, which are usually defined in `vllm_omni/model_executor/models/<YOUR_MODEL>/pipeline.py`.

## Core functions of stage benchmark

### `StagePool.build_stage_metrics()` — `vllm_omni/engine/stage_pool.py`
This is centralized method where stage metrics are computed. It is called by the Orchestrator exactly once per stage per request, at the moment `output.finished = True` is received. The method takes the raw stage output and a `submit_ts` timestamp, and returns a `StageRequestStats` object that flows downstream to the aggregator and the HTTP response.

If you want to check how the stage metrics are computed, it is recommended to add `logger.info()` inside this method.

### `print_stage_metrics()` — `vllm_omni/benchmarks/metrics/metrics.py`
This is the benchmark-side print entry point for per-stage results. It is called once per stage at the end of a benchmark run, after `_build_stage_metrics_from_outputs()` has aggregated all per-request `stage_metrics` snapshots into a `StageBenchmarkMetrics` object. All latency values go through `_print_percentile_metric()`, like mean, median, p99 (and any other percentiles in `selected_percentiles`) from the raw sample list in `StageBenchmarkMetrics`.

If you want to change the printing format of stage benchmark, you can edit this function.

## Stage Metrics design

### General design for stage local metrics
- stage_gen_time: Time from submitting a request to a specific stage (which is collected as `OrchestratorRequestState.stage_submit_ts[stage_id]`) to that stage finishing generation (which is collected when `StagePool.build_stage_metrics()` is called), which is the basic latency metric for stages.
- Generalized serving time to first output (TTFT) for streaming stages: Time from the __HTTP request being accepted by the serving frontend__ (to ignore the network latency which is measured by end-to-end benchmark, which is collected when `serve_http()` is called) to the stage producing its first __non-empty__ (to measure the time till users can get the result) output (which is collected as `StagePool._non_empty_first_output_timestamps_by_request`).
- Generalized Time-per-output-token (TPOT) and Inter-token-latency (ITL) for more types of streaming stages besides text output stage. Qwen3.5-Omni begins to use generalized abbreviations like TPOP.

### Special design for different output type stages
Stage local benchmark data can be varied for different output types. The output type of each stage should be fetched from model settings and avoid hardcode if possible. Here are some examples of customized metrics:
- Text output stage (like Thinker in Qwen3-Omni): generated tokens
- Audio output stage (like Code2wav in Qwen3-Omni): audio real-time factor
- Image output stage (like DiT in BAGEL): image generation latency
- Internal stream stage (like Talker in Qwen3-Omni): inter-chunk latency

`print_stage_metrics()` first determines the stage modality from `final_output_type` and `output_unit_type`, then dispatches to the appropriate sub-printer.

## `vllm_omni/metrics/definitions.py` — the single source of truth
This file is the central registry for all metric naming, constants, and shared formula helpers in vLLM-Omni. It is consumed by the server-side Prometheus pipeline, the benchmark client, and the stage metrics data path.

### What it contains
- Metric name constants, string keys used as dict keys in `stage_metrics` snapshots, `StageRequestStats` field names, `StageBenchmarkMetrics` field names, and Prometheus metric family names.
- Scalar defaults, like `DEFAULT_AUDIO_SAMPLE_RATE` as a fallback when a model does not populate `audio_sample_rate` in its output.
- Formula helpers, shared computations and extractions used by both the server (`build_stage_metrics`) and the benchmark client to keep results consistent.

### Always check here first when handling metrics
Before adding or renaming a metric anywhere in the codebase, search this file. The same string often already exists under a slightly different spelling. Using the wrong one silently produces a field mismatch or redundancy.

If a new name or constant is needed, add it here first, then import it:
```python
from vllm_omni.metrics import definitions as defs

# use the constant, avoid hardcoding the string
my_field = defs.MY_NEW_METRIC_MS
```
The same applies to new formula helpers — adding them to `definitions.py` ensures the server and benchmark client stay in sync by sharing exactly the same implementation.

## Adding a New Stage Metric Field
1. Add the field to `StageRequestStats` and its string key to `metrics/definitions.py`.
2. Populate it in `StagePool.build_stage_metrics()`.
3. The field propagates automatically through `_merge_stage_metric_event()` into the response dict.
4. For benchmark aggregation: add the field to `_STAGE_BENCHMARK_FIELDS` and format it in the relevant `_print_*_stage_metrics()`.

## General Troubleshooting
### No stage benchmark output at all
`print_stage_metrics()` is only called when `--print-stage` is passed to the benchmark runner. Confirm the flag is present. Without it, the entire stage section is silently skipped, even if stage metrics data is available.

### Stage name shows as stage_0, stage_1 instead of a meaningful name
The display name comes from `StagePipelineConfig.model_stage` in your model's pipeline definition. Check: `vllm_omni/model_executor/models/<YOUR_MODEL>/pipeline.py`.

For example, the `dynin_omni` pipeline sets `model_stage="token2text"` for stage 0, `model_stage="token2image"` for stage 1, and so on. If `model_stage` is missing or empty, `_build_stage_metrics_from_outputs()` falls back to `f"stage_{stage_id}"`.

```python
# vllm_omni/model_executor/models/<YOUR_MODEL>/pipeline.py
StagePipelineConfig(
    stage_id=1,
    model_stage="tts",   # ← This is taken as the stage name
    final_output=True,
    final_output_type="audio",
    ...
)
```

### A stage is missing from the output entirely
If a stage never appears in the printed table, check `StagePipelineConfig.final_output` in your pipeline definition.

Any stage whose output needs to be returned to the client (text, audio, image) should have `final_output=True`. This also controls when the request is considered finished from the client's perspective — a request only terminates after all `final_output=True` stages complete. Setting `final_output=True` on intermediate stages unnecessarily will cause premature request termination, so only set it on stages that genuinely produce client-facing output.

```python
# vllm_omni/model_executor/models/<YOUR_MODEL>/pipeline.py
StagePipelineConfig(
    stage_id=1,
    model_stage="tts",
    final_output=True,   # ← True for this stage to appear in benchmark output
    final_output_type="audio",
    ...
)
```

That said, you don't need to set every stage's `final_output=True` just to get it into the printed table. `_build_stage_metrics_snapshot()` iterates over `stage_events`, which includes events from intermediate stages recorded via `StageMetricsMessage`. As long as an intermediate stage finishes before the downstream `final_output=True` stage completes — meaning its `StageMetricsMessage` has already been consumed and written into `stage_events` — its metrics will appear in the snapshot that gets embedded in the response and subsequently printed. This is probably relevant for stages that generate Chain-of-Thought text.

### A stage appears but modality-specific metrics are missing
`print_stage_metrics()` dispatches to the correct sub-printer (`_print_audio_stage_metrics()`, `_print_image_stage_metrics()`, etc.) based on `StagePipelineConfig`.`final_output_type`. If the wrong sub-printer runs — or none does — check `final_output_type` in your pipeline definition:

| Expected metrics | Correct `final_output_type` |
| ---------------- | --------------------------- |
| TTFT, TPOT, ITL | "text" |
| TTFP, RTF, audio duration | "audio" |
| Image generation time, pixel count | "image" |
| TTFC, TPOC, inter-chunk latency | "internal_stream" |

A wrong or missing `final_output_type` causes `print_stage_metrics()` to print only the generic stage timing block with no modality-specific rows.

```python
# vllm_omni/model_executor/models/<YOUR_MODEL>/pipeline.py
StagePipelineConfig(
    stage_id=1,
    model_stage="tts",
    final_output=True,
    final_output_type="audio",   # ← This will be used to determine the modal of this stage
    ...
)
```
