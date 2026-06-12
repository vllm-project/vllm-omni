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

The name of stages are fetched from `StagePipelineConfig.model_stage`, which are usually defined in `vllm_omni/model_executor/modes/YOUR_MODEL/pipeline.py`.

## General design for stage local metrics
- stage_gen_time: Time from submitting a request to a specific stage (which is collected as `OrchestratorRequestState.stage_submit_ts[stage_id]`) to that stage finishing generation (which is collected when `StagePool.build_stage_metrics()` is called), which is the basic latency metric for stages.
- Generalized serving time to first output (TTFT) for streaming stages: Time from the __HTTP request being accepted by the serving frontend__ (to ignore the network latency which is measured by end-to-end benchmark, which is collected when `serve_http()` is called) to the stage producing its first __non-empty__ (to measure the time till users can get the result) output.
- Generalized Time-per-output-token (TPOT) and Inter-token-latency (ITL) for more types of streaming stages besides text output stage. Qwen3.5-Omni begins to use generalized abbreviations like TPOP.

## Special design for different output type stages
Stage local benchmark data can be varied for different output types. The output type of each stage should be fetched from model settings and avoid hardcode if possible. Here are some examples of customized metrics:
- Text output stage (like Thinker in Qwen3-Omni): generated tokens
- Audio output stage (like Code2wav in Qwen3-Omni): audio real-time factor
- Image output stage (like DiT in BAGEL): image generation latency
- Internal stream stage (like Talker in Qwen3-Omni): inter-chunk latency
