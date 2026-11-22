---
name: vllm-omni
description: Deploy, serve, debug, benchmark, and extend multi-stage omni models in vLLM-Omni (Qwen3-Omni, Qwen2.5-Omni, MiniCPM-o, and similar thinker/talker/code2wav pipelines). Use when writing or tuning deploy YAML under vllm_omni/deploy, launching `vllm serve --omni`, diagnosing stage startup hangs, OOM, connector stalls, or broken streaming audio, measuring TTFT/AUDIO_TTFP/AUDIO_RTF with `vllm bench serve --omni`, or navigating the orchestrator and stage runtime code. Routes new omni model integration to `add-omni-model`, omni performance optimization to `omni-perf-opt`, and diffusion, TTS-only, quantization, testing, recipe, and PR tasks to their dedicated skills.
---

# vLLM-Omni Multi-Stage Omni Models

Entry skill for multi-stage autoregressive omni models. Handle deployment,
serving, debugging, and benchmarking here; hand off to the dedicated skill when
the task belongs to one.

## Mental model

A request flows through a **stage graph**. For Qwen3-Omni:

```
stage 0 thinker (LLM_AR)  ->  stage 1 talker (LLM_AR)  ->  stage 2 code2wav (LLM_GENERATION)
  multimodal in, text out       hidden states -> codec codes     codes -> waveform
```

- **Pipeline** (code, `models/<model>/pipeline.py`) fixes the topology and the
  stage hand-off functions (`model_executor/stage_input_processors/`).
- **Deploy YAML** (`vllm_omni/deploy/*.yaml`) sets runtime: devices, memory,
  batch limits, CUDA graphs, connectors, `async_chunk`, `session_mode`,
  `model_runner` (v1 / MRv2).
- **Orchestrator** (`vllm_omni/engine/orchestrator.py`) routes each stage output to
  the next stage or the client; **connectors** move the data.
- **`async_chunk`** streams partial outputs between stages; it determines first
  audio packet latency.

## Task routing

| Task | Read |
|------|------|
| Understand code paths, find where something lives | [references/architecture.md](references/architecture.md) |
| Write or tune a deploy YAML, GPU layout, replicas | [references/stage-config.md](references/stage-config.md) |
| Launch a server, call APIs, offline `Omni` | [references/serving.md](references/serving.md) |
| Hang, OOM, crash, stall, bad or broken audio | [references/debugging.md](references/debugging.md) |
| Benchmark, compare runs, accuracy evals, profiling | [references/benchmarking.md](references/benchmarking.md) |
| Qwen3-Omni or Qwen2.5-Omni specifics | [references/models/qwen3-omni.md](references/models/qwen3-omni.md) |
| MiniCPM-o specifics | [references/models/minicpm-o.md](references/models/minicpm-o.md) |

Read only the references the task needs. For model-specific work, read the model
file together with the task reference.

## Hand off to dedicated skills

| Task | Skill |
|------|-------|
| New thinker/talker/code2wav omni model | `add-omni-model` |
| Omni latency, throughput, streaming smoothness, or GPU-count optimization | `omni-perf-opt` |
| New diffusion model / productionizing one | `add-diffusion-model`, `production-add-diffusion-model` |
| Diffusion performance | `diffusion-perf-opt` |
| TTS-only model | `add-tts-model` |
| Quantization | `quantization` |
| Writing or running tests, CI wiring | `vllm-omni-test` |
| Recipes under `recipes/` | `add-recipe` |
| Self-check before opening a PR | `precheck-pr` |
| Reviewing a PR | `review-pr` |
| NPU model runner upgrade | `vllm-omni-npu-model-runner-upgrade` |

## Working rules

1. **Resolve the effective config first.** Identify the pipeline, the deploy YAML
   actually loaded (default or `--deploy-config`), its `base_config` chain, the
   platform section, and CLI overrides before reasoning about behavior.
2. **Localize to a stage.** Use stage ids in logs, `--log-stats`, and
   `--print-stage`. Isolate the thinker with `"modalities": ["text"]`.
3. **Change one variable at a time** and keep the rest of the command identical.
4. **Check CUDA graphs both ways.** Compare suspicious output against
   `--enforce-eager` (or the stage's `enforce_eager`) before blaming model code.
5. **Treat source as authoritative.** Docs and design notes can lag the code;
   cite paths and confirm field names in `vllm_omni/config/stage_config.py` and
   the model package.
6. **Back claims with evidence.** For any performance or correctness claim, record
   vLLM and vLLM-Omni commits, hardware, deploy YAML, and exact commands, and pair
   performance results with an accuracy check when numerics may change.

## Reporting template

```markdown
## Summary
[One or two sentences: outcome and root cause or result]

## Setup
- vllm-omni: <commit>, vllm: <commit>, hardware: <GPUs>
- Deploy: <yaml + overrides>
- Command(s): <serve / bench / test>

## Findings
- [Stage-attributed observation with numbers or log evidence]

## Next steps
- [Concrete action]
```
