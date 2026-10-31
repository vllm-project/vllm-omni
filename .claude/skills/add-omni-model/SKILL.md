---
name: add-omni-model
description: Integrate a new multi-stage autoregressive omni model (thinker/talker/code2wav style, multimodal input with text and speech output) into vLLM-Omni, from the Hugging Face reference through stage separation, pipeline and deploy config, stage input processors, async_chunk streaming, parity validation, tests, and recipe. Use when adding a model like Qwen3-Omni, Qwen2.5-Omni, or MiniCPM-o, splitting an omni checkpoint into stages, writing `pipeline.py` or `stage_input_processors/<model>.py`, wiring `_OMNI_MODELS` / `OMNI_PIPELINES`, or debugging parity of a newly integrated omni model. Not for diffusion models (`add-diffusion-model`) or TTS-only models (`add-tts-model`).
---

# Add a Multi-Stage Omni Model

## Overview

```
HF reference -> Stage split -> Pipeline + registry -> Stage hand-off -> Deploy YAML
  (Phase 1)      (Phase 2)        (Phase 3)            (Phase 4)        (Phase 5)
     -> Parity ladder -> Tests + CI -> Recipe + docs -> precheck-pr
          (Phase 6)       (Phase 7)     (Phase 8)       (Phase 9)
```

Canonical in-tree examples:

| Model | Package | Stage names | Notable |
|-------|---------|-------------|---------|
| Qwen3-Omni | `vllm_omni/model_executor/models/qwen3_omni/` | `thinker` / `talker` / `code2wav` | Unified class, resolver for thinker-only variant, async_chunk default |
| Qwen2.5-Omni | `vllm_omni/model_executor/models/qwen2_5_omni/` | `thinker` / `talker` / `code2wav` | `async_chunk: false` default |
| MiniCPM-o 4.5 | `vllm_omni/model_executor/models/minicpmo_4_5/` | `llm` / `tts` / `code2wav` | Separate Code2Wav arch via `model_arch`, duplex plugin |

Developer guide: `docs/contributing/model/adding_omni_model.md`. Its stage
processor section still describes the older orchestrator-side
`custom_process_input_func(stage_list, ...)` contract; current omni models use
the producer-side hooks described in
[references/pipeline-and-handoff.md](references/pipeline-and-handoff.md). Its
e2e test path prefix `vllm_omni/tests/...` is wrong; tests live under `tests/`.

## Cross-cutting invariants

Check these at the end of every phase.

- **I1. Topology in code, runtime in YAML.** `model_stage`, `execution_type`,
  input sources, processor functions, and final outputs live in `pipeline.py`.
  The deploy YAML only sets devices, memory, batch limits, graphs, connectors,
  `async_chunk`, `session_mode`, and `model_runner`.
- **I2. One stage per process.** The unified class reads
  `vllm_config.model_config.model_stage` and builds only that stage's modules
  and weights. Never instantiate all stages in every process.
- **I3. Per-request state keyed by request id.** Any cache that survives a
  `forward()` call (codec buffers, streaming state, graph inputs) is keyed by
  request id and freed on finish. Shared buffers only fail under concurrency.
- **I4. Delta streaming.** Streaming paths emit only new audio per step;
  every `forward()` return path sets the output keys downstream expects.
- **I5. Both hand-off modes stay coherent.** For every stage edge, the
  `*_full_payload` (non-streaming) and `*_async_chunk` (streaming) processors
  must produce semantically equivalent next-stage inputs.
- **I6. Source is authoritative.** Confirm field names in
  `vllm_omni/config/stage_config.py` and an in-tree model before copying from
  docs or this skill.

## Phase 1: Hugging Face reference

1. Run the official implementation end to end with fixed inputs (text-only,
   image+text, audio+text, and a speech-output request).
2. Save golden artifacts: thinker text (greedy), talker codec ids for a fixed
   seed, and the code2wav waveform for those ids.
3. Record the architecture: sub-models, which hidden layers the talker
   consumes, special tokens, codec codebooks and frame rate, hop length,
   sample rate, and the HF `config.json` sub-config names.

Deliverable: a reference script and golden files you can diff against at
every later phase.

## Phase 2: Stage split and model package

Create `vllm_omni/model_executor/models/<name>/`:

- One module per stage (for example `<name>_thinker.py`, `<name>_talker.py`,
  `<name>_code2wav.py`).
- A unified class (`<Name>ForConditionalGeneration`) that switches on
  `model_stage`, builds only that stage, and routes `load_weights()` by prefix
  (`thinker.`, `talker.`, `code2wav.`). Follow `qwen3_omni/qwen3_omni.py`.
- `forward()` returns `OmniOutput` (`model_executor/models/output_templates.py`):
  `text_hidden_states`, `multimodal_outputs` (the payload keys the next stage
  reads, such as layer hidden states or codec codes), and optional
  `next_token_id`.
- Multimodal thinkers implement `SupportsMultiModal` (plus `SupportsMRoPE` /
  `SupportsPP` when applicable) and register their processor with
  `MULTIMODAL_REGISTRY.register_processor`.
- Keep `__init__.py` import-light; the registry imports lazily.
- If a stage needs a different class (MiniCPM-o's `MiniCPMO45Code2Wav`),
  give it its own arch and point the stage at it with `model_arch`.

## Phase 3: Pipeline and registry

1. Add every arch name to `_OMNI_MODELS` in
   `vllm_omni/model_executor/models/registry.py` as `(folder, module, class)`.
2. Define a frozen `PipelineConfig` in `<name>/pipeline.py` with one
   `StagePipelineConfig` per stage and `default_deploy_config_name`.
3. Register it in `OMNI_PIPELINES` in `vllm_omni/config/pipeline_registry.py`
   (alphabetical by `model_type`). Use `@pipeline_cfg_resolver` when the HF
   config selects a variant (Qwen3-Omni returns a thinker-only pipeline when
   `enable_audio_output` is false).

Field-by-field guidance: [references/pipeline-and-handoff.md](references/pipeline-and-handoff.md).

## Phase 4: Stage hand-off

Write `vllm_omni/model_executor/stage_input_processors/<name>.py` with, per
edge:

- `<a>2<b>_full_payload`: packs the complete upstream output when
  `async_chunk` is off.
- `<a>2<b>_async_chunk`: packs one incremental chunk when `async_chunk` is on.
- `<a>2<b>_token_only` (optional): orchestrator-side placeholder that only
  allocates `prompt_token_ids` of the right length for KV reservation.

Wire them through `custom_process_next_stage_input_func`,
`async_chunk_process_next_stage_input_func`, and `sync_process_input_func`.
`_select_processor_funcs` in `stage_config.py` picks the variant from the
effective `async_chunk`. Reference: `stage_input_processors/qwen3_omni.py`.

## Phase 5: Deploy YAML

Create `vllm_omni/deploy/<name>.yaml` with a header comment naming the
verified hardware. Start conservative and correct:

- Thinker on its own GPU; talker and code2wav share one GPU; per-GPU
  `gpu_memory_utilization` sums to at most 1.0.
- `enable_prefix_caching: false` on every stage unless the pipeline has been
  validated with it.
- Connector block (`SharedMemoryConnector` on a single node) with codec chunk
  settings when `async_chunk: true`.
- Talker `default_sampling_params` with a fixed `seed` for reproducible speech.
- `platforms:` sections only for platforms you actually ran.

Schema and layering: `.claude/skills/vllm-omni/references/stage-config.md`.
Performance profiles (MRv2, graph tuning, replicas) are a later step; use the
`omni-perf-opt` skill once parity passes.

## Phase 6: Parity ladder

Climb in order; do not move up until the current rung passes. Details and
comparison methods: [references/parity-and-validation.md](references/parity-and-validation.md).

1. Thinker text with `"modalities": ["text"]` matches HF greedy output.
2. Talker codec ids match HF for a fixed seed and identical conditioning.
3. Code2wav waveform matches HF for identical codec ids (within tolerance).
4. Full pipeline, `async_chunk` off: audio matches rung 3.
5. `async_chunk` on: audio equals non-streaming output apart from chunk
   boundaries; no truncation, repetition, or clicks.
6. Concurrency of at least 2: per-request outputs equal their single-request
   outputs (no crosstalk).
7. CUDA graphs on versus `--enforce-eager`: matching output.

## Phase 7: Tests and CI

- Component: `tests/model_executor/models/<name>/` and
  `tests/model_executor/stage_input_processors/test_<name>*.py` (CPU-only,
  cover both full-payload and async-chunk processors).
- E2E: `tests/e2e/offline_inference/test_<name>.py`,
  `tests/e2e/online_serving/test_<name>.py`, optional
  `tests/e2e/accuracy/<name>/`.
- Optional perf rows: `tests/dfx/perf/tests/test_<name>.json` (see
  `test_qwen3_omni_async_chunk.json` for the schema).
- Markers, levels, Buildkite wiring, and run commands: `vllm-omni-test` skill.

## Phase 8: Recipe and docs

- Recipe under `recipes/<Vendor>/<Model>.md` plus its `recipes/README.md` row
  via the `add-recipe` skill.
- Row in `docs/models/supported_models.md` listing only validated backends.
- No new model-specific Python under `examples/`
  (`.claude/skills/precheck-pr/references/examples-policy.md`).

## Phase 9: Pre-PR

Run the `precheck-pr` skill. The PR must include a head-versus-reference
accuracy and performance comparison with pinned revisions
(`.claude/skills/review-pr/references/checks/model-addition-checklist.md`).

## Integration checklist

```
Invariants
- [ ] I1 topology only in pipeline.py; deploy YAML has runtime keys only
- [ ] I2 each stage process builds and loads only its own modules
- [ ] I3 cross-step caches keyed by request id and freed on finish
- [ ] I4 streaming emits deltas; every forward() path sets payload keys
- [ ] I5 full_payload and async_chunk processors agree for each edge

Phases
- [ ] 1 HF reference runs; golden text, codec ids, waveform saved
- [ ] 2 Per-stage modules + unified class; forward returns OmniOutput
- [ ] 3 Arch names in _OMNI_MODELS; PipelineConfig in OMNI_PIPELINES
- [ ] 4 Stage input processors for every edge, both modes
- [ ] 5 Default deploy YAML with verified-hardware header
- [ ] 6 Parity ladder rungs 1-7 pass
- [ ] 7 Component + e2e tests, CI wiring via vllm-omni-test
- [ ] 8 Recipe, recipes/README.md row, supported_models.md row
- [ ] 9 precheck-pr passes; reference comparison in the PR
```

## References

- [references/pipeline-and-handoff.md](references/pipeline-and-handoff.md):
  `StagePipelineConfig` fields, processor hook contracts, skeletons
- [references/parity-and-validation.md](references/parity-and-validation.md):
  parity ladder methods, common failure causes, evidence to attach
- `.claude/skills/vllm-omni/references/models/qwen3-omni.md`,
  `.claude/skills/vllm-omni/references/models/minicpm-o.md`: worked examples
- `.claude/skills/vllm-omni/references/debugging.md`: symptom table
- `omni-perf-opt` skill: tuning after parity
