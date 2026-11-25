---
name: add-fullduplex-model
description: "Integrate a new full-duplex speech or multimodal model into vLLM-Omni on the unified engine-resident duplex framework (DuplexOmni, DuplexOmniEngine, DuplexOrchestrator, DuplexModelPlugin, Model Runner V2 hooks, WS /v1/realtime?duplex=1). Use when adding a new full-duplex model (autonomous listen/speak, frame-locked S2S, or VAD/turn-based duplex), implementing DuplexModelPlugin and DuplexDataPlane under vllm_omni/model_executor/models/<model>/duplex/, wiring PipelineConfig.duplex_plugin, debugging MRv2 resumable hooks/fencing/isolation, or validating multi-session duplex serving and Omni-DuplexEval benchmarks."
---

# Full-Duplex Model Integration Workflow

## Overview

```text
HF Reference -> Pipeline & Registry -> DuplexModelPlugin -> MRv2 Hooks & Isolation
  (Phase 1)          (Phase 2)             (Phase 3)              (Phase 4)
    -> Deploy YAML & Warmup -> Parity Ladder & Tests -> Recipe, Docs & Pre-PR
            (Phase 5)               (Phase 6)                 (Phase 7)
```

Active architecture (`docs/design/fullduplex.md`):

```text
WebSocket (/v1/realtime?duplex=1, alias /v1/duplex) or Python (DuplexClient / InlineDuplexClient)
  -> OmniDuplexSessionHandler (thin transport: WS I/O, resume/replay; NO session state)
  -> DuplexOmni(AsyncOmni) -> DuplexOmniEngine [loads PipelineConfig.duplex_plugin once]
  -> DuplexOrchestrator [DuplexStagePort]
      ├─ DuplexSessionManager (admission, backpressure, replica placement, lease reaper)
      └─ DuplexSessionRunner ×N (ONE DuplexEngineSession, ordered mailbox, VAD/tools/metrics)
           └─ <Model>DuplexPlugin (DuplexModelPlugin ABC + DuplexDataPlane)
                └─ Stage workers (vllm_omni/worker_v2/: resumable or turn-scoped MRv2 requests)
```

> **Prerequisites**: If the base model does not yet exist in `vllm-omni`, build its staged topology with [add-omni-model](../add-omni-model/SKILL.md) or [add-tts-model](../add-tts-model/SKILL.md) first (single-stage frame-locked models can build Stage 0 + `duplex/plugin.py` together).

### Choose the Integration Pattern

| Pattern | When to use | In-tree references | Request lifecycle & hooks | Turn & output policy |
|---------|-------------|--------------------|---------------------------|----------------------|
| **A: Autonomous Listen/Speak** (Resumable multi-stage) | Model samples control tokens (`<\|listen\|>` / `<\|speak\|>` / `<\|turn_eos\|>`) at unit boundaries | `minicpmo_4_5/duplex/` (`Thinker -> Talker -> Code2Wav`) | Resumable Stage 0 (`resumable=True`); `has_preprocess=True` + `preprocess_batch_mrv2`; `mrv2_custom_sampler` for logits masking | `decide_output()` returns `DIRECT_RESPONSE` on listen; `supports_model_native_turn_policy=True`, `supports_barge_in=True` |
| **B1: Frame-Locked Single-Stage** (In-step audio feedback) | Frame `t` audio latents/codes feed directly into frame `t+1` of the LM backbone | `haru/duplex/` (plugin and data plane), `haru/model_state.py` (MRv2 worker hooks) | Single AR stage (`async_chunk: false`, `recompute_preemption="fail"`); `RequestOutputKind.DELTA`; `create_mrv2_model_state` + `OmniSampler.sample_step` (`sparse_audio`) | `decide_output()` returns `None`; `FixedFramePcmAppendBuffer` + `CumulativeAudioTextDataPlane(audio_output_is_delta=True)` |
| **B2: Frame-Locked Multi-Stage** (Resumable LM -> Codec/TTS) | Fixed-frame temporal LM/Thinker feeds downstream Talker/Code2Wav via `async_chunk` | `personaplex/duplex/`, `nemotron_voicechat/duplex/` | Resumable Stage 0 (`max_tokens=1`, `ignore_eos=True`); `create_mrv2_model_state` + `post_sample` (or `has_postprocess=True`); downstream MRv2 stages set `supports_duplex_mrv2=True` | `decide_output()` returns `None` (or `DIRECT_RESPONSE` on stage-0 decisions); `FixedFramePcmAppendBuffer` + `CumulativeAudioTextDataPlane` |
| **C: Turn-Commit Ephemeral Duplex** (VAD / Smart Turn + interruptible output) | Turn-oriented / multimodal model with concurrent user input, server VAD, barge-in, tools, or memory | `qwen3_omni/duplex/`, `aura_omni/duplex/`, `joyai_vl_interaction/duplex/` | Fresh turn-scoped request per commit (`resumable=False`, `supports_core_resumable_request=False`); base MRv2 stages | `supports_model_native_turn_policy=False`, `supports_barge_in=True` (Qwen3/AURA) |

> **Config-only enablement is invalid**: Setting `duplex_plugin` and `session_mode: duplex` without wiring a real scheduler data plane advertises a broken endpoint whose model never receives the live stream.

## Cross-Cutting Invariants

Check all seven invariants at the end of every phase.

- **I1. Strict import & package boundaries.** Turn-based bases (`AsyncOmni`, `Orchestrator`, etc.) must import zero duplex code; `DuplexOmni`/`DuplexOrchestrator` must never eagerly import `<model>.duplex` (plugins load only via `PipelineConfig.duplex_plugin`). Import wire helpers only from `vllm_omni.protocol.duplex` (never `protocol.realtime`); use `abc.ABC`, not `typing.Protocol`.
- **I2. Single session authority & server-owned config.** `DuplexEngineSession` owns all session state — never store mutable per-session state on the plugin instance. Declare `private_runtime_config_keys`, validate `extra_body` and `model_custom_options`, reject client filesystem paths and forbidden live updates with `DuplexRuntimeConfigError`.
- **I3. Reuse framework building blocks.** Subclass `DefaultDuplexModelSessionState`, `FixedFramePcmAppendBuffer`, and `CumulativeAudioTextDataPlane` (`audio_output_is_delta=True` iff the final stage emits `RequestOutputKind.DELTA`); build appends via `build_duplex_append_prompt()` and `decode_pcm_f32le_payload()`.
- **I4. Atomic fences & preemption safety.** Derive stage request IDs from `DuplexFence(session_id=..., epoch=..., turn_id=...)` so barge-in/cancel advances `epoch` atomically. Set `recompute_preemption="fail"` on any stateful AR stage whose worker state cannot be recomputed from `prompt_token_ids`.
- **I5. Per-request MRv2 isolation & hot-loop discipline.** Key all worker/codec state by `request_id`, route batch metadata per row (never broadcast row `0`), release only finished IDs in `on_requests_finished()`, keep `enable_prefix_caching: false` on placeholder-slot stages, and avoid `.item()`/`.cpu()`/`.tolist()` syncs inside per-frame loops.
- **I6. Honest capabilities, mandatory `session_mode`, non-blocking loop.** Every duplex deploy YAML must set `session_mode: duplex` explicitly. Set `supports_input_append=True`, separate `supports_multi_session` from `supports_multi_session_same_replica`, offload blocking I/O in `prepare_runtime_config()` via `asyncio.to_thread()`, and set safe CUDA Graph / `warmup_frames` directly in the shipped deploy YAML.
- **I7. Source is authoritative & honest evidence.** Confirm exact field names and hook signatures in `stage_config.py`, `engine/duplex/plugin.py`, and `worker_v2/` before copying from docs. Never infer voiced speech quality or multi-session isolation from `/health` or silent warmup.

## Phase 1: Reference Analysis & Pattern Selection

**Goal**: Verify the upstream model on real speech and lock down its framing and state contract.

1. Run the upstream reference on real speech (plus multimodal/tool inputs if supported) and save golden audio + transcript outputs.
2. Select **Pattern A, B1, B2, or C** from the pattern table.
3. Record the streaming contract:
   - Input format, sample rate (`16000` vs `24000` Hz), unit/frame samples (round `chunk_period_ms` up so silence continuation never outruns real-time), and optional video/image contract.
   - Output sample rate, codec frame rate, acoustic delay, and `DELTA` vs cumulative emission.
   - Initial prefill layout (system prompt, voice prompt / `ref_audio`, persona tokens), exact `prompt_token_ids` length formula, special control tokens (fail fast on `unk_token_id`/`None`), and `silence_continuation_samples` / `silence_continuation_sample_rate_hz` (defaults are `16000`/`16000` — override for 24 kHz or sub-second units).

Deliverable: golden reference artifacts + written contract note.

## Phase 2: Base Pipeline, Registry & `duplex_plugin` Wiring

**Goal**: Register config, model classes, and pipeline topology so `vllm serve --omni` constructs `DuplexOmni`.

1. **Custom HF config** (if custom `model_type`): add `vllm_omni/transformers_utils/configs/<model>.py` calling `AutoConfig.register("<model_type>", <Model>Config)`, register it in `vllm_omni/transformers_utils/configs/__init__.py` (`_CLASS_TO_MODULE`, `__all__`, and eager bottom import), and match `PretrainedConfig.get_text_config(self, decoder: bool = False, encoder: bool = False)` for `mypy-3.10`.
2. **Register architectures** in `_OMNI_MODELS` (`vllm_omni/model_executor/models/registry.py`) and add inter-stage processors in `vllm_omni/model_executor/stage_input_processors/<model>.py` (skip processors for single-stage Pattern B1).
3. **Define `PipelineConfig`** in `vllm_omni/model_executor/models/<model>/pipeline.py` with `duplex_plugin="vllm_omni.model_executor.models.<model>.duplex.plugin.<Model>DuplexPlugin"`, `recompute_preemption="fail"` on stateful AR Stage 0, and `supports_duplex_mrv2=True` on any native MRv2 downstream stage (`input_sources` non-empty). Add `hf_config_predicate` if `architectures` overlaps an older generation.
4. **Register in `OMNI_PIPELINES`** (`vllm_omni/config/pipeline_registry.py`, alphabetical by `model_type`).

Deliverable: config, `_OMNI_MODELS`, and `OMNI_PIPELINES` wired with `duplex_plugin`.

## Phase 3: `DuplexModelPlugin` & `DuplexDataPlane`

**Goal**: Implement the model plugin package under `vllm_omni/model_executor/models/<model>/duplex/`.

```text
vllm_omni/model_executor/models/<model>/duplex/
├── __init__.py          # Import-light
├── plugin.py            # <Model>DuplexPlugin(DuplexModelPlugin) + SessionState
├── data_plane.py        # Subclass of CumulativeAudioTextDataPlane (or DuplexDataPlane)
└── capabilities.py      # Optional DuplexCapabilities builder (plus input.py if non-fixed-frame)
```

1. **Implement `<Model>DuplexPlugin(DuplexModelPlugin)`**:
   - Set class attributes: `plugin_id`, `private_runtime_config_keys`, `silence_continuation_samples`, `silence_continuation_sample_rate_hz`, and wire `self.data_plane = YourDataPlane(encode_audio)` in `__init__`.
   - Implement the six abstract methods: `configure_sampling_params()`, `plan_append()`, `decide_output()`, `create_session_state()`, `capabilities()`, `async prepare_runtime_config()` + `runtime_config_for_update()`.
   - Override optional lifecycle hooks (`configure_server`, `configure_pipeline`, `configure_async_chunk`, tool-call/history hooks) only when needed; buffered commit prefixes live on `PcmAppendBuffer.prepare_commit_prefix()`, not on the plugin.
2. **Implement `DuplexDataPlane`**: subclass `CumulativeAudioTextDataPlane` (`audio_output_is_delta = True` for `RequestOutputKind.DELTA`); override `_project_output()` only for custom transcript/listen/tool metadata.

Complete skeletons, startup validators, and capability matrix: [references/plugin-and-contracts.md](references/plugin-and-contracts.md).

## Phase 4: MRv2 Worker Hooks & Streaming State Isolation

**Goal**: Wire Stage 0 (and downstream stages) on Model Runner V2 (`vllm_omni/worker_v2/`) with strict per-request isolation.

1. **Stage 0 MRv2 hooks** (Patterns A, B1, B2):
   - Real per-step order: `on_requests_finished()` runs **first**, then `model_state.run_preprocess()`, `forward()`, `postprocess_model_output()`, `sample_with_output()`, and `model_state.post_sample()`.
   - Set `has_preprocess = True` (plus `have_multimodal_outputs = True` and `resume_streaming_rows_in_place = True` for lockstep single-stage audio).
   - Check `_omni_is_prefill` (never `span > 1`) and slice with `_omni_num_computed_tokens` / `_omni_prompt_len`; encode full prefill on `seq == 1`, delta unit only on `seq > 1`, and keep retried `(epoch, seq)` idempotent.
   - Accept `**kwargs` (or `model_intermediate_buffer=None`) in `forward()` for MRv2 `_call_model()` and `_dummy_run`.
   - Wire logits masking / in-step audio / post-sample state through the supported MRv2 seams (never legacy V1 bare hooks):
     - `mrv2_custom_sampler(self, sampler) -> tuple[Any, Any] | None` returning `(omni_sampler, rejection_sampler)` (for single-stage audio, return the `sparse_audio` payload with `include_hidden_states=False, owns_multimodal_outputs=True`).
     - `def create_mrv2_model_state(self, vllm_config, encoder_cache, device) -> OmniModelState` as an **instance method** (never `@classmethod`) returning a custom `OmniModelState` subclass.
   - Implement `on_requests_finished(self, finished_req_ids)` on every stateful stage model and custom `OmniModelState` (pass `set(finished_req_ids)` to `super()`).
2. **Stage 1/2 codec isolation** (Patterns A, B2): set `supports_duplex_mrv2=True` on native MRv2 downstream stages, keep one independent streaming codec per `request_id`, preserve acoustic de-delay across chunks, and free finished IDs in `on_requests_finished()`.

Hook signatures, `sparse_audio` contract, CUDA Graph rules, and deploy reference: [references/worker-v2-and-deploy.md](references/worker-v2-and-deploy.md).

## Phase 5: Deploy YAML, Warmup & Import Guard

**Goal**: Ship a deploy YAML that boots cleanly without CLI eager overrides and passes import-boundary guards.

1. Create `vllm_omni/deploy/<model>.yaml` (or `<model>_duplex.yaml` inheriting `base_config`) with explicit `pipeline: <model_type>`, `session_mode: duplex`, `duplex_session:` (`max_sessions`, `warmup_frames`, optional `collect_stats: true` / `replica_policy`), stage `max_num_seqs` matching per-replica capacity, `enable_prefix_caching: false` on placeholder stages, `async_scheduling: false` on lockstep Stage 0, and safe CUDA Graph mode (`enforce_eager: true` or `PIECEWISE`).
2. Add `"vllm_omni.model_executor.models.<model>.duplex"` to **both** `forbidden_prefixes` tuples in `tests/engine/test_duplex_import_boundary.py`.
3. Verify `vllm serve <model> --omni --deploy-config vllm_omni/deploy/<model>.yaml` (and optional `--duplex-engine-backend-as-process`).

## Phase 6: Parity Ladder, Tests & Eval

Climb in order; do not move up until the current rung passes. Wire markers, CI levels, and Buildkite via [vllm-omni-test](../vllm-omni-test/SKILL.md).

1. **Rung 1 — HF reference**: golden audio + transcript captured on real speech.
2. **Rung 2 — L1 CPU contracts & boundaries** (`core_model` + `cpu`): import boundaries (`test_duplex_import_boundary.py`, `test_duplex_protocol_facade.py`, `test_protocol_import_boundary.py`), plugin validators, deploy YAML parsing, PCM rollback, and MRv2 per-request isolation unit tests.
3. **Rung 3 — L2/L3 single-session live voiced E2E** (`core_model` / `advanced_model` + `omni` + `@hardware_test`): `/v1/realtime?duplex=1` handshake + real speech triggering voiced synthesis (`abs(pcm).max() > 0.05`, coherent transcript, malformed-input recovery). Silent warmup alone does not pass this rung.
4. **Rung 4 — L3 concurrent voiced isolation & admission**: staggered distinct spoken sessions up to effective capacity actively emitting speech with zero crosstalk, `resource_exhausted` at `capacity + 1`, and clean slot reuse after close.
5. **Rung 5 — L4 extended & eval** (`full_model` + `omni` + `@hardware_test`): barge-in / tools (if supported), `vllm bench omni-duplex-eval --omni`, and busy RTF (`collect_stats: true`).

Wire contract, handshake pitfalls, L1 pytest command, 9-point GPU acceptance bar, and troubleshooting: [references/realtime-testing-and-eval.md](references/realtime-testing-and-eval.md).

## Phase 7: Recipe, Docs & Pre-PR

- Add the model recipe (`recipes/<Vendor>/<Model>.md` + `recipes/README.md` row) via [add-recipe](../add-recipe/SKILL.md) and update `docs/models/supported_models.md` with validated backends only.
- Do not add new model-specific Python scripts under `examples/` (`.claude/skills/precheck-pr/references/examples-policy.md`).
- Run [precheck-pr](../precheck-pr/SKILL.md), `pre-commit run --files ...`, `git diff --check`, and sign commits with `git commit -s`; include pinned revision, hardware, deploy YAML, and voiced E2E / eval evidence (`.claude/skills/review-pr/references/checks/model-addition-checklist.md`).

## Integration Checklist

```text
Invariants
- [ ] I1 <model>.duplex in test_duplex_import_boundary.py; protocol.duplex only; ABCs
- [ ] I2 No per-session state on plugin; private keys, path traversal & live updates guarded
- [ ] I3 Reuses DefaultDuplexModelSessionState, FixedFramePcmAppendBuffer, CumulativeAudioTextDataPlane
- [ ] I4 DuplexFence keyword args; recompute_preemption="fail" on stateful AR Stage 0
- [ ] I5 Per-request MRv2 isolation, on_requests_finished(), no hot-loop GPU->CPU syncs
- [ ] I6 Explicit session_mode: duplex; honest capabilities; asyncio.to_thread() for blocking I/O
- [ ] I7 Signatures verified in source; voiced E2E + concurrency proved (not silent warmup)

Phases
- [ ] 1 HF reference golden audio/transcript; Pattern A/B1/B2/C & unit contract locked
- [ ] 2 Custom config (if any), _OMNI_MODELS, PipelineConfig.duplex_plugin, OMNI_PIPELINES
- [ ] 3 <model>/duplex/ passes load_duplex_plugin, sampling & replica policy validators
- [ ] 4 MRv2 hooks (preprocess, custom sampler / OmniModelState, on_requests_finished) isolated
- [ ] 5 Shipped deploy YAML boots with warmup & safe CUDA Graph mode
- [ ] 6 Parity ladder rungs 1-5 pass (L1 CPU + voiced single/concurrent E2E + eval)
- [ ] 7 Recipe, supported_models.md, precheck-pr, pre-commit & DCO git commit -s pass
```

## References

- [references/plugin-and-contracts.md](references/plugin-and-contracts.md) — `DuplexModelPlugin` skeletons, validators, `common/duplex/` helpers, tool hooks, capability matrix
- [references/worker-v2-and-deploy.md](references/worker-v2-and-deploy.md) — MRv2 hooks, `sparse_audio`, `OmniModelState`, `recompute_preemption`, `DuplexSessionRuntimeConfig` table
- [references/realtime-testing-and-eval.md](references/realtime-testing-and-eval.md) — 3-tier wire contract, WS handshake pitfalls, L1–L4 tests, 9-point GPU bar, `omni_duplex_eval`, troubleshooting
- `docs/design/fullduplex.md`, `docs/serving/full_duplex_api.md`, `docs/serving/realtime_duplex_api.md`
- Sibling skills: [add-omni-model](../add-omni-model/SKILL.md), [omni-perf-opt](../omni-perf-opt/SKILL.md), [vllm-omni-test](../vllm-omni-test/SKILL.md), [add-recipe](../add-recipe/SKILL.md), [precheck-pr](../precheck-pr/SKILL.md)
