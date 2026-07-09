# Worker / ModelRunner Review — reference

Supporting material for the `review-worker-runner` skill.

## Key files & roles

| File | Role | Notes |
| --- | --- | --- |
| `worker/base.py` | `OmniGPUWorkerBase` — per-process NVML memory accounting, sleep/wake + CuMem, profiler | **no unit tests** |
| `worker/mixins.py` | `OmniWorkerMixin` — loads omni plugins in worker processes | |
| `worker/gpu_ar_worker.py`, `gpu_generation_worker.py` | worker `init_device` + `self.model_runner = …` | |
| `worker/gpu_model_runner.py` | `OmniGPUModelRunner(GPUModelRunner)` — shared services (M-RoPE, prompt-embeds overlay, `model_intermediate_buffer`, pre/postprocess hooks, talker-MTP, Fish-KV, `_dummy_run`) | ~2145 L; the shared base — cleanups here propagate to NPU for free |
| `worker/gpu_ar_model_runner.py` | `GPUARModelRunner` — two-phase execute/sample, output build, async output, prefix cache, KV-transfer. Also `ExecuteModelState`, `OmniAsyncGPUModelRunnerOutput`, `_ensure_tensor_values` | |
| `worker/gpu_generation_model_runner.py` | `GPUGenerationModelRunner` — non-AR (code2wav) two-phase flow | **imports `ExecuteModelState` from the AR runner → backwards coupling** |
| `worker/omni_connector_model_runner_mixin.py` | connector data plane (recv threads, payload cache, KV transfer, async chunk) | **OUT OF SCOPE** — OmniConnector owners |
| `worker/gpu_memory_utils.py` | NVML per-process memory helpers | |
| `worker/memory_utils.py`, `worker/payload_span.py` | `request_memory_tolerant`; span helpers | **no unit tests** |
| `utils/mm_outputs.py` | `partition_payload_list`, `build_mm_cpu` | correctly shared GPU+NPU — keep single-source |
| `platforms/xpu/worker/*` | 19-line clean subclasses of the GPU runners | |
| `platforms/npu/worker/*` | diverged parallel hierarchy (~0.16 similarity) | OUT OF SCOPE (npu-upgrade skill) |

Two-phase flow (both upstream **and** omni): `execute_model()` runs forward, returns `None`, stores
an `ExecuteModelState`; `sample_tokens()` reads it, samples, builds output. Don't claim omni "added"
two-phase — it didn't.

## Model → feature map (which model needs the model-specific code)

| Feature / code | Gating signal | Model(s) |
| --- | --- | --- |
| talker-MTP (`_init_talker_mtp`, `_talker_mtp_forward`, AR graph capture) | `talker_mtp` / `talker` / `talker_mtp_graph_safe` | fish_speech, qwen3_tts, qwen3_omni |
| Fish-KV attention extensions (`_maybe_attach_attention_metadata_extensions`, `_prewarm_attention_capture_workspaces`) | Fish-KV attention backend | fish_speech (fish-kv) |
| GLM-Image M-RoPE decode fixup (`_calc_mrope_positions`, `_fixup_precomputed_mrope_decode_positions`) | `precomputed_mrope_decode` | glm_image_ar |
| Higgs `omni_query_start_loc` | `supports_omni_query_start_loc` | higgs_audio_v3_talker |
| MammothModa2 `generated_len` (image-grid EOL constraint) | (injected for all; only it consumes) | mammoth_moda2 |
| MiMoAudio req-infos | `__class__.__name__ == "MiMoAudioForConditionalGeneration"` | mimo_audio |
| `qwen3_tts_request_seed` | `sampling_params.extra_args["qwen3_tts_request_seed"]` | qwen3_tts |
| custom-sampler decode history (`_build_model_sampler_output_token_ids`, `_sampling_metadata_for_model_sampler`) | `prefer_model_sampler` | cosyvoice3, higgs_audio3, hunyuanimage3, glm_tts |
| force `output_token_ids` tracking (`_maybe_enable_output_token_ids_for_model_sampler`) | `prefer_model_sampler` + `logitsprocs_need_output_token_ids` | hunyuan_image3 |
| connector init allowlist (`_OMNI_CONNECTOR_INIT_ARCHS`) | `model_config.model_arch` ∈ set | Qwen3OmniMoe, Qwen2_5Omni, CovoAudio, MiMoAudioModel, Qwen3TTSTalker, Qwen3TTSCode2Wav, CosyVoice3, DyninOmni — **AR**: +IndexTTS2Talker; **generation**: +IndexTTS2S2MelDecoder (divergent!) |

## Capability-probe catalogue (~30 `getattr/hasattr(self.model, …)`)

These are the scattered probes that B-align consolidates into a declared capability object on
`OmniModelState`. Inputs/preprocess: `has_preprocess`, `preprocess_batch`, `preprocess_decode_batch`,
`prepare_runner_inputs`. Outputs/postprocess: `has_postprocess`, `postprocess_uses_hidden_states`,
`postprocess_uses_multimodal_outputs`, `postprocess_uses_req_infos`, `make_omni_output`,
`gpu_resident_buffer_keys`. Sampler: `prefer_model_sampler`, `sample`,
`skips_model_sampler_output_token_history`, `logitsprocs_need_output_token_ids`. Positions:
`supports_mrope`, `precomputed_mrope_decode`, `supports_omni_query_start_loc`,
`supports_omni_decode_step_metadata`. Talker-MTP: `talker_mtp`, `talker`, `talker_mtp_graph_safe`,
`mtp_hidden_size`. Connector/lifecycle: `flush_pending_metadata`, `on_requests_finished`,
`get_kv_transfer_metadata`. **Dead:** `supports_sampled_token_ids_cpu_override` (no model declares it).

Rule of thumb: if a probe reads a per-model flag → it's a capability that belongs on the model state;
if the runner branches on the *result* to run model-specific code → that code belongs behind a hook.

## Upstream-vs-omni tells

- **Upstream vocabulary:** `query_start_loc`, `CpuGpuBuffer` (`.cpu`/`.gpu`/`.np` — `.cpu` is a tensor
  *attribute*, never callable), `num_scheduled_tokens`, `slot_mappings`, `spec_decode_metadata`,
  `cudagraph_mode`, `ExecuteModelState`, `inputs_embeds`, `_prepare_inputs`.
- **Omni deltas:** `model_intermediate_buffer` / `additional_information` / `runtime_additional_information`
  (same concept, 3 names), `make_omni_output`, `model.preprocess`/`postprocess`, `talker_mtp`,
  connector (`init_omni_connectors`, `_local_stage_payload_cache`), `OmniKVTransferManager`,
  `prompt_embeds` cross-stage overlay, hardcoded arch names, `request_token_spans`.

## Async-omni-output pipeline map (PR #4476; qwen3_omni-only)

Two async layers: upstream `AsyncGPUModelRunnerOutput` (sampled-token/logprobs non-blocking D2H) +
omni's `OmniAsyncGPUModelRunnerOutput` (defers the whole `OmniModelRunnerOutput` build to a daemon
thread). Flow in `sample_tokens`: ① CPU metadata snapshot (scheduler_output `replace()`-copy,
req_ids, query_start_loc clone, …) → ② gate `_should_use_async_omni_output()` (async scheduling ∧
no prefix cache ∧ no spec decode ∧ `async_chunk` ∧ no routed-experts ∧ model flags) → ③ **eager
postprocess** on live GPU tensors (talker writes `hidden_states['last']` for next step's preprocess
— cannot defer) → ④ GPU payload snapshot: clone-on-producing-stream → dedicated copy stream D2H →
event (`_snapshot_tensor_payload_to_cpu_async`) → ⑤ construct async output (starts daemon builder
thread) → ⑥ `input_batch.set_async_sampled_token_ids(...)` (next step's sampler-history consumer).
Engine calls `get_output()` → join → re-raise background exception → `super().get_output()`.
Perf rationale: moves omni output construction off the decode critical path (measured
`next_cpu_start − prev_gpu_end` p50 from +0.8ms to −0.8ms). Refactor target:
the `async_omni_output_refactor_design.md` prior-art doc (OmniStepSnapshot / one-builder-two-modes /
AsyncOutputSpec / worker/output/ package).

## Correctness-bug catalogue (confirmed instances — check for recurrences)

| Pattern | Known instance |
| --- | --- |
| `clamp(max=size)` off-by-one (OOB index) | AR `sample_tokens` prompt_token_ids vocab correction |
| Early return shadows careful no-tokens guard (DP hang + kv_connector_no_forward skipped) | generation `execute_model` |
| OOB fallback returns `v[0]` (wrong request's payload, silent) | AR `_unwrap_lists` in combined prefix-cache mm payload |
| In-place mutation of engine-shared `scheduler_output` | AR execute_model `custom_metadata` write (ngram path `replace()`-copies; this doesn't) |
| `bool(value)` on maybe-tensor crash | AR `_is_sparse_audio_marker` |
| Joint asserts force `num_reqs==1` / len-1 list misaligns batch | generation `sample_tokens` tensor + list branches |
| capture≠replay: `_dummy_run` copy missing `has_preprocess` branch | generation `_dummy_run` |
| Silent KV-transfer-metadata drop (bare `except Exception`) | AR execute_model `get_kv_transfer_metadata` |
| `None`-fallthrough to default sampler (wrong tokens for custom sampler) | AR `_sample` `prefer_model_sampler` |
| Stale memoization keyed before data arrives | AR `_request_needs_downstream_stage_payload` (`final_stage_id` None → caches True forever) |

## Known gotchas (verify against these)

- **`typing.NamedTuple` cannot be extended by inheritance** — this is why `ExecuteModelState` is a full
  copy of upstream's, not a subclass. To reduce drift use composition (wrap upstream's tuple) or a
  field-parity test; inheritance only works if both become `@dataclass` (upstream won't).
- **`CpuGpuBuffer.cpu` is a tensor attribute, never callable** — so `if callable(query_start_loc_cpu)`
  guards are dead.
- **`get_model()` unwraps `CUDAGraphWrapper`/`UBatchWrapper`; `self.model` relies on `__getattr__`
  delegation** — `isinstance`/`supports_X` must use `get_model()`.
- **FA3 `scheduler_metadata` size = `1 + round_up(batch,4)*4`, split-count-independent** (verified by
  real run) — omni's resize to `max_num_seqs*max_num_splits+1` is a stale v0.16 over-provision (~10×).
- **`torch.Event()` may be CPU-backed**; upstream async output uses `torch.cuda.Event()` — verify
  device backing before trusting `.record()`/`.synchronize()` to gate GPU copies.
- **Async-output ↔ sampler-history coupling:** `OmniAsyncGPUModelRunnerOutput.__init__` writes
  `sampled_token_ids_cpu` + `async_copy_ready_event` onto `input_batch`; next step
  `_build_model_sampler_output_token_ids` reads + syncs them. Invisible, untested — move together.
- **Two-places coupling:** `_OMNI_CONNECTOR_INIT_ARCHS` (runner) **and**
  `omni_scheduling_coordinator._FULL_PAYLOAD_INPUT_STAGES` must be kept in lock-step — forgetting the
  latter is a **silent Stage-1 consumer hang**.
- **CUDA-graph capture == replay:** `_dummy_run` `has_preprocess` buffer path must mirror `_preprocess`.
- **Deferred-builder thread boundary:** the async output builder runs `_build_omni_model_runner_output_from_snapshot`
  on a daemon thread — any read of `self.requests`/`input_batch`/`model_intermediate_buffer` or write of
  connector accumulation state inside its call graph is a potential race. Verify each site's actual
  callers before claiming a race (e.g. `_resolve_global_request_id` turned out main-thread-only via
  `execute_model`); the confirmed live access is the `accumulate_full_payload_output` block.
- **Eager side-effects can't defer:** talker postprocess writes `hidden_states['last']` read by the
  *next* step's preprocess — it must run before snapshotting; only pure output assembly may defer.
- **Systemic docstring gap:** ~120 omni-added methods lack docstrings — report as ONE count-per-file
  sweep item, not per-line findings; keep *wrong* (drifted) docstrings separate as bugs.
- **Anchor hygiene:** inline comment insertions shift line numbers — after annotating, re-grep every
  anchor cited in the rollup/design docs and refresh (`grep -oE "file\.py:[0-9]+"` over the doc,
  verify each against the source).

## Tests & CI

- CPU-only L1/L2 suite in `tests/worker/` (`test_gpu_ar_model_runner.py`,
  `test_gpu_generation_model_runner.py`, `test_omni_gpu_model_runner.py`, `test_omni_connector_mixin.py`,
  `test_process_gpu_memory.py`). Markers: `core_model` = L1&L2, `advanced_model` = L3, `full_model` = L4;
  `omni`/`tts`/`diffusion`. Use the `vllm-omni-test` skill to add coverage.
- Untested (add characterization tests before refactor): `base.py`, `memory_utils.py`, `payload_span.py`.

## Rollup audit template

```markdown
# Worker / ModelRunner Audit

**Level 1 = error type** (model-specific issues listed individually). **Level 2 = `file:line` — description.**
Branch `<branch>`; comments-only, no behaviour change.

## Correctness — <bug class> (severity-tagged, FIRST)
## Dead / unused code
- `file:line` — <what> — <verified how> — <fix>.
## Stale over-provisioned workaround (verified)
## Deprecated / legacy remnant
## Possibly-redundant subsystem: <name>
## Divergent duplicate — needs merge (rfc8 A1)
## Implicit cross-phase state
## Base method reaching subclass-only attributes
## Wrapper-vs-raw model (get_model())
## Dead defensive guard
## Silent failure / swallowed exception
## Async / threading correctness
## Fork-fragility (rebase hazard)
## Redundant sync / D2H / perf
---
## Model-specific: talker-MTP baked into the generic runner
## Model-specific: Fish-KV attention …
## Model-specific: GLM-Image M-RoPE fixup …
## Model-specific: Higgs omni_query_start_loc …
## Model-specific: MammothModa2 generated_len …
## Model-specific: MiMoAudio class-name check …
## Model-specific: connector arch allowlist (divergent) …
## Model-specific: scattered getattr(self.model,…) probes …
---
## Naming inconsistency (additional_information)
## Stale / missing docstring
## Wrong type annotation
## Import inside function (hoist vs keep-lazy)
## Unclear signature (*args/**kwargs)
## Over-long / should-split function
## Fragile hack
## Missing tests (untested modules)
---
## Suggested order
1. Now (low risk, CPU-testable): dead/stale, deprecated, dead guard, silent-failure, wrapper-vs-raw,
   naming/docs/type-hints, non-lazy import hoists. Guard with tests/worker/; add missing char tests first.
2. Investigate (may unlock large removals): prompt_embeds subsystem; process-level memory estimation.
3. Structural (MR-V2 / B-align): merge duplicates + evict every Model-specific type into OmniModelState.
```
