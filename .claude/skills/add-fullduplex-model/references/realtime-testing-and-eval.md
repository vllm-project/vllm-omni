# Realtime Wire Contract, Testing, Eval & Troubleshooting

## 1. Three-Tier Wire Codec & Normative `/v1/realtime?duplex=1` Contract

The wire codec (`docs/design/fullduplex.md`, `docs/serving/realtime_duplex_api.md`) is split into three tiers:
- **Tier 1 (`vllm_omni/protocol/realtime/`)**: Pure OpenAI Realtime vocabulary.
- **Tier 2 & Tier 3 (`vllm_omni/protocol/duplex/`)**: OpenAI extensions (`session.created` + `resume_token`, `input_audio_buffer.append` + `video_frames`) and vLLM-Omni duplex events/commands (`response.listen`, `response.speak`, `playback.ack`, `session.stats.get`, etc.).
- **Strict import rule**: Engine, entrypoint, and model duplex code imports **only** `vllm_omni.protocol.duplex`, never `vllm_omni.protocol.realtime` directly (`tests/protocol/duplex/test_duplex_protocol_facade.py`).

### Key Server Event Lifecycle

| Public Event | Trigger & Ordering | Cardinality |
|--------------|--------------------|-------------|
| `session.created` | First event for a new session, before any held response output | Once per new session |
| `session.resumed` | Emitted on reattachment before replayed events | Once per resumed attachment |
| `session.updated` | Emitted when a client `session.update` opens/configures the session or after engine ACK of a live update | Once per accepted client `session.update` |
| `response.created` | First event for a visible assistant response; precedes item/speak/audio/text events | Exactly once per visible response |
| `response.speak` | Model selected speak (e.g., MiniCPM-o 4.5, Nemotron VoiceChat); after `response.created`, no later than first audio delta | At most once per response |
| `response.output_audio.delta` | Ordered audio chunk from final stage data plane (do **not** wait on short `response.audio.delta` — server emits `response.output_audio.*` only) | Zero or more |
| `response.output_audio_transcript.delta` | Append-only non-overlapping transcript delta (paired with audio delta or transcript-only; server emits `response.output_audio_transcript.*` only) | Zero or more |
| `response.output_audio.done` | Closes audio stream after final delta and before terminal events | At most once |
| `response.output_audio_transcript.done` | Exact concatenation of all transcript deltas for the response | At most once (non-empty transcript) |
| `response.function_call_arguments.done` | Published tool call recorded in session ledger before emission (`call_id`, `name`, `arguments`) | Zero or more |
| `response.output_item.done` / `conversation.item.done` | Finalize assistant item before `response.done` | At most once each |
| `response.done` | Terminal event for every created response (carries `response.metadata.vllm_omni.stage_metrics`) | Exactly once per created response |
| `rate_limits.updated` | Emitted immediately after `response.done` | Once per `response.done` |
| `response.listen` | Model-listen decision at a unit boundary (Pattern A), silent append in turn-based / non-native paths (`runner.py:1967–1983`), or silence-only / empty commit (`runner.py:3505–3511`); may occur without `response.created` | Zero or more per session |
| `session.closed` | Final event emitted after teardown attempt (note: if stage cleanup failed, the admission slot is retained for reaper retry) | Once per session |

### Wire Pitfalls to Watch For

1. **Standard `pcm16` input is resampled to 16 kHz**: `vllm_omni/protocol/duplex` resamples standard `pcm16` appends to 16 kHz for 16 kHz models. Models expecting **24 kHz** audio (Haru-4B, PersonaPlex) must have clients declare `input_audio_format: "pcm_f32le"`, `sample_rate_hz: 24000`, and send `format: "pcm_f32le"` + `sample_rate_hz: 24000` (see `tests/e2e/online_serving/test_haru_duplex.py`).
2. **Initial session configuration before streaming**: Pass `ref_audio`, `voice`, `instructions`, `watermarking`, and creation-time `extra_body` options in the opening `session.update` (with `autostart=false` if connecting with a `?model=` query param) — live `session.update` rejects changes to `model`, `ref_audio`, `watermarking`, post-audio `voice`, and fixed model options.
3. **Output buffer overflow**: Slow WebSocket consumers hitting `max_pending_output_bytes_per_session` (default `2 MiB`) or `max_pending_output_events_per_session` (`512`) fail the active response and close that session. Raise the byte limit in the deploy YAML when `session.created` echoes large `ref_audio` payloads (`haru.yaml` sets `64 MiB`).
4. **Output modality fail-closed**: Unknown output modalities fail with `unsupported_response_modality` — never silently project as text.
5. **Opening handshake & warmup settling**: Clients must send `session.update` first (waiting on `recv()` first times out with `config_timeout`), then drain **both** `session.created` and `session.updated` before sending audio or malformed-input probes. Wait for background `warmup_frames` to complete before asserting `max_sessions` admission or measuring first-frame latency.

## 2. L1–L4 Test Matrix (Aligned with `vllm-omni-test`)

Follow [vllm-omni-test](../../vllm-omni-test/SKILL.md) for markers, naming, and Buildkite wiring.

### L1: CPU Contract & Import-Boundary Tests (`@pytest.mark.core_model`, `@pytest.mark.cpu`)

1. **Import boundaries** (`tests/engine/test_duplex_import_boundary.py`, `tests/protocol/duplex/test_duplex_protocol_facade.py`, `tests/protocol/realtime/test_protocol_import_boundary.py`):
   - Add `"vllm_omni.model_executor.models.<model>.duplex"` to both `forbidden_prefixes` lists in `test_duplex_import_boundary.py`.
2. **Plugin & data-plane unit tests** (`tests/model_executor/models/<model>/test_<model>_duplex_plugin.py`):
   - `load_duplex_plugin()` + `validate_duplex_plugin_sampling()` + `validate_duplex_replica_policy()` pass.
   - `DuplexCapabilities`: `supports_input_append=True`, honest `supports_barge_in`, `chunk_period_ms`, `supports_multi_session` derived from `max_sessions`, and `supports_multi_session_same_replica` truthfully reflecting single-replica isolation (`False` if a replica holds at most one session).
   - `silence_continuation_samples` / `silence_continuation_sample_rate_hz` produce a payload accepted by `plan_append()`.
   - `private_runtime_config_keys`, path-traversal voice names, invalid `ref_audio` URIs, and forbidden live updates raise `DuplexRuntimeConfigError`.
   - `PcmAppendBuffer`: exact frame grouping, residual zero-padding on commit, FIFO rollback on failure, non-finite / wrong-rate / duplicate `operation_id` rejection.
   - `plan_append()`: exact `prompt_token_ids` length on `seq == 1` vs `seq > 1` (plus `video_frames` vision tokens if applicable).
   - `decide_output()`: emits `DuplexOutputDecision(DIRECT_RESPONSE)` whenever a non-final stage short-circuits the pipeline (e.g., MiniCPM listen, AURA silence, JoyAI non-forwarded observations, Nemotron stage-0 text/function decisions), and returns `None` otherwise (e.g., Haru, PersonaPlex, normal Qwen turns).
   - `DuplexDataPlane`: cumulative-to-delta slicing (or `audio_output_is_delta=True` chunk joining), no replay, `close_session()` cleans matching request IDs.
3. **Deploy config & MRv2 worker hook tests** (`tests/config/test_<model>_duplex_*deploy.py`, `tests/worker_v2/`, `tests/worker/test_native_duplex_hooks.py`):
   - Deploy YAML resolves `session_mode: duplex` and valid `duplex_session` / stage `max_num_seqs` (note: stage-level `enable_prefix_caching: false` is parsed into `stage.engine_extras["enable_prefix_caching"]`, not a top-level `StageDeployConfig` field).
   - Stage 0 `preprocess` handles `_omni_is_prefill=True` with `span == 1`, `_omni_num_computed_tokens` / `_omni_prompt_len` slicing (or translated offsets in a custom `OmniModelState`), retry idempotency, and `on_requests_finished()` without touching sibling requests.
   - Stage 1/2 interleaved multi-request streaming preserves independent codec histories.

Run L1 CPU suite locally (always pass `-m "core_model and cpu"` so GPU tests are skipped; add `-o addopts=""` if `pytest-xdist` is not installed):

```bash
pytest -q -m "core_model and cpu" \
  tests/engine/test_duplex_import_boundary.py \
  tests/protocol/duplex/test_duplex_protocol_facade.py \
  tests/protocol/realtime/test_protocol_import_boundary.py \
  tests/engine/duplex/test_duplex_plugin.py \
  tests/engine/duplex/test_duplex_deploy_config.py \
  tests/config/test_<model>_duplex_*deploy.py \
  tests/model_executor/models/<model>/
```

Repo pre-commit gotchas on new duplex files (`pre-commit run --files ...`, `git diff --check`, DCO `git commit -s`):
- Use `# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project` on all new `.py` files (`check-spdx-header`).
- Inside `vllm_omni/`, use `pybase64`, `regex`, `vllm_omni.transformers_utils.repo_utils.hf_api()`, and `current_omni_platform` instead of stdlib `base64`/`re`, direct `huggingface_hub` download imports, or `torch.cuda.*` (`check-forbidden-imports`, `check-torch-cuda-call`).
- Match Python 3.10 superclass signatures (`get_text_config(self, decoder: bool = False, encoder: bool = False)`, `super().on_requests_finished(set(finished_req_ids))`) for `mypy-3.10`.

### L2–L4: Live GPU E2E (`pytest.mark.omni` + `@hardware_test`)

References: `tests/e2e/online_serving/test_haru_duplex.py`, `test_personaplex_duplex.py`, `test_nemotron_voicechat_duplex.py`, `test_minicpmo_4_5_duplex.py`, `test_qwen3_omni_duplex_tools.py`.

- **L2 (`core_model` + `omni` + `@hardware_test`)**: WebSocket `/v1/realtime?duplex=1` protocol smoke (`session.created` -> `session.updated` -> `session.closed`, zero errors).
- **L3 (`advanced_model` + `omni` + `@hardware_test`)**: Staggered live paced streaming up to the deploy config's `max_sessions` (assert finite non-silent audio amplitude + non-empty transcript, zero errors), continuous overlap during playback, admission rejection (`resource_exhausted`) at `max_sessions + 1`, and replacement admission after close without state leakage.
- **L4 (`full_model` + `omni` + `@hardware_test`)**: Extended duration, soft interrupt / barge-in, tool-call round-trip, malformed-input resilience, and perf configs under `tests/dfx/perf/tests/test_<slug>_omni_duplex_eval.json` (note: `tests/e2e/accuracy/omni_duplex_eval/test_omni_duplex_eval_ci.py` is a CPU regression suite for the eval CI config/driver, not a live GPU L4 test).

E2E test hygiene:
- Inline reference audio as `data:audio/wav;base64,...` (CI hosts have no external URL access).
- Rely on the `omni_server` harness `/health` readiness gate plus `duplex_session.warmup_frames` — never add `time.sleep` for warmup.
- Keep one `OmniServerParams` set per file to avoid mid-module server restart races.

## 3. Performance Metrics & `omni-duplex-eval`

Never merge engine and client clocks into one timeline:
- **Engine per-segment stage metrics**: Wire `response.done` nests engine metrics under `response.metadata.vllm_omni.stage_metrics` (`vllm_ttft_ms`, `vllm_tpot_ms`, `vllm_itls_ms`); `EventCollector.timing_summary(after_s=...)` exposes raw per-stage snapshots in `["stage_metrics"]` (when present) and summary counters in `["request_metrics"]` (`vllm_omni/clients/duplex.py`).
- **Busy RTF & stall ratio (RFC #8325)**: Enable `duplex_session.collect_stats: true` in the deploy YAML, query `session.stats.get` on the WebSocket (single-host `CLOCK_MONOTONIC`), and summarize with `vllm_omni.benchmarks.duplex_session_metrics`.
- **Client receive-clock metrics**: Use `vllm_omni.clients.duplex.EventCollector.timing_summary()` (`audio_output` inter-chunk gaps, chunk durations, `response_created_to_first_audio_ms`, `commit_to_first_audio_ms`).
- **Automated duplex quality + perf benchmark**: Run the registered CLI command (writes `duplex_metrics.json` and prints per-stage percentiles with `--print-stage`):
  ```bash
  vllm bench omni-duplex-eval --omni generate --model "$MODEL" --response-root "$OUT_DIR" --print-stage
  ```

## 4. Mandatory 9-Point Remote GPU Acceptance Bar

Before claiming a new full-duplex model is ready on GPU, collect fresh evidence for all nine items:

1. **Shipped deploy YAML boots cleanly**: `vllm serve <model> --omni --deploy-config vllm_omni/deploy/<model>.yaml` boots with startup warmup and no local `--enforce-eager` override.
2. **Liveness + duplex readiness**: `/health` returns HTTP 200 and `/v1/realtime?duplex=1` completes the opening handshake (`session.created` + capabilities) with zero errors.
3. **Honest capabilities**: Advertises accurate `implementation_level` (`model_native_duplex` vs `turn_based_duplex`), `supports_input_append=true`, `chunk_period_ms`, and accurate `supports_multi_session` vs `supports_multi_session_same_replica` flags.
4. **Live paced audio & transcript (voiced path)**: Silent startup warmup does not exercise speech synthesis/vocoders; feed real spoken PCM at native sample rate and format and verify finite, non-silent audio deltas (e.g., `abs(pcm).max() > 0.05`) and non-empty, coherent transcript deltas.
5. **Continuous overlap during playback**: Appending live user audio while assistant audio streams does not cancel or corrupt a resumable request (or triggers clean barge-in for Pattern C when configured).
6. **Concurrent admission, voiced isolation & backpressure**: Simultaneous staggered sessions up to effective admission capacity (the deploy YAML's `max_sessions` under `legacy`, or bounded by `max_sessions_per_replica` under `replica_policy: least_sessions`) with distinct spoken inputs actively emitting voiced speech produce isolated, non-bleeding outputs; exceeding capacity is rejected with `resource_exhausted`.
7. **Isolated cleanup & slot reuse**: Closing a session releases only its stage request and GPU codec state; a replacement session opens and runs with a different voice/persona without leakage.
8. **Malformed input resilience**: Invalid base64, wrong sample rate, or non-finite PCM returns one typed `error` event without poisoning subsequent valid appends.
9. **Clean tree & contracts**: Focused L1 CPU tests (`-m "core_model and cpu"`), import-boundary tests, and `git diff --check` pass.

> **Not passing**: Empty audio, all-zero waveforms, `NaN`/`Inf` samples, or a session emitting only `response.listen`.

## 5. Troubleshooting Common Failures

| Symptom | Likely Root Cause | Fix |
|---------|-------------------|-----|
| `AttributeError: '<Model>' object has no attribute 'speculative_config'` in `init_omni_model_state` | `create_mrv2_model_state` was declared `@classmethod` instead of an instance method (argument shift when called via `getattr(type(model), ...)(model, vllm_config, ...)`) | Define `def create_mrv2_model_state(self, vllm_config, encoder_cache, device) -> OmniModelState:` as a regular instance method |
| `TypeError: forward() got an unexpected keyword argument 'model_intermediate_buffer'` during `_dummy_run` | MRv2 `OmniModelRunner._call_model()` passes `model_intermediate_buffer` via `self.model(**model_inputs)` | Add `**kwargs` (or `model_intermediate_buffer=None`) to `forward()` |
| `KeyError` on `pipeline: <name>` before startup | Pipeline missing from `OMNI_PIPELINES` | Register `"<name>": YOUR_PIPELINE` in `vllm_omni/config/pipeline_registry.py` |
| Startup raises `requires session_mode: turn or duplex` or `requires a deploy config with session_mode: duplex` | Duplex deploy YAML omitted `session_mode: duplex` (even if `default_session_mode` is set, `load_deploy_config()` defaults an omitted YAML field to `"turn"`) | Add explicit `session_mode: duplex` to the duplex deploy YAML (or inherited `base_config`) |
| Startup raises `model_runner v2 supports session_mode 'turn' only` on downstream stage | Native MRv2 downstream stage (`input_sources` non-empty) missing `supports_duplex_mrv2=True` | Set `supports_duplex_mrv2=True` on that `StagePipelineConfig` once duplex MRv2 hand-off is implemented |
| Startup fails in `load_duplex_plugin` or `validate_duplex_plugin_sampling` | Missing `plugin_id` / `DuplexDataPlane`, or `configure_sampling_params()` returned wrong length/type | Subclass `DuplexModelPlugin` + `DuplexDataPlane`, return `tuple` of cloned `SamplingParams` matching `len(defaults)` |
| Startup fails in `validate_duplex_replica_policy` | `replica_policy: least_sessions` with `max_sessions_per_replica > 1` while `supports_multi_session_same_replica` is `False` | Set `supports_multi_session_same_replica = max_sessions > 1` only when single-replica isolation is verified (otherwise keep `max_sessions_per_replica: 1`) |
| Startup warmup or silence continuation fails `decode_pcm_f32le_payload` | Plugin inherited default `16000` samples / `16000 Hz` silence unit | Override `silence_continuation_samples` and `silence_continuation_sample_rate_hz` on your plugin |
| Microphone appends disabled or rejected | `DuplexCapabilities` left `supports_input_append=False` | Set `supports_input_append=True` in `capabilities()` |
| 24 kHz model rejects client audio format | Client sent default `pcm16` (resampled to 16 kHz by wire codec) | Send explicit `format: "pcm_f32le"`, `sample_rate_hz: 24000` |
| Client hangs waiting for audio/transcript deltas | Client filters on short `response.audio.*` / `response.audio_transcript.*` names | Listen for `response.output_audio.*` and `response.output_audio_transcript.*` |
| `session.created` closes immediately with output buffer overflow | Echoed `ref_audio` exceeds default `2 MiB` `max_pending_output_bytes_per_session` | Raise `duplex_session.max_pending_output_bytes_per_session` in deploy YAML (e.g. `67108864`) |
| MRv2 silently skips audio generation or listen masking | Implemented bare `preprocess_batch` / `prepare_duplex_sampling` / `post_sample_multimodal_outputs` without MRv2 seams | Use `preprocess_batch_mrv2`, `mrv2_custom_sampler -> (OmniSampler, rejection_sampler)`, and/or custom `OmniModelState` (`create_mrv2_model_state` / `create_omni_model_state` + `post_sample`) |
| 1-frame resumable append treated as decode or chunked prefill replays first chunk | Stage 0 checked `span > 1` instead of `_omni_is_prefill`, or looked up `duplex_token_offset`/`duplex_prompt_len` without translating `_omni_num_computed_tokens`/`_omni_prompt_len` | Set `has_preprocess=True`, inspect `_omni_is_prefill`, and slice with `_omni_num_computed_tokens`/`_omni_prompt_len` |
| Garbage audio after first append | `prompt_token_ids` length mismatch, trailing pad instead of leading pad, or `enable_prefix_caching: true` on placeholder IDs | Match prompt budget (leading pad for Pattern A); set `enable_prefix_caching: false` |
| Audio repeats from step 0 or stutters | Cumulative Stage 1 output unsliced, or `audio_output_is_delta` mismatched with `RequestOutputKind` | Subclass `CumulativeAudioTextDataPlane`; set `audio_output_is_delta = True` iff final stage uses `RequestOutputKind.DELTA` |
| Crosstalk under concurrent sessions | Shared codec instance across requests, or batched `forward()` read row `0` | Allocate independent codec per `request_id`; route metadata per row; clean up in `on_requests_finished()` |
| `test_duplex_import_boundary.py` or `test_duplex_protocol_facade.py` fails | Turn-based stack imported duplex module, framework eagerly imported `<model>.duplex`, or code imported `vllm_omni.protocol.realtime` directly | Keep plugin loading behind `PipelineConfig.duplex_plugin`; import wire helpers only from `vllm_omni.protocol.duplex` |
