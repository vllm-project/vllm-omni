# Model Runner V2 (`worker_v2/`), Streaming Isolation & Deploy YAML

This reference covers worker-side Stage 0/1/2 hooks on Model Runner V2 (`vllm_omni/worker_v2/`), resumable request handling, multi-session state isolation, deploy YAML configuration, replica placement, and process mode.

## 1. Resumable vs Ephemeral Scheduler Lifecycles

**Resumable (Patterns A, B1, B2 — `DuplexStageSubmission.resumable=True`)**:

```text
RUNNING
  -> segment stop (max_tokens=1, stop_token_ids, or EOS; ignore_eos=True if EOS is semantic)
WAITING_FOR_STREAMING_REQ   (KV / Mamba state retained in worker)
  -> next append from DuplexSessionRunner (same duplex-s.<b64url>.e.<epoch>.r.stage0 id)
RUNNING
  -> session close or epoch cancel (CancelResponse / BargeIn)
FINISHED                    (KV released; on_requests_finished() fired on all stages)
```

**Ephemeral (Pattern C — `resumable=False`, e.g., Qwen3-Omni)**:
- Each committed utterance opens a fresh turn-scoped request ID (`duplex-s.<b64url>.e.<epoch>.r.stage{N}-turn{T}`) with full or policy-selected conversation history.
- Base MRv2 stages and Thinker prefix caching work unchanged.

**Preemption rule (`recompute_preemption="fail"`)**:
- Whenever Stage 0 maintains worker-local streaming encoder, audio latent, or recurrent state that cannot be reconstructed by recomputing placeholder `prompt_token_ids`, declare `recompute_preemption="fail"` on `StagePipelineConfig` (`vllm_omni/config/stage_config.py`).

## 2. Stage 0 MRv2 Hooks (`vllm_omni/worker_v2/`)

Model Runner V1 is rejected on vLLM 0.34 (`resolve_stage_model_runner()` in `vllm_omni/config/stage_config.py` requires `model_runner: v2`). While `vllm_omni/worker/` still holds shared helpers (`base.py`, `sparse_audio.py`, etc.), all AR/Generation stage execution runs through `vllm_omni/worker_v2/omni_model_runner.py`, `omni_ar_model_runner.py`, and `model_states/omni_model_state.py`.

### Real MRv2 Per-Step Execution Order

```text
1. model.on_requests_finished(finished_req_ids)      # Runs FIRST (before preprocessing) to free finished/aborted state
   + model_state.on_requests_finished(set(finished_req_ids))
2. model_state.run_preprocess(input_batch, model_inputs, req_states)
   # Generic OmniModelState dispatches (when model.has_preprocess is True):
   #   a. model.preprocess_prefill_rows_mrv2(...)    # Optional whole-batch prefill
   #   b. model.preprocess_batch_mrv2(req_infos=prefill_infos, device=embeds.device)  # Optional prefill batch
   #   c. model.preprocess_decode_batch_mrv2(...)    # Optional 1-token decode batch
   #   d. model.preprocess(input_ids, input_embeds, **info)  # Per remaining request
   # Or override run_preprocess() in a custom OmniModelState via:
   #   def create_mrv2_model_state(self, vllm_config, encoder_cache, device) -> OmniModelState  (instance method, NOT @classmethod!)
3. model.forward(..., **kwargs)                      # Backbone (PagedAttention / Mamba-Hybrid); MUST accept **kwargs (model_intermediate_buffer)
4. model_state.postprocess_model_output(...)         # Extracts (text_hidden, multimodal_outputs)
   (+ optional model_state.run_postprocess when postprocess_requires_multimodal_outputs=True)
5. sample_with_output(sampler, ...)                  # Standard sampler or OmniSampler from model_state.custom_sampler(sampler)
   #   - The generic model state delegates sampler registration to model.mrv2_custom_sampler(sampler)
   #   - Override model_state.compute_sample_logits to apply request-local logit rules (e.g. HaruModelState)
   #   - Override OmniSampler.__call__ for logits masking (e.g. MiniCPMO45DuplexSampler -> prepare_duplex_sampling)
   #   - Override OmniSampler.sample_step returning OmniSamplingOutput(multimodal_outputs=...) for in-step audio
6. model_state.post_sample(input_batch, text_hidden, sampled_token_ids, multimodal_outputs, ...)
   # Optional post-sample hook on a custom OmniModelState (e.g. PersonaPlexModelState)
```

Use the MRv2 hooks above for batch preparation, sampling, and audio output. Legacy callbacks such as `preprocess_batch`, `prepare_duplex_sampling`, `consume_sampled_token_ids_cpu_override`, and `post_sample_multimodal_outputs` require an explicit caller. Define `create_mrv2_model_state(self, vllm_config, encoder_cache, device)` as an instance method because the factory passes the model instance as the first argument.

Haru's model state extends `OmniMambaHybridModelState` to preserve Qwen3.5 recurrent state and clears the temporary request rows in `run_preprocess` before delegating. Its sampler uses standard token sampling and commits each accepted frame, including its audio, in `OmniSampler.sample_step`. `compute_sample_logits` applies Haru's text-stream rules during sampling, while prompt logprobs and sampler profiling use the backbone's stateless logit projection.

### Single-Stage In-Step Audio (`sparse_audio` Contract on MRv2)

When a single-stage Pattern B1 model (`final_output_type="audio"`, `engine_output_type="audio"`, `omni_pooler_payload_include_hidden = False`) emits audio from `OmniSampler.sample_step`, return the `sparse_audio` contract so rows without audio (e.g. partial prefills or warmup rows) get empty pooler payloads instead of misrouted hidden states:

```python
mm_outputs = {
    "model_outputs": chunks,  # list[Tensor], aligned 1-to-1 with audio_req_ids
    "sr": [torch.tensor(sample_rate_hz, dtype=torch.int32) for _ in chunks],
    "meta": {"req_id": audio_req_ids, "sparse_audio": ["1"]},
}
return OmniSamplingOutput(
    sampler_output=sampler_output,
    num_sampled=num_sampled,
    num_rejected=num_rejected,
    multimodal_outputs=mm_outputs,
    include_hidden_states=False,
    owns_multimodal_outputs=True,
)
```

### Critical Stage 0 Rules

1. **Always check `_omni_is_prefill`, never `span > 1`**:
   - A steady-state 1-frame resumable append (Haru, PersonaPlex, Nemotron) arrives with `input_ids.shape[0] == 1` **and** `info["_omni_is_prefill"] is True`. Checking `span > 1` silently misclassifies it as a decode step!
2. **Slice scheduled spans with `_omni_num_computed_tokens` and `_omni_prompt_len`**:
   - Generic `OmniModelState.run_preprocess()` populates `_omni_num_computed_tokens`, `_omni_prompt_len`, and `_omni_is_prefill` (`omni_model_state.py:756–761`). It does **not** populate `duplex_token_offset` / `duplex_prompt_len` unless a custom `OmniModelState` explicitly translates them (as `PersonaPlexModelState` does in `personaplex/model_state.py:57–58`).
   - Under chunked prefill and resumable appends, `input_ids` covers `[offset, offset + span)` where `offset = info["_omni_num_computed_tokens"]` and `prompt_len = info["_omni_prompt_len"]`. Slice `prepared.inputs_embeds[local_offset : local_offset + span]` and verify shape == `span`.
3. **Token-budget alignment**:
   - **Strict exact match (Pattern B1/B2 — Haru, PersonaPlex, Nemotron)**: `prepared.inputs_embeds.shape[0]` matches `scheduler_token_budget` (`prompt_slots` on `seq == 1`, `1` on `seq > 1`). Raise `ValueError` on out-of-range slices.
   - **Leading padding + undersized warning (Pattern A — MiniCPM-o 4.5)**: When `prompt_len > embeds.shape[0]`, prepend **leading** padding (`[pad_embeds, full_req_embeds]`, never trailing) so real audio/vision embeddings sit at the tail `[num_computed_tokens, prompt_len)` immediately before the listen/speak decode position. When `prompt_len < embeds.shape[0]`, warn and truncate (include vision slice budgets when `video_frames` are present).
4. **In-place batch row retention (Pattern B1 lockstep)**:
   - Set `resume_streaming_rows_in_place = True` on the model class when each conversation appends 1 frame per step (`haru/haru.py`) to avoid row churn.
5. **Key state by `request_id` and keep retries idempotent**:
   - `duplex_resource_request_id` is deterministic (`duplex-s.<b64url>.e.<epoch>.r.<role>`) and every new session starts at `epoch = 0`. Isolation across public sessions relies on server-generated unique session UUIDs (`DuplexOmni.open_session` / `api_server`) plus advancing `epoch` on cancel. Do not assume reopening with a manually reused `session_id` produces a different request ID — complete `on_requests_finished()` cleanup before admitting reuse.
   - Cache prepared append results by `(epoch, seq)` so a retried step does not advance a streaming audio encoder twice.
6. **Per-row batch metadata routing**:
   - In batched `forward()` or custom `OmniSampler` / `OmniModelState` post-sample hooks, emit outputs indexed per `req_ids` row — never read `runtime_additional_information[0]` for the whole batch.
7. **Fail-fast unsupported engine flags**:
   - Validate incompatible parallel/runtime options inside `__init__` (see `_check_supported(vllm_config)` in `haru/haru.py`).
8. **Accept `**kwargs` in `model.forward()`**:
   - `OmniModelRunner._call_model()` invokes `self.model(**model_inputs)` and includes `model_intermediate_buffer` even during `_dummy_run` profiling. Any MRv2 model class wrapping a backbone must accept `**kwargs` (or `model_intermediate_buffer=None`) in `forward()`.
9. **Optional `prepare_duplex_sampling()` via `mrv2_custom_sampler` (Pattern A)**:
   - On MRv2, expose `mrv2_custom_sampler(self, sampler) -> tuple[Any, Any] | None` returning `(duplex_sampler, rejection_sampler)` (e.g., `(MiniCPMO45DuplexSampler(sampler, self), None)` when `speculative_config is None` in `minicpmo_4_5/minicpmo_4_5_omni.py:1105–1124`) where `MiniCPMO45DuplexSampler` (`minicpmo_4_5/duplex/mrv2_sampling.py`) invokes `prepare_duplex_sampling()` using `DuplexSamplingRow` (`vllm_omni/model_executor/duplex_sampling.py`).

### `on_requests_finished()` Cleanup Contract

Implement on **every** stateful Stage 0, Stage 1, and Stage 2 model (and custom `OmniModelState` if stateful):

```python
def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
    for req_id in finished_req_ids:
        self._runtime.release_request(str(req_id))
```

Verify in unit tests that releasing `req_a` never mutates or resets `req_b`'s streaming encoder, decoder, or delayed-frame state.

## 3. Stage 1 / Stage 2 Streaming Codec Isolation & MRv2 Gate (Patterns A, B2)

- **Mandatory MRv2 downstream gate (`supports_duplex_mrv2=True`)**: `validate_native_mrv2_session()` (`vllm_omni/config/stage_config.py:1379–1394`) rejects any downstream stage (`input_sources` non-empty) that has `supports_native_mrv2_data_plane=True` and runs on `model_runner: v2` under `session_mode: duplex` unless the stage explicitly declares `supports_duplex_mrv2=True` in `pipeline.py` (see `personaplex/pipeline.py:75`).
- **One independent streaming codec per active request ID**: Size codec pools from `duplex_max_sessions` (propagated from `duplex_session.max_sessions` or `max_sessions_per_replica`). Never share convolution/transformer state across concurrent requests in a batch.
- **Acoustic de-delaying**: Retain trailing raw code frames across connector chunk boundaries when the talker shifts acoustic codebooks relative to text/semantic tokens.
- **Native talker vs vendored eager talker**: See `nemotron_labs_voicechat_duplex.yaml` (Stage 1 native Gemma3 talker on PagedAttention with `native_talker: true` and `async_chunk_prewarm_prompt_len`) and `personaplex_duplex_mrv2.yaml` (MRv2 Stage 0 `post_sample` + Stage 1 Mimi CUDA graphs).
- **CUDA Graph boundary**: Never capture Python-owned per-request generators or dynamic H2D codec copies inside a full CUDA graph. Configure `enforce_eager: true` or `compilation_config: { cudagraph_mode: PIECEWISE }` directly in the shipped deploy YAML.

## 4. Deploy Config Reference (`vllm_omni/deploy/`)

References: `haru.yaml` (Pattern B1), `personaplex_duplex_mrv2.yaml` & `nemotron_labs_voicechat_duplex.yaml` (Pattern B2), `minicpmo_4_5.yaml` / `minicpmo_4_5_duplex_h200.yaml` (Pattern A), `qwen3_omni_duplex_mrv2.yaml` (Pattern C).

```yaml
pipeline: your_model
session_mode: duplex                  # Mandatory: "duplex" or "turn" for any model with duplex_plugin
duplex_session:
  max_sessions: 2                     # Global service session cap
  warmup_frames: 4                    # Throwaway silent frames run before admitting real clients
  max_pending_output_bytes_per_session: 2097152  # Raise (e.g. 67108864) if session.created echoes large ref_audio
  collect_stats: false                # Opt-in busy-RTF & per-turn stage_work telemetry (session.stats.get)
  # Optional multi-replica placement:
  # replica_policy: least_sessions
  # max_sessions_per_replica: 2
  # replica_groups: [{0: 0, 1: 0}, {0: 1, 1: 1}]

async_chunk: true                     # false for single-stage Pattern B1 (e.g. haru.yaml)

stages:
  - stage_id: 0
    max_num_seqs: 2                   # >= max_sessions (or max_sessions_per_replica)
    enable_prefix_caching: false      # Mandatory when prompt_token_ids are placeholders (Pattern A/B); parsed into stage.engine_extras["enable_prefix_caching"]
    async_scheduling: false           # Recommended for lockstep Stage 0 so waiting frames step together
    # If Stage 0 lazy-loads an auxiliary codec/vocoder outside vLLM's KV pool, leave gpu_memory_utilization headroom and check its accelerator dtype.
    ...
```

### Server-Owned `DuplexSessionRuntimeConfig` Fields (`vllm_omni/config/stage_config.py`)

Clients cannot override any of these via `session.create`, `session.update`, or `extra_body`:

| Field | Default | Effect |
|-------|--------:|--------|
| `idle_ttl_s` | `300.0` | Expires inactive attached/detached sessions (`null` disables idle expiry only) |
| `disconnect_grace_s` | `30.0` | Grace window for detached sessions before cleanup |
| `reaper_interval_s` | `5.0` | Cadence for lease expiry and pending cleanup retries |
| `resume_replay_ttl_s` | `60.0` | TTL for replay events retained for resume/takeover |
| `resume_replay_max_bytes_per_session` | `8 MiB` | Caps per-session replay journal size |
| `max_pending_input_bytes_per_session` | `16 MiB` | Byte backpressure before mailbox admission (`input_backpressure`) |
| `max_pending_turns_per_session` | `4` | Caps queued turns awaiting runtime processing |
| `max_pending_output_bytes_per_session` | `2 MiB` | Slow-consumer compact-JSON output budget (overflow closes affected session) |
| `max_pending_output_events_per_session` | `512` | Slow-consumer event count cap (plus 64 KiB / 8-event reserve for errors/terminals) |
| `max_sessions` | `1` | Atomic global admission cap (closing sessions hold slots until cleanup finalizes) |
| `max_tool_calls_per_session` | `4096` | Total pending + terminal tombstone tool-call records per session |
| `replica_policy` | `"legacy"` | `"legacy"` or `"least_sessions"` (pins each session to a stage replica group) |
| `max_sessions_per_replica` | `null` | Per-replica cap when `replica_policy: least_sessions` |
| `replica_groups` | `null` | Optional explicit `[{stage_id: replica_id}]` groups (required for disjoint multi-stage `--api-server-count > 1`; single-stage auto-generated groups are already disjoint) |
| `warmup_frames` | `0` | Startup silent frames before admitting `/v1/realtime?duplex=1` clients (`< 0` disables all warmup) |
| `collect_stats` | `false` | Enables `DuplexStageWorkRecorder` + `session.stats.get` (single-host `CLOCK_MONOTONIC`) |
| `model_custom_options` | `{}` | Server-only model config validated by `plugin.configure_server()` |
| `server_vad_model_path` / `semantic_vad_model_path` | `null` | Local Silero VAD / Smart Turn v3.2 ONNX paths for `turn_detection` |
| `context_keep_media_turns` | `null` | Opt-in L1 candidate history view (Pattern C) |

### Process-Isolated Runtime & Multi-Frontend Notes

- `--duplex-engine-backend-as-process` isolates `DuplexOrchestrator` in a dedicated Python process (separate GIL, session-only — disables `/v1/chat/completions`, POSIX + `spawn`-safe args + MessagePack-compatible session config).
- `--api-server-count K` requires `replica_policy: least_sessions`, disjoint frontend replica ownership (explicit `replica_groups` for multi-stage pipelines; single-stage auto-generated groups are already disjoint), and plugins with `supports_session_resume=False`.
