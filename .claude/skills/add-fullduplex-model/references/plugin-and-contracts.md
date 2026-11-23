# `DuplexModelPlugin`, Reusable Helpers & Contracts

This reference covers the unified plugin contract in `vllm_omni/engine/duplex/plugin.py`, startup validation in `DuplexOmniEngine` / `DuplexSessionManager`, reusable helpers in `vllm_omni/model_executor/common/duplex/`, and the capability matrix across Patterns A, B1, B2, and C.

## 1. Pipeline Wiring & Startup Validation Sequence

Wire `duplex_plugin` in `vllm_omni/model_executor/models/<model>/pipeline.py` and register the pipeline in `OMNI_PIPELINES` (`vllm_omni/config/pipeline_registry.py`):

```python
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

YOUR_MODEL_PIPELINE = PipelineConfig(
    model_type="your_model",
    model_arch="YourStage0ForConditionalGeneration",
    default_deploy_config_name="your_model.yaml",
    duplex_plugin="vllm_omni.model_executor.models.your_model.duplex.plugin.YourDuplexPlugin",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="your_stage0",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            final_output=True,                 # True if single-stage (Pattern B1)
            final_output_type="audio",         # or "text" if Stage 0 emits text + downstream audio
            engine_output_type="audio",        # or "latent" if feeding downstream stages
            sampling_constraints={"detokenize": False},
            recompute_preemption="fail",       # Required when worker holds non-recomputable state
        ),
    ),
)
```

Exact fail-fast startup sequence executed by `api_server` -> `DuplexOmniEngine` -> `DuplexSessionManager`:
1. `_should_serve_duplex()` (`vllm_omni/entrypoints/openai/api_server.py`) resolves the deploy YAML (explicit `--deploy-config` or `pipeline_config.default_deploy_config_name`, including `base_config`), falls back to `pipeline_config.default_session_mode` if unset in YAML (e.g., `qwen3_omni/pipeline.py`), and **requires** the resolved `session_mode` to be `"duplex"` or `"turn"` — otherwise startup raises `ValueError`.
2. `DuplexOmniEngine._validate_deployment()` runs **before any stage process starts**:
   - `load_duplex_plugin(plugin_path, self._audio_encoder)` — verifies `isinstance(plugin, DuplexModelPlugin)`, non-empty `plugin.plugin_id`, and `isinstance(plugin.data_plane, DuplexDataPlane)`.
   - `plugin.configure_pipeline(pipeline_config)`
   - `plugin.configure_server(self.duplex_session_config)` — default raises `DuplexRuntimeConfigError` if `model_custom_options` is non-empty; override to validate model-specific server options.
   - `validate_duplex_replica_policy(plugin, self.duplex_session_config)` — under `replica_policy: least_sessions`, fails if `max_sessions_per_replica > 1` while `plugin.capabilities(max_sessions=cap).supports_multi_session_same_replica` is `False`.
3. `DuplexOmniEngine._create_orchestrator()` calls `plugin.configure_async_chunk(enabled=bool(self.async_chunk))`.
4. `DuplexSessionManager.__init__()` calls `validate_duplex_plugin_sampling(plugin, sampling_defaults=...)` — probes `configure_sampling_params(runtime_config={}, defaults=sampling_defaults)` and asserts a `tuple` of length `len(sampling_defaults)` where every non-`None` default matches `isinstance(value, type(default))` (allowing subclasses) — and then calls `plugin.configure_server(duplex_session_config)` again so direct manager construction validates server options too.

## 2. `DuplexModelPlugin` Skeleton (Pattern B1 / B2 Lockstep)

Minimal, production-grade plugin modeled on `vllm_omni/model_executor/models/haru/duplex/plugin.py` and `personaplex/duplex/plugin.py`:

```python
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputDecision
from vllm_omni.engine.duplex.intermediate import build_duplex_append_prompt
from vllm_omni.engine.duplex.plugin import (
    DefaultDuplexModelSessionState,
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
    EncodeAudio,
    reject_changed_runtime_value,
)
from vllm_omni.model_executor.common.duplex.payload import decode_pcm_f32le_payload
from vllm_omni.model_executor.common.duplex.pcm_buffer import FixedFramePcmAppendBuffer

if TYPE_CHECKING:
    from vllm.config import ModelConfig

SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1920           # e.g. 80 ms @ 24 kHz (Haru: 3200 @ 24 kHz, CHUNK_PERIOD_MS = 134)
CHUNK_PERIOD_MS = 80


def _frame_buffer() -> FixedFramePcmAppendBuffer:
    return FixedFramePcmAppendBuffer(
        sample_rate_hz=SAMPLE_RATE,
        frame_samples=FRAME_SAMPLES,
        chunk_period_ms=CHUNK_PERIOD_MS,
        model="YourModel",
    )


@dataclass
class YourSessionState(DefaultDuplexModelSessionState):
    audio_buffer: FixedFramePcmAppendBuffer = field(default_factory=_frame_buffer)


class YourDuplexPlugin(DuplexModelPlugin):
    plugin_id = "your_model"
    private_runtime_config_keys = frozenset({"your_prompt_slots", "your_model_path", "your_instructions"})
    # CRITICAL: override both so silence continuation and startup warmup emit valid frames!
    silence_continuation_samples = FRAME_SAMPLES
    silence_continuation_sample_rate_hz = SAMPLE_RATE

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = YourDataPlane(encode_audio)

    # ---- engine policy ----

    def configure_sampling_params(
        self, *, runtime_config: dict[str, object], defaults: tuple[object, ...]
    ) -> tuple[object, ...]:
        if not defaults or not isinstance(defaults[0], SamplingParams):
            return defaults
        stage0 = defaults[0].clone()
        stage0.max_tokens = 1
        stage0.output_kind = RequestOutputKind.DELTA  # Use DELTA when final stage emits per-step chunks
        stage0.detokenize = False
        return (stage0, *defaults[1:])

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        decode_pcm_f32le_payload(
            payload, sample_rate_hz=SAMPLE_RATE, exact_samples=FRAME_SAMPLES, model="YourModel"
        )
        slots = int(runtime_config.get("your_prompt_slots", 1)) if seq <= 1 else 1
        return DuplexAppendPlan(
            prompt=build_duplex_append_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                payload=dict(payload),  # type: ignore[arg-type]
                final=final,
                prompt_token_ids=[0] * slots,
                model_fields={},
            )
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        # Pattern B: always return None.
        # Pattern A: on stage_id < final_stage_id and segment_finished, inspect listen_token_id
        # and return DuplexOutputDecision(action=DuplexOutputAction.DIRECT_RESPONSE, metadata={...}).
        del stage_id, final_stage_id, segment_finished, segment_token_ids, segment_output_metadata, output
        return None

    # ---- session policy ----

    def create_session_state(self) -> YourSessionState:
        return YourSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        multi = max_sessions > 1
        return DuplexCapabilities(
            supports_model_native_turn_policy=True,
            supports_external_turn_signal=False,
            supports_client_commit=False,
            supports_barge_in=False,
            supports_playback_ack=True,
            supports_input_append=True,             # Required for live microphone appends
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_rollback_to_checkpoint=False,
            supports_turn_commit_only=False,
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_core_resumable_request=True,
            supports_stage_connector_handoff=False, # True for multi-stage (Pattern B2)
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=multi,
            supports_multi_session_same_replica=multi,
            supports_session_lease=True,
            supports_session_resume=False,
            session_admission_mode="engine_managed",
            supports_audio_truncate=False,
            supports_chat_completions=False,
            requires_model_runner_kv=True,
            requires_native_stage_role=True,
            adapter_patterns=["scheduler_data_plane"],
            signal_sources=["model_native", "client_event"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=CHUNK_PERIOD_MS,
            target_barge_in_latency_ms=None,
        )

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            raise DuplexRuntimeConfigError("model path is unavailable", code="model_path_unavailable")
        instructions = config.instructions or ""
        # Offload any disk/Hub/audio decoding so the orchestrator loop never blocks:
        slots = await asyncio.to_thread(compute_prefill_slots, model_path, instructions, config.voice)
        # Persist normalized creation-time fields that runtime_config_for_update() compares against!
        return {
            "your_model_path": model_path,
            "your_prompt_slots": slots,
            "your_instructions": instructions,
        }

    def runtime_config_for_update(
        self, config: DuplexSessionConfig, current: Mapping[str, object]
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        # Refuse live changes to options fixed at request creation:
        reject_changed_runtime_value(
            config.instructions or "",
            current.get("your_instructions", ""),
            message="instructions cannot be changed after the session is created",
            code="instructions_update_unsupported",
        )
        return deepcopy(dict(current))
```

## 3. Reusable `DuplexDataPlane` (`data_plane.py`)

Subclass `CumulativeAudioTextDataPlane` (`vllm_omni/model_executor/common/duplex/data_plane.py`):

```python
from vllm_omni.model_executor.common.duplex.data_plane import CumulativeAudioTextDataPlane


class YourDataPlane(CumulativeAudioTextDataPlane):
    stage_role = "tts"                 # or "llm" for single-stage / direct text
    default_sample_rate_hz = 24000
    audio_output_is_delta = True       # True when final stage uses RequestOutputKind.DELTA
```

Key behaviors inherited automatically:
- Per-request `_RequestCursor` (`audio_samples`, `text`, `terminal`).
- When `audio_output_is_delta=False`: slices `audio[state.audio_samples:]` and advances `state.audio_samples` so cumulative stage outputs never replay old audio.
- When `audio_output_is_delta=True`: concatenates `list[Tensor | ndarray]` chunks directly without slicing.
- `close_session(session_id, active_request_id=...)` removes both `active_request_id` and every request where `duplex_resource_request_belongs_to_session(req_id, session_id)` is `True`.
- For Pattern A listen decisions, inspect `get_duplex_output_decision(output)` from `vllm_omni.outputs.duplex` (authoritative over inner completion metadata).

## 4. Custom `PcmAppendBuffer` & Nested Session State Rules

Use `FixedFramePcmAppendBuffer` whenever your model takes fixed-length `pcm_f32le` frames. Only write a custom `PcmAppendBuffer` subclass when you need:
- Variable / silence-aware framing or `video_frames` paired with audio (Pattern A: `minicpmo_4_5/duplex/input.py`).
- Non-terminal commit prefixes via `prepare_commit_prefix()` + `retains_separate_vision_units()` (e.g., buffered camera units submitted with `force_listen=True` ahead of a speech commit).
- Single-terminal utterance buffering for VAD/commit models (Pattern C: `qwen3_omni/duplex/input.py`).
- Multi-unit draining per append for input-clocked sessions via `prepare_backlog()`.

Remember: `load_duplex_plugin()` validates only the plugin and `data_plane` at startup — verify `create_session_state()` and every `PcmAppendBuffer` method (`pending_byte_count`, `has_pending`, `has_reserved`, `clear`, `clear_force_listen`, `prepare_append`, `prepare_commit`, `flush`, commit/rollback order) in L1 unit tests!

## 5. Optional Plugin Capabilities (Tools, Auxiliary Memory, Multi-Stage Streaming)

- **Tool calling** (`tool_mode="declared"` or `"delegate_only"`, `max_tools`, `function_call_output_continuation`):
  - Implement `parse_function_call(model_result) -> {"call_id", "name", "arguments"} | None` and `runtime_config_for_function_output(config, current, item, *, history_len=...)`.
  - Session owns the call ledger (`pending -> accepted | timed_out | cancelled` with tombstones up to `max_tool_calls_per_session`).
  - Frame-locked models with queued function outputs (`nemotron_voicechat`) **must** implement `retire_pending_function_output()` (retire queued frames on cancel without rewinding monotonic worker watermarks) and `runtime_config_after_model_output()` (acknowledge worker consumption).
  - History-splicing models (`qwen3_omni`) implement `note_published_function_call()`, `has_pending_tool_followup()`, `runtime_patch_for_submitted_prompt()`, `bind_inflight_response()`, and `runtime_config_after_response()`.
  - Set `abandoned_tool_call_reasons` to notify the model of timeouts/cancellations via `abandoned_tool_call_notice()`.
- **Bounded auxiliary requests** (`supports_auxiliary_requests = True`, `uses_history_context = True`):
  - Uses `SessionMemoryRuntime` for private non-resumable background summarization (`prepare_auxiliary_request`, `auxiliary_policy`, `auxiliary_source_matches`, `apply_auxiliary_result`, `bind_history_context`). See `qwen3_omni/duplex/` and `joyai_vl_interaction/duplex/`.
- **Sentence-level / partial downstream forwarding**:
  - Override `plan_partial_stage_output()`, `partial_stage_followup()`, `resumable_downstream_stage_ids()`, `draining_stage_ids()`, and `release_concurrent_turn_requests()` (see `aura_omni/duplex/` and `joyai_vl_interaction/duplex/`).
- **Skip unnecessary downstream prewarm**:
  - Override `prewarms_downstream_stages(request_id) -> bool` when a request (e.g., text-only modalities) cannot reach downstream audio stages.

## 6. `DuplexCapabilities` Matrix Across Patterns

Contract source: `vllm_omni/engine/duplex/config.py` and each model's `duplex/capabilities.py` / `duplex/plugin.py`.

| Field | Pattern A (MiniCPM-o 4.5) | Pattern B1 (Haru-4B) | Pattern B2 (PersonaPlex) | Pattern B2 (Nemotron VoiceChat) | Pattern C (Qwen3-Omni) |
|-------|---------------------------|----------------------|--------------------------|---------------------------------|------------------------|
| `supports_model_native_turn_policy` | `True` | `True` | `True` | `True` | `False` (`turn_based_duplex`) |
| `supports_input_append` | `True` | `True` | `True` | `True` | `True` |
| `supports_client_commit` | `True` | `False` | `False` | `True` | `True` |
| `supports_external_turn_signal` | `True` | `False` | `False` | `False` | `True` |
| `supports_barge_in` | `True` | `False` | `False` | `False` | `True` |
| `supports_core_resumable_request` | `True` | `True` | `True` | `True` | `False` |
| `supports_stage_resumption` | `True` | `True` | `True` | `True` | `False` |
| `supports_stage_connector_handoff` | `True` | `False` | `True` | `True` | `False` (inherited default) |
| `supports_model_internal_state` | `True` | `True` | `True` | `True` | `False` |
| `supports_core_kv_lease` | `False` | `False` | `False` | `False` | `False` |
| `supports_multi_session` | `max_sessions > 1` | `max_sessions > 1` | `max_sessions > 1` | `max_sessions > 1` | `max_sessions > 1` |
| `supports_multi_session_same_replica` | `max_sessions > 1` | `max_sessions > 1` | `max_sessions > 1` | **`False`** (1 session/replica) | `max_sessions > 1` |
| `session_admission_mode` | `"engine_managed"` | `"engine_managed"` | `"engine_managed"` | `"engine_managed"` | `"engine_managed"` |
| `chunk_period_ms` | `1000` | `134` | `80` | `80` | `1000` (inherited default) |
| `tool_mode` | `"none"` | `"none"` | `"none"` | `"declared"` (`max_tools=5`) | `"declared"` |
| `supports_chat_completions` | `True` (`initial_user_text`) | `False` | `False` | `False` | `True` |
| `supports_session_stats` | *Engine-owned* | *Engine-owned* | *Engine-owned* | *Engine-owned* | *Engine-owned* |
