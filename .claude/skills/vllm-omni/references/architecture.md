# vLLM-Omni Architecture and Code Map

Authoritative module designs live in `docs/design/module/*.md`; start from
`docs/design/architecture_overview.md`. This file is a navigation aid, not a
replacement. The engine is still being refactored, so confirm names in code
before citing them.

## Contents
- Two config layers
- Request path
- Stage hand-off
- Duplex path
- Module design docs

## Two config layers

| Layer | Defines | Where |
|-------|---------|-------|
| **Pipeline** (code, frozen) | Topology: stages, `model_stage`, `execution_type` (`LLM_AR` / `LLM_GENERATION`), input sources, stage input processor functions, final outputs | `vllm_omni/model_executor/models/<model>/pipeline.py`, registered in `OMNI_PIPELINES` (`vllm_omni/config/pipeline_registry.py`) |
| **Deploy** (YAML, tunable) | Runtime: devices, memory, batch limits, eager vs graph, connectors, `async_chunk`, `session_mode`, `model_runner`, platform overrides | `vllm_omni/deploy/<name>.yaml`, schema in `vllm_omni/config/stage_config.py` |

The pipeline is auto-selected from the HF config (`config_factory.py`); its
`default_deploy_config_name` picks the deploy YAML unless `--deploy-config` is
given. Deploy YAML must not redefine topology.

## Request path

1. **Entry**: `vllm serve <model> --omni` → `vllm_omni/entrypoints/cli/main.py` →
   `cli/serve.py` → `entrypoints/openai/api_server.py`. Handlers: `serving_chat.py`,
   `serving_speech*.py`, `entrypoints/openai/realtime/`, `entrypoints/duplex/`.
2. **Client facade**: online `AsyncOmni` (`entrypoints/async_omni.py`), offline `Omni`
   (`entrypoints/omni.py`); both build `AsyncOmniEngine`
   (`engine/async_omni_engine.py`, `engine/omni_engine_base.py`).
3. **Stage processes**: `engine/stage_runtime.py`, `stage_engine_startup.py`,
   `stage_engine_core_proc*.py` spawn one EngineCore per stage replica;
   `stage_pool.py` + `stage_client.py` hold replicas and clients.
4. **Orchestration**: `engine/orchestrator.py` admits requests to stage 0, drains
   outputs, and routes each output either to the frontend or to the next stage.
5. **Workers / runners**: `execution_type` maps to worker type (`LLM_AR` → AR worker,
   `LLM_GENERATION` → generation worker). V1 runners: `vllm_omni/worker/`; MRv2
   (`model_runner: v2`) runners: `vllm_omni/worker_v2/`. Schedulers: `vllm_omni/core/sched/`.
6. **Output shaping**: `vllm_omni/outputs/output_processor.py` builds text, audio,
   and multimodal client outputs. Model forward returns `OmniOutput`
   (`model_executor/models/output_templates.py`).

## Stage hand-off

- **Semantic transform** (what the next stage receives): functions in
  `vllm_omni/model_executor/stage_input_processors/<model>.py`, for example
  `thinker2talker_full_payload` / `thinker2talker_async_chunk`. `pipeline.py` wires
  them; `_select_processor_funcs` in `stage_config.py` picks the variant based on
  `async_chunk`.
- **Transport** (how bytes move): named connectors in the deploy YAML, built by
  `distributed/omni_connectors/factory.py`. Registered types:
  `SharedMemoryConnector`, `MooncakeStoreConnector`, `MooncakeTransferEngineConnector`,
  `NixlConnector`, `MoriTransferEngineConnector`, `YuanrongConnector`,
  `YuanrongTransferEngineConnector`. Single-node default is shared memory.
- **`async_chunk`**: upstream emits partial results (hidden states, codec frames)
  so downstream stages start before upstream finishes. This is what makes first
  audio packet latency low. Resolved per pipeline when the YAML leaves it unset.

## Duplex path

`session_mode: duplex` plus a pipeline `duplex_plugin` switches to
`entrypoints/duplex_omni.py`, `engine/duplex_omni_engine.py`,
`engine/duplex_orchestrator.py`. Clients connect over WebSocket (`/v1/realtime`,
`/v1/realtime?duplex=1`, `/v1/duplex`). Design: `docs/design/fullduplex.md`.

## Module design docs

| Doc | Topic |
|-----|-------|
| `engine_orchestration.md` | Orchestrator routing and request state |
| `stage_runtime.md` | Stage process and replica lifecycle |
| `ar_runtime.md` | AR schedulers, workers, runners |
| `omni_connector.md` | Cross-stage transport invariants |
| `vllm_omni_config.md` | Pipeline and deploy config system |
| `model_integration.md` | Registering models |
| `input_output_modality_contracts.md` | Input/output modality contracts |
| `entrypoints.md` | Serving and client entrypoints |
| `error_contracts.md` | Error classification and propagation |
| `cache_management.md` | KV / multimodal caches |
| `observability.md`, `profiling.md`, `benchmarking.md` | Metrics, profiling, bench tooling |
| `execution_platforms.md` | CUDA / ROCm / NPU / XPU |
| `quantization.md`, `host_weight_runtime.md` | Quantization, host-weight runtime |
