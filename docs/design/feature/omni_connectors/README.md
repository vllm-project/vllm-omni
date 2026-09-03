# OmniConnector architecture

OmniConnectors provide transport and synchronization for data exchanged between vLLM-Omni pipeline stages. They carry stage payloads and KV-cache data, but do not select stages or implement model-specific execution policy.

## Configuration and runtime topology

Deploy configuration names the available connector backends and attaches them to directed stage edges:

```yaml
connectors:
  shm:
    name: SharedMemoryConnector

stages:
  - stage_id: 0
    output_connectors: {to_stage_1: shm}
  - stage_id: 1
    input_connectors: {from_stage_0: shm}
```

The configuration has two distinct layers:

- `connectors` is deploy-owned source configuration. Each entry contains a registered connector name and backend-specific `extra` options.
- `input_connectors` and `output_connectors` declare which directed stage edge uses each named connector.
- The resolver converts these entries into a per-stage `StageConnectorPlan`. The plan contains the resolved inbound and outbound `StageConnectorSpec` values used during stage initialization and connector construction.

`StageConnectorPlan` is therefore a runtime representation of deployment configuration, not a second user-facing routing mechanism. The orchestrator owns logical stage routing; the connector only transports data for an already selected edge.

## Connector lifecycle

1. The deploy YAML is parsed and connector definitions are validated.
2. Stage edge references are resolved into `StageConnectorPlan` instances.
3. The worker-side factory materializes the plan into receive and send connectors for the stage and replica.
4. Payload and KV operations use the common `put()` / `get()` contract.
5. Connectors release transport resources on completion, cancellation, failure, and shutdown.

Stages that receive connector-fed payloads must have an explicit inbound edge. Missing required edges fail during initialization rather than silently falling back to an unrelated stage. An omitted connector backend may use `SharedMemoryConnector` where the deployment resolver supports automatic local fallback.

## Backend designs

- [SharedMemoryConnector](shared_memory_connector.md)
- [MooncakeStoreConnector](mooncake_store_connector.md)
- [MooncakeTransferEngineConnector](mooncake_transfer_engine_connector.md)
- [MoriTransferEngineConnector](mori_transfer_engine_connector.md)
- [YuanrongConnector](yuanrong_connector.md)
- [YuanrongTransferEngineConnector](yuanrong_transfer_engine_connector.md)

For the complete deploy schema and CLI usage, see the [Pipeline and deploy configuration guide](../../../configuration/stage_configs.md). For the user-oriented setup guide, see [OmniConnectors](../../../user_guide/omni_connectors.md).
