# OmniConnectors

OmniConnectors transport payloads and KV-cache data between stages in a multi-stage vLLM-Omni pipeline. The connector transports data; stage routing and execution policy remain owned by the pipeline and orchestrator.

## Configure a connector

Define named connector implementations at the top level of a deploy YAML, then reference them from the sending and receiving stages:

```yaml
connectors:
  shm:
    name: SharedMemoryConnector

stages:
  - stage_id: 0
    output_connectors:
      to_stage_1: shm
  - stage_id: 1
    input_connectors:
      from_stage_0: shm
```

`connectors` contains deploy-owned connector definitions. The stage mappings identify the directed edge on which each definition is used. The runtime resolves these entries into an inbound/outbound `StageConnectorPlan` for each stage; users normally configure the YAML rather than constructing the plan directly.

If an expected edge has no explicit connector, vLLM-Omni can use the default `SharedMemoryConnector` where supported by the deployment. Configure every edge explicitly when stages run on different hosts or require a specific transport.

## Choose a backend

- `SharedMemoryConnector`: same-host transfers through shared memory.
- `MooncakeStoreConnector`: remote transfers through the Mooncake store.
- `MooncakeTransferEngineConnector`: peer-to-peer Mooncake TCP/RDMA transfers.
- `MoriTransferEngineConnector`: Mori IOEngine TCP/RDMA transfers.
- `YuanrongConnector`: remote transfers through Yuanrong Datasystem.
- `YuanrongTransferEngineConnector`: Yuanrong TransferEngine transfers on
  supported NPU deployments.

Backend-specific options and prerequisites are documented in the [OmniConnector design documentation](../design/feature/disaggregated_inference.md). For the complete deploy schema, see [Pipeline and deploy configurations](../configuration/stage_configs.md).

## Operational requirements

- Use matching connector configuration on both ends of an edge.
- Ensure remote backends, metadata services, and shared-memory permissions are available before starting the stages.
- Preserve stable stage and replica routing for requests whose payloads are transferred asynchronously.
- Treat connector initialization and cleanup errors as deployment errors; the runtime validates required inbound edges during stage initialization.
