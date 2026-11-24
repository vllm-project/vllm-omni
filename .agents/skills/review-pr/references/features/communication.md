# Communication Features

Use with the OmniConnector, stage-runtime, configuration, and I/O module
contracts. Load the concrete backend design from the reviewed head:

| Backend | Design |
| --- | --- |
| Mooncake store | [MooncakeStoreConnector](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/omni_connectors/mooncake_store_connector/) |
| Mooncake transfer engine | [MooncakeTransferEngineConnector](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/omni_connectors/mooncake_transfer_engine_connector/) |
| Mori transfer engine | [MoriTransferEngineConnector](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/omni_connectors/mori_transfer_engine_connector/) |
| Shared memory | [SharedMemoryConnector](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/omni_connectors/shared_memory_connector/) |
| Yuanrong store | [YuanrongConnector](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/omni_connectors/yuanrong_connector/) |
| Yuanrong transfer engine | [YuanrongTransferEngineConnector](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/omni_connectors/yuanrong_transfer_engine_connector/) |

## Feature checks

- Verify dependency, service, device/network, port, memory-pool, and environment
  prerequisites with actionable startup errors.
- Match key/metadata construction, sender and receiver roles, allocation,
  ownership, completion, and supported fast/serialized paths.
- Exercise timeout, retry, stale sender/buffer reclamation, partial transfer,
  cancellation, and connector close semantics.
- Validate the smallest supported same-host or cross-node topology and document
  backend limitations and operational diagnostics.
