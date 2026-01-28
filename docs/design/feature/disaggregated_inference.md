# Disaggregated Inference for Omni-Modality Models

This guide explains how to configure and use distributed connectors
(`vllm_omni/distributed/omni_connectors`) in vllm-omni for multi-stage pipelines.

Backend-specific setup lives in separate docs:

- [SharedMemoryConnector](omni_connectors/shared_memory_connector.md)
- [MooncakeConnector](omni_connectors/mooncake_connector.md)
- [YuanrongConnector](omni_connectors/yuanrong_connector.md)

## Overview

Connectors enable data transfer between pipeline stages (e.g., Thinker -> Talker).
Current connectors operate in D2H2D (device to host to device) mode.

## Connector Choices

| Use Case | Recommended Connector | Notes |
| :--- | :--- | :--- |
| Single node | SharedMemoryConnector | Auto-configured if no connector is specified. |
| Multi node (Mooncake) | MooncakeConnector | Requires Mooncake Master + metadata server. |
| Multi node (Yuanrong) | YuanrongConnector | Requires Yuanrong Datasystem + etcd. |

## Core API

The connector system is built around `OmniConnectorBase`.

```python
class OmniConnectorBase(ABC):
    @abstractmethod
    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, Optional[dict]]:
        """
        Store data.
        Returns: (success, serialized_size, metadata)
        """
        pass

    @abstractmethod
    def get(self, from_stage: str, to_stage: str, get_key: str, metadata: Optional[dict] = None) -> Optional[tuple[Any, int]]:
        """
        Retrieve data.
        Args: metadata - transport-specific handles returned by put() (e.g., SHM name).
        Returns: (object, serialized_size)
        """
        pass
```

### Metadata Passing

Some connectors (e.g., SharedMemoryConnector) generate transient resources during `put()`.
This `metadata` must be passed through the control plane so `get()` can locate the data.

## Configuration Model

Define connectors in runtime:

```yaml
runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 65536
```

Wire stages to connectors:

```yaml
stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: connector_of_shared_memory

  - stage_id: 1
    input_connectors:
      from_stage_0: connector_of_shared_memory
```

If a pipeline edge has no explicit connector, the system auto-creates a
SharedMemoryConnector for that edge.

## Relationship with vLLM

vLLM provides specialized distributed mechanisms for specific artifacts:

- KV Transfer (`vllm.distributed.kv_transfer`): optimized for KV caches.
- EC Transfer (`vllm.distributed.ec_transfer`): optimized for encoder embeddings.
- Device Communicators (`vllm.distributed.device_communicators`): low-level primitives (NCCL, SHM).

vllm-omni complements this with a generalized connector abstraction:

1. Unifies transport via a single `put`/`get` API for any stage artifact.
2. Enables DAG-style pipelines across processes or nodes with per-edge transports.
3. Can wrap vLLM-specific transfers for KV paths while keeping a consistent interface.

## Operational Notes

- Fail-fast config validation: missing expected edges cause startup failures.
- Missing payloads halt stages: verify connector wiring and metadata propagation.

## Future Roadmap: D2D Transport

Current connectors use D2H2D paths. Future versions will introduce direct
device-to-device connectors (NCCL, UCX, IPC) to reduce latency for large
tensor payloads.
