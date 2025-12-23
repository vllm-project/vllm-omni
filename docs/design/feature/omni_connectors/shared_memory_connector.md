# SharedMemoryConnector

## When to Use

Best for single-node deployments where stages run on the same host. It is
auto-configured when no explicit connector is specified for an edge.

## How It Works

- Small payloads (< threshold): serialized and passed inline in metadata.
- Large payloads (>= threshold): stored in shared memory; the SHM name is
  returned in metadata.

## Configuration

```yaml
runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 65536
```

## Notes

- Auto-mode uses SharedMemoryConnector if no connector is declared for an edge.
