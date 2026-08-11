# OmniConnector

Primary design: [OmniConnector](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/omni_connector/).

Use for connector contracts and model-agnostic data transfer across stages,
processes, devices, or nodes. Connector implementations transport and
synchronize data; they do not choose the next stage or implement model policy.

## Contract checks

- Prove producer and consumer agreement on request/stage identity, shape,
  layout, dtype, placement, ownership, ordering, and completion.
- Keep backend selection and metadata/key construction explicit; do not silently
  reinterpret or drop payloads across SHM, store, network, or vendor paths.
- Bound connect, put/get, retry, synchronization, and readiness waits with
  backpressure, cancellation, and actionable errors.
- Prevent route, endpoint, port, key, and buffer collisions across repeated
  startup, concurrent requests, and replica replacement.
- Release connections, handles, buffers, listener work, and stale sender state
  on success, cancellation, failure, and shutdown.

Load the matching communication feature design for a concrete backend. Require
focused producer-consumer tests and the smallest runnable multi-process or
multi-node path.
