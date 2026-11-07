---
title: OmniConnector
kind: module
status: draft
owners:
  - "@princepride"
  - "@yuanheng-zhao"
  - "@xuechendi"
  - "@natureofnature"
  - "@fake0fan"
primary_code_paths:
  - vllm_omni/distributed/omni_connectors/**
related_code_paths:
  - vllm_omni/platforms/*/omni_connectors/**
depends_on:
  - input_output_modality_contracts.md
  - vllm_omni_config.md
validation_paths:
  - tests/distributed/omni_connectors/**
upstream_refs:
  - vllm.distributed
last_reviewed: 2026-07-16
---

# OmniConnector

OmniConnector defines model-agnostic transport and synchronization contracts
for data exchanged across stages, processes, devices, and nodes.

## Candidate invariants

### CONNECTOR-INV-001: Connectors transport but do not route

**Rule:** A connector MUST NOT select the next stage or implement model-specific
execution policy.

### CONNECTOR-INV-002: Producer and consumer contracts agree

**Rule:** Both ends MUST agree on data identity, shape, dtype, placement,
ownership, and completion semantics.

### CONNECTOR-INV-003: Resources have deterministic cleanup

**Rule:** Connections, buffers, handles, and background work MUST be released on
normal completion, cancellation, failure, and shutdown.

## Safe-change guide

Test setup, transfer, synchronization, timeout, cancellation, failure, and
cleanup across every affected backend.

## Full-payload request lifecycle

The worker owns pending full-payload outputs under the internal request ID.
The scheduler supplies terminal status; the connector does not infer success
from a request disappearing from the active batch.

- **Completion:** stop and length-limit completion flush the accumulated
  payload once, then release its data and position tracking.
- **Cancellation or failure:** aborted, errored and ignored requests discard
  pending payloads without invoking the downstream processor. Already
  submitted connector writes are outside the pending accumulator's ownership.
- **Preemption and resume:** keep accepted AR outputs. The scheduler owns
  `delivered_upto`; the runner clips declared token fields before accumulation.
  After pending builders complete, resume truncates an unaccepted accumulated
  suffix to that boundary. The accumulator then appends the recovered or newly
  computed rows. Fixed embeddings use the processor's replacement policy and
  are never sliced by token count. One-shot generation payloads retain their
  existing append/replace policy.

Lifecycle regressions live in `tests/worker/test_full_payload_lifecycle.py`.
