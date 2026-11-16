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

### Shared-memory capacity failures

`SharedMemoryConnector` checks available space as an advisory admission hint,
then reserves backing storage with `posix_fallocate` before copying into the
mapping. Allocation failure unlinks the unpublished segment. Capacity polling
holds no UID-wide lock; per-key lock contention uses a nonblocking lock and the
same bounded wait as capacity polling.

A failed KV publication is not an extraction acknowledgement. All sender TP
ranks must agree on success; otherwise published local shards are cleaned up.
The AR scheduler retains the request's KV blocks and resubmits the transfer on
its next step. Cancellation removes the request rather than resurrecting it
for a retry. Restoring capacity permits publication and block release without
requiring a new model computation. Persistent rejection remains pending until
capacity returns or the request is cancelled; it is not reported as success.
