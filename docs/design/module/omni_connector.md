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

## Required companion KV receive failures

For legacy AR-to-diffusion KV reception, a successful primary receive does not
make a request usable when a role explicitly declared in `cfg_kv_request_ids`
is absent. An explicitly received empty KV object remains valid. Validate the
receiving rank's full payload before CFG scatter reduces it to rank-local roles.

Companion collector exceptions (including errors inside the Bagel collector)
propagate to the same verdict even if primary reception returned false or CFG
slots were already populated. A request declaring companion IDs requires a
collector; omitting it is a terminal configuration error, not permission to
reuse existing slots. Requests without companion IDs still allow no collector
and retain the primary-miss fallback. Collector exceptions abort collection;
request cleanup owns outstanding companions, rather than retrying a partially
consumed receive here.

Synchronous reception and prefetch consumption share a terminal receive verdict.
TP-active participants exchange errors in TP, CFG, then SP order; pure TP
includes errors in its existing KV-state consensus. This happens before any
rank clears its request-side KV references and raises `KVCacheReceiveError`.
The non-TP route retains the existing world-only distribution assumption: every
world member participates in the same request. This is not a new guarantee for
independent DP or PP requests on that route.

A consumed prefetch payload is never synchronously fetched again after a
terminal processing error. Primary misses without a terminal error retain the
existing fallback. This does not enable prefetch for models whose CFG collector
currently disables it. Process crashes, broken control collectives, and errors
during subsequent payload distribution are outside this receive verdict.
