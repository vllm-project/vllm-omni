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

## Native generation-stage scheduling metadata

The native async receive path uses `DefaultSchedulingMetadataAdapter` to
translate a runner-owned payload into scheduler-visible metadata. A producer
may supply `meta.next_stage_prompt_ids` to declare a non-empty list of
nonnegative scheduling token IDs. When `meta.next_stage_prompt_len` is also
present, it must match that list's length. The declaration takes precedence
over flattening `codes.audio`; the original codec tensor remains in the
runner data plane. This lets a generation stage schedule one placeholder per
chunk while decoding all RVQ frames, including missing residual entries
represented by `-1`. Payloads without an explicit declaration retain the
existing codec-token interpretation. AR consumers retain their prompt IDs.

`meta.aborted=true` distinguishes cancellation from normal completion for
consumers with incremental decoder history. The Zonos3 producer sends an empty
terminal control payload on cancellation; its Codec consumer releases history
without flushing synthesis tails or running a neural forward for that request.

For Zonos3, sampled logical EOS is distinct from transport completion. Its
materialized output retains the device Tail Drain completion bit. The producer
keeps request state until the native terminal notification is delivered after
all reserved outputs drain, then publishes one final identified Codec chunk
(including empty codes when all frames were already emitted). Token-limit
termination uses the same final flush, while cancellation discards history.
The native stop fence preserves the terminating output itself and rejects only
subsequent outputs; this also preserves other models' final embeddings/codes.

The producer and native consumer regression coverage is in
`tests/model_executor/stage_input_processors/test_zonos3.py`,
`tests/model_executor/models/test_zonos3_codec.py` and
`tests/worker/test_scheduling_metadata_adapter.py`.

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
