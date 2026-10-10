# Entrypoints

This package contains user-facing server, CLI, orchestration, and protocol
entrypoints for vLLM-Omni.

## What Belongs Here

- Runtime orchestration entrypoints such as `AsyncOmni`, `Omni`, and
  request-state helpers.
- CLI integration under `cli/`.
- Shared cross-entrypoint configuration and stage utilities.

## What Does Not Belong Here

- OpenAI endpoint-family route bodies and request helpers; those should move
  toward `openai/<family>/` packages as part of #5227.
- Generic model execution internals that belong under engine, worker, or model
  executor packages.

Some top-level utility files remain acceptable homes under Phase 0.2 (`P0.2`) of #5227.
They are not part of the OpenAI helper migration unless explicitly referenced by #5227.
