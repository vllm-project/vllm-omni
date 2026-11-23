# CLI Entrypoints

This package owns vLLM-Omni command-line integration.

## What Belongs Here

- Omni CLI command registration and dispatch.
- CLI argument parsing and validation for `vllm ... --omni`.
- Headless/runtime launch wiring that ultimately calls server bootstrap code.
- CLI-only presentation helpers such as the logo.

## What Does Not Belong Here

- FastAPI route bodies.
- OpenAI endpoint request helpers.
- Server app utilities that are shared outside CLI; those belong under
  `entrypoints/serve`.

The CLI currently imports `openai.api_server.omni_run_server`. If server
bootstrap moves later, update this import, but do not fold route helpers into
the CLI package. Related endpoint cleanup is tracked by #5227.
