# Serve Entrypoint Helpers

This package contains server-level helpers for assembling and operating the
HTTP/FastAPI serving process.

## What Belongs Here

- Helpers that are about the server app rather than a public OpenAI endpoint
  family.
- Operational route packages such as profiling and Omni control.
- Shared server utilities that mirror upstream `vllm.entrypoints.serve.utils`
  concepts.

## What Does Not Belong Here

- OpenAI protocol schemas or response bodies.
- Image, video, audio, or streaming-input request parsing.
- Serving model implementations.

Endpoint-family route bodies are still staged in `openai/api_server.py` during
the P0.2 helper migration. Route ownership (P0.3) and final folder cleanup (Phase 1)
are tracked by #5227.
