# Profile Routes

This package owns profiling-control schemas and, after route extraction, should
own the profiling route bodies.

## Put Here

- Request/response models for `/start_profile` and `/stop_profile`.
- Profile-specific helper functions.
- The profile `api_router.py` once route bodies move out of `api_server.py`.

## Do Not Put Here

- General server error handling.
- Engine profiling implementation internals.
- OpenAI modality endpoint logic.

The route bodies still live in `openai/api_server.py` under Phase 0.2 (`P0.2`) of #5227.
TODO(#5227, P0.3): move route bodies here during router extraction.
