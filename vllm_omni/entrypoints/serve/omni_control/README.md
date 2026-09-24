# Omni Control Routes

This package owns operational Omni server-control routes such as sleep and
wakeup.

## Put Here

- Request/response models for `/v1/omni/*` operational controls.
- Route bodies for Omni control endpoints after endpoint extraction.
- Small helpers that are specific to these operational controls.

## Do Not Put Here

- OpenAI modality endpoints.
- Generic server utilities that are shared outside Omni control.
- Engine implementation details for sleep or wakeup.

The current route bodies still live in `openai/api_server.py` and should move
here as part of #5227.
