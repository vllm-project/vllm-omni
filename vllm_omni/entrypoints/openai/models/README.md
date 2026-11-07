# OpenAI Models

This package mirrors upstream OpenAI model-list ownership.

## Put Here

- Helpers and serving shims that back `/v1/models`.
- Route bodies for `/v1/models` after endpoint extraction.
- Model-list protocol adapters that are OpenAI-specific.

## Do Not Put Here

- Actual model execution logic.
- Image/video/audio endpoint helpers.
- Generic app-state helpers.

`_DiffusionServingModels` lives here because pure diffusion servers still need
to satisfy the OpenAI `/v1/models` contract. The route body remains in
`openai/api_server.py` until the route extraction tracked by #5227.
