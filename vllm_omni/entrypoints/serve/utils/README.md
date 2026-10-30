# Serve Utilities

This directory is for server/app utilities shared by Omni entrypoints.

## Put Here

- FastAPI app/router mutation helpers (`routes.remove_route_from_app`,
  `routes._remove_route_from_router`).
- Server exception-handler registration.
- Engine-failure helpers that are not owned by one endpoint family.

## Do Not Put Here

- Endpoint restriction *policy* (which routes a pipeline shuts down, and why).
  That stays in `vllm_omni.config.endpoint_policy`; the policy module should
  *call* `remove_route_from_app` from here, not redefine it.
- OpenAI response-shape adapters. Those belong under
  `vllm_omni.entrypoints.openai`.
- Modality validation or payload parsing.
- Generic helpers without a clear server/app responsibility.

Keep this directory aligned with the spirit of upstream
`vllm.entrypoints.serve.utils`. Broader utility cleanup is tracked by #5227.
