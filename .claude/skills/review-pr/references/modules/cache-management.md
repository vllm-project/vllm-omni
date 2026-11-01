# Cache Management

Primary design: [Cache Management](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/cache_management/).

Use for AR or diffusion cache identity, reusable state, validity, lifecycle,
reset, eviction, and teardown.

## Contract checks

- Include every model, request, input, timestep, precision, device, topology,
  and feature property that can change a reused value in cache identity.
- Preserve a correct disabled and miss path; never reuse after validity inputs
  change.
- Isolate request-scoped state across concurrent requests and define ownership
  for model-scoped state.
- Make reset, invalidation, eviction, capacity, and teardown explicit on
  completion, cancellation, failure, model reload, and shutdown.
- Keep cache optimizations behind the owning runtime's hooks rather than adding
  a second request lifecycle.

Test hit, miss, invalidation, disabled behavior, concurrency, capacity, reset,
and teardown, with quality comparison when reuse is approximate.
