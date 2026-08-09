# Entrypoints and Serving Boundaries

Primary design: [Entrypoints and Serving Boundaries](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/entrypoints/).

Use for offline APIs, CLI and server composition, OpenAI-compatible routes,
request validation, response conversion, streaming, sessions, and engine
handoff. Entrypoints adapt public contracts; they do not own cross-stage
routing, stage lifecycle, configuration precedence, payload schemas, or error
classification.

## Contract checks

- Normalize and validate public values once before engine submission.
- Preserve request identity, modality, ordering, and exactly one terminal event
  across streaming, disconnect, cancellation, and failure.
- Translate internal outputs and errors explicitly for each advertised offline,
  HTTP, SSE, WebSocket, or asynchronous-job path.
- Keep model-specific behavior behind a common adapter or processor instead of
  adding model-name policy to generic entrypoints.
- Require API or CLI changes to explain necessity, alternatives, compatibility,
  migration, and the supported route or transport matrix.

Test the affected public protocol, normalization, engine handoff, rendering,
and terminal outcomes. Route orchestration, I/O schema, and semantic error
changes to their module contracts.
