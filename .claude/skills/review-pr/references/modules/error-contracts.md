# Error Classification, Propagation, and Rendering

Primary design: [Error Classification, Propagation, and Rendering](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/error_contracts/).

Use when a change classifies failures, crosses process boundaries, decides
fatal versus request-scoped impact, sanitizes context, or renders offline,
HTTP, SSE, WebSocket, or asynchronous-job errors. The I/O module owns the wire
schema; this module owns the meaning of its error fields.

## Contract checks

- Keep request-scoped failure isolated unless the underlying engine is no
  longer usable.
- Preserve stable classification and request context across process and
  protocol boundaries without exposing prompts, outputs, secrets, or internal
  stack details.
- Keep cancellation distinguishable from internal failure.
- Terminate each public transport exactly once with consistent status, body,
  event, finish reason, and metrics classification.
- Maintain compatibility across mixed or older producers only when the
  reviewed contract defines that behavior.

Test table-driven classification, wire round trips, fatal/request-scoped
propagation, cancellation, sanitization, and every affected public renderer.
