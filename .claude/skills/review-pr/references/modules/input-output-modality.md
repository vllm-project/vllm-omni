# Input, Output, and Modality Contracts

Primary design: [Input, Output, and Modality Contracts](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/input_output_modality_contracts/).

Use for requests, prompt and modality types, message schemas, serialization,
output types, accumulation, completion, and compatibility shims crossing
entrypoint, orchestration, stage, connector, or model boundaries.

## Contract checks

- Give every boundary value an explicit modality and validated typed schema.
- Preserve request and stage identity across conversion, serialization,
  streaming updates, batching, cancellation, and errors.
- Distinguish partial, final, empty, and failed output states for every modality.
- Verify producer and consumer agreement on shape, layout, dtype, device,
  ownership, metadata, ordering, and optional or unknown fields.
- Keep internal objects out of public protocols; translate them at the owning
  entrypoint.
- Preserve wire compatibility deliberately and give deprecated shims a bounded
  migration path.

Require construction, round-trip, accumulation, completion, rejection, and
compatibility tests at each changed producer-consumer boundary.
