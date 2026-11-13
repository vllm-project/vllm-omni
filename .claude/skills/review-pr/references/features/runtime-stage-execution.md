# Runtime and Stage Execution Features

Load the matching design from the reviewed head:

| Feature | Design | Primary module intersections |
| --- | --- | --- |
| Disaggregated inference | [Design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/disaggregated_inference/) | Configuration, stage runtime, OmniConnector, I/O contracts |
| Async chunk | [Design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/async_chunk/) | Model integration, orchestration, stage runtime, connector, I/O |
| Async diffusion output | [Design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/async_diffusion_output/) | Diffusion runtime, I/O contracts, entrypoints |
| Async Omni output materialization | [Design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/omni_async_output_materialization/) | AR runtime, model integration, I/O contracts, orchestration |
| Automatic prefix caching | [Design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/prefix_caching/) | AR runtime, cache management, I/O contracts |

## Feature checks

- Confirm the feature's supported models, entrypoints, stage types, topology,
  configuration, default, and compatibility limitations.
- Trace enablement from public/config ingress through every participating module
  and verify the advertised production dispatch actually selects it.
- Preserve a correct disabled or fallback path and make unsupported combinations
  fail or warn as documented.
- Test overlap, ordering, identity, backpressure, cancellation, failure, final
  partial output, and cleanup at each asynchronous boundary.
- Update the feature page, support matrices, examples, and recipes when behavior
  or coverage changes.
