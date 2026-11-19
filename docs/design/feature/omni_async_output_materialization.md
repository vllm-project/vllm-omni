# Async Omni Output Materialization

Omni AR and generation stages use Model Runner V2. The former V1 background
builder and its GPU/NPU/XPU runners have been removed. Model-owned flags from
that implementation do not enable a V1 fallback.

V2 snapshots and materializes stage outputs through `OmniAsyncOutput` and the
native output worker. Model states preserve request-local payload ownership
before graph or persistent buffers can be reused. Output snapshots retain the
storage and copy events required until CPU outputs become available.

## MRv2 native output ownership

Both `NativeOutputWorker` and the runner's inline finalizer used by multi-rank
executors call `OmniRunnerDataPlane.publish_output()` to publish materialized
inter-stage payloads. The data plane owns enqueueing and alias cleanup. After
enqueueing succeeds, it clears `pooler_output` only when a native send path
exists and that field aliases the same inter-stage payload list. Otherwise,
the scheduler would serialize those tensors again alongside text output.
Independent pooling results, client-facing `multimodal_outputs`, and inline
payloads for stages without a native send path remain available. Terminal
ordering and payload delivery stay owned by the native data plane.

`OmniAsyncOutput.get_output()` assigns the same list to `pooler_output` and
`inter_stage_outputs` in both full-payload branches, including model-owned
snapshots. Producer tests protect this identity contract. Async-chunk output
already leaves `pooler_output` empty. Publication does not compare tensor
contents or discard separately constructed pooling lists.

Enqueueing transfers a reference to `_NativeOutputBatch`; it does not wait for
downstream delivery. Clearing a field on the returned model output neither
clears the queued list nor its per-request entries. The queue retains those
objects until the output worker consumes them, including batches with absent
payloads for some requests. Delivery completion and terminal cleanup retain
their existing ordering; publication adds no synchronous delivery wait.

## AR and generation stage behavior

MRv2 AR output also honors the model's `omni_pooler_payload_include_hidden`
contract, in conjunction with the sampler's `include_hidden_states` flag.
Opting out omits hidden-state output copies, while logits and sampling continue
to receive the original hidden states. Models without the flag retain the
existing hidden-state payload behavior.

The MRv2 generation runner has its own control-only fast path. Both its
control-only and cached-request zero-token paths deliver pending request
prewarms after finished-request cleanup and run optional idle prefetch only on
consecutive idle steps. Dummy/profile runs do not enqueue or execute prewarms,
but a dummy run with scheduled tokens clears the idle flag so the next real
idle step cannot prefetch while that work may still be in flight.
Warmup uses the forward thread, stream and inference mode; failures log once
per hook at warning level (subsequent failures at debug) and leave the inline
reference preparation path available. Connector and auxiliary-output work
still runs on every idle step.

## Related files

- `vllm_omni/worker_v2/omni_ar_model_runner.py`: AR output snapshots and materialization.
- `vllm_omni/worker_v2/omni_generation_model_runner.py`: Generation output and idle steps.
- `vllm_omni/worker_v2/native_output_worker.py`: Native output worker lifecycle.
- `vllm_omni/worker_v2/omni_data_plane.py`: Payload publication and delivery ownership.
- `vllm_omni/platforms/npu/worker/npu_generation_model_runner_v2.py`: NPU generation outputs.
- `tests/worker_v2/test_omni_ar_model_runner.py`: AR output ownership tests.
- `tests/worker_v2/test_omni_generation_model_runner.py`: Generation and idle-step tests.
- [Async Chunk](async_chunk.md): Inter-stage chunking and scheduling design.
- [Automatic Prefix Caching](prefix_caching.md#implementation): Stage-output cache lifecycle.
