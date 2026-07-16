# Request-Level Batching

Request-level batching lets diffusion serving combine multiple compatible
logical requests into one pipeline `forward()` call. Each prompt remains a
separate request with its own `request_id`, sampling parameters, seed, output,
error, and abort state. The scheduler decides which requests can run together.

!!! warning "Prompt List Semantics"

    Diffusion request-level batching does not support a top-level packed
    list-prompt request. Submit multiple prompts as independent requests and let
    the scheduler batch compatible in-flight requests. Multimodal payloads stay
    inside a single prompt dict, for example
    `{"prompt": "...", "multi_modal_data": {"image": image}}`.

## Enablement

Increase `max_num_seqs` above `1` to allow the request scheduler to keep more
than one compatible request active:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --max-num-seqs 4
```

For bursty online traffic, you can also set a small admission wait window. This
lets the engine wait briefly before the first `schedule()` of a new wave so
nearby compatible requests can arrive and share the same fused forward pass:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --max-num-seqs 4 \
  --request-batch-max-wait-ms 20
```

`--request-batch-max-wait-ms 0` is the default and disables admission waiting,
so there is no added wait latency.

For deploy YAMLs, configure the diffusion stage engine args:

```yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    engine_args:
      max_num_seqs: 4
      request_batch_max_wait_ms: 20
```

## Compatibility

Only pipelines that declare request-batch support use the fused request-batch
path. The engine validates that the pipeline `forward()` uses the request-batch
contract and returns `list[DiffusionOutput]`. Pipelines that do not support this
contract do not use fused `pipeline.forward(batch)`; scheduled requests are
executed through per-request worker calls.

The scheduler batches only compatible requests. Compatibility is based on
shape-sensitive and guidance-sensitive sampling fields, including resolution,
frame count, CFG settings, output count, and LoRA identity. Requests with
different LoRA adapters or scales are kept in separate batches so the worker
activates one adapter per batch.

Request-level batching applies when `step_execution=False`. For the separate
step-wise runtime, see [Step Execution](step_execution.md).

## Tuning

- `max_num_seqs` caps the number of active compatible requests in one scheduler
  wave.
- `request_batch_max_wait_ms` is an upper bound on extra admission wait before a
  new wave starts. Keep it small for latency-sensitive serving; values such as
  `10` to `50` ms are a practical starting range for bursty HTTP ingress.
- `0` disables admission waiting and preserves the lowest first-request latency.
- FIFO ordering is conservative: an incompatible request at the front of the
  waiting queue can block later compatible requests from joining the current
  batch.

## Python API

When constructing `Omni`, pass the same engine arguments:

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    max_num_seqs=4,
    request_batch_max_wait_ms=20.0,
)

outputs = omni.generate(
    [
        "a cup of coffee on a table",
        "a toy dinosaur on a sandy beach",
        "a fox waking up in bed and yawning",
    ]
)
```

`Omni.generate([...])` submits each list item as its own logical diffusion
request. The runtime may batch those requests internally when their sampling
parameters are compatible.

## For Contributors

For implementation details and model-author guidance, see
[Request-Level Batching for Diffusion](../../design/feature/diffusion_request_level_batching.md).
