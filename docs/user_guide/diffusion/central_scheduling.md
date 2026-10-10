# Centralized Diffusion Scheduling

Enable centralized, non-preemptive admission through the existing server:

```bash
vllm serve Qwen/Qwen-Image --omni --no-async-chunk \
  --enable-tail-aware-scheduling \
  --tail-aware-scheduling-config '{"max_pending_requests":1024}'
```

Requests wait in a central FIFO queue. An idle replica takes the oldest request;
each replica executes one request at a time. Completion, cancellation or failure
releases the slot. Running requests are never preempted.

`max_pending_requests` defaults to 1024; admission overflow returns HTTP 429.
To use multiple local replicas, set stage 0's `num_replicas` and `devices` in
`--deploy-config`. The request API and worker execution settings stay unchanged.

The equivalent top-level YAML keys are `enable_tail_aware_scheduling` and
`tail_aware_scheduling_config`. Explicit CLI values override YAML values;
`--no-enable-tail-aware-scheduling` overrides an enabled deployment.

Supported scope: one API process, one local non-streaming diffusion stage and
static replicas. Disable `async_chunk`; multi-stage, distributed and duplex
execution are unsupported. FIFO admission requires no hardware calibration.
