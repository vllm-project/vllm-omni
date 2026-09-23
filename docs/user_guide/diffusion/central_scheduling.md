# Centralized Diffusion Scheduling

Enable centralized, non-preemptive admission through the existing server:

```bash
vllm serve Qwen/Qwen-Image --omni --no-async-chunk \
  --enable-tail-aware-scheduling \
  --tail-aware-scheduling-config '{"hardware_profile":"910B2"}'
```

Waiting requests are ranked by `waiting_time + beta * estimated_service_time`,
highest first. `risk_beta` defaults to 0.85; `band_risk_beta` (0.625) applies when
queue depth is within `band_min_pending`–`band_max_pending` (10–27 inclusive).
Service estimates use request size, steps and frames. Successful completions
update a moving average used to choose among idle replicas; failures do not.
Each replica executes one request at a time. Completion, cancellation or failure
releases the slot. Running requests are never preempted.

`max_pending_requests` defaults to 1024; admission overflow returns HTTP 429.
To use multiple local replicas, set stage 0's `num_replicas` and `devices` in
`--deploy-config`. The request API and worker execution settings stay unchanged.

The equivalent top-level YAML keys are `enable_tail_aware_scheduling` and
`tail_aware_scheduling_config`. Explicit CLI values override YAML values;
`--no-enable-tail-aware-scheduling` overrides an enabled deployment.

Supported scope: one API process, one local non-streaming diffusion stage and
static replicas. Disable `async_chunk`; multi-stage, distributed and duplex
execution are unsupported. Enabling risk scheduling requires an explicit
`hardware_profile`: `910B2` or `910B3`. Add it when upgrading a FIFO deployment.
Calibrated models are native Qwen-Image and Wan/Wan2.2 T2V, one output per request.
Custom timesteps/sigmas, custom pipelines and custom/Diffusers engines are unsupported;
other hardware or model geometries require new calibration and device validation.
