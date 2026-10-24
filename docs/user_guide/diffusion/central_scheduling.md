# Centralized Diffusion Scheduling

Enable centralized, non-preemptive admission through the existing server:

```bash
vllm serve Qwen/Qwen-Image --omni --no-async-chunk \
  --enable-tail-aware-scheduling \
  --tail-aware-scheduling-config '{"hardware_profile":"910B2"}'
```

Waiting Normal requests are ranked by `waiting_time + beta * estimated_service_time`,
highest first. `risk_beta` defaults to 0.85; `band_risk_beta` (0.625) applies when
queue depth is within `band_min_pending`–`band_max_pending` (10–27 inclusive).
Service estimates use request size, steps and frames. Successful completions
update a moving average used to choose among idle replicas; failures do not.
Each replica executes one request at a time. Cancelling queued work releases its
reservation immediately; submitted work holds its slot until the worker reports
termination or completion. Running requests are never preempted.

`max_pending_requests` defaults to 1024; admission overflow returns HTTP 429.
To use multiple local replicas, set stage 0's `num_replicas` and `devices` in
`--deploy-config`. Enabled admission requires `n`/`num_outputs_per_prompt=1` and
rejects custom timesteps/sigmas with HTTP 400. Each replica receives one request
at a time, so worker batching knobs do not combine centrally queued requests.

The equivalent top-level YAML keys are `enable_tail_aware_scheduling` and
`tail_aware_scheduling_config`. Explicit CLI values override YAML values;
`--no-enable-tail-aware-scheduling` overrides an enabled deployment. Generic diffusion
deployments merge these explicit inputs with YAML before deriving effective
parallelism from `--num-gpus`.

Supported scope: one API process, one local non-streaming diffusion stage and
static replicas. Disable `async_chunk`; multi-stage, distributed and duplex
execution are unsupported. Enabling risk scheduling requires an explicit
`hardware_profile`: `910B2` or `910B3`. Qwen-Image uses profile-specific estimates;
Wan/Wan2.2 T2V selects a table by profile for exact and scaled estimates. Both
tables currently contain identical reference data, without separate per-model
hardware calibration. Requests and reference keys use the same spatial and
temporal normalization before lookup and scaling.
Add the profile when upgrading a FIFO deployment. Only native pipelines with
one output per request are supported.
Custom timesteps/sigmas, custom pipelines and custom/Diffusers engines are unsupported;
Wan support is limited to standard T2V: TI2V, DMD and I2V model variants are
rejected before stage launch, and image-conditioned Wan requests are rejected
before admission. Other hardware or model geometries require new calibration
and device validation.

The head classifies requests into Normal and Tail queues. Every `quota_every`
arrivals (20) grant `quota_amount` Tail credits (1). Classification also requires
service time at least `threshold_ratio` (0.8) times the observed maximum and
`long_request_ratio` (1.5) times the observed minimum. Normal requests have
priority; waiting Tail requests concentrate on selected replicas, then backfill
idle slots newest-first. Sustained Normal arrivals can delay waiting Tail work.

Within `beam_min_pending`–`beam_max_pending` (10–27), the planner uses pending
Normal requests and estimated replica release times to choose the next request.
It minimizes Normal-request P95, with mean latency as a tie-breaker, using
recent completed Normal latencies and projected Normal latencies. Tail requests
affect replica availability but do not enter this latency objective.
Arrivals and completions update subsequent choices; outside this range the
scheduler uses risk ordering. `beam_horizon` (4), `beam_width` (16) and
`beam_branch_width` (6) bound the search. Forecast expiry never frees a running
slot: overdue occupied replicas and their uncertain latency projections are
excluded from the rollout until released by actual completion, cancellation
or failure. No step execution
or preemption is required.
