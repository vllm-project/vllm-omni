# Cosmos3 paired LIBERO evaluation

## Formal comparison contract

Compare one checkpoint at a time: the Framework DCP checkpoint and its
converted vLLM-Omni Diffusers artifact must contain the same model weights.
F0 and B0 are separate model comparisons. Keep the same LIBERO package,
initial states, task ordering, action statistics, evaluator Python environment,
camera order, preprocessing and action mapping. Record the checkpoint/export
manifest and GPU model, GPU count, device IDs, driver and concurrent GPU jobs
alongside the run lock. Use the same GPU type and count for both services and
avoid unrelated GPU load during measured requests. The runner evaluates the
backends sequentially; their model services may remain loaded on isolated GPUs.

The pinned evaluation is LIBERO-10, all 10 tasks, 50 trials per task, seed 0,
one environment, 256-pixel views in `agentview,wrist` order, 180-degree image
rotation, 10 environment stabilization actions, a full 16-action execution
horizon, 10 denoising steps, guidance 1.0, and no early stop. Both HAS and
constant Euler use nonstreaming full-chunk calls for the MoH timing comparison.
Streaming callbacks add network work inside the sampler and are measured
separately. See [MoH timing contract and field definitions](MOH_TIMING.md).
The default Framework Cosmos3-Edge scheduler config uses shift 1; serve
vLLM-Omni with `--flow-shift 1.0` for this contract. Do not save rollout GIFs,
comparison GIFs or generated videos during timed runs. vLLM-Omni always sends
`return_video=false` for native LIBERO actions. The Framework service must use
`--no-return-video` and no `--dump-dir`, because a dump directory also triggers
video decoding. Its `/info` must expose
`"video_decode_enabled": bool(self.cfg.return_video or self.cfg.dump_dir is not None)`;
the preflight refuses a missing field or a true value. This one-line metadata
addition is needed on the Framework server used for the comparison. Both
backends must also implement and advertise `timing_contract=moh_sample_actions_v1`:
measure the sampler with `time.monotonic()` after input transforms/H2D and
before output D2H, with no extra device synchronization. Do not add the label
to an existing round-trip measurement. An older Framework server must be
instrumented at that sampler boundary before the paired preflight can pass.

The first two completed model requests are excluded from steady-state latency
and throughput, while all valid episodes remain in success-rate accounting.
Environment stabilization actions are separate from request warmup and never
count as executed model actions. The primary comparable latency is
`policy_infer_ms`, measured inside each model worker. `client_ttfa_ms`
(HTTP submission to first ready action) and `http_total_ms` (submission to
complete response) remain transport diagnostics. For a
nonstreaming request, client TTFA equals full response time. Report these rates
without substituting requests for actions:

| Metric | Numerator | Denominator |
| --- | --- | --- |
| `steady_state_inference.actions_per_s` | Generated action positions in measured nonstreaming requests | Sum of worker sampler durations (`policy_infer_ms / 1000`) |
| `active_actions_per_s` | Generated action positions in measured requests | Union of measured HTTP request intervals |
| `sustained_actions_per_s` | Generated action positions in measured requests | First measured request start to last measured completion |
| `executed_actions_per_s` | Model action positions actually executed by LIBERO | Whole evaluation wall time, excluding service startup |

`server_infer_ms` and `worker_inference_ms` remain backend diagnostics and are
not substitutes for `policy_infer_ms`. MoH itself measures host wall time, not
CUDA kernel latency. Retain `request_timings.jsonl` and the
run lock for audit. A short smoke run with `--task-ids`, `--trials` or
`--max-steps` verifies integration only; it is not a formal success-rate or
throughput comparison.

## Run

Use the same converted model weights, action statistics and evaluator revision for
both backends. Start both policy servers first. The Framework server must be
started with `--no-return-video` and without `--dump-dir`. Start the vLLM-Omni
policy server with `--flow-shift 1.0`; the vLLM worker and frontend must use this
checkout. Both servers need 10 denoising steps, guidance 1.0, seed 0, the same
Euler action schedule and no early stop.

The preflight reads `/info` from both services and checks these settings before
any rollout. It verifies the SHA256 of each server's action statistics file,
checks the expected checkpoint paths, and records the evaluator script hash.
Path checks do **not** prove that the two checkpoint formats contain identical
weights; verify the conversion separately.

Example (replace paths and ports with those of the active services):

```bash
python examples/cosmos3/fair_eval.py \
  --framework-url http://127.0.0.1:8000 \
  --vllm-url http://127.0.0.1:8101 \
  --framework-checkpoint /path/to/dcp/checkpoint \
  --vllm-checkpoint /path/to/converted/diffusers \
  --stats-sha256 THE_SHA256_OF_THE_SHARED_STATS_FILE \
  --schedule has_euler \
  --python /path/to/evaluator/python \
  --output-root /path/to/new/results \
  --render-gpu 7
```

The default evaluator is `examples/cosmos3/closed_loop_eval.py` in this checkout.
It imports `eval_metrics.py` from the same directory and does not import
`cosmos_framework`. Both backend services use that one client for measurement.

Without `--run`, the command only validates and prints the locked settings and
two evaluation commands. Add `--run` to execute Framework first and vLLM-Omni
second. The output root must be new. `run_lock.json` records both `/info`
responses, the evaluator checksum and the exact commands. Use `--schedule
constant_euler` for a baseline pair; neither formal pair enables streaming.
Each run writes `request_timings.jsonl`, `summary.json`, and `metrics.json/csv/txt`.
The runner writes `comparison.json` after both runs finish. The first two
completed requests are excluded from steady state timing and throughput, but
remain in the raw records and success rate. Report request latency (client
TTFA and complete response), generated action throughput during active request
time and over the request span, and actions actually executed per evaluation
wall time separately. A request may return 16 actions, while the environment
may execute fewer because a task succeeds or an episode ends.
`comparison.json` includes client preprocessing, first and last action readiness,
complete response latency, server first/full readiness, and action throughput.
`comparison.json` includes `model_infer_mean_ms`, `model_infer_p95_ms`,
`model_chunks_per_s` and `model_actions_per_s`. Missing/incompatible model
timing aborts comparison export; raw reports remain available for diagnosis.
`worker_inference_ms` is retained separately and is never substituted for
the sampler measurement.

The runner inherits the calling environment, so set the evaluator's Python,
`PYTHONPATH`, LIBERO data configuration and GPU visibility consistently. It sets
the requested rendering backend for both runs. Use `--mujoco-gl osmesa
--render-gpu -1` when EGL devices are unavailable. For a short integration
check, add `--task-ids 0 --trials 1 --max-steps 64`; both backends receive the
same reduced task. The vLLM launcher uses a writable storage directory under
`COSMOS3_LOG_DIR` unless `VLLM_OMNI_SERVER_STORAGE__PATH` is explicitly set.
Video **decoding/output** is disabled;
this does not eliminate the model's internal video latent computation.
