# LingBot-World realtime-video benchmarks

Use the existing serving benchmark for LingBot's `WS /v1/realtime/video`
protocol. One benchmark request is one complete rollout. The shared
`vllm bench serve --omni` lifecycle owns request scheduling, warmup execution,
result saving and failures; the `realtime-video` backend records video chunks
and computes their cadence and playback metrics.

## Run

Start the model separately with a supported deployment, for example:

```bash
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --omni --deploy-config benchmarks/lingbot_world/configs/usp4_compiled.yaml \
  --port 8000

vllm bench serve --omni \
  --backend realtime-video --dataset-name realtime-video \
  --model robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --host 127.0.0.1 --port 8000 \
  --realtime-video-image /path/to/frame.png \
  --realtime-video-num-chunks 40 \
  --num-warmups 1 --realtime-video-warmup-num-chunks 8 \
  --num-prompts 3 --realtime-video-target-fps 12 \
  --realtime-video-save-video /tmp/rollout.mp4 \
  --save-result --result-dir /tmp --result-filename lingbot.json
```

An implicit `--num-prompts` means one measured rollout for this backend.
Rollouts are sequential (`--max-concurrency 1`, `--request-rate inf`): the
current deployment supports one session. Traffic ramps, background probes and
concurrent requests are rejected rather than changing the measurement.
There is no hidden full-rollout readiness request or tokenizer initialization.
Use explicit warmups after the server is ready. Without an image override, the
built-in workload uses the shared vLLM image asset; a local image/workload lets
an installed client run without fetching that default asset.

`single_gpu_eager.yaml` is also available for a single-GPU deployment. That
configuration and the four-GPU compiled configuration have different costs;
compare only matching deployment, runtime, input and hardware settings. The
compiled configuration uses Ulysses over four ranks and may pay compilation or
capture costs in the first rollout. A warmup does not prove compilation has
finished; retain its observed records.

NPU uses the same `realtime-video` backend against a server started with
`benchmarks/lingbot_world/configs/npu_usp4_eager.yaml` in the supported NPU
runtime. Its CI measurement count is `--num-prompts 7`, with one excluded
eight-chunk warmup per case. The client does not choose CUDA/NPU device APIs;
the server deployment and outer CI job own that allocation.

## Workload and warmup

Use `--dataset-path rollout.json` for camera scripts and mid-rollout prompt
updates:

```json
{
  "prompt": "The camera moves slowly forward through the scene.",
  "image": "/path/to/first_frame.png",
  "num_chunks": 40,
  "camera_pattern": "orbit",
  "fps": 16,
  "seed": 42,
  "prompt_updates": [
    {"after_chunk": 8, "prompt": "Rain begins to fall", "transition_chunks": 2}
  ]
}
```

An explicit `camera_action_script` takes precedence over `num_chunks` and
`camera_pattern`; each chunk has exactly three action entries. Prompt updates
must use distinct chunk boundaries. The namespaced image, prompt,
negative-prompt, width, height, fps, seed and flow-shift flags override matching
workload-file values. `--realtime-video-num-chunks` and camera-pattern apply to
the built-in workload, so specify their file equivalents with `--dataset-path`.

Session warmup and chunk warmup address different costs:

- `--num-warmups 1 --realtime-video-warmup-num-chunks 8` runs a distinct,
  excluded eight-chunk session. Without the optional shortening flag, warmup
  uses the full workload. Warmup raw records remain in the result.
- `--realtime-video-warmup-chunks 6` excludes the attention-window ramp from
  steady cadence. The terminal chunk also stays out of steady metrics because
  it skips next-chunk preparation. All chunks remain in end-to-end and playback
  calculations.

Forty chunks give 33 steady intervals per measured session, indices 6–38.
Three such sessions give 99 intervals, which still do not establish a reliable
p99. Percentiles are descriptive summaries; the CI p99 gate requires at least
1000 steady observations. Independent server starts are a separate CI concern,
not the same as multiple sessions on one already-running server.

## Video metrics

The domain report is `realtime_video` in the ordinary saved benchmark JSON.
It retains `workload`, `config`, measured `sessions`, excluded `warmups`, raw
chunk indices/arrival times/byte lengths/frame counts, errors and aggregate
metrics. Failed or partial requests remain failed, with their available raw
observations; they are not successful measurements or zero-valued timings.
Frames are not output tokens and chunk intervals are not token ITL.

| Field | Meaning |
| --- | --- |
| `ttfc_ms` | `session.start` sent to the first media chunk's bytes received. |
| `interval_all` | Inter-arrival summary over all post-first chunks. |
| `interval_steady` | Inter-arrival summary excluding the warmup and terminal chunks. |
| `video_rtf` | Wall seconds / generated video seconds; lower is better. |
| `video_rtfx` | Reciprocal of RTF; higher is better. |
| `frames_per_second` | Actual generated frames / session wall time. |
| `chunk_deadline_ms` | Twelve steady frames / target playback FPS. |
| `slo_attainment` | Fraction of steady intervals meeting that deadline. |
| `playback` | Simulated buffered playback underruns and startup/steady stalls. |

The opening chunk contains nine frames and later chunks contain twelve.
`--realtime-video-fps` labels the media sent by the server;
`--realtime-video-target-fps` states the playback rate used for deadlines,
RTF and playback simulation. It defaults to the workload's FPS. For example,
a twelve-frame chunk has a 750 ms budget at 16 FPS and 1000 ms at 12 FPS.
Neither choice changes the observed bytes or their arrival timestamps.
`--realtime-video-chunk-slo-ms` can set an explicit deadline.

The client timestamps received media, so it cannot split DiT, VAE, encoding or
transport costs. Server profiling is needed for that attribution. A slow
interval alone cannot establish a visible freeze: playback simulation also
uses the configured start buffer. Media saving occurs after measurement and
losslessly remuxes received packets, preserving pixels and presentation
timestamps.

## Legacy script

The existing script remains a compatibility entrypoint:

```bash
python benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py \
  --port 8000 --image /path/to/frame.png --num-chunks 40 \
  --warmup-sessions 1 --warmup-num-chunks 8 --sessions 3 \
  --target-fps 12 --output-json /tmp/lingbot-legacy.json
```

Its existing host/port, workload, session, warmup, video and output flags are
translated to the same serving benchmark. It has no separate request loop or
metric implementation. `--output-json` retains the old top-level domain JSON;
the common saved result is kept beside it as `*-serve-result.json`. Prefer the
normal `vllm bench serve --omni` command for new uses.

## CI cases and local checks

Performance uses the existing [`tests/dfx/perf`](../../tests/dfx/perf) JSON
configuration and recorded hardware baseline workflow with this common backend.
It runs the candidate without live A/B calibration or a separate scheduler.
A missing recorded hardware baseline is reported as unavailable, not a fabricated
regression pass. The new LingBot World recipe currently has no measured baseline;
its CI labels explicitly identify performance recording with a baseline pending.
Twelve FPS remains a playback/SLO basis, not an absolute gate.

The [accuracy tests](../../tests/e2e/accuracy/lingbot_world/README.md) keep
numerical precision, WBench and VBench as separate tests. This PR temporarily
uses explicit **record-only** CI: one candidate server lifetime captures the
fixed VAE inputs and one video per case, then the packaged `vllm bench serve
--omni` evaluation path records measurements after that server exits. Each test
prints its measured values and saves its own accuracy JSON alongside the raw
tensors, videos and evaluator outputs. Successful recording validates complete,
finite generation and scoring; it does not establish an accuracy regression
pass. A follow-up PR will fill in reviewed baseline values from real CI output.

CUDA uses two H100 GPUs; NPU capture uses the existing four-device A2
configuration before CUDA-based quality evaluation. Each accuracy job has a
120-minute timeout; this is a configured bound, not a measured runtime.
Long-stability tests are not included.

CPU tests under `tests/benchmarks/test_lingbot_world_*.py` exercise raw metrics,
scripted WebSocket transport, argument compatibility, malformed inputs and
independent accuracy results. They do not establish real-model accuracy,
performance or accelerator capacity.
