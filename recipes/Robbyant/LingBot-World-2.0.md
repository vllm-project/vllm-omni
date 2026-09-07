# LingBot-World 2.0

> Offline and experimental realtime interactive world generation

## Summary

- Vendor: Robbyant
- Model: `robbyant/lingbot-world-v2-14b-causal-fast-diffusers`
- Task: image-conditioned interactive world generation
- Modes: offline trajectory replay, stepwise one-request streaming, and in-process realtime AR-Diffusion ticks (deprecated)
- Hardware validated: NVIDIA H200 and B200
- Maintainer: Community

The checkpoint is separately licensed under CC BY-NC-SA and restricted to
non-commercial use. The vLLM-Omni integration code remains Apache-2.0.

## Which path to use

There are three ways to drive this model, each documented in its own section
below.

- **Realtime stepwise** — see [Streaming video serving](#streaming-video-serving).
  The suggested path: one `WS /v1/realtime/video` session produces the whole
  rollout, one video chunk per AR block. It cannot take realtime camera
  interaction yet; that is being added, and this path is the one that will
  remain.
- **Realtime tick** *(deprecated)* — see
  [Realtime in-process generation](#realtime-in-process-generation-deprecated).
  The only path that takes realtime camera interaction today, so it stays until
  the stepwise path can, but it takes no new work and will be removed.
- **Offline** — see [Offline generation](#offline-generation). Replays a fixed
  pose/intrinsics trajectory in one request and writes an MP4. Use it when the
  camera path is known up front and streaming is not needed.

## Offline generation

The offline path consumes one source image and an action directory containing
`poses.npy` with shape `[frames, 4, 4]` and `intrinsics.npy` with shape
`[frames, 4]`.

```bash
python examples/offline_inference/diffusion/lingbot_world_v2.py \
  --prompt "The camera moves slowly forward through the scene." \
  --image /path/to/first_frame.png \
  --action-dir /path/to/actions/forward \
  --num-frames 81 \
  --output lingbot_world_v2.mp4
```

The current causal-fast checkpoint uses four DMD sampling steps. Raw frame
counts must be `9 + 12k`, up to 117 frames. Camera files may contain a
longer bounded source trajectory (the official example contains 269 frames);
the runtime consumes only the prefix needed by the request.

## Realtime in-process generation (deprecated)

Use [Streaming video serving](#streaming-video-serving) instead unless you need
mid-session camera interaction, which only this path has. It still works and
emits no runtime warning; removal is tracked as B4 of the LingBot World 2.0
roadmap ([#6672](https://github.com/vllm-project/vllm-omni/issues/6672)).

The realtime example keeps AR-Diffusion KV and model-owned state across
requests. Each JSONL line describes the prompt and/or three latent-frame camera
actions applied at the next chunk boundary:

```json
{"event_id":1,"prompt":"A road through a forest","frames":[["j"],[],[]]}
{"event_id":2,"frames":[["w"],["w"],["w"]]}
{"event_id":3,"prompt":"The road enters a snowy valley","frames":[[],[],[]]}
```

Run:

```bash
python examples/offline_inference/diffusion/lingbot_world_v2_realtime.py \
  --image /path/to/first_frame.png \
  --events /path/to/events.jsonl \
  --output-dir /tmp/lingbot-realtime \
  --gpu-memory-fraction 0.6
```

The example writes one latent tensor and one metadata JSON file per chunk. It
exercises the same `ARDiffusionSessionManager -> ARDiffusionOmniTickConsumer
-> AsyncOmni -> ARDiffusionEngine` path used by a future HTTP or WebSocket
transport.

Realtime serving now goes through the generic `WS /v1/realtime/video`
transport, which exposes no LingBot-specific event fields; the tick loop above
stays available only for the older control plane.

## Streaming video serving

The stepwise path keeps AR-Diffusion paged KV but issues **one** request for
the whole rollout: `prepare_encode` runs once, then every AR block is four DMD
steps followed by one streamed chunk. Identity metadata uses
`session_id = request_id` with contiguous `chunk_index` values from zero.
Camera motion for this landing is request-scoped: one three-frame action list
per chunk, fixed when the rollout starts.

Serve it with the AR-Diffusion deploy config, which selects the AR-Diffusion
engine and enables streamed step execution:

```bash
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --omni \
  --deploy-config vllm_omni/deploy/lingbot_world_v2_stepwise.yaml \
  --port 8000
```

`--deploy-config` is required: the `lingbot_world` pipeline deliberately
registers no default deploy config, so offline replay and the tick example
keep their request-mode topology when no deploy config is given, and the
stepwise serving topology is only ever an explicit choice.

Clients then use the generic WebSocket protocol documented in
[`docs/serving/video_stream_api.md`](../../docs/serving/video_stream_api.md):
`session.start` begins one rollout and each AR block arrives as a binary video
chunk. The model is image-conditioned, so the first frame is required and rides
on `image_reference` as an `http(s)` or `data:` URL; per-chunk camera actions
ride on `extra_params`:

```json
{"type": "session.start", "model": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
 "prompt": "The camera moves slowly forward through the scene.",
 "image_reference": {"image_url": "data:image/png;base64,<first frame>"},
 "width": 832, "height": 480, "num_frames": 33,
 "extra_params": {"flow_shift": 5.0,
                  "camera_action_script": [[["w"], ["w"], ["w"]], [["a"], [], []], [[], [], []]]}}
```

`camera_action_script` must hold exactly one action list per generated
chunk, and a request generates `((num_frames - 1) // 4 + 1) // 3` chunks:
three for `num_frames: 33`, seven for `num_frames: 81`.

The bundled client speaks this protocol:

```bash
python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --model robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --prompt "The camera moves slowly forward through the scene." \
  --image-reference /path/to/first_frame.png \
  --width 832 --height 480 --num-frames 33 --fps 16 --seed 42 \
  --extra-params '{"camera_action_script":[[["w"],["w"],["w"]],[["a"],[],[]],[[],[],[]]]}' \
  --output lingbot_world_v2_stream.mp4
```

A served request may instead point at a pose/intrinsics trajectory with
`extra_params.action_path`, which is resolved inside the trusted root set by
`model_config.lingbot_action_root` or `VLLM_OMNI_LINGBOT_ACTION_ROOT`; a server
started without that root configured accepts only `camera_action_script`.

Requested `width`/`height` must match `ar_diffusion_width`/`ar_diffusion_height`
in the deploy config, because the AR cache geometry is fixed at load time.
Blocks are decoded independently, so seams between chunks are possible; a
session-owned streaming decoder is tracked separately.

Mid-session `session.interaction` for camera control is not wired yet, so a
served rollout follows the script it started with.

The deprecated tick example remains available for the older
one-block-per-`generate()` control plane.

## Realtime identity and controls

- `session_id` identifies the persistent world and its worker-owned state.
- `event_id` identifies a prompt/control update and remains monotonic across reset.
- `chunk_index` is contiguous from zero and restarts from zero after reset.
- `request_id` correlates one chunk snapshot with its output metadata.
- AsyncOmni uses a separate UUID-suffixed internal engine routing ID.

The generic runtime transports controls as opaque snapshots. LingBot's adapter
accepts:

- `lingbot.camera_actions.v1` for per-latent-frame key states such as `w`, `a`,
  `s`, `d`, `i`, `j`, `k`, and `l`;
- `lingbot.camera_trajectory.v1` for explicit pose/intrinsics trajectories.

## Validation

Real-checkpoint validation uses 480x832 output, four DMD steps, and seed 42.
The exercised matrix includes:

- TP=1 and TP=2 execution;
- two interleaved resident sessions;
- action input and prompt switching;
- seven contiguous chunks crossing the sink plus recent rolling window;
- direct versus paged replay;
- CUDA Graph execution;
- reset, close, failure cleanup, and exact metadata matching; and
- nine-frame VAE decode.

Official `generate.py` accuracy and performance numbers, together with
generation artifacts, are recorded in the PR validation section for the exact
tested commit.

## Current limitations

- Only the 14B causal-fast checkpoint is supported.
- The tick control plane is internal; the public transport is the stepwise
  `WS /v1/realtime/video` path, which cannot take mid-session camera
  interaction yet.
- Stepwise serving requires an explicit
  `--deploy-config vllm_omni/deploy/lingbot_world_v2_stepwise.yaml`; there is
  no default deploy config for this model.
- AR-Diffusion stages currently require one replica because session-affine
  routing across replicas is not implemented.
- Tick mode generates one AR block per request. Stepwise mode generates many
  AR blocks in one request. `max_num_seqs` must be one in both cases.
- Stateful streaming VAE decode is not implemented; the realtime example emits
  latent chunks.
- SP/USP, pipeline/CFG parallelism, HSDP, VAE parallelism, quantization,
  Cache-DiT, TeaCache, causal-pretrain, and the 1.3B checkpoint are not claimed.
- No AMD GPU, Ascend NPU, or Intel GPU support is claimed.

## References

- Checkpoint: <https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast-diffusers>
- Official implementation: <https://github.com/robbyant/lingbot-world-v2>
- Offline example: [`examples/offline_inference/diffusion/lingbot_world_v2.py`](../../examples/offline_inference/diffusion/lingbot_world_v2.py)
- Streaming serving client: [`examples/online_serving/streaming_video_generation/`](../../examples/online_serving/streaming_video_generation/README.md)
- Streaming serving end-to-end test: [`tests/e2e/online_serving/test_lingbot_world_v2_stepwise.py`](../../tests/e2e/online_serving/test_lingbot_world_v2_stepwise.py)
- Stepwise latent-level end-to-end test: [`tests/e2e/offline_inference/test_lingbot_world_v2_stepwise.py`](../../tests/e2e/offline_inference/test_lingbot_world_v2_stepwise.py)
- Realtime design: [`docs/design/feature/realtime_ar_diffusion.md`](../../docs/design/feature/realtime_ar_diffusion.md)
