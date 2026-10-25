# Hy-Nomas

Hy-Nomas generates video autoregressively across text-controlled segments, with
optional camera trajectories. This integration exposes the distilled checkpoint
through `Omni`, `/v1/videos`, and AR-Diffusion. The pipeline owns the DiT, Qwen text
encoder, PixelUMM decoder, and generation loop. Component implementations and the
Euler scheduler come from the pinned reference package, which is a runtime dependency.
The DiT uses the standard `weights_sources` and `load_weights` interfaces.

## Installation

Use Python 3.12 and a CUDA environment compatible with both vLLM-Omni and the
Hy-Nomas reference kernels. Install vLLM-Omni first. Install the reference checkout
without replacing vLLM-Omni's dependency versions:

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-Nomas.git
git -C Hy-Nomas checkout --detach 911cea6458a89c2663513ff7099561ec190dd491
python -m pip install --no-deps -e ./Hy-Nomas
python -m pip install hydra-core==1.3.2 omegaconf==2.3.0 loguru einops
```

Keep this editable checkout: the adapter reads its `configs/config.yaml` and
`configs/experiment/inference.yaml`. Install FlashAttention 2 for the environment's
PyTorch/CUDA versions following the reference repository's installation instructions.
FlexAttention supplies segment and camera range masks; the reference text/refiner
path still requires FlashAttention 2. Verify imports before loading weights:

```bash
python -c 'from hymm.pipeline.causal_ar import CPTCausalInferencePipeline'
```

The validated environment uses Python 3.12, PyTorch 2.13.0/CUDA 13.0, vLLM 0.32.0,
Transformers 5.16.1, Diffusers 0.40.0, and FlashAttention 2.8.3 on H200.
The supplied deployment profiles select the `mp` executor. Python callers must set
`distributed_executor_backend="mp"` for single-device execution. The pipeline checks
process isolation before importing reference components, which use process-global
Torch state. SDPA backend selection is scoped to component execution inside the worker.

## Checkpoints

`tencent/Hy-Nomas` and a local checkpoint directory are recognized without a generated
`model_index.json`. The pipeline requires these two checkpoint files:

- `DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors`
- `VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors`

Download these files into a local directory before using the shared entrypoints:

```bash
hf download tencent/Hy-Nomas \
  DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors \
  VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors \
  --local-dir ./Hy-Nomas-weights
```

Passing a Hub ID directly to `Omni` or `vllm serve` follows the standard snapshot
download behavior and can fetch other checkpoint variants. Using the prepared local
directory limits the download to the inference files above.

The DiT checkpoint must contain every required parameter, including camera weights.

The text encoder separately loads `Qwen/Qwen2.5-VL-7B-Instruct`. A local Hy-Nomas
directory must retain the same two relative paths. To use a local text encoder,
pass `custom_pipeline_args={"text_encoder": "/path/to/Qwen2.5-VL-7B-Instruct"}` to
`Omni` or the equivalent `--custom-pipeline-args` JSON to the server.

## Offline generation

Run the shared text-to-video example from the vLLM-Omni repository:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model ./Hy-Nomas-weights --enforce-eager \
  --deploy-config vllm_omni/deploy/hy_nomas.yaml \
  --prompt 'A forest beside a river. ||| The camera moves along the river.' \
  --height 704 --width 1280 --num-frames 121 --num-inference-steps 8 \
  --guidance-scale 1 --fps 24 --seed 42 \
  --extra-body '{"segment_frame_boundaries":[0,60,121]}' \
  --output hy_nomas.mp4
```

Each `|||`-separated text is one segment, not an independent batch item. For
structured prompts, pass each already-prepared reference segment JSON as that
segment's text. The adapter does not run the separate Nomas-PE prompt enhancer or
the reference dataset/benchmark prompt-file loader.

`segment_frame_boundaries` uses pixel-frame indices: start at zero, end at
`num_frames`, and provide one interval per segment. Internal boundaries must be
multiples of four. A single segment defaults to `[0, num_frames]`.

Optional offline `--extra-body` fields `camera_trajectory` and `segment_has_camera` are
forwarded to the reference camera-conditioning path. The trajectory is a reference
generator name such as `pan_left`, or a JSON/NPZ trajectory path accessible to the
worker. `segment_has_camera`, when specified, contains one boolean per segment.

## Online generation

Start the server:

```bash
vllm serve ./Hy-Nomas-weights --omni --port 8098 \
  --deploy-config vllm_omni/deploy/hy_nomas.yaml
```

In another terminal, submit a video job:

```bash
curl -sS http://localhost:8098/v1/videos \
  -F 'prompt=A forest beside a river.' -F 'size=1280x704' \
  -F 'num_frames=121' -F 'fps=24' -F 'num_inference_steps=8' \
  -F 'guidance_scale=1' -F 'seed=42'
```

Use the returned `id` with `GET /v1/videos/{id}` until its status is `completed`
or `failed`. Download completed output from `GET /v1/videos/{id}/content`.
Multi-segment and camera parameters use the endpoint's JSON `extra_params` form field,
for example `-F 'extra_params={"segment_frame_boundaries":[0,60,121],"camera_trajectory":"pan_left","segment_has_camera":[true,true]}'`
with a two-segment prompt and 121 frames.

Specify the frame count with `num_frames`, or omit it and pass
`-F 'extra_params={"duration":4}'` to request 97 frames at 24 fps, including the
initial frame. Offline callers can pass `--extra-body '{"duration":4}'`.
An explicit `num_frames` must agree with `duration`; fractional durations must
span a positive multiple of four frame intervals. Use these model parameters
instead of the top-level `seconds` field to select the clip length.
`generate_sound=false` is accepted; audio generation is unsupported.

## Sequence parallelism

Use two CUDA devices with the shared offline example's `--ulysses-degree 2`,
or start a two-device server:

```bash
vllm serve ./Hy-Nomas-weights --omni --enforce-eager --ulysses-degree 2
```

For the Python API, pass `parallel_config={"ulysses_degree": 2}` to `Omni`.
The adapter reuses the worker's Ulysses process group for the reference DiT's
sequence splitting, all-to-all attention, and output gathering. Text encoding and
PixelUMM decoding run independently on each device. Weights remain replicated.

The degree must divide the DiT's 16 attention heads. The number of spatial tokens
per latent frame must also divide evenly across devices; invalid dimensions are
rejected before inference. The default 704×1280 shape meets these constraints.
Use strict Ulysses with the reference all-to-all implementation. Ring attention,
AllGather-KV, UAA, tensor/pipeline/data/CFG parallelism, HSDP, text encoder sharding,
and VAE parallelism are not supported by this adapter.
Real-model validation covers Ulysses degree two on H200; larger degrees have not
been measured.

## CPU offload

Use the public module-offload configuration to select DiT and text encoder weights:

```bash
vllm serve ./Hy-Nomas-weights --omni \
  --deploy-config vllm_omni/deploy/hy_nomas.yaml \
  --diffusion-offload-config '{"mode":"module","components":["dit","text_encoder"]}'
```

The compatibility option `--enable-cpu-offload` stages the DiT, text encoder, and
PixelUMM decoder. Ordinary request inference keeps the DiT on the device throughout
each chunk's denoising and KV writes. Camera projection uses its own context. AR step execution obtains
a residency context for each DiT call, so module offload adds weight transfers per
step. Text encoding and VAE decoding use their own component contexts, including
cleanup after exceptions. Explicit component selection covers
only the named DiT/text encoder and leaves the VAE resident, as defined by the
shared offload API.

Offload reduces generation-time weight residency and adds host-device transfers.
Checkpoint initialization places components on the GPU; allow enough
memory for all components during loading. Module CPU offload can be combined with
Ulysses SP; each rank stages its own component copies. Layerwise and distributed
layerwise offload are not supported.

## Streaming and live interaction

Use the AR-Diffusion deployment to stream video through `/v1/realtime/video`:

```bash
vllm serve ./Hy-Nomas-weights --omni --port 8098 \
  --deploy-config vllm_omni/deploy/hy_nomas_ar.yaml
```

The deployment uses two GPUs with Ulysses SP. Each chunk contains up to 16 latent
frames. `denoise_step` performs one DiT forward, and `step_scheduler` performs one
reference Euler update. At the chunk boundary, the adapter recomputes KV using the
reference cache-noise setting and commits every layer atomically into AR-Diffusion's
paged storage. Text is cached once per prompt segment; a live prompt update starts
a new segment at the next boundary.

PixelUMM emits only stable frames through `push()`. Its overlapping temporal windows
retain a tail between chunks, so the number of output frames can vary and short
chunks can emit no frames. Completing the request calls `finish()` and emits the
remaining tail. Frames already sent are never replaced. Stopping or disconnecting
cancels the request and releases its session, including any withheld tail.

The existing client supports prompt and camera updates:

```bash
python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --model ./Hy-Nomas-weights --port 8098 --no-helios-distilled-preset \
  --prompt 'A forest beside a river.' --height 704 --width 1280 \
  --num-frames 241 --num-inference-steps 8 --guidance-scale 1 --fps 24 --seed 42 \
  --prompt-updates '[{"at":2,"prompt":"The camera approaches a wooden bridge.","transition_chunks":0}]' \
  --camera-updates '[{"at":3,"camera":{"camera_trajectory":"pan_left"},"transition_chunks":0}]' \
  --output /path/to/results/hy_nomas_stream.mp4
```

Hy-Nomas applies discrete updates with `transition_chunks=0` (or omitted when
sending protocol messages directly). A camera update uses
`event.multi_modal_data.camera={"camera_trajectory":"pan_left"}`; a null
`camera_trajectory` disables camera control. The payload accepts the same reference
generator names and trajectory files as offline generation. A composite prompt and
camera update is validated before either part is queued. Chunk metadata acknowledges
applied and coalesced event IDs.

The deployment reserves history for up to 120 latent frames (approximately 20 seconds),
16 prompt segments, and one active session. Configure `ar_height`, `ar_width`,
`ar_chunk_latent_frames`, `ar_max_latent_frames`, `ar_max_segments`, and
`ar_session_capacity` under `custom_pipeline_args`. Requests must fit the deployed
spatial limits. Capacity exhaustion returns an error; it does not silently discard
history. Increasing these limits increases the required memory. The fixed noise
timeline uses `ar_max_latent_frames`, so keep it fixed when comparing seeds.

## Multi-turn continuation

Persistent turns use the AR-Diffusion tick protocol with `step_execution=false`.
Every tick supplies a `session_id`, consecutive `chunk_index` starting at zero,
and its prompt/control snapshot. The worker retains KV, the noise generator and
PixelUMM decoder across turns. Height, width and seed remain fixed until reset.
Each tick must fit one chunk; the default 16-latent limit accepts up to 61 frames.

Run the session example with two successive prompts:

```bash
python examples/offline_inference/diffusion/hy_nomas_session.py \
  --model ./Hy-Nomas-weights --ulysses-degree 2 \
  --prompts 'A forest beside a river.' 'The camera approaches a wooden bridge.' \
  --height 704 --width 1280 --num-frames 49 --num-inference-steps 8 \
  --output-dir /path/to/results/hy_nomas_session
```

The example uses `ARDiffusionOmniTickConsumer` and writes each newly stable chunk,
the complete video, and frame/identity metadata. Its final tick sets
`close_session=true`, which flushes the decoder before the runner releases the
session. `reset=true` starts a fresh timeline for the same ID. Explicit lifecycle
close, cancellation, failed forwards and runner eviction discard model state and
release KV pages. The streaming request API owns one active request; persistent
tick sessions use their separate IDs and lifecycle calls.

For typed camera controls, use `track="camera"`, `schema="hy_nomas.camera.v1"`, and
`data={"camera_trajectory":"pan_left"}`. Prompt snapshots use the tick's `prompt`
field. The returned `metadata.ar_diffusion` preserves tick identity and applied
event IDs. A turn can include frames withheld from the previous turn; concatenate
the returned chunks in order, using `metadata.hy_nomas.frame_start`.

AR mode uses the reference chunk-AR schedule. The ordinary request path continues
to use the reference segment-AR schedule; comparisons must use matching chunk
boundaries, noise capacity, prompts, camera inputs, sampling steps and precision.

## Scope and validation

- CUDA, BF16 distilled DiT, BF16 PixelUMM, one output per request.
- Four, six, or eight denoising steps; guidance scale one and flow shift five.
  Engine-level and request-level flow-shift overrides use the same constraint.
- 24 fps, dimensions divisible by 32, and `num_frames=4*n+1` with `n >= 1`.
  The reference's 720-pixel height is reduced to 704 by its 32-pixel latent grid;
  this API requires the actual output height explicitly and defaults to 704.
- Ordinary requests release their reference KV history after completion. AR-Diffusion
  owns persistent KV and session cleanup for streaming and continuation.
- Generic startup warmup is disabled because its two-step, CFG-zero request is
  outside the distilled model's sampling contract. The first real request includes
  reference-kernel compilation.
- Use `--enforce-eager`. Full-transformer compilation is rejected. A request for
  generic regional compilation reports that the reference cached forward method
  is unsupported and keeps the reference execution path; FlexAttention kernels
  retain their own compilation.
- Quantization, LoRA, Cache-DiT, latent output, and input images are unsupported.

CPU contract tests run in the existing L1 diffusion-model job. GPU tests require
the external reference runtime and are selected explicitly with `local_model`:

```bash
pytest tests/diffusion/models/hy_nomas/test_hy_nomas.py -m 'core_model and cpu' --run-level core_model
pytest tests/diffusion/models/hy_nomas/test_reference_config.py -m 'local_model and cpu' --run-level core_model
pytest tests/diffusion/models/hy_nomas/test_reference_parallel.py -m 'local_model and cpu' --run-level core_model
pytest tests/diffusion/models/hy_nomas/test_ar.py -m 'core_model and cpu' --run-level core_model
pytest tests/diffusion/models/hy_nomas/test_reference_streaming.py -m 'local_model and cpu' --run-level core_model
pytest tests/diffusion/models/hy_nomas/test_reference_parity.py -m 'local_model and cuda' --run-level core_model
pytest tests/e2e/offline_inference/test_hy_nomas.py tests/e2e/online_serving/test_hy_nomas.py \
  -m 'local_model and cuda' --run-level core_model
```

GPU validation requires the reference runtime, model weights, text encoder, and a
CUDA device with enough memory for all three components plus KV history. The local
test configuration targets one H200 for single-device cases and two H200s for SP.
Select SP endpoint tests with `-m 'local_model and cuda and cards_2'`.
The reference-parity test loads independent
upstream and Omni instances, and checks exact decoded-frame equality for text,
segments, and camera control in resident and offload modes. It requires memory for
both model instances; set `HY_NOMAS_MODEL` and optionally `HY_NOMAS_TEXT_ENCODER` to
local checkpoint directories. The 256×256, nine-frame cases check execution,
4/6-step sampling, and sequential requests in resident and offload modes. Quality acceptance additionally
requires identical seeds, structured segment prompts, dimensions, frame boundaries,
sampling parameters, and camera trajectories in reference and Omni runs, including
a longer multi-segment sequence.

The AR CPU tests exercise the real KV allocator, state binding, failure cleanup,
interactions and batch metadata with substituted learned model components. The
reference streaming CPU tests exercise PixelUMM's actual window bookkeeping with
a substituted window predictor. Neither validates learned video quality or GPU
collectives; streaming and continuation require separate real-model acceptance.
