# Hy-Nomas

Hy-Nomas generates video autoregressively across text-controlled segments, with
optional camera trajectories. This integration exposes the distilled checkpoint
through `Omni` and `/v1/videos`. It calls Tencent's reference DiT, Qwen text encoder,
Euler sampler, and PixelUMM decoder; the reference package is a runtime dependency.

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
The adapter scopes PyTorch SDPA backend selection to each inference call to preserve
the reference's cuDNN-enabled policy, then restores the previous settings.

## Checkpoints

`tencent/Hy-Nomas` is recognized without a generated `model_index.json`. Only these
two checkpoint files are requested by the pipeline:

- `DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors`
- `VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors`

The text encoder separately loads `Qwen/Qwen2.5-VL-7B-Instruct`. A local Hy-Nomas
directory must retain the same two relative paths. To use a local text encoder,
pass `custom_pipeline_args={"text_encoder": "/path/to/Qwen2.5-VL-7B-Instruct"}` to
`Omni` or the equivalent `--custom-pipeline-args` JSON to the server.

## Offline generation

Run the shared text-to-video example from the vLLM-Omni repository:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model tencent/Hy-Nomas --enforce-eager \
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
vllm serve tencent/Hy-Nomas --omni --enforce-eager --port 8098
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

## Sequence parallelism

Use two CUDA devices with the shared offline example's `--ulysses-degree 2`,
or start a two-device server:

```bash
vllm serve tencent/Hy-Nomas --omni --enforce-eager --ulysses-degree 2
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
vllm serve tencent/Hy-Nomas --omni --enforce-eager \
  --diffusion-offload-config '{"mode":"module","components":["dit","text_encoder"]}'
```

The compatibility option `--enable-cpu-offload` stages the DiT, text encoder, and
PixelUMM decoder. The DiT stays on the device for the complete camera-projection,
denoising, and KV-write phase. Text encoding and VAE decoding use their own component
contexts, including cleanup after exceptions. Explicit component selection covers
only the named DiT/text encoder and leaves the VAE resident, as defined by the
shared offload API.

Offload reduces generation-time weight residency and adds host-device transfers.
Checkpoint initialization still uses the reference loaders on the GPU; allow enough
memory for all components during loading. Module CPU offload can be combined with
Ulysses SP; each rank stages its own component copies. Layerwise and distributed
layerwise offload are not supported.

## Scope and validation

- CUDA, BF16 distilled DiT, BF16 PixelUMM, one output per request.
- Four, six, or eight denoising steps; guidance scale one and flow shift five.
- 24 fps, dimensions divisible by 32, and `num_frames=4*n+1` with `n >= 1`.
  The reference's 720-pixel height is reduced to 704 by its 32-pixel latent grid;
  this API requires the actual output height explicitly and defaults to 704.
- KV history is shared between segments within one request and released on success
  or failure. Requests do not share an interactive session.
- Generic startup warmup is disabled because its two-step, CFG-zero request is
  outside the distilled model's sampling contract. The first real request includes
  reference-kernel compilation.
- Quantization, LoRA, Cache-DiT, latent output,
  input images, and incremental streaming are not supported by this adapter.

CPU contract tests run in the existing L1 diffusion-model job. GPU tests require
the external reference runtime and are selected explicitly with `local_model`:

```bash
pytest tests/diffusion/models/hy_nomas/test_hy_nomas.py -m 'core_model and cpu' --run-level core_model
pytest tests/diffusion/models/hy_nomas/test_reference_config.py -m 'local_model and cpu' --run-level core_model
pytest tests/diffusion/models/hy_nomas/test_reference_parallel.py -m 'local_model and cpu' --run-level core_model
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
