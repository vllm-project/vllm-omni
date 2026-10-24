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

The current development checks do not establish runtime compatibility of the
reference package with every vLLM-Omni dependency version. GPU generation and
reference-output comparison remain required before treating this draft as validated.

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

Optional `extra_body` fields `camera_trajectory` and `segment_has_camera` are
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
Multi-segment and camera parameters use the endpoint's JSON `extra_body` form field.

## Scope and validation

- Single CUDA device, BF16 distilled DiT, BF16 PixelUMM, one output per request.
- Four, six, or eight denoising steps; guidance scale one and flow shift five.
- 24 fps, dimensions divisible by 32, and `num_frames=4*n+1` with `n >= 1`.
  The reference's 720-pixel height is reduced to 704 by its 32-pixel latent grid;
  this API requires the actual output height explicitly and defaults to 704.
- KV history is shared between segments within one request and released on success
  or failure. Requests do not share an interactive session.
- Generic startup warmup is disabled because its two-step, CFG-zero request is
  outside the distilled model's sampling contract. The first real request includes
  reference-kernel compilation.
- Parallelism, vLLM-Omni offload, quantization, LoRA, Cache-DiT, latent output,
  input images, and incremental streaming are not supported by this adapter.

CPU contract tests run in the existing L1 diffusion-model job. GPU tests are local
validation cases while the reference dependencies and generation remain unverified:

```bash
pytest tests/diffusion/models/hy_nomas/test_hy_nomas.py -m 'core_model and cpu' --run-level core_model
pytest tests/diffusion/models/hy_nomas/test_reference_config.py -m 'local_model and cpu' --run-level core_model
pytest tests/e2e/offline_inference/test_hy_nomas.py tests/e2e/online_serving/test_hy_nomas.py \
  -m 'local_model and cuda' --run-level core_model
```

GPU validation requires the reference runtime, model weights, text encoder, and a
CUDA device with enough memory for all three components plus KV history. The local
test configuration targets one H200. The 256×256, nine-frame cases check execution
and request isolation, not generation quality. Quality acceptance additionally
requires identical seeds, structured segment prompts, dimensions, frame boundaries,
sampling parameters, and camera trajectories in reference and Omni runs, including
a longer multi-segment sequence.
