# LingBot-Video

LingBot-Video uses one `LingBotVideoPipeline` for text-to-image (T2I),
text-to-video (T2V), and text-image-to-video (TI2V) generation. Both the dense
and MoE checkpoints use the same request format.

## Start the server

```bash
MODEL=robbyant/lingbot-video-dense-1.3b bash run_server.sh
```

The MoE checkpoint uses the same server and request scripts, but requires
substantially more GPU memory:

```bash
MODEL=robbyant/lingbot-video-moe-30b-a3b bash run_server.sh
```

### Memory offload

Use model-level (sequential) CPU offload when the Base checkpoint and shared
components do not fit on one GPU. The Qwen3-VL text encoder and Base Transformer
take turns on the GPU; the VAE remains resident by default:

```bash
MODEL=robbyant/lingbot-video-moe-30b-a3b \
  bash run_server.sh --enable-cpu-offload --enforce-eager
```

For the dense checkpoint, layerwise offload instead keeps the Transformer block
weights in pinned CPU memory and prefetches blocks as they execute. The text
encoder, VAE, and non-block Transformer modules remain GPU-resident:

```bash
MODEL=robbyant/lingbot-video-dense-1.3b \
  bash run_server.sh --enable-layerwise-offload --enforce-eager
```

The two offload modes are mutually exclusive. Both reduce GPU memory at the
cost of extra CPU-GPU transfers and startup work; they are capacity features,
not speedups.

For an additional low-memory T2I/T2V request, include
`"offload_vae_during_denoise":true` in `extra_params`. LingBot temporarily moves
the VAE to CPU during denoising and restores it before decode. This option is
separate from the shared sequential policy, where the VAE normally remains on
GPU.

CPU/layerwise offload does not load the optional Refiner by itself. Refiner
loading is an explicit startup choice.

### Optional Refiner

The official MoE package stores a second 30B-A3B Transformer under `refiner/`.
Enable it at server startup through the diffusion stage's `model_config`.
Sequential CPU offload keeps only the active Base or Refiner DiT on the GPU:

```bash
MODEL=robbyant/lingbot-video-moe-30b-a3b \
  bash run_server.sh \
  --enable-cpu-offload --enforce-eager \
  --stage-overrides '{"0":{"model_config":{"lingbot_refiner":{"enabled":true,"default_run":false,"offload_vae_during_denoise":true}}}}'
```

`default_run=false` keeps ordinary video requests on the Base path. Set
`RUN_REFINER=true` in the video client to opt into the already-loaded Refiner:

```bash
RUN_REFINER=true bash run_curl_text_image_to_video.sh
```

The same command supports TI2V when `INPUT_IMAGE` is set. T2I bypasses the
Refiner by default and rejects an explicit `run_refiner=true`. Base and Refiner
share one text encoder, processor, and VAE, but use independent native weight
sources, schedulers, generators, and denoise schedules.

The client defaults to a small `320x192`, 9-frame Refiner validation workload.
Production deployments can override `REFINER_HEIGHT`, `REFINER_WIDTH`,
`REFINER_STEPS`, and `REFINER_MAX_VIDEO_FRAMES`; the official high-resolution
default is `1920x1088`. Both CPU offload and HSDP are capacity strategies and
may increase latency.

## Text to image

The image endpoint selects T2I mode and always generates one frame:

```bash
bash run_curl_text_to_image.sh
```

The script sends a `320x192`, two-step smoke request and writes
`lingbot_t2i.png`.

## Text or text-image to video

Run the video script without an image to select T2V mode:

```bash
bash run_curl_text_image_to_video.sh
```

Pass a first-frame image to the same script to select TI2V mode:

```bash
INPUT_IMAGE=/path/to/input.png bash run_curl_text_image_to_video.sh
```

The client scripts omit the optional `model` request field, so they target
whichever dense or MoE checkpoint the server loaded. The video example uses the
lightweight `320x192`, 9-frame, two-step configuration.

Until the shared `/v1/videos` reference-image resizing is removed, TI2V target
dimensions must be sent through `extra_params`, for example
`{"size":"320x192"}`. Do not use the top-level `size`, `width`, or `height`
fields for TI2V because the serving layer currently applies those dimensions
to the reference image before the model receives it. T2V requests can continue
to use the top-level dimension fields.

LingBot video frame counts use the causal VAE `4n+1` grid. The pipeline rounds
any requested frame count upward to the next valid value. An explicit
`num_frames` takes precedence over `seconds`; otherwise, the server first
resolves `seconds * fps` and the pipeline applies the same alignment.

Official `resolution`/`ratio` presets can be sent through `extra_params`, for
example `{"resolution":"720p","ratio":"16:9"}`. The `2k` and `4k` entries
only define output dimensions; whether they run successfully depends on the
checkpoint, GPU memory, and memory optimizations available in the deployment.

For `/v1/images/generations`, the server resolves these aliases to their final
pixel dimensions before applying `--max-generated-image-size`. Requests above
the configured limit return HTTP 400 before engine dispatch. LingBot produces
one output per prompt; image requests with `n>1` are also rejected with HTTP
400.

LingBot TI2V accepts exactly one image reference. Image editing, video
references, audio references, and batching are not supported by this pipeline
mode. When the Refiner is loaded, TI2V rebuilds text-only Refiner conditioning
and reinjects the geometry-aligned clean first-frame latent after every step.
