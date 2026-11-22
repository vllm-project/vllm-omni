# Cosmos3-Nano-Sim-Transfer — GB200

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Cosmos3-Nano-Sim-Transfer`, loaded with
  `Cosmos3NanoSimTransferPipeline`
- Task: control-video-conditioned video generation from an edge, blur, depth,
  or segmentation hint
- Mode: offline full-clip inference (`offline_transfer`)
- Hardware: NVIDIA GB200
- Maintainer: Maciej Bala

## When to use this recipe

Use it to generate one complete clip that follows a single control video. Each
request is an independent full rollout: tick transport, streaming, and session
continuation are not supported. For action-conditioned rollouts, use
[Cosmos3-Nano-Sim-Bimanual](Cosmos3-Nano-Sim-Bimanual.md).

## Supported model contract

### Inputs

The offline adapter reads one JSON record.

| Field | Contract |
| --- | --- |
| `prompt` or `prompt_path` | Plain text, or a structured JSON caption passed to the pipeline as JSON. `--prompt-path` overrides the record. |
| Hint | Exactly one of `edge`, `blur`, `depth`, or `seg`, as `true` or an object. Edge and blur can be computed from `vision_path`; depth and segmentation normally provide `control_path` in the hint object. |
| `vision_path` | Optional source video. When set, it must provide at least `num_frames` frames. |
| `num_frames` | Positive pixel frame count; default 97. |
| Window settings | Optional `kv_cache_inference_size`, `attention_sink_size`, and `emphasize_control_in_prompt`. |

Relative caption and control paths are resolved against the JSON record's
directory. Direct pipeline requests can pass the control clip as
`control_video`, with a named hint or `control_hint`; it cannot be combined
with a hint's `control` or `control_path`. Requests with actions, seed images,
or `initial_latent` are rejected. The control weight, `control_guidance`, and
`guidance_scale` must all be 1.0, and `num_first_chunk_conditional_frames` must
be 0.

### Outputs

| Output | Contract |
| --- | --- |
| Video | One clip with exactly `num_frames` frames. Generation pads to the VAE's `4k+1` grid and trims the result. |
| Frame rate | The decoded source video's FPS when it carries one; otherwise the request FPS (`--fps`), then the deployment `default_fps` (15). |
| Geometry | `resolution` selects the 256, 480, 704, or 720 bucket (CLI fallback: 480). Without `aspect_ratio`, the source video, else the hint's control clip, selects the nearest aspect. Inputs are resized directly to the bucket without center cropping. |
| Latent | `--output-type latent` saves the pre-VAE latent tensor with `torch.save`. |

The deployment's `max_pixels: 921600` (720×1280) bounds every bucket, and the
shared Cosmos3-Nano-Sim-Bimanual resolution policy validates the result.

### History window

The window counts temporal frames, including sinks and the current frame. Each
retained past frame holds both its control and clean latents. The checkpoint
manifest sets the default; the converted checkpoint used here keeps 51 frames
with 1 sink. Deployment `model_config.kv_cache_inference_size` and
`attention_sink_size` replace that default, and request fields override both.
Sinks require a checkpoint chunk size of one. A direct pipeline request with
`kv_cache_inference_size: null` keeps full history and requires
`attention_sink_size: 0`. Changing the window can change generated outputs.

## References

- [Supported models](../../docs/models/supported_models.md) and the
  [diffusion feature matrix](../../docs/user_guide/diffusion_features.md)
- Base model recipe: [Cosmos3-Nano](Cosmos3-Nano.md)
- Sibling recipe: [Cosmos3-Nano-Sim-Bimanual](Cosmos3-Nano-Sim-Bimanual.md)
- Default deployment:
  [`cosmos3_nano_sim_transfer.yaml`](../../vllm_omni/deploy/cosmos3_nano_sim_transfer.yaml)

## Hardware

- Accelerator model and per-device memory: NVIDIA GB200, 186 GB HBM3e per GPU
- Number of devices: 1
- Device interconnect: not used by the single-GPU deployment
- Host memory: not recorded
- Qualification scope: single-GPU offline inference with the default dense
  deployment. No memory or latency profile is recorded yet.

## Software environment

- OS: Linux (distribution not recorded)
- Python: not recorded (vLLM-Omni supports 3.10–3.13)
- Driver / runtime: not recorded
- vLLM version: not recorded
- vLLM-Omni version or commit: the commit that adds this recipe
- Extra packages: `imageio[ffmpeg]` to read `vision_path` and control videos

## Command

For the depth example, save this record as `/data/sim_transfer_depth.json` and
place `depth.mp4` beside it.

```json
{"prompt":"A robot moves a block.","depth":{"control_path":"depth.mp4"},"num_frames":121}
```

```bash
python -m vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.offline_transfer \
  --model nvidia/Cosmos3-Nano-Sim-Transfer \
  --input-json /data/sim_transfer_depth.json \
  --resolution 480 \
  --num-frames 121 \
  --fps 30 \
  --kv-cache-inference-size 30 \
  --attention-sink-size 3 \
  --no-emphasize-control-in-prompt \
  --seed 42 \
  --output cosmos3_nano_sim_transfer.mp4
```

The window, sink, emphasis, and aspect-ratio options override the same JSON
fields. Omitted settings come from the deployment, then the checkpoint.

## Verification

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,r_frame_rate,nb_read_frames \
  -of default=noprint_wrappers=1 cosmos3_nano_sim_transfer.mp4
```

Expect `nb_read_frames=121` and `r_frame_rate=30/1`, at the 480p bucket that
matches the control clip's aspect ratio (for example 832×480 for 16:9). Review
the video against the depth sequence. To compare runs, use the same checkpoint,
input, seed, and flags, save `--output-type latent`, and compare the tensors
before comparing decoded videos.

## Notes

- Memory usage: not profiled yet. The dense K/V storage holds the text plus two
  latent frames (control and clean) for each frame in the window.
- Key flags: the default deployment runs eager with dense K/V and overlaps Wan
  VAE decode with generation for fresh full-video CUDA requests. Set
  `model_config.overlap_vae_decode: false` for serial decode. Control hints are
  VAE-encoded up front, then consumed one checkpoint chunk at a time.
- Known limitations:
    - Other accelerators are not qualified by this recipe.
    - Online serving declares the same request fields but is not qualified here.
      Online requests cannot send `kv_cache_inference_size: null`; set a full
      history window in the deployment instead.
    - One Transfer session runs per worker; use stage `num_replicas` for
      independent workers.

## Supported features

| Feature | Status on GB200 | Guide |
| --- | --- | --- |
| Regional compilation and CUDA graphs | Opt-in, not qualified: set `enforce_eager: false`, `diffusion_compile_granularity: regional`, and `model_config.use_cuda_graphs: true` | [Regional compilation](../../docs/user_guide/diffusion/regional_compilation.md) |
| Tensor parallelism | ❓ inherited from Sim-Bimanual, not validated for Transfer | [Tensor parallel](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md) |
| Sequence, pipeline, data, and VAE-patch parallelism | ❌ rejected at load time | — |
| Cache-DiT / TeaCache | ❌ disabled for this pipeline | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| CFG parallelism | ❌ distilled inference requires guidance 1.0 | — |
| Quantized diffusion KV cache | ❌ rejected at load time | — |
| Tick transport and session continuation | ❌ rejected at admission | — |
