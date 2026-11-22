# Cosmos3-Nano-Sim-Transfer — GB200

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Cosmos3-Nano-Sim-Transfer`, loaded with
  `Cosmos3NanoSimTransferPipeline`
- Task: control-video-conditioned video generation from an edge, blur, depth,
  or segmentation hint
- Mode: offline full-clip inference with `Omni.generate()`
- Hardware: NVIDIA GB200
- Maintainer: Maciej Bala

## When to use this recipe

Use it to generate one complete clip that follows a single control video. Each
request is an independent full rollout: tick transport, streaming, and session
continuation are not supported. For action-conditioned rollouts, use
[Cosmos3-Nano-Sim-Bimanual](Cosmos3-Nano-Sim-Bimanual.md).

## Supported model contract

### Inputs

Pass the caption and optional source video in the prompt dictionary. Pass the
frame count in `OmniDiffusionSamplingParams` and the hint and window settings in
`extra_args`.

| Field | Contract |
| --- | --- |
| `prompt` | Plain text or a structured caption serialized as a JSON string. |
| Hint | Exactly one of `edge`, `blur`, `depth`, or `seg`, as `True` or a dictionary. Edge and blur can be computed from the source video. Depth and segmentation typically use `control_path` in the hint dictionary. |
| `multi_modal_data["video"]` | Optional source video, such as a list of RGB PIL frames. Provide at least `num_frames` frames. |
| `num_frames` | Positive pixel frame count. |
| Window settings | Optional `kv_cache_inference_size`, `attention_sink_size`, and `emphasize_control_in_prompt`. |

You can pass the control clip as `control_video` with a named hint or
`control_hint`. Supply one control source per request. Requests with actions,
seed images, or `initial_latent` are rejected. The control weight,
`control_guidance`, and `guidance_scale` must all be 1.0, and
`num_first_chunk_conditional_frames` must be 0.

### Outputs

| Output | Contract |
| --- | --- |
| Video | One clip with exactly `num_frames` frames. Generation pads to the VAE's `4k+1` grid and trims the result. |
| Frame rate | The decoded source video's FPS when it carries one; otherwise the request `frame_rate` or `fps`, then the deployment `default_fps` (15). |
| Geometry | `resolution` selects the 256, 480, 704, or 720 bucket; the deployment defaults to 720 (`default_resolution: [720, 1280]`). Without `aspect_ratio`, the source video, else the hint's control clip, selects the nearest aspect. Inputs are resized directly to the bucket without center cropping. |
| Latent | Set `output_type="latent"` to return the pre-VAE latent tensor. |

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
- Extra packages: `imageio[ffmpeg]` for video decoding and MP4 export

## Offline inference

Run the example from the repository root with a depth clip at `/data/depth.mp4`.

```python
from diffusers.utils import export_to_video

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if __name__ == "__main__":
    fps = 30
    params = OmniDiffusionSamplingParams(
        num_frames=121,
        num_inference_steps=4,
        frame_rate=fps,
        guidance_scale=1.0,
        seed=42,
        output_type="np",
        extra_args={
            "depth": {"control_path": "/data/depth.mp4"},
            "resolution": "480",
            "kv_cache_inference_size": 30,
            "attention_sink_size": 3,
            "emphasize_control_in_prompt": False,
        },
    )
    omni = Omni(
        model="nvidia/Cosmos3-Nano-Sim-Transfer",
        model_class_name="Cosmos3NanoSimTransferPipeline",
        deploy_config="vllm_omni/deploy/cosmos3_nano_sim_transfer.yaml",
    )
    try:
        outputs = omni.generate({"prompt": "A robot moves a block."}, params)
    finally:
        omni.close()
    export_to_video(outputs[0].images[0], "cosmos3_nano_sim_transfer.mp4", fps=fps)
```

Omit `resolution` to use the deployment's 720 bucket. Omitted window and sink
settings use the deployment configuration, then the checkpoint.

## Verification

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,r_frame_rate,nb_read_frames \
  -of default=noprint_wrappers=1 cosmos3_nano_sim_transfer.mp4
```

Expect `nb_read_frames=121` and `r_frame_rate=30/1`, at the 480p bucket that
matches the control clip's aspect ratio (for example 832×480 for 16:9). Review
the video against the depth sequence. To compare runs, use the same checkpoint,
input, seed, and sampling parameters. Set `output_type="latent"` and compare
the tensors in `outputs[0].images[0]` before comparing decoded videos.

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
