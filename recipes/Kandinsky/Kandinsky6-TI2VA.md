# Kandinsky 6 TI2VA — H100 (v1, single GPU, CPU offload)

> Text/image-to-video-and-audio serving with Kandinsky 6 Pro

## Summary

- Vendor: Kandinsky Lab ([`kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers))
- Model: Kandinsky 6 Pro (TI2VA — text/image-to-video-and-audio)
- Task: Joint text/image-to-video-and-audio generation
- Mode: Offline (`Omni(...)`) and online serving with the OpenAI-compatible
  `/v1/videos` API
- Hardware: 1x NVIDIA H100 80GB HBM3 with component-level CPU offload
- Maintainer: Community

## When to use this recipe

Use this recipe to serve Kandinsky 6's TI2VA pipeline — a single DiT that
jointly denoises video and audio latents, conditioned on a text prompt and,
optionally, a reference image (image-to-video). Tensor parallel, CFG
parallel, Ulysses, ring, pipeline parallel, HSDP, layerwise offload,
MagCache, TeaCache, Cache-DiT, FP8, and tiled VAE patch parallel are wired.
NaviCache and a registered NABLA backend are not. TeaCache coefficients are
uncalibrated. See "Known limitations".

## Supported model contract

| Task                     | Entrypoint                                                                                                                    | Input                         | Output                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------- |
| Text-to-video-and-audio  | `vllm serve kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers --omni` / `Omni(model="kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers")` | text prompt                   | synchronized MP4 (H.264 video + AAC 44.1 kHz mono audio)         |
| Image-to-video-and-audio | same                                                                                                                          | text prompt + reference image | synchronized MP4, first frame conditioned on the reference image |

- Audio generation is on by default (`sample_audio=True`); pass
  `extra_args.sample_audio=false` to generate video only.
- Batch size is 1 request per generation call (upstream limitation of the
  ported denoise loop; concurrent requests are still served by vLLM-Omni's
  own request scheduler, just not batched together within one DiT forward
  pass yet).
- Serving defaults follow the Pro production geometry registered in
  `vllm_omni/model_extras/kandinsky6.py`: **864x480, 125 frames @ 24 fps
  (5.2 s), 50 steps, CFG 5.0**. Any `4k+1` frame count and 16-divisible
  resolution can be requested explicitly via `height`/`width`/`num_frames`/
  `num_inference_steps` — the model is *not* fixed-duration.
- Audio is always 44.1 kHz; its length is derived from the video duration.
- Negative prompt defaults to the pipeline's built-in K6 negative prompt
  when omitted.

## References

- Hub checkpoint: [`kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers)
- Joint video+audio native-model reference: [`recipes/MiniMaxAI/MiniMax-H3.md`](../MiniMaxAI/MiniMax-H3.md)
- Offline example: [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
- Image-to-video example: [`examples/offline_inference/image_to_video/image_to_video.py`](../../examples/offline_inference/image_to_video/image_to_video.py)

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 80GB HBM3
  (driver 570.133.20)
- Number of devices: 1 (multi-GPU tensor parallelism is implemented in the
  transformer's linear layers but not yet validated end-to-end)
- Device interconnect: N/A (single-device path)
- Host memory: 1.4 TB installed; the bf16 DiT (~56 GB) plus the text
  encoders are staged on the host under `--enable-cpu-offload`, so budget
  at least ~90 GB of free host RAM.
- Qualification scope: single-GPU T2VA at the Pro default geometry and at a
  reduced smoke geometry, offline and via `vllm serve`. Numbers below were
  measured once on the shared box described above; treat them as
  indicative, not as a benchmark.

## Software environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.13
- Driver / runtime: CUDA 12.9 (torch 2.13.0+cu129)
- vLLM version: 0.29.0
- vLLM-Omni version or commit: this repository, `vllm_omni/diffusion/models/kandinsky6/`
- diffusers 0.40.0, transformers 5.14.1
- Attention: FlashAttention-3 on this box (`flash_attn_interface`). The Pro
  timings below use it. Without FA3, `attention_engine: "auto"` falls back
  to PyTorch SDPA; an earlier SDPA Pro run on the same GPU was 26.9 s/it
  and 1377 s.

## Weights

`Kandinsky6TI2VAPipeline._load_components` loads the public Diffusers layout
from [`kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers)
(`model_index.json`, `transformer/`, `vae/`, `text_encoder/`, `tokenizer/`,
`text_encoder_2/`, `tokenizer_2/`, `audio_vae/`, `scheduler/`). Pass that
public repo id to `vllm serve` / `Omni(model=...)`.

Each weight folder is loaded on its own (`transformer/`, `vae/`, `text_encoder/`, `text_encoder_2/`, `audio_vae/`). The DiT is built with initialization skipped, then the framework loader fills it from `transformer/`.

## Command

```bash
vllm serve kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers --omni \
  --host 127.0.0.1 --port 8091 \
  --num-gpus 1 --enable-cpu-offload
```

`--enable-cpu-offload` enables model-level offload with mutual exclusion
between the transformer and the two text encoders (they never co-reside on
the GPU). Without it the bf16 DiT (~56 GB) plus Qwen2.5-VL-7B (~16 GB) plus
the VAEs and activations do not fit an 80 GB device at the Pro geometry.

Offline (writes an MP4 with the audio track muxed in):

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers \
  --prompt "A golden retriever runs along a sunny beach, waves crashing, cinematic footage" \
  --seed 42 --enable-cpu-offload --output kandinsky6_t2va.mp4
# Video only: add --extra-body '{"sample_audio": false}'
```

Image-to-video-and-audio uses the shared image-to-video example:

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers \
  --image first_frame.png \
  --prompt "A golden retriever runs along a sunny beach, waves crashing, cinematic footage" \
  --seed 42 --enable-cpu-offload --output kandinsky6_i2va.mp4
```

The generic scripts pick up the Pro defaults from
`vllm_omni/model_extras/kandinsky6.py`.

## Verification

`/v1/videos` takes multipart form fields (not a JSON body). Submit, poll,
then download:

```bash
VID=$(curl -s -X POST http://127.0.0.1:8091/v1/videos \
  -F prompt="A golden retriever runs along a sunny beach, waves crashing" \
  -F size=864x480 -F num_frames=125 -F num_inference_steps=50 -F seed=42 \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

# Poll until "status": "completed"
curl -s http://127.0.0.1:8091/v1/videos/$VID

curl -s -o kandinsky6.mp4 http://127.0.0.1:8091/v1/videos/$VID/content
# Expect an MP4 with an H.264 video stream (125 frames, 24 fps, 864x480)
# and an AAC audio stream (44100 Hz, mono) of matching duration.
```

Smoke-size request for quick validation (~30 s end-to-end on one H100):
`-F size=512x320 -F num_frames=25 -F num_inference_steps=10`.

## Measurements

Single H100 80GB, seed 42, prompt
"A golden retriever runs along a sunny beach, waves crashing, cinematic footage",
negative prompt left at the pipeline default, MagCache off, prompt expansion
off. Pro geometry: 864x480, 125 frames, 50 steps, CFG 5.0, audio on.

vLLM-Omni: `text_to_video.py --model kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers --enable-cpu-offload --model-class-name Kandinsky6TI2VAPipeline`,
FlashAttention-3.

| Wall / generation                               | Step time         | Peak GPU                                 | After load |
| ----------------------------------------------- | ----------------- | ---------------------------------------- | ---------- |
| 751.7 s generation (12.5 min); engine init 41 s | 14.4 s/it average | 78.39 GiB reserved (78.0 GiB nvidia-smi) | 17.74 GiB  |

An earlier SDPA vLLM-Omni Pro run on the same GPU was 26.9 s/it and 1377 s,
peak reserved 74.97 GiB. A 512x320 / 25-frame / 10-step SDPA smoke was 1.4 s/it
and 29.5 s at the same reserved peak.

The MP4 is H.264 864x480, 125 frames at 24 fps (5.22 s), plus AAC 44.1 kHz
mono (230400 samples): a golden retriever on a sunny beach with breaking waves.

vLLM-Omni's VAE decode logged two failed CUDA allocations (7.15 GiB, then
4.27 GiB) and still wrote a complete file.

Audio is peak-normalized to full scale by default; pass
`extra_args.audio_normalization="clip"` to keep the raw decoded amplitude
instead.

## Notes

- Key flags: `extra_args.sample_audio` (bool, default true),
  `extra_args.audio_normalization` (`"normalize"` default | `"clip"`),
  `extra_args.visual_cond_scheme` (defaults to `pretrain` for text-only,
  `tail_cond_first_frame` when an image is supplied).
- The post-process payload is a flat dict (`video`, `audio` as float32 in
  `[-1, 1]`, `audio_sample_rate=44100`, `fps`) so the serving muxer picks
  up the correct sample rate.
- Acceleration that is wired:
    - Tensor parallel: the DiT loader narrows full checkpoint tensors onto
    each rank's `ColumnParallelLinear` / `RowParallelLinear` shard.
    - CFG parallel (`--cfg-parallel-size 2`): rank 0 runs the conditional
    DiT, rank 1 the unconditional one, then both all-gather video and
    audio velocities.
    - HSDP: `_hsdp_shard_conditions` covers the indexed transformer blocks.
    Do not combine with tensor parallel.
    - Sequence parallel: visual tokens (and their RoPE) are sharded;
    Ulysses all-to-all runs inside visual self-attention, Ring uses the
    PyTorch ring kernel. Text stays replicated. Audio queries gather the
    full visual sequence.
    - Pipeline parallel: `visual_transformer_blocks` are split with
    `PPMissingLayer`. Embeddings stay on the first stage, output heads on
    the last. Activations and the final velocity are sent between stages.
    - VAE patch parallel: Hunyuan decode splits the latent width when
    `vae_patch_parallel_size > 1` and tiling is on.
    - MagCache / TeaCache / Cache-DiT are registered. MagCache reuses the
    Pro `mag_ratios`. TeaCache coefficients are **uncalibrated** (copied
    from Qwen-Image) so skips are not quality-neutral. Cache-DiT targets
    `visual_transformer_blocks` only and falls back to the same step cache
    if the fused `(video, audio)` block return is rejected.
    - FP8: online quantization runs in the DiT loader via
    `process_weights_after_loading` after the sharded assign.
    - `attention_engine: auto` uses FlashAttention-3 when
    `flash_attn_interface` imports.
- Known limitations:
    - NABLA sparse attention is not registered in the framework attention
    backend. NaviCache is not wired.
    - Prompt expansion, NF4 Qwen, and the super-resolution cascade are not
    part of this port.
    - Distributed layerwise offload still uses one process unless another
    parallel axis sets `world_size > 1` (for example tensor parallel).
    - `_load_components` loads the Diffusers checkpoint directly rather than the
    streamed-prefetch loader.

## Supported features

| Feature                                          | Status                                                       | Guide                                                           |
| ------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------------- |
| Text-to-video-and-audio                          | Supported (validated on H100)                                | —                                                               |
| Image-to-video-and-audio                         | Supported (not yet validated on hardware)                    | —                                                               |
| CPU offload (`--enable-cpu-offload`)             | Supported (required on 80 GB)                                | —                                                               |
| Layerwise offload (`--enable-layerwise-offload`) | Supported (about 36 GiB peak on the smoke geometry)          | —                                                               |
| Tensor parallelism                               | Wired (sharded DiT load)                                     | —                                                               |
| CFG parallelism                                  | Wired (`--cfg-parallel-size 2`)                              | —                                                               |
| Ulysses / Ring sequence parallel                 | Wired (visual tokens only)                                   | —                                                               |
| Pipeline parallelism                             | Wired (visual block split)                                   | —                                                               |
| HSDP                                             | Wired (not with tensor parallel)                             | —                                                               |
| VAE patch parallel                               | Wired (width-split decode)                                   | —                                                               |
| MagCache                                         | Wired (Pro ratios, step reuse)                               | —                                                               |
| TeaCache                                         | Wired, coefficients uncalibrated                             | —                                                               |
| Cache-DiT                                        | Wired on `visual_transformer_blocks`                         | —                                                               |
| FP8                                              | Wired (online quant; smoke peak about 46 GiB vs 75 GiB bf16) | [`FP8 quantization`](../../docs/user_guide/quantization/fp8.md) |
| NaviCache                                        | Not supported                                                | —                                                               |
| LoRA                                             | Not supported                                                | —                                                               |
