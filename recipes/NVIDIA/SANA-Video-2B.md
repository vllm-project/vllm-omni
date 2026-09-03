# SANA-Video 2B

> Native text-to-video and image-to-video generation at 480p and 720p,
> plus a validated Diffusers-adapter compatibility path

## Summary

- Vendor: NVIDIA
- Model: `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`,
  `Efficient-Large-Model/SANA-Video_2B_720p_diffusers`
- Task: Text-to-video and image-to-video
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When to use this recipe

Use the native `SanaVideoPipeline` for T2V and
`SanaImageToVideoPipeline` for I2V. Both native pipelines support the 480p
and 720p checkpoints. Use `--diffusion-load-format diffusers` when you need
the black-box Diffusers compatibility baseline; adapter T2V and I2V are
validated at both resolutions.

The native pipeline loads the 480p checkpoint through
`DistributedAutoencoderKLWan` and the 720p checkpoint through
`DistributedAutoencoderKLLTX2Video`. These are vLLM-Omni distributed wrappers
around the corresponding Diffusers autoencoders, not independent VAE
implementations. The denoising loop also intentionally loads Diffusers'
`DPMSolverMultistepScheduler` from the checkpoint to preserve its scheduler
configuration and numerical behavior.

## References

- Upstream project: <https://github.com/NVlabs/Sana>
- Model cards:
  [480p](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers),
  [720p](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_720p_diffusers)
- Diffusers documentation: <https://huggingface.co/docs/diffusers/api/pipelines/sana_video>
- Online serving guides:
  [Text-to-Video](../../docs/user_guide/examples/online_serving/text_to_video.md),
  [Image-to-Video](../../docs/user_guide/examples/online_serving/image_to_video.md)
- Offline T2V example:
  [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
- Offline I2V example:
  [`examples/offline_inference/image_to_video/image_to_video.py`](../../examples/offline_inference/image_to_video/image_to_video.py)
- Support discussion: [vLLM-Omni issue #5432](https://github.com/vllm-project/vllm-omni/issues/5432)

## Hardware Support

## GPU

### 1x RTX 5090 32GB

#### Environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.3
- Driver / runtime: NVIDIA driver 580.95.05; PyTorch 2.11.0+cu130
- Diffusers: 0.38.0
- vLLM version: 0.26.0
- vLLM-Omni version or commit: PR #5508, commit `22037901`

#### Command

##### Native text-to-video inference

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_720p_diffusers \
  --model-class-name SanaVideoPipeline \
  --prompt "A cat walking on the grass, facing the camera." \
  --negative-prompt "blurry, low quality, temporal artifacts" \
  --height 704 --width 1280 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 6 \
  --extra-body '{"motion_score": 30}' \
  --fps 16 --seed 42 --output sana_video_720p.mp4
```

For 480p, select `SANA-Video_2B_480p_diffusers` and use
`--height 480 --width 832`.

##### Native image-to-video inference

SANA checkpoints declare `SanaVideoPipeline` in `model_index.json`, so I2V
must be selected explicitly with `--model-class-name
SanaImageToVideoPipeline`.

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --model-class-name SanaImageToVideoPipeline \
  --image input.jpg \
  --prompt "A cat turns toward the camera with smooth, natural motion." \
  --negative-prompt "blurry, low quality, temporal artifacts" \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 6 \
  --extra-body '{"motion_score": 30}' \
  --fps 16 --seed 42 --output sana_video_i2v_480p.mp4
```

The same pipeline class supports the 720p checkpoint through vLLM-Omni's
distributed LTX-2 VAE wrapper; use `--height 704 --width 1280`.

For online I2V serving:

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video.sh

INPUT_IMAGE=input.jpg OUTPUT_PATH=sana_video_i2v.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh
```

##### Native online serving

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/text_to_video/run_server_sana_video.sh

bash examples/online_serving/text_to_video/run_curl_sana_video.sh
```

##### Parallel native serving

The native pipelines support tensor parallelism (up to 2 GPUs) and CFG
parallelism. Tensor parallelism on 2 GPUs:

```bash
vllm serve Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --omni \
  --model-class-name SanaVideoPipeline \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --port 8091
```

For CFG parallelism on 2 GPUs, replace `--tensor-parallel-size 2` with
`--cfg-parallel-size 2`; it splits the guided and unguided branches across the
two GPUs and only helps when `guidance_scale` is above 1. Passing both flags
combines the two on 4 GPUs.

Whether tensor parallelism lowers latency depends on the interconnect. Each
transformer block adds an all-reduce, so it speeds up generation only on fast
GPU links such as NVLink and can be slower than a single GPU on PCIe-only
systems. Measure on your hardware before enabling it.

To run the black-box compatibility backend for T2V, replace the server script
with `run_server_sana_video_diffusers.sh`. The same `/v1/videos` request
works; `num_frames` is adapted to Diffusers' `frames` argument. The script
selects `TORCH_SDPA` because SANA-Video uses an attention mask that the
AITER-backed Diffusers attention path does not accept.

##### Diffusers-adapter image-to-video serving

The validated I2V adapter commands are:

```bash
# 480p
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video_diffusers.sh

INPUT_IMAGE=input.jpg OUTPUT_PATH=sana_video_i2v_adapter.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh

# 720p
MODEL=Efficient-Large-Model/SANA-Video_2B_720p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video_diffusers.sh

INPUT_IMAGE=input.jpg WIDTH=1280 HEIGHT=704 \
  OUTPUT_PATH=sana_video_i2v_adapter_720p.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh
```

#### Verification

Check the encoded 720p output metadata after running a generation command:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 sana_video_720p.mp4
```

The standard 720p request above should report:

```text
width=1280
height=704
r_frame_rate=16/1
nb_frames=81
```

For 480p, expect `width=832` and `height=480`.

The automated serving matrix covers both checkpoint variants:

| Backend | 480p T2V | 720p T2V | 480p I2V | 720p I2V |
| --- | --- | --- | --- | --- |
| Native vLLM-Omni | Validated | Validated | Validated | Validated |
| Diffusers adapter | Validated | Validated | Validated | Validated |

Use the native `SanaVideoPipeline` and `SanaImageToVideoPipeline` for the
primary SANA execution paths. The Diffusers adapter is retained as a
validated compatibility/reference backend.

#### Native Cache-DiT and CPU offload

The native T2V and I2V pipelines support Cache-DiT, model-level CPU offload,
and layerwise CPU offload through the common diffusion flags. Layerwise mode
keeps non-block transformer modules, the text encoder, and the VAE on the
runtime device while prefetching DiT blocks in order.

Add one of the following flag sets to either native offline command above:

```bash
# Cache-DiT
--cache-backend cache_dit

# Model-level CPU offload
--enable-cpu-offload

# Layerwise CPU offload
--enable-layerwise-offload

# Cache-DiT plus model-level offload
--cache-backend cache_dit --enable-cpu-offload

# Cache-DiT plus layerwise offload
--cache-backend cache_dit --enable-layerwise-offload

# Distributed layerwise offload (weights sharded across the DP group)
--enable-distributed-layerwise-offload
```

If both CPU offload flags are supplied, the common offloader keeps its existing
layerwise precedence. Cache-DiT refreshes each request from the explicit
`num_inference_steps`; when that field is omitted, both native SANA pipelines
default to 50 inference steps.

Cache-DiT and CPU offload require TP1, CFG1, and SP1. Combining them with
tensor, CFG, or sequence parallelism raises before checkpoint components are
loaded, as do other cache backends such as TeaCache. Cache-DiT cannot be
combined with distributed layerwise offload: per-rank cache skips would
desynchronize the weight AllGather.

Measured on one A800-SXM4-80GB (driver 580.126.09, CUDA 13.0, PyTorch
2.11.0+cu130, Diffusers 0.38.0) with `DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN`,
81 frames, 50 steps, seed 42, each configuration in its own process:

| Configuration | 480p T2V latency | 480p T2V generation peak |
|---|---:|---:|
| Baseline | 120.19 s | 24.07 GiB |
| Cache-DiT | 77.11 s (1.56x) | 24.07 GiB |
| Model CPU offload | 121.15 s | 16.81 GiB (-7.26 GiB) |
| Layerwise offload | 118.63 s | 20.75 GiB (-3.32 GiB) |

Cache-DiT gives 1.54x on 720p I2V (37.0 s to 24.1 s). Latency is the median of
three measured runs after one warmup. Offload trades startup peak too: 14.28
GiB baseline against 6.14 GiB for model-level and 6.80 GiB for layerwise.

Cache-DiT is approximate: it skips block computations, so its output differs
from an uncached run. Similarity against a reference run is not used as a gate
here because this pipeline does not reproduce its own trajectory on this
hardware -- two runs of identical code differ by SSIM 0.9712 / rel_l2 8.7e-2 at
50 steps. Cache-DiT output was checked frame by frame against the uncached run
on 480p T2V and 720p I2V instead, and matches it in subject, composition,
sharpness, and lighting.

The numbers above use the offline example's default cache configuration.
Lowering `residual_diff_threshold` in `cache_config` caches fewer steps,
trading speed for a trajectory closer to the uncached run; see the
[Cache-DiT guide](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md)
for the tuning knobs.

#### Notes

- Key flags: select I2V explicitly with `--model-class-name
  SanaImageToVideoPipeline`; SANA checkpoint `model_index.json` files declare
  the T2V class. Pass `motion_score` through `--extra-body`. The supplied
  Diffusers-adapter server scripts select `TORCH_SDPA` because the AITER-backed
  path does not accept SANA's attention mask.
- Output profile: 81 frames at 16 FPS is the standard checkpoint profile and
  produces approximately five seconds of video. Minute-scale generation
  requires the separate LongSANA/LongLive block-autoregressive workflow.
- Backend boundary: the native pipelines and Transformer are owned by
  vLLM-Omni. The 480p Wan VAE and 720p LTX-2 VAE run through vLLM-Omni
  distributed wrappers derived from the corresponding Diffusers VAE classes.
  The denoising loop intentionally retains the checkpoint-compatible
  Diffusers `DPMSolverMultistepScheduler`.
- Known limitations:
    - Tensor parallelism (up to 2 GPUs) and CFG parallelism are supported for
    the native pipeline. Sequence parallelism, TeaCache, and step execution
    are not validated.
    - Cache-DiT and CPU offload are limited to TP1/CFG1/SP1. Cache-DiT with
    distributed layerwise offload is not supported by the native pipeline.
    - Cache-DiT speedup and offload memory numbers above are single-GPU A800
    measurements; other hardware will differ. Distributed layerwise offload is
    only exercised single-rank here.
    - The Diffusers backend is a compatibility path and does not provide native
    vLLM-Omni parallelism or continuous batching.
    - Native describes pipeline and Transformer ownership, not a zero-Diffusers
    dependency guarantee.

### 1x NVIDIA L20X

#### Environment

- Driver: 570.133.20
- PyTorch: 2.13.0+cu130
- Diffusers: 0.38.0
- vLLM: 0.27.0
- Nsight Systems: 2026.1
- vLLM-Omni commit: `54fcc6d`

#### Same-workload T2V comparison

Both backends used
`Efficient-Large-Model/SANA-Video_2B_480p_diffusers`, the same T2V prompt
and seeds, 832×480 output, 81 frames, and four inference steps. One warmup
request was excluded, followed by three measured requests.

| Metric | Native vLLM-Omni | Diffusers adapter | Native delta |
| --- | ---: | ---: | ---: |
| E2E latency | 7.1600 ± 0.0043 s | 6.9761 ± 0.0517 s | +2.636% |
| Throughput | 0.1397 req/s | 0.1433 req/s | -2.568% |
| Peak reserved VRAM | 23,936 MiB | 19,060 MiB | +25.582% |
| Initialization | 11.669 s | 20.236 s | -42.336% |

The peak-memory result is not a Transformer-only comparison. For the 480p
checkpoint, the native pipeline deliberately loads and decodes with the Wan
VAE in FP32. This matches the upstream SANA-Video reference and the recorded
accuracy-golden setup for this integration. The adapter launch uses
`--dtype bfloat16` as the Diffusers pipeline-wide dtype, so its Wan VAE also
runs in BF16.

Peak VRAM is the CUDA allocator's maximum reserved memory over the complete
request. It includes VAE decode activations and allocator-reserved blocks, not
only live Transformer tensors. The different VAE precision policies are a
material code-level difference and a likely contributor to the higher native
high-water mark; the available full-request measurements do not isolate how
much of the delta they account for. The 25.582% result should therefore not be
interpreted as an isolated native Transformer memory regression. Exact
attribution requires both paths to use the same VAE dtype. Changing the native
480p VAE to BF16 would instead change the recorded golden/reference setup and
requires a new accuracy study.
