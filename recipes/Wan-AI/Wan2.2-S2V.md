# Wan2.2 Speech To Video

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.2-S2V-14B`
- Task: Speech-to-video generation (talking-head / singing from reference image + audio)
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to generate talking-head or singing videos from a
reference image and audio clip using the Wan2.2 S2V 14B model with vLLM-Omni.
Two configurations are provided:

1. **Multi-GPU with tensor parallelism** — splits the DiT across GPUs for faster
   per-step latency.
2. **Multi-GPU with tensor parallelism + CPU offload** — enables generation on
   GPUs with limited VRAM by offloading idle model components to CPU.

## References

- Upstream model card: <https://huggingface.co/Wan-AI/Wan2.2-S2V-14B>
- Example reference assets: <https://github.com/Wan-Video/Wan2.2/tree/main/examples>

## Hardware Support

## CUDA

### 2× NVIDIA A100/H100 (80 GB)

#### Environment

- OS: Linux
- Python: 3.10+
- Driver: NVIDIA driver with CUDA 12.x
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Prerequisites

None


#### Command

```bash
wget -O "Five Hundred Miles.png" \
  "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.png"
wget -O "Five Hundred Miles.MP3" \
  "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.MP3"
```

**TP=2 (no CPU offload, requires ~60 GB VRAM per GPU):**

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python examples/offline_inference/speech_to_video/speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" \
  --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 \
  --fps 16 \
  --tensor-parallel-size 2 \
  --vae-use-slicing --vae-use-tiling \
  --output s2v_720p_tp2.mp4
```

**TP=1 with CPU offload (reduces VRAM usage, adds ~12s overhead):**

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python examples/offline_inference/speech_to_video/speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" \
  --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 \
  --fps 16 \
  --tensor-parallel-size 1 \
  --enable-cpu-offload \
  --vae-use-slicing --vae-use-tiling \
  --output s2v_720p_tp1_offload.mp4
```

**TP=1 * cfg=2 with CPU offload:**

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python examples/offline_inference/speech_to_video/speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" \
  --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 \
  --fps 16 \
  --tensor-parallel-size 1 \
  --enable-cpu-offload \
  --cfg-parallel-size 2 \
  --vae-use-slicing --vae-use-tiling \
  --output s2v_720p_tp1_cfg2_offload.mp4
```

## XPU

### 4x or 8x Intel Arc B70 (32 GB)

#### Environment

- OS: Linux
- Python: 3.10+
- torch: 2.11.0+xpu
- triton-xpu: 3.7.0
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Prerequisites

None


#### Command

```bash
wget -O "Five Hundred Miles.png" \
  "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.png"
wget -O "Five Hundred Miles.MP3" \
  "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.MP3"
```

**TP=4:**

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python examples/offline_inference/speech_to_video/speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" \
  --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 \
  --fps 16 \
  --tensor-parallel-size 4 \
  --vae-use-slicing --vae-use-tiling \
  --output s2v_720p_tp4.mp4
```

**TP=4 cfg=2:**

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python examples/offline_inference/speech_to_video/speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" \
  --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 \
  --fps 16 \
  --tensor-parallel-size 4 \
  --cfg-parallel-size 2 \
  --vae-use-slicing --vae-use-tiling \
  --output s2v_720p_tp4_cfg2_offload.mp4
```


#### Verification

The output video should show a talking/singing person matching the reference
image with lip movements synchronized to the audio. Check:

1. Video file is generated at the specified output path.
2. Audio is muxed into the output MP4.
3. Lip sync quality is reasonable for the given inference steps.

#### Notes

- **Key flags:**
  - `--tensor-parallel-size <N>` — splits the DiT across N GPUs for TP.
  - `--cfg-parallel-size <N>` - splits the conditioned / unconditioned pred to 2 GPUs run parallel.
  - `--enable-cpu-offload` — enables model-level CPU offloading (transformer
    and text_encoder alternate on GPU). Adds ~12s overhead but reduces VRAM.
  - `--vae-use-slicing --vae-use-tiling` — reduces VAE peak memory by
    processing frames in slices and spatial tiles.
  - `--num-frames 81` — generates ~5s of video at 16 fps.
  - `--num-inference-steps` — quality/speed tradeoff. 4 steps for fast preview,
    40 steps for production quality.
- **Performance tips:**
  - Use `--enforce-eager` to skip torch.compile if you encounter recompilation
    issues or want deterministic profiling.
  - For profiling, add `--enable-diffusion-pipeline-profiler` for stage-level
    timing, or `--profiler-config '{"profiler":"torch","torch_profiler_dir":"./perf","torch_profiler_record_shapes":true,"torch_profiler_with_stack":false}'`
    for detailed op-level torch profiler traces.
  - The first clip in multi-clip generation includes warmup overhead; subsequent
    clips run at steady-state speed.
- **Known limitations:**
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn` is required for multi-GPU offline
    inference to avoid CUDA context issues with forked processes.
  - CPU offload adds ~12s latency for model-level shuttling between CPU and GPU.
  - S2V self-attention is ~8.5× more expensive than T2V due to audio-visual
    conditioning fused into self-attn blocks.
