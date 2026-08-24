# Speech-To-Video

This example demonstrates how to generate talking-head videos from a reference image and an audio clip using the Wan2.2 Speech-to-Video (S2V) pipeline with vLLM-Omni's offline inference API.

The S2V pipeline generates multiple video clips autoregressively — the last frames of each clip become the motion context for the next — producing a seamless video that spans the full audio duration.

## Prerequisites

The S2V pipeline requires `librosa` for audio loading:

```bash
pip install librosa decord
```

## Local CLI Usage

### Singing Example (480p)

```bash
wget -O "Five Hundred Miles.png" "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.png"
wget -O "Five Hundred Miles.MP3" "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.MP3"
python speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" \
  --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 448 --width 832 \
  --num-inference-steps 5 \
  --output s2v_singing_480p.mp4
```

## Key Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *(required)* | Path to Wan2.2 S2V model (local path or HuggingFace ID) |
| `--image` | *(required)* | Path to reference image (face/portrait) |
| `--audio` | *(required)* | Path to audio file (wav/mp3) |
| `--prompt` | `"A person speaking naturally"` | Text prompt describing the scene |
| `--negative-prompt` | *(S2V default)* | Negative prompt (built-in Chinese quality filter if not set) |
| `--height` / `--width` | *auto* | Output resolution (auto-calculated from image aspect ratio if not set, must be divisible by 64) |
| `--num-frames` | `80` | Number of frames per clip (should be divisible by 4) |
| `--num-inference-steps` | `40` | Number of denoising steps |
| `--guidance-scale` | `4.5` | Classifier-free guidance scale |
| `--flow-shift` | `3.0` | Scheduler flow shift |
| `--seed` | `42` | Random seed for reproducibility |
| `--fps` | `16` | Frames per second for the saved MP4 |
| `--init-first-frame` | `False` | Use reference image as the first frame |
| `--output` | `s2v_output.mp4` | Output video file path |
| `--vae-use-slicing` | `False` | Enable VAE slicing for memory optimization |
| `--vae-use-tiling` | `False` | Enable VAE tiling for memory optimization |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism (TP) inside the DiT |
| `--cfg-parallel-size` | `1` | Number of GPUs for CFG parallelism (1 or 2) |
| `--vae-patch-parallel-size` | `1` | Number of GPUs for VAE patch parallelism |
| `--enable-cpu-offload` | `False` | Enable CPU offloading |
| `--enable-layerwise-offload` | `False` | Enable layerwise offloading on DiT |

> If you encounter OOM errors, try `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage, or `--enable-cpu-offload` for more aggressive memory saving.

## Performance Tips

The dominant cost in S2V is self-attention across the full token sequence (~80K at 720p), which scales quadratically with resolution. Strategies to speed up generation:

| Strategy | Speedup | How |
|---|---|---|
| **Lower resolution** (480p) | ~3.5x | `--height 448 --width 832` — reduces tokens from 80K to 33K |
| **Tensor parallelism** (2 GPUs) | ~1.4x | `--tensor-parallel-size 2` — splits attention/GEMM across GPUs |
| **Fewer steps** | Linear | `--num-inference-steps 5` — trades quality for speed |
| **Combine all above** | ~5x+ | 480p + TP=2 + 5 steps |

## Profiling

To profile the S2V pipeline and identify performance bottlenecks:

### Enable Built-in Profiler

Use `--enable-diffusion-pipeline-profiler` to display stage durations:

```bash
python speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" \
  --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 448 --width 832 \
  --num-inference-steps 5 \
  --enable-diffusion-pipeline-profiler \
  --output s2v_output.mp4
```

### Torch Profiler (Device-Level Profiling)

For detailed GPU/XPU kernel-level profiling with Chrome trace visualization:

```bash
python speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" \
  --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 448 --width 832 \
  --num-inference-steps 1 \
  --profile-dir /tmp/s2v_profile \
  --profile-record-shapes \
  --profile-with-memory \
  ----profile-with-stack \
  --output s2v_profiled.mp4
```

**Profiler Arguments:**

| Parameter | Description |
|-----------|-------------|
| `--profile-dir` | Enable torch profiler and save traces to this directory |
| `--profile-record-shapes` | Record tensor shapes (increases trace size) |
| `--profile-with-stack` | Record stack traces (increases overhead) |
| `--profile-with-memory` | Profile memory usage |
| `--profile-with-flops` | Estimate FLOPs for operations |

**Output Files:**
- `trace_rank*.json.gz` - Chrome trace (open in `chrome://tracing`)
- `ops_rank*.xlsx` - Excel workbook with operation statistics

> **Note:** Use `--num-inference-steps 1` for faster profiling. The trace captures one full denoising step with all transformer blocks and attention operations.

## How It Works

The S2V pipeline processes in the following stages:

1. **Pre-process**: The reference image is resized/cropped to fit the target resolution. The audio path is validated.
2. **Text encoding**: The prompt is encoded using the UMT5 text encoder.
3. **Audio encoding**: The audio file is loaded via `librosa`, processed through Wav2Vec2, and bucketed to align with video frame-rate. The audio length determines the number of clips (`num_repeat`).
4. **Reference image encoding**: The reference image is VAE-encoded into latent tokens.
5. **Multi-clip denoising**: For each clip, random noise is generated and denoised through the `WanModel_S2V` transformer with audio injection. The last frames of each decoded clip become the motion context for the next.
6. **Post-process**: All clips are concatenated and converted to pixel-space video frames.
