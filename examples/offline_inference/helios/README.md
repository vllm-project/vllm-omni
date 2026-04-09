# Helios Video Generation

Helios is a text-to-video (T2V), image-to-video (I2V), and video-to-video (V2V) diffusion model. This example demonstrates end-to-end video generation using vLLM-Omni with three model variants:

| Variant | Description | Key Features |
|---------|-------------|--------------|
| **Helios-Base** | Base model, Stage 1 only | Single-stage denoising, `guidance_scale=5.0` |
| **Helios-Mid** | Mid model, Stage 2 pyramid | Multi-stage pyramid denoising, CFG-Zero* support |
| **Helios-Distilled** | Distilled model, Stage 2+3 | Few-step inference with DMD, `guidance_scale=1.0` |

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Run Examples

Get into the example folder:
```bash
cd examples/offline_inference/helios
```

### Text-to-Video (T2V)

**Helios-Base** (Stage 1 only):
```bash
python end2end.py \
    --model BestWishYsh/Helios-Base \
    --sample-type t2v \
    --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
    --guidance-scale 5.0 \
    --output helios_t2v_base.mp4
```

**Helios-Mid** (Stage 2 + CFG-Zero*):
```bash
python end2end.py \
    --model BestWishYsh/Helios-Mid \
    --sample-type t2v \
    --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
    --guidance-scale 5.0 \
    --is-enable-stage2 \
    --pyramid-num-inference-steps-list 20 20 20 \
    --use-cfg-zero-star --use-zero-init --zero-steps 1 \
    --output helios_t2v_mid.mp4
```

**Helios-Distilled** (Stage 2 pyramid + DMD):
```bash
python end2end.py \
    --model BestWishYsh/Helios-Distilled \
    --sample-type t2v \
    --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
    --num-frames 240 \
    --guidance-scale 1.0 \
    --is-enable-stage2 \
    --pyramid-num-inference-steps-list 2 2 2 \
    --is-amplify-first-chunk \
    --output helios_t2v_distilled.mp4
```

### Image-to-Video (I2V)

**Helios-Base**:
```bash
python end2end.py \
    --model BestWishYsh/Helios-Base \
    --sample-type i2v \
    --image-path /path/to/image.jpg \
    --prompt "A towering emerald wave surges forward, its crest curling with raw power and energy." \
    --guidance-scale 5.0 \
    --output helios_i2v_base.mp4
```

**Helios-Mid** (Stage 2 + CFG-Zero*):
```bash
python end2end.py \
    --model BestWishYsh/Helios-Mid \
    --sample-type i2v \
    --image-path /path/to/image.jpg \
    --prompt "A towering emerald wave surges forward, its crest curling with raw power and energy." \
    --guidance-scale 5.0 \
    --is-enable-stage2 \
    --pyramid-num-inference-steps-list 20 20 20 \
    --use-cfg-zero-star --use-zero-init --zero-steps 1 \
    --output helios_i2v_mid.mp4
```

**Helios-Distilled**:
```bash
python end2end.py \
    --model BestWishYsh/Helios-Distilled \
    --sample-type i2v \
    --image-path /path/to/image.jpg \
    --prompt "A towering emerald wave surges forward, its crest curling with raw power and energy." \
    --num-frames 240 \
    --guidance-scale 1.0 \
    --is-enable-stage2 \
    --pyramid-num-inference-steps-list 2 2 2 \
    --is-amplify-first-chunk \
    --output helios_i2v_distilled.mp4
```

### Video-to-Video (V2V)

**Helios-Base**:
```bash
python end2end.py \
    --model BestWishYsh/Helios-Base \
    --sample-type v2v \
    --video-path /path/to/video.mp4 \
    --prompt "A bright yellow Lamborghini speeds along a curving mountain road." \
    --guidance-scale 5.0 \
    --output helios_v2v_base.mp4
```

**Helios-Mid** (Stage 2 + CFG-Zero*):
```bash
python end2end.py \
    --model BestWishYsh/Helios-Mid \
    --sample-type v2v \
    --video-path /path/to/video.mp4 \
    --prompt "A bright yellow Lamborghini speeds along a curving mountain road." \
    --guidance-scale 5.0 \
    --is-enable-stage2 \
    --pyramid-num-inference-steps-list 20 20 20 \
    --use-cfg-zero-star --use-zero-init --zero-steps 1 \
    --output helios_v2v_mid.mp4
```

**Helios-Distilled**:
```bash
python end2end.py \
    --model BestWishYsh/Helios-Distilled \
    --sample-type v2v \
    --video-path /path/to/video.mp4 \
    --prompt "A bright yellow Lamborghini speeds along a curving mountain road." \
    --num-frames 240 \
    --guidance-scale 1.0 \
    --is-enable-stage2 \
    --pyramid-num-inference-steps-list 2 2 2 \
    --is-amplify-first-chunk \
    --output helios_v2v_distilled.mp4
```

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `BestWishYsh/Helios-Base` | Model ID or local path |
| `--sample-type` | `t2v` | Generation mode: `t2v`, `i2v`, or `v2v` |
| `--prompt` | — | Text prompt describing the video |
| `--negative-prompt` | *(see source)* | Negative prompt for CFG (includes anti-static terms) |
| `--image-path` | — | Input image (required for `i2v`) |
| `--video-path` | — | Input video (required for `v2v`) |
| `--height` | `384` | Video height in pixels |
| `--width` | `640` | Video width in pixels |
| `--num-frames` | `99` | Number of output frames |
| `--num-inference-steps` | `50` | Denoising steps (Stage 1 only) |
| `--guidance-scale` | `5.0` | CFG scale (`1.0` for Distilled) |
| `--seed` | `42` | Random seed |
| `--fps` | `16` | Output video frame rate |
| `--output` | `helios_output.mp4` | Output file path |

### Stage 2 / Pyramid Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--is-enable-stage2` | off | Enable pyramid multi-stage denoising |
| `--pyramid-num-stages` | `3` | Number of pyramid stages |
| `--pyramid-num-inference-steps-list` | `10 10 10` | Steps per pyramid stage |
| `--is-amplify-first-chunk` | off | DMD amplification (Distilled) |

### CFG-Zero* Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-cfg-zero-star` | off | Enable CFG-Zero* guidance (Mid) |
| `--use-zero-init` | off | Zero init for first steps |
| `--zero-steps` | `1` | Number of zero-init steps |

### Memory & Parallelism

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vae-use-slicing` | off | Enable VAE slicing |
| `--vae-use-tiling` | off | Enable VAE tiling |
| `--enforce-eager` | off | Disable torch.compile |
| `--enable-cpu-offload` | off | CPU offloading |
| `--enable-layerwise-offload` | off | Layerwise offloading |
| `--ulysses-degree` | `1` | Ulysses SP degree |
| `--ring-degree` | `1` | Ring SP degree |
| `--cfg-parallel-size` | `1` | CFG parallel size (1 or 2) |
| `--tensor-parallel-size` | `1` | Tensor parallelism size |
