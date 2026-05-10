# Cosmos-Predict2.5

NVIDIA Cosmos-Predict2.5 supports text-to-video (T2W), image-to-video (I2W), and video-to-video (V2W) generation.

> **Note:** Model requires `--revision diffusers/base/post-trained` to locate weights inside the HF repo.

## Run Examples

```bash
cd examples/offline_inference/cosmos
```

### Text-to-Video (T2W)

```bash
python cosmos_predict2_5.py \
  --mode text2world \
  --prompt "A camera moving through a forest" \
  --output output_t2w.mp4
```

### Image-to-Video (I2W)

```bash
python cosmos_predict2_5.py \
  --mode image2world \
  --image /path/to/image.jpg \
  --prompt "The robot continues welding the metal structure" \
  --output output_i2w.mp4
```

### Video-to-Video (V2W)

```bash
python cosmos_predict2_5.py \
  --mode video2world \
  --video /path/to/video.mp4 \
  --prompt "The robot continues pouring liquid into the container" \
  --num-latent-conditional-frames 2 \
  --output output_v2w.mp4
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | required | `text2world`, `image2world`, or `video2world` |
| `--model` | `nvidia/Cosmos-Predict2.5-2B` | Model ID or local path |
| `--revision` | `diffusers/base/post-trained` | Required — model revision/branch |
| `--prompt` | required | Text prompt |
| `--negative-prompt` | `""` | Negative prompt |
| `--image` | — | Input image (required for `image2world`) |
| `--video` | — | Input video (required for `video2world`) |
| `--num-latent-conditional-frames` | — | V2W only; must be `1` or `2` |
| `--height` / `--width` | `704` / `1280` | Video resolution |
| `--num-frames` | `93` | Number of output frames |
| `--num-inference-steps` | `36` | Denoising steps |
| `--guidance-scale` | `7.0` | CFG scale |
| `--seed` | `42` | Random seed |
| `--fps` | `16` | Output video frame rate |
| `--output` | `./output.mp4` | Output file path |

### Memory / Runtime

| Argument | Description |
|---|---|
| `--vae-use-slicing` | Enable VAE slicing |
| `--vae-use-tiling` | Enable VAE tiling |
| `--enable-cpu-offload` | Offload diffusion modules to CPU |
| `--enforce-eager` | Disable `torch.compile` |

## V2W Frame Requirements

Input video must have at least `4 * (num_latent_conditional_frames - 1) + 1` frames:
- `num_latent_conditional_frames=1` → minimum 1 frame
- `num_latent_conditional_frames=2` → minimum 5 frames
