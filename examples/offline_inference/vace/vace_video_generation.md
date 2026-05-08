# VACE Video Generation

[VACE](https://github.com/ali-vilab/VACE) (Video All-in-one Creation Engine) supports multiple video tasks through a single model.

| Model | Architecture | Model Weights (bf16) | HuggingFace |
|-------|-------------|----------------------|-------------|
| Wan2.1-VACE (1.3B) | Wan2.1 | ~10 GB | [Wan-AI/Wan2.1-VACE-1.3B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers) |
| Wan2.1-VACE (14B) | Wan2.1 | ~38 GB | [Wan-AI/Wan2.1-VACE-14B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers) |

## Text-to-Video (T2V)

```bash
python vace_video_generation.py \
  --mode t2v \
  --prompt "A sleek robot stands in a vast warehouse filled with boxes" \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --output t2v_output.mp4
```

## Image-to-Video (I2V)

First frame is kept, remaining frames are generated:

```bash
python vace_video_generation.py \
  --mode i2v \
  --image astronaut.jpg \
  --prompt "An astronaut emerging from a cracked egg on the moon" \
  --height 480 --width 832 --num-frames 81 \
  --output i2v_output.mp4
```

## First-Last-Frame Interpolation (FLF2V)

```bash
python vace_video_generation.py \
  --mode flf2v \
  --image first_frame.jpg --last-image last_frame.jpg \
  --prompt "A bird takes off from a branch and lands on another" \
  --height 512 --width 512 --num-frames 81 \
  --output flf2v_output.mp4
```

## Inpainting

Center vertical stripe is masked and regenerated:

```bash
python vace_video_generation.py \
  --mode inpaint \
  --image scene.jpg \
  --prompt "Shrek walks out of a building" \
  --height 480 --width 832 --num-frames 81 \
  --output inpaint_output.mp4
```

## Reference Image-guided (R2V)

```bash
python vace_video_generation.py \
  --mode r2v \
  --image reference.jpg \
  --prompt "Camera slowly zooms out from the character" \
  --height 480 --width 832 --num-frames 81 \
  --output r2v_output.mp4
```

## Key Arguments

- `--mode`: VACE task mode (`t2v`, `i2v`, `flf2v`, `inpaint`, `r2v`).
- `--model`: Model ID (default: `Wan-AI/Wan2.1-VACE-1.3B-diffusers`).
- `--image`: Input image for I2V, inpainting, and R2V modes.
- `--last-image`: Last frame image for FLF2V mode.
- `--prompt`: Text description of desired video.
- `--height/--width`: Output resolution (default 480x832). Dimensions should be multiples of 16.
- `--num-frames`: Number of frames (default 81).
- `--guidance-scale`: CFG scale (default 5.0).
- `--flow-shift`: Scheduler flow shift (default 5.0).
- `--num-inference-steps`: Number of denoising steps (default 30).
- `--fps`: Frames per second for the saved MP4 (default 16).
- `--output`: Path to save the generated video.
- `--vae-use-tiling`: Enable VAE tiling for memory optimization.
- `--ulysses-degree`: Ulysses sequence parallelism degree for multi-GPU.
- `--cfg-parallel-size`: CFG parallel size for multi-GPU.
- `--tensor-parallel-size`: Tensor parallel size.

> If you encounter OOM errors, try `--vae-use-tiling` or multi-GPU parallelism options.
