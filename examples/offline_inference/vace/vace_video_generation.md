# VACE Video Generation

[VACE](https://github.com/ali-vilab/VACE) (Video All-in-one Creation Engine) supports multiple video tasks through a single model.

Supported models: `Wan-AI/Wan2.1-VACE-1.3B-diffusers`, `Wan-AI/Wan2.1-VACE-14B-diffusers`

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

## Sequence Parallelism (2 GPUs)

```bash
python vace_video_generation.py \
  --mode t2v \
  --prompt "A robot in a warehouse" \
  --ulysses-degree 2 \
  --output t2v_sp2.mp4
```
